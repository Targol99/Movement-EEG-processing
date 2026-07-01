"""
kinematics_engine_live_write.py
===============================
Main loop + live PyVista viewer + CSV recording. CALIBRATED (tare) mode only.

This file no longer touches any hardware SDK. Acquisition is handled by two
independent producer threads:

  * eeg_engine.eeg_loop      -> EEGBuffer        (latest EEG sample)
  * movella_engine.movella_loop -> MovellaBuffer (latest orientation per MAC)

This file then runs:

  * a PRODUCER thread that reads raw orientations from MovellaBuffer, applies
    calibration + forward kinematics (from kinematics_core), and publishes a
    computed FrameBuffer;
  * a CSV WRITER thread (CsvWriter) that consumes ONLY the buffers
    (FrameBuffer + EEGBuffer) and writes timestamped rows;
  * an optional PyVista visualizer (main thread) that consumes FrameBuffer.

Visualization can be disabled with `--no-viz` (a.k.a. `--headless`).

Workflow (sensors connect ONCE at startup, then re-run from step 2 freely):
  1. Connect IMUs + EEG.                                [done one time]
  2. Choose a config, map each joint to a connected IMU by index.
  3. Place subject in neutral pose, press 'R' to calibrate; tracking starts.
  4. Press 'A' to reconfigure (sensors stay connected), or 'Q' / close window
     to quit.

Hotkeys (type in THIS console):
  'R' = re-tare to current pose
  'A' = change config (returns to step 2, sensors stay connected)
  'W' = start/stop CSV recording (each start opens recordings/recording_*.csv)
  'T' = write a manual timestamp marker row into the active recording
  'Q' = quit (also: close the viewer window)
"""

import sys
import csv
import time
import json
import threading
import msvcrt
from datetime import datetime
from pathlib import Path
import numpy as np

import aquisition_and_processing.kinematics_core as kc
from visualisation.visualizer import SkeletonVisualizer
from aquisition_and_processing.eeg_processing import extract_features_multichannel, CHANNEL_NAMES

from aquisition_and_processing.eeg_engine import EEGBuffer, eeg_loop
from aquisition_and_processing.movella_engine import MovellaBuffer, movella_loop

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_RATE = 30
MAX_IMUS = 4
CONFIG_DIR = Path(__file__).parent              # folder scanned for *.json configs
AXIS_LENGTH = 12.0                              # length of the per-joint orientation triad
STATUS_PERIOD_S = 5.0                           # console heartbeat period
CSV_PERIOD_DEFAULT_S = 0.05                     # default CSV logging period
CSV_DIR = Path(__file__).parent / "recordings"  # where recording_*.csv files are written
CONNECT_TIMEOUT_S = 40.0                         # how long to wait for IMUs to connect
ENABLE_VISUALIZATION = True                      # default; overridable with --no-viz

# EEG channel layout written to CSV (must match the EEG stream order).
EEG_CHANNELS = ("F7", "C3", "PZ", "CZ", "F8", "O1", "O2", "C4", "REF")


# ==============================================================================
# THREAD-SAFE COMPUTED-FRAME HAND-OFF
# ==============================================================================
class FrameBuffer:
    """Single-slot, last-write-wins buffer for the computed frame snapshot."""
    def __init__(self):
        self._lock = threading.Lock() 
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


# ==============================================================================
# CSV RECORDING (consumes ONLY the buffers)
# ==============================================================================
def _wrap180(d):
    """Wrap a degree difference into (-180, 180] so derivatives don't spike on flips."""
    return (d + 180.0) % 360.0 - 180.0


class CsvWriter:
    """
    Periodic CSV logger that interacts ONLY with the buffers it is handed:
    the computed FrameBuffer (for per-joint world rotations) and the EEGBuffer
    (for the latest EEG sample). It holds no hardware references and performs no
    kinematics beyond reading rotation matrices out of the latest frame.

    Each row holds, per joint, the roll/pitch/yaw of its calibrated world
    rotation (deg) plus angular velocity (deg/s) and acceleration (deg/s^2) per
    axis (finite differences between consecutive *logged* rows), followed by the
    EEG channels. Timing uses time.perf_counter(); t = 0.0 is the instant
    recording starts.

    Runs on its own thread (run()); recording is toggled with 'W' and markers
    dropped with 'T' from the keypress thread. A lock guards all file access so
    those two threads can't collide.
    """
    AXES = ("roll", "pitch", "yaw")

    def __init__(self, sorted_joints, frame_buffer, eeg_buffer,
                 out_dir, period=CSV_PERIOD_DEFAULT_S):
        self.joints = list(sorted_joints)
        self.frame_buffer = frame_buffer
        self.eeg_buffer = eeg_buffer
        self.out_dir = Path(out_dir)
        self.period = float(period)

        self._lock = threading.Lock()
        self.recording = False
        self._fh = None
        self._writer = None
        self._origin = 0.0
        self._next_write = 0.0
        self._marker_count = 0
        # Finite-difference history (advanced on periodic rows only).
        self._prev_t = None
        self._prev_ang = None
        self._prev_vel = None

    # ------------------------------------------------------ header / columns
    def _header(self):
        cols = ["time_s", "marker"]
        for j in self.joints:
            # World angles
            for ax in self.AXES:
                cols += [f"{j}_world_{ax}_deg"]
            # Intrinsic angles with derivatives
            for ax in self.AXES:
                cols += [f"{j}_intrinsic_{ax}_deg",
                         f"{j}_intrinsic_{ax}_vel_dps",
                         f"{j}_intrinsic_{ax}_acc_dps2"]
        cols += [f"EEG_{ch}" for ch in EEG_CHANNELS]
        # Add EEG feature columns (theta, alpha, beta per channel)
        for ch in CHANNEL_NAMES:
            cols += [f"EEG_{ch}_theta_pw",
                     f"EEG_{ch}_alpha_pw",
                     f"EEG_{ch}_beta_pw"]
        return cols

    # ----------------------------------------------------- buffer readers
    def _angles_from_frame(self):
        """Pull per-joint roll/pitch/yaw (deg) from the latest computed frame.
        Returns a dict with 'world' and 'intrinsic' keys, each mapping joint -> (r, p, y).
        """
        frame = self.frame_buffer.get()
        if frame is None:
            return None
        joints = frame.get("joints", {})
        world_ang = {}
        intrinsic_ang = {}
        for name in self.joints:
            jr = joints.get(name)
            if jr is None:
                return None  # frame not yet consistent with this joint set
            world_ang[name] = kc.matrix_to_rpy_xyz(jr["R"])
            intrinsic_ang[name] = kc.matrix_to_rpy_xyz(jr.get("R_intrinsic", np.eye(3)))
        return {"world": world_ang, "intrinsic": intrinsic_ang}

    def _eeg_row(self):
        """Latest EEG sample as exactly len(EEG_CHANNELS) values; NaN-padded."""
        latest = self.eeg_buffer.get_latest()
        n = len(EEG_CHANNELS)
        if latest is None:
            return [float("nan")] * n
        _ts, sample = latest
        vals = list(sample)[:n]
        vals += [float("nan")] * (n - len(vals))
        return vals

    def _eeg_features(self):
        """Extract EEG band power features (theta, alpha, beta) from the sliding window.
        Returns a flat list: [F7_theta, F7_alpha, F7_beta, C3_theta, C3_alpha, ...]
        Returns NaN-padded if window is not yet available.
        """
        window = self.eeg_buffer.get_window()
        n_channels = len(CHANNEL_NAMES)
        if window is None:
            return [float("nan")] * (n_channels * 3)  # 3 features per channel
        
        features = extract_features_multichannel(window)
        
        result = []
        for ch_name in CHANNEL_NAMES:
            if ch_name in features:
                result += [features[ch_name].get("theta", float("nan")),
                           features[ch_name].get("alpha", float("nan")),
                           features[ch_name].get("beta", float("nan"))]
            else:
                result += [float("nan"), float("nan"), float("nan")]
        return result

    # ----------------------------------------------------- start / stop
    def toggle(self, now):
        """Start a new recording, or stop the current one. Bound to 'W'."""
        with self._lock:
            if self.recording:
                self._close_locked()
                print("[csv] recording STOPPED.")
                return
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"recording_{datetime.now():%Y%m%d_%H%M%S}.csv"
            self._fh = open(path, "w", newline="")
            self._writer = csv.writer(self._fh)
            self._writer.writerow(self._header())
            self._origin = now
            self._next_write = now            # emit the first sample immediately at t=0
            self._marker_count = 0
            self._prev_t = self._prev_ang = self._prev_vel = None
            self.recording = True
            print(f"[csv] recording STARTED -> {path.name} "
                  f"(period {self.period:.3f}s, t=0 now).")

    def _close_locked(self):
        """Flush and close the file. Caller must hold the lock. Idempotent."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._fh = None
        self._writer = None
        self.recording = False

    def close(self):
        with self._lock:
            self._close_locked()

    # ----------------------------------------------------- derivatives / row
    def _vel_acc(self, t, intrinsic_ang):
        """(vel, acc) per joint/axis vs the previous periodic sample, computed from intrinsic angles.
        intrinsic_ang is a dict of joint -> (r, p, y) tuples.
        Returns (vel, acc) dicts with same structure; zeros if none yet.
        """
        if self._prev_t is None:
            zero = {j: (0.0, 0.0, 0.0) for j in self.joints}
            return zero, dict(zero)
        dt = max(t - self._prev_t, 1e-9)
        have_vel = self._prev_vel is not None
        vel, acc = {}, {}
        for j in self.joints:
            v, a = [], []
            for k in range(3):
                vk = _wrap180(intrinsic_ang[j][k] - self._prev_ang[j][k]) / dt
                v.append(vk)
                a.append((vk - self._prev_vel[j][k]) / dt if have_vel else 0.0)
            vel[j], acc[j] = tuple(v), tuple(a)
        return vel, acc

    def _row_locked(self, t, marker, world_ang, intrinsic_ang, vel, acc, eeg_row, eeg_features):
        """Write one row. Caller must hold the lock and be recording.
        vel and acc are derived from intrinsic angles.
        """
        row = [f"{t - self._origin:.6f}", marker]
        for j in self.joints:
            # World angles (no derivatives)
            for k in range(3):
                row.append(f"{world_ang[j][k]:.4f}")
            # Intrinsic angles with derivatives
            for k in range(3):
                row += [f"{intrinsic_ang[j][k]:.4f}", f"{vel[j][k]:.4f}", f"{acc[j][k]:.4f}"]
        row += [f"{x:.4f}" for x in eeg_row]
        row += [f"{x:.4f}" for x in eeg_features]
        self._writer.writerow(row)
        self._fh.flush()

    # ----------------------------------------------------- periodic write
    def _maybe_write_locked(self, now):
        """Emit one periodic row if recording and the period has elapsed. Holds lock."""
        if not self.recording or now < self._next_write:
            return
        ang_data = self._angles_from_frame()
        if ang_data is None:
            return  # no usable frame yet; try again next tick
        world_ang = ang_data["world"]
        intrinsic_ang = ang_data["intrinsic"]
        vel, acc = self._vel_acc(now, intrinsic_ang)
        self._row_locked(now, "", world_ang, intrinsic_ang, vel, acc, self._eeg_row(), self._eeg_features())
        # Advance on a fixed grid; resync if we fell behind (avoid bursts).
        self._next_write += self.period
        if self._next_write <= now:
            self._next_write = now + self.period
        self._prev_t, self._prev_ang, self._prev_vel = now, intrinsic_ang, vel

    def add_marker(self, now):
        """Write an exact-time marker row. Bound to 'T'. No-op if not recording."""
        with self._lock:
            if not self.recording:
                print("[csv] 'T' ignored (not recording; press 'W' to start).")
                return
            ang_data = self._angles_from_frame()
            if ang_data is None:
                print("[csv] 'T' ignored (no frame available yet).")
                return
            self._marker_count += 1
            world_ang = ang_data["world"]
            intrinsic_ang = ang_data["intrinsic"]
            vel, acc = self._vel_acc(now, intrinsic_ang)   # vs last periodic sample; history untouched
            self._row_locked(now, self._marker_count, world_ang, intrinsic_ang, vel, acc, self._eeg_row(), self._eeg_features())
            print(f"[csv] marker #{self._marker_count} at t={now - self._origin:.3f}s")

    # ----------------------------------------------------- thread entry point
    def run(self, stop_event):
        """Periodic logging loop. Runs on its own thread until stop_event is set."""
        tick = min(self.period / 4.0, 0.01)
        while not stop_event.is_set():
            now = time.perf_counter()
            with self._lock:
                self._maybe_write_locked(now)
            time.sleep(tick)
        self.close()


# ==============================================================================
# KEYBOARD (Windows console)
# ==============================================================================
def read_key():
    """Non-blocking single-key read. Returns a lowercase char, or None if no key."""
    if msvcrt.kbhit():
        return msvcrt.getch().decode("utf-8", "ignore").lower()
    return None


# ==============================================================================
# SENSOR <-> JOINT MAPPING (MAC-agnostic; no SDK here)
# ==============================================================================
def samples_by_joint(raw_by_mac, joint_to_mac):
    """Translate {mac: R} into {joint_name: R} using the active mapping."""
    return {joint: raw_by_mac[mac]
            for joint, mac in joint_to_mac.items()
            if mac in raw_by_mac}


def list_config_files(config_dir):
    return sorted(config_dir.glob("*.json"))


def choose_config_file(config_dir):
    """Let the user pick a JSON config by index. Returns parsed dict, or None to quit."""
    files = list_config_files(config_dir)
    if not files:
        print(f"[ERROR] No .json config files found in {config_dir}")
        return None

    print("\n" + "-" * 60)
    print("Available configs:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")
    print("  [q] Quit")

    while True:
        choice = input("Select config index: ").strip().lower()
        if choice == "q":
            return None
        if choice.isdigit() and int(choice) < len(files):
            path = files[int(choice)]
            try:
                with open(path, "r") as fh:
                    config = json.load(fh)
                print(f"-> Loaded '{path.name}'")
                return config
            except Exception as e:
                print(f"[ERROR] Failed to parse {path.name}: {e}")
                continue
        print("  Invalid choice, try again.")


def map_sensors_to_joints(config, connected_macs):
    """
    Walk the joints in the config and, for each, ask which connected IMU drives
    it. The terminal shows an index -> MAC table so the user types a short index
    instead of a full address. Returns {joint_name: mac}; joints left blank are
    skipped (treated as inactive for this session).
    """
    mac_by_index = {i: mac for i, mac in enumerate(connected_macs)}
    joint_names = list(config.get("state", {}).keys())
    joint_to_mac = {}
    used_macs = set()

    print("\n" + "-" * 60)
    print("Map each joint to a connected IMU (press Enter to skip a joint).")
    print("Connected IMUs:")
    for i, mac in mac_by_index.items():
        print(f"  [{i}] {mac}")

    for joint in joint_names:
        while True:
            choice = input(f"  IMU index for joint '{joint}' (Enter to skip): ").strip()
            if choice == "":
                print(f"    -> '{joint}' skipped.")
                break
            if not choice.isdigit() or int(choice) not in mac_by_index:
                print("    Invalid index, try again.")
                continue
            mac = mac_by_index[int(choice)]
            if mac in used_macs:
                print(f"    {mac} is already assigned, choose another.")
                continue
            joint_to_mac[joint] = mac
            used_macs.add(mac)
            print(f"    -> '{joint}' <- [{choice}] {mac}")
            break

    return joint_to_mac


# ==============================================================================
# LIVE STATE BOOTSTRAP / CALIBRATION (reads from MovellaBuffer; no SDK here)
# ==============================================================================
def update_live_states(current_states, new_samples, sorted_joints):
    for joint in sorted_joints:
        if joint in new_samples:
            current_states[joint] = new_samples[joint]
    return current_states


def initialize_live_states(movella_buffer, sorted_joints, joint_to_mac):
    """Block until we have one orientation sample for every active joint."""
    live_states = {}
    while len(live_states) < len(sorted_joints):
        raw = movella_buffer.get_orientations()
        samples = samples_by_joint(raw, joint_to_mac)
        for joint in sorted_joints:
            if joint in samples and joint not in live_states:
                live_states[joint] = samples[joint]
        time.sleep(0.005)
    return live_states


def snapshot_calibration(live_states, sorted_joints):
    return {joint: live_states[joint].copy() for joint in sorted_joints}


def wait_for_calibration_key():
    """Block until 'R' is pressed. The Movella thread keeps the buffer fresh."""
    print("\n>> Place the subject in the NEUTRAL pose, then press 'R' to calibrate. <<")
    while True:
        if read_key() == "r":
            print("[!] Tare captured. Building transforms...")
            return
        time.sleep(0.01)


# ==============================================================================
# FRAME ASSEMBLY (pulls math from core, builds the viewer's snapshot)
# ==============================================================================
def assemble_frame(state_dict, sorted_joints, endpoints, joint_pos, endpoint_pos, world_rot, intrinsic_rot=None):
    import numpy as np
    joints = {
        name: {
            "name": name,
            "parent": state_dict[name].get("parent"),
            "pos": joint_pos[name],
            "R": world_rot[name],
            "R_intrinsic": intrinsic_rot[name],
        }
        for name in sorted_joints
    }
    eps = {
        ep_name: {"parent": endpoints[ep_name]["parent"], "pos": endpoint_pos[ep_name]}
        for ep_name in endpoint_pos
    }
    return {"joints": joints, "endpoints": eps}


def compute_frame(state_dict, sorted_joints, active_joints, endpoints,
                  frame_change, calibration, live_states):
    world_rot = kc.build_world_rotations_calibrated(frame_change, calibration, live_states, sorted_joints)
    parent_dict = {j: state_dict[j].get("parent") for j in sorted_joints}
    intrinsic_rot = kc.build_intrinsic_joint_rotations(frame_change, calibration, live_states, sorted_joints, active_joints, parent_dict)
    joint_pos, endpoint_pos = kc.run_forward_kinematics_absolute(
        state_dict, sorted_joints, active_joints, world_rot, endpoints)
    return assemble_frame(state_dict, sorted_joints, endpoints, joint_pos, endpoint_pos, world_rot, intrinsic_rot)


# ==============================================================================
# PRODUCER LOOP (background thread): MovellaBuffer -> compute -> FrameBuffer
# ==============================================================================
def engine_loop(stop_event, reconfigure_event, frame_buffer, movella_buffer, recorder,
                state_dict, sorted_joints, active_joints, endpoints,
                frame_change, calibration, live_states, joint_to_mac):
    last_status = time.time()
    frames = 0
    while not stop_event.is_set():
        raw = movella_buffer.get_orientations()              # {mac: R}
        new_samples = samples_by_joint(raw, joint_to_mac)    # {joint: R}
        live_states = update_live_states(live_states, new_samples, sorted_joints)
        frame = compute_frame(
            state_dict, sorted_joints, active_joints, endpoints,
            frame_change, calibration, live_states)
        frame_buffer.set(frame)

        # Hotkeys: R re-tare, A reconfigure, Q quit, W start/stop CSV, T marker.
        key = read_key()
        if key == "r":
            calibration = snapshot_calibration(live_states, sorted_joints)
            print("[engine] Re-calibrated to current pose.")
        elif key == "a":
            print("[engine] Reconfigure requested. Closing session...")
            reconfigure_event.set()
            return
        elif key == "q":
            print("[engine] Quit requested.")
            stop_event.set()
            return
        elif key == "w":
            recorder.toggle(time.perf_counter())
        elif key == "t":
            recorder.add_marker(time.perf_counter())

        frames += 1
        now = time.time()
        if now - last_status >= STATUS_PERIOD_S:
            print(f"[engine] {frames / (now - last_status):5.1f} Hz")
            frames, last_status = 0, now

        time.sleep(0.002)


# ==============================================================================
# ONE TRACKING SESSION (steps 2 -> 3 -> 4). Returns True to reconfigure again.
# ==============================================================================
def prompt_csv_period(default=CSV_PERIOD_DEFAULT_S):
    """Ask for the CSV logging period in seconds; Enter accepts the default."""
    raw = input(f"CSV log period in seconds [{default}]: ").strip()
    if raw == "":
        return default
    try:
        val = float(raw)
        if val <= 0.0:
            raise ValueError
        return val
    except ValueError:
        print(f"  Invalid period; using {default}s.")
        return default


def run_session(movella_buffer, eeg_buffer, enable_viz):
    # ---- Step 2: choose config + map sensors to joints ----
    config = choose_config_file(CONFIG_DIR)
    if config is None:
        return False  # user chose to quit

    joint_to_mac = map_sensors_to_joints(config, movella_buffer.get_macs())
    if not joint_to_mac:
        print("[!] No joints mapped. Returning to config selection.")
        return True

    active_joints = set(joint_to_mac.keys())
    state_dict, sorted_joints, endpoints = kc.parse_topology(config, active_joints)
    frame_change = kc.get_frame_change_matrices(state_dict, sorted_joints)

    print(f"\n[!] Tracking {len(sorted_joints)} joint(s): {', '.join(sorted_joints)}")
    if endpoints:
        print(f"    + {len(endpoints)} endpoint(s): {', '.join(endpoints)}")

    # ---- Step 3: calibrate, then build the first frame ----
    wait_for_calibration_key()
    live_states = initialize_live_states(movella_buffer, sorted_joints, joint_to_mac)
    calibration = snapshot_calibration(live_states, sorted_joints)

    frame_buffer = FrameBuffer()
    frame_buffer.set(compute_frame(
        state_dict, sorted_joints, active_joints, endpoints,
        frame_change, calibration, live_states))

    csv_period = prompt_csv_period()
    recorder = CsvWriter(sorted_joints, frame_buffer, eeg_buffer, CSV_DIR, csv_period)

    stop_event = threading.Event()
    reconfigure_event = threading.Event()

    producer = threading.Thread(
        target=engine_loop,
        args=(stop_event, reconfigure_event, frame_buffer, movella_buffer, recorder,
              state_dict, sorted_joints, active_joints, endpoints,
              frame_change, calibration, live_states, joint_to_mac),
        daemon=True,
    )
    csv_thread = threading.Thread(target=recorder.run, args=(stop_event,), daemon=True)
    producer.start()
    csv_thread.start()

    print("\n[!] Engine online.")
    print("    'R' = re-tare   |   'A' = change config   |   'Q' = quit")
    print(f"    'W' = start/stop CSV ({csv_period:.3f}s)   |   'T' = timestamp marker\n")

    if enable_viz:
        # Viewer blocks on the main thread until the window closes, 'A', or 'Q'.
        SkeletonVisualizer(axis_length=AXIS_LENGTH, title="Live Skeleton").run(
            frame_buffer.get,
            should_close=lambda: reconfigure_event.is_set() or stop_event.is_set())
    else:
        # Headless: keep the main thread alive until 'A' (reconfigure) or 'Q' (quit).
        print("[!] Headless mode (visualization disabled). "
              "'A' to reconfigure, 'Q' to quit.")
        while not reconfigure_event.is_set() and not stop_event.is_set():
            time.sleep(0.05)

    # Session ending: stop producer + CSV threads cleanly, finalize the recording.
    stop_event.set()
    producer.join(timeout=1.0)
    csv_thread.join(timeout=1.0)
    recorder.close()

    # ---- Step 4: 'A' -> reconfigure; otherwise (window close / 'Q') -> quit ----
    if reconfigure_event.is_set():
        print("\n[!] Returning to config selection (sensors stay connected)...")
        return True
    print("[!] Session closed.")
    return False


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    enable_viz = ENABLE_VISUALIZATION and not any(
        arg in ("--no-viz", "--headless") for arg in sys.argv[1:])

    # ---- Step 1: start the acquisition threads (connect ONCE) ----
    eeg_buffer = EEGBuffer()
    movella_buffer = MovellaBuffer()
    movella_stop = threading.Event()

    eeg_thread = threading.Thread(target=eeg_loop, args=(eeg_buffer,), daemon=True)
    movella_thread = threading.Thread(
        target=movella_loop,
        args=(movella_buffer, movella_stop, MAX_IMUS, OUTPUT_RATE),
        daemon=True,
    )
    eeg_thread.start()
    movella_thread.start()

    print("Waiting for Movella IMUs to connect...")
    if not movella_buffer.wait_until_ready(timeout=CONNECT_TIMEOUT_S):
        print("[ERROR] Movella connection timed out. Exiting.")
        movella_stop.set()
        return

    connected_macs = movella_buffer.get_macs()
    if not connected_macs:
        print("[ERROR] No IMUs connected. Exiting.")
        movella_stop.set()
        return

    print(f"\n[!] {len(connected_macs)} IMU(s) connected and streaming. "
          f"Visualization {'ON' if enable_viz else 'OFF'}.")

    try:
        # ---- Session loop: each pass = choose config, map, calibrate, view ----
        while run_session(movella_buffer, eeg_buffer, enable_viz):
            pass
    finally:
        movella_stop.set()
        movella_thread.join(timeout=2.0)
        print("-> Shutdown complete.")


if __name__ == "__main__":
    main()
