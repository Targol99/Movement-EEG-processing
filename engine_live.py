"""
kinematics_engine_live.py
=========================
Hardware producer + live PyVista viewer. CALIBRATED (tare) mode only.

This file owns all SDK I/O. It imports the math from kinematics_core (which is
MAC-agnostic and works in joint-name space) and the rendering from visualizer.

Workflow (sensors are connected ONCE, then you can re-run from step 2 freely):
  1. Connect the IMUs.                                  [done one time]
  2. Choose a config file, then map each joint to one connected IMU
     by index (the terminal shows  index -> MAC).
  3. Place the subject in the neutral pose and press 'R' to calibrate;
     the 3D window opens and live tracking starts.
  4. Press 'A' (or close the window then re-run) to drop back to step 2 and
     pick a different config / re-map sensors WITHOUT disconnecting.

Hotkeys while the viewer is open (type in THIS console):
  'R' = re-tare to the current pose
  'A' = change config (returns to step 2, sensors stay connected)
  close the window = quit
"""

import sys
import time
import json
import threading
import msvcrt
from pathlib import Path

import aquisition_and_processing.kinematics_core as kc
from visualisation.visualizer import SkeletonVisualizer

# Movella PC SDK
sys.path.append(r"C:\Program Files\Movella\DOT PC SDK 2023.6\SDK Files\Examples\python") 
from xdpchandler import * # type: ignore

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_RATE = 30
MAX_IMUS = 4
CONFIG_DIR = Path(__file__).parent / "configs"          # folder scanned for *.json configs
AXIS_LENGTH = 12.0                           # length of the per-joint orientation triad
STATUS_PERIOD_S = 2.0                        # console heartbeat period


# ==============================================================================
# THREAD-SAFE FRAME HAND-OFF
# ==============================================================================
class FrameBuffer:
    """Single-slot, last-write-wins buffer shared between producer and viewer."""
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
# KEYBOARD (Windows console)
# ==============================================================================
def read_key():
    """Non-blocking single-key read. Returns a lowercase char, or None if no key."""
    if msvcrt.kbhit():
        return msvcrt.getch().decode("utf-8", "ignore").lower()
    return None


# ==============================================================================
# HARDWARE ORCHESTRATION (SDK I/O lives only in this file)
# ==============================================================================
def initialize_hardware_handler():
    handler = XdpcHandler() # type: ignore
    if not handler.initialize():
        print("[ERROR] Failed to initialize Movella DOT PC SDK.")
        sys.exit(-1)
    return handler


def connect_and_configure_imus(handler, max_imus, output_rate):
    print("[1/3] Scanning for nearby Movella DOT devices...")
    handler.scanForDots()
    time.sleep(3.0)

    detected = handler.detectedDots()
    if len(detected) == 0:
        print("[ERROR] No hardware targets found during scan."); handler.cleanup(); sys.exit(-1)

    print(f"[2/3] Found {len(detected)} sensors. Connecting (Cap: {max_imus})...")
    handler.connectDots()
    time.sleep(6.0)

    connected = handler.connectedDots()
    if len(connected) == 0:
        print("[ERROR] Connection routine timed out."); handler.cleanup(); sys.exit(-1)

    active_sensors = connected[:max_imus]
    connected_macs = [s.portInfo().bluetoothAddress() for s in active_sensors]

    print("\n[Hardware Matrix Status] Active Links:")
    for mac in connected_macs:
        print(f"  -> {mac}")

    for sensor in active_sensors:
        sensor.setOnboardFilterProfile("General")
        sensor.setOutputRate(output_rate)
        if not sensor.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler): # type: ignore
            print("[ERROR] Measurement streaming failed."); handler.cleanup(); sys.exit(-1)

    return handler, connected_macs


def flush_buffer_noise(handler, connected_macs):
    flush_start = time.time()
    while time.time() - flush_start < 1.0:
        if handler.packetsAvailable():
            for mac in connected_macs:
                handler.getNextPacket(mac)
        time.sleep(0.01)


def capture_raw_sensor_orientations(handler, macs):
    """Pull one orientation sample per MAC. Returns {mac: R}."""
    raw_matrices = {}
    if handler.packetsAvailable():
        for mac in macs:
            packet = handler.getNextPacket(mac)
            if packet is not None and packet.containsOrientation():
                euler = packet.orientationEuler()
                raw_matrices[mac] = kc.rpy_to_rotation_matrix_xyz(
                    euler.roll(), euler.pitch(), euler.yaw())
    return raw_matrices


# ==============================================================================
# SENSOR <-> JOINT MAPPING (the MAC-agnostic part)
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
# LIVE STATE BOOTSTRAP / CALIBRATION (all dicts keyed by joint name)
# ==============================================================================
def update_live_states(current_states, new_samples, sorted_joints):
    for joint in sorted_joints:
        if joint in new_samples:
            current_states[joint] = new_samples[joint]
    return current_states


def initialize_live_states(handler, sorted_joints, joint_to_mac):
    """Block until we have one orientation sample for every active joint."""
    macs = list(joint_to_mac.values())
    live_states = {}
    while len(live_states) < len(sorted_joints):
        raw = capture_raw_sensor_orientations(handler, macs)
        samples = samples_by_joint(raw, joint_to_mac)
        for joint in sorted_joints:
            if joint in samples and joint not in live_states:
                live_states[joint] = samples[joint]
        time.sleep(0.005)
    return live_states


def snapshot_calibration(live_states, sorted_joints):
    return {joint: live_states[joint].copy() for joint in sorted_joints}


def wait_for_calibration_key(handler, joint_to_mac):
    """Block until 'R' is pressed, pumping packets so the buffers stay fresh."""
    macs = list(joint_to_mac.values())
    print("\n>> Place the subject in the NEUTRAL pose, then press 'R' to calibrate. <<")
    while True:
        _ = capture_raw_sensor_orientations(handler, macs)
        if read_key() == "r":
            print("[!] Tare captured. Building transforms...")
            return
        time.sleep(0.01)


# ==============================================================================
# FRAME ASSEMBLY (pulls math from core, builds the viewer's snapshot)
# ==============================================================================
def assemble_frame(state_dict, sorted_joints, endpoints, joint_pos, endpoint_pos, world_rot):
    joints = {
        name: {
            "name": name,
            "parent": state_dict[name].get("parent"),
            "pos": joint_pos[name],
            "R": world_rot[name],
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
    joint_pos, endpoint_pos = kc.run_forward_kinematics_absolute(
        state_dict, sorted_joints, active_joints, world_rot, endpoints)
    return assemble_frame(state_dict, sorted_joints, endpoints, joint_pos, endpoint_pos, world_rot)


# ==============================================================================
# PRODUCER LOOP (background thread; SERIAL CALLS ONLY)
# ==============================================================================
def engine_loop(handler, stop_event, reconfigure_event, frame_buffer, state_dict,
                sorted_joints, active_joints, endpoints, frame_change, calibration,
                live_states, joint_to_mac):
    macs = list(joint_to_mac.values())
    last_status = time.time()
    frames = 0
    while not stop_event.is_set():
        # Hotkeys: 'R' re-tares, 'A' requests a return to config selection.
        key = read_key()
        if key == "r":
            calibration = snapshot_calibration(live_states, sorted_joints)
            print("[engine] Re-calibrated to current pose.")
        elif key == "a":
            print("[engine] Reconfigure requested. Closing viewer...")
            reconfigure_event.set()   # viewer sees this via should_close and exits
            return

        raw = capture_raw_sensor_orientations(handler, macs)
        new_samples = samples_by_joint(raw, joint_to_mac)
        live_states = update_live_states(live_states, new_samples, sorted_joints)
        frame_buffer.set(compute_frame(
            state_dict, sorted_joints, active_joints, endpoints,
            frame_change, calibration, live_states))

        frames += 1
        now = time.time()
        if now - last_status >= STATUS_PERIOD_S:
            print(f"[engine] {frames / (now - last_status):5.1f} Hz")
            frames, last_status = 0, now

        time.sleep(0.002)


# ==============================================================================
# ONE TRACKING SESSION (steps 2 -> 3 -> 4). Returns True to reconfigure again.
# ==============================================================================
def run_session(handler, connected_macs):
    # ---- Step 2: choose config + map sensors to joints ----
    config = choose_config_file(CONFIG_DIR)
    if config is None:
        return False  # user chose to quit

    joint_to_mac = map_sensors_to_joints(config, connected_macs)
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
    wait_for_calibration_key(handler, joint_to_mac)
    live_states = initialize_live_states(handler, sorted_joints, joint_to_mac)
    calibration = snapshot_calibration(live_states, sorted_joints)

    frame_buffer = FrameBuffer()
    frame_buffer.set(compute_frame(
        state_dict, sorted_joints, active_joints, endpoints,
        frame_change, calibration, live_states))

    stop_event = threading.Event()
    reconfigure_event = threading.Event()
    thread = threading.Thread(
        target=engine_loop,
        args=(handler, stop_event, reconfigure_event, frame_buffer, state_dict,
              sorted_joints, active_joints, endpoints, frame_change, calibration,
              live_states, joint_to_mac),
        daemon=True,
    )
    thread.start()

    print("\n[!] Engine online.")
    print("    'R' = re-tare   |   'A' = change config   |   close window = quit\n")

    # Viewer blocks here until the window closes OR 'A' was pressed.
    SkeletonVisualizer(axis_length=AXIS_LENGTH, title="Live Skeleton").run(
        frame_buffer.get, should_close=reconfigure_event.is_set)

    # Viewer closed: stop the producer thread cleanly.
    stop_event.set()
    thread.join(timeout=1.0)

    # ---- Step 4: 'A' -> reconfigure; window closed manually -> quit ----
    if reconfigure_event.is_set():
        print("\n[!] Returning to config selection (sensors stay connected)...")
        flush_buffer_noise(handler, connected_macs)  # drop stale packets
        return True
    print("[!] Viewer closed.")
    return False


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # ---- Step 1: connect the hardware ONCE ----
    raw_handler = initialize_hardware_handler()
    handler, connected_macs = connect_and_configure_imus(raw_handler, MAX_IMUS, OUTPUT_RATE)
    flush_buffer_noise(handler, connected_macs)
    print(f"\n[!] {len(connected_macs)} IMU(s) connected and streaming.")

    try:
        # ---- Session loop: each pass = choose config, map, calibrate, view ----
        while run_session(handler, connected_macs):
            pass
    finally:
        handler.cleanup()
        print("-> SDK resources released. Connections closed.")


if __name__ == "__main__":
    main()
