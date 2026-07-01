"""
kinematics_engine_offline_write.py
==================================
Main loop + live PyVista viewer + OFFLINE recording. CALIBRATED (tare) mode only.

This is the offline-recording counterpart of kinematics_engine_live_write.py.
The acquisition + visualization architecture is unchanged; only the recording
strategy differs:

  LIVE version  : a CsvWriter thread resampled BOTH streams onto one fixed grid
                  (CSV_PERIOD), keeping only the latest sample per tick. IMU got
                  aliased, EEG got decimated, and the two streams drifted out of
                  alignment.

  OFFLINE version (this file):
    * 'W' (1st press) ARMS recording. From that instant every native sample of
      EVERY stream is appended to an UNBOUNDED in-RAM list (no maxlen, no grid,
      no resampling). Acquisition time is assumed reasonable, so OOM is not a
      concern.
    * 'W' (2nd press) STOPS recording, then computes forward kinematics over the
      captured IMU stream and writes the results. Nothing is written during
      acquisition.
    * Because IMU (~OUTPUT_RATE Hz) and EEG (256 Hz) stream at different rates,
      two files are written with the same base name plus a stream suffix:
          recording_<stamp>_imu.csv   (one row per IMU frame, full kinematics)
          recording_<stamp>_eeg.csv   (one row per EEG sample, raw + features)

Timing model (device/source clocks, not arrival time)
-----------------------------------------------------
Each row's primary `time_s` comes from the stream's OWN regular clock, not from
when the host happened to dequeue the sample (arrival time is smeared by
Bluetooth/transport buffering and is unusable at these rates):

  * IMU  -> Movella `sampleTimeFine` (device microsecond clock). Each sensor is
            normalised to its first sample in the recording, and frames are
            aligned across sensors on the reference sensor's device grid, so the
            IMU file comes out regular at the configured output rate.
  * EEG  -> the LSL source timestamp returned by pull_sample (already a true
            sample time from the source clock), normalised to the first EEG
            sample in the recording.

Both files ALSO carry `host_recv_s`, a time.perf_counter() arrival stamp on one
shared host clock, for COARSE cross-stream alignment (good to within the
transport jitter). For PRECISE cross-stream sync, drop a 'T' marker at a known
physical event: markers are stamped on the host clock and annotated into the
nearest row of BOTH files, giving an exact common reference.

"minimal changes" footprint
----------------------------
To capture full native-rate streams the buffers must accumulate; rather than
editing the buffer classes in place this file SUBCLASSES EEGBuffer and
MovellaBuffer and overrides add_sample to append to an unbounded list while
armed. eeg_engine.py, kinematics_core.py and eeg_processing.py are UNTOUCHED.
The one edit outside this file is in movella_engine.py: capture_raw_sensor_
orientations now returns the packet's sampleTimeFine, and movella_loop unwraps
it and stores device time instead of time.time() -- the fix for IMU timing.

Hotkeys (type in THIS console) -- identical to the live version:
  'R' = re-tare to current pose
  'A' = change config (returns to step 2, sensors stay connected)
  'W' = start/stop recording (stop -> compute + write *_imu.csv and *_eeg.csv)
  'T' = drop a timestamped marker (annotated into the nearest row of both files)
  'Q' = quit (also: close the viewer window)
"""

import sys
import csv
import time
import json
import queue
import bisect
import threading
import msvcrt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

import aquisition_and_processing.kinematics_core as kc
from visualisation.visualizer import SkeletonVisualizer
from aquisition_and_processing.eeg_processing import windowed_band_features, CHANNEL_NAMES

from aquisition_and_processing.eeg_engine import RecordingEEGBuffer, eeg_loop
from aquisition_and_processing.movella_engine import RecordingMovellaBuffer, movella_loop

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MOVELLA_OUTPUT_RATE = 60  # output-rate for Movella IMUs; do not change.
EEG_SAMPLE_RATE = 256     # EEG stream rate; do not change.

MAX_IMUS = 4
CONFIG_DIR = Path(__file__).parent / "configs"  # folder scanned for *.json configs
CSV_DIR = Path(__file__).parent / "recordings"  # where recording_*.csv files are written

AXIS_LENGTH = 12.0       # length of the per-joint orientation triad
STATUS_PERIOD_S = 2.0    # console heartbeat period
CONNECT_TIMEOUT_S = 40.0 # how long to wait for IMUs to connect
POLLING_PERIOD_S = 0.1 # how often movella_loop polls the SDK for new samples
ENABLE_VISUALIZATION = False   # default; overridable with --no-viz

# EEG channel layout written to CSV (must match the EEG stream order).
EEG_CHANNELS = ("F7", "C3", "PZ", "CZ", "F8", "O1", "O2", "C4", "REF")

# Offline EEG band-power features (theta/alpha/beta per channel), computed on a
# trailing 512-sample window, exactly like the live version did -- but now at
# native sample times. Set WRITE_EEG_FEATURES = False to write raw channels only
# (much faster). EEG_FEATURE_STRIDE controls how often features are computed:
# stride=1 -> every sample (slowest); stride=8 -> ~32 Hz feature rate, the rows
# in between get NaN feature cells. Raise the stride if offline processing feels
# slow; raw channels are always written at full native rate regardless.
WRITE_EEG_FEATURES = True
EEG_FEATURE_STRIDE = 1
EEG_FEATURE_WINDOW = 512

WRITING_TIME_WINDOW: float | None = 5.0  # slice length in seconds. If set, the
#   writer keeps the first int(round(window * rate)) samples PER STREAM, sliced
#   AFTER time-ordering (IMU frames on the device grid; EEG on the LSL clock).
#   None = keep everything captured between the two 'W' presses.
WRITING_CAPTURE_MARGIN_S = 0.2  # when the auto-stop timer is used, record this
#   much LONGER than WRITING_TIME_WINDOW so there are always >= window*rate
#   samples available to slice (your "record a little longer, then slice").
WAIT_TIME_BEFORE_MOVEMENT = 2.05  # seconds to wait after the first 'W' before the subject is expected to start moving.


# ==============================================================================
# THREAD-SAFE COMPUTED-FRAME HAND-OFF (for the live viewer; unchanged)
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
# OFFLINE RECORDER (no thread, no live file I/O)
# ------------------------------------------------------------------------------
# 'W' toggles arm/disarm. On disarm it pulls the captured raw streams out of the
# buffers, computes forward kinematics over the IMU stream, and writes the two
# CSVs. 'T' records a marker time; markers are annotated onto the nearest row of
# each file when written.
# ==============================================================================
def _wrap180(d):
    """Wrap a degree difference into (-180, 180] so derivatives don't spike on flips."""
    return (d + 180.0) % 360.0 - 180.0


class OfflineRecorder:
    AXES = ("roll", "pitch", "yaw")

    def __init__(self, sorted_joints, active_joints, state_dict, endpoints,
                 frame_change, joint_to_mac, eeg_buffer, movella_buffer, out_dir):
        self.joints = list(sorted_joints)
        self.active_joints = set(active_joints)
        self.state_dict = state_dict
        self.endpoints = endpoints
        self.frame_change = frame_change
        self.joint_to_mac = joint_to_mac
        self.eeg_buffer = eeg_buffer
        self.movella_buffer = movella_buffer
        self.out_dir = Path(out_dir)

        self.recording = False
        self.movement_prompt_shown = False
        self._t0 = 0.0             # perf_counter at recording start (shared origin)
        self._calibration = None   # locked at recording start
        self._markers = []         # list of perf_counter marker times
        self._base = None          # base file name (no suffix)
        self._timer_thread = None  # timer thread for auto-stop

        # Serialises start/stop transitions so the auto-stop timer, a manual 'W',
        # and session finalize can never run _stop concurrently or stop the wrong
        # recording. _epoch tags each recording so a stale timer is ignored.
        self._lifecycle_lock = threading.Lock()
        self._epoch = 0

        # Background writer: _stop() snapshots + slices the captured data and
        # enqueues a self-contained job; this thread does the (slow) kinematics +
        # CSV write so the producer/viewer thread never blocks. A single worker
        # serialises jobs, so back-to-back recordings can't interleave on disk.
        self._jobs = queue.Queue()
        self._worker = threading.Thread(target=self._worker_run,
                                        name="offline-writer", daemon=True)
        self._worker.start()

    # ----------------------------------------------------------- arm / disarm
    def toggle(self, now, calibration):
        """Bound to 'W'. now = time.perf_counter(); calibration is the active tare."""
        if not self.recording:
            self._start(now, calibration)
        else:
            self._stop()

    def _start(self, now, calibration):
        with self._lifecycle_lock:
            if self.recording:
                return
            self._t0 = now
            # Lock the calibration used for this recording (re-taring mid-recording
            # does not retroactively change a capture in progress).
            self._calibration = {j: np.asarray(calibration[j], float).copy() for j in self.joints}
            self._markers = []
            self._base = f"recording_{datetime.now():%Y%m%d_%H%M%S}"
            self._epoch += 1
            epoch = self._epoch
            self.eeg_buffer.start_recording()
            self.movella_buffer.start_recording()
            self.recording = True

        # If WRITING_TIME_WINDOW is set, start a timer thread to auto-stop. It
        # carries this recording's epoch so a leftover timer can't stop a later one.
        if WRITING_TIME_WINDOW is not None:
            self._timer_thread = threading.Thread(
                target=self._timer_callback, args=(epoch,), daemon=True)
            self._timer_thread.start()
            print(f"[offline] recording STARTED (t=0 now). Auto-stop in {WRITING_TIME_WINDOW}s. Base name '{self._base}'.")
        else:
            print(f"[offline] recording STARTED (t=0 now). Base name '{self._base}'.")

    def add_marker(self, now):
        """Bound to 'T'. No-op if not recording."""
        if not self.recording:
            print("[offline] 'T' ignored (not recording; press 'W' to start).")
            return
        self._markers.append(now)
        print(f"[offline] marker #{len(self._markers)} at t={now - self._t0:.3f}s")

    def finalize_if_recording(self):
        """Called on session end so a capture in progress is not lost."""
        if self.recording:
            print("[offline] session ending while recording; finalizing...")
            self._stop()

    def _timer_callback(self, epoch):
        """Auto-stop this recording after WRITING_TIME_WINDOW s (ignored if stale).

        Sleeps a little LONGER than the slice window (capture margin) so the
        writer always has at least window*rate samples to slice down to.
        """
        time.sleep(WRITING_TIME_WINDOW + WRITING_CAPTURE_MARGIN_S)
        with self._lifecycle_lock:
            stale = (epoch != self._epoch) or (not self.recording)
        if stale:
            return
        print("[offline] timer expired, auto-stopping...")
        self._stop(epoch=epoch)

    def _stop(self, epoch=None):
        """
        Runs on the producer (or timer / finalize) thread. Atomically disarm +
        snapshot only (cheap), then hand a self-contained job to the writer
        thread. Returns immediately so the viewer/keys never stall on the
        kinematics + CSV write. `epoch`, if given, must still be current.
        """
        # set movement prompt flag
        self.movement_prompt_shown = False

        with self._lifecycle_lock:
            if not self.recording or (epoch is not None and epoch != self._epoch):
                return
            self.recording = False
            t0, base = self._t0, self._base
            calibration, markers = self._calibration, list(self._markers)
            # Disarm + snapshot inside the lock so a new _start can't race in and
            # have its buffer drained by this stop.
            eeg_records = self.eeg_buffer.stop_recording()
            imu_records = self.movella_buffer.stop_recording()

        # NOTE: no slicing here. The window cut is integer-based and applied in
        # the writer AFTER the samples are time-ordered (raw records are in
        # arrival order, which is not sample order for interleaved sensors).
        job = {
            "base": base,
            "t0": t0,
            "calibration": calibration,
            "markers": markers,
            "imu_records": imu_records,
            "eeg_records": eeg_records,
            "window": WRITING_TIME_WINDOW,   # seconds, or None to keep everything
        }
        self._jobs.put(job)
        print(f"[offline] recording STOPPED. Captured "
              f"{len(imu_records)} IMU sample(s), {len(eeg_records)} EEG sample(s). "
              f"Queued for background writing (pending jobs: {self._jobs.qsize()}).")

    # ----------------------------------------------------------- writer thread
    def _worker_run(self):
        """Process write jobs one at a time until a None sentinel is received."""
        while True:
            job = self._jobs.get()
            try:
                if job is None:           # shutdown sentinel
                    return
                self._process_job(job)
            except Exception as e:        # never let a bad job kill the writer
                print(f"[offline] writer error: {e}")
            finally:
                self._jobs.task_done()

    def _process_job(self, job):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        imu_path = self.out_dir / f"{job['base']}_imu.csv"
        eeg_path = self.out_dir / f"{job['base']}_eeg.csv"

        # Integer slice targets = window * stream rate. Applied AFTER ordering,
        # inside each writer. None -> keep everything.
        window = job.get("window")
        imu_keep_low = int(round(EEG_FEATURE_WINDOW / EEG_SAMPLE_RATE * MOVELLA_OUTPUT_RATE)) if window is not None else None
        eeg_keep_low = EEG_FEATURE_WINDOW if window is not None else None
        imu_keep_high = int(round(window * MOVELLA_OUTPUT_RATE)) if window is not None else None
        eeg_keep_high = int(round(window * EEG_SAMPLE_RATE)) if window is not None else None
        imu_keep_slice = slice(imu_keep_low, imu_keep_high) if window is not None else None
        eeg_keep_slice = slice(eeg_keep_low, eeg_keep_high) if window is not None else None
        
        t_start = time.perf_counter()
        n_imu = self._write_imu(imu_path, job["imu_records"], job["calibration"],
                                job["markers"], job["t0"], keep_slice=imu_keep_slice)
        n_eeg = self._write_eeg(eeg_path, job["eeg_records"],
                                job["markers"], job["t0"], keep_slice=eeg_keep_slice)
        dt = time.perf_counter() - t_start
        print(f"[offline] wrote {n_imu} IMU frame(s) -> {imu_path}")
        print(f"[offline] wrote {n_eeg} EEG row(s)   -> {eeg_path}  ({dt:.2f}s)")

    def shutdown(self):
        """Block until all queued writes finish, then stop the worker thread."""
        if not self._worker.is_alive():
            return
        if not self._jobs.empty():
            print("[offline] waiting for pending writes to finish...")
        self._jobs.put(None)              # sentinel after any queued jobs (FIFO)
        self._worker.join()

    # ----------------------------------------------------------- markers -> rows
    def _marker_row_indices(self, row_times, markers):
        """Map each marker time to the nearest row index. Returns {row_idx: 'id'}."""
        out = {}
        if not row_times or not markers:
            return out
        for n, m in enumerate(markers, start=1):
            pos = bisect.bisect_left(row_times, m)
            cands = [c for c in (pos - 1, pos) if 0 <= c < len(row_times)]
            if not cands:
                continue
            best = min(cands, key=lambda c: abs(row_times[c] - m))
            out[best] = f"{out[best]};{n}" if best in out else str(n)
        return out

    # ----------------------------------------------------------- IMU file
    def _imu_header(self):
        # time_s   : device clock (sampleTimeFine), regular at the output rate.
        # host_recv_s: perf_counter arrival on the shared host clock (coarse
        #              cross-stream alignment / marker anchoring).
        cols = ["time_s", "host_recv_s", "marker"]
        for j in self.joints:
            for ax in self.AXES:                       # world angles, no derivatives
                cols.append(f"{j}_world_{ax}_deg")
            for ax in self.AXES:                       # intrinsic angles + derivatives
                cols += [f"{j}_intrinsic_{ax}_deg",
                         f"{j}_intrinsic_{ax}_vel_dps",
                         f"{j}_intrinsic_{ax}_acc_dps2"]
        return cols

    def _group_imu_frames(self, records):
        """
        Reconstruct synchronised frames on the DEVICE clock.

        records are (t_perf_host, dev_s, mac, R). sampleTimeFine is per-sensor
        (it counts from each sensor's own power-on), so every sensor is first
        normalised to its first sample in the recording; all sensors then share
        t=0 at recording start and run on their regular device grids.

        The sensor with the most samples is the frame reference. For each of its
        samples we take every other sensor's NEAREST sample within half a nominal
        period (zero-order hold if a sensor dropped that one). One reference
        sample -> one fully-populated frame, on a regular grid at the configured
        rate -- which is the actual fix for the jitter.

        Yields (t_ref_rel, t_perf_host, {mac: R}) in time order.
        """
        if not records:
            return []
        per_mac = defaultdict(list)             # mac -> [(dev_s, t_perf, R), ...]
        for t_perf, dev_s, mac, R in records:
            per_mac[mac].append((dev_s, t_perf, R))
        for mac in per_mac:
            per_mac[mac].sort(key=lambda x: x[0])   # by device time
        print(f"[offline] grouped {len(records)} IMU samples into {len(per_mac)} sensor streams.")
        num_samples = {mac : len(per_mac[mac]) for mac in per_mac}
        print(f"[offline] collected\n {num_samples}\n samples for each sensor.")
        macs = list(per_mac.keys())
        t0_dev = {m: per_mac[m][0][0] for m in macs}        # per-sensor normaliser
        ref = max(macs, key=lambda m: len(per_mac[m]))
        others = [m for m in macs if m != ref]
        tol = 0.5 / MOVELLA_OUTPUT_RATE

        ptr = {m: 0 for m in others}
        last_R = {m: per_mac[m][0][2] for m in macs}        # hold value per sensor

        frames = []
        for dev_s, t_perf, R_ref in per_mac[ref]:
            t_ref = dev_s - t0_dev[ref]
            mac_map = {ref: R_ref}
            for m in others:
                seq = per_mac[m]
                i = ptr[m]
                # monotonic walk to the sample nearest t_ref
                while (i + 1 < len(seq) and
                       abs((seq[i + 1][0] - t0_dev[m]) - t_ref) <=
                       abs((seq[i][0]     - t0_dev[m]) - t_ref)):
                    i += 1
                ptr[m] = i
                if abs((seq[i][0] - t0_dev[m]) - t_ref) <= tol:
                    last_R[m] = seq[i][2]       # fresh sample, within tolerance
                mac_map[m] = last_R[m]          # else hold previous (rare drop)
            frames.append((t_ref, t_perf, mac_map))
        return frames

    def _write_imu(self, path, records, calibration, markers, t0, keep_slice=None):
        frames = self._group_imu_frames(records)        # device-time ordered
        if keep_slice is not None:
            frames = frames[keep_slice]                     # integer slice AFTER ordering

        # Seed every joint with its calibration pose -> identity world rotation
        # until that joint's first real sample arrives.
        live_states = {j: calibration[j].copy() for j in self.joints}

        rows, host_times = [], []
        prev_t = prev_ang = prev_vel = None
        for t_ref, t_perf, mac_map in frames:
            new_samples = samples_by_joint(mac_map, self.joint_to_mac)
            live_states = update_live_states(live_states, new_samples, self.joints)
            frame = compute_frame(self.state_dict, self.joints, self.active_joints,
                                  self.endpoints, self.frame_change, calibration, live_states)
            jr = frame["joints"]
            world_ang = {j: kc.matrix_to_rpy_xyz(jr[j]["R"]) for j in self.joints}
            intrinsic_ang = {j: kc.matrix_to_rpy_xyz(jr[j]["R_intrinsic"]) for j in self.joints}

            # Finite-difference derivatives vs the previous frame, using the
            # regular DEVICE-clock dt (no longer the smeared arrival dt).
            if prev_t is None:
                vel = {j: (0.0, 0.0, 0.0) for j in self.joints}
                acc = {j: (0.0, 0.0, 0.0) for j in self.joints}
            else:
                dt = max(t_ref - prev_t, 1e-9)
                have_vel = prev_vel is not None
                vel, acc = {}, {}
                for j in self.joints:
                    v, a = [], []
                    for k in range(3):
                        vk = _wrap180(intrinsic_ang[j][k] - prev_ang[j][k]) / dt
                        v.append(vk)
                        a.append((vk - prev_vel[j][k]) / dt if have_vel else 0.0)
                    vel[j], acc[j] = tuple(v), tuple(a)

            rows.append((t_ref, t_perf - t0, world_ang, intrinsic_ang, vel, acc))
            host_times.append(t_perf)
            prev_t, prev_ang, prev_vel = t_ref, intrinsic_ang, vel

        marker_idx = self._marker_row_indices(host_times, markers)

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self._imu_header())
            for i, (t, host_s, world_ang, intrinsic_ang, vel, acc) in enumerate(rows):
                row = [f"{t:.6f}", f"{host_s:.6f}", marker_idx.get(i, "")]
                for j in self.joints:
                    for k in range(3):
                        row.append(f"{world_ang[j][k]:.4f}")
                    for k in range(3):
                        row += [f"{intrinsic_ang[j][k]:.4f}",
                                f"{vel[j][k]:.4f}",
                                f"{acc[j][k]:.4f}"]
                w.writerow(row)
        return len(rows)

    # ----------------------------------------------------------- EEG file
    def _eeg_header(self):
        # time_s     : LSL SOURCE clock (true sample time), regular at 256 Hz.
        # host_recv_s: perf_counter arrival, shared with the IMU file for coarse
        #              cross-stream alignment / marker anchoring.
        cols = ["time_s", "host_recv_s", "marker"]
        cols += [f"EEG_{ch}" for ch in EEG_CHANNELS]
        if WRITE_EEG_FEATURES:
            for ch in CHANNEL_NAMES:
                cols += [f"EEG_{ch}_theta_pw",
                         f"EEG_{ch}_alpha_pw",
                         f"EEG_{ch}_beta_pw"]
        return cols

    def _compute_eeg_features(self, samples):
        """Trailing-window band power (theta/alpha/beta x 8 channels) at stride.
        Rows before a full window, and rows skipped by EEG_FEATURE_STRIDE, stay
        NaN. Delegates to the vectorised eeg_processing.windowed_band_features
        (filter-once + one welch per window), ~25x faster than per-window calls.
        """
        n_feat = len(CHANNEL_NAMES) * 3
        n = len(samples)
        out = [[float("nan")] * n_feat for _ in range(n)]
        if n < EEG_FEATURE_WINDOW:
            return out
        arr = np.asarray(samples, float)
        idxs, feats = windowed_band_features(
            arr, EEG_FEATURE_WINDOW, EEG_FEATURE_STRIDE, fs=EEG_SAMPLE_RATE)
        for k, i in enumerate(idxs):
            out[int(i)] = feats[k].tolist()
        return out

    def _write_eeg(self, path, records, markers, t0, keep_slice= None):
        # Order by the TRUE sample time (LSL source timestamp, r[1]) -- not host
        # arrival -- so the integer slice keeps the first n_keep samples in time
        # and time_s is monotonic in the file.
        if keep_slice is None:
            keep_slice = slice(0, len(records))
        records = sorted(records, key=lambda r: r[1])
        n_ch = len(EEG_CHANNELS)
        host_times = [t_perf for t_perf, _, _ in records]
        marker_idx = self._marker_row_indices(host_times, markers)
        eeg_t0 = records[keep_slice.start][1] if records else 0.0      # first LSL source timestamp

        feat_cache = None
        if WRITE_EEG_FEATURES:
            feat_cache = self._compute_eeg_features([s for _, _, s in records])

        n_feat = len(CHANNEL_NAMES) * 3
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self._eeg_header())

            for i in range(keep_slice.start, keep_slice.stop):
                t_perf, eeg_ts, sample = records[i]
                vals = list(sample)[:n_ch]
                vals += [float("nan")] * (n_ch - len(vals))
                row = [f"{eeg_ts - eeg_t0:.6f}",        # time_s: LSL source clock
                       f"{t_perf - t0:.6f}",            # host_recv_s: shared host clock
                       marker_idx.get(i, "")]
                row += [f"{x:.4f}" for x in vals]
                if WRITE_EEG_FEATURES:
                    feats = feat_cache[i] if feat_cache is not None else [float("nan")] * n_feat
                    row += [f"{x:.6f}" for x in feats]
                w.writerow(row)
        return len(records)


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
# ------------------------------------------------------------------------------
# Identical to the live version EXCEPT that 'W'/'T' now drive the OfflineRecorder
# (arm/disarm + marker) instead of a periodic CSV writer. The recorder reads the
# raw capture lists directly, so this loop never touches files.
# ==============================================================================
def engine_loop(stop_event, reconfigure_event, frame_buffer, movella_buffer, recorder: OfflineRecorder,
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

        if recorder.recording and not recorder.movement_prompt_shown:
            if time.perf_counter() - recorder._t0 >= WAIT_TIME_BEFORE_MOVEMENT:
                print(f"F-ING MOVE NOW!")
                recorder.movement_prompt_shown = True

        # Hotkeys: R re-tare, A reconfigure, Q quit, W arm/disarm recording, T marker.
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
            # Start arms capture; stop snapshots + slices the data and hands it to
            # the background writer thread, returning immediately (no viewer stall).
            recorder.toggle(time.perf_counter(), calibration)
        elif key == "t":
            recorder.add_marker(time.perf_counter())

        frames += 1
        now = time.time()
        # if now - last_status >= STATUS_PERIOD_S:
        #     print(f"[engine] {frames / (now - last_status):5.1f} Hz")
        #     frames, last_status = 0, now

        time.sleep(0.002)


# ==============================================================================
# ONE TRACKING SESSION (steps 2 -> 3 -> 4). Returns True to reconfigure again.
# ==============================================================================
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

    recorder = OfflineRecorder(
        sorted_joints, active_joints, state_dict, endpoints,
        frame_change, joint_to_mac, eeg_buffer, movella_buffer, CSV_DIR)

    stop_event = threading.Event()
    reconfigure_event = threading.Event()

    producer = threading.Thread(
        target=engine_loop,
        args=(stop_event, reconfigure_event, frame_buffer, movella_buffer, recorder,
              state_dict, sorted_joints, active_joints, endpoints,
              frame_change, calibration, live_states, joint_to_mac),
        daemon=True,
    )
    producer.start()

    print("\n[!] Engine online.")
    print("    'R' = re-tare   |   'A' = change config   |   'Q' = quit")
    print("    'W' = start/stop recording (offline write on stop)   |   'T' = marker\n")

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

    # Session ending: stop the producer, enqueue any capture in progress, then
    # block until the background writer has flushed every queued job.
    stop_event.set()
    producer.join(timeout=1.0)
    recorder.finalize_if_recording()
    recorder.shutdown()

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
    # NOTE: the ONLY change vs the live engine's main() is using the recording
    # subclasses here. The producer loops are unchanged and call add_sample()
    # exactly as before.
    eeg_buffer = RecordingEEGBuffer()
    movella_buffer = RecordingMovellaBuffer()
    movella_stop = threading.Event()

    eeg_thread = threading.Thread(
        target=eeg_loop, 
        args=(eeg_buffer,), 
        daemon=True
    )
    
    movella_thread = threading.Thread(
        target=movella_loop,
        args=(movella_buffer, movella_stop, MAX_IMUS, MOVELLA_OUTPUT_RATE),
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