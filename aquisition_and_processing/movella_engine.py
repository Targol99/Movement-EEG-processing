"""
movella_engine.py
=================
Movella DOT hardware producer. Mirrors eeg_engine.py: a thread-safe buffer
class plus a loop function that runs on its own thread and owns ALL Movella
PC SDK I/O (scan, connect, configure, stream).

The rest of the pipeline never touches the SDK. Consumers read the most recent
orientation per sensor (keyed by MAC) from `MovellaBuffer`, exactly like the
EEG side reads the latest EEG sample from `EEGBuffer`.

Orientations are stored as 3x3 rotation matrices already converted out of the
SDK's Euler representation, so downstream code stays MAC- and SDK-agnostic.
"""

import sys
import time
import threading

import aquisition_and_processing.kinematics_core as kc

# Movella PC SDK (only this file depends on it)
sys.path.append(r"C:\Program Files\Movella\DOT PC SDK 2023.6\SDK Files\Examples\python")
from xdpchandler import *  # type: ignore  # noqa: F401,F403


# ==============================================================================
# THREAD-SAFE ORIENTATION BUFFER (keyed by MAC)
# ==============================================================================
class MovellaBuffer:
    """
    Latest-sample-wins store of sensor orientations, keyed by MAC. Thread-safe.

    The streaming thread calls add_sample(); consumers call get_orientations()
    to read the most recent rotation matrix for every connected sensor. The set
    of connected MACs and a readiness Event are exposed so the main thread can
    block until the (one-time) hardware connection has completed.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._latest = {}            # mac -> (timestamp, R 3x3 ndarray)
        self._connected_macs = []
        self._ready = threading.Event()

    # ---- connection handshake (set once by the streaming thread) ----
    def set_connected(self, macs):
        """Publish the connected MAC list and signal readiness. Called once."""
        with self.lock:
            self._connected_macs = list(macs)
        self._ready.set()

    def wait_until_ready(self, timeout=None):
        """Block until the streaming thread has finished connecting."""
        return self._ready.wait(timeout)

    def get_macs(self):
        with self.lock:
            return list(self._connected_macs)

    # ---- live samples ----
    def add_sample(self, mac, timestamp, R):
        with self.lock:
            self._latest[mac] = (timestamp, R)

    def get_latest(self):
        """{mac: (timestamp, R)} snapshot of the most recent sample per sensor."""
        with self.lock:
            return dict(self._latest)

    def get_orientations(self):
        """{mac: R} snapshot — convenience for consumers that only need rotations."""
        with self.lock:
            return {mac: val[1] for mac, val in self._latest.items()}
        
# ==============================================================================
# RECORDING BUFFER
# ------------------------------------------------------------------------------
# This subclass the production buffer and add an unbounded capture list that
# is filled ONLY while armed. The base behaviour (latest sample / sliding window
# / connection handshake) is preserved for the live viewer and for movella_loop.
# This is the entire reason no other file needs to change.
# ==============================================================================

class RecordingMovellaBuffer(MovellaBuffer):
    """MovellaBuffer + unbounded capture of every native sample while armed."""
    def __init__(self):
        super().__init__()
        self._rec_lock = threading.Lock()
        self._recording = False
        self._record = []          # list of (t_perf, dev_s, mac, R)

    def start_recording(self):
        with self._rec_lock:
            self._record = []
            self._recording = True

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False
            return list(self._record)

    def add_sample(self, mac, timestamp, R):
        # `timestamp` is now the sensor's DEVICE time in seconds (unwrapped
        # sampleTimeFine), supplied by the patched movella_loop. t_perf is host
        # arrival, used only for coarse cross-stream alignment + markers.
        # Keep the base latest-per-MAC store working for the live viewer.
        super().add_sample(mac, timestamp, R)
        if self._recording:
            t_perf = time.perf_counter()
            with self._rec_lock:
                if self._recording:
                    self._record.append((t_perf, timestamp, mac, R))



# ==============================================================================
# HARDWARE ORCHESTRATION (SDK I/O lives only in this file)
# ==============================================================================
def initialize_hardware_handler():
    handler = XdpcHandler()  # type: ignore
    if not handler.initialize():
        raise RuntimeError("Failed to initialize Movella DOT PC SDK.")
    return handler


def connect_and_configure_imus(handler, max_imus, output_rate):
    """Scan, connect, and start measurement. Returns the list of connected MACs."""
    print("[1/3] Scanning for nearby Movella DOT devices...")
    handler.scanForDots()
    time.sleep(3.0)

    detected = handler.detectedDots()
    if len(detected) == 0:
        raise RuntimeError("No hardware targets found during scan.")

    print(f"[2/3] Found {len(detected)} sensors. Connecting (Cap: {max_imus})...")
    handler.connectDots()
    time.sleep(6.0)

    connected = handler.connectedDots()
    if len(connected) == 0:
        raise RuntimeError("Connection routine timed out.")

    active_sensors = connected[:max_imus]
    connected_macs = [s.portInfo().bluetoothAddress() for s in active_sensors]

    print("\n[Hardware Matrix Status] Active Links:")
    for mac in connected_macs:
        print(f"  -> {mac}")

    for sensor in active_sensors:
        sensor.setOnboardFilterProfile("General")
        sensor.setOutputRate(output_rate)
        if not sensor.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):  # type: ignore # noqa: F405
            raise RuntimeError("Measurement streaming failed.")

    return connected_macs


def flush_buffer_noise(handler, connected_macs, duration_s=1.0):
    """Drain stale packets accumulated before/while configuring."""
    flush_start = time.time()
    while time.time() - flush_start < duration_s:
        if handler.packetsAvailable():
            for mac in connected_macs:
                handler.getNextPacket(mac)
        time.sleep(0.01)


def capture_raw_sensor_orientations(handler, macs):
    """
    Pull any available orientation packets. Returns {mac: (sample_time_us, R)}
    for the sensors that produced a fresh orientation packet.

    sample_time_us is the sensor's OWN device timestamp (packet.sampleTimeFine(),
    microseconds, per the Movella DOT manual). It is regular at the configured
    output rate -- unlike wall-clock arrival time, which is smeared by Bluetooth
    buffering. It counts from sensor power-on and is therefore per-sensor; it also
    wraps at 2**32 us (~1.2 h). The streaming loop unwraps it and the recorder
    normalises each sensor to recording start, so neither the wrap nor the
    per-sensor epoch reaches downstream code.
    """
    raw_matrices = {}
    if handler.packetsAvailable():
        for mac in macs:
            packet = handler.getNextPacket(mac)
            if packet is not None and packet.containsOrientation():
                euler = packet.orientationEuler()
                R = kc.rpy_to_rotation_matrix_xyz(
                    euler.roll(), euler.pitch(), euler.yaw())
                raw_matrices[mac] = (packet.sampleTimeFine(), R)
    return raw_matrices


# ==============================================================================
# STREAMING THREAD (mirrors eeg_loop)
# ==============================================================================
def movella_loop(
    buffer: MovellaBuffer, stop_event: threading.Event, max_imus: int, output_rate: int):
    """
    Connect the IMUs ONCE, then stream forever until stop_event is set.

    The connection is performed inside this thread; the connected MAC list and a
    readiness signal are published via the buffer so the main thread can wait on
    them. On any fatal error the buffer is still marked ready (with an empty MAC
    list) so the main thread is never left blocked.
    """
    print("Looking for Movella DOT sensors...")
    handler = None
    try:
        handler = initialize_hardware_handler()
        connected_macs = connect_and_configure_imus(handler, max_imus, output_rate)
        flush_buffer_noise(handler, connected_macs)
        buffer.set_connected(connected_macs)   # signals ready -> unblocks main
        print(f"Connected to {len(connected_macs)} Movella DOT sensor(s); streaming.")

        # Per-MAC state for unwrapping the 32-bit microsecond sampleTimeFine.
        _US_WRAP = 1 << 32
        last_raw_us = {}      # mac -> last raw sampleTimeFine seen
        wrap_count = {}       # mac -> number of rollovers so far
        while not stop_event.is_set():
            orientations = capture_raw_sensor_orientations(handler, connected_macs)
            if orientations:
                for mac, (st_us, R) in orientations.items():
                    prev = last_raw_us.get(mac)
                    if prev is not None and st_us < prev:
                        wrap_count[mac] = wrap_count.get(mac, 0) + 1
                    last_raw_us[mac] = st_us
                    # Device sample time in seconds (monotonic per sensor). This,
                    # not arrival time, is what downstream timing is built from.
                    device_ts = (st_us + wrap_count.get(mac, 0) * _US_WRAP) * 1e-4
                    buffer.add_sample(mac, device_ts, R)

    except Exception as e:
        print(f"[movella] fatal error: {e}")
        buffer.set_connected([])   # unblock main with an empty MAC list -> error
    finally:
        if handler is not None:
            handler.cleanup()
            print("-> Movella SDK resources released. Connections closed.")
