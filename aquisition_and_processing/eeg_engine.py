import time
from pylsl import StreamInlet, resolve_stream
from collections import deque
import threading


class EEGBuffer:

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = deque(maxlen=512)

    def add_sample(self, timestamp, sample):
        with self.lock:
            self.samples.append((timestamp, sample))

    def get_latest(self):
        with self.lock:
            if len(self.samples) == 0:
                return None
            return self.samples[-1]
    def get_window(self):

     with self.lock:

         if len(self.samples) < 512:
            return None

         return [sample for _, sample in self.samples]


def eeg_loop(buffer):

    print("Looking for EEG stream...")

    streams = resolve_stream('type', 'EEG')

    inlet = StreamInlet(streams[0])

    print("Connected to EEG stream")

    while True:

        sample, timestamp = inlet.pull_sample()
        

        buffer.add_sample(
            timestamp,
            sample
        )


# ==============================================================================
# RECORDING BUFFER
# ------------------------------------------------------------------------------
# This subclass the production buffer and add an unbounded capture list that
# is filled ONLY while armed. The base behaviour (latest sample / sliding window
# / connection handshake) is preserved for the live viewer and for movella_loop.
# This is the entire reason no other file needs to change.
# ==============================================================================
class RecordingEEGBuffer(EEGBuffer):
    """EEGBuffer + unbounded capture of every native sample while armed."""
    def __init__(self):
        super().__init__()
        self._rec_lock = threading.Lock()
        self._recording = False
        self._record = []          # list of (t_perf, eeg_ts, sample)

    def start_recording(self):
        with self._rec_lock:
            self._record = []
            self._recording = True

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False
            return list(self._record)

    def add_sample(self, timestamp, sample):
        # `timestamp` is the LSL SOURCE timestamp (a true sample time), captured
        # exactly as eeg_loop receives it from pull_sample(). t_perf is the host
        # arrival time, used only for coarse cross-stream alignment + markers.
        # Keep the base latest/window store working for anything that reads it.
        super().add_sample(timestamp, sample)
        if self._recording:
            t_perf = time.perf_counter()
            with self._rec_lock:
                if self._recording:
                    self._record.append((t_perf, timestamp, sample))
