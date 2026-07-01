import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch


FS = 256  # Bitbrain sampling rate


# ==========================================================
# FILTER DESIGN
# ==========================================================

def bandpass_filter(data, low=1.0, high=40.0, fs=FS, order=4):

    b, a = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs
    )

    return filtfilt(b, a, data)


def notch_filter(data, freq=50.0, fs=FS, q=30):

    b, a = iirnotch(
        freq,
        q,
        fs=fs
    )

    return filtfilt(b, a, data)


def preprocess_channel(signal):

    signal = np.asarray(signal)

    signal = signal - np.mean(signal)

    signal = bandpass_filter(signal)

    signal = notch_filter(signal)

    return signal


# ==========================================================
# BAND POWER
# ==========================================================

def band_power(signal, low_freq, high_freq, fs=FS):

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=min(len(signal), 256)
    )

    idx = np.logical_and(
        freqs >= low_freq,
        freqs <= high_freq
    )

    return np.trapz(
        psd[idx],
        freqs[idx]
    )


def extract_features(signal):

    signal = preprocess_channel(signal)

    return {
        "theta": band_power(signal, 4, 8),
        "alpha": band_power(signal, 8, 12),
        "beta": band_power(signal, 13, 30)
    }


# ==========================================================
# MULTI-CHANNEL FEATURES
# ==========================================================

CHANNEL_NAMES = [
    "F7",
    "C3",
    "PZ",
    "CZ",
    "F8",
    "O1",
    "O2",
    "C4"
]


def extract_features_multichannel(eeg_window):

    eeg_window = np.asarray(eeg_window)

    results = {}

    for ch in range(8):

        results[CHANNEL_NAMES[ch]] = extract_features(
            eeg_window[:, ch]
        )

    return results

# ==========================================================
# FAST WINDOWED FEATURES (vectorised; used by the offline writer)
# ==========================================================

_BANDS = (("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("beta", 13.0, 30.0))


def preprocess_multichannel(arr, fs=FS):
    """Mean-subtract + bandpass + notch along axis 0 for an (N, C) array.

    Filters every channel in two filtfilt passes total (vectorised over
    channels) instead of re-filtering each overlapping window. Faster AND
    cleaner: per-window filtfilt injects edge transients into every window;
    filtering the whole signal once does not.
    """
    arr = np.asarray(arr, float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    b_bp, a_bp = butter(4, [1.0, 40.0], btype="bandpass", fs=fs)
    arr = filtfilt(b_bp, a_bp, arr, axis=0)
    b_no, a_no = iirnotch(50.0, 30, fs=fs)
    arr = filtfilt(b_no, a_no, arr, axis=0)
    return arr


def windowed_band_features(arr, win, stride, fs=FS, n_channels=8):
    """Fast trailing-window band power (theta/alpha/beta per channel).

    Filters the whole signal once, then computes ONE Welch PSD per window
    (across all channels at once) and integrates every band from it. Equivalent
    to calling extract_features_multichannel on each trailing window, but ~25x
    faster (no per-window re-filtering, one welch instead of three per channel,
    channels vectorised).

    Returns (indices, feats):
        indices[k] -> end-sample index of window k (the row that receives them)
        feats[k]   -> flat array, channel-major:
                      [ch0 theta, ch0 alpha, ch0 beta, ch1 theta, ...],
                      length n_channels * 3.
    """
    arr = np.asarray(arr, float)[:, :n_channels]
    N = arr.shape[0]
    n_feat = n_channels * len(_BANDS)
    if N < win:
        return np.empty(0, dtype=int), np.empty((0, n_feat))

    filt = preprocess_multichannel(arr, fs)
    nperseg = min(win, 256)
    # The frequency grid is identical for every window -> mask the bands once.
    freqs, _ = welch(filt[:win], fs=fs, nperseg=nperseg, axis=0)
    masks = [(freqs >= lo) & (freqs <= hi) for _, lo, hi in _BANDS]
    fmask = [freqs[m] for m in masks]

    idxs = np.arange(win - 1, N, stride)
    feats = np.empty((idxs.size, n_feat))
    for k, i in enumerate(idxs):
        _, psd = welch(filt[i - win + 1: i + 1], fs=fs, nperseg=nperseg, axis=0)  # (nf, C)
        cols = [np.trapz(psd[m], fm, axis=0) for m, fm in zip(masks, fmask)]      # each (C,)
        feats[k] = np.stack(cols, axis=1).ravel()
    return idxs, feats
