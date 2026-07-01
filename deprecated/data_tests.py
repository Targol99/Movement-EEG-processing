import pandas as pd
import numpy as np

recording = pd.read_csv('recordings\\recording_20260616_180913_eeg.csv')

# get times
times = np.asarray(recording["time_s"]) 
dts = np.diff(times)

dt_mean = np.mean(dts)
dt_std = np.std(dts)
dt_span = np.max(dts) - np.min(dts)

freq_mean = 1 / dt_mean
freq_std = dt_std / (dt_mean ** 2)
freq_span = 1 / np.min(dts) - 1 / np.max(dts)

print(f"Mean dt: {dt_mean:.6f} s, std: {dt_std:.6f} s, span: {dt_span:.6f} s")
print(f"Mean freq: {freq_mean:.2f} Hz, std: {freq_std:.2f} Hz, span: {freq_span:.2f} Hz")
print(f"dts: {dts} s")