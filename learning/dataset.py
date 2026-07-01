import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler


def subsample_file_pairs(pairs: list, n: int | None, seed: int | None =None) -> list:
    """
    Deterministically keep at most `n` file pairs (for dataset-size studies).
    n=None or n>=len(pairs) returns the input unchanged. Selection is seeded so a
    learning curve is reproducible; the held-out validation set is left untouched
    by subsampling only the training pairs in the caller.
    """
    if n is None or n >= len(pairs):
        return pairs
    rng = random.Random(seed)
    chosen = rng.sample(range(len(pairs)), n)
    return [pairs[i] for i in sorted(chosen)]


def get_file_pairs(data_dir):
    """
    Finds and pairs matching EEG and IMU files based on their prefix.
    """
    eeg_files = sorted(glob.glob(os.path.join(data_dir, "*_eeg.csv")))
    pairs = []
    
    for eeg_path in eeg_files:
        base_name = os.path.basename(eeg_path)
        prefix = base_name.replace("_eeg.csv", "")
        imu_path = os.path.join(data_dir, f"{prefix}_imu.csv")
        
        if os.path.exists(imu_path):
            pairs.append({"prefix": prefix, "eeg": eeg_path, "imu": imu_path})
        else:
            print(f"Warning: No matching IMU file for {eeg_path}")
            
    return pairs


class CrossModalScaler:
    """
    Pure statistical normalizer. Operates on raw numeric arrays passed by the dataset,
    completely decoupled from pandas column name string keys.
    """
    def __init__(self):
        self.eeg_volt_scaler = RobustScaler()
        self.eeg_pow_scaler = StandardScaler()
        self.imu_scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, volt_vals, pow_vals, kinematic_vals):
        """Fits statistical parameters using raw training arrays."""
        self.eeg_volt_scaler.fit(volt_vals)
        
        # Apply the Log-transform to power bands prior to fitting variance
        log_pow = np.log1p(pow_vals)
        self.eeg_pow_scaler.fit(log_pow)
        
        if kinematic_vals.size > 0:
            self.imu_scaler.fit(kinematic_vals)
        
        self.is_fitted = True

    def transform_eeg(self, volt_vals, pow_vals):
        """Normalizes EEG blocks using pre-fit training distribution metrics."""
        volt_scaled = self.eeg_volt_scaler.transform(volt_vals)
        
        pow_log = np.log1p(pow_vals)
        pow_scaled = self.eeg_pow_scaler.transform(pow_log)
        
        return np.hstack([volt_scaled, pow_scaled]).astype(np.float32)

    def transform_imu_kinematics(self, kinematic_vals):
        """Normalizes linear kinematics (velocities/accelerations)."""
        if kinematic_vals.size == 0:
            return np.empty((len(kinematic_vals), 0), dtype=np.float32)
        return self.imu_scaler.transform(kinematic_vals).astype(np.float32) # type: ignore


def split_imu_columns(imu_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split IMU column names into angle (_deg) and kinematic groups."""
    angle_cols = [c for c in imu_cols if c.endswith('_deg')]
    kinematic_cols = [c for c in imu_cols if not c.endswith('_deg')]
    return angle_cols, kinematic_cols


def build_imu_tensor_from_arrays(
    kinematic_vals: np.ndarray | None,
    angle_vals_deg: np.ndarray | None,
) -> np.ndarray:
    """Stack kinematics, sin(angles), cos(angles) into the canonical IMU tensor layout."""
    imu_features = []
    if kinematic_vals is not None and kinematic_vals.size > 0:
        imu_features.append(kinematic_vals)
    if angle_vals_deg is not None and angle_vals_deg.size > 0:
        angles_rad = np.radians(angle_vals_deg)
        imu_features.append(np.sin(angles_rad))
        imu_features.append(np.cos(angles_rad))
    if not imu_features:
        raise ValueError("At least one IMU feature group must be non-empty.")
    return np.hstack(imu_features).astype(np.float32)


def build_imu_cols_to_idx(imu_cols: list[str]) -> dict[str, int | list[int]]:
    """
    Map logical IMU column names to tensor channel indices.

    Tensor layout: [kinematics | sin(angles) | cos(angles)]
    Kinematic columns map to a single index; angle columns map to [sin_idx, cos_idx].
    """
    angle_cols, kinematic_cols = split_imu_columns(imu_cols)
    n_kin = len(kinematic_cols)
    n_ang = len(angle_cols)

    idx_map: dict[str, int | list[int]] = {}
    for i, col in enumerate(kinematic_cols):
        idx_map[col] = i
    for i, col in enumerate(angle_cols):
        idx_map[col] = [n_kin + i, n_kin + n_ang + i]
    return idx_map


def resolve_feature_indices(cols_to_idx: dict[str, int | list[int]], target_cols: list[str]) -> list[int]:
    """Flatten column-to-index map entries into a list usable for tensor slicing."""
    indices: list[int] = []
    for col in target_cols:
        entry = cols_to_idx[col]
        if isinstance(entry, list):
            indices.extend(entry)
        else:
            indices.append(entry)
    return indices


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset that restricts execution strictly to selected columns 
    provided during instantiation, handling alignment, dropping, and cyclic encoding.
    """
    def __init__(self, file_pairs: list, scaler: CrossModalScaler, eeg_volt_cols: list[str], eeg_pow_cols: list[str], imu_cols: list[str], 
                 eeg_len: int, imu_len: int, is_training: bool = True):
        self.scaler = scaler
        self.eeg_data = []
        self.imu_data = []
        
        # Save explicitly targeted column constraints
        self.eeg_volt_cols = eeg_volt_cols
        self.eeg_pow_cols = eeg_pow_cols
        self.all_eeg_cols = eeg_volt_cols + eeg_pow_cols
        
        self.imu_angle_cols, self.imu_kinematic_cols = split_imu_columns(imu_cols)
        
        raw_eeg_dfs = []
        raw_imu_dfs = []
        
        # 1. Row/Length validation loop
        for pair in file_pairs:
            df_eeg = pd.read_csv(pair["eeg"])
            df_imu = pd.read_csv(pair["imu"])
            
            if len(df_eeg) < eeg_len or len(df_imu) < imu_len:
                print(f"Skipping trial {pair['prefix']}: Insufficient length "
                      f"(EEG: {len(df_eeg)}/{eeg_len}, IMU: {len(df_imu)}/{imu_len})")
                continue
            
            # Crop timeline lengths to match exact sequence boundaries
            raw_eeg_dfs.append(df_eeg.iloc[:eeg_len].copy())
            raw_imu_dfs.append(df_imu.iloc[:imu_len].copy())
            
        if not raw_eeg_dfs:
            raise RuntimeError("No file pairs matched the specified sequence length constraints.")
            
        # 2. Extract and pass data arrays to fit the scaler globally if training
        if is_training and not self.scaler.is_fitted:
            concat_eeg = pd.concat(raw_eeg_dfs, ignore_index=True)
            concat_imu = pd.concat(raw_imu_dfs, ignore_index=True)
            
            # Pull strict raw matrix values based on training script columns
            v_vals = concat_eeg[self.eeg_volt_cols].values
            p_vals = concat_eeg[self.eeg_pow_cols].values
            k_vals = concat_imu[self.imu_kinematic_cols].values if self.imu_kinematic_cols else np.empty((0,0))
            
            self.scaler.fit(v_vals, p_vals, k_vals)
            
        # 3. Transform data using only designated columns and build tensors
        for df_eeg, df_imu in zip(raw_eeg_dfs, raw_imu_dfs):
            # Split and normalize EEG channels
            v_vals = df_eeg[self.eeg_volt_cols].values
            p_vals = df_eeg[self.eeg_pow_cols].values
            scaled_eeg = self.scaler.transform_eeg(v_vals, p_vals)
            eeg_tensor = torch.tensor(scaled_eeg, dtype=torch.float32)
            
            k_vals = (
                self.scaler.transform_imu_kinematics(df_imu[self.imu_kinematic_cols].values)
                if self.imu_kinematic_cols else None
            )
            angle_vals = df_imu[self.imu_angle_cols].values if self.imu_angle_cols else None
            imu_tensor = torch.tensor(build_imu_tensor_from_arrays(k_vals, angle_vals), dtype=torch.float32)
            
            self.eeg_data.append(eeg_tensor)
            self.imu_data.append(imu_tensor)

        self.imu_cols_to_idx = build_imu_cols_to_idx(imu_cols)
        self.eeg_cols_to_idx = {col: idx for idx, col in enumerate(self.all_eeg_cols)}

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.imu_data[idx]
    

class DummyDataset(Dataset):
    """Synthetic dataset mirroring TimeSeriesDataset channel layout and preprocessing."""
    def __init__(
        self,
        num_samples: int,
        eeg_volt_cols: list[str],
        eeg_pow_cols: list[str],
        imu_cols: list[str],
        eeg_len: int,
        imu_len: int,
    ):
        self.num_samples = num_samples
        self.eeg_len = eeg_len
        self.imu_len = imu_len
        self.eeg_volt_cols = eeg_volt_cols
        self.eeg_pow_cols = eeg_pow_cols
        self.all_eeg_cols = eeg_volt_cols + eeg_pow_cols
        self.imu_angle_cols, self.imu_kinematic_cols = split_imu_columns(imu_cols)
        self.imu_cols_to_idx = build_imu_cols_to_idx(imu_cols)
        self.eeg_cols_to_idx = {col: idx for idx, col in enumerate(self.all_eeg_cols)}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        eeg_dummy = torch.randn(self.eeg_len, len(self.all_eeg_cols))
        k_vals = (
            np.random.randn(self.imu_len, len(self.imu_kinematic_cols)).astype(np.float32)
            if self.imu_kinematic_cols else None
        )
        angle_vals = (
            np.random.uniform(-180.0, 180.0, (self.imu_len, len(self.imu_angle_cols))).astype(np.float32)
            if self.imu_angle_cols else None
        )
        imu_dummy = torch.tensor(build_imu_tensor_from_arrays(k_vals, angle_vals), dtype=torch.float32)
        return eeg_dummy, imu_dummy