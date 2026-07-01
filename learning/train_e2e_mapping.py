import os
import io
import base64
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from dataset import (
    get_file_pairs, subsample_file_pairs, CrossModalScaler,
    TimeSeriesDataset, resolve_feature_indices,
)
from models import EEGToMovementNetwork, TargetFeatureLoss
import metrics
import report_utils as ru

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_DIR = "./recordings"
OUTPUT_DIR = "./saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TYPE = "direct_conv1d" 
HIDDEN_DIM = 32
BATCH_SIZE = 32         
EPOCHS = 100
LR = 1e-3
SEED = 42

TRAIN_SUBSET_SIZE = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
torch.manual_seed(SEED)

EEG_LEN = 768   # 256 Hz x 3.0 s
IMU_LEN = 180   # 60 Hz  x 3.0 s

IMU_COLS = [
    'Shoulder_intrinsic_roll_deg', 'Shoulder_intrinsic_roll_vel_dps', 'Shoulder_intrinsic_roll_acc_dps2',
    'Shoulder_intrinsic_pitch_deg', 'Shoulder_intrinsic_pitch_vel_dps', 'Shoulder_intrinsic_pitch_acc_dps2',
    'Shoulder_intrinsic_yaw_deg', 'Shoulder_intrinsic_yaw_vel_dps', 'Shoulder_intrinsic_yaw_acc_dps2',
    'Elbow_intrinsic_roll_deg', 'Elbow_intrinsic_roll_vel_dps', 'Elbow_intrinsic_roll_acc_dps2',
    'Elbow_intrinsic_pitch_deg', 'Elbow_intrinsic_pitch_vel_dps', 'Elbow_intrinsic_pitch_acc_dps2',
    'Elbow_intrinsic_yaw_deg', 'Elbow_intrinsic_yaw_vel_dps', 'Elbow_intrinsic_yaw_acc_dps2',
    'Wrist_intrinsic_roll_deg', 'Wrist_intrinsic_roll_vel_dps', 'Wrist_intrinsic_roll_acc_dps2',
    'Wrist_intrinsic_pitch_deg', 'Wrist_intrinsic_pitch_vel_dps', 'Wrist_intrinsic_pitch_acc_dps2',
    'Wrist_intrinsic_yaw_deg', 'Wrist_intrinsic_yaw_vel_dps', 'Wrist_intrinsic_yaw_acc_dps2'
]
TARGET_IMU_COLS = [c for c in IMU_COLS if c.endswith('_vel_dps')]

EEG_VOLT_COLS = ['EEG_F7', 'EEG_C3', 'EEG_PZ', 'EEG_CZ', 'EEG_F8', 'EEG_O1', 'EEG_O2', 'EEG_C4', 'EEG_REF']
EEG_POW_COLS = [
    'EEG_F7_theta_pw', 'EEG_F7_alpha_pw', 'EEG_F7_beta_pw', 'EEG_C3_theta_pw', 'EEG_C3_alpha_pw', 'EEG_C3_beta_pw',
    'EEG_PZ_theta_pw', 'EEG_PZ_alpha_pw', 'EEG_PZ_beta_pw', 'EEG_CZ_theta_pw', 'EEG_CZ_alpha_pw', 'EEG_CZ_beta_pw',
    'EEG_F8_theta_pw', 'EEG_F8_alpha_pw', 'EEG_F8_beta_pw', 'EEG_O1_theta_pw', 'EEG_O1_alpha_pw', 'EEG_O1_beta_pw',
    'EEG_O2_theta_pw', 'EEG_O2_alpha_pw', 'EEG_O2_beta_pw', 'EEG_C4_theta_pw', 'EEG_C4_alpha_pw', 'EEG_C4_beta_pw'
]

# ==========================================
# 2. DATA PREPARATION
# ==========================================
all_pairs = get_file_pairs(DATA_DIR)
train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=SEED)
train_pairs = subsample_file_pairs(train_pairs, TRAIN_SUBSET_SIZE, seed=SEED)
print(f"Trials -> train: {len(train_pairs)} | val: {len(val_pairs)} (subset={TRAIN_SUBSET_SIZE})")

scaler = CrossModalScaler()
train_dataset = TimeSeriesDataset(
    file_pairs=train_pairs, scaler=scaler,
    eeg_volt_cols=EEG_VOLT_COLS, eeg_pow_cols=EEG_POW_COLS, imu_cols=IMU_COLS,
    eeg_len=EEG_LEN, imu_len=IMU_LEN, is_training=True,
)
val_dataset = TimeSeriesDataset(
    file_pairs=val_pairs, scaler=scaler,
    eeg_volt_cols=EEG_VOLT_COLS, eeg_pow_cols=EEG_POW_COLS, imu_cols=IMU_COLS,
    eeg_len=EEG_LEN, imu_len=IMU_LEN, is_training=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

RUN_DIR = ru.make_run_dir(
    OUTPUT_DIR, "eeg_to_movement_direct", MODEL_TYPE, f"n{len(train_dataset)}"
)
print("Experiment folder:", RUN_DIR)
joblib.dump(scaler, os.path.join(RUN_DIR, "fitted_scaler.pkl"))

sample_eeg, sample_imu = train_dataset[0]
EEG_CHANNELS = sample_eeg.shape[1]
IMU_CHANNELS = sample_imu.shape[1]

# ==========================================
# 3. MODEL, LOSS SETUP
# ==========================================
model = EEGToMovementNetwork(
    eeg_channels=EEG_CHANNELS, imu_channels=IMU_CHANNELS, 
    eeg_len=EEG_LEN, imu_len=IMU_LEN, hidden_dim=HIDDEN_DIM
).to(DEVICE)

target_idx = resolve_feature_indices(train_dataset.imu_cols_to_idx, TARGET_IMU_COLS)
criterion = TargetFeatureLoss(target_features_idx=target_idx, target_weight=0.95, auxiliary_weight=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
hist = {k: [] for k in ["tr_tgt", "tr_aux", "va_tgt", "va_aux"]}
best_val = float("inf")
best_path = os.path.join(RUN_DIR, "best_direct_eeg_to_movement.pth")

print(f"\nDirect training on {DEVICE} | EEG channels={EEG_CHANNELS}, IMU channels={IMU_CHANNELS}\n")
pbar = tqdm(range(EPOCHS), desc="Training")
for epoch in pbar:
    model.train()
    agg = {"tgt": 0.0, "aux": 0.0, "tot": 0.0, "n": 0}
    for eeg, imu in train_loader:
        eeg, imu = eeg.to(DEVICE), imu.to(DEVICE)
        optimizer.zero_grad()
        
        pred_imu = model(eeg)
        loss = criterion(pred_imu, imu)
        
        loss.backward()
        optimizer.step()
        
        t_mse, a_mse = metrics.split_target_aux_mse(pred_imu.detach(), imu, target_idx)
        agg["tgt"] += t_mse * eeg.size(0)
        agg["aux"] += a_mse * eeg.size(0)
        agg["tot"] += loss.item() * eeg.size(0)
        agg["n"] += eeg.size(0)

    model.eval()
    vagg = {"tgt": 0.0, "aux": 0.0, "tot": 0.0, "n": 0}
    recon_chunks, true_chunks = [], []
    with torch.no_grad():
        for eeg, imu in val_loader:
            eeg, imu = eeg.to(DEVICE), imu.to(DEVICE)
            
            pred_imu = model(eeg)
            loss = criterion(pred_imu, imu)
            
            t_mse, a_mse = metrics.split_target_aux_mse(pred_imu, imu, target_idx)
            vagg["tgt"] += t_mse * eeg.size(0)
            vagg["aux"] += a_mse * eeg.size(0)
            vagg["tot"] += loss.item() * eeg.size(0)
            vagg["n"] += eeg.size(0)
            
            recon_chunks.append(pred_imu.cpu())
            true_chunks.append(imu.cpu())

    hist["tr_tgt"].append(agg["tgt"] / agg["n"]);  hist["va_tgt"].append(vagg["tgt"] / vagg["n"])
    hist["tr_aux"].append(agg["aux"] / agg["n"]);  hist["va_aux"].append(vagg["aux"] / vagg["n"])

    val_tot = vagg["tot"] / vagg["n"]
    pbar.set_postfix({
        "val_tgt": f"{hist['va_tgt'][-1]:.6f}",
        "val_aux": f"{hist['va_aux'][-1]:.6f}",
        "val_tot": f"{val_tot:.6f}"
    })
          
    if val_tot < best_val:
        best_val = val_tot
        torch.save(model.state_dict(), best_path)
        pbar.write(f"Epoch {epoch+1:03d}: --> saved new best direct mapping model")

# ==========================================
# 5. EVALUATION IN PHYSICAL UNITS
# ==========================================
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
recon_all = torch.cat(recon_chunks)
true_all = torch.cat(true_chunks)

report = metrics.reconstruction_report(recon_all, true_all, val_dataset, TARGET_IMU_COLS)
print("\nDirect End-to-End Validation Performance (physical units):")
print(f"  target mean R^2 = {report['summary']['target_mean_r2']:.4f} | "
      f"target mean NRMSE = {report['summary']['target_mean_nrmse']:.4f}")

ru.save_metrics_json(RUN_DIR, {
    "phase": "direct_e2e", "run_dir": RUN_DIR, "model_type": MODEL_TYPE,
    "epochs": EPOCHS, "train_trials": len(train_dataset), "val_trials": len(val_dataset),
    "target_cols": TARGET_IMU_COLS, "best_val_loss": best_val,
    "summary": report["summary"],
    "per_feature": {c: {k: v for k, v in m.items()} for c, m in report["per_feature"].items()},
})

# ==========================================
# 6. EXPORT DIAGNOSTIC PLOT
# ==========================================
epochs_range = range(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_range, hist["tr_tgt"], color="#2563eb", lw=2, label="Train target MSE")
ax.plot(epochs_range, hist["va_tgt"], color="#1d4ed8", lw=2, ls="--", label="Val target MSE")
ax.plot(epochs_range, hist["tr_aux"], color="#f59e0b", lw=1.5, label="Train aux MSE")
ax.plot(epochs_range, hist["va_aux"], color="#b45309", lw=1.5, ls="--", label="Val aux MSE")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (scaled)")
ax.set_title("Direct End-to-End Tracking Error Trajectory")
ax.legend(); ax.grid(True, ls="--", alpha=0.5)
plt.tight_layout()

buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
plot_b64 = base64.b64encode(buf.getvalue()).decode(); plt.close()

# ==========================================
# 7. GENERATING THE EXPERIMENT DASHBOARD HTML
# ==========================================
# Reusing the HTML builder pattern found in Phase 1/Phase 2 reports.
arch_rows = [
    ("Architecture Paradigm", "Direct End-to-End Regression"),
    ("Compute Device", DEVICE.type.upper()),
    ("EEG Input Matrix", f"[{BATCH_SIZE}, {EEG_LEN}, {EEG_CHANNELS}]"),
    ("Direct IMU Output Matrix", f"[{BATCH_SIZE}, {IMU_LEN}, {IMU_CHANNELS}]"),
]
param_rows = [
    ("Direct Network", ru.format_param_count(sum(p.numel() for p in model.parameters())), 
     ru.format_param_count(sum(p.numel() for p in model.parameters() if p.requires_grad)))
]

html = ru.build_imu_report({
    "run_name": os.path.basename(RUN_DIR), "plot_b64": plot_b64, "report": report,
    "best_val_loss": best_val, "train_val_str": f"{len(train_dataset)} / {len(val_dataset)}",
    "window_note": "Direct Window Mapping | IMU 60 Hz (180 samples) vs EEG 256 Hz (768 samples)",
    "target_cols": TARGET_IMU_COLS,
    "kinematic_cols": train_dataset.imu_kinematic_cols, "angle_cols": train_dataset.imu_angle_cols,
    "config": [
        ("Model Paradigm", f"{MODEL_TYPE.upper()}"),
        ("Learning rate", str(LR)), ("Batch size", str(BATCH_SIZE)), ("Epochs", str(EPOCHS)),
        ("Loss weights (tgt/aux)", "0.9 / 0.1"),
    ],
    "data": [
        ("Train trials", str(len(train_dataset))), ("Val trials", str(len(val_dataset))),
        ("EEG features", str(EEG_CHANNELS)), ("IMU features", str(IMU_CHANNELS)),
    ],
    "arch_rows": arch_rows, "param_rows": param_rows,
})

report_path = os.path.join(RUN_DIR, "direct_e2e_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n--> Execution Completed successfully!\n--> Report: {report_path}\n--> Weights: {best_path}")