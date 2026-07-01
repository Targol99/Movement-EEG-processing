import os
import io
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from dataset import (
    get_file_pairs, subsample_file_pairs, TimeSeriesDataset, resolve_feature_indices,
)
from models import IMUAutoencoder, EEGMappingNetwork
import metrics
import report_utils as ru

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "./recordings"

# ---- Reference the Phase 1 experiment folder here (created by train_imu_ae.py) ----
PHASE1_DIR = "./saved_models/imu_ae_conv1d_vae_ld16_n674_2506_1654"
# -----------------------------------------------------------------------------------

# Must match the Phase 1 run referenced above.
MODEL_TYPE = "conv1d"
LATENT_TYPE = "vae"
LATENT_DIM = 16

# Hyperparameters for the EEG->IMU mapping network (Phase 2).
HIDDEN_DIM = 16
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
SEED = None

# Dataset-size study (training trials only; validation stays fixed).
TRAIN_SUBSET_SIZE = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if SEED is not None:
    torch.manual_seed(SEED)

EEG_LEN = 768
IMU_LEN = 180

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
# The features whose decoded reconstruction you care about. Vary this and compare
# the report's "End-to-end target R^2" across runs to find which features map best.
TARGET_IMU_COLS = [c for c in IMU_COLS if c.startswith("Shoulder") and (c.endswith('_vel_dps') or c.endswith('_deg'))]

EEG_VOLT_COLS = ['EEG_F7', 'EEG_C3', 'EEG_PZ', 'EEG_CZ', 'EEG_F8', 'EEG_O1', 'EEG_O2', 'EEG_C4', 'EEG_REF']
EEG_POW_COLS = [
    'EEG_F7_theta_pw', 'EEG_F7_alpha_pw', 'EEG_F7_beta_pw', 'EEG_C3_theta_pw', 'EEG_C3_alpha_pw', 'EEG_C3_beta_pw',
    'EEG_PZ_theta_pw', 'EEG_PZ_alpha_pw', 'EEG_PZ_beta_pw', 'EEG_CZ_theta_pw', 'EEG_CZ_alpha_pw', 'EEG_CZ_beta_pw',
    'EEG_F8_theta_pw', 'EEG_F8_alpha_pw', 'EEG_F8_beta_pw', 'EEG_O1_theta_pw', 'EEG_O1_alpha_pw', 'EEG_O1_beta_pw',
    'EEG_O2_theta_pw', 'EEG_O2_alpha_pw', 'EEG_O2_beta_pw', 'EEG_C4_theta_pw', 'EEG_C4_alpha_pw', 'EEG_C4_beta_pw'
]

# ==========================================
# 2. DATA + SCALER RECOVERY (from PHASE1_DIR)
# ==========================================
scaler_path = os.path.join(PHASE1_DIR, "fitted_scaler.pkl")
weights_path = os.path.join(PHASE1_DIR, "best_imu_autoencoder.pth")
for p in (scaler_path, weights_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing Phase 1 artifact: {p}. Check PHASE1_DIR.")
scaler = joblib.load(scaler_path)

all_pairs = get_file_pairs(DATA_DIR)
train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=SEED)
train_pairs = subsample_file_pairs(train_pairs, TRAIN_SUBSET_SIZE, seed=SEED)
print(f"Trials -> train: {len(train_pairs)} | val: {len(val_pairs)}")

# is_training=False everywhere: the frozen Phase 1 scaler must not be refit.
train_dataset = TimeSeriesDataset(
    file_pairs=train_pairs, scaler=scaler, eeg_volt_cols=EEG_VOLT_COLS, eeg_pow_cols=EEG_POW_COLS,
    imu_cols=IMU_COLS, eeg_len=EEG_LEN, imu_len=IMU_LEN, is_training=False,
)
val_dataset = TimeSeriesDataset(
    file_pairs=val_pairs, scaler=scaler, eeg_volt_cols=EEG_VOLT_COLS, eeg_pow_cols=EEG_POW_COLS,
    imu_cols=IMU_COLS, eeg_len=EEG_LEN, imu_len=IMU_LEN, is_training=False,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

sample_eeg, sample_imu = train_dataset[0]
EEG_IN_CHANNELS = sample_eeg.shape[1]
IMU_IN_CHANNELS = sample_imu.shape[1]
TARGET_IMU_LEN = sample_imu.shape[0]
target_idx = resolve_feature_indices(train_dataset.imu_cols_to_idx, TARGET_IMU_COLS)

# Phase 2 output folder lives inside the Phase 1 run for clean lineage.
RUN_DIR = ru.make_run_dir(PHASE1_DIR, "phase2_eeg", MODEL_TYPE, f"n{len(train_dataset)}")
print("experiment folder:", RUN_DIR)

# ==========================================
# 3. FROZEN IMU AE + TRAINABLE EEG MAPPER
# ==========================================
imu_model = IMUAutoencoder(
    in_channels=IMU_IN_CHANNELS, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
    target_len=TARGET_IMU_LEN, model_type=MODEL_TYPE, latent_type=LATENT_TYPE,
).to(DEVICE)
imu_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
imu_model.eval()
for p in imu_model.parameters():
    p.requires_grad = False

eeg_model = EEGMappingNetwork(
    in_channels=EEG_IN_CHANNELS, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
    sequence_length=EEG_LEN, model_type=MODEL_TYPE,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(eeg_model.parameters(), lr=LR)


def encode_target(imu):
    if LATENT_TYPE == "vae":
        mu, _ = imu_model.encoder(imu)  # map to the stable mean
        return mu
    return imu_model.encoder(imu)


# ==========================================
# 4. TRAINING LOOP (latent MSE trains; e2e R^2 tracked, not trained)
# ==========================================
hist = {"tr_mse": [], "va_mse": [], "va_e2e_r2": []}
best_val = float("inf")
best_path = os.path.join(RUN_DIR, "best_eeg_mapping_network.pth")

print(f"\nPhase 2 mapping on {DEVICE} | EEG {EEG_IN_CHANNELS}ch -> latent {LATENT_DIM}\n")
pbar = tqdm(range(EPOCHS), desc="Training")
for epoch in pbar:
    eeg_model.train()
    tr_loss, tr_n = 0.0, 0
    for eeg, imu in train_loader:
        eeg, imu = eeg.to(DEVICE), imu.to(DEVICE)
        with torch.no_grad():
            target_z = encode_target(imu)
        optimizer.zero_grad()
        loss = criterion(eeg_model(eeg), target_z)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * eeg.size(0); tr_n += eeg.size(0)

    eeg_model.eval()
    va_loss, va_n = 0.0, 0
    recon_chunks, true_chunks = [], []
    with torch.no_grad():
        for eeg, imu in val_loader:
            eeg, imu = eeg.to(DEVICE), imu.to(DEVICE)
            target_z = encode_target(imu)
            pred_z = eeg_model(eeg)
            va_loss += criterion(pred_z, target_z).item() * eeg.size(0); va_n += eeg.size(0)
            recon_chunks.append(imu_model.decoder(pred_z).cpu())
            true_chunks.append(imu.cpu())

    hist["tr_mse"].append(tr_loss / tr_n)
    hist["va_mse"].append(va_loss / va_n)
    recon_val = torch.cat(recon_chunks); true_val = torch.cat(true_chunks)
    hist["va_e2e_r2"].append(metrics.target_r2_scaled(recon_val, true_val, target_idx))

    pbar.set_postfix({
        "tr_mse": f"{hist['tr_mse'][-1]:.6f}",
        "va_mse": f"{hist['va_mse'][-1]:.6f}",
        "va_e2e_r2": f"{hist['va_e2e_r2'][-1]:.4f}"
    })
    if hist["va_mse"][-1] < best_val:
        best_val = hist["va_mse"][-1]
        torch.save(eeg_model.state_dict(), best_path)
        pbar.write(f"Epoch {epoch+1:03d}: --> saved new best EEG mapping network")

# ==========================================
# 5. FINAL END-TO-END PHYSICAL METRICS (best checkpoint)
# ==========================================
eeg_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
recon_all, true_all = metrics.collect_eeg_e2e_reconstruction(eeg_model, imu_model, val_loader, LATENT_TYPE, DEVICE)
report = metrics.reconstruction_report(recon_all, true_all, val_dataset, TARGET_IMU_COLS)
print("\nEnd-to-end EEG->IMU reconstruction (physical units):")
print(f"  target mean R^2 = {report['summary']['target_mean_r2']:.4f} | "
      f"target mean RMSE = {report['summary']['target_mean_rmse']:.4f} dps")

ru.save_metrics_json(RUN_DIR, {
    "phase": 2, "run_dir": RUN_DIR, "phase1_dir": PHASE1_DIR,
    "model_type": MODEL_TYPE, "latent_type": LATENT_TYPE, "latent_dim": LATENT_DIM,
    "epochs": EPOCHS, "train_trials": len(train_dataset), "val_trials": len(val_dataset),
    "target_cols": TARGET_IMU_COLS, "best_val_latent_mse": best_val,
    "summary": report["summary"],
    "per_feature": {c: {k: v for k, v in m.items()} for c, m in report["per_feature"].items()},
})

# ==========================================
# 6. PLOTS (latent MSE + tracked end-to-end target R^2)
# ==========================================
epochs = range(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist["tr_mse"], color="#4f46e5", lw=2, label="Train latent MSE")
ax.plot(epochs, hist["va_mse"], color="#f59e0b", lw=2, label="Val latent MSE")
ax.set_xlabel("Epoch"); ax.set_ylabel("Latent MSE (training signal)")
ax.grid(True, ls="--", alpha=0.5)
ax2 = ax.twinx()
ax2.plot(epochs, hist["va_e2e_r2"], color="#16a34a", lw=1.8, ls="--", label="Val e2e target R^2")
ax2.set_ylabel("End-to-end target R^2 (tracked)")
ax.set_title(f"EEG->IMU mapping ({MODEL_TYPE.upper()})")
l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, loc="center right")
plt.tight_layout()
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
plot_b64 = base64.b64encode(buf.getvalue()).decode(); plt.close()

# ==========================================
# 7. DASHBOARD
# ==========================================
info = ru.probe_eeg_mapping(eeg_model, imu_model, sample_eeg, sample_imu, LATENT_TYPE, DEVICE)
arch_rows = [
    ("EEG backbone", MODEL_TYPE.upper()), ("Target latent (frozen)", LATENT_TYPE.upper()),
    ("Shared latent dim", str(LATENT_DIM)), ("EEG hidden dim", str(info.get("eeg_hidden_dim", "N/A"))),
    ("Compute device", DEVICE.type.upper()),
    ("EEG input", info["eeg_input_shape"]), ("IMU input", info["imu_input_shape"]),
    ("Frozen IMU latent target", info["target_latent_shape"]),
    ("EEG mapper output", info["predicted_latent_shape"]),
]
param_rows = [
    ("Frozen IMU AE", ru.format_param_count(info["imu_total_params"]), "0"),
    ("Trainable EEG mapper", ru.format_param_count(info["eeg_params"]), ru.format_param_count(info["eeg_params_trainable"])),
]
html = ru.build_eeg_report({
    "run_name": os.path.basename(RUN_DIR), "phase1_dir": PHASE1_DIR, "plot_b64": plot_b64,
    "report": report, "best_val_latent_mse": best_val,
    "train_val_str": f"{len(train_dataset)} / {len(val_dataset)}",
    "window_note": "Window 3.0 s | IMU 60 Hz (180 samples), EEG 256 Hz (768 samples)",
    "target_cols": TARGET_IMU_COLS,
    "config": [
        ("EEG backbone", MODEL_TYPE.upper()), ("Latent type (frozen)", LATENT_TYPE.upper()),
        ("Latent dim", str(LATENT_DIM)), ("Hidden dim", str(HIDDEN_DIM)),
        ("Learning rate", str(LR)), ("Batch size", str(BATCH_SIZE)), ("Epochs", str(EPOCHS)),
        ("Training signal", "latent MSE (mu)"),
    ],
    "data": [
        ("Train trials", str(len(train_dataset))), ("Val trials", str(len(val_dataset))),
        ("Subset size", str(TRAIN_SUBSET_SIZE)),
        ("EEG channels", str(EEG_IN_CHANNELS)), ("IMU channels", str(IMU_IN_CHANNELS)),
        ("EEG seq length", str(EEG_LEN)), ("IMU seq length", str(IMU_LEN)),
        ("# target features", str(len(TARGET_IMU_COLS))),
    ],
    "arch_rows": arch_rows, "param_rows": param_rows,
})
report_path = os.path.join(RUN_DIR, "phase2_experiment_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n--> report: {report_path}\n--> weights: {best_path}\n--> metrics.json saved in {RUN_DIR}")

