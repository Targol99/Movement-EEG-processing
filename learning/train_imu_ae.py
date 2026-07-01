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
from models import IMUAutoencoder, TargetFeatureLoss
import metrics
import report_utils as ru

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_DIR = "./recordings"
OUTPUT_DIR = "./saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TYPE = "conv1d"   # "conv1d" or "lstm"
LATENT_TYPE = "vae"     # "ae" or "vae"
LATENT_DIM = 16
HIDDEN_DIM = 16
BATCH_SIZE = 32         # 221 trials -> 6 full batches + 1 of 29 (safe for BatchNorm)
EPOCHS = 100
LR = 1e-3
SEED = None

# Loss weights: target features get more weight than auxiliary features.
TARGET_WEIGHT = 0.9
AUX_WEIGHT = 0.1

# VAE KLD linear annealing
KLD_MAX_WEIGHT = 0.007
KLD_START_EPOCH = 20
KLD_END_EPOCH = 60

# Dataset-size study: cap the number of TRAINING trials (validation stays fixed).
# None = use all available training trials.
TRAIN_SUBSET_SIZE = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", DEVICE)
if SEED is not None:
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
# Which channels the autoencoder should focus on. Change this to study which
# target sets are easiest to reconstruct (compare the report's "Target mean R2").
TARGET_IMU_COLS = [c for c in IMU_COLS if c.startswith("Shoulder") and (c.endswith('_vel_dps') or c.endswith('_deg'))]

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

# Create the experiment folder up front and stash the scaler inside it.
RUN_DIR = ru.make_run_dir(
    OUTPUT_DIR, "imu_ae", MODEL_TYPE, LATENT_TYPE, f"ld{LATENT_DIM}", f"n{len(train_dataset)}"
)
print("experiment folder:", RUN_DIR)
joblib.dump(scaler, os.path.join(RUN_DIR, "fitted_scaler.pkl"))

_, sample_imu = train_dataset[0]
TARGET_LEN = sample_imu.shape[0]
IN_CHANNELS = sample_imu.shape[1]

# ==========================================
# 3. MODEL, LOSS, METRIC INDICES
# ==========================================
model = IMUAutoencoder(
    in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
    target_len=TARGET_LEN, model_type=MODEL_TYPE, latent_type=LATENT_TYPE,
).to(DEVICE)

target_idx = resolve_feature_indices(train_dataset.imu_cols_to_idx, TARGET_IMU_COLS)
criterion = TargetFeatureLoss(target_features_idx=target_idx, target_weight=TARGET_WEIGHT, auxiliary_weight=AUX_WEIGHT)
optimizer = optim.Adam(model.parameters(), lr=LR)


def kld_term(mu, logvar):
    return torch.clip(-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()), min=-1000, max=1000)


# ==========================================
# 4. TRAINING LOOP (tracks target vs auxiliary MSE)
# ==========================================
hist = {k: [] for k in ["tr_tgt", "tr_aux", "va_tgt", "va_aux", "tr_kld", "va_kld", "kld_w"]}
best_val = float("inf")
best_path = os.path.join(RUN_DIR, "best_imu_autoencoder.pth")

print(f"\nTraining on {DEVICE} | IMU channels={IN_CHANNELS}, T={TARGET_LEN}\n")

pbar = tqdm(range(EPOCHS), desc="Training")

for epoch in pbar:
    if epoch < KLD_START_EPOCH:
        kld_w = 0.0
    elif epoch > KLD_END_EPOCH:
        kld_w = KLD_MAX_WEIGHT
    else:
        kld_w = KLD_MAX_WEIGHT * (epoch - KLD_START_EPOCH) / (KLD_END_EPOCH - KLD_START_EPOCH)
    hist["kld_w"].append(kld_w)

    model.train()
    agg = {"tgt": 0.0, "aux": 0.0, "kld": 0.0, "tot": 0.0, "n": 0}
    for _, imu in train_loader:
        imu = imu.to(DEVICE)
        optimizer.zero_grad()
        if LATENT_TYPE == "vae":
            recon, mu, logvar = model(imu)
            kld = kld_term(mu, logvar)
            loss = criterion(recon, imu) + kld_w * kld
            agg["kld"] += kld.item() * imu.size(0)
        else:
            recon, _ = model(imu)
            loss = criterion(recon, imu)
        loss.backward()
        optimizer.step()
        t_mse, a_mse = metrics.split_target_aux_mse(recon.detach(), imu, target_idx)
        agg["tgt"] += t_mse * imu.size(0)
        agg["aux"] += a_mse * imu.size(0)
        agg["tot"] += loss.item() * imu.size(0)
        agg["n"] += imu.size(0)

    model.eval()
    vagg = {"tgt": 0.0, "aux": 0.0, "kld": 0.0, "tot": 0.0, "n": 0}
    with torch.no_grad():
        for _, imu in val_loader:
            imu = imu.to(DEVICE)
            if LATENT_TYPE == "vae":
                recon, mu, logvar = model(imu)
                kld = kld_term(mu, logvar)
                loss = criterion(recon, imu) + kld_w * kld
                vagg["kld"] += kld.item() * imu.size(0)
            else:
                recon, _ = model(imu)
                loss = criterion(recon, imu)
            t_mse, a_mse = metrics.split_target_aux_mse(recon, imu, target_idx)
            vagg["tgt"] += t_mse * imu.size(0)
            vagg["aux"] += a_mse * imu.size(0)
            vagg["tot"] += loss.item() * imu.size(0)
            vagg["n"] += imu.size(0)

    hist["tr_tgt"].append(agg["tgt"] / agg["n"]);  hist["va_tgt"].append(vagg["tgt"] / vagg["n"])
    hist["tr_aux"].append(agg["aux"] / agg["n"]);  hist["va_aux"].append(vagg["aux"] / vagg["n"])
    hist["tr_kld"].append(agg["kld"] / agg["n"]);  hist["va_kld"].append(vagg["kld"] / vagg["n"])

    val_tot = vagg["tot"] / vagg["n"]
    pbar.set_postfix({
        "val_tgt": f"{hist['va_tgt'][-1]:.6f}",
        "val_aux": f"{hist['va_aux'][-1]:.6f}",
        "val_tot": f"{val_tot:.6f}",
        "kld_w": f"{kld_w:.5f}"
    })
    
    # 3. For occasional alerts (like saving a model), use pbar.write
    if val_tot < best_val:
        best_val = val_tot
        torch.save(model.state_dict(), best_path)
        pbar.write(f"Epoch {epoch+1:03d}: --> saved new best IMU autoencoder")

# ==========================================
# 5. FINAL PHYSICAL-UNIT METRICS (best checkpoint, on validation)
# ==========================================
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
recon_all, true_all = metrics.collect_ae_reconstruction(model, val_loader, LATENT_TYPE, DEVICE)
report = metrics.reconstruction_report(recon_all, true_all, val_dataset, TARGET_IMU_COLS)
print("\nValidation reconstruction (physical units):")
print(f"  target mean R^2 = {report['summary']['target_mean_r2']:.4f} | "
      f"target mean NRMSE = {report['summary']['target_mean_nrmse']:.4f}")

ru.save_metrics_json(RUN_DIR, {
    "phase": 1, "run_dir": RUN_DIR, "model_type": MODEL_TYPE, "latent_type": LATENT_TYPE,
    "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, "epochs": EPOCHS,
    "train_trials": len(train_dataset), "val_trials": len(val_dataset),
    "target_cols": TARGET_IMU_COLS, "best_val_loss": best_val,
    "summary": report["summary"],
    "per_feature": {c: {k: v for k, v in m.items()} for c, m in report["per_feature"].items()},
})

# ==========================================
# 6. PLOTS (target vs auxiliary; KLD only if VAE)
# ==========================================
epochs = range(1, EPOCHS + 1)
ncols = 2 if LATENT_TYPE == "vae" else 1
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
axes = axes if LATENT_TYPE == "vae" else [axes]
axes[0].plot(epochs, hist["tr_tgt"], color="#2563eb", lw=2, label="Train target MSE")
axes[0].plot(epochs, hist["va_tgt"], color="#1d4ed8", lw=2, ls="--", label="Val target MSE")
axes[0].plot(epochs, hist["tr_aux"], color="#f59e0b", lw=1.5, label="Train aux MSE")
axes[0].plot(epochs, hist["va_aux"], color="#b45309", lw=1.5, ls="--", label="Val aux MSE")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (scaled)")
axes[0].set_title("Target vs auxiliary reconstruction loss")
axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.5)
if LATENT_TYPE == "vae":
    axes[1].plot(epochs, hist["tr_kld"], color="#9333ea", lw=2, label="Train KLD")
    axes[1].plot(epochs, hist["va_kld"], color="#7e22ce", lw=2, ls="--", label="Val KLD")
    tw = axes[1].twinx(); tw.plot(epochs, hist["kld_w"], color="#06b6d4", lw=1.2, ls=":", label="KLD weight")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("KLD"); tw.set_ylabel("anneal weight")
    axes[1].set_title("KL divergence & annealing"); axes[1].grid(True, ls="--", alpha=0.5)
    l1, lb1 = axes[1].get_legend_handles_labels(); l2, lb2 = tw.get_legend_handles_labels()
    axes[1].legend(l1 + l2, lb1 + lb2, loc="upper right")
plt.tight_layout()
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
plot_b64 = base64.b64encode(buf.getvalue()).decode(); plt.close()

# ==========================================
# 7. DASHBOARD
# ==========================================
info = ru.probe_imu_autoencoder(model, sample_imu, LATENT_TYPE, DEVICE)
arch_rows = [
    ("Backbone", MODEL_TYPE.upper()), ("Latent type", LATENT_TYPE.upper()),
    ("Latent dim", str(LATENT_DIM)), ("Hidden dim", str(info.get("hidden_dim", "N/A"))),
    ("Compute device", DEVICE.type.upper()),
]
if "encoder_base_len" in info:
    arch_rows.append(("Conv encoder base length", str(info["encoder_base_len"])))
arch_rows += [
    ("IMU input", info["input_shape"]), ("Encoder output", info["encoder_output"]),
    ("Latent z", info["latent_shape"]), ("Reconstruction", info["recon_shape"]),
]
param_rows = [
    ("Encoder", ru.format_param_count(info["encoder_params"]), ru.format_param_count(info["encoder_params"])),
    ("Decoder", ru.format_param_count(info["decoder_params"]), ru.format_param_count(info["decoder_params"])),
    ("Full AE", ru.format_param_count(info["total_params"]), ru.format_param_count(info["total_params"])),
]
html = ru.build_imu_report({
    "run_name": os.path.basename(RUN_DIR), "plot_b64": plot_b64, "report": report,
    "best_val_loss": best_val, "train_val_str": f"{len(train_dataset)} / {len(val_dataset)}",
    "window_note": "Window 3.0 s | IMU 60 Hz (180 samples), EEG 256 Hz (768 samples)",
    "target_cols": TARGET_IMU_COLS,
    "kinematic_cols": train_dataset.imu_kinematic_cols, "angle_cols": train_dataset.imu_angle_cols,
    "config": [
        ("Model / latent", f"{MODEL_TYPE.upper()} / {LATENT_TYPE.upper()}"),
        ("Latent dim", str(LATENT_DIM)), ("Hidden dim", str(HIDDEN_DIM)),
        ("Learning rate", str(LR)), ("Batch size", str(BATCH_SIZE)), ("Epochs", str(EPOCHS)),
        ("Loss weights (tgt/aux)", "0.9 / 0.1"),
        ("KLD max / anneal", f"{KLD_MAX_WEIGHT} / {KLD_START_EPOCH}-{KLD_END_EPOCH}"),
    ],
    "data": [
        ("Train trials", str(len(train_dataset))), ("Val trials", str(len(val_dataset))),
        ("Subset size", str(TRAIN_SUBSET_SIZE)), ("IMU channels", str(IN_CHANNELS)),
        ("IMU seq length", str(IMU_LEN)), ("EEG seq length", str(EEG_LEN)),
        ("# target features", str(len(TARGET_IMU_COLS))),
    ],
    "arch_rows": arch_rows, "param_rows": param_rows,
})
report_path = os.path.join(RUN_DIR, "experiment_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n--> report: {report_path}\n--> weights: {best_path}\n--> metrics.json saved in {RUN_DIR}")