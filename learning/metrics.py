"""
Reconstruction metrics for the IMU autoencoder (Phase 1) and the end-to-end
EEG -> latent -> IMU pipeline (Phase 2).

Design goal
-----------
We need numbers that are *comparable across different TARGET_IMU_COLS choices*.
Raw MSE is not comparable: a velocity channel (dps) and an acceleration channel
(dps^2) live on different scales, and even two velocity channels can have very
different dynamic ranges. So the headline metric is R^2 (fraction of variance
explained), which is dimensionless, bounded above by 1, and equals 0 for a model
that just predicts the per-channel mean. NRMSE (RMSE / std) is reported as a
secondary scale-free number, and RMSE/MAE are reported in physical units for
interpretability.

R^2 invariance note
-------------------
RobustScaler is a per-channel affine map x' = (x - median) / IQR. R^2 is invariant
to applying the *same* affine map to both prediction and target, so the R^2 we
compute in scaled space is identical to the R^2 in physical (dps) space. That lets
us track target R^2 cheaply every epoch without inverse-transforming, while still
reporting physical-unit RMSE/MAE in the final table.

Sampling rates: IMU 60 Hz, EEG 256 Hz. Both windows (180 / 768 samples) cover the
same 3.0 s, so velocities are already per-second quantities and RMSE in dps is
directly meaningful. dt = 1/60 s is exposed for optional displacement integration.
"""

import math
import numpy as np
import torch

IMU_DT_SECONDS = 1.0 / 60.0
EEG_DT_SECONDS = 1.0 / 256.0


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def unit_for_column(col: str) -> str:
    if col.endswith("_vel_dps"):
        return "dps"
    if col.endswith("_acc_dps2"):
        return "dps^2"
    if col.endswith("_deg"):
        return "deg"
    return "scaled"


# ----------------------------------------------------------------------------
# inverse transform: scaled model space -> physical units
# ----------------------------------------------------------------------------
def inverse_imu_to_physical(tensor, scaler, n_kin: int, n_ang: int):
    """
    tensor: (..., C) in scaled model space, layout [kinematics | sin | cos].
    Returns (kin_phys (..., n_kin) in original units, angle_deg (..., n_ang) in degrees).
    Either element may be None if that group is empty.
    """
    arr = _np(tensor).astype(np.float64)
    lead, c = arr.shape[:-1], arr.shape[-1]
    flat = arr.reshape(-1, c)

    kin_phys = None
    if n_kin > 0:
        kin_phys = scaler.imu_scaler.inverse_transform(flat[:, :n_kin])
        kin_phys = kin_phys.reshape(*lead, n_kin)

    angle_deg = None
    if n_ang > 0:
        sin = flat[:, n_kin:n_kin + n_ang]
        cos = flat[:, n_kin + n_ang:n_kin + 2 * n_ang]
        angle_deg = np.degrees(np.arctan2(sin, cos)).reshape(*lead, n_ang)

    return kin_phys, angle_deg


# ----------------------------------------------------------------------------
# per-feature scoring
# ----------------------------------------------------------------------------
def _linear_metrics(pred, true) -> dict:
    pred = pred.ravel().astype(np.float64)
    true = true.ravel().astype(np.float64)
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    std = float(true.std())
    nrmse = float(rmse / std) if std > 0 else float("nan")
    if pred.std() > 0 and true.std() > 0:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse, "corr": corr}


def _circular_metrics(pred_deg, true_deg) -> dict:
    """Angle reconstruction, wrapped to [-180, 180]. cos_sim in [-1, 1], 1 = perfect."""
    diff = np.radians(pred_deg.ravel() - true_deg.ravel())
    wrapped = np.degrees(np.arctan2(np.sin(diff), np.cos(diff)))
    return {
        "rmse": float(np.sqrt(np.mean(wrapped ** 2))),
        "mae": float(np.mean(np.abs(wrapped))),
        "cos_sim": float(np.mean(np.cos(np.radians(wrapped)))),
    }


def _skill(m: dict) -> float:
    """Single comparable 'goodness' score per feature: R^2 for linear, cos_sim for angles."""
    return m["r2"] if "r2" in m else m["cos_sim"]


def reconstruction_report(recon, true, dataset, target_cols) -> dict:
    """
    Full per-feature reconstruction report in physical units, split into target vs
    auxiliary features. `recon` and `true` are (N, T, C) tensors in scaled model space.

    Returns:
      {
        "per_feature": {col: {kind, unit, is_target, rmse, mae, r2/cos_sim, ...}},
        "summary": {target_mean_r2, target_mean_nrmse, target_mean_skill,
                    aux_mean_skill, n_target_kin, n_target_ang, ...}
      }
    """
    scaler = dataset.scaler
    kin_cols = list(dataset.imu_kinematic_cols)
    ang_cols = list(dataset.imu_angle_cols)
    n_kin, n_ang = len(kin_cols), len(ang_cols)
    target_set = set(target_cols)

    rk, ra = inverse_imu_to_physical(recon, scaler, n_kin, n_ang)
    tk, ta = inverse_imu_to_physical(true, scaler, n_kin, n_ang)

    per_feature: dict[str, dict] = {}
    for i, col in enumerate(kin_cols):
        m = _linear_metrics(rk[..., i], tk[..., i])
        m.update(kind="kinematic", unit=unit_for_column(col), is_target=col in target_set)
        per_feature[col] = m
    for i, col in enumerate(ang_cols):
        m = _circular_metrics(ra[..., i], ta[..., i])
        m.update(kind="angle", unit="deg", is_target=col in target_set)
        per_feature[col] = m

    tgt = [m for m in per_feature.values() if m["is_target"]]
    aux = [m for m in per_feature.values() if not m["is_target"]]
    tgt_kin = [m for m in tgt if m["kind"] == "kinematic"]

    def nanmean(vals):
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "target_mean_skill": nanmean([_skill(m) for m in tgt]),
        "aux_mean_skill": nanmean([_skill(m) for m in aux]),
        "target_mean_r2": nanmean([m["r2"] for m in tgt_kin]),
        "target_mean_nrmse": nanmean([m["nrmse"] for m in tgt_kin]),
        "target_mean_rmse": nanmean([m["rmse"] for m in tgt]),
        "n_target": len(tgt),
        "n_aux": len(aux),
        "n_target_kinematic": len(tgt_kin),
        "n_target_angle": len(tgt) - len(tgt_kin),
    }
    return {"per_feature": per_feature, "summary": summary}


# ----------------------------------------------------------------------------
# cheap per-epoch trackers (scaled space)
# ----------------------------------------------------------------------------
def split_target_aux_mse(recon, true, target_idx) -> tuple[float, float]:
    """Unweighted target vs auxiliary MSE in scaled space, for the loss curves."""
    tset = set(target_idx)
    aux_idx = [i for i in range(recon.shape[2]) if i not in tset]
    tgt = torch.nn.functional.mse_loss(recon[:, :, target_idx], true[:, :, target_idx]).item()
    if aux_idx:
        aux = torch.nn.functional.mse_loss(recon[:, :, aux_idx], true[:, :, aux_idx]).item()
    else:
        aux = 0.0
    return tgt, aux


def target_r2_scaled(recon, true, target_idx) -> float:
    """
    Mean R^2 over the target channels, computed in scaled space.
    Equals physical-unit R^2 because RobustScaler is a per-channel affine map.
    Cheap enough to call every epoch on the validation set.
    """
    r = _np(recon)[..., target_idx].reshape(-1, len(target_idx))
    t = _np(true)[..., target_idx].reshape(-1, len(target_idx))
    r2s = []
    for j in range(r.shape[1]):
        ss_res = np.sum((r[:, j] - t[:, j]) ** 2)
        ss_tot = np.sum((t[:, j] - t[:, j].mean()) ** 2)
        if ss_tot > 0:
            r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s)) if r2s else float("nan")


# ----------------------------------------------------------------------------
# reconstruction collectors (run a loader, return stacked recon/true tensors)
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_ae_reconstruction(model, loader, latent_type, device):
    """Phase 1: recon = autoencoder(imu)."""
    model.eval()
    recons, trues = [], []
    for _, imu in loader:
        imu = imu.to(device)
        out = model(imu)
        recon = out[0]  # (recon, mu, logvar) for vae, (recon, z) for ae
        recons.append(recon.cpu())
        trues.append(imu.cpu())
    return torch.cat(recons), torch.cat(trues)


@torch.no_grad()
def collect_eeg_e2e_reconstruction(eeg_model, imu_model, loader, latent_type, device):
    """Phase 2: recon = frozen_decoder(eeg_model(eeg)). Not used for training, only eval."""
    eeg_model.eval()
    imu_model.eval()
    recons, trues = [], []
    for eeg, imu in loader:
        eeg = eeg.to(device)
        z = eeg_model(eeg)
        recon = imu_model.decoder(z)
        recons.append(recon.cpu())
        trues.append(imu.cpu())
    return torch.cat(recons), torch.cat(trues)