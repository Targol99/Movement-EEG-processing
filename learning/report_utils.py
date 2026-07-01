
import os
import json
import math
import datetime

import torch
import torch.nn as nn
from models import IMUAutoencoder

# ============================================================================
# experiment folder + metrics persistence
# ============================================================================
def make_run_dir(base_dir: str, *tag_pieces) -> str:
    """
    Create ./<base_dir>/<tag1_tag2_..._DDMM_HHMM>/ and return its path.
    Each experiment gets its own folder so nothing overwrites a previous run.
    """
    ts = datetime.datetime.now().strftime("%d%m_%H%M")
    name = "_".join(str(p) for p in tag_pieces if p not in ("", None)) + f"_{ts}"
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_metrics_json(run_dir: str, payload: dict, fname: str = "metrics.json") -> str:
    """Persist a machine-readable metrics summary for cross-experiment comparison."""
    path = os.path.join(run_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ============================================================================
# parameter counting + shape probes (used in the Architecture section)
# ============================================================================
def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1e3:.1f}K)"
    return f"{n:,}"


def _fmt_shape(shape: tuple) -> str:
    return " x ".join(str(d) for d in shape)


def probe_imu_autoencoder(model: IMUAutoencoder, sample_input: torch.Tensor, latent_type: str, device: torch.device) -> dict:
    model.eval()
    x = sample_input.unsqueeze(0).to(device)
    with torch.no_grad():
        if latent_type == "vae":
            mu, logvar = model.encoder(x)
            z = mu
            encoder_out = f"mu, logvar: each {_fmt_shape(mu.shape)}"
        else:
            z = model.encoder(x)
            encoder_out = _fmt_shape(z.shape)
        recon = model.decoder(z)

    enc_total, enc_train = count_parameters(model.encoder)
    dec_total, dec_train = count_parameters(model.decoder)
    full_total, full_train = count_parameters(model)
    info = {
        "input_shape": _fmt_shape(x.shape), "encoder_output": encoder_out,
        "latent_shape": _fmt_shape(z.shape), "recon_shape": _fmt_shape(recon.shape),
        "encoder_params": enc_total, "decoder_params": dec_total, "total_params": full_total,
    }
    if hasattr(model.encoder, "base_len"):
        info["encoder_base_len"] = model.encoder.base_len
    if hasattr(model.encoder, "feature_extractor"):
        info["hidden_dim"] = model.encoder.feature_extractor[0].out_channels
    elif hasattr(model.encoder, "lstm"):
        info["hidden_dim"] = model.encoder.lstm.hidden_size
        info["lstm_layers"] = model.encoder.lstm.num_layers
    return info


def probe_eeg_mapping(eeg_model, imu_model, sample_eeg, sample_imu, latent_type, device) -> dict:
    eeg_model.eval()
    imu_model.eval()
    eeg_x = sample_eeg.unsqueeze(0).to(device)
    imu_x = sample_imu.unsqueeze(0).to(device)
    with torch.no_grad():
        if latent_type == "vae":
            target_z, _ = imu_model.encoder(imu_x)
        else:
            target_z = imu_model.encoder(imu_x)
        predicted_z = eeg_model(eeg_x)

    eeg_total, eeg_train = count_parameters(eeg_model)
    imu_total, _ = count_parameters(imu_model)
    info = {
        "eeg_input_shape": _fmt_shape(eeg_x.shape), "imu_input_shape": _fmt_shape(imu_x.shape),
        "target_latent_shape": _fmt_shape(target_z.shape),
        "predicted_latent_shape": _fmt_shape(predicted_z.shape),
        "eeg_params": eeg_total, "eeg_params_trainable": eeg_train, "imu_total_params": imu_total,
    }
    if hasattr(eeg_model, "feature_extractor"):
        info["eeg_hidden_dim"] = eeg_model.feature_extractor[0].out_channels
    elif hasattr(eeg_model, "lstm"):
        info["eeg_hidden_dim"] = eeg_model.lstm.hidden_size
    return info


# ============================================================================
# shared HTML pieces
# ============================================================================
_CSS = """
  body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:32px; }
  .container { max-width:1180px; margin:0 auto; }
  h1 { color:#fff; font-size:24px; margin:0 0 4px; }
  h2 { color:#93c5fd; font-size:17px; margin:34px 0 12px; border-bottom:1px solid #1e293b; padding-bottom:6px; }
  .sub { color:#94a3b8; font-size:13px; margin:0 0 4px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:18px 0; }
  .kpi { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; }
  .kpi .lbl { font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:.04em; }
  .kpi .val { font-size:26px; font-weight:700; color:#fff; margin-top:6px; }
  .kpi.hero { background:linear-gradient(135deg,#1d4ed8,#4338ca); border:none; }
  .kpi.hero .lbl, .kpi.hero .val { color:#fff; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px; }
  .card h3 { margin:0 0 10px; font-size:14px; color:#cbd5e1; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { padding:8px 10px; text-align:left; border-bottom:1px solid #283549; }
  th { color:#94a3b8; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
  td code, .mono { font-family:ui-monospace,monospace; color:#a5b4fc; }
  tr.tgt { background:rgba(59,130,246,.14); }
  tr.tgt td:first-child { border-left:3px solid #3b82f6; font-weight:600; color:#fff; }
  .pill { display:inline-block; background:#0b3b6f; color:#bfdbfe; border-radius:999px; padding:2px 9px; font-size:11px; margin:2px 3px 0 0; }
  .pill.tgt { background:#1d4ed8; color:#fff; }
  .chart { background:#fff; border-radius:12px; padding:14px; margin:14px 0; text-align:center; }
  .chart img { max-width:100%; height:auto; }
  .good { color:#4ade80; } .mid { color:#fbbf24; } .bad { color:#f87171; }
  .foot { color:#64748b; font-size:12px; margin-top:30px; }
"""


def _r2_class(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if v >= 0.7:
        return "good"
    if v >= 0.3:
        return "mid"
    return "bad"


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "&mdash;"
    return f"{v:.{nd}f}"


def _kpi(label, value, hero=False):
    cls = "kpi hero" if hero else "kpi"
    return f'<div class="{cls}"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _pills(cols: list[str], target_set: set) -> str:
    return "".join(
        f'<span class="pill {"tgt" if c in target_set else ""}">{c}</span>' for c in cols
    ) or "<span class='sub'>none</span>"


def _kinematic_table(report: dict) -> str:
    pf = report["per_feature"]
    rows = ""
    for col, m in pf.items():
        if m["kind"] != "kinematic":
            continue
        cls = "tgt" if m["is_target"] else ""
        rows += (
            f"<tr class='{cls}'><td>{col}</td><td class='mono'>{m['unit']}</td>"
            f"<td>{_f(m['rmse'])}</td><td>{_f(m['mae'])}</td>"
            f"<td class='{_r2_class(m['r2'])}'>{_f(m['r2'])}</td>"
            f"<td>{_f(m['nrmse'])}</td><td>{_f(m['corr'])}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<div class='card' style='margin-bottom:16px;'><h3>Per-feature reconstruction "
        "&mdash; kinematics (physical units)</h3><table>"
        "<tr><th>Feature</th><th>Unit</th><th>RMSE</th><th>MAE</th><th>R&sup2;</th>"
        "<th>NRMSE</th><th>Corr</th></tr>" + rows + "</table>"
        "<p class='sub' style='margin-top:8px'>Highlighted rows are target features. "
        "R&sup2; and NRMSE are scale-free and comparable across target-column choices.</p></div>"
    )


def _angle_table(report: dict) -> str:
    pf = report["per_feature"]
    rows = ""
    for col, m in pf.items():
        if m["kind"] != "angle":
            continue
        cls = "tgt" if m["is_target"] else ""
        rows += (
            f"<tr class='{cls}'><td>{col}</td>"
            f"<td>{_f(m['rmse'], 2)}&deg;</td><td>{_f(m['mae'], 2)}&deg;</td>"
            f"<td class='{_r2_class(m['cos_sim'])}'>{_f(m['cos_sim'])}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<div class='card' style='margin-bottom:16px;'><h3>Per-feature reconstruction "
        "&mdash; angles (circular, degrees)</h3><table>"
        "<tr><th>Feature</th><th>RMSE</th><th>MAE</th><th>Cos-sim</th></tr>"
        + rows + "</table><p class='sub' style='margin-top:8px'>Angles are scored on the "
        "wrapped error; cos-sim is 1.0 for a perfect match.</p></div>"
    )


def _arch_card(arch_rows: list[tuple[str, str]], param_rows: list[tuple[str, str, str]]) -> str:
    prm = "".join(
        f"<tr><td>{a}</td><td class='mono'>{b}</td><td class='mono'>{c}</td></tr>"
        for a, b, c in param_rows
    )
    return (
        "<div class='grid2'>"
        f"<div class='card'><h3>Architecture</h3>{_kv_table(arch_rows)}</div>"
        "<div class='card'><h3>Parameter counts</h3><table>"
        "<tr><th>Component</th><th>Total</th><th>Trainable</th></tr>"
        f"{prm}</table></div></div>"
    )


# ============================================================================
# Phase 1 dashboard
# ============================================================================
def build_imu_report(ctx: dict) -> str:
    """ctx keys: run_name, plot_b64, report(metrics), config(list of kv), data(list of kv),
    arch_rows, param_rows, target_cols, kinematic_cols, angle_cols, window_note."""
    s = ctx["report"]["summary"]
    tset = set(ctx["target_cols"])
    hero = _kpi("Target mean R&sup2;", _f(s["target_mean_r2"], 3), hero=True)
    cards = hero + "".join([
        _kpi("Target mean NRMSE", _f(s["target_mean_nrmse"], 3)),
        _kpi("Aux mean skill", _f(s["aux_mean_skill"], 3)),
        _kpi("Best val loss", _f(ctx["best_val_loss"], 5)),
        _kpi("Train / Val trials", ctx["train_val_str"]),
    ])
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Phase 1 &mdash; IMU Autoencoder</title><style>{_CSS}</style></head><body><div class="container">
<h1>Phase 1 &mdash; IMU Autoencoder</h1>
<p class="sub">Run <span class="mono">{ctx['run_name']}</span> &middot; generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}</p>
<p class="sub">{ctx['window_note']}</p>
<div class="cards">{cards}</div>

<h2>Target features being optimized</h2>
<div class="card">{_pills(ctx['target_cols'], tset)}
<p class="sub" style="margin-top:8px">These are weighted in the loss; everything else is auxiliary. The headline R&sup2; is the mean over these channels &mdash; use it to compare different target-column choices across runs.</p></div>

<h2>Training curves</h2>
<div class="chart"><img src="data:image/png;base64,{ctx['plot_b64']}" alt="loss curves"></div>

<h2>Reconstruction quality</h2>
{_kinematic_table(ctx['report'])}
{_angle_table(ctx['report'])}

<h2>Configuration</h2>
<div class="grid2">
<div class="card"><h3>Hyperparameters</h3>{_kv_table(ctx['config'])}</div>
<div class="card"><h3>Dataset</h3>{_kv_table(ctx['data'])}</div>
</div>

<h2>Architecture &amp; tensor shapes</h2>
{_arch_card(ctx['arch_rows'], ctx['param_rows'])}

<p class="foot">All kinematic metrics inverse-transformed to physical units (dps, dps&sup2;). R&sup2; is invariant to the per-channel affine scaler, so the per-epoch curve and this table agree.</p>
</div></body></html>"""


# ============================================================================
# Phase 2 dashboard
# ============================================================================
def build_eeg_report(ctx: dict) -> str:
    """ctx keys: run_name, phase1_dir, plot_b64, report(end-to-end metrics),
    config, data, arch_rows, param_rows, target_cols, window_note,
    best_val_latent_mse, train_val_str."""
    s = ctx["report"]["summary"]
    tset = set(ctx["target_cols"])
    hero = _kpi("End-to-end target R&sup2;", _f(s["target_mean_r2"], 3), hero=True)
    cards = hero + "".join([
        _kpi("Target mean NRMSE", _f(s["target_mean_nrmse"], 3)),
        _kpi("Target mean RMSE", _f(s["target_mean_rmse"], 2) + " dps"),
        _kpi("Best val latent MSE", _f(ctx["best_val_latent_mse"], 5)),
        _kpi("Train / Val trials", ctx["train_val_str"]),
    ])
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Phase 2 &mdash; EEG to IMU mapping</title><style>{_CSS}</style></head><body><div class="container">
<h1>Phase 2 &mdash; EEG &rarr; IMU latent mapping</h1>
<p class="sub">Run <span class="mono">{ctx['run_name']}</span> &middot; generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}</p>
<p class="sub">Phase 1 source: <span class="mono">{ctx['phase1_dir']}</span></p>
<p class="sub">{ctx['window_note']}</p>
<div class="cards">{cards}</div>

<h2>Target features being decoded</h2>
<div class="card">{_pills(ctx['target_cols'], tset)}
<p class="sub" style="margin-top:8px">Training optimizes latent MSE only. The headline R&sup2; measures the real objective &mdash; decoder(eeg_model(eeg)) vs. true IMU in physical units &mdash; and is comparable across target-column choices to find which features map best from EEG.</p></div>

<h2>Curves</h2>
<div class="chart"><img src="data:image/png;base64,{ctx['plot_b64']}" alt="mapping curves"></div>

<h2>End-to-end reconstruction quality (EEG &rarr; IMU)</h2>
{_kinematic_table(ctx['report'])}
{_angle_table(ctx['report'])}

<h2>Configuration</h2>
<div class="grid2">
<div class="card"><h3>Hyperparameters</h3>{_kv_table(ctx['config'])}</div>
<div class="card"><h3>Dataset</h3>{_kv_table(ctx['data'])}</div>
</div>

<h2>Architecture &amp; tensor shapes</h2>
{_arch_card(ctx['arch_rows'], ctx['param_rows'])}

<p class="foot">Latent MSE is the training signal; the physical-unit reconstruction R&sup2; is tracked but never backpropagated. IMU encoder/decoder weights are frozen from Phase 1.</p>
</div></body></html>"""
