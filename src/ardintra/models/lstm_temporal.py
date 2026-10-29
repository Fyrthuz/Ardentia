"""LSTM temporal sobre ventanas de 10 días de met interpolada (vía Kondylatos 2022).

Cada fila (celda, día) del dataset → secuencia (10 días × n_vars dinámicas) con
los met/FWI/sequía IDW de las 3 estaciones más cercanas. LSTM(64) → MLP → logit.
Entrena en GPU; early-stop por PR-AUC en val 2022. Compara contra el LGBM flat.

Uso: uv run python -m ardintra.models.lstm_temporal
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

from ardintra.features.build_dataset import load_met_stations, MET_VARS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 10
STATIC_FEATS = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos",
                "elev_m", "dist_to_road", "road_density_1km", "road_fuel_interface"]


def build_sequences(df: pd.DataFrame, met: pd.DataFrame, cells: pd.DataFrame):
    """Devuelve (X_seq [N, WIN, n_vars], X_static [N, n_static], y, masks)."""
    met = met.dropna(subset=["lat", "lon"])
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    d, idx = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=3)
    w = 1.0 / np.maximum(d, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)

    met = met.copy()
    met["fecha"] = pd.to_datetime(met["fecha"]).dt.normalize()
    piv = met.pivot_table(index="fecha", columns="idema", values=MET_VARS, aggfunc="first")
    dates = pd.DatetimeIndex(piv.index)
    date_pos = {d_: i for i, d_ in enumerate(dates)}
    st_pos = {s: j for j, s in enumerate(piv.columns.levels[1])}

    n_vars = len(MET_VARS)
    arrays = {v: np.full((len(dates), len(st_meta)), np.nan) for v in MET_VARS}
    for var in MET_VARS:
        blk = piv[var]
        for j, sid in enumerate(st_meta["idema"]):
            if sid in blk.columns:
                arrays[var][:, j] = blk[sid].to_numpy()

    rows_ll = df[["h9"]].merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    dd, ii3 = tree.query(np.radians(rows_ll[["lat", "lon"]].fillna(42.8).to_numpy()), k=3)
    wv = 1.0 / np.maximum(dd, 1e-5) ** 2
    wv = wv / wv.sum(axis=1, keepdims=True)

    dpos = np.array([date_pos.get(d_, -1) for d_ in df["fecha"].dt.normalize()])
    seq = np.zeros((len(df), WIN, n_vars), dtype="float32")
    # pesos efectivos por fila/día (normaliza la media IDW cuando faltan estaciones)
    wsum = np.zeros((len(df), WIN), dtype="float32")
    for k in range(3):
        st_j = ii3[:, k]
        w_k = wv[:, k]
        for off in range(WIN):
            dp = dpos - (WIN - 1) + off
            has = (dp >= 0) & (dp < len(dates)) & (dpos >= 0)
            v = arrays  # acceso por variable dentro del bucle principal
            for vi, var in enumerate(MET_VARS):
                vv = arrays[var][np.clip(dp, 0, None)[has], st_j[has]]
                m = ~np.isnan(vv)
                seq[has, off, vi] += np.nan_to_num(vv) * w_k[has]
            wsum[has, off] += w_k[has]
    wsum = np.maximum(wsum, 1e-9)
    seq = seq / wsum[:, :, None]  # media IDW por día
    seq[dpos < 0] = 0
    return seq


def main(target: str = "big10", epochs: int = 40):
    import ardintra.features.build_dataset as BD

    df = pd.read_parquet("data/processed/features/v1/dataset.parquet")
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    if target == "big10":
        y = np.where((df["label"] == 1) & (surf >= 10), 1, 0)
    else:
        y = df["label"].to_numpy()

    cells = pd.read_parquet("data/processed/h3_cells/res9_static.parquet", columns=["h9", "lat", "lon"])
    met = BD.load_met_stations()
    print("construyendo secuencias…", flush=True)
    X_seq = build_sequences(df, met, cells)
    print("seq:", X_seq.shape)

    # normaliza por variable (train stats) y rellena NaN→0
    tr_all = (df["fecha"].dt.year <= 2022).to_numpy()
    for v in range(X_seq.shape[2]):
        flat = X_seq[:, :, v].reshape(-1)
        mu, sd = np.nanmean(flat[tr_all.repeat(WIN)]), np.nanstd(flat[tr_all.repeat(WIN)]) + 1e-6
        flat = (flat - mu) / sd
        X_seq[:, :, v] = flat.reshape(len(df), WIN)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0, neginf=0)

    tr = ((df["fecha"].dt.year <= 2021)).to_numpy() & df["fwi"].notna().to_numpy()
    val = ((df["fecha"].dt.year == 2022)).to_numpy() & df["fwi"].notna().to_numpy()
    te = ((df["fecha"].dt.year >= 2023)).to_numpy() & df["fwi"].notna().to_numpy()

    Xt = torch.tensor(X_seq, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    posw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    model = nn.Sequential(
        nn.LSTM(X_seq.shape[2], 64, num_layers=1, batch_first=True),
        TakeLast(),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw, device=DEVICE))
    Xg, yg = Xt.to(DEVICE), yt.to(DEVICE)
    best_ap, best_state, best_ep = -1, None, -1
    tr_idx = np.where(tr)[0]
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + ep)
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), 4096):
            sl = torch.tensor(tr_idx[i:i + 4096], device=DEVICE)
            opt.zero_grad()
            logits = model(Xg[sl])
            loss = lossf(logits.squeeze(-1), yg[sl])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xt[val].to(DEVICE)).squeeze(-1)).cpu().numpy()
        ap = average_precision_score(y[val], pv) if val.sum() else -1
        if ep % 5 == 4:
            print(f"  ep{ep}: val AP {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, best_ep, best_state = ap, ep, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        p_te = torch.sigmoid(model(Xt[te].to(DEVICE)).squeeze(-1)).cpu().numpy()
    yte = y[te]
    ap = average_precision_score(yte, p_te)
    o = np.argsort(-p_te); k5 = max(int(len(p_te) * .05), 1)
    print(f"RESULTADO LSTM-TEMPORAL [{target}] test PR-AUC={ap:.3f} ({ap/max(yte.mean()+1e-9,1e-9):5.1f}× azar) · "
          f"recall@5%={yte[o[:k5]].sum()/max(yte.sum(),1):.0%} · ROC-AUC={roc_auc_score(yte, p_te):.3f} · mejor ep {best_ep}")


class TakeLast(nn.Module):
    # expone el último estado del LSTM al head
    def forward(self, tup):
        out, _ = tup
        return out[:, -1, :]


if __name__ == "__main__":
    main("big10" if "big" in (sys.argv[1] if len(sys.argv) > 1 else "big10") else "ign")
