"""CNN-2D sobre parches espaciales 5×5 de features (Ronda 3 — Kondylatos/Zhang/LOAN).

Para cada (celda, día) del test: extrae un parche 5×5 de las celdas vecinas
(estáticas + dinámicas del día) y entrena una CNN-2D que captura la estructura
espacial local (corredores de combustible, bordes WUI, gradientes de pendiente).

Los features por píxel del parche:
  - estáticas: forest_frac, fuel_model, slope, aspect_sin/cos, elev, dist_road, road_dens
  - dinámicas del día: fwi, tmax, rh, wind, prec, swc, vpd, ndmi, spei

Entrena en GPU (RTX 5080) con early stopping por PR-AUC en val 2022.
Compara contra el LGBM flat (nuestro mejor tabular).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 5  # 5×5 celdas alrededor del target
STATIC_COLS = ["forest_frac", "fuel_model", "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
               "dist_to_road_log", "road_density_log", "road_fuel_interface"]
DYN_COLS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
            "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30",
            "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
            "es_finde", "dens_x_weekend"]


def build_patches(df: pd.DataFrame, cells: pd.DataFrame, k: int = 12) -> np.ndarray:
    """Para cada fila del dataset: parche (PATCH × PATCH × n_features) de las k
    celdas más cercanas espacialmente (del MISMO dataset, que ya trae dinámicas).

    k = número de vecinos a incluir (rellenamos el parche 5×5=25 con los k más
    cercanos + padding). k=12 captura el anillo-1 + la propia celda.
    """
    # celdas únicas con coordenadas
    cell_ll = df[["h9", "lat", "lon"]].drop_duplicates("h9").reset_index(drop=True)
    tree = cKDTree(np.radians(cell_ll[["lat", "lon"]].to_numpy()))

    # features completas por celda única (estáticas + dinámicas medias del período)
    # para el parche espacial usamos las features del dataset (que ya las trae)
    feat_cols = [c for c in STATIC_COLS + DYN_COLS if c in df.columns]
    cell_features = df.groupby("h9")[feat_cols].mean().reset_index()
    cell_map = {h: i for i, h in enumerate(cell_features["h9"])}

    # para cada fila del dataset: buscar las k celdas más cercanas y apilar sus features
    ll_rows = df[["h9"]].merge(cell_ll, on="h9", how="left")
    pts = np.radians(ll_rows[["lat", "lon"]].fillna(42.8).to_numpy())
    _, nn_idx = tree.query(pts, k=k + 1)  # +1 incluye la propia

    n_feats = len(feat_cols)
    feat_matrix = cell_features[feat_cols].to_numpy(dtype="float32")

    # parche: (n_rows, k, n_feats) → reshape a (n_rows, PATCH, PATCH, n_feats) con padding
    patches = np.zeros((len(df), PATCH * PATCH, n_feats), dtype="float32")
    for i in range(len(df)):
        neighbor_idx = [cell_map.get(cell_ll.loc[j, "h9"], -1) for j in nn_idx[i, 1:k+1]]
        for j, ci in enumerate(neighbor_idx[:PATCH * PATCH]):
            if ci >= 0:
                patches[i, j] = feat_matrix[ci]
    # normaliza por feature (global) y elimina NaN
    for f in range(n_feats):
        col = patches[:, :, f]
        mu, sd = np.nanmean(col), np.nanstd(col) + 1e-6
        col = np.clip((col - mu) / sd, -5, 5)
        patches[:, :, f] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0).reshape(len(df), PATCH, PATCH, n_feats)


class CNN2D(nn.Module):
    def __init__(self, n_feats: int, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_feats, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 3, padding=1), nn.BatchNorm2d(channels * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_cnn(patches: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray,
              epochs: int = 30, batch: int = 1024, lr: float = 5e-4):
    Xg = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2).to(DEVICE)  # N,C,H,W
    yg = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    posw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    model = CNN2D(patches.shape[3]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw, device=DEVICE))
    tr_idx = np.where(tr)[0]
    best_ap, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + ep)
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), batch):
            sl = torch.tensor(tr_idx[i:i+batch], device=DEVICE)
            opt.zero_grad()
            logits = model(Xg[sl])
            loss = lossf(logits, yg[sl])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xg[val])).cpu().numpy()
        ap = average_precision_score(y[val], pv) if val.sum() else -1
        if ep % 5 == 4:
            print(f"  ep{ep}: val AP {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, best_ep = ap, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_all = torch.sigmoid(model(Xg)).cpu().numpy()
    print(f"mejor época {best_ep} · val AP {best_ap:.4f}")
    return p_all, model


def main(target: str = "big10"):
    from ardintra.features.build_dataset import load_met_stations

    df = pd.read_parquet("data/processed/features/v1/dataset.parquet")
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    ok = df["fwi"].notna().to_numpy()
    if target == "big10":
        y = np.where((df["label"] == 1) & (surf >= 10), 1, 0)[ok]
    else:
        y = df["label"].to_numpy()[ok]
    df = df[ok].reset_index(drop=True).copy()
    cells = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")

    print("construyendo parches espaciales…", flush=True)
    patches = build_patches(df, cells)
    print(f"parches: {patches.shape}")

    tr = (df["fecha"].dt.year <= 2021).to_numpy()
    val = (df["fecha"].dt.year == 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()

    print("entrenando CNN-2D…", flush=True)
    p_all, model = train_cnn(patches, y, tr, val, epochs=25)
    yte = y[te]
    p_te = p_all[te]
    ap = average_precision_score(yte, p_te)
    roc = roc_auc_score(yte, p_te)
    br = brier_score_loss(yte, p_te)
    o = np.argsort(-p_te); k5 = max(int(len(p_te) * .05), 1)
    r5 = yte[o[:k5]].sum() / max(yte.sum(), 1)
    print(f"RESULTADO CNN-2D [{target}] test PR-AUC={ap:.4f} ({ap/max(yte.mean()+1e-9,1e-9):.0f}× azar) · "
          f"ROC={roc:.4f} · Brier={br:.4f} · recall@5%={r5:.0%}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "big10")
