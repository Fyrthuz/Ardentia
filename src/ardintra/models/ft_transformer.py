"""FT-Transformer con ATENCIÓN CRUZADA static↔dynamic para WildfireRisk-ES.

Arquitectura (oporstile deducciones del paper "Revisiting DL for Tabular Data"):
  - Tokenizador por feature: cada variable → token d_model
      · numéricas → Linear(1→d) + bias learned + LayerNorm
      · categórica (road_class) → Embedding(idx→d)
  - Dos grupos con tokens propia:
      · STATIC GROUP: terrain/combustible/viaria (estáticos)
      · DYNAMIC GROUP: met/era5/satélite/recencia/calendario (variación diaria)
  -Encoders: cada grupo → 1 self-attention block
  - CRUZADA: dynamic tokens ↔ static tokens (bidirectional attention block) para
    codificar interacciones del tipo "arcén×sequedad" sin feature engineering manual
  - cabeza: [CLS-dynamic] → MLP → logit

Entrenamiento: GPU (RTX 5080) · batches 4096 · AdamW + cosine LR · PR-AUC en val.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NUM_STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
              "dist_to_road", "road_density_1km", "road_fuel_interface"]
NUM_DYNAMIC = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
               "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
               "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
               "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
               "es_finde", "dens_x_weekend", "fwi_7d", "fwi_prev"]
CAT_COL = "road_class"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureTokenizer(nn.Module):
    def __init__(self, num_cols: list[str], cat_card: int, d_model: int):
        super().__init__()
        self.num_cols = num_cols
        self.weight = nn.Parameter(torch.empty(len(num_cols), d_model))
        self.bias = nn.Parameter(torch.empty(len(num_cols), d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.bias, 0, 0.02)
        self.cat_emb = nn.Embedding(cat_card + 1, d_model, padding_idx=cat_card)

    def forward(self, Xn, Xc=None):
        w, b = self.weight, self.bias
        tok = Xn.unsqueeze(2) * w.unsqueeze(0) + b.unsqueeze(0)   # [B, n, d]
        if Xc is not None:
            tok = torch.cat([tok, self.cat_emb(Xc.squeeze(1)).unsqueeze(1)], dim=1)
        return tok  # [B, n(+1 cat), d]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff, d_model), nn.Dropout(dropout))

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        attn_out, _ = self.attn(self.norm1(x), self.norm1(kv), self.norm1(kv))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class FTAttention(nn.Module):
    def __init__(self, num_static: int, num_dynamic: int, cat_card: int,
                 d_model: int = 64, heads: int = 8, layers_cross: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tok_static = FeatureTokenizer(NUM_STATIC, cat_card, d_model)
        self.tok_dyn = FeatureTokenizer(NUM_DYNAMIC, cat_card, d_model)
        self.cls_static = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_dyn = nn.Parameter(torch.randn(1, 1, d_model))
        self.enc_static = TransformerBlock(d_model, heads, d_model * 4, dropout)
        self.enc_dyn = TransformerBlock(d_model, heads, d_model * 4, dropout)
        # atención cruzada bidireccional (2 bloquees)
        self.cross_dyn2stat = TransformerBlock(d_model, heads, d_model * 4, dropout)
        self.cross_stat2dyn = TransformerBlock(d_model, heads, d_model * 4, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, Xn_s, Xn_d, Xc_s):
        # 1) tokenizar
        ts = self.tok_static(Xn_s, None)
        td = self.tok_dyn(Xn_d, Xc_s)
        # 2) [CLS] prepended en cada grupo
        ts = torch.cat([self.cls_static.expand(Xn_s.shape[0], -1, -1), ts], dim=1)
        td = torch.cat([self.cls_dyn.expand(Xn_d.shape[0], -1, -1), td], dim=1)
        # 3) self-attention por grupo
        ts = self.enc_static(ts)
        td = self.enc_dyn(td)
        # 4) CROSS: dynamic Query sobre static KV · y viceversa
        td = self.cross_dyn2stat(td, kv=ts)
        ts = self.cross_stat2dyn(ts, kv=td)
        # 5) cabeza desde el [CLS] del grupo dinámico (por predicción diaria)
        return self.head(td[:, 0]).squeeze(-1)


def prepare(df: pd.DataFrame):
    Xs = df[NUM_STATIC].copy().astype("float32")
    Xd = df[[c for c in NUM_DYNAMIC if c in df.columns]].copy().astype("float32")
    # normalización robusta: medianas/cuartiles (Training data only preferible, MVP: global)
    for c in list(Xs.columns) + list(Xd.columns):
        med = Xs[c].median() if c in Xs.columns else Xd[c].median()
        q1, q99 = Xs[c].quantile([0.01, 0.99]) if c in Xs.columns else Xd[c].quantile([0.01, 0.99])
        if c in Xs.columns:
            Xs[c] = Xs[c].fillna(med).clip(q1 - 6 * (q99 - q1 or 1), q99 + 6 * (q99 - q1) or np.inf)
        else:
            Xd[c] = Xd[c].fillna(med).clip(q1 - 6 * (q99 - q1 or 1), q99 + 6 * (q99 - q1) or np.inf)
    cat = df[CAT_COL].fillna("none").astype(str)
    codes, uniq = pd.factorize(cat)
    Xc = torch.tensor(codes, dtype=torch.long).unsqueeze(1)
    Xs_t = torch.tensor(Xs.to_numpy(), dtype=torch.float32)
    Xd_t = torch.tensor(Xd.to_numpy(), dtype=torch.float32)
    return Xs_t, Xd_t, Xc, len(uniq) + 1


def train_ft(df: pd.DataFrame, y: np.ndarray, tr_mask: np.ndarray, val_mask: np.ndarray,
             epochs: int = 24, batch: int = 4096, lr: float = 1e-3) -> np.ndarray:
    Xs, Xd, Xc, card = prepare(df)
    model = FTAttention(num_static=len(NUM_STATIC), num_dynamic=len(NUM_DYNAMIC), cat_card=card).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    posw = (y[tr_mask] == 0).sum() / max((y[tr_mask] == 1).sum(), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(posw), device=DEVICE))
    Xn_s, Xn_d, Xc_t = Xs.to(DEVICE), Xd.to(DEVICE), Xc.to(DEVICE)
    ys = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    y_s = ys.cpu().numpy()
    tr_idx = np.where(tr_mask)[0]
    best_ap, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + ep)
        np.random.shuffle(tr_idx)
        for i in range(0, len(tr_idx), batch):
            bi = tr_idx[i : i + batch]
            sl = torch.tensor(bi, device=DEVICE)
            opt.zero_grad()
            logits = model(Xn_s[sl], Xn_d[sl], Xc_t[sl])
            loss = lossf(logits, ys[sl])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xn_s[val_mask], Xn_d[val_mask], Xc_t[val_mask])).cpu().numpy()
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_s[val_mask], pv) if val_mask.sum() else -1
        if ep % 4 == 3:
            print(f"  ep{ep}: val PR-AUC {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, best_ep = ap, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    # predicción en lotes (la atención sobre 235k filas en un paso reventaba la VRAM)
    p_all = np.empty(len(y))
    with torch.no_grad():
        for i in range(0, len(y), 8192):
            p_all[i:i+8192] = torch.sigmoid(
                model(Xn_s[i:i+8192], Xn_d[i:i+8192], Xc_t[i:i+8192])).cpu().numpy()
    print(f"mejor época {best_ep} · val PR-AUC {best_ap:.4f} · device {DEVICE}")
    return p_all, model
