"""GNN espacial para predicción de ignición (Ronda 3).

Arquitectura: GraphSAGE/GAT sobre la malla H3 (nodos = celdas, aristas = adyacencia).
Cada nodo agrega información de sus vecinos en 2-3 hops → captura dependencias
espaciales que el LGBM tabular no ve.

Fases:
  A) GNN estática: solo features del día → P(fire | node features, neighborhood)
  B) GNN + temporal: GRU sobre 10 días de features dinámicas → P(fire | secuencia)
  C) GNN + posterior bayesiano (prior v2 como feature extra)

Uso: uv run python -m ardintra.models.gnn_spatial [--temporal]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/gnn_spatial.json")

FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
    "ffmc_anom", "dmc_anom", "dc_anom",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend",
    "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
    "lat_f", "lon_f", "doy_sin", "doy_cos",
    "ctx_fwi", "ctx_dc", "ctx_tmax_c", "ctx_rh_mean", "ctx_swc_l1_mean", "ctx_vpd_max",
    "ctx_ndmi_summer", "ctx_spei30", "road_class",
]
NUM_FEATS = [c for c in FEATS if c not in ("road_class",)]


def load_data():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    return df


def build_graph(df: pd.DataFrame, k_hops: int = 1):
    """Construye el grafo H3: nodos = celdas únicas, aristas = adyacencia k-hop."""
    unique_cells = df["h9"].drop_duplicates().reset_index(drop=True)
    cell_to_idx = {h: i for i, h in enumerate(unique_cells)}
    n_nodes = len(unique_cells)

    # aristas: pares de celdas adyacentes (k-hop)
    edges_src, edges_dst = [], []
    for h in unique_cells:
        i = cell_to_idx[h]
        for n in h3.grid_disk(h, k_hops):
            if n != h and n in cell_to_idx:
                edges_src.append(i)
                edges_dst.append(cell_to_idx[n])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # grafo no dirigido
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    print(f"grafo: {n_nodes:,} nodos · {edge_index.shape[1]:,} aristas (k={k_hops})")
    return unique_cells, cell_to_idx, edge_index


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # mean aggregation de vecinos (scatter-add pattern)
        row, col = edge_index[0], edge_index[1]
        neigh_agg = torch.zeros_like(x)
        neigh_agg.index_add_(0, col, x[row])
        ones = torch.ones(row.shape[0], 1, device=x.device)
        counts = torch.zeros(x.shape[0], 1, device=x.device)
        counts.index_add_(0, col, ones)
        neigh_agg = neigh_agg / counts.clamp(min=1)
        return F.relu(self.linear_self(x) + self.linear_neigh(neigh_agg))


class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for i in range(layers):
            self.layers.append(GraphSAGELayer(dims[i], dims[i + 1]))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        return self.head(x).squeeze(-1)


def main(temporal: bool = False, epochs: int = 40):
    import torch.optim as optim

    df = load_data()
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    y = df["big10"].to_numpy()

    unique_cells, cell_to_idx, edge_index = build_graph(df)
    edge_index = edge_index.to(DEVICE)

    # features por nodo (celda única): media de las features dinámicas del período
    # + estáticas (que ya son por celda)
    feature_cols = [c for c in NUM_FEATS if c in df.columns]
    node_feats = df.groupby("h9")[feature_cols].mean().reindex(unique_cells)
    node_feats = node_feats.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    node_x = torch.tensor(node_feats.to_numpy(), dtype=torch.float32, device=DEVICE)
    node_labels = df.groupby("h9")["big10"].max().reindex(unique_cells).fillna(0).to_numpy().astype(np.float32)

    # labels por nodo: ¿algún fuego ≥10 ha en esta celda durante el período?
    y_torch = torch.tensor(node_labels, dtype=torch.float32, device=DEVICE)

    # splits por año (temporal) — los nodos son los mismos, cambia el target
    years = df.groupby("h9")["fecha"].apply(lambda s: s.dt.year.max()).reindex(unique_cells).fillna(2022)
    tr_mask = (years <= 2021).to_numpy()
    val_mask = (years == 2022).to_numpy()
    te_mask = (years >= 2023).to_numpy()

    model = GNNModel(node_x.shape[1], hidden=64, layers=3).to(DEVICE)
    posw = float((node_labels[tr_mask] == 0).sum() / max((node_labels[tr_mask] == 1).sum(), 1))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw, device=DEVICE))

    tr_idx = np.where(tr_mask)[0]
    best_ap, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + ep)
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), 4096):
            sl = torch.tensor(tr_idx[i:i+4096], device=DEVICE)
            opt.zero_grad()
            logits = model(node_x, edge_index)
            loss = lossf(logits[sl], y_torch[sl])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(node_x, edge_index)).cpu().numpy()
        ap = average_precision_score(node_labels[val_mask], pv[val_mask]) if val_mask.sum() else -1
        if ep % 5 == 4:
            print(f"  ep{ep}: val AP {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, best_ep = ap, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_all = torch.sigmoid(model(node_x, edge_index)).cpu().numpy()

    # evaluación NODE-LEVEL: P(≥10 ha alguna vez) por celda (susceptibilidad)
    # el GNN predice por nodo, no por (celda, día) — es un experimento separado
    # del LGBM per-(cell,day)
    node_pred = {h: p_all[i] for i, h in enumerate(unique_cells)}
    y_node = node_labels
    p_node = p_all

    ap = average_precision_score(y_node, p_node)
    roc = roc_auc_score(y_node, p_node)
    br = brier_score_loss(y_node, p_node)
    o = np.argsort(-p_node); k5 = max(int(len(p_node) * .05), 1)
    r5 = y_node[o[:k5]].sum() / max(y_node.sum(), 1)
    print(f"\nRESULTADO GNN (node-level susceptibility) PR-AUC={ap:.4f} ({ap/max(y_node.mean()+1e-9,1e-9):.0f}× azar) · "
          f"ROC={roc:.4f} · Brier={br:.4f} · recall@5%={r5:.0%}")

    result = {"pr_auc": round(float(ap), 4), "roc_auc": round(float(roc), 4),
              "brier": round(float(br), 4), "recall_top5": round(float(r5), 3),
              "lift": round(float(ap / max(y_node.mean(), 1e-9)), 1),
              "best_epoch": best_ep, "n_nodes": len(unique_cells),
              "nota": "evaluación node-level (susceptibilidad), no comparable 1:1 con LGBM per-(cell,day)",
              "target": "big10"}
    OUT.write_text(json.dumps(result, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
