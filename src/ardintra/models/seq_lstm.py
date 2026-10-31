"""Secuencias temporales: LSTM vs LGBM con contexto temporal explícito.

Fase 3 (estilo Mesogeos track A, adaptado a nuestro muestreo):
  a) LGBM + resúmenes de ventana (mean/max/min/delta de 14 días)
  b) LGBM + secuencia aplanada (112 features)
  c) LSTM (PyTorch, GPU) sobre la secuencia 14×8

Target: big10 (ignición ≥10 ha) · Train ≤2022 · Test 2023-24 ·
Early stopping del LSTM en 2022 (val).

Uso: uv run python -m ardintra.models.seq_lstm
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

SEQ = Path("data/processed/features/v1/sequences.npz")
OUT = Path("reports/seq_results.json")


def metrics(y, p):
    return {"auc": round(float(roc_auc_score(y, p)), 4),
            "ap": round(float(average_precision_score(y, p)), 4)}


def main():
    d = np.load(SEQ, allow_pickle=True)
    y = d["big10"].astype(int)
    years = d["years"]
    tr, te = years <= 2022, years >= 2023
    val = years == 2022  # para early stopping del LSTM
    print(f"train {tr.sum():,} · test {te.sum():,} · positivos test {y[te].sum()}")

    results = {}

    # --- a) LGBM + resúmenes
    Xs = d["X_sum"]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                           min_child_samples=100, scale_pos_weight=38, random_state=42, verbosity=-1)
    m.fit(Xs[tr], y[tr])
    results["lgbm_sumarios"] = metrics(y[te], m.predict_proba(Xs[te])[:, 1])
    print("LGBM sumarios 14d:", results["lgbm_sumarios"])

    # --- b) LGBM + secuencia aplanada
    Xf = d["X_flat"]
    m2 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                            min_child_samples=100, scale_pos_weight=38, random_state=42, verbosity=-1)
    m2.fit(Xf[tr], y[tr])
    results["lgbm_flat14"] = metrics(y[te], m2.predict_proba(Xf[te])[:, 1])
    print("LGBM flat 14d:", results["lgbm_flat14"])

    # --- c) LSTM
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = d["X_seq"].astype(np.float32)                     # [n, T, V]
    mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    Xn = (X - mu) / sd

    class LSTMNet(nn.Module):
        def __init__(self, v=X.shape[2], h=64):
            super().__init__()
            self.lstm = nn.LSTM(v, h, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1]).squeeze(-1)

    torch.manual_seed(42)
    net = LSTMNet().to(dev)
    posw = torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def loader(idx, shuffle=False):
        ds = TensorDataset(torch.from_numpy(Xn[idx]), torch.from_numpy(y[idx]).float())
        return DataLoader(ds, batch_size=4096, shuffle=shuffle, num_workers=2, pin_memory=True)

    best_auc, best_state, bad = -1, None, 0
    for ep in range(30):
        net.train()
        for xb, yb in loader(np.where(tr)[0], shuffle=True):
            opt.zero_grad()
            out = net(xb.to(dev))
            loss = lossf(out, yb.to(dev))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xb.to(dev)).cpu() for xb, _ in loader(np.where(val)[0])])
            pv = torch.sigmoid(torch.tensor(pv)).numpy()
        auc_v = roc_auc_score(y[val], pv)
        if auc_v > best_auc:
            best_auc, best_state, bad = auc_v, {k: t.cpu().clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or bad >= 5:
            print(f"  ep{ep}: val AUC {auc_v:.4f} (best {best_auc:.4f})", flush=True)
        if bad >= 5:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pt = torch.sigmoid(torch.cat([net(xb.to(dev)).cpu() for xb, _ in loader(np.where(te)[0])])).numpy()
    results["lstm_14d"] = metrics(y[te], pt)
    print("LSTM 14d:", results["lstm_14d"])
    print(f"(val AUC best: {best_auc:.4f})")

    # referencia: LGBM tabular día-D (sin secuencia) — del run previo
    results["ref_lgbm_tabular_diaD"] = {"auc": 0.8970, "ap": 0.0364}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
