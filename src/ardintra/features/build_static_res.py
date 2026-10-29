"""Agrega estáticas res9 → res{N} (H3 parent) — parametrizado por ARDENTIA_RES."""
from __future__ import annotations

import os
from pathlib import Path

import h3
import numpy as np
import pandas as pd

RES = int(os.environ.get("ARDENTIA_RES", "7"))

df = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
df[f"h{RES}"] = [h3.cell_to_parent(x, RES) for x in df["h9"]]

agg = df.groupby(f"h{RES}").agg(
    forest_frac=("forest_frac", "mean"), land_px=("land_px", "sum"), fuel_px=("fuel_px", "sum"),
    elev_m=("elev_m", "mean"), slope_deg=("slope_deg", "mean"), aspect_deg=("aspect_deg", "mean"),
    dist_to_road=("dist_to_road", "min"), road_density_1km=("road_density_1km", "mean"),
    road_fuel_interface=("road_fuel_interface", "max"),
).reset_index()
agg["fuel_model"] = df.groupby(f"h{RES}").apply(lambda g: g.loc[g["land_px"].idxmax(), "fuel_model"]).values
agg["road_class"] = df.loc[df.groupby(f"h{RES}")["dist_to_road"].idxmin()].set_index(f"h{RES}")["road_class"].reindex(agg[f"h{RES}"]).values
agg["forest_frac"] = agg["fuel_px"] / agg["land_px"].clip(lower=1)
agg["aspect_sin"] = np.sin(np.radians(agg["aspect_deg"]))
agg["aspect_cos"] = np.cos(np.radians(agg["aspect_deg"]))
agg["dist_to_road_log"] = np.log1p(agg["dist_to_road"].clip(0, 5000))
agg["road_density_log"] = np.log1p(agg["road_density_1km"])
coords = [h3.cell_to_latlng(x) for x in agg[f"h{RES}"]]
agg["lat"] = [c[0] for c in coords]
agg["lon"] = [c[1] for c in coords]

# el id SIEMPRE se llama h9 aguas abajo (convención del builder)
agg = agg.rename(columns={f"h{RES}": "h9"})
out = Path(f"data/processed/h3_cells/res{RES}_static.parquet")
agg.to_parquet(out, index=False)
print(f"-> {out}: {len(agg):,} celdas res{RES} (desde {len(df):,} res9)")
