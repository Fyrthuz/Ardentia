"""Agrega estáticas res9 -> res8 (H3 parent)."""
from __future__ import annotations
from pathlib import Path
import h3
import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
df["h8"] = [h3.cell_to_parent(x, 8) for x in df["h9"]]

agg = df.groupby("h8").agg(
    forest_frac=("forest_frac", "mean"), land_px=("land_px", "sum"), fuel_px=("fuel_px", "sum"),
    elev_m=("elev_m", "mean"), slope_deg=("slope_deg", "mean"), aspect_deg=("aspect_deg", "mean"),
    dist_to_road=("dist_to_road", "min"), road_density_1km=("road_density_1km", "mean"),
    road_fuel_interface=("road_fuel_interface", "max"),
).reset_index()
agg["fuel_model"] = df.groupby("h8").apply(lambda g: g.loc[g["land_px"].idxmax(), "fuel_model"]).values
agg["road_class"] = df.loc[df.groupby("h8")["dist_to_road"].idxmin()].set_index("h8")["road_class"].reindex(agg["h8"]).values
agg["forest_frac"] = agg["fuel_px"] / agg["land_px"].clip(lower=1)
agg = agg.rename(columns={"h8": "h9"})
agg["aspect_sin"] = np.sin(np.radians(agg["aspect_deg"]))
agg["aspect_cos"] = np.cos(np.radians(agg["aspect_deg"]))
agg["dist_to_road_log"] = np.log1p(agg["dist_to_road"].clip(0, 5000))
agg["road_density_log"] = np.log1p(agg["road_density_1km"])
coords = [h3.cell_to_latlng(x) for x in agg["h9"]]
agg["lat"] = [c[0] for c in coords]
agg["lon"] = [c[1] for c in coords]
out = Path("data/processed/h3_cells/res8_static.parquet")
agg.to_parquet(out, index=False)
print(f"-> {out}: {len(agg):,} celdas res8 (desde {len(df):,} res9)")
