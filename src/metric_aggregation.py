"""Aggregate the per-forecast metrics written by scripts/calculate-metrics.py.

Typical use:

    import metric_aggregation as ma
    metrics = {m: ma.load_metrics(SAVE_DIR / m) for m in ("CPDiT", "GSO_OF")}
    metrics = ma.common_sample(metrics)
    scores = {m: ma.aggregate(ds) for m, ds in metrics.items()}  # dims: lead_time

`aggregate` reduces over issue times, so it can also be applied per group, e.g.
`ds.groupby("time.month").map(ma.aggregate)`.
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import xarray as xr


def load_metrics(model_dir: Path) -> xr.Dataset:
    """Open all daily per-forecast metric files for one model, concatenated on issue time."""
    files = sorted(Path(model_dir).glob("*.nc"))
    return xr.open_mfdataset(files, combine="nested", concat_dim="time").load()


def common_sample(
    metrics: dict[str, xr.Dataset], complete_leads: bool = False
) -> dict[str, xr.Dataset]:
    """Restrict every model to the forecasts that are valid for all models.

    If `complete_leads` is True, also drop issue times that are not valid at every lead
    time, so each lead time is averaged over exactly the same issue times. Otherwise the
    set of issue times can differ slightly between lead times (e.g. where a truth field is
    missing at one valid time).
    """
    aligned = xr.align(*metrics.values(), join="inner")
    valid = reduce(operator.and_, [ds["valid"].fillna(0).astype(bool) for ds in aligned])
    if complete_leads:
        valid = valid.all(dim="lead_time")
    return {
        name: ds.where(valid).assign(valid=valid.broadcast_like(ds["valid"]).astype(np.int8))
        for name, ds in zip(metrics, aligned)
    }


def _pooled_fss(ds: xr.Dataset, window: str, dim: str) -> xr.DataArray:
    fcst = ds[f"fss_{window}_fcst"].sum(dim)
    obs = ds[f"fss_{window}_obs"].sum(dim)
    diff = ds[f"fss_{window}_diff"].sum(dim)
    denom = fcst + obs
    # FSS is undefined when no events were forecast or observed
    return (1 - diff / denom).where(denom > 0)


def aggregate(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """Aggregate per-forecast metrics over `dim`, weighting every valid forecast equally.

    Returns the number of valid forecasts, RMSE (root of the pooled MSE), MAE, mean SSIM
    and pooled FSS for the small and large windows.
    """
    valid = ds["valid"].fillna(0).astype(bool)
    ds = ds.where(valid)
    return xr.Dataset(
        {
            "n": valid.sum(dim),
            "rmse": np.sqrt(ds["mse"].mean(dim)),
            "mae": ds["mae"].mean(dim),
            "ssim": ds["ssim"].mean(dim),
            "fss_small": _pooled_fss(ds, "small", dim),
            "fss_large": _pooled_fss(ds, "large", dim),
        }
    )
