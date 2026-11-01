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
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import xarray as xr


def _is_per_forecast(path: Path) -> bool:
    with xr.open_dataset(path) as ds:
        return "time" in ds.dims and "valid" in ds


def load_metrics(model_dir: Path) -> xr.Dataset:
    """Open all daily per-forecast metric files for one model, concatenated on issue time.

    Files without per-forecast metrics (no `time` dim or `valid` flag) are skipped. These are
    left over from older versions of calculate-metrics.py, which saved daily aggregates on a
    `date` dim, and survive a rerun on days where that model had no forecasts to score.
    """
    files = sorted(Path(model_dir).glob("*.nc"))
    stale = [f for f in files if not _is_per_forecast(f)]
    if stale:
        warnings.warn(
            f"Skipping {len(stale)} file(s) in {model_dir} without per-forecast metrics: "
            + ", ".join(f.name for f in stale)
        )
    files = [f for f in files if f not in stale]
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


def _pooled_fss(ds: xr.Dataset, window: str, level: str, dim: str) -> xr.DataArray:
    fcst = ds[f"fss_{window}_{level}_fcst"].sum(dim)
    obs = ds[f"fss_{window}_{level}_obs"].sum(dim)
    diff = ds[f"fss_{window}_{level}_diff"].sum(dim)
    denom = fcst + obs
    # FSS is undefined when no events were forecast or observed
    return (1 - diff / denom).where(denom > 0)


def aggregate(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """Aggregate per-forecast metrics over `dim`, weighting every valid forecast equally.

    Returns the number of valid forecasts, RMSE (root of the pooled MSE), MAE, mean SSIM,
    mean CRPS (ensemble models only) and pooled FSS (clear-sky index) for the small and
    large windows at the light (0.8) and deep (0.5) cloud thresholds, named e.g.
    `fss_small_light`.
    """
    valid = ds["valid"].fillna(0).astype(bool)
    ds = ds.where(valid)
    return xr.Dataset(
        {
            "n": valid.sum(dim),
            # Every field has the same number of pixels, so this is the pooled RMSE
            "rmse": np.sqrt((ds["rmse"] ** 2).mean(dim)),
            "mae": ds["mae"].mean(dim),
            "ssim": ds["ssim"].mean(dim),
            **({"crps": ds["crps"].mean(dim)} if "crps" in ds else {}),
            **{
                f"fss_{window}_{level}": _pooled_fss(ds, window, level, dim)
                for window in ("small", "large")
                for level in ("light", "deep")
            },
        }
    )
