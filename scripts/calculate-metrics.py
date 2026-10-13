"""Calculate error metrics for solar irradiance nowcasts against satellite-derived truth.

For every day of the requested year, the DiT and GSO optical-flow (GSO-OF) nowcasts are
compared with the Himawari truth product. GSO-OF is only evaluated at the DiT issue times,
so both models are scored on the same forecasts.

Metrics are saved per forecast, i.e. with dims (time, lead_time) where `time` is the issue
time, as one NetCDF file per model per day. Forecasts without a complete truth field are
flagged with `valid = 0` and have NaN metrics. FSS is saved as its decomposed components so
it can be pooled over any set of forecasts; see `metric_aggregation.py` in src/.
"""

import argparse
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore")

sys.path.append("/home/548/cd3022/repos/nowcast-evaluation/src")
import logger  # noqa: E402
import nowcast_preprocess  # noqa: E402

LOG = logger.get_logger(__name__)

VARIABLE = "surface_global_irradiance"

DIT_DIR = Path("/scratch/er8/cd3022/CPDiT/inference")
GSO_OF_DIR = Path("/g/data/er8/nowcasts/nwc_20240701_20251231")
# Truth directories are organised by local (Australian) day, e.g. 2025/01/15 holds
# 2025-01-14T18:30 to 2025-01-15T11:30 UTC, whereas the nowcasts are organised by UTC day
TRUTH_DIR = Path("/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest")
SAVE_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/error-metric-data/per-forecast")

SPATIAL_DIMS = ["latitude", "longitude"]

# Buffer (degrees) removed from each edge of the DiT domain so evaluation is not
# impacted by boundary effects
DOMAIN_BUFFER = 1.0


# DATA LOADING
def _region_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Slice from lo to hi, respecting whether the coordinate is ascending or descending."""
    ascending = coord.values[0] <= coord.values[-1]
    return slice(lo, hi) if ascending else slice(hi, lo)


def _crop_to_region(ds: xr.Dataset, bounds: dict) -> xr.Dataset:
    return ds.sel(
        latitude=_region_slice(ds["latitude"], bounds["lat_min"], bounds["lat_max"]),
        longitude=_region_slice(ds["longitude"], bounds["lon_min"], bounds["lon_max"]),
    )[[VARIABLE]].reset_coords(drop=True)


def preprocess_dit(ds: xr.Dataset) -> xr.Dataset:
    ds = nowcast_preprocess.preprocess_nowcast_dataset(
        ds,
        f_dim_name="lead_time",
        new_f_dim_name="lead_time",
    )
    # Drop non-index coords (init_time, valid_time) so they don't leak into the metrics
    return ds[[VARIABLE]].mean(dim="member").reset_coords(drop=True)


def preprocess_gso_of(ds: xr.Dataset, bounds: dict) -> xr.Dataset:
    ds = nowcast_preprocess.preprocess_nowcast_dataset(
        ds,
        f_dim_name="time",
        new_f_dim_name="lead_time",
        relative_leadtime=True,
    )
    return _crop_to_region(ds, bounds)


def preprocess_truth(ds: xr.Dataset, bounds: dict) -> xr.Dataset:
    return _crop_to_region(ds, bounds)


def open_files(files: list[Path], preprocess) -> xr.Dataset:
    """Open many single-time files as one dataset concatenated along `time`."""
    return xr.open_mfdataset(
        files,
        preprocess=preprocess,
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=True,
        chunks="auto",
    )


def _files_at_times(files: list[Path], times: set[pd.Timestamp], time_fmt: str) -> list[Path]:
    """Keep the files whose timestamp (parsed from the file name) is in `times`."""
    return [f for f in files if pd.to_datetime(f.stem, format=time_fmt) in times]


def _truth_file_time(path: Path) -> pd.Timestamp:
    """Truth files are named IDE00326.YYYYmmddHHMM.nc"""
    return pd.to_datetime(path.stem.split(".")[1], format="%Y%m%d%H%M")


# METRIC CALCULATION
def _fss_components_2d(
    f: np.ndarray, o: np.ndarray, threshold: float, window: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decomposed FSS components for each (lat, lon) field in the trailing two axes.

    Follows the `scores.spatial.fss_2d` definition (event = value > threshold, windows
    restricted to the interior of the domain), but returns the components per field:
    mean(Pf^2), mean(Po^2) and mean((Pf - Po)^2), where P are neighbourhood event
    fractions. FSS over any set of fields is 1 - sum(diff) / (sum(fcst) + sum(obs)).
    """
    wh, ww = window

    def fractions(x: np.ndarray) -> np.ndarray:
        events = (x > threshold).astype(np.float64)
        csum = np.pad(events.cumsum(-2).cumsum(-1), [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)])
        window_sum = (
            csum[..., wh:, ww:] - csum[..., :-wh, ww:] - csum[..., wh:, :-ww] + csum[..., :-wh, :-ww]
        )
        return window_sum / (wh * ww)

    pf, po = fractions(f), fractions(o)
    spatial_axes = (-2, -1)
    return (
        np.mean(pf**2, axis=spatial_axes),
        np.mean(po**2, axis=spatial_axes),
        np.mean((pf - po) ** 2, axis=spatial_axes),
    )


def fss_components(
    f: xr.DataArray, o: xr.DataArray, threshold: float, window: tuple[int, int]
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """FSS components (fcst, obs, diff) for each (lat, lon) field."""
    return xr.apply_ufunc(
        _fss_components_2d,
        f,
        o,
        input_core_dims=[SPATIAL_DIMS, SPATIAL_DIMS],
        output_core_dims=[[], [], []],
        kwargs={"threshold": threshold, "window": window},
    )


def _ssim_2d(f2d: np.ndarray, o2d: np.ndarray, data_range: float, sigma: float) -> float:
    """SSIM for a single (lat, lon) field. NaN if either field has missing data."""
    if np.isnan(f2d).any() or np.isnan(o2d).any():
        return np.nan
    return structural_similarity(
        f2d,
        o2d,
        data_range=data_range,
        gaussian_weights=True,  # Wang et al. (2004) formulation
        sigma=sigma,
        use_sample_covariance=False,
    )


def ssim_field(f: xr.DataArray, o: xr.DataArray, data_range: float, sigma: float) -> xr.DataArray:
    """SSIM for each (lat, lon) field, looping over all other dimensions (e.g. time)."""
    return xr.apply_ufunc(
        _ssim_2d,
        f,
        o,
        input_core_dims=[SPATIAL_DIMS, SPATIAL_DIMS],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"data_range": data_range, "sigma": sigma},
    )


def calc_error_metrics(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    fss_threshold: float = 500,  # W/m2
    fss_window_small: tuple[int, int] = (5, 5),  # height * width
    fss_window_large: tuple[int, int] = (100, 100),  # height * width
    ssim_data_range: float = 1200,  # W/m2, range of values in image
    ssim_sigma: float = 1.5,
) -> xr.Dataset:
    """Return per-forecast MSE, MAE, SSIM and FSS components (small/large window).

    `fcst` has dims (time, lead_time, latitude, longitude), where `time` is the issue time.
    `obs` has dims (time, latitude, longitude), where `time` is the valid time.
    The output has dims (time, lead_time). A forecast is only scored (`valid` = 1) when both
    the forecast and the observed field at its valid time are complete; otherwise all of its
    metrics are NaN. Missing data must not be scored: FSS would treat it as "no event".
    """
    # Put obs on exactly the forecast grid (guards against float coordinate mismatch)
    obs = obs.sel(
        {"latitude": fcst["latitude"], "longitude": fcst["longitude"]}, method="nearest"
    )
    obs = obs.assign_coords({"latitude": fcst["latitude"], "longitude": fcst["longitude"]})

    per_lead = []
    for lead in fcst["lead_time"].values:
        # Needed for FSS and SSIM, which operate on in-memory arrays
        f = fcst.sel(lead_time=lead).load()
        # Observations at each forecast's valid time (NaN where obs are missing)
        o = obs.reindex(time=f["time"].values + lead)
        o = o.assign_coords(time=f["time"].values).load()

        valid = f.notnull().all(dim=SPATIAL_DIMS) & o.notnull().all(dim=SPATIAL_DIMS)

        err = f - o
        metrics = {
            "valid": valid.astype(np.int8),
            "mse": (err**2).mean(dim=SPATIAL_DIMS),
            "mae": np.abs(err).mean(dim=SPATIAL_DIMS),
            "ssim": ssim_field(f, o, data_range=ssim_data_range, sigma=ssim_sigma),
        }
        for name, window in (("small", fss_window_small), ("large", fss_window_large)):
            comps = fss_components(f, o, threshold=fss_threshold, window=window)
            for comp_name, comp in zip(("fcst", "obs", "diff"), comps):
                metrics[f"fss_{name}_{comp_name}"] = comp
        ds = xr.Dataset(metrics)
        # Missing data is never scored
        ds = ds.where(valid).assign(valid=metrics["valid"])
        per_lead.append(ds)

    metrics = xr.concat(per_lead, dim=fcst["lead_time"]).transpose("time", "lead_time")
    metrics.attrs = {
        "fss_threshold_W_m-2": fss_threshold,
        "fss_window_small": list(fss_window_small),
        "fss_window_large": list(fss_window_large),
        "ssim_data_range_W_m-2": ssim_data_range,
        "ssim_sigma": ssim_sigma,
        "note": (
            "time is the issue time. Metrics are per forecast and NaN where valid == 0 "
            "(incomplete truth or forecast field). FSS = 1 - sum(diff) / (sum(fcst) + "
            "sum(obs)), summed over the forecasts being aggregated."
        ),
    }
    return metrics


# DAILY PROCESSING
def _day_dir(date: pd.Timestamp) -> Path:
    return Path(f"{date.year}/{date.month:02d}/{date.day:02d}")


def calc_daily_metrics(date: pd.Timestamp) -> dict[str, xr.Dataset] | None:
    """Return {model name: metrics} for one UTC day, or None if there is nothing to score."""
    dit_files = sorted((DIT_DIR / _day_dir(date)).glob("*.nc"))
    if not dit_files:
        LOG.warning(f"Skipping {date.date()}: no DiT files in {DIT_DIR / _day_dir(date)}")
        return None
    issue_times = {nowcast_preprocess.file_name_to_datetime64(f) for f in dit_files}
    issue_times = set(pd.DatetimeIndex(list(issue_times)))

    LOG.info(f"Calculating metrics for {date.date()} ({len(dit_files)} DiT issue times)")

    dit = open_files(dit_files, preprocess_dit)
    LOG.info("Loaded DiT nowcasts")

    # Use the DiT domain, minus a buffer, as the region for all datasets
    bounds = {
        "lat_min": dit["latitude"].min().item() + DOMAIN_BUFFER,
        "lat_max": dit["latitude"].max().item() - DOMAIN_BUFFER,
        "lon_min": dit["longitude"].min().item() + DOMAIN_BUFFER,
        "lon_max": dit["longitude"].max().item() - DOMAIN_BUFFER,
    }
    dit = _crop_to_region(dit, bounds)

    # Truth for valid times up to issue time + max lead time. These span two local-day
    # truth directories, so search both and keep only the valid times that are needed.
    valid_times = {
        t + pd.Timedelta(lead) for t in issue_times for lead in dit["lead_time"].values
    }
    truth_files = [
        f
        for d in (date, date + pd.Timedelta(days=1))
        for f in sorted((TRUTH_DIR / _day_dir(d)).glob("*.nc"))
        if _truth_file_time(f) in valid_times
    ]
    if not truth_files:
        LOG.warning(f"Skipping {date.date()}: no truth files for the DiT valid times")
        return None
    truth = open_files(truth_files, partial(preprocess_truth, bounds=bounds))
    LOG.info(f"Loaded truth data ({len(truth_files)}/{len(valid_times)} valid times found)")

    nowcasts = {"CPDiT": dit}

    # GSO-OF only at the DiT issue times, so both models are scored on the same forecasts
    gso_of_files = _files_at_times(
        sorted((GSO_OF_DIR / _day_dir(date)).glob("*.nc")), issue_times, "%Y%m%dT%H%M%S"
    )
    if gso_of_files:
        nowcasts["GSO_OF"] = open_files(gso_of_files, partial(preprocess_gso_of, bounds=bounds))
        LOG.info(f"Loaded GSO-OF nowcasts ({len(gso_of_files)}/{len(issue_times)} issue times)")
    else:
        LOG.warning(f"No GSO-OF files at the DiT issue times on {date.date()}")

    results = {}
    for name, nowcast in nowcasts.items():
        metrics = calc_error_metrics(nowcast[VARIABLE], truth[VARIABLE])
        # Lead time in minutes
        results[name] = metrics.assign_coords(
            lead_time=metrics["lead_time"] / np.timedelta64(1, "m")
        )
        n_valid = int(metrics["valid"].sum())
        LOG.info(f"Calculated {name} metrics: {n_valid}/{metrics['valid'].size} forecasts valid")

    return results


def main(date: pd.Timestamp, save_dir: Path) -> None:
    metrics = calc_daily_metrics(date)

    if not metrics:
        LOG.info(f"No data for {date.date()}")
        return

    for name, ds in metrics.items():
        out_dir = save_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Saving {name} metrics...")
        ds.to_netcdf(out_dir / f"{date.year}{date.month:02d}{date.day:02d}.nc")
    LOG.info("Saved metrics to netcdf files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default="2025-01-01")
    parser.add_argument("--n-workers", type=int, default=24)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    args = parser.parse_args()

    with Client(n_workers=args.n_workers, threads_per_worker=1):
        main(pd.Timestamp(args.date), args.save_dir)
