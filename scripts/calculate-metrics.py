"""Calculate error metrics for solar irradiance nowcasts against satellite-derived truth.

For every day of the requested year, the DiT and GSO optical-flow (GSO-OF) nowcasts and a
smart persistence baseline are compared with the Himawari truth product. GSO-OF and smart
persistence are only evaluated at the DiT issue times, so all models are scored on the same
forecasts.

Metrics are saved per forecast, i.e. with dims (time, lead_time) where `time` is the issue
time, as one NetCDF file per model per day. Forecasts without a complete truth field are
flagged with `valid = 0` and have NaN metrics. FSS is saved as its decomposed components so
it can be pooled over any set of forecasts; see `metric_aggregation.py` in src/.

RMSE, MAE, SSIM and CRPS (ensembles only, currently DiT) are computed on irradiance. FSS is computed on the clear-sky index
(CSI = all-sky irradiance / clear-sky irradiance), where a cloud event is CSI < threshold.
Clear-sky irradiance comes from BARRA-C2.
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

DIT_DIR = Path("/scratch/er8/cd3022/CPDiT/stage2_checkpoints/inference/")
GSO_OF_DIR = Path("/g/data/er8/nowcasts/nwc_20240701_20251231")
# Truth directories are organised by local (Australian) day, e.g. 2025/01/15 holds
# 2025-01-14T18:30 to 2025-01-15T11:30 UTC, whereas the nowcasts are organised by UTC day
TRUTH_DIR = Path("/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest")
SAVE_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/error-metric-data")

# BARRA-C2 hourly-mean clear-sky irradiance, in monthly files. Timestamps are the middle of
# the averaging hour (HH:30)
CLEAR_SKY_DIR = Path(
    "/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/rsdscs/latest"
)

SPATIAL_DIMS = ["latitude", "longitude"]
ENSEMBLE_DIM = "member"

# Clear-sky index thresholds below which a pixel counts as cloudy
CSI_THRESHOLDS = {"light": 0.8, "deep": 0.5}  # light cloud, deep cloud

# CSI is undefined when the sun is low or down. Pixels with clear-sky irradiance below this
# (W/m2) get a NaN CSI, and so are never an event in the forecast or the truth
MIN_CLEAR_SKY = 50.0

# Smart persistence holds the issue-time CSI fixed. Where the CSI is undefined (sun too low)
# the sky is assumed clear (CSI = 1), and the CSI is capped so a noisy low-sun value isn't
# scaled up by a much larger clear-sky irradiance later in the day
SMART_PERSISTENCE_MAX_CSI = 1.2

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
    # Keep the ensemble members for CRPS. Drop non-index coords (init_time, valid_time) so
    # they don't leak into the metrics
    return ds[[VARIABLE]].reset_coords(drop=True)


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


def load_clear_sky(grid: xr.DataArray, start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
    """Clear-sky irradiance on the lat/lon grid of `grid`, covering [start, end].

    One hour of padding is loaded either side so it can later be interpolated in time.
    """
    months = pd.period_range(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), freq="M")
    files = []
    for m in months:
        pattern = f"rsdscs_*_{m.year}{m.month:02d}-{m.year}{m.month:02d}.nc"
        match = next(CLEAR_SKY_DIR.glob(pattern), None)
        if match is None:
            # On Gadi this also happens when gdata/ob53 is missing from the PBS storage list
            raise FileNotFoundError(f"No clear-sky file matching {CLEAR_SKY_DIR / pattern}")
        files.append(match)
    csk = xr.open_mfdataset(files, combine="by_coords", chunks="auto")["rsdscs"]
    csk = csk.rename({"lat": "latitude", "lon": "longitude"}).reset_coords(drop=True)
    csk = csk.sel(time=slice(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1)))
    # Pad the crop by a grid cell so the interpolation has neighbours at the edges
    pad = 0.1
    csk = csk.sel(
        latitude=slice(grid["latitude"].min().item() - pad, grid["latitude"].max().item() + pad),
        longitude=slice(grid["longitude"].min().item() - pad, grid["longitude"].max().item() + pad),
    )
    return csk.interp(latitude=grid["latitude"], longitude=grid["longitude"]).load()


def smart_persistence(
    obs: xr.DataArray,
    clear_sky: xr.DataArray,
    grid: xr.DataArray,
    issue_times: pd.DatetimeIndex,
    lead_times: np.ndarray,
) -> xr.DataArray:
    """Smart persistence forecast: the observed clear-sky index at the issue time, times the
    clear-sky irradiance at each valid time (clouds persist, the sun moves).

    Returns dims (time, lead_time, latitude, longitude) on the lat/lon grid of `grid`, where
    `time` is the issue time. NaN where the observation at the issue time is missing.
    """
    lat, lon = grid["latitude"].values, grid["longitude"].values
    obs = obs.sel(latitude=lat, longitude=lon, method="nearest")
    obs = obs.assign_coords(latitude=lat, longitude=lon)

    o0 = obs.reindex(time=issue_times)
    csk0 = clear_sky.interp(time=issue_times.values)
    csi0 = (o0 / csk0).where(csk0 >= MIN_CLEAR_SKY, 1.0).clip(max=SMART_PERSISTENCE_MAX_CSI)
    csi0 = csi0.where(o0.notnull()).load()

    per_lead = []
    for lead in lead_times:
        csk = clear_sky.interp(time=issue_times.values + lead)
        per_lead.append(csi0 * csk.assign_coords(time=issue_times.values))
    fcst = xr.concat(per_lead, dim=xr.DataArray(lead_times, dims="lead_time", name="lead_time"))
    return fcst.transpose("time", "lead_time", *SPATIAL_DIMS).rename(VARIABLE)


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
    f: np.ndarray, o: np.ndarray, threshold: float, window: tuple[int, int], below: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decomposed FSS components for each (lat, lon) field in the trailing two axes.

    Follows the `scores.spatial.fss_2d` definition (event = value > threshold, or
    value < threshold if `below`; NaN is never an event; windows
    restricted to the interior of the domain), but returns the components per field:
    mean(Pf^2), mean(Po^2) and mean((Pf - Po)^2), where P are neighbourhood event
    fractions. FSS over any set of fields is 1 - sum(diff) / (sum(fcst) + sum(obs)).
    """
    wh, ww = window

    def fractions(x: np.ndarray) -> np.ndarray:
        events = ((x < threshold) if below else (x > threshold)).astype(np.float64)
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
    f: xr.DataArray, o: xr.DataArray, threshold: float, window: tuple[int, int], below: bool = False
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """FSS components (fcst, obs, diff) for each (lat, lon) field."""
    return xr.apply_ufunc(
        _fss_components_2d,
        f,
        o,
        input_core_dims=[SPATIAL_DIMS, SPATIAL_DIMS],
        output_core_dims=[[], [], []],
        kwargs={"threshold": threshold, "window": window, "below": below},
    )


def crps_ensemble(f: xr.DataArray, o: xr.DataArray, member_dim: str = ENSEMBLE_DIM) -> xr.DataArray:
    """Pixelwise ensemble CRPS (empirical CDF of the members, not the "fair" CRPS).

    CRPS = mean|X - y| - 0.5 * mean|X - X'|, with the spread term computed from the sorted
    members as (2 / M^2) * sum_i (2i - M - 1) x_(i), avoiding the M x M pairwise
    differences. Same as `scores.probability.crps_for_ensemble(method="ecdf")`.
    """
    m = f.sizes[member_dim]
    skill = np.abs(f - o).mean(dim=member_dim)
    sorted_members = xr.apply_ufunc(
        np.sort,
        f,
        input_core_dims=[[member_dim]],
        output_core_dims=[[member_dim]],
        kwargs={"axis": -1},
    )
    weights = xr.DataArray(2 * np.arange(1, m + 1) - m - 1, dims=member_dim)
    spread = 2 * (sorted_members * weights).sum(dim=member_dim) / m**2
    return skill - 0.5 * spread


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
    clear_sky: xr.DataArray,
    fss_thresholds: dict[str, float] = CSI_THRESHOLDS,  # clear-sky index
    fss_window_small: tuple[int, int] = (5, 5),  # height * width
    fss_window_large: tuple[int, int] = (100, 100),  # height * width
    ssim_data_range: float = 1200,  # W/m2, range of values in image
    ssim_sigma: float = 1.5,
) -> xr.Dataset:
    """Return per-forecast RMSE, MAE, SSIM, FSS components (small/large window) and, for
    ensemble forecasts, CRPS.

    `fcst` has dims (time, lead_time, latitude, longitude), where `time` is the issue time,
    plus a `member` dim for an ensemble. RMSE, MAE, SSIM and FSS are computed on the
    ensemble mean and CRPS on the members.
    `obs` has dims (time, latitude, longitude), where `time` is the valid time.
    `clear_sky` is the clear-sky irradiance on the forecast grid, with dims (time,
    latitude, longitude) and a time axis that brackets all valid times.
    The output has dims (time, lead_time). A forecast is only scored (`valid` = 1) when both
    the forecast and the observed field at its valid time are complete; otherwise all of its
    metrics are NaN. Missing data must not be scored: FSS would treat it as "no event".
    """
    # Put obs on exactly the forecast grid (guards against float coordinate mismatch)
    obs = obs.sel(
        {"latitude": fcst["latitude"], "longitude": fcst["longitude"]}, method="nearest"
    )
    obs = obs.assign_coords({"latitude": fcst["latitude"], "longitude": fcst["longitude"]})

    is_ensemble = ENSEMBLE_DIM in fcst.dims

    per_lead = []
    for lead in fcst["lead_time"].values:
        # Needed for FSS and SSIM, which operate on in-memory arrays
        f_members = fcst.sel(lead_time=lead).load()
        # Observations at each forecast's valid time (NaN where obs are missing)
        o = obs.reindex(time=f_members["time"].values + lead)
        o = o.assign_coords(time=f_members["time"].values).load()

        # A forecast is complete only if every member is
        f_complete = f_members.notnull().all(
            dim=SPATIAL_DIMS + ([ENSEMBLE_DIM] if is_ensemble else [])
        )
        valid = f_complete & o.notnull().all(dim=SPATIAL_DIMS)
        f = f_members.mean(dim=ENSEMBLE_DIM) if is_ensemble else f_members

        # Clear-sky index for FSS, with the clear-sky irradiance interpolated in time to
        # each valid time. NaN (never an event) where the sun is too low for CSI to be defined
        csk = clear_sky.interp(time=f["time"].values + lead)
        csk = csk.assign_coords(time=f["time"].values)
        csk = csk.where(csk >= MIN_CLEAR_SKY)
        f_csi, o_csi = f / csk, o / csk

        err = f - o
        metrics = {
            "valid": valid.astype(np.int8),
            "rmse": np.sqrt((err**2).mean(dim=SPATIAL_DIMS)),
            "mae": np.abs(err).mean(dim=SPATIAL_DIMS),
            "ssim": ssim_field(f, o, data_range=ssim_data_range, sigma=ssim_sigma),
        }
        if is_ensemble:
            metrics["crps"] = crps_ensemble(f_members, o).mean(dim=SPATIAL_DIMS)
        for name, window in (("small", fss_window_small), ("large", fss_window_large)):
            for level, threshold in fss_thresholds.items():
                comps = fss_components(f_csi, o_csi, threshold, window, below=True)
                for comp_name, comp in zip(("fcst", "obs", "diff"), comps):
                    metrics[f"fss_{name}_{level}_{comp_name}"] = comp
        ds = xr.Dataset(metrics)
        # Missing data is never scored
        ds = ds.where(valid).assign(valid=metrics["valid"])
        per_lead.append(ds)

    metrics = xr.concat(per_lead, dim=fcst["lead_time"]).transpose("time", "lead_time")
    metrics.attrs = {
        **{f"fss_csi_threshold_{k}": v for k, v in fss_thresholds.items()},
        "fss_event": "clear-sky index < threshold",
        "clear_sky_min_W_m-2": MIN_CLEAR_SKY,
        "fss_window_small": list(fss_window_small),
        "fss_window_large": list(fss_window_large),
        "ssim_data_range_W_m-2": ssim_data_range,
        "ssim_sigma": ssim_sigma,
        "note": (
            "time is the issue time. Metrics are per forecast and NaN where valid == 0 "
            "(incomplete truth or forecast field). For ensembles, RMSE, MAE, SSIM and FSS are "
            "of the ensemble mean and CRPS (empirical CDF) is of the members. Pool RMSE as "
            "sqrt(mean(rmse^2)), not mean(rmse). FSS (on clear-sky "
            "index) = 1 - sum(diff) / (sum(fcst) + sum(obs)), summed over the forecasts being "
            "aggregated."
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

    # Truth for valid times up to issue time + max lead time, plus the issue times themselves
    # for smart persistence. These span two local-day truth directories, so search both and
    # keep only the times that are needed.
    valid_times = {
        t + pd.Timedelta(lead) for t in issue_times for lead in dit["lead_time"].values
    }
    truth_times = valid_times | issue_times
    truth_files = [
        f
        for d in (date, date + pd.Timedelta(days=1))
        for f in sorted((TRUTH_DIR / _day_dir(d)).glob("*.nc"))
        if _truth_file_time(f) in truth_times
    ]
    if not truth_files:
        LOG.warning(f"Skipping {date.date()}: no truth files for the DiT valid times")
        return None
    truth = open_files(truth_files, partial(preprocess_truth, bounds=bounds))
    LOG.info(f"Loaded truth data ({len(truth_files)}/{len(truth_times)} times found)")

    clear_sky = load_clear_sky(
        dit,
        min(issue_times),
        max(issue_times) + dit["lead_time"].max().to_numpy(),
    )
    LOG.info("Loaded clear-sky irradiance")

    nowcasts = {"CPDiT": dit}

    sorted_issue_times = pd.DatetimeIndex(sorted(issue_times))
    nowcasts["SmartPersistence"] = smart_persistence(
        truth[VARIABLE], clear_sky, dit, sorted_issue_times, dit["lead_time"].values
    ).to_dataset()
    LOG.info("Calculated smart persistence nowcasts")

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
        metrics = calc_error_metrics(nowcast[VARIABLE], truth[VARIABLE], clear_sky)
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
