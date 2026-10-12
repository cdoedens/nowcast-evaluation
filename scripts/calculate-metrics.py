"""Calculate error metrics for solar irradiance nowcasts against satellite-derived truth.

For every day of the requested year, the DiT and GSO optical-flow (GSO-OF) nowcasts are
compared with the Himawari truth product. Metrics are computed per lead time, concatenated
along a `date` dimension and saved as one NetCDF file per model.
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
from scores.spatial import fss_2d
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore")

sys.path.append("/home/548/cd3022/repos/nowcast-evaluation/src")
import logger  # noqa: E402
import nowcast_preprocess  # noqa: E402

LOG = logger.get_logger(__name__)

VARIABLE = "surface_global_irradiance"

DIT_DIR = Path("/scratch/er8/cd3022/CPDiT/v0.1.0/inference")
GSO_OF_DIR = Path("/g/data/er8/nowcasts/nwc_20240701_20251231")
TRUTH_DIR = Path("/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest")
SAVE_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/error-metric-data")

SPATIAL_DIMS = ["latitude", "longitude"]


# DATA LOADING
def _region_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Slice from lo to hi, respecting whether the coordinate is ascending or descending."""
    ascending = coord.values[0] <= coord.values[-1]
    return slice(lo, hi) if ascending else slice(hi, lo)


def _crop_to_region(ds: xr.Dataset, bounds: dict) -> xr.Dataset:
    return ds.sel(
        latitude=_region_slice(ds["latitude"], bounds["lat_min"], bounds["lat_max"]),
        longitude=_region_slice(ds["longitude"], bounds["lon_min"], bounds["lon_max"]),
    )[[VARIABLE]]


def preprocess_dit(ds: xr.Dataset) -> xr.Dataset:
    ds = nowcast_preprocess.preprocess_nowcast_dataset(
        ds,
        f_dim_name="lead_time",
        new_f_dim_name="lead_time",
    )
    return ds[[VARIABLE]].mean(dim="member")


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


# METRIC CALCULATION
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
    """Return RMSE, MAE, FSS (small/large window) and SSIM, each as a function of lead time.

    `fcst` has dims (time, lead_time, latitude, longitude), where `time` is the issue time.
    `obs` has dims (time, latitude, longitude), where `time` is the valid time.
    RMSE and SSIM are calculated per forecast then averaged over issue times; MAE is
    averaged over all times and grid points.
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

        err = f - o
        fss_kwargs = {
            "event_threshold": fss_threshold,
            "spatial_dims": SPATIAL_DIMS,
        }
        per_lead.append(
            xr.Dataset(
                {
                    "rmse": np.sqrt((err**2).mean(dim=SPATIAL_DIMS)).mean(dim="time"),
                    "mae": np.abs(err).mean(dim=[*SPATIAL_DIMS, "time"]),
                    "fss_small": fss_2d(f, o, window_size=fss_window_small, **fss_kwargs),
                    "fss_large": fss_2d(f, o, window_size=fss_window_large, **fss_kwargs),
                    "ssim": ssim_field(
                        f, o, data_range=ssim_data_range, sigma=ssim_sigma
                    ).mean(dim="time"),
                }
            )
        )

    return xr.concat(per_lead, dim=fcst["lead_time"])


# DAILY PROCESSING
def calc_daily_metrics(date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset] | None:
    """Return (GSO-OF, DiT) metrics for one day, or None if any input data is missing."""
    day_dir = Path(f"{date.year}/{date.month:02d}/{date.day:02d}")

    dit_files = sorted((DIT_DIR / day_dir).glob("*.nc"))
    gso_of_files = sorted((GSO_OF_DIR / day_dir).rglob("*.nc"))
    truth_files = sorted((TRUTH_DIR / day_dir).rglob("*.nc"))
    if not (dit_files and gso_of_files and truth_files):
        LOG.info(f"Skipping {date.date()}: missing data")
        return None

    LOG.info(f"Calculating metrics for {date.date()}")

    dit = open_files(dit_files, preprocess_dit)
    LOG.info("Loaded DiT nowcasts")

    # Use the DiT domain as the region for the other datasets
    bounds = {
        "lat_min": dit["latitude"].min().item(),
        "lat_max": dit["latitude"].max().item(),
        "lon_min": dit["longitude"].min().item(),
        "lon_max": dit["longitude"].max().item(),
    }

    gso_of = open_files(gso_of_files, partial(preprocess_gso_of, bounds=bounds))
    LOG.info("Loaded GSO-OF nowcasts")
    truth = open_files(truth_files, partial(preprocess_truth, bounds=bounds))
    LOG.info("Loaded truth data")

    results = []
    for nowcast in (gso_of, dit):
        metrics = calc_error_metrics(nowcast[VARIABLE], truth[VARIABLE])
        # Lead time in minutes, and the day as a dimension for concatenating days
        metrics = metrics.assign_coords(lead_time=metrics["lead_time"] / np.timedelta64(1, "m"))
        results.append(metrics.expand_dims(date=[date]))
    LOG.info("Calculated metrics")

    return tuple(results)


def main(year: int) -> None:
    daily_of, daily_dit = [], []
    for date in pd.date_range(f"{year}-01-01", f"{year}-12-31"):
        daily = calc_daily_metrics(date)
        if daily is None:
            continue
        daily_of.append(daily[0])
        daily_dit.append(daily[1])

    if not daily_of:
        LOG.error(f"No days with complete data found for {year}")
        return

    # JOIN DAILY DATA INTO COMBINED DATASETS
    of_metrics = xr.concat(daily_of, dim="date")
    dit_metrics = xr.concat(daily_dit, dim="date")
    LOG.info(f"Concatenated daily metrics into full datasets for {year}")

    # SAVE METRICS AS NETCDF
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving GSO-OF metrics...")
    of_metrics.to_netcdf(SAVE_DIR / f"gso-of_metrics_{year}.nc")
    LOG.info("Saving DiT metrics...")
    dit_metrics.to_netcdf(SAVE_DIR / f"dit_metrics_{year}.nc")
    LOG.info("Saved metrics to netcdf files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--n-workers", type=int, default=24)
    args = parser.parse_args()

    with Client(n_workers=args.n_workers, threads_per_worker=1):
        main(args.year)
