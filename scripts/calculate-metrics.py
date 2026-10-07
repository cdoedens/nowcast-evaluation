import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from dask.distributed import Client
import sys, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/548/cd3022/repos/nowcast-evaluation/src")
import nowcast_preprocess
import logger


def preprocess_gso_of(ds):
    ds = nowcast_preprocess.datetime_to_lead_time(ds)
    return ds.sel(
        latitude=slice(-28.5, -35),
        longitude=slice(145, 151.5)
    )[["surface_global_irradiance"]]

def preprocess_truth(ds):
    return ds.sel(
        latitude=slice(-35, -28.5),
        longitude=slice(145, 151.5)
    )[["surface_global_irradiance"]]


def calc_error_metrics(
    fcst: xr.DataArray,
    obs: xr.DataArray,
) -> xr.DataArray:
    """Return RMSE with a single dimension: lead time."""
    # forecast_time must be a lead time (timedelta), not a valid time
    if not np.issubdtype(fcst["forecast_time"].dtype, np.timedelta64):
        raise TypeError(
            f"'{lead_dim}' should hold lead times (timedelta64). If it holds valid "
            f"times, convert it first: lead = valid_time - init_time."
        )

    # Put obs on exactly the forecast grid (guards against float coordinate mismatch)
    obs = obs.sel({"latitude": fcst["latitude"], "longitude": fcst["longitude"]}, method="nearest")
    obs = obs.assign_coords({"latitude": fcst["latitude"], "longitude": fcst["longitude"]})

    rmse_list = []
    mae_list = []
    for lead in fcst["forecast_time"].values:
        f = fcst.sel({"forecast_time": lead})
        valid_times = f["time"].values + lead
        # Observations at each forecast's valid time (NaN where obs are missing)
        o = obs.reindex({"time": valid_times})
        o = o.assign_coords({"time": f["time"].values})

        sq_err = (f - o) ** 2
        mse = sq_err.mean(dim=["latitude", "longitude"], skipna=True)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        mae = f - o
        mae_list.append(mae)

    rmse_array = xr.concat(rmse_list, dim=fcst["forecast_time"]).rename("rmse")
    mae_array = xr.concat(mae_list, dim=fcst["forecast_time"]).rename("mae")

    return xr.merge([rmse_array, mae_array])

    
def main(YEAR):
    LOG = logger.get_logger(__name__)
    daily_metrics = []
    for month in range(1, 13):
        MONTH = f"{month:02d}"
        LOG.info(f"Starting month: {MONTH}")
        
        for day in range(1, 32):
            DAY = f"{day:02d}"
            LOG.info(f"Opening data for {YEAR}-{MONTH}-{DAY}")
            gso_of_path = Path(f"/g/data/er8/nowcasts/nwc_20240701_20251231/{YEAR}/{MONTH}/{DAY}")
            gso_of_files = [f for f in gso_of_path.glob("*.nc")]
            if len(gso_of_files) == 0:
                continue
            gso_of = xr.open_mfdataset(
                gso_of_files,
                preprocess = preprocess_gso_of,
                concat_dim='time',
                combine='nested',
                data_vars='minimal',
                coords='minimal',
                compat='override',
                parallel=True,
                chunks='auto'
            )
            
            truth_path = Path(f"/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest/{YEAR}/{MONTH}/{DAY}")
            truth_files = [f for f in truth_path.glob("*.nc")]
            if len(truth_files) == 0:
                continue
            truth = xr.open_mfdataset(
                truth_files,
                preprocess = preprocess_truth,
                concat_dim='time',
                combine='nested',
                data_vars='minimal',
                coords='minimal',
                compat='override',
                parallel=True,
                chunks='auto'
            )

            LOG.info(f"Calculating error metrics for {YEAR}-{MONTH}-{DAY}")
            metrics = calc_error_metrics(
                gso_of["surface_global_irradiance"],
                truth["surface_global_irradiance"]
            )
            daily_metrics.append(metrics)

    LOG.info(f"Saving error metric dataset for {YEAR}")
    annual_metrics = daily_metrics.concat(daily_metrics, dim="time")
    annual_metrics.to_netcdf("/scratch/er8/cd3022/nowcast-evaluation/error-metric-data/gso-of_2025.nc")
            

if __name__ == "__main__":
    client = Client(
        n_workers=24,
        threads_per_worker=1
    )
    YEAR = "2025"
    main(YEAR)