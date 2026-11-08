"""Calculate the RMSE of CPDiT ensemble-mean nowcasts as a function of the number of ensemble
members.

Every CPDiT forecast has 10 members. The error of an N-member ensemble is estimated by
drawing subsets of N of the 10 members (without replacement), taking the mean of each subset
as the forecast, and averaging the squared error over the subsets. Where there are at most
`--n-samples` distinct subsets (N = 1, 9, 10, and N = 2, 8 for the default of 50) all of
them are used, so those values are exact; otherwise `--n-samples` subsets are drawn at random.

Data loading, cropping, truth gap-filling and the clear-sky index (CSI) are taken from
calculate-metrics.py, so N = 10 reproduces the CPDiT RMSE in the error metrics. Errors are on
the CSI over pixels where clear-sky irradiance >= MIN_CLEAR_SKY, and a forecast is only scored
(`valid = 1`) when every member and the observed field at its valid time are complete.

Output: one NetCDF per day with `mse` of dims (time, lead_time, n_members), where `time` is the
issue time and `lead_time` is in minutes. `mse` is the mean over subsets of the pixel-mean
squared error, so pool over forecasts as RMSE = sqrt(mean(mse)).
"""

import argparse
import importlib.util
import sys
import warnings
from functools import partial
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

REPO = Path("/home/548/cd3022/repos/nowcast-evaluation")
sys.path.append(str(REPO / "src"))
import logger  # noqa: E402
import nowcast_preprocess  # noqa: E402
from run_days import run_days  # noqa: E402

# Reuse the loaders and constants of calculate-metrics.py (hyphenated name, so load by path)
_spec = importlib.util.spec_from_file_location("calc_metrics", REPO / "scripts/calculate-metrics.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

LOG = logger.get_logger(__name__)

VARIABLE = cm.VARIABLE
SPATIAL_DIMS = cm.SPATIAL_DIMS
ENSEMBLE_DIM = cm.ENSEMBLE_DIM

SAVE_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/ensemble-size-data")
DEFAULT_N_SAMPLES = 50
DEFAULT_SEED = 0


def member_subsets(m: int, n: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Subsets of n of the m members, as an (n_subsets, n) array of member indices: all of
    them if there are at most n_samples, otherwise n_samples drawn at random (a subset may be
    drawn more than once, which doesn't bias the mean)."""
    if comb(m, n) <= n_samples:
        return np.array(list(combinations(range(m), n)))
    return np.array([rng.choice(m, size=n, replace=False) for _ in range(n_samples)])


def subset_mse(err: np.ndarray, subsets_by_n: dict[int, np.ndarray]) -> np.ndarray:
    """Mean squared error of the subset-mean forecast, averaged over subsets, for each N.

    `err` is (members, pixels): each member's error over the pixels being scored. The error of
    a subset mean is the mean of its members' errors, so it is a weighted sum of rows of `err`.
    """
    m = err.shape[0]
    out = np.empty(len(subsets_by_n))
    for i, (n, subsets) in enumerate(subsets_by_n.items()):
        weights = np.zeros((len(subsets), m))
        np.put_along_axis(weights, subsets, 1.0 / n, axis=1)
        out[i] = np.mean((weights @ err) ** 2)  # mean over subsets and pixels
    return out


def calc_daily_mse(date: pd.Timestamp, n_samples: int, seed: int) -> xr.Dataset | None:
    """Per-forecast MSE vs number of members for one UTC day, or None if nothing to score."""
    day_dir = cm._day_dir(date)
    dit_files = sorted((cm.DIT_DIR / day_dir).glob("*.nc"))
    if not dit_files:
        LOG.warning(f"Skipping {date.date()}: no DiT files in {cm.DIT_DIR / day_dir}")
        return None
    issue_times = pd.DatetimeIndex(sorted({nowcast_preprocess.file_name_to_datetime64(f) for f in dit_files}))
    LOG.info(f"Calculating MSE vs ensemble size for {date.date()} ({len(dit_files)} DiT issue times)")

    dit = cm.open_files(dit_files, cm.preprocess_dit)
    dit["lead_time"] = dit["lead_time"].astype("timedelta64[ns]")
    # Same region as calculate-metrics.py: the DiT domain minus a buffer
    bounds = {
        "lat_min": dit["latitude"].min().item() + cm.DOMAIN_BUFFER,
        "lat_max": dit["latitude"].max().item() - cm.DOMAIN_BUFFER,
        "lon_min": dit["longitude"].min().item() + cm.DOMAIN_BUFFER,
        "lon_max": dit["longitude"].max().item() - cm.DOMAIN_BUFFER,
    }
    dit = cm._crop_to_region(dit, bounds)[VARIABLE]
    dit = dit.reindex(time=issue_times).transpose("time", "lead_time", ENSEMBLE_DIM, *SPATIAL_DIMS).load()
    leads = dit["lead_time"].values
    n_members_total = dit.sizes[ENSEMBLE_DIM]
    LOG.info(f"Loaded DiT nowcasts ({n_members_total} members, {len(leads)} lead times)")

    # Truth at the valid times, plus neighbours for filling single-step gaps
    valid_times = {t + pd.Timedelta(lead) for t in issue_times for lead in leads}
    load_times = valid_times | {t + s for t in valid_times for s in (-cm.TRUTH_STEP, cm.TRUTH_STEP)}
    truth_files = [
        f
        for d in (date, date + pd.Timedelta(days=1))
        for f in sorted((cm.TRUTH_DIR / cm._day_dir(d)).glob("*.nc"))
        if cm._truth_file_time(f) in load_times
    ]
    if not truth_files:
        LOG.warning(f"Skipping {date.date()}: no truth files for the DiT valid times")
        return None
    truth = cm.open_files(truth_files, partial(cm.preprocess_truth, bounds=bounds))[VARIABLE].load()
    truth = cm.fill_truth_gaps(truth, valid_times)
    truth = truth.sel(latitude=dit["latitude"], longitude=dit["longitude"], method="nearest")
    truth = truth.assign_coords(latitude=dit["latitude"], longitude=dit["longitude"])
    LOG.info(f"Loaded truth data ({len(truth_files)} files for {len(valid_times)} times)")

    clear_sky = cm.load_clear_sky(dit, issue_times.min(), issue_times.max() + leads.max())
    LOG.info("Loaded clear-sky irradiance")

    # Reproducible, but different subsets on different days
    rng = np.random.default_rng([seed, date.toordinal()])
    n_members = np.arange(1, n_members_total + 1)

    mse = np.full((len(issue_times), len(leads), len(n_members)), np.nan)
    valid = np.zeros((len(issue_times), len(leads)), dtype=np.int8)
    for j, lead in enumerate(leads):
        vt = issue_times + lead
        obs = truth.reindex(time=vt).values
        csk = clear_sky.interp(time=vt.values).values
        csk = np.where(csk >= cm.MIN_CLEAR_SKY, csk, np.nan)
        for i in range(len(issue_times)):
            f = dit.isel(time=i, lead_time=j).values  # (member, lat, lon)
            o = obs[i]
            if not (np.isfinite(f).all() and np.isfinite(o).all()):
                continue  # incomplete forecast or truth: not scored, as in the error metrics
            defined = np.isfinite(csk[i])
            if not defined.any():
                continue
            err = (f[:, defined] - o[defined]) / csk[i][defined]  # CSI error, (member, pixel)
            subsets = {n: member_subsets(n_members_total, n, n_samples, rng) for n in n_members}
            mse[i, j] = subset_mse(err, subsets)
            valid[i, j] = 1

    ds = xr.Dataset(
        {
            "mse": (("time", "lead_time", "n_members"), mse),
            "valid": (("time", "lead_time"), valid),
        },
        coords={
            "time": issue_times,
            "lead_time": leads / np.timedelta64(1, "m"),
            "n_members": n_members,
        },
    )
    ds["lead_time"].attrs["long_name"] = "lead time (minutes)"
    ds["n_subsets"] = ("n_members", [min(comb(n_members_total, n), n_samples) for n in n_members])
    ds.attrs = {
        "model": "CPDiT",
        "metric_variable": "clear-sky index (all-sky / clear-sky irradiance)",
        "clear_sky_min_W_m-2": cm.MIN_CLEAR_SKY,
        "n_samples": n_samples,
        "seed": seed,
        "note": (
            "time is the issue time. mse is the pixel-mean squared CSI error of the mean of N "
            "members, averaged over subsets of N of the members (all subsets if there are at "
            "most n_samples, otherwise n_samples random ones). NaN where valid == 0. Pool as "
            "RMSE = sqrt(mean(mse))."
        ),
    }
    LOG.info(f"Calculated MSE: {int(valid.sum())}/{valid.size} forecasts valid")
    return ds


def main(date: pd.Timestamp, n_samples: int, seed: int, save_dir: Path) -> None:
    ds = calc_daily_mse(date, n_samples, seed)
    if ds is None:
        LOG.info(f"No data for {date.date()}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(save_dir / f"{date.year}{date.month:02d}{date.day:02d}.nc")
    LOG.info(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=str, default="2025-01-01")
    parser.add_argument(
        "--end-date", type=str, default=None, help="if given, process every day from --date to this"
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="max subsets per ensemble size"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--n-workers", type=int, default=1, help="days processed in parallel")
    args = parser.parse_args()

    dates = pd.date_range(args.date, args.end_date or args.date, freq="D")
    failed = run_days(main, dates, args.n_workers, args.n_samples, args.seed, args.save_dir)
    sys.exit(1 if failed else 0)
