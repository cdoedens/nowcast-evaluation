"""Calculate radially averaged power spectra of CPDiT and GSO-OF nowcasts and of the
Himawari observations at their valid times.

For every forecast (issue time x lead time) of one UTC day, the 2D power spectrum of the field
is computed and averaged in rings of constant wavenumber (the radially averaged power spectral
density, RAPSD). Data loading, cropping, truth gap-filling and the clear-sky index definition
are taken from calculate-metrics.py, so the spectra are of exactly the fields that were scored.

By default spectra are of the clear-sky index (CSI), consistent with the error metrics, so the
large-scale clear-sky gradient across the domain (sun angle) does not dominate the low
wavenumbers. Use `--variable irradiance` for spectra of the all-sky irradiance instead. A field
is only used (`valid = 1`) when it has no missing pixels; for CSI this also requires the sun to
be high enough over the whole domain (clear-sky irradiance >= MIN_CLEAR_SKY).

Before the FFT each field has its mean removed and is tapered with a 2D Tukey window, which
stops the discontinuity between opposite (non-periodic) edges leaking power into all
wavenumbers.

For CPDiT two spectra are saved: `psd`, the mean of the spectra of the individual members (the
spectrum of a single realisation, which is what should match the observations), and
`psd_ens_mean`, the spectrum of the ensemble mean (smoothed by averaging over members).

Output: one NetCDF per model per day with dims (time, lead_time, wavenumber), where `time` is
the issue time, `lead_time` is in minutes and `wavenumber` is in cycles per grid cell.
"""

import argparse
import importlib.util
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal.windows import tukey

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

SAVE_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/power-spectra-data")
DEFAULT_LEAD_TIMES = [60, 120, 180]  # minutes

# Fraction of each axis tapered by the Tukey window (0 = no taper, 1 = Hann window)
TUKEY_ALPHA = 0.25

EARTH_RADIUS_KM = 6371.0


# SPECTRA
def grid_spacing_km(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Grid spacing (dy, dx) in km at the centre of a regular lat/lon grid."""
    deg = np.pi / 180 * EARTH_RADIUS_KM
    dy = abs(np.diff(lat).mean()) * deg
    dx = abs(np.diff(lon).mean()) * deg * np.cos(np.deg2rad(lat.mean()))
    return dy, dx


class RadialSpectrum:
    """Radially averaged power spectral density for fields of a fixed shape (ny, nx).

    Wavenumbers are in cycles per grid cell, binned in rings of width 1 / min(ny, nx), from
    the lowest non-zero ring up to the Nyquist wavenumber 0.5. PSD is normalised so that the
    sum over all 2D Fourier modes equals the (windowed) field variance times the number of
    grid cells, i.e. it is comparable between models on the same grid.
    """

    def __init__(self, ny: int, nx: int, alpha: float = TUKEY_ALPHA):
        self.shape = (ny, nx)
        self.window = np.outer(tukey(ny, alpha), tukey(nx, alpha))
        self.norm = (self.window**2).sum()

        n = min(ny, nx)
        ky = np.fft.fftfreq(ny)
        kx = np.fft.fftfreq(nx)
        k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
        ring = np.rint(k * n).astype(int).ravel()
        self.n_bins = n // 2
        # Ring 0 is the mean (removed) and rings beyond n/2 are the corners beyond Nyquist
        self.keep = (ring >= 1) & (ring <= self.n_bins)
        self.ring = ring[self.keep] - 1
        self.counts = np.bincount(self.ring, minlength=self.n_bins)
        self.wavenumber = np.arange(1, self.n_bins + 1) / n

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """RAPSD of a 2D field; all NaN if the field has any missing pixel."""
        if not np.isfinite(field).all():
            return np.full(self.n_bins, np.nan)
        x = (field - field.mean()) * self.window
        power = np.abs(np.fft.fft2(x)) ** 2 / self.norm
        return np.bincount(self.ring, weights=power.ravel()[self.keep], minlength=self.n_bins) / self.counts


def rapsd(da: xr.DataArray, spectrum: RadialSpectrum) -> xr.DataArray:
    """RAPSD of every (latitude, longitude) field of `da`, looping over all other dims."""
    out = xr.apply_ufunc(
        spectrum,
        da,
        input_core_dims=[SPATIAL_DIMS],
        output_core_dims=[["wavenumber"]],
        vectorize=True,
        output_dtypes=[float],
    )
    return out.assign_coords(wavenumber=spectrum.wavenumber)


# DAILY PROCESSING
def to_grid(da: xr.DataArray, grid: xr.DataArray) -> xr.DataArray:
    """Put `da` on exactly the lat/lon grid of `grid` (guards against float mismatch)."""
    da = da.sel(latitude=grid["latitude"], longitude=grid["longitude"], method="nearest")
    return da.assign_coords(latitude=grid["latitude"], longitude=grid["longitude"])


def calc_daily_spectra(
    date: pd.Timestamp, lead_times_min: list[int], variable: str
) -> dict[str, xr.Dataset] | None:
    """Return {model name: spectra} for one UTC day, or None if there is nothing to do."""
    day_dir = cm._day_dir(date)
    dit_files = sorted((cm.DIT_DIR / day_dir).glob("*.nc"))
    if not dit_files:
        LOG.warning(f"Skipping {date.date()}: no DiT files in {cm.DIT_DIR / day_dir}")
        return None
    issue_times = set(pd.DatetimeIndex([nowcast_preprocess.file_name_to_datetime64(f) for f in dit_files]))
    sorted_issue_times = pd.DatetimeIndex(sorted(issue_times))
    leads = np.array([np.timedelta64(m, "m") for m in lead_times_min]).astype("timedelta64[ns]")
    LOG.info(f"Calculating power spectra for {date.date()} ({len(dit_files)} DiT issue times)")

    dit = cm.open_files(dit_files, cm.preprocess_dit)
    dit["lead_time"] = dit["lead_time"].astype("timedelta64[ns]")
    missing = set(leads) - set(dit["lead_time"].values)
    if missing:
        raise ValueError(f"Lead times {sorted(missing)} not in the DiT nowcasts")

    # Same region as calculate-metrics.py: the DiT domain minus a buffer
    bounds = {
        "lat_min": dit["latitude"].min().item() + cm.DOMAIN_BUFFER,
        "lat_max": dit["latitude"].max().item() - cm.DOMAIN_BUFFER,
        "lon_min": dit["longitude"].min().item() + cm.DOMAIN_BUFFER,
        "lon_max": dit["longitude"].max().item() - cm.DOMAIN_BUFFER,
    }
    dit = cm._crop_to_region(dit, bounds)[VARIABLE].sel(lead_time=leads).load()
    LOG.info("Loaded DiT nowcasts")

    nowcasts = {"CPDiT": dit}
    gso_of_files = cm._files_at_times(
        sorted((cm.GSO_OF_DIR / day_dir).glob("*.nc")), issue_times, "%Y%m%dT%H%M%S"
    )
    if gso_of_files:
        gso = cm.open_files(gso_of_files, partial(cm.preprocess_gso_of, bounds=bounds))[VARIABLE]
        gso["lead_time"] = gso["lead_time"].astype("timedelta64[ns]")
        gso = gso.reindex(lead_time=leads)  # NaN (so invalid) if a lead time is missing
        nowcasts["GSO_OF"] = to_grid(gso, dit).load()
        LOG.info(f"Loaded GSO-OF nowcasts ({len(gso_of_files)}/{len(issue_times)} issue times)")
    else:
        LOG.warning(f"No GSO-OF files at the DiT issue times on {date.date()}")

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
    truth = to_grid(cm.fill_truth_gaps(truth, valid_times), dit)
    # Observations at each forecast's valid time, with dims (time=issue time, lead_time, ...)
    obs = xr.concat(
        [
            truth.reindex(time=sorted_issue_times + lead).assign_coords(time=sorted_issue_times)
            for lead in leads
        ],
        dim=xr.DataArray(leads, dims="lead_time", name="lead_time"),
    ).transpose("time", "lead_time", *SPATIAL_DIMS)
    nowcasts["Observations"] = obs
    LOG.info(f"Loaded truth data ({len(truth_files)} files for {len(valid_times)} times)")

    if variable == "csi":
        clear_sky = cm.load_clear_sky(dit, min(issue_times), max(issue_times) + leads.max())
        csk = xr.concat(
            [
                clear_sky.interp(time=sorted_issue_times + lead).assign_coords(time=sorted_issue_times)
                for lead in leads
            ],
            dim=xr.DataArray(leads, dims="lead_time", name="lead_time"),
        )
        csk = csk.where(csk >= cm.MIN_CLEAR_SKY)
        LOG.info("Loaded clear-sky irradiance")

    ny, nx = dit.sizes["latitude"], dit.sizes["longitude"]
    spectrum = RadialSpectrum(ny, nx)
    dy, dx = grid_spacing_km(dit["latitude"].values, dit["longitude"].values)

    results = {}
    for name, da in nowcasts.items():
        da = da.reindex(time=sorted_issue_times)  # every model on the same issue times
        if variable == "csi":
            da = da / csk
        if ENSEMBLE_DIM in da.dims:
            psd_members = rapsd(da, spectrum)
            ds = xr.Dataset(
                {
                    # NaN if any member is incomplete, so every member is always included
                    "psd": psd_members.mean(ENSEMBLE_DIM, skipna=False),
                    "psd_ens_mean": rapsd(da.mean(ENSEMBLE_DIM, skipna=False), spectrum),
                }
            )
        else:
            ds = xr.Dataset({"psd": rapsd(da, spectrum)})
        ds["valid"] = ds["psd"].notnull().all("wavenumber").astype(np.int8)
        ds = ds.transpose("time", "lead_time", "wavenumber")
        ds = ds.assign_coords(lead_time=ds["lead_time"] / np.timedelta64(1, "m"))
        ds["lead_time"].attrs["long_name"] = "lead time (minutes)"
        ds["wavenumber"].attrs["units"] = "cycles per grid cell"
        ds.attrs = {
            "variable": "clear-sky index" if variable == "csi" else f"{VARIABLE} (W m-2)",
            "clear_sky_min_W_m-2": cm.MIN_CLEAR_SKY,
            "grid_shape": [ny, nx],
            "grid_spacing_km_y": dy,
            "grid_spacing_km_x": dx,
            "tukey_alpha": TUKEY_ALPHA,
            "note": (
                "time is the issue time. Radially averaged power spectral density of each field "
                "after removing its mean and applying a 2D Tukey window, normalised by the sum of "
                "the squared window. Wavelength = grid spacing / wavenumber. NaN where valid == 0 "
                "(field with missing pixels). For ensembles, psd is the mean of the member "
                "spectra and psd_ens_mean the spectrum of the ensemble mean."
            ),
        }
        results[name] = ds
        LOG.info(f"Calculated {name} spectra: {int(ds['valid'].sum())}/{ds['valid'].size} fields valid")

    return results


def main(date: pd.Timestamp, lead_times: list[int], variable: str, save_dir: Path) -> None:
    spectra = calc_daily_spectra(date, lead_times, variable)
    if not spectra:
        LOG.info(f"No data for {date.date()}")
        return
    for name, ds in spectra.items():
        out_dir = save_dir / variable / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_dir / f"{date.year}{date.month:02d}{date.day:02d}.nc")
    LOG.info(f"Saved power spectra to {save_dir / variable}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=str, default="2025-01-01")
    parser.add_argument(
        "--end-date", type=str, default=None, help="if given, process every day from --date to this"
    )
    parser.add_argument("--lead-times", type=int, nargs="+", default=DEFAULT_LEAD_TIMES, help="minutes")
    parser.add_argument("--variable", choices=["csi", "irradiance"], default="csi")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--n-workers", type=int, default=1, help="days processed in parallel")
    args = parser.parse_args()

    dates = pd.date_range(args.date, args.end_date or args.date, freq="D")
    failed = run_days(main, dates, args.n_workers, args.lead_times, args.variable, args.save_dir)
    sys.exit(1 if failed else 0)
