"""Plot the mean radially averaged power spectra of CPDiT and GSO-OF nowcasts against the
observations, one figure per lead time.

Reads the daily files written by calculate-power-spectra.py. Only forecasts that are valid for
every model (CPDiT, GSO-OF and the observations at the valid time) are averaged, so all
spectra are over exactly the same set of cases.

Each figure has two panels sharing the wavelength axis: the mean PSD (log-log), and the ratio
of each forecast PSD to the observed PSD (1 = the forecast has the observed variance at that
scale; < 1 = too smooth; > 1 = too much small-scale noise).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

DATA_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/power-spectra-data")
FIG_DIR = Path("/home/548/cd3022/repos/nowcast-evaluation/outputs/figures/power-spectra")
MODELS = ["CPDiT", "GSO_OF"]
OBS = "Observations"
DEFAULT_LEAD_TIMES = [60, 120, 180]  # minutes

# Fixed colour per model (identity), observations in neutral ink
STYLES = {
    OBS: dict(color="#0b0b0b", lw=2.0, label="Observations (Himawari)"),
    "CPDiT": dict(color="#2a78d6", lw=2.0, label="CPDiT (mean of member spectra)"),
    "CPDiT_ens_mean": dict(color="#2a78d6", lw=2.0, ls="--", label="CPDiT ensemble mean"),
    "GSO_OF": dict(color="#eb6834", lw=2.0, label="GSO-OF"),
}


def load(model: str, data_dir: Path, start: str | None, end: str | None) -> xr.Dataset:
    files = sorted((data_dir / model).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No power spectra files in {data_dir / model}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", decode_timedelta=False).sortby("time").load()
    return ds.sel(time=slice(start, end))


def plot_lead_time(spectra: dict[str, xr.DataArray], n_cases: int, lead: int, attrs: dict, out: Path):
    dx = np.sqrt(attrs["grid_spacing_km_x"] * attrs["grid_spacing_km_y"])
    obs = spectra[OBS]
    wavelength = dx / obs["wavenumber"].values

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )
    for name, psd in spectra.items():
        ax.loglog(wavelength, psd.values, **STYLES[name])
        if name != OBS:
            axr.semilogx(wavelength, (psd / obs).values, **STYLES[name])

    axr.axhline(1.0, color="#52514e", lw=1, zorder=0)
    axr.set_yscale("log")
    axr.set_ylabel("Forecast / observed PSD")
    axr.set_xlabel("Wavelength (km)")
    ax.set_ylabel(f"PSD ({attrs['variable']})")
    # Large scales on the left, small scales on the right
    ax.set_xlim(wavelength.max(), wavelength.min())
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500) if wavelength.min() <= t <= wavelength.max()]
    axr.set_xticks(ticks, [str(t) for t in ticks])
    axr.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lead_label = f"{lead // 60} h" if lead % 60 == 0 else f"{lead} min"
    ax.set_title(f"Radially averaged power spectrum, {lead_label} lead time (n = {n_cases} forecasts)")
    ax.legend(frameon=False)
    for a in (ax, axr):
        a.grid(True, which="major", color="#e4e3df", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main(args):
    data_dir = args.data_dir / args.variable
    ds = {m: load(m, data_dir, args.start, args.end) for m in [*MODELS, OBS]}

    # Cases where every model and the observations are valid
    valid = None
    for d in ds.values():
        v = d["valid"].astype(bool)
        valid = v if valid is None else (valid & v).fillna(False)

    for lead in args.lead_times:
        mask = valid.sel(lead_time=lead)
        n = int(mask.sum())
        if n == 0:
            print(f"No forecasts valid for every model at lead time {lead} min, skipping")
            continue
        spectra = {OBS: ds[OBS]["psd"].sel(lead_time=lead).where(mask).mean("time")}
        for m in MODELS:
            spectra[m] = ds[m]["psd"].sel(lead_time=lead).where(mask).mean("time")
            if m == "CPDiT" and args.ens_mean and "psd_ens_mean" in ds[m]:
                spectra["CPDiT_ens_mean"] = ds[m]["psd_ens_mean"].sel(lead_time=lead).where(mask).mean("time")
        out = args.fig_dir / args.variable / f"power_spectra_{lead:03d}min.png"
        plot_lead_time(spectra, n, lead, ds[OBS].attrs, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", choices=["csi", "irradiance"], default="csi")
    parser.add_argument("--lead-times", type=int, nargs="+", default=DEFAULT_LEAD_TIMES, help="minutes")
    parser.add_argument("--start", type=str, default=None, help="first issue date to include")
    parser.add_argument("--end", type=str, default=None, help="last issue date to include")
    parser.add_argument("--no-ens-mean", dest="ens_mean", action="store_false",
                        help="don't plot the spectrum of the CPDiT ensemble mean")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    main(parser.parse_args())
