"""Plot CPDiT ensemble-mean RMSE against the number of ensemble members.

Reads the daily files written by calculate-ensemble-size.py and pools over all valid forecasts
as RMSE = sqrt(mean(mse)). Every ensemble size is scored on the same forecasts. Two panels:
the RMSE of the clear-sky index, and the RMSE relative to a single member.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

DATA_DIR = Path("/scratch/er8/cd3022/nowcast-evaluation/ensemble-size-data")
FIG_DIR = Path("/home/548/cd3022/repos/nowcast-evaluation/outputs/figures/ensemble-size")
DEFAULT_LEAD_TIMES = [60, 120, 180]  # minutes

# Fixed colour per lead time, in this order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def lead_label(minutes: float) -> str:
    return f"{int(minutes) // 60} h" if minutes % 60 == 0 else f"{int(minutes)} min"


def main(args):
    files = sorted(args.data_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No files in {args.data_dir}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", data_vars="minimal", decode_timedelta=False)
    ds = ds.sortby("time").sel(time=slice(args.start, args.end)).load()
    if len(args.lead_times) > len(COLORS):
        raise ValueError(f"At most {len(COLORS)} lead times per figure")

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.28})
    n_members = ds["n_members"].values
    for lead, color in zip(args.lead_times, COLORS):
        d = ds.sel(lead_time=lead)
        valid = d["valid"].astype(bool)
        n = int(valid.sum())
        rmse = np.sqrt(d["mse"].where(valid).mean("time")).values
        label = f"{lead_label(lead)} (n = {n})"
        style = dict(color=color, lw=2, marker="o", ms=5)
        ax.plot(n_members, rmse, label=label, **style)
        axr.plot(n_members, 100 * (rmse / rmse[0] - 1), label=label, **style)
        print(f"{lead_label(lead)}: RMSE {rmse[0]:.4f} (1 member) -> {rmse[-1]:.4f} "
              f"({n_members[-1]} members), {100 * (1 - rmse[-1] / rmse[0]):.1f}% lower")

    ax.set_ylabel("RMSE (clear-sky index)")
    ax.set_title("CPDiT ensemble-mean RMSE")
    axr.set_ylabel("Change in RMSE from 1 member (%)")
    axr.set_title("Relative to a single member")
    axr.axhline(0, color="#52514e", lw=1, zorder=0)
    for a in (ax, axr):
        a.set_xlabel("Number of ensemble members")
        a.set_xticks(n_members)
        a.grid(True, color="#e4e3df", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Lead time", frameon=False)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    leads = "_".join(f"{int(lt):03d}" for lt in args.lead_times)
    out = args.fig_dir / f"rmse_vs_ensemble_size_{leads}min.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-times", type=int, nargs="+", default=DEFAULT_LEAD_TIMES, help="minutes")
    parser.add_argument("--start", type=str, default=None, help="first issue date to include")
    parser.add_argument("--end", type=str, default=None, help="last issue date to include")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    main(parser.parse_args())
