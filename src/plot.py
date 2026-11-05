"""Plot coverage in 30-day blocks for the fixed threshold and ACI.

Usage:
    python src/plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ALPHA, GAMMA, MONTH, ROOT, score_pair  # noqa: E402
from conformal import aci_thresholds, covered, fixed_thresholds  # noqa: E402

PANELS = {"bike": "Bike sharing", "metro": "Metro traffic", "pm25": "Beijing PM2.5"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
FIXED, ACI = "#2a78d6", "#eb6834"


def block_series(times: pd.Series, hits: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    blocks = len(hits) // MONTH
    values = hits[: blocks * MONTH].reshape(blocks, MONTH).mean(axis=1)
    middle = times.iloc[MONTH // 2 :: MONTH].iloc[:blocks]
    return middle, values


def main() -> None:
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for axis, (dataset, title) in zip(axes, PANELS.items()):
        frame = pd.read_csv(ROOT / "results" / "predictions" / f"{dataset}__temporal__tabpfn.csv.gz").sort_values("position")
        calibration, test = frame[frame["role"] == "calibration"], frame[frame["role"] == "test"]
        cal_scores, test_scores = score_pair(calibration, "cqr"), score_pair(test, "cqr")
        times = pd.to_datetime(test["time"])
        series = {
            "Fixed threshold": (fixed_thresholds(cal_scores, len(test), ALPHA), FIXED, "o"),
            "ACI": (aci_thresholds(cal_scores, test_scores, ALPHA, GAMMA), ACI, "s"),
        }
        for label, (thresholds, color, marker) in series.items():
            x, y = block_series(times, covered(test_scores, thresholds))
            axis.plot(x, y, color=color, linewidth=2, marker=marker, markersize=5, label=label)
        axis.axhline(1 - ALPHA, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="90% target")
        axis.set_facecolor(SURFACE)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(0.6, 1.02)
        axis.yaxis.grid(True, color=GRID, linewidth=1)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(MUTED)
        axis.tick_params(colors=MUTED, length=0)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4 if dataset == "bike" else 8))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].set_ylabel("Coverage in each 30-day block")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.995, 1.0))
    figure.suptitle("TabPFN-2 with CQR, time-ordered split", x=0.012, ha="left", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    target = ROOT / "results" / "coverage_over_time.png"
    figure.savefig(target, dpi=170, facecolor=SURFACE)
    print(f"saved {target}")


if __name__ == "__main__":
    main()
