"""Measure 90% interval coverage under random and time-ordered splits.

Reads the predictions saved by colab_predict.ipynb from results/predictions/ and
writes results/summary.json.

Usage:
    python src/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import (  # noqa: E402
    absolute_residual_scores,
    aci_thresholds,
    block_bootstrap_interval,
    covered,
    cqr_scores,
    fixed_thresholds,
    rolling_thresholds,
    worst_block_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.10
GAMMA = 0.005
REFRESH = 168  # recalibrate every 168 rows, one week of hourly data
MONTH = 720  # rows in a 30-day block of hourly data
DATASETS = ("bike", "metro", "pm25")
MODELS = ("gbdt", "tabpfn")


def score_pair(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "absolute":
        return absolute_residual_scores(frame["y"].to_numpy(), frame["q50"].to_numpy())
    return cqr_scores(frame["y"].to_numpy(), frame["q05"].to_numpy(), frame["q95"].to_numpy())


def interval_widths(frame: pd.DataFrame, thresholds: np.ndarray, kind: str) -> np.ndarray:
    if kind == "absolute":
        return 2 * thresholds
    return (frame["q95"].to_numpy() - frame["q05"].to_numpy()) + 2 * thresholds


def evaluate(frame: pd.DataFrame, thresholds: np.ndarray, kind: str, scale: float) -> dict:
    hits = covered(score_pair(frame, kind), thresholds)
    low, high = block_bootstrap_interval(hits, MONTH)
    return {
        "coverage": float(hits.mean()),
        "coverage_low": low,
        "coverage_high": high,
        "worst_month": worst_block_coverage(hits, MONTH),
        "width": float(np.maximum(interval_widths(frame, thresholds, kind), 0).mean() / scale),
    }


def dataset_scale(path: Path) -> float:
    """IQR of the target in the time-ordered calibration block, shared by all runs of a dataset."""
    frame = pd.read_csv(path)
    return float(np.subtract(*np.percentile(frame.loc[frame["role"] == "calibration", "y"], [75, 25])))


def analyze_prediction_file(path: Path, shifted: bool, scale: float) -> dict:
    frame = pd.read_csv(path).sort_values("position")
    calibration = frame[frame["role"] == "calibration"]
    test = frame[frame["role"] == "test"]
    y_test = test["y"].to_numpy()
    result = {
        "n_calibration": len(calibration),
        "n_test": len(test),
        "mae_over_iqr": float(np.abs(y_test - test["q50"].to_numpy()).mean() / scale),
        "raw_band_coverage": float(((y_test >= test["q05"]) & (y_test <= test["q95"])).mean()),
    }
    for kind in ("absolute", "cqr"):
        cal_scores, test_scores = score_pair(calibration, kind), score_pair(test, kind)
        methods = {"fixed": fixed_thresholds(cal_scores, len(test), ALPHA)}
        if shifted:
            window = len(calibration)
            methods["rolling"] = rolling_thresholds(cal_scores, test_scores, ALPHA, window, REFRESH)
            methods["aci"] = aci_thresholds(cal_scores, test_scores, ALPHA, GAMMA)
        result[kind] = {name: evaluate(test, values, kind, scale) for name, values in methods.items()}
    return result


def main() -> None:
    predictions = ROOT / "results" / "predictions"
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model in MODELS:
            random_file = predictions / f"{dataset}__random__{model}.csv.gz"
            temporal_file = predictions / f"{dataset}__temporal__{model}.csv.gz"
            if not (random_file.exists() and temporal_file.exists()):
                print(f"missing predictions for {dataset} {model}")
                continue
            scale = dataset_scale(predictions / f"{dataset}__temporal__{MODELS[0]}.csv.gz")
            summary[dataset][model] = {
                "random": analyze_prediction_file(random_file, shifted=False, scale=scale),
                "temporal": analyze_prediction_file(temporal_file, shifted=True, scale=scale),
            }

    print("coverage (target 0.90) | worst 30-day block | width / calibration IQR")
    header = f"{'data':<7}{'model':<10}{'score':<10}{'method':<18}{'coverage':>9}{'95% interval':>17}{'worst':>7}{'width':>7}"
    print(header)
    for dataset, models in summary.items():
        for model, splits in models.items():
            for kind in ("absolute", "cqr"):
                rows = [("random fixed", splits["random"][kind]["fixed"])]
                rows += [(f"temporal {name}", splits["temporal"][kind][name]) for name in ("fixed", "rolling", "aci")]
                for name, row in rows:
                    span = f"{row['coverage_low']:.3f} to {row['coverage_high']:.3f}"
                    print(
                        f"{dataset:<7}{model:<10}{kind:<10}{name:<18}{row['coverage']:>9.3f}{span:>17}"
                        f"{row['worst_month']:>7.3f}{row['width']:>7.2f}"
                    )
    print("\nerror and raw model band")
    for dataset, models in summary.items():
        for model, splits in models.items():
            for split in ("random", "temporal"):
                row = splits[split]
                print(
                    f"{dataset:<7}{model:<10}{split:<10}MAE/IQR {row['mae_over_iqr']:.3f}"
                    f" | raw 5-95 band coverage {row['raw_band_coverage']:.3f}"
                )
    target = ROOT / "results" / "summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
