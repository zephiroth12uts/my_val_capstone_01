"""Household contents-value regression: figures.

The figures are drawn from the files that the notebooks and the training module save, so no
model is fitted here. They repeat the main charts of notebooks 03_ky, 04_ky and 05_ky.
"""

from pathlib import Path

from loguru import logger
import matplotlib
from sklearn.metrics import mean_absolute_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from my_val_regression.config import (
    FIGURES_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)

app = typer.Typer()

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def plot_tree_curves(reports_dir: Path = REPORTS_DIR, models_dir: Path = MODELS_DIR):
    """Validation MAE against the number of trees for the two tuned models."""
    hyperparameters = pd.read_csv(models_dir / "final_hyperparameters.csv").set_index("model")
    panels = [
        ("XGBoost tuned", "tree_curve_xgboost.csv", "xgboost_log_tuned"),
        ("LightGBM tuned", "tree_curve_lightgbm.csv", "lightgbm_log_tuned"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (title, file_name, model) in zip(axes, panels):
        curve = pd.read_csv(reports_dir / file_name)
        used = int(hyperparameters.loc[model, "n_estimators"])
        searched = int(hyperparameters.loc[model, "searched_max_trees"])
        mean, sd = curve["val_MAE_mean"], curve["val_MAE_sd"]
        ax.plot(curve["trees"], mean)
        ax.fill_between(curve["trees"], mean - sd, mean + sd, alpha=0.2)
        ax.axvline(used, color="C3", label=f"used: {used} trees")
        ax.axvline(searched, color="grey", ls=":", label=f"searched maximum: {searched}")
        ax.set_title(title)
        ax.set_xlabel("number of trees")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("validation MAE ($)")
    fig.tight_layout()
    return fig


def plot_blend_weight_curve(validation_predictions: pd.DataFrame):
    """Validation MAE of the blend for every weight of the XGBoost share."""
    val = validation_predictions
    weights = np.round(np.arange(0, 1.0001, 0.02), 2)
    mae = [
        mean_absolute_error(
            val["true_total_value"], w * val["xgb_tuned_pred"] + (1 - w) * val["lgbm_tuned_pred"]
        )
        for w in weights
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(weights, mae, marker="o", ms=3, label="validation")
    ax.axvline(weights[int(np.argmin(mae))], color="grey", ls="--", lw=1)
    ax.set_xlabel("w (XGBoost share)")
    ax.set_ylabel("validation MAE ($)")
    ax.set_title("Blend weight on the validation set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_success_rate_by_n(results_path: Path):
    """Row-level success rate of the 70% interval per basket size for the three sets."""
    table = pd.read_csv(results_path, header=[0, 1], index_col=0)
    rates = table["success rate (%)"].drop(index=["ALL"], errors="ignore")
    rates = rates[[c for c in ("train", "validation", "test") if c in rates.columns]]
    labels = {"train": "train (in-sample)", "validation": "validation (in-sample)", "test": "test"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(rates))
    width = 0.27
    for j, column in enumerate(rates.columns):
        ax.bar(x + (j - 1) * width, rates[column], width, label=labels[column])
    ax.axhline(70, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={i}" for i in rates.index])
    ax.set_ylabel("success rate (%)")
    ax.set_title("70% interval: train, validation (in-sample) and test, row level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


@app.command()
def main(
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    interim_dir: Path = INTERIM_DIR,
    figures_dir: Path = FIGURES_DIR,
):
    """Save the three regression figures as PNG files."""
    logger.info("Generating the regression figures...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_tree_curves(reports_dir, models_dir).savefig(
        figures_dir / "validation_tree_curves.png", dpi=150
    )
    predictions = pd.read_parquet(interim_dir / "tuned_model_predictions.parquet")
    plot_blend_weight_curve(predictions).savefig(figures_dir / "blend_weight_curve.png", dpi=150)
    results_path = reports_dir / "interval_results_train_validation_test.csv"
    plot_success_rate_by_n(results_path).savefig(figures_dir / "success_rate_by_n.png", dpi=150)
    logger.success(f"Three figures saved to {figures_dir}.")


if __name__ == "__main__":
    app()
