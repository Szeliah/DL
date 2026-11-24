from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results_100_epochs")
IMG_DIR = Path("img")

EXPECTED_MODELS = [
    ("CNN", 3),
    ("CNN", 5),
    ("CNN", 7),
    ("ResNet", 3),
    ("ResNet", 5),
    ("ResNet", 7),
]

HISTORY_COLUMNS = {
    "model",
    "n",
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_error",
    "val_loss",
    "val_accuracy",
    "val_error",
}

SUMMARY_COLUMNS = {
    "model",
    "n",
    "test_loss",
    "test_accuracy",
    "test_error",
}


def load_results():
    """Load and validate the two experiment CSV files."""
    history_path = RESULTS_DIR / "history.csv"
    summary_path = RESULTS_DIR / "summary.csv"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Cannot find {history_path}. "
            "Run this script from the project directory containing results_100_epochs/."
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. "
            "The final test results file is missing."
        )

    history = pd.read_csv(history_path)
    summary = pd.read_csv(summary_path)

    missing_history = HISTORY_COLUMNS - set(history.columns)
    missing_summary = SUMMARY_COLUMNS - set(summary.columns)

    if missing_history:
        raise ValueError(
            f"history.csv is missing columns: {sorted(missing_history)}"
        )

    if missing_summary:
        raise ValueError(
            f"summary.csv is missing columns: {sorted(missing_summary)}"
        )

    # Convert columns to numeric and catch malformed CSV values.
    for column in HISTORY_COLUMNS - {"model"}:
        history[column] = pd.to_numeric(history[column], errors="raise")

    for column in SUMMARY_COLUMNS - {"model"}:
        summary[column] = pd.to_numeric(summary[column], errors="raise")

    history["model"] = history["model"].astype(str)
    summary["model"] = summary["model"].astype(str)

    history = history.sort_values(["model", "n", "epoch"])

    # Check that every experiment has the expected 100 epoch rows.
    print("\nHistory CSV:")
    print(f"  Rows: {len(history)}")

    for model_name, n in EXPECTED_MODELS:
        run = history[
            (history["model"] == model_name) &
            (history["n"] == n)
        ]
        if len(run) == 0:
            print(f"  WARNING: missing history for {model_name}, n={n}")
        elif len(run) != 100:
            print(
                f"  WARNING: {model_name}, n={n} has {len(run)} rows "
                "(expected 100)"
            )

    print("\nSummary CSV:")
    print(f"  Rows: {len(summary)}")

    return history, summary


def add_lr_markers(ax):
    """Mark the epochs after which the learning rate was reduced."""
    ax.axvline(
        x=50,
        linestyle="--",
        linewidth=1,
        label="LR reduced after epoch 50",
    )
    ax.axvline(
        x=75,
        linestyle=":",
        linewidth=1.5,
        label="LR reduced after epoch 75",
    )


def save_and_collect(fig, filename, figures):
    """Save a figure and retain it for the final plt.show()."""
    output_path = IMG_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    figures.append(fig)
    print(f"Saved: {output_path}")


def plot_depth_comparison(
    final_history,
    summary,
    metric,
    title,
    y_label,
    filename,
    figures,
    source="history",
):
    """Plot one final metric against the number of weighted layers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = final_history if source == "history" else summary.copy()
    data["depth"] = 6 * data["n"] + 2

    for model_name in ["CNN", "ResNet"]:
        model_data = data[data["model"] == model_name].sort_values("depth")

        if not model_data.empty:
            ax.plot(
                model_data["depth"],
                model_data[metric],
                marker="o",
                linewidth=2,
                label=model_name,
            )

    ax.set_title(title)
    ax.set_xlabel("Network depth (weighted layers)")
    ax.set_ylabel(y_label)
    ax.set_xticks([20, 32, 44])
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_and_collect(fig, filename, figures)


def plot_epoch_curves(history, metric, title, y_label, filename, figures):
    """Plot an epoch-by-epoch metric for all six models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, n in EXPECTED_MODELS:
        run = history[
            (history["model"] == model_name) &
            (history["n"] == n)
        ].sort_values("epoch")

        if run.empty:
            continue

        ax.plot(
            run["epoch"],
            run[metric],
            linewidth=1.7,
            label=f"{model_name}, n={n}",
        )

    add_lr_markers(ax)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    save_and_collect(fig, filename, figures)


def print_comparison_table(final_history, summary):
    """Print a compact table combining final train/validation and test scores."""
    final_history = (
        final_history[
            [
                "model", "n", "epoch",
                "train_accuracy", "train_error",
                "val_accuracy", "val_error",
            ]
        ]
        .rename(columns={"epoch": "last_epoch"})
    )

    merged = final_history.merge(
        summary[
            ["model", "n", "test_loss", "test_accuracy", "test_error"]
        ],
        on=["model", "n"],
        how="outer",
        validate="one_to_one",
    )

    merged["depth"] = 6 * merged["n"] + 2
    merged = merged.sort_values(["n", "model"])

    columns = [
        "model", "n", "depth",
        "train_accuracy", "val_accuracy",
        "test_accuracy", "test_error", "test_loss",
    ]

    print("\nFinal comparison (percentages except loss):")
    print(
        merged[columns].to_string(
            index=False,
            formatters={
                "train_accuracy": lambda x: f"{x:.2f}",
                "val_accuracy": lambda x: f"{x:.2f}",
                "test_accuracy": lambda x: f"{x:.2f}",
                "test_error": lambda x: f"{x:.2f}",
                "test_loss": lambda x: f"{x:.4f}",
            },
        )
    )

    return merged


def main(show_plots=True):
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    history, summary = load_results()

    # Last recorded training/validation measurements for each model.
    final_history = (
        history.sort_values("epoch")
        .groupby(["model", "n"], as_index=False)
        .tail(1)
        .copy()
    )

    figures = []

    # 1. Training error vs depth.
    plot_depth_comparison(
        final_history,
        summary,
        metric="train_error",
        title="Training error vs network depth",
        y_label="Training error (%)",
        filename="01_training_error_vs_depth.png",
        figures=figures,
        source="history",
    )

    # 2. Validation error vs depth.
    plot_depth_comparison(
        final_history,
        summary,
        metric="val_error",
        title="Validation error vs network depth",
        y_label="Validation error (%)",
        filename="02_validation_error_vs_depth.png",
        figures=figures,
        source="history",
    )

    # 3. Test error vs depth.
    plot_depth_comparison(
        final_history,
        summary,
        metric="test_error",
        title="Test error vs network depth",
        y_label="Test error (%)",
        filename="03_test_error_vs_depth.png",
        figures=figures,
        source="summary",
    )

    # 4. Training loss vs epoch.
    plot_epoch_curves(
        history,
        metric="train_loss",
        title="Training loss during training",
        y_label="Training loss",
        filename="04_training_loss_vs_epoch.png",
        figures=figures,
    )

    # 5. Validation loss vs epoch.
    plot_epoch_curves(
        history,
        metric="val_loss",
        title="Validation loss during training",
        y_label="Validation loss",
        filename="05_validation_loss_vs_epoch.png",
        figures=figures,
    )

    # 6. Training accuracy vs epoch.
    plot_epoch_curves(
        history,
        metric="train_accuracy",
        title="Training accuracy during training",
        y_label="Training accuracy (%)",
        filename="06_training_accuracy_vs_epoch.png",
        figures=figures,
    )

    # 7. Validation accuracy vs epoch.
    plot_epoch_curves(
        history,
        metric="val_accuracy",
        title="Validation accuracy during training",
        y_label="Validation accuracy (%)",
        filename="07_validation_accuracy_vs_epoch.png",
        figures=figures,
    )

    # 8. Test accuracy vs depth.
    plot_depth_comparison(
        final_history,
        summary,
        metric="test_accuracy",
        title="Test accuracy vs network depth",
        y_label="Test accuracy (%)",
        filename="08_test_accuracy_vs_depth.png",
        figures=figures,
        source="summary",
    )

    comparison = print_comparison_table(final_history, summary)
    comparison.to_csv(IMG_DIR / "comparison_metrics.csv", index=False)
    print(f"\nSaved comparison table: {IMG_DIR / 'comparison_metrics.csv'}")

    print(f"\nAll plots saved in: {IMG_DIR.resolve()}")

    if show_plots:
        plt.show()

    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CIFAR-10 CNN vs ResNet experiment results."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save PNG plots; do not open plot windows."
    )
    args = parser.parse_args()

    main(show_plots=not args.no_show)
