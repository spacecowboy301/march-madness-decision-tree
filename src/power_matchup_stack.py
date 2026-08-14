from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .analyze_factor_importance import (
    RANDOM_STATE,
    clean_label,
    feature_role,
    fit_temporally_calibrated,
    grouped_model_importance,
    metric_dict,
    numeric_pipeline,
    temporal_oof_probabilities,
    tune_candidates,
)
from .factor_matchups import build_matchup_dataset


REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
MODEL_PATH = Path("models/power_matchup_stack.joblib")
METRICS_PATH = REPORTS / "power_matchup_metrics.json"
COMPARISON_PATH = REPORTS / "power_matchup_model_comparison.csv"
PREDICTIONS_PATH = REPORTS / "power_matchup_validation_predictions.csv"
CONDITIONAL_IMPORTANCE_PATH = REPORTS / "conditional_factor_importance.csv"
CHANNEL_IMPORTANCE_PATH = REPORTS / "channel_importance.csv"
CONDITIONAL_MECHANISM_PATH = REPORTS / "conditional_matchup_mechanism_importance.csv"
SUMMARY_PATH = REPORTS / "power_matchup_summary.md"
COMPARISON_CHART_PATH = FIGURES / "power_matchup_model_comparison.png"
CONDITIONAL_CHART_PATH = FIGURES / "conditional_factor_importance.png"
CONDITIONAL_MECHANISM_CHART_PATH = FIGURES / "conditional_matchup_mechanism_importance.png"

POWER_FEATURE = "regular_season_power_gap"


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


@dataclass
class TwoChannelStackedModel:
    factor_model: object
    power_model: object
    stacker: LogisticRegression

    def channel_probabilities(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        factor_probability = self.factor_model.predict_proba(x)[:, 1]
        power_probability = self.power_model.predict_proba(x)[:, 1]
        return factor_probability, power_probability

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        factor_probability, power_probability = self.channel_probabilities(x)
        channels = np.column_stack([logit(power_probability), logit(factor_probability)])
        return self.stacker.predict_proba(channels)

    def predict_with_neutral_channel(self, x: pd.DataFrame, neutral: str) -> np.ndarray:
        factor_probability, power_probability = self.channel_probabilities(x)
        power_logit = logit(power_probability)
        factor_logit = logit(factor_probability)
        if neutral == "power":
            power_logit = np.zeros_like(power_logit)
        elif neutral == "factor":
            factor_logit = np.zeros_like(factor_logit)
        else:
            raise ValueError(f"Unknown channel: {neutral}")
        return self.stacker.predict_proba(np.column_stack([power_logit, factor_logit]))[:, 1]


def fit_stack(
    factor_candidate,
    power_candidate,
    x: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> TwoChannelStackedModel:
    factor_oof, factor_labels = temporal_oof_probabilities(factor_candidate, x, y, seasons)
    power_oof, power_labels = temporal_oof_probabilities(power_candidate, x, y, seasons)
    if not np.array_equal(factor_labels, power_labels):
        raise ValueError("Factor and power temporal folds are not aligned.")
    stacker = LogisticRegression(C=0.25, max_iter=3000, random_state=RANDOM_STATE)
    stacker.fit(np.column_stack([logit(power_oof), logit(factor_oof)]), factor_labels)
    factor_model = fit_temporally_calibrated(factor_candidate, x, y, seasons)
    power_model = fit_temporally_calibrated(power_candidate, x, y, seasons)
    return TwoChannelStackedModel(factor_model, power_model, stacker)


def plot_comparison(comparison: pd.DataFrame) -> None:
    ordered = comparison.sort_values("log_loss", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].barh(ordered["model"].map(clean_label), ordered["accuracy"], color="#287271")
    axes[0].set_xlim(0.5, max(0.78, ordered["accuracy"].max() + 0.03))
    axes[0].set_xlabel("Held-out accuracy")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(ordered["model"].map(clean_label), ordered["log_loss"], color="#d99b45")
    axes[1].set_xlabel("Held-out log loss (lower is better)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Separated Power Baseline And Matchup Signal")
    fig.tight_layout()
    fig.savefig(COMPARISON_CHART_PATH, dpi=190)
    plt.close(fig)


def plot_conditional_importance(importance: pd.DataFrame) -> None:
    plotted = importance.sort_values("importance_mean").tail(14)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plotted["factor_group"].map(clean_label), plotted["importance_mean"], color="#4f6d7a")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Factor Importance Conditional On Regular-Season Power")
    ax.set_xlabel("Increase in held-out log loss when factor family is shuffled")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CONDITIONAL_CHART_PATH, dpi=190)
    plt.close(fig)


def plot_conditional_mechanisms(importance: pd.DataFrame) -> None:
    plotted = importance.sort_values("importance_mean").tail(12)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(plotted["factor_group"].map(clean_label), plotted["importance_mean"], color="#d99b45")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Matchup Mechanisms Conditional On Regular-Season Power")
    ax.set_xlabel("Increase in held-out log loss when mechanism is shuffled")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CONDITIONAL_MECHANISM_CHART_PATH, dpi=190)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    factor_x, y, meta, feature_families = build_matchup_dataset()
    observed = factor_x.notna().any(axis=0)
    factor_x = factor_x.loc[:, observed]
    feature_families = {feature: feature_families[feature] for feature in factor_x.columns}
    full_x = factor_x.copy()
    full_x[POWER_FEATURE] = meta[POWER_FEATURE].to_numpy()
    train = meta["season"] <= 2016
    valid = meta["season"].between(2017, 2025)

    factor_options = {
        "regularized_logistic": [
            (
                f"C={c}",
                numeric_pipeline(
                    LogisticRegression(C=c, max_iter=4000, random_state=RANDOM_STATE),
                    list(factor_x.columns),
                    scale=True,
                ),
            )
            for c in [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
        ]
    }
    factor_candidates, factor_tuning = tune_candidates(
        factor_options,
        full_x.loc[train],
        y.loc[train],
        meta.loc[train, "season"],
    )
    factor_candidate = factor_candidates["regularized_logistic"]

    power_options = {
        "regular_season_power": [
            (
                f"C={c}",
                numeric_pipeline(
                    LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE),
                    [POWER_FEATURE],
                    scale=True,
                ),
            )
            for c in [0.02, 0.05, 0.1, 0.25, 1.0, 10.0]
        ]
    }
    power_candidates, power_tuning = tune_candidates(
        power_options,
        full_x.loc[train],
        y.loc[train],
        meta.loc[train, "season"],
    )
    power_candidate = power_candidates["regular_season_power"]
    model = fit_stack(
        factor_candidate,
        power_candidate,
        full_x.loc[train],
        y.loc[train],
        meta.loc[train, "season"],
    )

    valid_x = full_x.loc[valid].reset_index(drop=True)
    valid_y = y.loc[valid].reset_index(drop=True)
    valid_meta = meta.loc[valid].reset_index(drop=True)
    factor_probability, power_probability = model.channel_probabilities(valid_x)
    stacked_probability = model.predict_proba(valid_x)[:, 1]
    comparison = pd.DataFrame(
        [
            {"model": "factor_only", **metric_dict(valid_y, factor_probability)},
            {"model": "regular_season_power_only", **metric_dict(valid_y, power_probability)},
            {"model": "two_channel_stack", **metric_dict(valid_y, stacked_probability)},
        ]
    ).sort_values("log_loss")

    predictions = valid_meta.copy()
    predictions["factor_only_probability"] = factor_probability
    predictions["power_baseline_probability"] = power_probability
    predictions["combined_probability"] = stacked_probability
    predictions["matchup_adjustment_probability_points"] = 100.0 * (
        stacked_probability - power_probability
    )
    predictions["combined_prediction_correct"] = (
        (stacked_probability >= 0.5).astype(int) == valid_y.to_numpy()
    )

    conditional = grouped_model_importance(
        {"two_channel_stack": model},
        valid_x,
        valid_y,
        valid_meta["season"],
        feature_families,
        repeats=40,
    )
    mechanism_families = {feature: feature_role(feature) for feature in factor_x.columns}
    conditional_mechanisms = grouped_model_importance(
        {"two_channel_stack": model},
        valid_x,
        valid_y,
        valid_meta["season"],
        mechanism_families,
        repeats=40,
    )
    stacked_loss = metric_dict(valid_y, stacked_probability)["log_loss"]
    factor_loss = metric_dict(valid_y, factor_probability)["log_loss"]
    power_loss = metric_dict(valid_y, power_probability)["log_loss"]
    channels = pd.DataFrame(
        [
            {
                "channel": "regular_season_power",
                "stacker_coefficient": float(model.stacker.coef_[0, 0]),
                "incremental_log_loss_gain_over_other_channel": float(factor_loss - stacked_loss),
            },
            {
                "channel": "four_factors_matchups",
                "stacker_coefficient": float(model.stacker.coef_[0, 1]),
                "incremental_log_loss_gain_over_other_channel": float(power_loss - stacked_loss),
            },
        ]
    )

    metrics = {
        "architecture": "separate regular-season power and Four Factors matchup channels",
        "power_source": "opponent-adjusted scoring efficiency from regular-season detailed games",
        "factor_source": "opponent-adjusted, reliability-shrunk Four Factors and miscellaneous rates",
        "kenpom_rank_used": False,
        "regular_season_games_used": int(json.loads(Path("reports/data_audit.json").read_text())["regular_season_games_used"]),
        "train_seasons": sorted(meta.loc[train, "season"].unique().astype(int).tolist()),
        "held_out_seasons": sorted(meta.loc[valid, "season"].unique().astype(int).tolist()),
        "factor_features": int(factor_x.shape[1]),
        "factor_tuning": factor_tuning.loc[factor_tuning["selected_configuration"], ["configuration", "cv_log_loss"]].to_dict("records"),
        "power_tuning": power_tuning.loc[power_tuning["selected_configuration"], ["configuration", "cv_log_loss"]].to_dict("records"),
        "stacker": {
            "intercept": float(model.stacker.intercept_[0]),
            "power_coefficient": float(model.stacker.coef_[0, 0]),
            "factor_coefficient": float(model.stacker.coef_[0, 1]),
        },
        "held_out_metrics": comparison.set_index("model").to_dict("index"),
        "data_limitation": "Current official-format local tournament outcomes end in 2023; 2024-2025 remain unavailable without a current competition-data download.",
    }

    top_conditional = conditional.sort_values("importance_mean", ascending=False).head(8)
    top_mechanisms = conditional_mechanisms.sort_values("importance_mean", ascending=False).head(8)
    summary = [
        "# Power Baseline + Interpretable Matchup Model",
        "",
        "## Architecture",
        "",
        "The model keeps opponent-adjusted regular-season scoring power in one channel and Four Factors matchup interactions in another. KenPom rank, net rating, luck, tournament seed, and season-end KenPom values are not inputs.",
        "",
        f"Regular-season games used to estimate pre-tournament team strength: **{metrics['regular_season_games_used']:,}**.",
        "",
        "## Held-out comparison",
        "",
    ]
    summary.extend(
        f"- {clean_label(row.model)}: {row.accuracy:.3f} accuracy, {row.roc_auc:.3f} ROC AUC, {row.log_loss:.3f} log loss"
        for row in comparison.itertuples(index=False)
    )
    summary.extend(
        [
            "",
            "## Channel weights",
            "",
            f"- Regular-season power coefficient: {model.stacker.coef_[0, 0]:.3f}",
            f"- Four Factors matchup coefficient: {model.stacker.coef_[0, 1]:.3f}",
            "",
            "These coefficients describe the final stack, not feature importance. The factor-only and conditional permutation reports remain the interpretation surfaces.",
            "",
            "## Conditional factor importance",
            "",
        ]
    )
    summary.extend(
        f"- {clean_label(row.factor_group)}: {row.importance_mean:.4f} held-out log-loss increase"
        for row in top_conditional.itertuples(index=False)
    )
    summary.extend(["", "## Conditional matchup mechanisms", ""])
    summary.extend(
        f"- {clean_label(row.factor_group)}: {row.importance_mean:.4f} held-out log-loss increase"
        for row in top_mechanisms.itertuples(index=False)
    )
    summary.extend(["", "## Data boundary", "", metrics["data_limitation"]])

    joblib.dump(
        {
            "model": model,
            "factor_features": list(factor_x.columns),
            "power_feature": POWER_FEATURE,
            "feature_families": feature_families,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    COMPARISON_PATH.write_text(comparison.to_csv(index=False))
    PREDICTIONS_PATH.write_text(predictions.to_csv(index=False))
    CONDITIONAL_IMPORTANCE_PATH.write_text(conditional.to_csv(index=False))
    CONDITIONAL_MECHANISM_PATH.write_text(conditional_mechanisms.to_csv(index=False))
    CHANNEL_IMPORTANCE_PATH.write_text(channels.to_csv(index=False))
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")
    plot_comparison(comparison)
    plot_conditional_importance(conditional)
    plot_conditional_mechanisms(conditional_mechanisms)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
