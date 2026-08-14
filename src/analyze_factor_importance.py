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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .build_pretournament_features import main as build_pretournament_features
from .factor_matchups import FACTOR_SPECS, FEATURES_PATH, add_percentiles, build_matchup_dataset


REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
MODEL_PATH = Path("models/factor_importance_model.joblib")
METRICS_PATH = REPORTS / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS / "model_comparison.csv"
ROLLING_PATH = REPORTS / "rolling_validation.csv"
PREDICTIONS_PATH = REPORTS / "validation_predictions.csv"
TOP_FEATURES_PATH = REPORTS / "feature_importance.csv"
GROUP_IMPORTANCE_PATH = REPORTS / "group_importance.csv"
INTERACTION_IMPORTANCE_PATH = REPORTS / "interaction_importance.csv"
TUNING_PATH = REPORTS / "hyperparameter_tuning.csv"
UPSET_PATH = REPORTS / "upset_analysis.csv"
RESULTS_MD_PATH = REPORTS / "summary.md"
FEATURE_IMPORTANCE_PATH = FIGURES / "feature_importance.png"
GROUP_IMPORTANCE_CHART_PATH = FIGURES / "group_importance.png"
INTERACTION_IMPORTANCE_CHART_PATH = FIGURES / "interaction_importance.png"
MODEL_COMPARISON_CHART_PATH = FIGURES / "model_comparison.png"
YEAR_CHART_PATH = FIGURES / "accuracy_by_year.png"
CALIBRATION_CHART_PATH = FIGURES / "calibration.png"
QUADRANT_CHART_PATH = FIGURES / "matchup_quadrants.png"

RANDOM_STATE = 42


@dataclass
class SigmoidCalibratedModel:
    estimator: object
    calibrator: LogisticRegression

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.estimator.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        logits = np.log(raw / (1 - raw)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)


@dataclass
class AveragingCalibratedEnsemble:
    models: tuple[SigmoidCalibratedModel, ...]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([model.predict_proba(x) for model in self.models], axis=0)


def numeric_pipeline(estimator, features: list[str], scale: bool = False) -> Pipeline:
    transforms = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        transforms.append(("scale", StandardScaler()))
    return Pipeline(
        [
            ("prepare", ColumnTransformer([("num", Pipeline(transforms), features)], remainder="drop")),
            ("model", estimator),
        ]
    )


def candidate_configurations(features: list[str]) -> dict[str, list[tuple[str, Pipeline]]]:
    return {
        "regularized_logistic": [
            (
                f"C={c}",
                numeric_pipeline(
                    LogisticRegression(C=c, max_iter=4000, random_state=RANDOM_STATE), features, scale=True
                ),
            )
            for c in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
        ],
        "decision_tree": [
            (
                f"depth={depth},leaf={leaf}",
                numeric_pipeline(
                    DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        criterion="log_loss",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    features,
                ),
            )
            for depth, leaf in [(3, 20), (4, 20), (5, 25), (6, 35)]
        ],
        "random_forest": [
            (
                f"depth={depth},leaf={leaf},max_features={max_features}",
                numeric_pipeline(
                    RandomForestClassifier(
                        n_estimators=450,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        max_features=max_features,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                    features,
                ),
            )
            for depth, leaf, max_features in [(5, 8, 0.35), (7, 10, 0.45), (9, 12, 0.55), (None, 16, 0.45)]
        ],
        "hist_gradient_boosting": [
            (
                f"rate={rate},leaves={leaves},l2={l2}",
                numeric_pipeline(
                    HistGradientBoostingClassifier(
                        learning_rate=rate,
                        max_iter=260,
                        max_leaf_nodes=leaves,
                        min_samples_leaf=20,
                        l2_regularization=l2,
                        random_state=RANDOM_STATE,
                    ),
                    features,
                ),
            )
            for rate, leaves, l2 in [(0.025, 7, 4.0), (0.04, 7, 2.0), (0.03, 15, 4.0), (0.05, 15, 2.0)]
        ],
    }


def metric_dict(y: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def temporal_oof_probabilities(
    estimator: Pipeline, x: pd.DataFrame, y: pd.Series, seasons: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    unique_years = sorted(seasons.unique())
    for year in unique_years:
        prior = seasons < year
        current = seasons == year
        if seasons.loc[prior].nunique() < 4 or current.sum() == 0:
            continue
        fold = clone(estimator)
        fold.fit(x.loc[prior], y.loc[prior])
        probabilities.extend(fold.predict_proba(x.loc[current])[:, 1])
        labels.extend(y.loc[current])
    return np.asarray(probabilities), np.asarray(labels)


def tune_candidates(
    configurations: dict[str, list[tuple[str, Pipeline]]],
    x: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    selected: dict[str, Pipeline] = {}
    rows = []
    for family, options in configurations.items():
        family_rows = []
        for configuration, estimator in options:
            probability, labels = temporal_oof_probabilities(estimator, x, y, seasons)
            row = {
                "model": family,
                "configuration": configuration,
                "oof_games": int(len(labels)),
                **{f"cv_{key}": value for key, value in metric_dict(labels, probability).items()},
            }
            family_rows.append((row, estimator))
            rows.append(row)
        best_row, best_estimator = min(family_rows, key=lambda item: item[0]["cv_log_loss"])
        selected[family] = best_estimator
        best_row["selected_configuration"] = True

    tuning = pd.DataFrame(rows)
    tuning["selected_configuration"] = tuning["selected_configuration"].fillna(False).astype(bool)
    return selected, tuning.sort_values(["model", "cv_log_loss"]).reset_index(drop=True)


def training_cv_comparison(
    candidates: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, seasons: pd.Series
) -> pd.DataFrame:
    rows = []
    probabilities = []
    labels = None
    for name, candidate in candidates.items():
        probability, candidate_labels = temporal_oof_probabilities(candidate, x, y, seasons)
        probabilities.append(probability)
        labels = candidate_labels
        rows.append({"model": name, **{f"training_cv_{key}": value for key, value in metric_dict(labels, probability).items()}})
    ensemble_probability = np.mean(probabilities, axis=0)
    rows.append(
        {
            "model": "soft_voting_ensemble",
            **{f"training_cv_{key}": value for key, value in metric_dict(labels, ensemble_probability).items()},
        }
    )
    return pd.DataFrame(rows)


def fit_temporally_calibrated(
    estimator: Pipeline, x: pd.DataFrame, y: pd.Series, seasons: pd.Series
) -> SigmoidCalibratedModel:
    oof_probability, oof_y = temporal_oof_probabilities(estimator, x, y, seasons)
    if len(oof_probability) < 100:
        raise ValueError("Not enough temporal out-of-fold predictions to calibrate probabilities.")
    logits = np.log(np.clip(oof_probability, 1e-6, 1 - 1e-6) / np.clip(1 - oof_probability, 1e-6, 1))
    calibrator = LogisticRegression(C=100.0, max_iter=2000, random_state=RANDOM_STATE)
    calibrator.fit(logits.reshape(-1, 1), oof_y)
    fitted = clone(estimator).fit(x, y)
    return SigmoidCalibratedModel(fitted, calibrator)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(center - margin), float(center + margin)


def fit_and_compare(
    candidates: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    train = meta["season"] <= 2016
    valid = meta["season"].between(2017, 2025)
    fitted = {}
    rows = []
    for name, candidate in candidates.items():
        calibrated = fit_temporally_calibrated(candidate, x.loc[train], y.loc[train], meta.loc[train, "season"])
        fitted[name] = calibrated

    fitted["soft_voting_ensemble"] = AveragingCalibratedEnsemble(tuple(fitted.values()))
    for name, calibrated in fitted.items():
        if name == "soft_voting_ensemble":
            raw_probability = np.mean(
                [model.estimator.predict_proba(x.loc[valid])[:, 1] for model in calibrated.models], axis=0
            )
        else:
            raw_probability = calibrated.estimator.predict_proba(x.loc[valid])[:, 1]
        calibrated_probability = calibrated.predict_proba(x.loc[valid])[:, 1]
        row = {"model": name}
        row.update({f"raw_{key}": value for key, value in metric_dict(y.loc[valid], raw_probability).items()})
        row.update({f"calibrated_{key}": value for key, value in metric_dict(y.loc[valid], calibrated_probability).items()})
        rows.append(row)
    return fitted, pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)


def rolling_validation(
    candidates: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for year in sorted(meta.loc[meta["season"] >= 2017, "season"].unique()):
        yearly_probabilities = []
        for name, candidate in candidates.items():
            train = meta["season"] < year
            test = meta["season"] == year
            model = clone(candidate).fit(x.loc[train], y.loc[train])
            probability = model.predict_proba(x.loc[test])[:, 1]
            yearly_probabilities.append(probability)
            rows.append({"model": name, "season": int(year), "games": int(test.sum()), **metric_dict(y.loc[test], probability)})
        ensemble_probability = np.mean(yearly_probabilities, axis=0)
        rows.append(
            {
                "model": "soft_voting_ensemble",
                "season": int(year),
                "games": int(test.sum()),
                **metric_dict(y.loc[test], ensemble_probability),
            }
        )
    return pd.DataFrame(rows)


def select_model(comparison: pd.DataFrame) -> tuple[str, bool]:
    eligible = comparison[
        (comparison["training_cv_accuracy"] >= 0.60) & (comparison["training_cv_roc_auc"] >= 0.65)
    ]
    if eligible.empty:
        return str(comparison.sort_values("training_cv_log_loss").iloc[0]["model"]), False
    selected = eligible.sort_values(
        ["training_cv_log_loss", "training_cv_roc_auc"], ascending=[True, False]
    ).iloc[0]
    return str(selected["model"]), True


def permute_within_season(
    frame: pd.DataFrame, columns: list[str], seasons: pd.Series, rng: np.random.Generator
) -> pd.DataFrame:
    permuted = frame.copy()
    for _season, indices in seasons.groupby(seasons).groups.items():
        positions = np.asarray(list(indices))
        shuffled = rng.permutation(positions)
        permuted.loc[positions, columns] = frame.loc[shuffled, columns].to_numpy()
    return permuted


def permutation_importance_by_season(
    model: SigmoidCalibratedModel,
    x: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    feature_families: dict[str, str],
    repeats: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_probability = model.predict_proba(x)[:, 1]
    baseline = {
        season: log_loss(y.loc[seasons == season], baseline_probability[seasons.to_numpy() == season], labels=[0, 1])
        for season in sorted(seasons.unique())
    }
    rng = np.random.default_rng(RANDOM_STATE)
    raw_rows = []
    for feature in x.columns:
        for repeat in range(repeats):
            permuted = permute_within_season(x, [feature], seasons, rng)
            probability = model.predict_proba(permuted)[:, 1]
            for season in baseline:
                mask = seasons == season
                delta = log_loss(y.loc[mask], probability[mask.to_numpy()], labels=[0, 1]) - baseline[season]
                raw_rows.append(
                    {
                        "feature": feature,
                        "family": feature_families[feature],
                        "season": int(season),
                        "repeat": repeat,
                        "log_loss_increase": float(delta),
                    }
                )
    raw = pd.DataFrame(raw_rows)
    summary = raw.groupby(["feature", "family"])["log_loss_increase"].agg(
        importance_mean="mean",
        importance_std="std",
        importance_low=lambda values: values.quantile(0.025),
        importance_high=lambda values: values.quantile(0.975),
        positive_fraction=lambda values: (values > 0).mean(),
    ).reset_index()
    season_means = raw.groupby(["feature", "season"])["log_loss_increase"].mean()
    positive_seasons = season_means.groupby("feature").apply(lambda values: float((values > 0).mean()))
    summary["positive_season_fraction"] = summary["feature"].map(positive_seasons)
    summary = summary.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary, raw


def grouped_model_importance(
    models: dict[str, object],
    x: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    feature_families: dict[str, str],
    repeats: int = 30,
) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for feature, family in feature_families.items():
        groups.setdefault(family, []).append(feature)
    rows = []
    for model_name, model in models.items():
        baseline = log_loss(y, model.predict_proba(x)[:, 1], labels=[0, 1])
        rng = np.random.default_rng(RANDOM_STATE)
        for group, columns in groups.items():
            deltas = []
            for _repeat in range(repeats):
                permuted = permute_within_season(x, columns, seasons, rng)
                delta = log_loss(y, model.predict_proba(permuted)[:, 1], labels=[0, 1]) - baseline
                deltas.append(delta)
            rows.append(
                {
                    "model": model_name,
                    "factor_group": group,
                    "feature_count": len(columns),
                    "importance_mean": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas, ddof=1)),
                    "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "importance_mean"], ascending=[True, False])


def feature_role(feature: str) -> str:
    if "opponent_adjusted_matchup_edge" in feature:
        return "opponent_adjusted_edges"
    if "recent_form" in feature:
        return "recent_form"
    if "reliability" in feature:
        return "reliability"
    if "volatility" in feature:
        return "volatility"
    if "_raw_" in feature:
        return "raw_matchup_edges"
    if "strength_vs_strength" in feature:
        return "strength_vs_strength"
    if "strength_vs_weakness" in feature or feature == "weakness_exploitation_composite":
        return "strength_vs_weakness"
    if "weakness_vs_strength" in feature:
        return "weakness_vs_strength"
    if feature.endswith("_net_matchup_edge"):
        return "net_matchup_edges"
    if feature.endswith("_matchup_environment"):
        return "matchup_environments"
    if feature.startswith("three_point_rate_"):
        return "three_point_volume_style"
    return "engineered_composites"


def build_validation_predictions(
    model: SigmoidCalibratedModel, x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame
) -> pd.DataFrame:
    probability = model.predict_proba(x)[:, 1]
    output = meta.copy()
    output["pred_team_a_win_prob"] = probability
    output["pred_team_a_win"] = (probability >= 0.5).astype(int)
    output["prediction_correct"] = output["pred_team_a_win"].eq(y.to_numpy())
    output["predicted_winner"] = np.where(output["pred_team_a_win"].eq(1), output["team_a"], output["team_b"])
    return output


def upset_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    seeded = predictions.dropna(subset=["actual_upset"]).copy()
    rows = []
    for label, subset in [("actual_upsets", seeded[seeded["actual_upset"].eq(True)]), ("favorite_wins", seeded[seeded["actual_upset"].eq(False)]), ("all_seeded_games", seeded)]:
        rows.append(
            {
                "segment": label,
                "games": int(len(subset)),
                "accuracy": float(subset["prediction_correct"].mean()),
                "mean_confidence": float(
                    np.maximum(subset["pred_team_a_win_prob"], 1 - subset["pred_team_a_win_prob"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def clean_label(value: str) -> str:
    label = value.replace("_", " ").replace("nonsteal", "non-steal").title()
    return (
        label.replace("Efg", "eFG")
        .replace(" Pct", " %")
        .replace(" Off ", " Offense ")
        .replace(" Def ", " Defense ")
        .replace(" Std", " Variability")
    )


def plot_feature_importance(importance: pd.DataFrame) -> None:
    plotted = importance.head(18).iloc[::-1]
    xerr = np.vstack(
        [
            (plotted["importance_mean"] - plotted["importance_low"]).clip(lower=0),
            (plotted["importance_high"] - plotted["importance_mean"]).clip(lower=0),
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(plotted["feature"].map(clean_label), plotted["importance_mean"], color="#287271", xerr=xerr, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Validation Permutation Importance: Raw + Engineered Factors")
    ax.set_xlabel("Increase in log loss when shuffled (larger = more important)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=190)
    plt.close(fig)


def plot_group_importance(grouped: pd.DataFrame, path: Path, title: str) -> None:
    consensus = grouped.groupby("factor_group")["importance_mean"].agg(["mean", "min", "max"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    xerr = np.vstack([consensus["mean"] - consensus["min"], consensus["max"] - consensus["mean"]])
    ax.barh(consensus.index.map(clean_label), consensus["mean"], xerr=xerr, color="#d99b45", alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Mean held-out log-loss increase; whiskers show model range")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    ordered = comparison.sort_values("calibrated_log_loss")
    labels = ordered["model"].map(clean_label)
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(positions, ordered["calibrated_accuracy"], color="#287271")
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlim(0.5, max(0.75, ordered["calibrated_accuracy"].max() + 0.03))
    axes[0].set_xlabel("Held-out accuracy")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(positions, ordered["calibrated_log_loss"], color="#d99b45")
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlabel("Held-out log loss (lower is better)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Leakage-Safe Factors-Only Model Comparison")
    fig.tight_layout()
    fig.savefig(MODEL_COMPARISON_CHART_PATH, dpi=190)
    plt.close(fig)


def plot_accuracy_by_year(predictions: pd.DataFrame) -> None:
    yearly = predictions.groupby("season")["prediction_correct"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(yearly.index, yearly["mean"], marker="o", color="#287271", linewidth=2)
    ax.axhline(0.5, color="#777777", linewidth=1, linestyle="--")
    for year, row in yearly.iterrows():
        ax.annotate(f"{row['mean']:.0%}", (year, row["mean"]), xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_ylim(0.45, max(0.8, yearly["mean"].max() + 0.08))
    ax.set_title("Selected Model Accuracy by Held-Out Season")
    ax.set_xlabel("Tournament season")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(YEAR_CHART_PATH, dpi=190)
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame) -> None:
    bins = pd.cut(predictions["pred_team_a_win_prob"], bins=np.linspace(0, 1, 9), include_lowest=True)
    calibration = predictions.groupby(bins, observed=False).agg(
        predicted=("pred_team_a_win_prob", "mean"), actual=("actual_team_a_win", "mean"), games=("actual_team_a_win", "size")
    ).dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.plot(calibration["predicted"], calibration["actual"], marker="o", color="#287271", linewidth=2)
    for _, row in calibration.iterrows():
        ax.annotate(str(int(row["games"])), (row["predicted"], row["actual"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Held-Out Probability Calibration")
    ax.set_xlabel("Mean predicted Team A win probability")
    ax.set_ylabel("Observed Team A win rate")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CALIBRATION_CHART_PATH, dpi=190)
    plt.close(fig)


def plot_matchup_quadrants(meta: pd.DataFrame) -> None:
    team_features = add_percentiles(pd.read_csv(FEATURES_PATH)).set_index(["season", "team_id"])
    factor_names = ["efg", "turnover", "offensive_rebound", "free_throw_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#eeeeee")
    minimum_cell_games = 10
    for ax, name in zip(axes.flat, factor_names):
        off_col, def_col, *_rest = FACTOR_SPECS[name]
        matrix = np.full((2, 2), np.nan)
        counts = np.zeros((2, 2), dtype=int)
        wins = np.zeros((2, 2), dtype=float)
        for game in meta.itertuples(index=False):
            a = team_features.loc[(game.season, game.team_a_id), f"{off_col}_strength"]
            b = team_features.loc[(game.season, game.team_b_id), f"{def_col}_strength"]
            if a >= 0.67:
                off_bin = 1
            elif a <= 0.33:
                off_bin = 0
            else:
                continue
            if b >= 0.67:
                def_bin = 1
            elif b <= 0.33:
                def_bin = 0
            else:
                continue
            counts[off_bin, def_bin] += 1
            wins[off_bin, def_bin] += game.actual_team_a_win
        np.divide(wins, counts, out=matrix, where=counts > 0)
        matrix[counts < minimum_cell_games] = np.nan
        image = ax.imshow(matrix, vmin=0.3, vmax=0.7, cmap=color_map)
        for off_bin in range(2):
            for def_bin in range(2):
                text = (
                    f"n={counts[off_bin, def_bin]}\nlow sample"
                    if counts[off_bin, def_bin] < minimum_cell_games
                    else f"{matrix[off_bin, def_bin]:.0%}\nn={counts[off_bin, def_bin]}"
                )
                ax.text(def_bin, off_bin, text, ha="center", va="center", color="#111111")
        ax.set_xticks([0, 1], ["Weak defense", "Strong defense"])
        ax.set_yticks([0, 1], ["Weak offense", "Strong offense"])
        ax.set_title(clean_label(name))
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Observed Team A win rate",
        orientation="horizontal",
        shrink=0.72,
        pad=0.08,
    )
    fig.suptitle("Held-Out Strength-vs-Weakness and Strength-vs-Strength Outcomes")
    fig.savefig(QUADRANT_CHART_PATH, dpi=190)
    plt.close(fig)


def write_summary(
    metrics: dict,
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    grouped: pd.DataFrame,
    interactions: pd.DataFrame,
    upset: pd.DataFrame,
) -> None:
    selected = metrics["selected_model"]
    selected_row = comparison.set_index("model").loc[selected]
    top_features = importance.head(8)
    consensus = grouped.groupby("factor_group")["importance_mean"].mean().sort_values(ascending=False)
    interaction_consensus = (
        interactions.groupby("factor_group")["importance_mean"].mean().sort_values(ascending=False)
    )
    lines = [
        "# Trustworthy Four Factors + Misc Importance Analysis",
        "",
        "## Data timing",
        "",
        "Features are computed only from NCAA regular-season detailed box scores. Tournament results are stored separately, so the feature snapshot is confirmed pre-tournament rather than inferred from a season-end KenPom page.",
        "",
        "## Selected model",
        "",
        f"- Model: {clean_label(selected)}",
        f"- Held-out accuracy: {selected_row['calibrated_accuracy']:.3f}",
        f"- Held-out ROC AUC: {selected_row['calibrated_roc_auc']:.3f}",
        f"- Held-out log loss: {selected_row['calibrated_log_loss']:.3f}",
        f"- Accuracy 95% Wilson interval: {metrics['trustworthiness']['accuracy_wilson_95'][0]:.3f}-{metrics['trustworthiness']['accuracy_wilson_95'][1]:.3f}",
        f"- Importance interpretation gate passed: {metrics['trustworthiness']['importance_gate_passed']}",
        f"- Model selection basis: {metrics['selection_basis']}",
        "",
        "## Most important features",
        "",
    ]
    lines.extend(
        f"- {clean_label(row.feature)}: {row.importance_mean:.4f} mean held-out log-loss increase; positive in {row.positive_season_fraction:.0%} of seasons"
        for row in top_features.itertuples(index=False)
    )
    lines.extend(["", "## Factor-group consensus", ""])
    lines.extend(f"- {clean_label(name)}: {value:.4f}" for name, value in consensus.items())
    lines.extend(["", "## Matchup-mechanism importance", ""])
    lines.extend(f"- {clean_label(name)}: {value:.4f}" for name, value in interaction_consensus.items())
    lines.extend(["", "## Upset behavior", ""])
    lines.extend(
        f"- {clean_label(row.segment)}: {row.accuracy:.3f} accuracy across {int(row.games)} games"
        for row in upset.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "Permutation importance measures predictive reliance, not causality. Correlated engineered features share credit, which is why the grouped chart and cross-model range are more reliable than a single feature's exact rank.",
            "Tournament seeds, KenPom rank, net rating, luck, and adjusted efficiency are excluded from the factor-only model. The separate two-channel report uses an internally estimated regular-season scoring-efficiency baseline; seeds are used only after prediction to label upset evaluation segments.",
        ]
    )
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    build_pretournament_features()
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    x, y, meta, feature_families = build_matchup_dataset()
    observed = x.notna().any(axis=0)
    x = x.loc[:, observed]
    feature_families = {feature: feature_families[feature] for feature in x.columns}
    train = meta["season"] <= 2016
    valid = meta["season"].between(2017, 2025)
    if not train.any() or not valid.any():
        raise ValueError("The factor dataset does not cover both requested train and validation periods.")

    configurations = candidate_configurations(list(x.columns))
    candidates, tuning = tune_candidates(
        configurations, x.loc[train], y.loc[train], meta.loc[train, "season"]
    )
    training_comparison = training_cv_comparison(
        candidates, x.loc[train], y.loc[train], meta.loc[train, "season"]
    )
    models, comparison = fit_and_compare(candidates, x, y, meta)
    comparison = comparison.merge(training_comparison, on="model", how="left")
    rolling = rolling_validation(candidates, x, y, meta)
    selected_name, selection_gate = select_model(comparison)
    selected = models[selected_name]

    valid_x = x.loc[valid].reset_index(drop=True)
    valid_y = y.loc[valid].reset_index(drop=True)
    valid_meta = meta.loc[valid].reset_index(drop=True)
    importance, _raw_importance = permutation_importance_by_season(
        selected, valid_x, valid_y, valid_meta["season"], feature_families
    )
    grouped = grouped_model_importance(models, valid_x, valid_y, valid_meta["season"], feature_families)
    feature_roles = {feature: feature_role(feature) for feature in x.columns}
    interactions = grouped_model_importance(models, valid_x, valid_y, valid_meta["season"], feature_roles)
    predictions = build_validation_predictions(selected, valid_x, valid_y, valid_meta)
    upset = upset_analysis(predictions)

    selected_metrics = comparison.set_index("model").loc[selected_name]
    correct = int(predictions["prediction_correct"].sum())
    total = int(len(predictions))
    interval = wilson_interval(correct, total)
    stable_top_features = int((importance.head(15)["positive_season_fraction"] >= 0.60).sum())
    importance_gate = bool(selection_gate and interval[0] > 0.5 and stable_top_features >= 5)
    metrics = {
        "analysis_goal": "Interpret Four Factors and misc importance before optimizing maximum predictive accuracy",
        "snapshot_status": "confirmed_pre_tournament",
        "feature_source": "NCAA regular-season detailed box scores using KenPom-compatible formulas",
        "excluded_model_features": ["tournament seed", "KenPom rank", "net rating", "luck", "adjusted efficiency"],
        "requested_train_years": "1997-2016",
        "actual_train_years": sorted(meta.loc[train, "season"].unique().astype(int).tolist()),
        "requested_validation_years": "2017-2025",
        "actual_validation_years": sorted(meta.loc[valid, "season"].unique().astype(int).tolist()),
        "one_row_per_game": True,
        "feature_count": int(x.shape[1]),
        "raw_matchup_feature_count": int(sum("_raw_" in feature for feature in x.columns)),
        "selected_model": selected_name,
        "selection_basis": "hyperparameters and model family selected only by pre-2017 expanding-window cross-validation; 2017-2023 is evaluation-only",
        "selected_metrics": {key: float(value) for key, value in selected_metrics.items() if key != "model"},
        "trustworthiness": {
            "importance_gate_passed": importance_gate,
            "selection_accuracy_auc_gate_passed": bool(selection_gate),
            "validation_used_for_model_family_selection": False,
            "final_untouched_test_set_available": False,
            "accuracy_wilson_95": list(interval),
            "stable_features_in_top_15": stable_top_features,
            "minimum_stability_definition": "positive permutation importance in at least 60% of held-out seasons",
        },
    }

    joblib.dump(
        {
            "model": selected,
            "features": list(x.columns),
            "feature_families": feature_families,
            "feature_roles": feature_roles,
            "snapshot_status": "confirmed_pre_tournament",
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    tuning.to_csv(TUNING_PATH, index=False)
    rolling.to_csv(ROLLING_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    importance.to_csv(TOP_FEATURES_PATH, index=False)
    grouped.to_csv(GROUP_IMPORTANCE_PATH, index=False)
    interactions.to_csv(INTERACTION_IMPORTANCE_PATH, index=False)
    upset.to_csv(UPSET_PATH, index=False)

    plot_feature_importance(importance)
    plot_group_importance(
        grouped,
        GROUP_IMPORTANCE_CHART_PATH,
        "Basketball Factor Importance Across Candidate Models",
    )
    plot_group_importance(
        interactions,
        INTERACTION_IMPORTANCE_CHART_PATH,
        "Raw And Matchup-Interaction Importance",
    )
    plot_model_comparison(comparison)
    plot_accuracy_by_year(predictions)
    plot_calibration(predictions)
    plot_matchup_quadrants(valid_meta)
    write_summary(metrics, comparison, importance, grouped, interactions, upset)

    print(json.dumps({"selected_model": selected_name, "metrics": metrics["selected_metrics"], "trustworthiness": metrics["trustworthiness"]}, indent=2))
    print(f"Wrote {FEATURE_IMPORTANCE_PATH}")
    print(f"Wrote {GROUP_IMPORTANCE_CHART_PATH}")
    print(f"Wrote {INTERACTION_IMPORTANCE_CHART_PATH}")


if __name__ == "__main__":
    main()
