from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .team_names import normalize_team_name
from .train_decision_tree import load_kenpom_features, load_tournament_games


RAW_2026_RESULTS_PATH = Path("data/raw/ncaa_2026_tournament_results.csv")
KENPOM_2026_PATH = Path("data/raw/kenpom/kenpom_2026.csv")
MODEL_PATH = Path("models/decision_tree_four_factors_misc.joblib")
METRICS_PATH = Path("reports/factor_model_metrics.json")
VALIDATION_PREDICTIONS_PATH = Path("reports/factor_model_validation_predictions.csv")
PREDICTIONS_2026_PATH = Path("reports/2026_factor_model_predictions.csv")
SUMMARY_2026_PATH = Path("reports/2026_factor_model_summary.json")
FEATURE_IMPORTANCE_PATH = Path("reports/factor_model_feature_importance.png")
TOP_FEATURES_PATH = Path("reports/factor_model_top_features.csv")


FOUR_FACTOR_MATCHUPS = {
    "efg": {
        "off_col": "ff_off_efg",
        "def_col": "ff_def_efg",
        "off_higher_is_better": True,
        "def_higher_is_better": False,
        "label": "eFG",
    },
    "turnover": {
        "off_col": "ff_off_to",
        "def_col": "ff_def_to",
        "off_higher_is_better": False,
        "def_higher_is_better": True,
        "label": "Turnover",
    },
    "oreb": {
        "off_col": "ff_off_or",
        "def_col": "ff_def_or",
        "off_higher_is_better": True,
        "def_higher_is_better": False,
        "label": "Off Rebound",
    },
    "ftrate": {
        "off_col": "ff_off_ftrate",
        "def_col": "ff_def_ftrate",
        "off_higher_is_better": True,
        "def_higher_is_better": False,
        "label": "Free Throw Rate",
    },
}

MISC_STRENGTHS = {
    "threep": ("misc_3p", True),
    "twop": ("misc_2p", True),
    "ft": ("misc_ft", True),
    "blk": ("misc_blk", True),
    "stl": ("misc_stl", True),
    "nst": ("misc_nst", False),
    "assist": ("misc_a", True),
    "threepa": ("misc_3pa", True),
}


def coalesce_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "misc_3p": "misc_threep",
        "misc_3p_rank": "misc_threep_rank",
        "misc_2p": "misc_twop",
        "misc_2p_rank": "misc_twop_rank",
        "misc_2p_dist": "misc_twop_dist",
        "misc_2p_dist_rank": "misc_twop_dist_rank",
        "misc_a": "misc_assist",
        "misc_a_rank": "misc_assist_rank",
        "misc_3pa": "misc_threepa",
        "misc_3pa_rank": "misc_threepa_rank",
    }
    for canonical, alias in aliases.items():
        if canonical in df.columns and alias in df.columns:
            df[canonical] = pd.to_numeric(df[canonical], errors="coerce").combine_first(
                pd.to_numeric(df[alias], errors="coerce")
            )
        elif alias in df.columns:
            df[canonical] = pd.to_numeric(df[alias], errors="coerce")
    return df


def add_strength_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = coalesce_alias_columns(df)
    df = df.copy()

    for spec in FOUR_FACTOR_MATCHUPS.values():
        for side, col, higher_is_better in [
            ("off", spec["off_col"], spec["off_higher_is_better"]),
            ("def", spec["def_col"], spec["def_higher_is_better"]),
        ]:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            score = values if higher_is_better else -values
            df[f"{col}_strength_pct"] = score.groupby(df["season"]).rank(pct=True)

    for name, (col, higher_is_better) in MISC_STRENGTHS.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        score = values if higher_is_better else -values
        df[f"{col}_strength_pct"] = score.groupby(df["season"]).rank(pct=True)

    return df


def factor_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = []
    for spec in FOUR_FACTOR_MATCHUPS.values():
        base_cols.extend([spec["off_col"], spec["def_col"]])
        base_cols.extend([f"{spec['off_col']}_strength_pct", f"{spec['def_col']}_strength_pct"])
    for col, _higher_is_better in MISC_STRENGTHS.values():
        base_cols.extend([col, f"{col}_strength_pct"])

    available = []
    for col in base_cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            available.append(col)
    return available


def get_team_features(game: pd.Series, team_prefix: str, col: str):
    return game.get(f"{team_prefix}_{col}")


def add_four_factor_matchup_features(row: dict, game: pd.Series, team_a_prefix: str, team_b_prefix: str) -> None:
    for name, spec in FOUR_FACTOR_MATCHUPS.items():
        a_off = get_team_features(game, team_a_prefix, spec["off_col"])
        a_def = get_team_features(game, team_a_prefix, spec["def_col"])
        b_off = get_team_features(game, team_b_prefix, spec["off_col"])
        b_def = get_team_features(game, team_b_prefix, spec["def_col"])
        a_off_strength = get_team_features(game, team_a_prefix, f"{spec['off_col']}_strength_pct")
        a_def_strength = get_team_features(game, team_a_prefix, f"{spec['def_col']}_strength_pct")
        b_off_strength = get_team_features(game, team_b_prefix, f"{spec['off_col']}_strength_pct")
        b_def_strength = get_team_features(game, team_b_prefix, f"{spec['def_col']}_strength_pct")

        if pd.notna(a_off) and pd.notna(b_def):
            if name in {"efg", "oreb", "ftrate"}:
                row[f"{name}_a_attack_edge"] = a_off - b_def
            else:
                row[f"{name}_a_attack_edge"] = b_def - a_off
        if pd.notna(b_off) and pd.notna(a_def):
            if name in {"efg", "oreb", "ftrate"}:
                row[f"{name}_b_attack_edge"] = b_off - a_def
            else:
                row[f"{name}_b_attack_edge"] = a_def - b_off
        if f"{name}_a_attack_edge" in row and f"{name}_b_attack_edge" in row:
            row[f"{name}_net_matchup_edge"] = row[f"{name}_a_attack_edge"] - row[f"{name}_b_attack_edge"]
            row[f"{name}_matchup_edge_abs"] = abs(row[f"{name}_net_matchup_edge"])

        if pd.notna(a_off_strength) and pd.notna(b_def_strength):
            row[f"{name}_a_strength_vs_strength"] = a_off_strength * b_def_strength
            row[f"{name}_a_strength_vs_weakness"] = a_off_strength * (1 - b_def_strength)
            row[f"{name}_a_weakness_vs_strength"] = (1 - a_off_strength) * b_def_strength
            row[f"{name}_a_weakness_vs_weakness"] = (1 - a_off_strength) * (1 - b_def_strength)
        if pd.notna(b_off_strength) and pd.notna(a_def_strength):
            row[f"{name}_b_strength_vs_strength"] = b_off_strength * a_def_strength
            row[f"{name}_b_strength_vs_weakness"] = b_off_strength * (1 - a_def_strength)
            row[f"{name}_b_weakness_vs_strength"] = (1 - b_off_strength) * a_def_strength
            row[f"{name}_b_weakness_vs_weakness"] = (1 - b_off_strength) * (1 - a_def_strength)
        for suffix in [
            "strength_vs_strength",
            "strength_vs_weakness",
            "weakness_vs_strength",
            "weakness_vs_weakness",
        ]:
            a_key = f"{name}_a_{suffix}"
            b_key = f"{name}_b_{suffix}"
            if a_key in row and b_key in row:
                row[f"{name}_net_{suffix}"] = row[a_key] - row[b_key]


def add_misc_features(row: dict, game: pd.Series, team_a_prefix: str, team_b_prefix: str) -> None:
    for name, (col, _higher_is_better) in MISC_STRENGTHS.items():
        a = get_team_features(game, team_a_prefix, col)
        b = get_team_features(game, team_b_prefix, col)
        a_strength = get_team_features(game, team_a_prefix, f"{col}_strength_pct")
        b_strength = get_team_features(game, team_b_prefix, f"{col}_strength_pct")
        if pd.notna(a) and pd.notna(b):
            row[f"misc_{name}_diff"] = a - b
            row[f"misc_{name}_abs_diff"] = abs(a - b)
        if pd.notna(a_strength) and pd.notna(b_strength):
            row[f"misc_{name}_strength_diff"] = a_strength - b_strength
            row[f"misc_{name}_strength_abs_diff"] = abs(a_strength - b_strength)


def attach_features(games: pd.DataFrame, kenpom: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    left = games.merge(
        kenpom[["season", "team_norm", "team", *feature_cols]].add_prefix("winner_"),
        left_on=["season", "winner_norm"],
        right_on=["winner_season", "winner_team_norm"],
        how="left",
    )
    full = left.merge(
        kenpom[["season", "team_norm", "team", *feature_cols]].add_prefix("loser_"),
        left_on=["season", "loser_norm"],
        right_on=["loser_season", "loser_team_norm"],
        how="left",
    )
    return full.drop(columns=["winner_season", "loser_season"], errors="ignore")


def build_examples(games_with_features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows = []
    labels = []
    meta = []
    for _, game in games_with_features.iterrows():
        if game[[f"winner_{col}" for col in feature_cols]].isna().all() or game[
            [f"loser_{col}" for col in feature_cols]
        ].isna().all():
            continue
        for team_a_prefix, team_b_prefix, label in [("winner", "loser", 1), ("loser", "winner", 0)]:
            row = {}
            add_four_factor_matchup_features(row, game, team_a_prefix, team_b_prefix)
            add_misc_features(row, game, team_a_prefix, team_b_prefix)
            rows.append(row)
            labels.append(label)
            meta.append(
                {
                    "season": game["season"],
                    "team_a": game[team_a_prefix],
                    "team_b": game[team_b_prefix],
                    "actual_team_a_win": label,
                }
            )
    return pd.DataFrame(rows), pd.Series(labels, name="team_a_win"), pd.DataFrame(meta)


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 else None,
    }


def train_model() -> tuple[Pipeline, list[str], pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    games = load_tournament_games()
    kenpom = add_strength_percentiles(load_kenpom_features())
    feature_cols = factor_feature_columns(kenpom)
    merged = attach_features(games, kenpom, feature_cols)
    x, y, meta = build_examples(merged, feature_cols)
    train_mask = meta["season"].between(1997, 2016)
    valid_mask = meta["season"].between(2017, 2025)
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("No train or validation rows after joining tournament games to KenPom factors.")
    x = x.loc[:, x.loc[train_mask].notna().any(axis=0)]

    model = Pipeline(
        steps=[
            ("imputer", ColumnTransformer([("num", SimpleImputer(strategy="median"), x.columns)], remainder="drop")),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=25,
                    criterion="log_loss",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x.loc[train_mask], y.loc[train_mask])

    metrics = {
        "model_name": "Four Factors + Misc matchup-engineered decision tree",
        "excluded_feature_families": ["KenPom rank", "net rating", "luck", "broad efficiency ratings"],
        "actual_train_years": sorted(meta.loc[train_mask, "season"].unique().tolist()),
        "actual_validation_years": sorted(meta.loc[valid_mask, "season"].unique().tolist()),
        "feature_count": int(x.shape[1]),
        "rows": {
            "train": int(train_mask.sum()),
            "validation": int(valid_mask.sum()),
            "all_examples": int(len(x)),
        },
        "metrics": {
            "train": evaluate(model, x.loc[train_mask], y.loc[train_mask]),
            "validation": evaluate(model, x.loc[valid_mask], y.loc[valid_mask]),
        },
        "tree": export_text(model.named_steps["tree"], feature_names=list(x.columns)),
    }
    return model, list(x.columns), x, y, meta, metrics


def predict_2026(model: Pipeline, model_features: list[str]) -> tuple[pd.DataFrame, dict]:
    games = pd.read_csv(RAW_2026_RESULTS_PATH)
    games["season"] = 2026
    games["winner_norm"] = games["winner"].map(normalize_team_name)
    games["loser_norm"] = games["loser"].map(normalize_team_name)

    kenpom = pd.read_csv(KENPOM_2026_PATH)
    kenpom["team_norm"] = kenpom["team"].map(normalize_team_name)
    kenpom["season"] = 2026
    kenpom = add_strength_percentiles(kenpom)
    feature_cols = factor_feature_columns(kenpom)
    merged = attach_features(games, kenpom, feature_cols)
    x_2026, _y_unused, _meta_unused = build_examples(merged, feature_cols)
    x_2026 = x_2026.iloc[::2].reset_index(drop=True)
    x_2026 = x_2026.reindex(columns=model_features)

    winner_prob = model.predict_proba(x_2026)[:, 1]
    pred_actual_winner = (winner_prob >= 0.5).astype(int)
    predictions = games[["round", "date", "winner", "loser", "winner_score", "loser_score", "source"]].copy()
    predictions["actual_winner"] = predictions["winner"]
    predictions["actual_loser"] = predictions["loser"]
    predictions["actual_margin"] = predictions["winner_score"] - predictions["loser_score"]
    predictions["pred_actual_winner_prob"] = winner_prob
    predictions["predicted_winner"] = np.where(pred_actual_winner == 1, predictions["winner"], predictions["loser"])
    predictions["prediction_correct"] = pred_actual_winner.astype(bool)

    y_true = np.ones(len(predictions), dtype=int)
    summary = {
        "season": 2026,
        "games": int(len(predictions)),
        "correct": int(predictions["prediction_correct"].sum()),
        "accuracy": float(accuracy_score(y_true, pred_actual_winner)),
        "brier": float(brier_score_loss(y_true, winner_prob)),
        "log_loss": float(log_loss(y_true, winner_prob, labels=[0, 1])),
        "by_round": predictions.groupby("round")["prediction_correct"].agg(["sum", "count", "mean"]).to_dict("index"),
    }
    return predictions, summary


def feature_importance_frame(model: Pipeline, features: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": features, "importance": model.named_steps["tree"].feature_importances_})
        .sort_values("importance", ascending=False)
        .query("importance > 0")
    )


def plot_feature_importance(importance: pd.DataFrame) -> None:
    plotted = importance.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plotted["feature"], plotted["importance"], color="#3f6f5f")
    ax.set_title("Factors-Only Matchup Model Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    model, features, x, y, meta, metrics = train_model()
    predictions_2026, summary_2026 = predict_2026(model, features)
    importance = feature_importance_frame(model, features)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    valid_mask = meta["season"].between(2017, 2025)
    valid_meta = meta.loc[valid_mask].copy()
    valid_meta["pred_team_a_win_prob"] = model.predict_proba(x.loc[valid_mask, features])[:, 1]
    valid_meta["pred_team_a_win"] = (valid_meta["pred_team_a_win_prob"] >= 0.5).astype(int)
    valid_meta.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    predictions_2026.to_csv(PREDICTIONS_2026_PATH, index=False)
    SUMMARY_2026_PATH.write_text(json.dumps(summary_2026, indent=2))
    importance.to_csv(TOP_FEATURES_PATH, index=False)
    plot_feature_importance(importance)

    print(json.dumps({"validation": metrics["metrics"]["validation"], "2026": summary_2026}, indent=2))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {PREDICTIONS_2026_PATH}")
    print(f"Wrote {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
