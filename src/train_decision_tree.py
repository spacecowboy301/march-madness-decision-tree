from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .team_names import normalize_team_name


RAW_DIR = Path("data/raw")
KENPOM_PATH = Path("data/processed/kenpom_team_features.csv")
MODEL_PATH = Path("models/decision_tree_march_madness.joblib")
PREDICTIONS_PATH = Path("reports/validation_predictions.csv")
METRICS_PATH = Path("reports/metrics.json")


def load_tournament_games() -> pd.DataFrame:
    teams_path = RAW_DIR / "MTeams.csv"
    results_path = RAW_DIR / "MNCAATourneyCompactResults.csv"
    missing = [str(path) for path in [teams_path, results_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing NCAA data files. Download the Kaggle men's March Machine Learning Mania files "
            f"and place these in data/raw/: {', '.join(missing)}"
        )

    teams = pd.read_csv(teams_path)
    results = pd.read_csv(results_path)
    team_names = teams.set_index("TeamID")["TeamName"].to_dict()

    games = results[["Season", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()
    games["winner"] = games["WTeamID"].map(team_names)
    games["loser"] = games["LTeamID"].map(team_names)
    games["winner_norm"] = games["winner"].map(normalize_team_name)
    games["loser_norm"] = games["loser"].map(normalize_team_name)
    return games.rename(columns={"Season": "season"})


def load_kenpom_features() -> pd.DataFrame:
    if not KENPOM_PATH.exists():
        raise FileNotFoundError(
            f"Missing {KENPOM_PATH}. Run `python -m src.kenpom_scraper --start-year 2002 --end-year 2025` first."
        )
    kenpom = pd.read_csv(KENPOM_PATH)
    kenpom["team_norm"] = kenpom["team_norm"].fillna(kenpom["team"].map(normalize_team_name))
    return kenpom


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"season", "team", "team_norm"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            df[col] = series
            cols.append(col)
    return cols


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
    full = full.drop(columns=["winner_season", "loser_season"], errors="ignore")
    return full


def build_examples(games_with_features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows = []
    labels = []
    meta = []
    for _, game in games_with_features.iterrows():
        if game[[f"winner_{col}" for col in feature_cols]].isna().all() or game[[f"loser_{col}" for col in feature_cols]].isna().all():
            continue
        for team_a_prefix, team_b_prefix, label in [("winner", "loser", 1), ("loser", "winner", 0)]:
            row = {}
            for col in feature_cols:
                a = game.get(f"{team_a_prefix}_{col}")
                b = game.get(f"{team_b_prefix}_{col}")
                row[f"{col}_diff"] = a - b if pd.notna(a) and pd.notna(b) else np.nan
                row[f"{col}_abs_diff"] = abs(a - b) if pd.notna(a) and pd.notna(b) else np.nan
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
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
    return metrics


def main() -> None:
    games = load_tournament_games()
    kenpom = load_kenpom_features()
    feature_cols = numeric_feature_columns(kenpom)
    merged = attach_features(games, kenpom, feature_cols)
    x, y, meta = build_examples(merged, feature_cols)

    train_mask = meta["season"].between(1997, 2016)
    valid_mask = meta["season"].between(2017, 2025)
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("No train or validation rows after joining tournament games to KenPom features.")

    observed_train_cols = x.loc[train_mask].notna().any(axis=0)
    x = x.loc[:, observed_train_cols]

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
    train_metrics = evaluate(model, x.loc[train_mask], y.loc[train_mask])
    valid_metrics = evaluate(model, x.loc[valid_mask], y.loc[valid_mask])
    actual_train_years = sorted(meta.loc[train_mask, "season"].unique().tolist())
    actual_validation_years = sorted(meta.loc[valid_mask, "season"].unique().tolist())

    valid_meta = meta.loc[valid_mask].copy()
    valid_meta["pred_team_a_win_prob"] = model.predict_proba(x.loc[valid_mask])[:, 1]
    valid_meta["pred_team_a_win"] = (valid_meta["pred_team_a_win_prob"] >= 0.5).astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(x.columns)}, MODEL_PATH)
    valid_meta.to_csv(PREDICTIONS_PATH, index=False)

    tree_text = export_text(model.named_steps["tree"], feature_names=list(x.columns))
    metrics = {
        "requested_train_years": "1997-2016",
        "requested_validation_years": "2017-2025",
        "actual_train_years": actual_train_years,
        "actual_validation_years": actual_validation_years,
        "rows": {
            "train": int(train_mask.sum()),
            "validation": int(valid_mask.sum()),
            "all_examples": int(len(x)),
        },
        "metrics": {"train": train_metrics, "validation": valid_metrics},
        "tree": tree_text,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics["metrics"], indent=2))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
