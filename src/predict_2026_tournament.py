from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .team_names import normalize_team_name


RESULTS_2026_PATH = Path("data/raw/ncaa_2026_tournament_results.csv")
KENPOM_2026_PATH = Path("data/raw/kenpom/kenpom_2026.csv")
MODEL_PATH = Path("models/decision_tree_march_madness.joblib")
PREDICTIONS_PATH = Path("reports/2026_tournament_predictions.csv")
SUMMARY_PATH = Path("reports/2026_tournament_summary.json")


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


def build_prediction_rows(games: pd.DataFrame, kenpom: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    feature_cols = numeric_feature_columns(kenpom)
    missing_model_inputs = set(model_features) - {f"{col}_diff" for col in feature_cols} - {f"{col}_abs_diff" for col in feature_cols}
    if missing_model_inputs:
        raise ValueError(f"2026 KenPom data cannot produce model inputs: {sorted(missing_model_inputs)[:10]}")

    rows = []
    for _, game in games.iterrows():
        winner = kenpom.loc[kenpom["team_norm"] == game["winner_norm"]]
        loser = kenpom.loc[kenpom["team_norm"] == game["loser_norm"]]
        if winner.empty or loser.empty:
            missing = []
            if winner.empty:
                missing.append(game["winner"])
            if loser.empty:
                missing.append(game["loser"])
            raise ValueError(f"Missing KenPom 2026 features for: {missing}")

        winner_features = winner.iloc[0]
        loser_features = loser.iloc[0]
        row = {}
        for col in feature_cols:
            a = winner_features.get(col)
            b = loser_features.get(col)
            row[f"{col}_diff"] = a - b if pd.notna(a) and pd.notna(b) else np.nan
            row[f"{col}_abs_diff"] = abs(a - b) if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)[model_features]


def main() -> None:
    games = pd.read_csv(RESULTS_2026_PATH)
    games["season"] = 2026
    games["winner_norm"] = games["winner"].map(normalize_team_name)
    games["loser_norm"] = games["loser"].map(normalize_team_name)

    kenpom = pd.read_csv(KENPOM_2026_PATH)
    kenpom["team_norm"] = kenpom["team"].map(normalize_team_name)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_features = artifact["features"]

    x_2026 = build_prediction_rows(games, kenpom, model_features)
    winner_prob = model.predict_proba(x_2026)[:, 1]
    predicted_winner_is_actual_winner = (winner_prob >= 0.5).astype(int)

    predictions = games[
        ["round", "date", "winner", "loser", "winner_score", "loser_score", "source"]
    ].copy()
    predictions["actual_winner"] = predictions["winner"]
    predictions["actual_loser"] = predictions["loser"]
    predictions["actual_margin"] = predictions["winner_score"] - predictions["loser_score"]
    predictions["pred_actual_winner_prob"] = winner_prob
    predictions["predicted_winner"] = np.where(winner_prob >= 0.5, predictions["winner"], predictions["loser"])
    predictions["prediction_correct"] = predicted_winner_is_actual_winner.astype(bool)

    y_true = np.ones(len(predictions), dtype=int)
    summary = {
        "season": 2026,
        "games": int(len(predictions)),
        "correct": int(predictions["prediction_correct"].sum()),
        "accuracy": float(accuracy_score(y_true, predicted_winner_is_actual_winner)),
        "brier": float(brier_score_loss(y_true, winner_prob)),
        "log_loss": float(log_loss(y_true, winner_prob, labels=[0, 1])),
        "roc_auc": None,
        "note": "Predictions are scored from actual-winner orientation, so ROC AUC is undefined because y_true contains only wins.",
        "by_round": predictions.groupby("round")["prediction_correct"].agg(["sum", "count", "mean"]).to_dict("index"),
    }

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
