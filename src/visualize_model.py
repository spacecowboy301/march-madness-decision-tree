from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .predict_2026_tournament import (
    KENPOM_2026_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    RESULTS_2026_PATH,
    build_prediction_rows,
)
from .team_names import normalize_team_name


REPORTS_DIR = Path("reports")
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.png"
ROUND_ACCURACY_PATH = REPORTS_DIR / "2026_accuracy_by_round.png"
CONFIDENCE_PATH = REPORTS_DIR / "2026_prediction_confidence.png"
BRACKET_HTML_PATH = REPORTS_DIR / "2026_bracket_predictions.html"
EXPLANATIONS_PATH = REPORTS_DIR / "2026_prediction_explanations.csv"

ROUND_ORDER = [
    "First Four",
    "First Round",
    "Second Round",
    "Sweet 16",
    "Elite Eight",
    "Final Four",
    "Championship",
]


def clean_feature_name(name: str) -> str:
    text = name.removeprefix("eff_")
    text = text.replace("_abs_diff", " abs diff").replace("_diff", " diff")
    text = text.replace("_", " ")
    text = text.replace("netrtg", "net rating").replace("ortg", "off rating").replace("drtg", "def rating")
    text = text.replace("adjt", "tempo").replace("ncsos", "non-conf SOS").replace("sos", "SOS")
    return text.title().replace("Sos", "SOS")


def load_model_and_data():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_features = artifact["features"]

    games = pd.read_csv(RESULTS_2026_PATH)
    games["season"] = 2026
    games["winner_norm"] = games["winner"].map(normalize_team_name)
    games["loser_norm"] = games["loser"].map(normalize_team_name)

    kenpom = pd.read_csv(KENPOM_2026_PATH)
    kenpom["team_norm"] = kenpom["team"].map(normalize_team_name)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    x_2026 = build_prediction_rows(games, kenpom, model_features)
    return model, model_features, games, kenpom, predictions, x_2026


def feature_importance_frame(model, model_features: list[str]) -> pd.DataFrame:
    tree = model.named_steps["tree"]
    return (
        pd.DataFrame({"feature": model_features, "importance": tree.feature_importances_})
        .assign(label=lambda df: df["feature"].map(clean_feature_name))
        .sort_values("importance", ascending=False)
    )


def plot_feature_importance(model, model_features: list[str], top_n: int = 16) -> Path:
    importance = feature_importance_frame(model, model_features)
    importance = importance[importance["importance"] > 0].head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance["label"], importance["importance"], color="#356f8f")
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
    plt.close(fig)
    return FEATURE_IMPORTANCE_PATH


def plot_round_accuracy(predictions: pd.DataFrame) -> Path:
    by_round = (
        predictions.assign(round=pd.Categorical(predictions["round"], ROUND_ORDER, ordered=True))
        .groupby("round", observed=True)["prediction_correct"]
        .mean()
        .reindex(ROUND_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_round.index.astype(str), by_round.values, color="#5f8f5b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("2026 Prediction Accuracy By Round")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(by_round.values):
        ax.text(idx, value + 0.025, f"{value:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROUND_ACCURACY_PATH, dpi=180)
    plt.close(fig)
    return ROUND_ACCURACY_PATH


def plot_prediction_confidence(predictions: pd.DataFrame) -> Path:
    confidence = predictions["pred_actual_winner_prob"].where(
        predictions["predicted_winner"].eq(predictions["winner"]),
        1 - predictions["pred_actual_winner_prob"],
    )
    colors = np.where(predictions["prediction_correct"], "#356f8f", "#b6544f")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(confidence, predictions["actual_margin"], c=colors, alpha=0.82)
    ax.axvline(0.5, color="#333333", linewidth=1, linestyle="--")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Model Confidence In Its Pick")
    ax.set_ylabel("Actual Winner Margin")
    ax.set_title("2026 Pick Confidence vs Actual Margin")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(CONFIDENCE_PATH, dpi=180)
    plt.close(fig)
    return CONFIDENCE_PATH


def decision_rules_for_row(model, row: pd.Series, model_features: list[str]) -> str:
    imputer = model.named_steps["imputer"]
    tree = model.named_steps["tree"]
    transformed = imputer.transform(pd.DataFrame([row], columns=model_features))
    node_indicator = tree.decision_path(transformed)
    leaf_id = tree.apply(transformed)[0]
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    rules = []
    for node_id in node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]:
        if node_id == leaf_id:
            continue
        feature_idx = feature[node_id]
        feature_name = model_features[feature_idx]
        value = transformed[0, feature_idx]
        op = "<=" if value <= threshold[node_id] else ">"
        rules.append(f"{clean_feature_name(feature_name)} {op} {threshold[node_id]:.2f} (value {value:.2f})")
    return " | ".join(rules)


def top_matchup_drivers(row: pd.Series, predicted_winner: str, actual_winner: str, actual_loser: str) -> str:
    # Express the biggest model-known gaps from the predicted winner's point of view.
    sign = 1 if predicted_winner == actual_winner else -1
    candidates = [
        "eff_rk_diff",
        "eff_netrtg_diff",
        "eff_ortg_diff",
        "eff_drtg_diff",
        "eff_adjt_diff",
        "eff_luck_diff",
        "eff_sos_netrtg_diff",
        "eff_ncsos_netrtg_diff",
    ]
    bits = []
    for feature in candidates:
        if feature in row and pd.notna(row[feature]):
            value = row[feature] * sign
            bits.append((abs(value), feature, value))
    bits.sort(reverse=True)
    return "; ".join(f"{clean_feature_name(feature)} {value:+.2f}" for _, feature, value in bits[:3])


def build_explanations(model, model_features: list[str], predictions: pd.DataFrame, x_2026: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, pred in predictions.iterrows():
        row = x_2026.iloc[idx]
        predicted_prob = (
            pred["pred_actual_winner_prob"]
            if pred["predicted_winner"] == pred["winner"]
            else 1 - pred["pred_actual_winner_prob"]
        )
        rows.append(
            {
                "round": pred["round"],
                "matchup": f"{pred['winner']} vs {pred['loser']}",
                "actual_winner": pred["winner"],
                "predicted_winner": pred["predicted_winner"],
                "predicted_winner_probability": predicted_prob,
                "prediction_correct": pred["prediction_correct"],
                "top_matchup_drivers": top_matchup_drivers(row, pred["predicted_winner"], pred["winner"], pred["loser"]),
                "decision_tree_rules": decision_rules_for_row(model, row, model_features),
            }
        )
    explanations = pd.DataFrame(rows)
    explanations.to_csv(EXPLANATIONS_PATH, index=False)
    return explanations


def build_bracket_html(predictions: pd.DataFrame, explanations: pd.DataFrame) -> Path:
    merged = predictions.copy()
    merged["why"] = explanations["top_matchup_drivers"]
    merged["tree_rules"] = explanations["decision_tree_rules"]
    merged["pick_prob"] = explanations["predicted_winner_probability"]

    round_columns = []
    for round_name in ROUND_ORDER:
        games = merged[merged["round"] == round_name]
        cards = []
        for _, game in games.iterrows():
            correct = bool(game["prediction_correct"])
            cls = "correct" if correct else "miss"
            cards.append(
                f"""
                <article class="game {cls}">
                  <div class="roundline">{html.escape(str(game['date']))}</div>
                  <div class="teams">
                    <div><strong>{html.escape(str(game['winner']))}</strong> {int(game['winner_score'])}</div>
                    <div>{html.escape(str(game['loser']))} {int(game['loser_score'])}</div>
                  </div>
                  <div class="pick">Model pick: <strong>{html.escape(str(game['predicted_winner']))}</strong> ({game['pick_prob']:.0%})</div>
                  <div class="why">{html.escape(str(game['why']))}</div>
                  <details><summary>Decision path</summary><p>{html.escape(str(game['tree_rules']))}</p></details>
                </article>
                """
            )
        round_columns.append(
            f"""
            <section class="round">
              <h2>{html.escape(round_name)}</h2>
              {''.join(cards)}
            </section>
            """
        )

    payload = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>2026 March Madness Model Bracket</title>
      <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f4ef; color: #20242a; }}
        header {{ padding: 24px 28px 12px; }}
        h1 {{ margin: 0 0 8px; font-size: 28px; }}
        .note {{ max-width: 980px; color: #555; line-height: 1.45; }}
        .bracket {{ display: grid; grid-template-columns: repeat(7, minmax(230px, 1fr)); gap: 14px; padding: 16px 24px 32px; overflow-x: auto; }}
        .round {{ min-width: 230px; }}
        h2 {{ font-size: 15px; margin: 0 0 10px; border-bottom: 2px solid #222; padding-bottom: 6px; }}
        .game {{ background: white; border: 1px solid #d6d1c8; border-left: 5px solid #356f8f; border-radius: 8px; padding: 10px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
        .game.miss {{ border-left-color: #b6544f; }}
        .roundline {{ font-size: 11px; color: #777; margin-bottom: 5px; }}
        .teams {{ font-size: 13px; line-height: 1.45; }}
        .pick {{ margin-top: 8px; font-size: 12px; }}
        .why {{ margin-top: 6px; font-size: 11px; color: #555; line-height: 1.35; }}
        details {{ margin-top: 6px; font-size: 11px; }}
        summary {{ cursor: pointer; color: #333; }}
        details p {{ color: #555; line-height: 1.35; }}
      </style>
    </head>
    <body>
      <header>
        <h1>2026 March Madness Decision Tree Bracket</h1>
        <p class="note">Each card shows the actual matchup, model pick, confidence in that pick, top matchup gaps from the pick's point of view, and the decision-tree path used by the model. Blue cards were correct; red cards were misses.</p>
      </header>
      <main class="bracket">{''.join(round_columns)}</main>
    </body>
    </html>
    """
    BRACKET_HTML_PATH.write_text(payload)
    return BRACKET_HTML_PATH


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model, model_features, _games, _kenpom, predictions, x_2026 = load_model_and_data()
    explanations = build_explanations(model, model_features, predictions, x_2026)
    paths = {
        "feature_importance": str(plot_feature_importance(model, model_features)),
        "round_accuracy": str(plot_round_accuracy(predictions)),
        "confidence": str(plot_prediction_confidence(predictions)),
        "explanations": str(EXPLANATIONS_PATH),
        "bracket": str(build_bracket_html(predictions, explanations)),
    }
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
