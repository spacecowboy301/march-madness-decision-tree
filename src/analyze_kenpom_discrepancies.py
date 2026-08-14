from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .analyze_factor_importance import RANDOM_STATE, metric_dict
from .factor_matchups import build_matchup_dataset
from .team_names import normalize_team_name


KENPOM_PATH = Path("data/processed/kenpom_team_features.csv")
MODEL_PREDICTIONS_PATH = Path("reports/power_matchup_validation_predictions.csv")
OUTPUT_PATH = Path("reports/kenpom_discrepancy_matchups.csv")
METRICS_PATH = Path("reports/kenpom_discrepancy_metrics.json")
SUMMARY_PATH = Path("reports/kenpom_discrepancy_summary.md")
CHART_PATH = Path("reports/figures/model_vs_kenpom_discrepancies.png")


KENPOM_NAME_CANDIDATES = {
    "abilene chr": ("abilene christian",),
    "american univ": ("american",),
    "ark little rock": ("arkansas little rock", "little rock"),
    "ark pine bluff": ("arkansas pine bluff",),
    "boston univ": ("boston university",),
    "c michigan": ("central michigan",),
    "cs bakersfield": ("cal st. bakersfield",),
    "cs fullerton": ("cal st. fullerton",),
    "cs northridge": ("cal st. northridge",),
    "central conn": ("central connecticut",),
    "coastal car": ("coastal carolina",),
    "col charleston": ("charleston",),
    "e kentucky": ("eastern kentucky",),
    "e washington": ("eastern washington",),
    "etsu": ("east tennessee st.",),
    "f dickinson": ("fairleigh dickinson",),
    "fl atlantic": ("florida atlantic",),
    "fl gulf coast": ("florida gulf coast",),
    "g washington": ("george washington",),
    "il chicago": ("uic",),
    "kennesaw": ("kennesaw st.",),
    "kent": ("kent st.",),
    "ms valley st.": ("mississippi valley st.",),
    "mtsu": ("middle tennessee", "middle tennessee st."),
    "monmouth nj": ("monmouth",),
    "mt st. marys": ("mount st. marys",),
    "n colorado": ("northern colorado",),
    "n dakota st.": ("north dakota st.",),
    "n kentucky": ("northern kentucky",),
    "nc a&t": ("north carolina a&t",),
    "nc central": ("north carolina central",),
    "northwestern la": ("northwestern st.",),
    "prairie view": ("prairie view a&m",),
    "s carolina st.": ("south carolina st.",),
    "s dakota st.": ("south dakota st.",),
    "s illinois": ("southern illinois",),
    "se louisiana": ("southeastern louisiana",),
    "se missouri st.": ("southeast missouri st.", "southeast missouri"),
    "sf austin": ("stephen f. austin",),
    "suny albany": ("albany",),
    "southern univ": ("southern",),
    "st. josephs pa": ("saint josephs",),
    "st. louis": ("saint louis",),
    "st. marys ca": ("saint mary's",),
    "st. peters": ("saint peters",),
    "tam c. christi": ("texas a&m corpus chris",),
    "tx southern": ("texas southern",),
    "troy": ("troy st.",),
    "ut san antonio": ("utsa", "texas san antonio"),
    "w michigan": ("western michigan",),
    "wi green bay": ("green bay",),
    "wi milwaukee": ("milwaukee",),
    "wku": ("western kentucky",),
}


def candidate_names(team: str) -> tuple[str, ...]:
    normalized = normalize_team_name(team)
    return (normalized, *KENPOM_NAME_CANDIDATES.get(normalized, ()))


def load_kenpom_lookup() -> dict[tuple[int, str], float]:
    if not KENPOM_PATH.exists():
        raise FileNotFoundError(f"Missing {KENPOM_PATH}; run the optional KenPom scraper first.")
    kenpom = pd.read_csv(KENPOM_PATH)
    kenpom["team_norm"] = kenpom["team_norm"].fillna(kenpom["team"].map(normalize_team_name))
    kenpom["eff_netrtg"] = pd.to_numeric(kenpom["eff_netrtg"], errors="coerce")
    observed = kenpom.dropna(subset=["season", "team_norm", "eff_netrtg"])
    return {
        (int(row.season), str(row.team_norm)): float(row.eff_netrtg)
        for row in observed.itertuples(index=False)
    }


def lookup_rating(lookup: dict[tuple[int, str], float], season: int, team: str) -> float:
    for candidate in candidate_names(team):
        value = lookup.get((int(season), candidate))
        if value is not None:
            return value
    return np.nan


def attach_kenpom_ratings(frame: pd.DataFrame, lookup: dict[tuple[int, str], float]) -> pd.DataFrame:
    output = frame.copy()
    output["kenpom_netrtg_a"] = [
        lookup_rating(lookup, season, team)
        for season, team in zip(output["season"], output["team_a"])
    ]
    output["kenpom_netrtg_b"] = [
        lookup_rating(lookup, season, team)
        for season, team in zip(output["season"], output["team_b"])
    ]
    output["kenpom_netrtg_gap"] = output["kenpom_netrtg_a"] - output["kenpom_netrtg_b"]
    return output


def fit_kenpom_probability_benchmark(
    lookup: dict[tuple[int, str], float]
) -> tuple[LogisticRegression, dict[str, object]]:
    _x, y, meta, _families = build_matchup_dataset()
    training = attach_kenpom_ratings(meta.loc[meta["season"] <= 2016], lookup)
    training["actual_team_a_win"] = y.loc[training.index].to_numpy()
    training = training.dropna(subset=["kenpom_netrtg_gap"])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    model.fit(training[["kenpom_netrtg_gap"]], training["actual_team_a_win"])
    details = {
        "training_games": int(len(training)),
        "training_seasons": sorted(training["season"].unique().astype(int).tolist()),
        "intercept": float(model.intercept_[0]),
        "net_rating_gap_coefficient": float(model.coef_[0, 0]),
    }
    return model, details


def matchup_label(row: pd.Series) -> str:
    upset = " upset" if bool(row["actual_upset"]) else ""
    seeds = ""
    if pd.notna(row["winner_seed"]) and pd.notna(row["loser_seed"]):
        seeds = f" ({int(row['winner_seed'])} over {int(row['loser_seed'])}{upset})"
    return f"{int(row['season'])}: {row['actual_winner']} over {row['actual_loser']}{seeds}"


def report_line(row: pd.Series) -> str:
    return (
        f"- {matchup_label(row)}: model {row['model_actual_winner_probability']:.1%}, "
        f"KenPom-derived {row['kenpom_actual_winner_probability']:.1%}, "
        f"gap {row['actual_winner_probability_gap_pp']:+.1f} points."
    )


def plot_discrepancies(comparison: pd.DataFrame) -> None:
    model_calls = comparison.loc[comparison["model_correct"]].nlargest(
        7, "actual_winner_probability_gap_pp"
    )
    model_misses = comparison.loc[~comparison["model_correct"]].nsmallest(
        7, "actual_winner_probability_gap_pp"
    )
    plotted = pd.concat([model_misses, model_calls]).copy()
    plotted["label"] = plotted.apply(
        lambda row: f"{int(row.season)} {row.actual_winner} over {row.actual_loser}", axis=1
    )
    colors = np.where(plotted["model_correct"], "#287271", "#c44e52")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(plotted["label"], plotted["actual_winner_probability_gap_pp"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_title("Largest Model vs KenPom-Derived Probability Gaps")
    ax.set_xlabel("Model minus KenPom-derived probability for the actual winner (points)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=190)
    plt.close(fig)


def main() -> None:
    lookup = load_kenpom_lookup()
    benchmark, benchmark_details = fit_kenpom_probability_benchmark(lookup)
    predictions = pd.read_csv(MODEL_PREDICTIONS_PATH)
    comparison = attach_kenpom_ratings(predictions, lookup)
    total_games = len(comparison)
    comparison = comparison.dropna(subset=["kenpom_netrtg_gap"]).reset_index(drop=True)
    comparison["kenpom_team_a_probability"] = benchmark.predict_proba(
        comparison[["kenpom_netrtg_gap"]]
    )[:, 1]
    comparison["model_team_a_probability"] = comparison["combined_probability"]
    actual = comparison["actual_team_a_win"].astype(int).to_numpy()
    comparison["model_actual_winner_probability"] = np.where(
        actual == 1,
        comparison["model_team_a_probability"],
        1.0 - comparison["model_team_a_probability"],
    )
    comparison["kenpom_actual_winner_probability"] = np.where(
        actual == 1,
        comparison["kenpom_team_a_probability"],
        1.0 - comparison["kenpom_team_a_probability"],
    )
    comparison["actual_winner_probability_gap_pp"] = 100.0 * (
        comparison["model_actual_winner_probability"]
        - comparison["kenpom_actual_winner_probability"]
    )
    comparison["model_correct"] = (
        (comparison["model_team_a_probability"] >= 0.5).astype(int) == actual
    )
    comparison["kenpom_correct"] = (
        (comparison["kenpom_team_a_probability"] >= 0.5).astype(int) == actual
    )
    comparison["model_predicted_winner"] = np.where(
        comparison["model_team_a_probability"] >= 0.5,
        comparison["team_a"],
        comparison["team_b"],
    )
    comparison["kenpom_predicted_winner"] = np.where(
        comparison["kenpom_team_a_probability"] >= 0.5,
        comparison["team_a"],
        comparison["team_b"],
    )
    comparison["comparison_category"] = np.select(
        [
            comparison["model_correct"] & ~comparison["kenpom_correct"],
            ~comparison["model_correct"] & comparison["kenpom_correct"],
            comparison["model_correct"] & comparison["kenpom_correct"],
        ],
        ["model_only_correct", "kenpom_only_correct", "both_correct"],
        default="both_incorrect",
    )

    model_metrics = metric_dict(actual, comparison["model_team_a_probability"])
    kenpom_metrics = metric_dict(actual, comparison["kenpom_team_a_probability"])
    metrics = {
        "benchmark_label": "KenPom-derived probability benchmark; not an official KenPom game probability",
        "benchmark_timing": "retrospective season-end KenPom snapshots",
        "timing_warning": "KenPom snapshots may include tournament games; use this only as an exploratory discrepancy analysis.",
        "requested_validation_games": int(total_games),
        "matched_validation_games": int(len(comparison)),
        "coverage": float(len(comparison) / total_games),
        "benchmark_fit": benchmark_details,
        "held_out_metrics": {
            "two_channel_model": model_metrics,
            "kenpom_derived_benchmark": kenpom_metrics,
        },
        "disagreement_counts": comparison["comparison_category"].value_counts().to_dict(),
    }

    model_only = comparison.loc[comparison["comparison_category"] == "model_only_correct"].nlargest(
        8, "actual_winner_probability_gap_pp"
    )
    higher_and_correct = comparison.loc[
        comparison["model_correct"]
        & comparison["kenpom_correct"]
        & comparison["actual_winner_probability_gap_pp"].gt(0)
    ].nlargest(8, "actual_winner_probability_gap_pp")
    kenpom_only = comparison.loc[comparison["comparison_category"] == "kenpom_only_correct"].nsmallest(
        8, "actual_winner_probability_gap_pp"
    )
    confident_misses = comparison.loc[~comparison["model_correct"]].nsmallest(
        8, "model_actual_winner_probability"
    )

    summary = [
        "# Notable Model vs KenPom-Derived Discrepancies",
        "",
        "## Scope And Caveat",
        "",
        "This is a retrospective comparison against a probability benchmark calibrated from KenPom adjusted net-rating gaps. It is **not an official KenPom game probability**. The cached historical KenPom pages are season-end snapshots and may include tournament games, while this project's model features are confirmed pre-tournament. Treat the matchup rankings as exploratory examples, not a fair head-to-head leaderboard.",
        "",
        f"Coverage: **{len(comparison)} of {total_games} held-out games ({len(comparison) / total_games:.1%})**.",
        "",
        "## Overall Comparison",
        "",
        f"- Two-channel model: {model_metrics['accuracy']:.1%} accuracy, {model_metrics['roc_auc']:.3f} ROC AUC, {model_metrics['log_loss']:.3f} log loss.",
        f"- KenPom-derived benchmark: {kenpom_metrics['accuracy']:.1%} accuracy, {kenpom_metrics['roc_auc']:.3f} ROC AUC, {kenpom_metrics['log_loss']:.3f} log loss.",
        f"- Model-only correct calls: {int((comparison['comparison_category'] == 'model_only_correct').sum())}; KenPom-only correct calls: {int((comparison['comparison_category'] == 'kenpom_only_correct').sum())}.",
        "",
        "## Biggest Model-Only Correct Calls",
        "",
    ]
    summary.extend(report_line(row) for _, row in model_only.iterrows())
    summary.extend(
        [
            "",
            "## Correct Calls Where The Model Assigned More Win Probability",
            "",
        ]
    )
    summary.extend(report_line(row) for _, row in higher_and_correct.iterrows())
    summary.extend(["", "## Model Misses Where KenPom-Derived Was Correct", ""])
    summary.extend(report_line(row) for _, row in kenpom_only.iterrows())
    summary.extend(["", "## Most Confident Model Misses", ""])
    summary.extend(report_line(row) for _, row in confident_misses.iterrows())
    summary.extend(
        [
            "",
            "## Reading The Gaps",
            "",
            "A positive gap means this model assigned more probability to the eventual winner. A negative gap means the KenPom-derived benchmark did. The gap does not prove which basketball factor caused the result, and a single upset should not be treated as validation of a mechanism.",
        ]
    )

    OUTPUT_PATH.write_text(comparison.sort_values("actual_winner_probability_gap_pp", ascending=False).to_csv(index=False))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")
    plot_discrepancies(comparison)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
