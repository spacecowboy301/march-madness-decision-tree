from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .team_names import normalize_team_name


RAW_RESULTS_PATH = Path("data/raw/MRegularSeasonDetailedResults.csv")
TEAMS_PATH = Path("data/raw/MTeams.csv")
OUTPUT_PATH = Path("data/processed/pretournament_team_factors.csv")
AUDIT_PATH = Path("reports/pretournament_feature_audit.json")
KENPOM_REFERENCE_PATH = Path("data/processed/kenpom_team_features.csv")
MIN_COMPLETE_DAY = 125

BOX_COLUMNS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def safe_rate(numerator: pd.Series, denominator: pd.Series, scale: float = 100.0) -> pd.Series:
    return scale * numerator.div(denominator.replace(0, np.nan))


def team_game_rows(results: pd.DataFrame) -> pd.DataFrame:
    common = results[["Season", "DayNum"]].rename(columns={"Season": "season", "DayNum": "day_num"})
    frames = []
    for side, opponent in [("W", "L"), ("L", "W")]:
        frame = common.copy()
        frame["team_id"] = results[f"{side}TeamID"]
        frame["opponent_id"] = results[f"{opponent}TeamID"]
        for col in BOX_COLUMNS:
            frame[f"team_{col.lower()}"] = results[f"{side}{col}"]
            frame[f"opp_{col.lower()}"] = results[f"{opponent}{col}"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_team_seasons(results: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    season_max_day = results.groupby("Season")["DayNum"].max()
    complete_seasons = season_max_day[season_max_day >= MIN_COMPLETE_DAY].index
    eligible = results[results["Season"].isin(complete_seasons)].copy()
    rows = team_game_rows(eligible)

    sum_cols = [
        col
        for col in rows.columns
        if col.startswith(("team_", "opp_")) and col not in {"team_id", "opponent_id"}
    ]
    grouped = rows.groupby(["season", "team_id"], as_index=False).agg(
        games=("day_num", "size"),
        data_through_day=("day_num", "max"),
        **{col: (col, "sum") for col in sum_cols},
    )

    poss_team = grouped["team_fga"] + 0.475 * grouped["team_fta"] - grouped["team_or"] + grouped["team_to"]
    poss_opp = grouped["opp_fga"] + 0.475 * grouped["opp_fta"] - grouped["opp_or"] + grouped["opp_to"]
    grouped["possessions"] = (poss_team + poss_opp) / 2.0

    grouped["ff_off_efg"] = safe_rate(grouped["team_fgm"] + 0.5 * grouped["team_fgm3"], grouped["team_fga"])
    grouped["ff_off_to"] = safe_rate(grouped["team_to"], grouped["possessions"])
    grouped["ff_off_or"] = safe_rate(grouped["team_or"], grouped["team_or"] + grouped["opp_dr"])
    grouped["ff_off_ftrate"] = safe_rate(grouped["team_fta"], grouped["team_fga"])
    grouped["ff_def_efg"] = safe_rate(grouped["opp_fgm"] + 0.5 * grouped["opp_fgm3"], grouped["opp_fga"])
    grouped["ff_def_to"] = safe_rate(grouped["opp_to"], grouped["possessions"])
    grouped["ff_def_or"] = safe_rate(grouped["opp_or"], grouped["opp_or"] + grouped["team_dr"])
    grouped["ff_def_ftrate"] = safe_rate(grouped["opp_fta"], grouped["opp_fga"])

    team_2pm = grouped["team_fgm"] - grouped["team_fgm3"]
    team_2pa = grouped["team_fga"] - grouped["team_fga3"]
    opp_2pm = grouped["opp_fgm"] - grouped["opp_fgm3"]
    opp_2pa = grouped["opp_fga"] - grouped["opp_fga3"]

    grouped["misc_off_3p"] = safe_rate(grouped["team_fgm3"], grouped["team_fga3"])
    grouped["misc_off_2p"] = safe_rate(team_2pm, team_2pa)
    grouped["misc_off_ft"] = safe_rate(grouped["team_ftm"], grouped["team_fta"])
    grouped["misc_off_blk"] = safe_rate(grouped["opp_blk"], team_2pa)
    grouped["misc_off_stl"] = safe_rate(grouped["opp_stl"], grouped["possessions"])
    grouped["misc_off_nst"] = safe_rate((grouped["team_to"] - grouped["opp_stl"]).clip(lower=0), grouped["possessions"])
    grouped["misc_off_a"] = safe_rate(grouped["team_ast"], grouped["team_fgm"])
    grouped["misc_off_3pa"] = safe_rate(grouped["team_fga3"], grouped["team_fga"])

    grouped["misc_def_3p"] = safe_rate(grouped["opp_fgm3"], grouped["opp_fga3"])
    grouped["misc_def_2p"] = safe_rate(opp_2pm, opp_2pa)
    grouped["misc_def_ft"] = safe_rate(grouped["opp_ftm"], grouped["opp_fta"])
    grouped["misc_def_blk"] = safe_rate(grouped["team_blk"], opp_2pa)
    grouped["misc_def_stl"] = safe_rate(grouped["team_stl"], grouped["possessions"])
    grouped["misc_def_nst"] = safe_rate((grouped["opp_to"] - grouped["team_stl"]).clip(lower=0), grouped["possessions"])
    grouped["misc_def_a"] = safe_rate(grouped["opp_ast"], grouped["opp_fgm"])
    grouped["misc_def_3pa"] = safe_rate(grouped["opp_fga3"], grouped["opp_fga"])

    names = teams[["TeamID", "TeamName"]].rename(columns={"TeamID": "team_id", "TeamName": "team"})
    grouped = grouped.merge(names, on="team_id", how="left")
    grouped["snapshot_status"] = "confirmed_pre_tournament"
    grouped["source"] = "NCAA regular-season detailed box scores"
    grouped["formula_version"] = "four_factors_misc_v1"

    feature_cols = [col for col in grouped.columns if col.startswith(("ff_", "misc_"))]
    keep = [
        "season",
        "team_id",
        "team",
        "games",
        "data_through_day",
        "snapshot_status",
        "source",
        "formula_version",
        *feature_cols,
    ]
    return grouped[keep].sort_values(["season", "team_id"]).reset_index(drop=True)


def build_features() -> tuple[pd.DataFrame, dict]:
    results = pd.read_csv(RAW_RESULTS_PATH)
    teams = pd.read_csv(TEAMS_PATH)
    features = aggregate_team_seasons(results, teams)
    available = sorted(features["season"].unique().tolist())
    raw_max_days = results.groupby("Season")["DayNum"].max().astype(int).to_dict()
    excluded = {str(year): day for year, day in raw_max_days.items() if year not in available}
    audit = {
        "timing_status": "confirmed_pre_tournament",
        "source": str(RAW_RESULTS_PATH),
        "formula_note": "Aggregated only NCAA regular-season detailed box scores; NCAA tournament games are a separate file.",
        "minimum_complete_regular_season_day": MIN_COMPLETE_DAY,
        "available_seasons": available,
        "excluded_incomplete_seasons": excluded,
        "team_season_rows": int(len(features)),
        "feature_columns": [col for col in features.columns if col.startswith(("ff_", "misc_"))],
    }
    if KENPOM_REFERENCE_PATH.exists():
        reference = pd.read_csv(KENPOM_REFERENCE_PATH, low_memory=False)
        comparison = features.copy()
        comparison["team_norm"] = comparison["team"].map(normalize_team_name)
        joined = comparison.merge(reference, on=["season", "team_norm"], suffixes=("_pre", "_kenpom"))
        correlations = {}
        for col in [name for name in features.columns if name.startswith("ff_")]:
            left = pd.to_numeric(joined[f"{col}_pre"], errors="coerce")
            right = pd.to_numeric(joined[f"{col}_kenpom"], errors="coerce")
            correlations[col] = float(left.corr(right))
        audit["kenpom_formula_reference"] = {
            "note": "Season-end KenPom values are used only to verify formula agreement, never as model inputs.",
            "joined_team_seasons": int(len(joined)),
            "pearson_correlations": correlations,
        }
    return features, audit


def main() -> None:
    features, audit = build_features()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2))
    print(f"Wrote {OUTPUT_PATH} with {len(features):,} team-seasons")
    print(f"Confirmed pre-tournament seasons: {audit['available_seasons'][0]}-{audit['available_seasons'][-1]}")
    if audit["excluded_incomplete_seasons"]:
        print(f"Excluded incomplete seasons: {audit['excluded_incomplete_seasons']}")


if __name__ == "__main__":
    main()
