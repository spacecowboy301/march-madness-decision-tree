from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from .team_names import normalize_team_name


RAW_RESULTS_PATH = Path("data/raw/MRegularSeasonDetailedResults.csv")
TEAMS_PATH = Path("data/raw/MTeams.csv")
OUTPUT_PATH = Path("data/processed/pretournament_team_factors.csv")
AUDIT_PATH = Path("reports/data_audit.json")
KENPOM_REFERENCE_PATH = Path("data/processed/kenpom_team_features.csv")
MIN_COMPLETE_DAY = 125
RECENT_GAMES = 10
PRIOR_GAMES = 6.0
RECENT_PRIOR_GAMES = 4.0
OPPONENT_ADJUSTMENT_ALPHA = 20.0

BOX_COLUMNS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

RATE_COLUMNS = {
    "efg": ("ff_off_efg", "ff_def_efg"),
    "turnover": ("ff_off_to", "ff_def_to"),
    "offensive_rebound": ("ff_off_or", "ff_def_or"),
    "free_throw_rate": ("ff_off_ftrate", "ff_def_ftrate"),
    "three_point": ("misc_off_3p", "misc_def_3p"),
    "two_point": ("misc_off_2p", "misc_def_2p"),
    "free_throw_pct": ("misc_off_ft", "misc_def_ft"),
    "block": ("misc_off_blk", "misc_def_blk"),
    "steal": ("misc_off_stl", "misc_def_stl"),
    "nonsteal_turnover": ("misc_off_nst", "misc_def_nst"),
    "assist": ("misc_off_a", "misc_def_a"),
    "three_point_rate": ("misc_off_3pa", "misc_def_3pa"),
}


def safe_rate(numerator: pd.Series, denominator: pd.Series, scale: float = 100.0) -> pd.Series:
    return scale * numerator.div(denominator.replace(0, np.nan))


def team_game_rows(results: pd.DataFrame) -> pd.DataFrame:
    common = results[["Season", "DayNum"]].rename(columns={"Season": "season", "DayNum": "day_num"})
    common["game_id"] = np.arange(len(results))
    frames = []
    for side, opponent in [("W", "L"), ("L", "W")]:
        frame = common.copy()
        frame["team_id"] = results[f"{side}TeamID"]
        frame["opponent_id"] = results[f"{opponent}TeamID"]
        frame["team_score"] = results[f"{side}Score"]
        frame["opp_score"] = results[f"{opponent}Score"]
        frame["won"] = float(side == "W")
        location = results["WLoc"].fillna("N")
        if side == "L":
            location = location.map({"H": "A", "A": "H", "N": "N"}).fillna("N")
        frame["location"] = location
        frame["location_value"] = frame["location"].map({"H": 1.0, "A": -1.0, "N": 0.0})
        for col in BOX_COLUMNS:
            frame[f"team_{col.lower()}"] = results[f"{side}{col}"]
            frame[f"opp_{col.lower()}"] = results[f"{opponent}{col}"]
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    return add_game_rates(rows)


def add_game_rates(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    team_2pm = rows["team_fgm"] - rows["team_fgm3"]
    team_2pa = rows["team_fga"] - rows["team_fga3"]
    opp_2pm = rows["opp_fgm"] - rows["opp_fgm3"]
    opp_2pa = rows["opp_fga"] - rows["opp_fga3"]
    team_poss = rows["team_fga"] + 0.475 * rows["team_fta"] - rows["team_or"] + rows["team_to"]
    opp_poss = rows["opp_fga"] + 0.475 * rows["opp_fta"] - rows["opp_or"] + rows["opp_to"]
    rows["game_possessions"] = (team_poss + opp_poss) / 2.0

    components = {
        "efg": (rows["team_fgm"] + 0.5 * rows["team_fgm3"], rows["team_fga"], rows["opp_fgm"] + 0.5 * rows["opp_fgm3"], rows["opp_fga"]),
        "turnover": (rows["team_to"], rows["game_possessions"], rows["opp_to"], rows["game_possessions"]),
        "offensive_rebound": (rows["team_or"], rows["team_or"] + rows["opp_dr"], rows["opp_or"], rows["opp_or"] + rows["team_dr"]),
        "free_throw_rate": (rows["team_fta"], rows["team_fga"], rows["opp_fta"], rows["opp_fga"]),
        "three_point": (rows["team_fgm3"], rows["team_fga3"], rows["opp_fgm3"], rows["opp_fga3"]),
        "two_point": (team_2pm, team_2pa, opp_2pm, opp_2pa),
        "free_throw_pct": (rows["team_ftm"], rows["team_fta"], rows["opp_ftm"], rows["opp_fta"]),
        "block": (rows["opp_blk"], team_2pa, rows["team_blk"], opp_2pa),
        "steal": (rows["opp_stl"], rows["game_possessions"], rows["team_stl"], rows["game_possessions"]),
        "nonsteal_turnover": ((rows["team_to"] - rows["opp_stl"]).clip(lower=0), rows["game_possessions"], (rows["opp_to"] - rows["team_stl"]).clip(lower=0), rows["game_possessions"]),
        "assist": (rows["team_ast"], rows["team_fgm"], rows["opp_ast"], rows["opp_fgm"]),
        "three_point_rate": (rows["team_fga3"], rows["team_fga"], rows["opp_fga3"], rows["opp_fga"]),
    }
    for name, (team_num, team_den, opp_num, opp_den) in components.items():
        rows[f"rate_{name}"] = team_num.div(team_den.replace(0, np.nan))
        rows[f"opportunity_{name}"] = team_den
        rows[f"opp_rate_{name}"] = opp_num.div(opp_den.replace(0, np.nan))
        rows[f"opp_opportunity_{name}"] = opp_den
    rows["points_per_100"] = 100.0 * rows["team_score"].div(rows["game_possessions"].replace(0, np.nan))
    return rows


def ridge_design(rows: pd.DataFrame, team_ids: np.ndarray) -> sparse.csr_matrix:
    team_index = {int(team_id): index for index, team_id in enumerate(team_ids)}
    row_index = np.arange(len(rows))
    team_columns = rows["team_id"].map(team_index).to_numpy()
    opponent_columns = len(team_ids) + rows["opponent_id"].map(team_index).to_numpy()
    location_column = np.full(len(rows), 2 * len(team_ids))
    matrix = sparse.coo_matrix(
        (
            np.concatenate([np.ones(len(rows)), np.ones(len(rows)), rows["location_value"].to_numpy()]),
            (
                np.concatenate([row_index, row_index, row_index]),
                np.concatenate([team_columns, opponent_columns, location_column]),
            ),
        ),
        shape=(len(rows), 2 * len(team_ids) + 1),
    )
    return matrix.tocsr()


def fit_adjusted_rate(
    design: sparse.csr_matrix,
    values: pd.Series,
    opportunities: pd.Series,
    team_ids: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    observed = values.notna() & opportunities.gt(0)
    weights = opportunities.loc[observed].to_numpy(dtype=float)
    weights = weights / max(float(np.mean(weights)), 1.0)
    model = Ridge(alpha=OPPONENT_ADJUSTMENT_ALPHA, solver="lsqr", fit_intercept=True)
    model.fit(design[observed.to_numpy()], values.loc[observed].to_numpy(dtype=float), sample_weight=weights)
    count = len(team_ids)
    offense = model.coef_[:count]
    defense = model.coef_[count : 2 * count]
    baseline = float(model.intercept_ + np.mean(offense) + np.mean(defense))
    offense_adjusted = baseline + offense - np.mean(offense)
    defense_adjusted = baseline + defense - np.mean(defense)
    return (
        dict(zip(team_ids.astype(int), 100.0 * offense_adjusted)),
        dict(zip(team_ids.astype(int), 100.0 * defense_adjusted)),
    )


def add_opponent_adjusted_features(rows: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    grouped = grouped.copy()
    adjusted_parts = []
    for season, season_rows in rows.groupby("season"):
        team_ids = np.sort(season_rows["team_id"].unique())
        design = ridge_design(season_rows, team_ids)
        season_output = pd.DataFrame({"season": season, "team_id": team_ids})
        for name, (off_col, def_col) in RATE_COLUMNS.items():
            offense, defense = fit_adjusted_rate(
                design,
                season_rows[f"rate_{name}"],
                season_rows[f"opportunity_{name}"],
                team_ids,
            )
            season_output[f"adj_{off_col}"] = season_output["team_id"].map(offense)
            season_output[f"adj_{def_col}"] = season_output["team_id"].map(defense)

        offense_efficiency, defense_efficiency = fit_adjusted_rate(
            design,
            season_rows["points_per_100"] / 100.0,
            season_rows["game_possessions"],
            team_ids,
        )
        season_output["power_off_eff"] = season_output["team_id"].map(offense_efficiency)
        season_output["power_def_eff"] = season_output["team_id"].map(defense_efficiency)
        season_output["power_rating"] = season_output["power_off_eff"] - season_output["power_def_eff"]
        adjusted_parts.append(season_output)
    adjusted = pd.concat(adjusted_parts, ignore_index=True)
    return grouped.merge(adjusted, on=["season", "team_id"], how="left", validate="one_to_one")


def add_shrinkage_form_and_volatility(rows: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    grouped = grouped.copy()
    recent = rows.sort_values(["season", "team_id", "day_num", "game_id"]).groupby(
        ["season", "team_id"], group_keys=False
    ).tail(RECENT_GAMES)
    grouped_index = grouped.set_index(["season", "team_id"])
    derived: dict[str, pd.Series] = {}

    for name, (off_col, def_col) in RATE_COLUMNS.items():
        for column, rate_column, opportunity_column in [
            (off_col, f"rate_{name}", f"opportunity_{name}"),
            (def_col, f"opp_rate_{name}", f"opp_opportunity_{name}"),
        ]:
            full = rows.groupby(["season", "team_id"]).apply(
                lambda frame: pd.Series(
                    {
                        "numerator": float((frame[rate_column] * frame[opportunity_column]).sum()),
                        "opportunities": float(frame[opportunity_column].sum()),
                        "mean_opportunities": float(frame[opportunity_column].mean()),
                    }
                ),
                include_groups=False,
            )
            season_numerator = full["numerator"].groupby(level=0).transform("sum")
            season_denominator = full["opportunities"].groupby(level=0).transform("sum")
            league_rate = season_numerator.div(season_denominator.replace(0, np.nan))
            prior = PRIOR_GAMES * full["mean_opportunities"].groupby(level=0).transform("median")
            shrunk = (full["numerator"] + prior * league_rate).div(full["opportunities"] + prior)
            reliability = full["opportunities"].div(full["opportunities"] + prior)
            derived[f"shrunk_{column}"] = 100.0 * shrunk
            derived[f"reliability_{column}"] = reliability

            recent_aggregate = recent.groupby(["season", "team_id"]).apply(
                lambda frame: pd.Series(
                    {
                        "numerator": float((frame[rate_column] * frame[opportunity_column]).sum()),
                        "opportunities": float(frame[opportunity_column].sum()),
                        "mean_opportunities": float(frame[opportunity_column].mean()),
                        "volatility": float((100.0 * frame[rate_column]).std(ddof=1)),
                    }
                ),
                include_groups=False,
            )
            recent_prior = RECENT_PRIOR_GAMES * recent_aggregate["mean_opportunities"]
            recent_rate = (
                recent_aggregate["numerator"] + recent_prior * shrunk.reindex(recent_aggregate.index)
            ).div(recent_aggregate["opportunities"] + recent_prior)
            derived[f"recent_{column}"] = 100.0 * recent_rate
            derived[f"recent_delta_{column}"] = 100.0 * recent_rate - 100.0 * shrunk
            derived[f"volatility_{column}"] = recent_aggregate["volatility"]

    return grouped_index.join(pd.DataFrame(derived)).reset_index()


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

    grouped = add_shrinkage_form_and_volatility(rows, grouped)
    grouped = add_opponent_adjusted_features(rows, grouped)

    names = teams[["TeamID", "TeamName"]].rename(columns={"TeamID": "team_id", "TeamName": "team"})
    grouped = grouped.merge(names, on="team_id", how="left")
    grouped["snapshot_status"] = "confirmed_pre_tournament"
    grouped["source"] = "NCAA regular-season detailed box scores"
    grouped["formula_version"] = "four_factors_misc_opponent_adjusted_v2"

    feature_cols = [
        col
        for col in grouped.columns
        if col.startswith(
            (
                "ff_",
                "misc_",
                "shrunk_",
                "reliability_",
                "recent_",
                "recent_delta_",
                "volatility_",
                "adj_",
                "power_",
            )
        )
    ]
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
        "regular_season_games_used": int(len(results[results["Season"].isin(available)])),
        "regular_season_team_game_observations": int(2 * len(results[results["Season"].isin(available)])),
        "opponent_adjustment": {
            "method": "season-specific weighted ridge offense and defense effects",
            "alpha": OPPONENT_ADJUSTMENT_ALPHA,
            "home_court_control": True,
        },
        "shrinkage": {
            "full_season_prior_equivalent_games": PRIOR_GAMES,
            "recent_games": RECENT_GAMES,
            "recent_prior_equivalent_games": RECENT_PRIOR_GAMES,
        },
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
