from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path("data/processed/pretournament_team_factors.csv")
RESULTS_PATH = Path("data/raw/MNCAATourneyCompactResults.csv")
TEAMS_PATH = Path("data/raw/MTeams.csv")
SEEDS_PATH = Path("data/raw/MNCAATourneySeeds.csv")


FACTOR_SPECS = {
    "efg": ("ff_off_efg", "ff_def_efg", True, False, "four_factors"),
    "turnover": ("ff_off_to", "ff_def_to", False, True, "four_factors"),
    "offensive_rebound": ("ff_off_or", "ff_def_or", True, False, "four_factors"),
    "free_throw_rate": ("ff_off_ftrate", "ff_def_ftrate", True, False, "four_factors"),
    "three_point": ("misc_off_3p", "misc_def_3p", True, False, "misc_shooting"),
    "two_point": ("misc_off_2p", "misc_def_2p", True, False, "misc_shooting"),
    "free_throw_pct": ("misc_off_ft", "misc_def_ft", True, False, "misc_shooting"),
    "block": ("misc_off_blk", "misc_def_blk", False, True, "misc_ball_security"),
    "steal": ("misc_off_stl", "misc_def_stl", False, True, "misc_ball_security"),
    "nonsteal_turnover": ("misc_off_nst", "misc_def_nst", False, True, "misc_ball_security"),
    "assist": ("misc_off_a", "misc_def_a", True, False, "misc_creation_style"),
}

STYLE_SPECS = {
    "three_point_rate": ("misc_off_3pa", "misc_def_3pa", "misc_creation_style"),
}


def add_percentiles(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for _name, (off_col, def_col, off_higher, def_higher, _family) in FACTOR_SPECS.items():
        for col, higher in [(off_col, off_higher), (def_col, def_higher)]:
            values = pd.to_numeric(features[col], errors="coerce")
            score = values if higher else -values
            features[f"{col}_strength"] = score.groupby(features["season"]).rank(pct=True)
    for _name, (off_col, def_col, _family) in STYLE_SPECS.items():
        for col in [off_col, def_col]:
            values = pd.to_numeric(features[col], errors="coerce")
            features[f"{col}_percentile"] = values.groupby(features["season"]).rank(pct=True)
    return features


def stable_winner_first(season: int, winner_id: int, loser_id: int) -> bool:
    key = f"{season}:{min(winner_id, loser_id)}:{max(winner_id, loser_id)}".encode("ascii")
    return hashlib.sha256(key).digest()[0] % 2 == 0


def seed_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    digits = "".join(char for char in text[1:] if char.isdigit())
    return float(digits[:2]) if len(digits) >= 2 else np.nan


def load_tournament_games() -> pd.DataFrame:
    results = pd.read_csv(RESULTS_PATH)
    teams = pd.read_csv(TEAMS_PATH).set_index("TeamID")["TeamName"]
    games = results.rename(
        columns={
            "Season": "season",
            "WTeamID": "winner_id",
            "LTeamID": "loser_id",
            "WScore": "winner_score",
            "LScore": "loser_score",
        }
    )
    games["winner"] = games["winner_id"].map(teams)
    games["loser"] = games["loser_id"].map(teams)

    if SEEDS_PATH.exists():
        seeds = pd.read_csv(SEEDS_PATH).rename(columns={"Season": "season", "TeamID": "team_id", "Seed": "seed"})
        winner_seeds = seeds.rename(columns={"team_id": "winner_id", "seed": "winner_seed_code"})
        loser_seeds = seeds.rename(columns={"team_id": "loser_id", "seed": "loser_seed_code"})
        games = games.merge(winner_seeds, on=["season", "winner_id"], how="left")
        games = games.merge(loser_seeds, on=["season", "loser_id"], how="left")
        games["winner_seed"] = games["winner_seed_code"].map(seed_number)
        games["loser_seed"] = games["loser_seed_code"].map(seed_number)
    else:
        games["winner_seed"] = np.nan
        games["loser_seed"] = np.nan
    return games


def matchup_terms(a: pd.Series, b: pd.Series) -> tuple[dict[str, float], dict[str, str]]:
    row: dict[str, float] = {}
    families: dict[str, str] = {}
    attack_edges: dict[str, float] = {}
    overall_strengths_a: dict[str, float] = {}
    overall_strengths_b: dict[str, float] = {}

    for name, (off_col, def_col, off_higher, def_higher, family) in FACTOR_SPECS.items():
        off_direction = 1.0 if off_higher else -1.0
        def_direction = 1.0 if def_higher else -1.0
        raw_values = {
            f"{name}_raw_off_diff": off_direction * (a[off_col] - b[off_col]),
            f"{name}_raw_def_diff": def_direction * (a[def_col] - b[def_col]),
        }
        row.update(raw_values)
        for key in raw_values:
            families[key] = name

        a_off = a[f"{off_col}_strength"]
        a_def = a[f"{def_col}_strength"]
        b_off = b[f"{off_col}_strength"]
        b_def = b[f"{def_col}_strength"]

        values = {
            f"{name}_off_strength_diff": a_off - b_off,
            f"{name}_def_strength_diff": a_def - b_def,
            f"{name}_overall_strength_diff": (a_off + a_def - b_off - b_def) / 2.0,
            f"{name}_net_matchup_edge": (a_off - b_def) - (b_off - a_def),
            f"{name}_net_strength_vs_strength": a_off * b_def - b_off * a_def,
            f"{name}_net_strength_vs_weakness": a_off * (1 - b_def) - b_off * (1 - a_def),
            f"{name}_net_weakness_vs_strength": (1 - a_off) * b_def - (1 - b_off) * a_def,
        }
        row.update(values)
        for key in values:
            families[key] = name
        attack_edges[name] = values[f"{name}_net_matchup_edge"]
        overall_strengths_a[name] = (a_off + a_def) / 2.0
        overall_strengths_b[name] = (b_off + b_def) / 2.0

    for name, (off_col, def_col, _family) in STYLE_SPECS.items():
        raw_values = {
            f"{name}_raw_off_diff": a[off_col] - b[off_col],
            f"{name}_raw_def_diff": a[def_col] - b[def_col],
        }
        row.update(raw_values)
        for key in raw_values:
            families[key] = name

        a_off = a[f"{off_col}_percentile"]
        a_def = a[f"{def_col}_percentile"]
        b_off = b[f"{off_col}_percentile"]
        b_def = b[f"{def_col}_percentile"]
        values = {
            f"{name}_off_style_diff": a_off - b_off,
            f"{name}_def_style_diff": a_def - b_def,
            f"{name}_expected_diff": (a_off + b_def - b_off - a_def) / 2.0,
            f"{name}_environment": (a_off + a_def + b_off + b_def) / 4.0,
        }
        row.update(values)
        for key in values:
            families[key] = name

    four = [attack_edges[name] for name in ["efg", "turnover", "offensive_rebound", "free_throw_rate"]]
    composites = {
        "four_factor_edge_mean": float(np.mean(four)),
        "four_factor_edge_min": float(np.min(four)),
        "four_factor_edge_max": float(np.max(four)),
        "four_factor_edge_std": float(np.std(four)),
        "four_factor_positive_edge_count": float(sum(value > 0 for value in four) - sum(value < 0 for value in four)),
        "possession_creation_edge": float(np.mean([attack_edges["turnover"], attack_edges["offensive_rebound"]])),
        "shooting_pressure_edge": float(np.mean([attack_edges["efg"], attack_edges["free_throw_rate"]])),
        "perimeter_edge": float(np.mean([attack_edges["efg"], attack_edges["three_point"]])),
        "interior_edge": float(
            np.mean(
                [
                    attack_edges["two_point"],
                    attack_edges["offensive_rebound"],
                    attack_edges["free_throw_rate"],
                    attack_edges["block"],
                ]
            )
        ),
        "ball_security_edge": float(
            np.mean([attack_edges["turnover"], attack_edges["steal"], attack_edges["nonsteal_turnover"]])
        ),
        "four_factor_balance_diff": float(
            min(overall_strengths_a[name] for name in ["efg", "turnover", "offensive_rebound", "free_throw_rate"])
            - min(overall_strengths_b[name] for name in ["efg", "turnover", "offensive_rebound", "free_throw_rate"])
        ),
        "four_factor_breadth_diff": float(
            sum(overall_strengths_a[name] >= 0.67 for name in ["efg", "turnover", "offensive_rebound", "free_throw_rate"])
            - sum(overall_strengths_b[name] >= 0.67 for name in ["efg", "turnover", "offensive_rebound", "free_throw_rate"])
        ),
        "strength_vs_strength_composite": float(
            np.mean([row[f"{name}_net_strength_vs_strength"] for name in FACTOR_SPECS])
        ),
        "weakness_exploitation_composite": float(
            np.mean([row[f"{name}_net_strength_vs_weakness"] for name in FACTOR_SPECS])
        ),
        "three_point_leverage_edge": float(
            attack_edges["three_point"] * row["three_point_rate_environment"]
        ),
    }
    row.update(composites)
    for key in composites:
        families[key] = "engineered_composites"
    return row, families


def build_matchup_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, str]]:
    features = add_percentiles(pd.read_csv(FEATURES_PATH)).set_index(["season", "team_id"])
    games = load_tournament_games()
    rows = []
    labels = []
    meta = []
    feature_families: dict[str, str] = {}

    for game in games.itertuples(index=False):
        winner_key = (game.season, game.winner_id)
        loser_key = (game.season, game.loser_id)
        if winner_key not in features.index or loser_key not in features.index:
            continue
        winner_first = stable_winner_first(game.season, game.winner_id, game.loser_id)
        if winner_first:
            a, b = features.loc[winner_key], features.loc[loser_key]
            team_a, team_b = game.winner, game.loser
            team_a_id, team_b_id = game.winner_id, game.loser_id
            label = 1
        else:
            a, b = features.loc[loser_key], features.loc[winner_key]
            team_a, team_b = game.loser, game.winner
            team_a_id, team_b_id = game.loser_id, game.winner_id
            label = 0
        row, families = matchup_terms(a, b)
        rows.append(row)
        labels.append(label)
        feature_families.update(families)
        meta.append(
            {
                "season": game.season,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "actual_team_a_win": label,
                "actual_winner": game.winner,
                "actual_loser": game.loser,
                "winner_seed": game.winner_seed,
                "loser_seed": game.loser_seed,
                "actual_upset": bool(game.winner_seed > game.loser_seed)
                if pd.notna(game.winner_seed) and pd.notna(game.loser_seed)
                else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.Series(labels, name="team_a_win"), pd.DataFrame(meta), feature_families
