import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_factor_importance import select_model
from src.factor_matchups import FEATURES_PATH, add_percentiles, build_matchup_dataset, matchup_terms


class EnhancedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.team_features = add_percentiles(pd.read_csv(FEATURES_PATH)).set_index(["season", "team_id"])

    def test_adjusted_form_and_power_features_exist(self):
        columns = self.team_features.columns
        self.assertIn("adj_ff_off_efg", columns)
        self.assertIn("recent_delta_ff_off_efg", columns)
        self.assertIn("volatility_ff_off_efg", columns)
        self.assertIn("power_rating", columns)

    def test_matchup_directions_reverse_cleanly(self):
        season = 2023
        teams = self.team_features.loc[season].sort_values("power_rating", ascending=False).head(2)
        a = teams.iloc[0]
        b = teams.iloc[1]
        forward, _ = matchup_terms(a, b)
        reverse, _ = matchup_terms(b, a)
        directional = [
            name
            for name in forward
            if name.endswith(
                (
                    "_edge",
                    "_expected_diff",
                    "_strength_vs_strength",
                    "_strength_vs_weakness",
                    "_weakness_vs_strength",
                )
            )
        ]
        for name in directional:
            self.assertAlmostEqual(forward[name], -reverse[name], places=9, msg=name)

    def test_power_is_separate_from_factor_inputs(self):
        x, _y, meta, families = build_matchup_dataset()
        self.assertNotIn("regular_season_power_gap", x.columns)
        self.assertNotIn("power_baseline", set(families.values()))
        self.assertFalse(meta["regular_season_power_gap"].isna().any())
        forbidden = ("kenpom", "seed", "luck", "net_rating", "netrtg")
        self.assertFalse(any(any(term in column.lower() for term in forbidden) for column in x.columns))

    def test_regular_season_games_are_recorded_as_model_inputs(self):
        audit = json.loads(Path("reports/data_audit.json").read_text())
        self.assertGreater(audit["regular_season_games_used"], 100_000)
        self.assertEqual(
            audit["regular_season_team_game_observations"],
            2 * audit["regular_season_games_used"],
        )

    def test_model_selection_ignores_held_out_results(self):
        comparison = pd.DataFrame(
            [
                {
                    "model": "training_winner",
                    "training_cv_accuracy": 0.66,
                    "training_cv_roc_auc": 0.74,
                    "training_cv_log_loss": 0.58,
                    "calibrated_log_loss": 0.70,
                },
                {
                    "model": "validation_winner",
                    "training_cv_accuracy": 0.65,
                    "training_cv_roc_auc": 0.72,
                    "training_cv_log_loss": 0.62,
                    "calibrated_log_loss": 0.50,
                },
            ]
        )

        selected, gate_passed = select_model(comparison)

        self.assertEqual(selected, "training_winner")
        self.assertTrue(gate_passed)


if __name__ == "__main__":
    unittest.main()
