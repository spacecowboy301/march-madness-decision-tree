# Four Factors + Misc Matchup Model

This model intentionally excludes broad KenPom quality signals:

- KenPom rank
- Net rating
- Luck
- Broad adjusted efficiency ratings

It uses Four Factors and Miscellaneous stats only, with matchup engineering for:

- Strength vs strength
- Strength vs weakness
- Weakness vs strength
- Weakness vs weakness
- Team A attack edge vs Team B defense
- Team B attack edge vs Team A defense

## Performance

Validation seasons: 2017, 2018, 2019, 2021, 2022, 2023

- Validation accuracy: 56.40%
- Validation ROC AUC: 0.608
- Validation Brier score: 0.252
- Validation log loss: 0.798

2026 tournament:

- Correct predictions: 42 / 67
- Accuracy: 62.69%
- Brier score: 0.201
- Log loss: 0.591

## Most Important Features

| Feature | Importance | Interpretation |
| --- | ---: | --- |
| `efg_net_strength_vs_weakness` | 0.388 | Net edge when a strong eFG offense attacks a weak eFG defense. |
| `misc_nst_strength_diff` | 0.178 | Difference in non-steal turnover strength. |
| `oreb_net_weakness_vs_strength` | 0.106 | Net penalty when offensive rebounding weakness meets defensive rebounding strength. |
| `oreb_a_weakness_vs_strength` | 0.083 | Team A offensive rebounding weakness against Team B defensive rebounding strength. |
| `oreb_b_weakness_vs_strength` | 0.080 | Team B offensive rebounding weakness against Team A defensive rebounding strength. |
| `oreb_a_attack_edge` | 0.046 | Team A offensive rebounding rate vs Team B allowed offensive rebounding rate. |
| `oreb_b_attack_edge` | 0.045 | Team B offensive rebounding rate vs Team A allowed offensive rebounding rate. |
| `ftrate_b_weakness_vs_strength` | 0.037 | Team B free-throw-rate weakness against Team A defensive free-throw-rate strength. |
| `misc_threep_abs_diff` | 0.037 | Absolute 3-point shooting gap. |

## Takeaway

When the broad team-strength signals are removed, the tree leans heavily on matchup texture:

- eFG offense vs eFG defense mismatches
- offensive rebounding weakness vs defensive rebounding strength
- offensive rebounding attack edges
- turnover profile differences
- 3-point shooting gaps

The performance drop is meaningful. The full KenPom model performs much better because rank/net rating summarize team quality very efficiently, while this model is better for explaining specific matchup dynamics.
