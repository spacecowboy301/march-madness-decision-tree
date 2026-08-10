# Trustworthy Four Factors + Misc Importance Analysis

## Data timing

Features are computed only from NCAA regular-season detailed box scores. Tournament results are stored separately, so the feature snapshot is confirmed pre-tournament rather than inferred from a season-end KenPom page.

## Selected model

- Model: Hist Gradient Boosting
- Held-out accuracy: 0.658
- Held-out ROC AUC: 0.694
- Held-out log loss: 0.634
- Accuracy 95% Wilson interval: 0.611-0.703
- Importance interpretation gate passed: True
- Model selection basis: hyperparameters tuned by pre-2017 expanding-window cross-validation; model family selected by calibrated accuracy on the requested 2017-2025 validation period

## Most important features

- Offensive Rebound Raw Offense Diff: 0.0124 mean held-out log-loss increase; positive in 100% of seasons
- Four Factor Edge Mean: 0.0077 mean held-out log-loss increase; positive in 100% of seasons
- Non-Steal Turnover Net Weakness Vs Strength: 0.0074 mean held-out log-loss increase; positive in 100% of seasons
- Offensive Rebound Offense Strength Diff: 0.0035 mean held-out log-loss increase; positive in 83% of seasons
- Block Raw Defense Diff: 0.0035 mean held-out log-loss increase; positive in 67% of seasons
- eFG Net Weakness Vs Strength: 0.0025 mean held-out log-loss increase; positive in 83% of seasons
- Possession Creation Edge: 0.0022 mean held-out log-loss increase; positive in 67% of seasons
- Free Throw % Net Weakness Vs Strength: 0.0020 mean held-out log-loss increase; positive in 83% of seasons

## Factor-group consensus

- Engineered Composites: 0.0239
- Offensive Rebound: 0.0201
- eFG: 0.0101
- Block: 0.0076
- Non-Steal Turnover: 0.0058
- Free Throw Rate: 0.0045
- Steal: 0.0045
- Free Throw %: 0.0023
- Three Point: 0.0009
- Turnover: 0.0008
- Three Point Rate: 0.0004
- Two Point: 0.0000
- Assist: -0.0001

## Matchup-mechanism importance

- Raw Rate Differences: 0.0170
- Engineered Composites: 0.0169
- Percentile Strength Differences: 0.0068
- Weakness Vs Strength: 0.0064
- Strength Vs Strength: 0.0025
- Strength Vs Weakness: 0.0008
- Three Point Volume Style: 0.0004
- Net Matchup Edges: 0.0003

## Upset behavior

- Actual Upsets: 0.354 accuracy across 113 games
- Favorite Wins: 0.778 accuracy across 288 games
- All Seeded Games: 0.658 accuracy across 401 games

## Interpretation guardrails

Permutation importance measures predictive reliance, not causality. Correlated engineered features share credit, which is why the grouped chart and cross-model range are more reliable than a single feature's exact rank.
Tournament seeds, KenPom rank, net rating, luck, and adjusted efficiency are excluded from model features. Seeds are used only after prediction to label upset evaluation segments.
