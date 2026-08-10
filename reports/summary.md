# Trustworthy Four Factors + Misc Importance Analysis

## Data timing

Features are computed only from NCAA regular-season detailed box scores. Tournament results are stored separately, so the feature snapshot is confirmed pre-tournament rather than inferred from a season-end KenPom page.

## Selected model

- Model: Regularized Logistic
- Held-out accuracy: 0.641
- Held-out ROC AUC: 0.706
- Held-out log loss: 0.625
- Accuracy 95% Wilson interval: 0.593-0.686
- Importance interpretation gate passed: True
- Model selection basis: hyperparameters tuned by pre-2017 expanding-window cross-validation; model family selected by calibrated accuracy on the requested 2017-2025 validation period

## Most important features

- Offensive Rebound Net Strength Vs Strength: 0.0110 mean held-out log-loss increase; positive in 100% of seasons
- Offensive Rebound Raw Matchup Edge: 0.0100 mean held-out log-loss increase; positive in 67% of seasons
- Block Net Strength Vs Strength: 0.0079 mean held-out log-loss increase; positive in 100% of seasons
- eFG Raw Matchup Edge: 0.0038 mean held-out log-loss increase; positive in 67% of seasons
- Free Throw % Net Strength Vs Strength: 0.0032 mean held-out log-loss increase; positive in 83% of seasons
- Turnover Raw Matchup Edge: 0.0027 mean held-out log-loss increase; positive in 67% of seasons
- Free Throw Rate Net Strength Vs Weakness: 0.0023 mean held-out log-loss increase; positive in 100% of seasons
- Two Point Raw Matchup Edge: 0.0023 mean held-out log-loss increase; positive in 50% of seasons

## Factor-group consensus

- Engineered Composites: 0.0239
- Offensive Rebound: 0.0124
- eFG: 0.0090
- Non-Steal Turnover: 0.0055
- Steal: 0.0044
- Block: 0.0032
- Two Point: 0.0032
- Free Throw %: 0.0018
- Turnover: 0.0016
- Three Point: 0.0015
- Free Throw Rate: 0.0010
- Three Point Rate: 0.0002
- Assist: 0.0001

## Matchup-mechanism importance

- Engineered Composites: 0.0168
- Raw Matchup Edges: 0.0160
- Strength Vs Strength: 0.0090
- Weakness Vs Strength: 0.0070
- Net Matchup Edges: 0.0019
- Strength Vs Weakness: 0.0016
- Matchup Environments: 0.0004
- Three Point Volume Style: 0.0002

## Upset behavior

- Actual Upsets: 0.336 accuracy across 113 games
- Favorite Wins: 0.760 accuracy across 288 games
- All Seeded Games: 0.641 accuracy across 401 games

## Interpretation guardrails

Permutation importance measures predictive reliance, not causality. Correlated engineered features share credit, which is why the grouped chart and cross-model range are more reliable than a single feature's exact rank.
Tournament seeds, KenPom rank, net rating, luck, and adjusted efficiency are excluded from model features. Seeds are used only after prediction to label upset evaluation segments.
