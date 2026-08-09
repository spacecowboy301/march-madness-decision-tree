# Trustworthy Four Factors + Misc Importance Analysis

## Data timing

Features are computed only from NCAA regular-season detailed box scores. Tournament results are stored separately, so the feature snapshot is confirmed pre-tournament rather than inferred from a season-end KenPom page.

## Selected model

- Model: Regularized Logistic
- Held-out accuracy: 0.641
- Held-out ROC AUC: 0.707
- Held-out log loss: 0.624
- Accuracy 95% Wilson interval: 0.593-0.686
- Importance interpretation gate passed: True

## Most important engineered features

- Offensive Rebound Offense Strength Diff: 0.0092 mean held-out log-loss increase; positive in 100% of seasons
- Four Factor Edge Variability: 0.0083 mean held-out log-loss increase; positive in 83% of seasons
- Block Defense Strength Diff: 0.0075 mean held-out log-loss increase; positive in 100% of seasons
- eFG Offense Strength Diff: 0.0072 mean held-out log-loss increase; positive in 83% of seasons
- eFG Net Matchup Edge: 0.0063 mean held-out log-loss increase; positive in 83% of seasons
- Four Factor Edge Max: 0.0061 mean held-out log-loss increase; positive in 83% of seasons
- eFG Overall Strength Diff: 0.0058 mean held-out log-loss increase; positive in 67% of seasons
- Three Point Leverage Edge: 0.0058 mean held-out log-loss increase; positive in 83% of seasons

## Factor-group consensus

- Engineered Composites: 0.0320
- eFG: 0.0280
- Offensive Rebound: 0.0139
- Steal: 0.0079
- Block: 0.0072
- Non-Steal Turnover: 0.0071
- Three Point: 0.0068
- Free Throw Rate: 0.0066
- Free Throw %: 0.0026
- Two Point: 0.0021
- Turnover: 0.0018
- Three Point Rate: -0.0002
- Assist: -0.0003

## Upset behavior

- Actual Upsets: 0.363 accuracy across 113 games
- Favorite Wins: 0.750 accuracy across 288 games
- All Seeded Games: 0.641 accuracy across 401 games

## Interpretation guardrails

Permutation importance measures predictive reliance, not causality. Correlated engineered features share credit, which is why the grouped chart and cross-model range are more reliable than a single feature's exact rank.
Tournament seeds, KenPom rank, net rating, luck, and adjusted efficiency are excluded from model features. Seeds are used only after prediction to label upset evaluation segments.
