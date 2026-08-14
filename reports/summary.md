# Trustworthy Four Factors + Misc Importance Analysis

## Data timing

Features are computed only from NCAA regular-season detailed box scores. Tournament results are stored separately, so the feature snapshot is confirmed pre-tournament rather than inferred from a season-end KenPom page.

## Selected model

- Model: Regularized Logistic
- Held-out accuracy: 0.661
- Held-out ROC AUC: 0.731
- Held-out log loss: 0.616
- Accuracy 95% Wilson interval: 0.613-0.705
- Importance interpretation gate passed: True
- Model selection basis: hyperparameters and model family selected only by pre-2017 expanding-window cross-validation; 2017-2023 is evaluation-only

## Most important features

- eFG Opponent Adjusted Matchup Edge: 0.0070 mean held-out log-loss increase; positive in 83% of seasons
- Two Point Opponent Adjusted Matchup Edge: 0.0053 mean held-out log-loss increase; positive in 83% of seasons
- Offensive Rebound Opponent Adjusted Matchup Edge: 0.0052 mean held-out log-loss increase; positive in 67% of seasons
- Non-Steal Turnover Net Weakness Vs Strength: 0.0033 mean held-out log-loss increase; positive in 100% of seasons
- Offensive Rebound Net Strength Vs Strength: 0.0030 mean held-out log-loss increase; positive in 100% of seasons
- Free Throw Rate Net Strength Vs Weakness: 0.0028 mean held-out log-loss increase; positive in 100% of seasons
- Block Opponent Adjusted Matchup Edge: 0.0022 mean held-out log-loss increase; positive in 100% of seasons
- Free Throw % Net Strength Vs Strength: 0.0022 mean held-out log-loss increase; positive in 67% of seasons

## Factor-group consensus

- Engineered Composites: 0.0330
- Offensive Rebound: 0.0107
- eFG: 0.0095
- Two Point: 0.0079
- Non-Steal Turnover: 0.0046
- Turnover: 0.0024
- Block: 0.0017
- Free Throw %: 0.0015
- Steal: 0.0015
- Three Point: 0.0005
- Assist: 0.0004
- Free Throw Rate: -0.0001
- Three Point Rate: -0.0001

## Matchup-mechanism importance

- Opponent Adjusted Edges: 0.0257
- Engineered Composites: 0.0231
- Weakness Vs Strength: 0.0069
- Recent Form: 0.0049
- Net Matchup Edges: 0.0029
- Strength Vs Strength: 0.0017
- Strength Vs Weakness: 0.0009
- Three Point Volume Style: 0.0000
- Reliability: -0.0002
- Volatility: -0.0004
- Raw Matchup Edges: -0.0033

## Upset behavior

- Actual Upsets: 0.283 accuracy across 113 games
- Favorite Wins: 0.809 accuracy across 288 games
- All Seeded Games: 0.661 accuracy across 401 games

## Interpretation guardrails

Permutation importance measures predictive reliance, not causality. Correlated engineered features share credit, which is why the grouped chart and cross-model range are more reliable than a single feature's exact rank.
Tournament seeds, KenPom rank, net rating, luck, and adjusted efficiency are excluded from the factor-only model. The separate two-channel report uses an internally estimated regular-season scoring-efficiency baseline; seeds are used only after prediction to label upset evaluation segments.
