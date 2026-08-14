# Power Baseline + Interpretable Matchup Model

## Architecture

The model keeps opponent-adjusted regular-season scoring power in one channel and Four Factors matchup interactions in another. KenPom rank, net rating, luck, tournament seed, and season-end KenPom values are not inputs.

Regular-season games used to estimate pre-tournament team strength: **107,634**.

## Held-out comparison

- Two Channel Stack: 0.653 accuracy, 0.730 ROC AUC, 0.602 log loss
- Regular Season Power Only: 0.658 accuracy, 0.727 ROC AUC, 0.606 log loss
- Factor Only: 0.656 accuracy, 0.732 ROC AUC, 0.615 log loss

## Channel weights

- Regular-season power coefficient: 0.837
- Four Factors matchup coefficient: 0.090

These coefficients describe the final stack, not feature importance. The factor-only and conditional permutation reports remain the interpretation surfaces.

## Conditional factor importance

- Turnover: 0.0004 held-out log-loss increase
- Non-Steal Turnover: 0.0003 held-out log-loss increase
- Steal: 0.0003 held-out log-loss increase
- Three Point: 0.0003 held-out log-loss increase
- Block: 0.0003 held-out log-loss increase
- Offensive Rebound: 0.0002 held-out log-loss increase
- Assist: 0.0002 held-out log-loss increase
- Three Point Rate: -0.0000 held-out log-loss increase

## Conditional matchup mechanisms

- Recent Form: 0.0012 held-out log-loss increase
- Strength Vs Strength: 0.0000 held-out log-loss increase
- Weakness Vs Strength: 0.0000 held-out log-loss increase
- Three Point Volume Style: -0.0000 held-out log-loss increase
- Reliability: -0.0000 held-out log-loss increase
- Net Matchup Edges: -0.0001 held-out log-loss increase
- Volatility: -0.0002 held-out log-loss increase
- Opponent Adjusted Edges: -0.0002 held-out log-loss increase

## Data boundary

Current official-format local tournament outcomes end in 2023; 2024-2025 remain unavailable without a current competition-data download.
