# NCAA Men's March Madness Factor Importance

This project measures how the Four Factors and miscellaneous team stats contribute to NCAA men's tournament predictions. The primary analysis prioritizes trustworthy feature importance before optimizing maximum predictive accuracy.

Requested split:

- Training seasons: 1997-2016
- Validation seasons: 2017-2025

Important timing note: KenPom's season-only historical pages contain season-end values. They are retained as a legacy benchmark, but they are not used for the primary importance conclusions. The trustworthy factors pipeline computes KenPom-compatible rates only from regular-season detailed box scores, which are stored separately from NCAA tournament games.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Input Data

### NCAA tournament results

Download the men's March Machine Learning Mania files from Kaggle and place these files in `data/raw/`:

- `MTeams.csv`
- `MNCAATourneyCompactResults.csv`

The primary factors analysis uses:

- `MTeams.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv` (evaluation only; never a model feature)
- `MRegularSeasonDetailedResults.csv`

For convenience, this project also includes a downloader for a Kaggle-format mirror of those files:

```bash
./.venv/bin/python -m src.download_ncaa_data
```

### KenPom

The scraper uses KenPom login credentials from environment variables:

```bash
export KENPOM_EMAIL="your-email"
export KENPOM_PASSWORD="your-password"
```

Then scrape the requested years:

```bash
./.venv/bin/python -m src.kenpom_scraper --start-year 2002 --end-year 2025
```

The scraper writes yearly CSVs into `data/raw/kenpom/` and a combined feature table to:

```text
data/processed/kenpom_team_features.csv
```

It attempts to collect:

- Efficiency ratings from `index.php`
- Four Factors from `stats.php`
- Miscellaneous team stats from `teamstats.php`
- Optional team-page details from `team.php?team=...&y=...`

## Train and Validate

### Jupyter notebook

Open and run:

```text
notebooks/march_madness_decision_tree.ipynb
```

The notebook downloads NCAA tournament results if needed, combines or scrapes KenPom features, trains the decision tree, evaluates it, and writes the model/prediction artifacts.

### Python script

```bash
./.venv/bin/python -m src.train_decision_tree
```

Outputs:

- `models/decision_tree_march_madness.joblib`
- `reports/validation_predictions.csv`
- `reports/metrics.json`
- `reports/2026_tournament_predictions.csv`
- `reports/2026_tournament_summary.json`
- `reports/2026_tournament_results.md`
- `reports/feature_importance.png`
- `reports/2026_accuracy_by_round.png`
- `reports/2026_prediction_confidence.png`
- `reports/2026_prediction_explanations.csv`
- `reports/2026_bracket_predictions.html`

## Run 2026 Tournament Predictions

After scraping/caching KenPom 2026 ratings:

```bash
./.venv/bin/python -m src.predict_2026_tournament
```

The included 2026 results file is `data/raw/ncaa_2026_tournament_results.csv`.

The existing `reports/2026_factor_model_*` files are legacy outputs built from
season-end KenPom snapshots. They are not used in the confirmed pre-tournament
factor-importance analysis below and should not be interpreted as leakage-safe
2026 predictions.

## Visualize The Model

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python -m src.visualize_model
```

This generates feature-importance, round-accuracy, confidence, matchup-explanation, and bracket-view reports under `reports/`.

## Trustworthy Four Factors + Misc Analysis

Build confirmed pre-tournament Four Factors and misc rates:

```bash
./.venv/bin/python -m src.build_pretournament_features
```

Then compare and calibrate four model families, run one-row-per-game frozen and rolling validation, and measure held-out permutation importance:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python -m src.analyze_factor_importance
```

This excludes tournament seed, KenPom rank, net rating, luck, and adjusted efficiency from model inputs. It includes strength-vs-strength, strength-vs-weakness, factor balance, weakness exploitation, shooting leverage, interior/perimeter, and ball-security features.

Core outputs:

- `models/decision_tree_four_factors_misc.joblib`
- `reports/factor_model_metrics.json`
- `reports/factor_model_comparison.csv`
- `reports/factor_model_rolling_validation.csv`
- `reports/factor_model_top_features.csv`
- `reports/factor_model_feature_importance.png`
- `reports/factor_model_group_importance.csv`
- `reports/factor_model_group_importance.png`
- `reports/factor_model_calibration.png`
- `reports/factor_model_accuracy_by_year.png`
- `reports/factor_matchup_quadrants.png`
- `reports/factor_model_upset_analysis.csv`
- `reports/factor_model_results.md`

The model artifact retains its historical filename for compatibility. Its
contents are the calibrated model selected by held-out validation, which is
currently regularized logistic regression rather than a decision tree.

The feature-importance chart uses held-out permutation importance rather than a single tree's impurity counts. The grouped chart shuffles each basketball concept as a block and shows the range across logistic regression, decision tree, random forest, and gradient boosting.

## Modeling Approach

The trustworthy factors analysis uses one deterministic orientation per tournament game, so every game counts exactly once. Team A orientation is assigned by a stable hash, preventing winner-side bias without duplicating validation observations.

Training data currently covers 2003-2016 because detailed box scores begin in 2003. The downloaded source supports complete held-out tournaments for 2017-2019 and 2021-2023; 2020 had no tournament, and the local 2024 regular-season file is incomplete and therefore fails closed. The audit is saved to `reports/pretournament_feature_audit.json`.
