# NCAA Men's March Madness Factor Importance

This project measures how pre-tournament Four Factors and miscellaneous team
statistics contribute to NCAA men's tournament predictions. The analysis
prioritizes trustworthy feature importance before maximum predictive accuracy.

## Method

- Build team-season features only from regular-season detailed box scores.
- Include direction-correct raw matchup edges for every Four Factor and
  miscellaneous statistic: Team A offense against Team B defense, compared
  with the reverse matchup.
- Engineer strength-vs-strength, strength-vs-weakness, balance, shooting,
  rebounding, and ball-security matchup features.
- Exclude tournament seed, KenPom rank, net rating, luck, and adjusted
  efficiency from model inputs.
- Compare logistic regression, decision tree, random forest, and histogram
  gradient boosting models.
- Calibrate probabilities and evaluate one row per game on future seasons.
- Measure held-out permutation importance by feature and basketball concept.

The requested split is 1997-2016 for training and 2017-2025 for validation.
Detailed box scores begin in 2003, so the realized training period is
2003-2016. The available complete validation tournaments are 2017-2019 and
2021-2023; 2020 had no tournament and the downloaded 2024 regular-season file
is incomplete, so the pipeline excludes it automatically.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Run

The analysis command downloads missing NCAA source files, rebuilds the
pre-tournament features, compares the models, and regenerates every report:

```bash
./.venv/bin/python -m src.download_ncaa_data
MPLCONFIGDIR=.mplconfig ./.venv/bin/python -m src.analyze_factor_importance
```

Or open and run:

```text
notebooks/march_madness_factor_importance.ipynb
```

## Outputs

```text
models/
  factor_importance_model.joblib
reports/
  data_audit.json
  feature_importance.csv
  group_importance.csv
  interaction_importance.csv
  hyperparameter_tuning.csv
  metrics.json
  model_comparison.csv
  rolling_validation.csv
  summary.md
  upset_analysis.csv
  validation_predictions.csv
  figures/
    accuracy_by_year.png
    calibration.png
    feature_importance.png
    group_importance.png
    interaction_importance.png
    matchup_quadrants.png
    model_comparison.png
```

The main feature-importance chart uses validation permutation importance rather
than decision-tree impurity. One grouped chart shuffles each basketball concept
as a block; another separately measures raw rates, ordinary strength
differences, strength-vs-strength, strength-vs-weakness, and other matchup
mechanisms. No model feature compares offense only with offense or defense only
with defense.

See [`docs/factor_reference.md`](docs/factor_reference.md) for formulas,
directionality, matchup interpretations, engineered-feature definitions, and
importance-reading guidance for every factor.

## Optional KenPom Reference

`src/kenpom_scraper.py` is retained as an optional formula-reference utility.
KenPom's season-only historical Four Factors and miscellaneous pages are
season-end snapshots, so scraped values are never model inputs in the primary
analysis.

Credentials are read only from environment variables and are excluded from
Git:

```bash
export KENPOM_EMAIL="your-email"
export KENPOM_PASSWORD="your-password"
./.venv/bin/python -m src.kenpom_scraper --start-year 2002 --end-year 2025
```

The data audit may compare regular-season formulas with cached KenPom values,
but that comparison is diagnostic only. Raw downloads, cached KenPom data,
processed feature tables, credentials, and local Jupyter state are ignored.
