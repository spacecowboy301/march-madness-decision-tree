# NCAA Men's March Madness Decision Tree

This project builds a decision tree model to predict NCAA men's tournament game winners from pre-tournament KenPom team features.

Requested split:

- Training seasons: 1997-2016
- Validation seasons: 2017-2025

Important data note: KenPom's public historical archive currently exposes ratings from 2002 onward. The pipeline keeps the requested split, but games before the first scraped KenPom season are dropped because the requested KenPom features do not exist locally for those years.

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

The loader only needs those two files.

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

## Run 2026 Tournament Predictions

After scraping/caching KenPom 2026 ratings:

```bash
./.venv/bin/python -m src.predict_2026_tournament
```

The included 2026 results file is `data/raw/ncaa_2026_tournament_results.csv`.

## Modeling Approach

Each tournament game becomes two training examples:

- Team A vs Team B, label `1`
- Team B vs Team A, label `0`

For every numeric KenPom feature, the model receives:

- `feature_diff = team_a_feature - team_b_feature`
- `feature_abs_diff = abs(team_a_feature - team_b_feature)`

This avoids giving the model a fixed winner-side bias and lets a plain decision tree learn matchup thresholds.
