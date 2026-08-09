# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NCAA Men's March Madness Factor Importance
#
# This notebook studies how the Four Factors and miscellaneous team stats affect NCAA men's tournament predictions. It compares several model families, calibrates probabilities, and measures importance on held-out seasons.
#
# Requested split:
#
# - Training: 1997-2016
# - Validation: 2017-2025
#
# Timing note: KenPom's season-only historical pages are end-of-season snapshots, so the original KenPom decision tree below is retained only as a legacy benchmark. The primary analysis in section 9 computes KenPom-compatible Four Factors and misc rates exclusively from regular-season detailed box scores. That source is confirmed pre-tournament and does not include tournament games.

# %%
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.download_ncaa_data import main as download_ncaa_data
from src.kenpom_scraper import combine_cached_years, scrape_year, KenPomClient
from src.train_decision_tree import (
    load_tournament_games,
    load_kenpom_features,
    numeric_feature_columns,
    attach_features,
    build_examples,
    evaluate,
)

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
MODEL_PATH = PROJECT_ROOT / 'models' / 'decision_tree_march_madness.joblib'
PREDICTIONS_PATH = PROJECT_ROOT / 'reports' / 'validation_predictions.csv'
METRICS_PATH = PROJECT_ROOT / 'reports' / 'metrics.json'

PROJECT_ROOT

# %% [markdown]
# ## 1. Download NCAA Tournament Results
#
# The analysis expects Kaggle-format team, tournament, seed, and detailed regular-season files.

# %%
needed_ncaa_files = [
    RAW_DIR / 'MTeams.csv',
    RAW_DIR / 'MNCAATourneyCompactResults.csv',
    RAW_DIR / 'MNCAATourneySeeds.csv',
    RAW_DIR / 'MRegularSeasonDetailedResults.csv',
]
if not all(path.exists() for path in needed_ncaa_files):
    download_ncaa_data()
else:
    print('NCAA files already exist:', [path.name for path in needed_ncaa_files])

# %% [markdown]
# ## 2. Legacy KenPom Benchmark Data
#
# This optional benchmark uses the historical KenPom pages. Those season-only pages contain season-end values and are not used for the trustworthy feature-importance conclusions in section 9.
#
# Set `RUN_KENPOM_SCRAPE = True` to refresh missing seasons. If `KENPOM_EMAIL` and `KENPOM_PASSWORD` are available in the environment, the scraper also attempts subscriber Four Factors and Miscellaneous pages. Without credentials, it uses public efficiency ratings only.
#
# Recommended credential handling from a terminal before launching Jupyter:
#
# ```bash
# export KENPOM_EMAIL='your-email'
# export KENPOM_PASSWORD='your-password'
# jupyter notebook
# ```

# %%
START_YEAR = 2002
END_YEAR = 2025
RUN_KENPOM_SCRAPE = False
SCRAPE_DELAY_SECONDS = 10.0

has_kenpom_login = bool(os.getenv('KENPOM_EMAIL') and os.getenv('KENPOM_PASSWORD'))
public_efficiency_only = not has_kenpom_login

if RUN_KENPOM_SCRAPE:
    client = KenPomClient(os.getenv('KENPOM_EMAIL'), os.getenv('KENPOM_PASSWORD'), delay=SCRAPE_DELAY_SECONDS)
    client.login()
    for year in range(START_YEAR, END_YEAR + 1):
        print(f'Scraping KenPom {year}')
        scrape_year(
            client,
            year,
            include_team_pages=has_kenpom_login,
            public_efficiency_only=public_efficiency_only,
            use_cache=True,
        )

kenpom = combine_cached_years(START_YEAR, END_YEAR)
print(kenpom.shape)
kenpom.head()

# %% [markdown]
# ## 3. Build Matchup-Level Training Rows
#
# Each tournament game becomes two examples: winner-vs-loser labeled `1`, and loser-vs-winner labeled `0`. For every numeric KenPom feature, the model gets both the signed feature difference and the absolute difference.

# %%
games = load_tournament_games()
kenpom = load_kenpom_features()
feature_cols = numeric_feature_columns(kenpom)

games_with_features = attach_features(games, kenpom, feature_cols)
x, y, meta = build_examples(games_with_features, feature_cols)

train_mask = meta['season'].between(1997, 2016)
valid_mask = meta['season'].between(2017, 2025)

observed_train_cols = x.loc[train_mask].notna().any(axis=0)
x = x.loc[:, observed_train_cols]

summary = {
    'feature_count': len(feature_cols),
    'model_column_count': x.shape[1],
    'all_examples': int(len(x)),
    'train_examples': int(train_mask.sum()),
    'validation_examples': int(valid_mask.sum()),
    'actual_train_years': sorted(meta.loc[train_mask, 'season'].unique().tolist()),
    'actual_validation_years': sorted(meta.loc[valid_mask, 'season'].unique().tolist()),
}
summary

# %% [markdown]
# ## 4. Train Decision Tree Model

# %%
if train_mask.sum() == 0 or valid_mask.sum() == 0:
    raise ValueError('No train or validation rows after joining tournament games to KenPom features.')

model = Pipeline(
    steps=[
        ('imputer', ColumnTransformer([('num', SimpleImputer(strategy='median'), x.columns)], remainder='drop')),
        (
            'tree',
            DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=25,
                criterion='log_loss',
                random_state=42,
            ),
        ),
    ]
)

model.fit(x.loc[train_mask], y.loc[train_mask])
model

# %% [markdown]
# ## 5. Evaluate On Validation Seasons

# %%
train_metrics = evaluate(model, x.loc[train_mask], y.loc[train_mask])
valid_metrics = evaluate(model, x.loc[valid_mask], y.loc[valid_mask])

metrics = {
    'requested_train_years': '1997-2016',
    'requested_validation_years': '2017-2025',
    'actual_train_years': summary['actual_train_years'],
    'actual_validation_years': summary['actual_validation_years'],
    'rows': {
        'train': int(train_mask.sum()),
        'validation': int(valid_mask.sum()),
        'all_examples': int(len(x)),
    },
    'metrics': {'train': train_metrics, 'validation': valid_metrics},
}

pd.DataFrame(metrics['metrics']).T

# %% [markdown]
# ## 6. Inspect The Tree And Feature Importance

# %%
tree_text = export_text(model.named_steps['tree'], feature_names=list(x.columns))
print(tree_text)

# %%
importance = pd.DataFrame(
    {
        'feature': x.columns,
        'importance': model.named_steps['tree'].feature_importances_,
    }
).sort_values('importance', ascending=False)

importance.query('importance > 0').head(20)

# %% [markdown]
# ## 7. Run 2026 Tournament Predictions

# %%
from src.predict_2026_tournament import main as predict_2026_tournament

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({'model': model, 'features': list(x.columns)}, MODEL_PATH)
predict_2026_tournament()

summary_2026 = json.loads((PROJECT_ROOT / 'reports' / '2026_tournament_summary.json').read_text())
predictions_2026 = pd.read_csv(PROJECT_ROOT / 'reports' / '2026_tournament_predictions.csv')

pd.DataFrame(summary_2026['by_round']).T

# %%
predictions_2026[[
    'round',
    'winner',
    'loser',
    'winner_score',
    'loser_score',
    'pred_actual_winner_prob',
    'predicted_winner',
    'prediction_correct',
]].head(20)

# %%
predictions_2026.loc[
    ~predictions_2026['prediction_correct'],
    ['round', 'winner', 'loser', 'winner_score', 'loser_score', 'pred_actual_winner_prob', 'predicted_winner']
]

# %% [markdown]
# ## 8. Generated Model Visualizations
#
# These cells create saved visual explanations for the decision tree: feature importance, round accuracy, confidence vs margin, and a bracket-style HTML report for the 2026 tournament.

# %%
from IPython.display import HTML, IFrame, Image, display
from src.visualize_model import main as create_model_visualizations

create_model_visualizations()


# %% [markdown]
# ### Feature Importance Chart
#
# This shows which matchup-difference features the decision tree actually used most. Higher bars mean the feature reduced impurity more across tree splits.

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'feature_importance.png')))


# %% [markdown]
# ### Other Model Diagnostics
#
# Round accuracy shows where the model held up or struggled. The confidence chart compares how sure the model was with the actual game margin.

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / '2026_accuracy_by_round.png')))
display(Image(filename=str(PROJECT_ROOT / 'reports' / '2026_prediction_confidence.png')))


# %% [markdown]
# ### 2026 Bracket Diagram With Model Picks
#
# The HTML bracket has one card per played matchup. Blue cards are correct picks, red cards are misses. Each card includes the model pick, confidence, top matchup gaps, and the decision-tree path that explains the pick.

# %%
display(IFrame(src=str(PROJECT_ROOT / 'reports' / '2026_bracket_predictions.html'), width='100%', height=900))


# %% [markdown]
# ### Matchup-Level Explanations
#
# Use this table when you want the plain-text reason for a pick without opening the bracket diagram.

# %%
explanations_2026 = pd.read_csv(PROJECT_ROOT / 'reports' / '2026_prediction_explanations.csv')
explanations_2026[[
    'round',
    'matchup',
    'predicted_winner',
    'predicted_winner_probability',
    'prediction_correct',
    'top_matchup_drivers',
    'decision_tree_rules',
]].head(20)


# %% [markdown]
# ## 9. Trustworthy Four Factors + Misc Importance Analysis
#
# This is the primary analysis. It:
#
# - computes Four Factors and misc rates only from regular-season box scores;
# - excludes seed, KenPom rank, net rating, luck, and adjusted efficiency from model inputs;
# - uses one deterministic row per tournament game;
# - compares logistic regression, a decision tree, random forest, and gradient boosting;
# - calibrates probabilities using earlier-season out-of-fold predictions;
# - validates on future seasons and measures permutation importance stability by season;
# - engineers strength-vs-strength, strength-vs-weakness, balance, shooting-leverage, and ball-security features.

# %%
from src.build_pretournament_features import main as build_pretournament_features
from src.analyze_factor_importance import main as run_factor_importance_analysis

build_pretournament_features()
run_factor_importance_analysis()

factor_metrics = json.loads((PROJECT_ROOT / 'reports' / 'factor_model_metrics.json').read_text())
factor_comparison = pd.read_csv(PROJECT_ROOT / 'reports' / 'factor_model_comparison.csv')

print('Selected model:', factor_metrics['selected_model'])
print('Pre-tournament timing:', factor_metrics['snapshot_status'])
print('Importance gate passed:', factor_metrics['trustworthiness']['importance_gate_passed'])
factor_comparison


# %% [markdown]
# ### Feature Importance
#
# Bars are held-out permutation importance: the increase in log loss when a feature is shuffled within each validation season. Whiskers show uncertainty across seasons and repeated shuffles. Larger positive values mean the model relied more on that feature.

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_model_feature_importance.png')))


# %% [markdown]
# ### Grouped Factor Importance
#
# Correlated engineered features split credit. This grouped chart shuffles each basketball concept as a block and shows the range across all four model families.

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_model_group_importance.png')))


# %% [markdown]
# ### Model Comparison And Calibration

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_model_comparison.png')))
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_model_calibration.png')))
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_model_accuracy_by_year.png')))


# %% [markdown]
# ### Strength Matchup Quadrants
#
# These held-out descriptive rates compare strong and weak offenses against strong and weak defenses. Cells with fewer than 10 games are suppressed rather than over-interpreted.

# %%
display(Image(filename=str(PROJECT_ROOT / 'reports' / 'factor_matchup_quadrants.png')))


# %% [markdown]
# ### Stable Features And Upset Diagnostic
#
# Seeds are used only here to label actual upsets after predictions are made. They are never model features.

# %%
factor_top_features = pd.read_csv(PROJECT_ROOT / 'reports' / 'factor_model_top_features.csv')
factor_upsets = pd.read_csv(PROJECT_ROOT / 'reports' / 'factor_model_upset_analysis.csv')
factor_rolling = pd.read_csv(PROJECT_ROOT / 'reports' / 'factor_model_rolling_validation.csv')

display(factor_top_features.head(20))
display(factor_upsets)
factor_rolling[factor_rolling['model'].eq(factor_metrics['selected_model'])]


# %% [markdown]
# ## 10. Save Legacy Benchmark Model, Predictions, And Metrics

# %%
valid_meta = meta.loc[valid_mask].copy()
valid_meta['pred_team_a_win_prob'] = model.predict_proba(x.loc[valid_mask])[:, 1]
valid_meta['pred_team_a_win'] = (valid_meta['pred_team_a_win_prob'] >= 0.5).astype(int)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

metrics['tree'] = tree_text
joblib.dump({'model': model, 'features': list(x.columns)}, MODEL_PATH)
valid_meta.to_csv(PREDICTIONS_PATH, index=False)
METRICS_PATH.write_text(json.dumps(metrics, indent=2))

print(f'Wrote {MODEL_PATH}')
print(f'Wrote {PREDICTIONS_PATH}')
print(f'Wrote {METRICS_PATH}')
valid_meta.head()
