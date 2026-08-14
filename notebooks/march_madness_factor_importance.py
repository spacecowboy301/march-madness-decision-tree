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
# This notebook measures how pre-tournament Four Factors and miscellaneous team
# statistics contribute to tournament predictions. It compares factor-only
# models, calibrates their probabilities, and measures importance only on
# future held-out seasons. A second model keeps regular-season power and matchup
# signal in separate channels so overall quality cannot erase factor readability.
#
# Tournament seed, KenPom rank, net rating, luck, and adjusted efficiency are
# excluded from the factor-only model. A separate baseline uses internally
# estimated regular-season scoring efficiency. All inputs come from NCAA
# regular-season detailed box scores using KenPom-compatible formulas.

# %%
from pathlib import Path
import json
import os
import sys

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_enhanced_analysis import main as run_analysis
from src.download_ncaa_data import main as download_ncaa_data

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
PROJECT_ROOT

# %% [markdown]
# ## Factor Reference
#
# The complete data dictionary below defines each raw factor, its offensive and
# defensive direction, its matchup interpretation, and every engineered feature.

# %%
display(Markdown((PROJECT_ROOT / "docs" / "factor_reference.md").read_text()))

# %% [markdown]
# ## 1. Prepare NCAA Data
#
# Raw data are downloaded only when required files are missing. They remain
# excluded from Git.

# %%
required_files = [
    PROJECT_ROOT / "data/raw/MTeams.csv",
    PROJECT_ROOT / "data/raw/MNCAATourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/MNCAATourneySeeds.csv",
    PROJECT_ROOT / "data/raw/MRegularSeasonDetailedResults.csv",
]

if not all(path.exists() for path in required_files):
    download_ncaa_data()
else:
    print("NCAA source files already available.")

# %% [markdown]
# ## 2. Build Features, Train Models, And Validate
#
# This single entry point rebuilds the pre-tournament team features, engineers
# matchup variables, compares and calibrates all candidate models, and writes
# the model and reports.

# %%
run_analysis()

metrics = json.loads((REPORTS / "metrics.json").read_text())
comparison = pd.read_csv(REPORTS / "model_comparison.csv")
power_metrics = json.loads((REPORTS / "power_matchup_metrics.json").read_text())
power_comparison = pd.read_csv(REPORTS / "power_matchup_model_comparison.csv")

print("Selected model:", metrics["selected_model"])
print("Pre-tournament timing:", metrics["snapshot_status"])
print("Importance gate passed:", metrics["trustworthiness"]["importance_gate_passed"])
comparison

# %% [markdown]
# ## 3. Regular-Season Data Contribution
#
# Every complete regular-season game contributes two team-game scoring
# observations for opponent adjustment, reliability shrinkage, recent form,
# volatility, and the separate scoring-power baseline. Tournament outcomes
# remain the only binary labels used to fit and evaluate the tournament
# classifier, avoiding leakage from a game's own final box score.

# %%
print("Regular-season games used:", f"{power_metrics['regular_season_games_used']:,}")
print("KenPom rank used:", power_metrics["kenpom_rank_used"])
power_comparison

# %% [markdown]
# ## 4. Feature Importance
#
# Bars show the held-out increase in log loss when each feature is shuffled
# within a validation season. Whiskers capture uncertainty across seasons and
# repeated shuffles. Larger positive values indicate greater predictive reliance.

# %%
display(Image(filename=str(FIGURES / "feature_importance.png")))

# %% [markdown]
# ## 5. Grouped Factor Importance
#
# Correlated engineered variables can split credit. This chart shuffles each
# basketball concept as a block and shows its importance range across logistic
# regression, decision tree, random forest, and gradient boosting.

# %%
display(Image(filename=str(FIGURES / "group_importance.png")))

# %% [markdown]
# ## 6. Raw Rates And Matchup Mechanisms
#
# This view directly compares raw offense-vs-opposing-defense matchup edges,
# strength-vs-strength, strength-vs-weakness, and the remaining engineered
# mechanisms. No feature compares offense only with offense or defense only
# with defense.

# %%
display(Image(filename=str(FIGURES / "interaction_importance.png")))

# %% [markdown]
# ## 7. Model Diagnostics

# %%
display(Image(filename=str(FIGURES / "model_comparison.png")))
display(Image(filename=str(FIGURES / "calibration.png")))
display(Image(filename=str(FIGURES / "accuracy_by_year.png")))

# %% [markdown]
# ## 8. Power Baseline And Conditional Matchup Signal
#
# The factor-only model remains the interpretation model. The two-channel stack
# combines it with opponent-adjusted regular-season scoring power. Conditional
# permutation importance shuffles factor groups while the power baseline stays
# fixed.

# %%
display(Image(filename=str(FIGURES / "power_matchup_model_comparison.png")))
display(Image(filename=str(FIGURES / "conditional_factor_importance.png")))
display(Image(filename=str(FIGURES / "conditional_matchup_mechanism_importance.png")))

# %% [markdown]
# ## 9. Model vs KenPom-Derived Discrepancies
#
# This retrospective comparison converts KenPom adjusted net-rating gaps into
# probabilities using the pre-2017 training period. These are not official
# KenPom game probabilities, and the cached KenPom pages are season-end
# snapshots that may include tournament results. The model inputs remain
# confirmed pre-tournament.

# %%
display(Image(filename=str(FIGURES / "model_vs_kenpom_discrepancies.png")))
display(Markdown((REPORTS / "kenpom_discrepancy_summary.md").read_text()))

# %% [markdown]
# ## 10. Strength Matchups
#
# These descriptive held-out rates compare strong and weak offenses with strong
# and weak defenses. Cells with fewer than ten games are suppressed.

# %%
display(Image(filename=str(FIGURES / "matchup_quadrants.png")))

# %% [markdown]
# ## 11. Detailed Results
#
# Seeds appear only in the post-prediction upset diagnostic and never enter the
# model. Permutation importance measures predictive reliance, not causality.

# %%
feature_importance = pd.read_csv(REPORTS / "feature_importance.csv")
group_importance = pd.read_csv(REPORTS / "group_importance.csv")
interaction_importance = pd.read_csv(REPORTS / "interaction_importance.csv")
hyperparameter_tuning = pd.read_csv(REPORTS / "hyperparameter_tuning.csv")
rolling_validation = pd.read_csv(REPORTS / "rolling_validation.csv")
upset_analysis = pd.read_csv(REPORTS / "upset_analysis.csv")
conditional_factor_importance = pd.read_csv(REPORTS / "conditional_factor_importance.csv")
conditional_mechanisms = pd.read_csv(REPORTS / "conditional_matchup_mechanism_importance.csv")
channel_importance = pd.read_csv(REPORTS / "channel_importance.csv")
kenpom_discrepancies = pd.read_csv(REPORTS / "kenpom_discrepancy_matchups.csv")

display(feature_importance.head(20))
display(group_importance.head(20))
display(interaction_importance.head(20))
display(upset_analysis)
display(channel_importance)
display(conditional_factor_importance.head(20))
display(conditional_mechanisms.head(20))
display(kenpom_discrepancies.head(20))
rolling_validation[rolling_validation["model"].eq(metrics["selected_model"])]

# %% [markdown]
# ## 12. Saved Summaries

# %%
display(Markdown((REPORTS / "summary.md").read_text()))
display(Markdown((REPORTS / "power_matchup_summary.md").read_text()))
