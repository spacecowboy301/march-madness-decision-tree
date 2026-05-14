# 2026 NCAA Men's Tournament Predictions

Model: `models/decision_tree_march_madness.joblib`

Features: KenPom efficiency ratings, Four Factors, and Miscellaneous team stats joined to each tournament matchup.

Results source: NCAA.com schedule/scores pages, plus ESPN/NCAA.com confirmation for the Final Four and championship.

## Summary

- Games scored: 67
- Correct predictions: 58
- Accuracy: 86.57%
- Brier score: 0.1398
- Log loss: 0.4382

## Round Accuracy

| Round | Correct | Games | Accuracy |
| --- | ---: | ---: | ---: |
| First Four | 3 | 4 | 75.00% |
| First Round | 30 | 32 | 93.75% |
| Second Round | 14 | 16 | 87.50% |
| Sweet 16 | 6 | 8 | 75.00% |
| Elite Eight | 3 | 4 | 75.00% |
| Final Four | 1 | 2 | 50.00% |
| Championship | 1 | 1 | 100.00% |

## Missed Games

| Round | Actual Winner | Actual Loser | Score | Model Pick | Actual Winner Probability |
| --- | --- | --- | --- | --- | ---: |
| First Four | Miami OH | SMU | 89-79 | SMU | 13.84% |
| First Round | TCU | Ohio State | 66-64 | Ohio State | 32.35% |
| First Round | High Point | Wisconsin | 83-82 | Wisconsin | 13.84% |
| Second Round | Texas | Gonzaga | 74-68 | Gonzaga | 2.27% |
| Second Round | Iowa | Florida | 73-72 | Florida | 13.84% |
| Sweet 16 | Iowa | Nebraska | 77-71 | Nebraska | 26.15% |
| Sweet 16 | Tennessee | Iowa State | 76-62 | Iowa State | 32.35% |
| Elite Eight | UConn | Duke | 73-72 | Duke | 32.35% |
| Final Four | UConn | Illinois | 71-62 | Illinois | 32.35% |

Detailed predictions are saved in `reports/2026_tournament_predictions.csv`.
