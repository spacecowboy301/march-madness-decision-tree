# 2026 NCAA Men's Tournament Predictions

Model: `models/decision_tree_march_madness.joblib`

Features: public KenPom 2026 efficiency ratings joined to each tournament matchup.

Results source: NCAA.com schedule/scores pages, plus ESPN/NCAA.com confirmation for the Final Four and championship.

## Summary

- Games scored: 67
- Correct predictions: 51
- Accuracy: 76.12%
- Brier score: 0.1570
- Log loss: 0.9530

## Round Accuracy

| Round | Correct | Games | Accuracy |
| --- | ---: | ---: | ---: |
| First Four | 1 | 4 | 25.00% |
| First Round | 28 | 32 | 87.50% |
| Second Round | 11 | 16 | 68.75% |
| Sweet 16 | 6 | 8 | 75.00% |
| Elite Eight | 3 | 4 | 75.00% |
| Final Four | 1 | 2 | 50.00% |
| Championship | 1 | 1 | 100.00% |

## Missed Games

| Round | Actual Winner | Actual Loser | Score | Model Pick | Actual Winner Probability |
| --- | --- | --- | --- | --- | ---: |
| First Four | Texas | NC State | 68-66 | NC State | 47.37% |
| First Four | Prairie View A&M | Lehigh | 67-55 | Lehigh | 47.37% |
| First Four | Miami OH | SMU | 89-79 | SMU | 13.10% |
| First Round | TCU | Ohio State | 66-64 | Ohio State | 32.04% |
| First Round | High Point | Wisconsin | 83-82 | Wisconsin | 13.10% |
| First Round | Kentucky | Santa Clara | 89-84 | Santa Clara | 47.37% |
| First Round | Utah State | Villanova | 86-76 | Villanova | 47.37% |
| Second Round | Texas | Gonzaga | 74-68 | Gonzaga | 0.00% |
| Second Round | St. John's | Kansas | 67-65 | Kansas | 47.37% |
| Second Round | Tennessee | Virginia | 79-72 | Virginia | 47.37% |
| Second Round | Iowa | Florida | 73-72 | Florida | 13.10% |
| Second Round | Alabama | Texas Tech | 90-65 | Texas Tech | 47.37% |
| Sweet 16 | Iowa | Nebraska | 77-71 | Nebraska | 24.74% |
| Sweet 16 | Tennessee | Iowa State | 76-62 | Iowa State | 32.04% |
| Elite Eight | UConn | Duke | 73-72 | Duke | 32.04% |
| Final Four | UConn | Illinois | 71-62 | Illinois | 32.04% |

Detailed predictions are saved in `reports/2026_tournament_predictions.csv`.
