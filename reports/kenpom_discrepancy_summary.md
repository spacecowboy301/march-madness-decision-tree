# Notable Model vs KenPom-Derived Discrepancies

## Scope And Caveat

This is a retrospective comparison against a probability benchmark calibrated from KenPom adjusted net-rating gaps. It is **not an official KenPom game probability**. The cached historical KenPom pages are season-end snapshots and may include tournament games, while this project's model features are confirmed pre-tournament. Treat the matchup rankings as exploratory examples, not a fair head-to-head leaderboard.

Coverage: **401 of 401 held-out games (100.0%)**.

## Overall Comparison

- Two-channel model: 65.3% accuracy, 0.730 ROC AUC, 0.602 log loss.
- KenPom-derived benchmark: 73.3% accuracy, 0.824 ROC AUC, 0.519 log loss.
- Model-only correct calls: 14; KenPom-only correct calls: 46.

## Biggest Model-Only Correct Calls

- 2021: Abilene Chr over Texas (14 over 3 upset): model 62.3%, KenPom-derived 16.7%, gap +45.6 points.
- 2021: North Texas over Purdue (13 over 4 upset): model 56.4%, KenPom-derived 20.7%, gap +35.7 points.
- 2021: Loyola-Chicago over Illinois (8 over 1 upset): model 53.0%, KenPom-derived 31.0%, gap +22.0 points.
- 2019: Murray St over Marquette (12 over 5 upset): model 57.7%, KenPom-derived 38.5%, gap +19.2 points.
- 2022: Murray St over San Francisco (7 over 10): model 61.6%, KenPom-derived 43.0%, gap +18.6 points.
- 2017: MTSU over Minnesota (12 over 5 upset): model 63.0%, KenPom-derived 44.9%, gap +18.1 points.
- 2017: Oregon over Kansas (3 over 1 upset): model 54.1%, KenPom-derived 39.5%, gap +14.6 points.
- 2018: St Bonaventure over UCLA (11 over 11): model 54.6%, KenPom-derived 41.7%, gap +13.0 points.

## Correct Calls Where The Model Assigned More Win Probability

- 2021: Drake over Wichita St (11 over 11): model 77.7%, KenPom-derived 57.0%, gap +20.7 points.
- 2018: Gonzaga over Ohio St (4 over 5): model 71.9%, KenPom-derived 56.6%, gap +15.3 points.
- 2023: FL Atlantic over Kansas St (9 over 3 upset): model 69.2%, KenPom-derived 54.4%, gap +14.8 points.
- 2021: Norfolk St over Appalachian St (16 over 16): model 60.3%, KenPom-derived 50.1%, gap +10.3 points.
- 2023: FL Atlantic over Memphis (9 over 8 upset): model 62.1%, KenPom-derived 52.5%, gap +9.5 points.
- 2017: Cincinnati over Kansas St (6 over 11): model 77.5%, KenPom-derived 69.0%, gap +8.5 points.
- 2018: Nevada over Texas (7 over 10): model 69.3%, KenPom-derived 60.9%, gap +8.5 points.
- 2019: Belmont over Temple (11 over 11): model 74.5%, KenPom-derived 67.7%, gap +6.8 points.

## Model Misses Where KenPom-Derived Was Correct

- 2021: UCLA over Abilene Chr (11 over 14): model 31.5%, KenPom-derived 87.5%, gap -56.1 points.
- 2021: Villanova over Winthrop (5 over 12): model 43.0%, KenPom-derived 91.5%, gap -48.5 points.
- 2021: Oklahoma St over Liberty (4 over 13): model 42.3%, KenPom-derived 83.6%, gap -41.4 points.
- 2022: Wisconsin over Colgate (3 over 14): model 48.1%, KenPom-derived 84.9%, gap -36.8 points.
- 2022: Arkansas over Vermont (4 over 13): model 36.0%, KenPom-derived 72.6%, gap -36.5 points.
- 2019: Texas Tech over Buffalo (3 over 6): model 48.1%, KenPom-derived 83.3%, gap -35.1 points.
- 2023: San Diego St over Col Charleston (5 over 12): model 45.3%, KenPom-derived 80.3%, gap -35.0 points.
- 2022: Providence over S Dakota St (4 over 13): model 36.1%, KenPom-derived 70.9%, gap -34.8 points.

## Most Confident Model Misses

- 2018: UMBC over Virginia (16 over 1 upset): model 6.1%, KenPom-derived 1.0%, gap +5.1 points.
- 2023: F Dickinson over Purdue (16 over 1 upset): model 8.5%, KenPom-derived 0.6%, gap +7.9 points.
- 2021: Oregon St over Loyola-Chicago (12 over 8 upset): model 10.4%, KenPom-derived 22.4%, gap -12.0 points.
- 2023: Miami FL over Houston (5 over 1 upset): model 13.2%, KenPom-derived 17.7%, gap -4.4 points.
- 2022: St Peter's over Kentucky (15 over 2 upset): model 13.8%, KenPom-derived 5.0%, gap +8.8 points.
- 2018: Syracuse over Michigan St (11 over 3 upset): model 14.4%, KenPom-derived 15.8%, gap -1.4 points.
- 2022: Arkansas over Gonzaga (4 over 1 upset): model 15.6%, KenPom-derived 14.0%, gap +1.6 points.
- 2022: St Peter's over Murray St (15 over 7 upset): model 17.3%, KenPom-derived 20.5%, gap -3.2 points.

## Reading The Gaps

A positive gap means this model assigned more probability to the eventual winner. A negative gap means the KenPom-derived benchmark did. The gap does not prove which basketball factor caused the result, and a single upset should not be treated as validation of a mechanism.
