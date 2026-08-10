# Factor Reference

This reference defines every base basketball factor and engineered matchup
concept used by the factor-importance model. The formulas below describe this
project's implementation exactly.

## Reading The Features

All percentages are calculated from NCAA regular-season detailed box scores and
multiplied by 100. Tournament games are stored separately and never contribute
to a team's feature snapshot.

Raw rates are converted to within-season percentile strengths before matchup
features are constructed:

- `1.00` means one of the strongest values in that season.
- `0.50` is approximately average.
- `0.00` means one of the weakest values in that season.
- Lower-is-better statistics are reversed before percentile ranking.
- A positive matchup feature favors Team A; a negative value favors Team B.

Team A is assigned by a deterministic hash rather than by winner, favorite, or
seed. This prevents the row orientation from leaking the outcome.

## Possession Estimate

Most turnover and steal rates use estimated possessions:

```text
team possessions = FGA + 0.475 * FTA - OR + TO
opponent possessions = opponent FGA + 0.475 * opponent FTA
                       - opponent OR + opponent TO
possessions = (team possessions + opponent possessions) / 2
```

Using the average of both estimates reduces box-score bookkeeping noise.

## Four Factors

### Effective Field-Goal Percentage

**Feature family:** `efg`

```text
Offensive eFG% = 100 * (FGM + 0.5 * 3PM) / FGA
Defensive eFG% = 100 * (opponent FGM + 0.5 * opponent 3PM) / opponent FGA
```

Effective field-goal percentage gives an extra half-make of credit to a made
three because a three is worth 50% more than a two. It captures shooting
accuracy and shot value in one number.

- Higher offensive eFG% is better.
- Lower defensive eFG% allowed is better.
- Offensive strength can reflect shot quality, shooting talent, spacing, and
  finishing.
- Defensive strength can reflect contest quality, rim protection, scheme, and
  forcing inefficient shot locations.

**Strength vs strength:** an elite shooting offense meets a defense that is
elite at suppressing shot value. The result indicates which side's advantage
has historically translated more strongly after accounting for the reverse
matchup.

**Strength vs weakness:** an elite shooting offense meets a defense that allows
high eFG%. This is a favorable efficiency mismatch, but three-point variance can
still make a single-game result volatile.

**Caution:** eFG% describes outcomes, not the exact shot-quality process. Two
teams can have the same eFG% with very different three-point volume and rim-shot
profiles.

### Turnover Percentage

**Feature family:** `turnover`

```text
Offensive TO% = 100 * team turnovers / possessions
Defensive TO% = 100 * opponent turnovers / possessions
```

Turnover percentage estimates how frequently a possession ends without a shot
or free-throw opportunity.

- Lower offensive TO% is better because the offense protects possessions.
- Higher defensive TO% is better because the defense ends opponent possessions.
- The separate steal and non-steal-turnover factors explain how those turnovers
  occur.

**Strength vs strength:** a low-turnover offense faces a high-turnover defense.
This is a direct contest between ball security and disruption.

**Strength vs weakness:** a careful offense faces a defense that rarely forces
turnovers, or a disruptive defense faces a careless offense. The second case
can create transition chances in addition to denying shots.

**Caution:** turnover rates depend on style and opponent pressure. A team can
accept more turnovers while creating higher-value shots on its successful
possessions.

### Offensive Rebounding Percentage

**Feature family:** `offensive_rebound`

```text
Offensive OR% = 100 * team OR / (team OR + opponent DR)
Defensive OR% allowed = 100 * opponent OR / (opponent OR + team DR)
```

Offensive rebounding percentage estimates the share of available offensive
rebounds captured. The defensive version measures the share conceded.

- Higher offensive OR% is better.
- Lower defensive OR% allowed is better.
- Offensive strength creates second-chance possessions and can draw fouls.
- Defensive strength finishes stops and limits repeated shot attempts.

**Strength vs strength:** an elite offensive rebounding team meets an elite
defensive rebounding team. This is the clearest example of a direct Four Factors
collision.

**Strength vs weakness:** a strong offensive rebounding team faces a poor
defensive rebounding team. The offense may gain extra attempts even if its
first-shot efficiency is ordinary.

**Caution:** teams that prioritize transition defense may intentionally send
fewer players to the offensive glass. OR% therefore reflects both ability and
strategic choice.

### Free-Throw Rate

**Feature family:** `free_throw_rate`

```text
Offensive FTRate = 100 * FTA / FGA
Defensive FTRate allowed = 100 * opponent FTA / opponent FGA
```

Free-throw rate measures how frequently a team earns free-throw attempts
relative to field-goal attempts.

- Higher offensive FTRate is better in the model's directional ranking.
- Lower defensive FTRate allowed is better.
- Offensive strength often reflects rim pressure, physicality, and foul drawing.
- Defensive strength reflects defending without fouling.

**Strength vs strength:** a foul-drawing offense faces a defense that avoids
fouls. The matchup can depend heavily on personnel, officiating, and whether the
offense reaches the paint.

**Strength vs weakness:** a high-FTRate offense faces a foul-prone defense,
creating a potential efficiency and lineup-depth advantage.

**Caution:** this implementation uses attempts divided by field-goal attempts.
It does not directly include free-throw accuracy, which is modeled separately.

## Miscellaneous Shooting Factors

### Three-Point Percentage

**Feature family:** `three_point`

```text
Offensive 3P% = 100 * 3PM / 3PA
Defensive 3P% allowed = 100 * opponent 3PM / opponent 3PA
```

- Higher offensive 3P% is better.
- Lower defensive 3P% allowed is better.
- It isolates perimeter conversion from two-point shooting.

**Matchup meaning:** strength vs strength compares elite perimeter shooting with
elite suppression. Strength vs weakness identifies an offense capable of
punishing a defense that has allowed efficient three-point shooting.

**Caution:** three-point percentage is volatile in small samples, and defensive
3P% can contain more shooting luck than statistics such as attempt location or
contest quality.

### Two-Point Percentage

**Feature family:** `two_point`

```text
2PM = FGM - 3PM
2PA = FGA - 3PA
Offensive 2P% = 100 * 2PM / 2PA
Defensive 2P% allowed = 100 * opponent 2PM / opponent 2PA
```

- Higher offensive 2P% is better.
- Lower defensive 2P% allowed is better.
- It combines rim finishing, post play, and midrange shooting.

**Matchup meaning:** strength vs strength often represents finishing against rim
protection or disciplined interior defense. Strength vs weakness can expose a
defense that struggles to protect the paint.

**Caution:** without shot-location data, the factor cannot distinguish rim
attempts from midrange attempts.

### Free-Throw Percentage

**Feature family:** `free_throw_pct`

```text
Offensive FT% = 100 * FTM / FTA
Defensive opponent FT% = 100 * opponent FTM / opponent FTA
```

- Higher offensive FT% is better.
- Lower opponent FT% is treated as stronger defense for consistent directional
  feature construction.
- Offensive FT% helps determine whether foul drawing becomes actual scoring.

**Matchup meaning:** offensive FT% is largely a team skill rather than a direct
interaction with the opponent. The defensive value should be interpreted with
care because defenses have limited control over opponent free-throw accuracy.

**Caution:** defensive opponent FT% may mostly reflect opponent mix and random
variation. A low importance estimate is therefore unsurprising.

## Miscellaneous Ball-Security Factors

### Block Rate

**Feature family:** `block`

```text
Offensive block rate = 100 * opponent blocks / team 2PA
Defensive block rate = 100 * team blocks / opponent 2PA
```

The offensive version measures how often a team's two-point attempts are
blocked; the defensive version measures rim disruption.

- Lower offensive block rate is better.
- Higher defensive block rate is better.

**Strength vs strength:** an offense that rarely gets blocked faces an elite
shot-blocking defense. **Strength vs weakness:** a strong rim-protecting defense
faces an offense whose two-point attempts are blocked frequently.

**Caution:** blocks capture only some rim deterrence. Great interior defenders
can alter or discourage attempts without recording a block.

### Steal Rate

**Feature family:** `steal`

```text
Offensive steal rate allowed = 100 * opponent steals / possessions
Defensive steal rate = 100 * team steals / possessions
```

- Lower offensive steal rate allowed is better.
- Higher defensive steal rate is better.
- Steals are especially valuable because they often create transition offense.

**Strength vs strength:** a secure offense faces an aggressive defense that
creates live-ball turnovers. **Strength vs weakness:** a high-steal defense
faces a team vulnerable to ball pressure.

**Caution:** aggressive steal attempts can also create defensive breakdowns;
the rate does not directly measure that tradeoff.

### Non-Steal Turnover Rate

**Feature family:** `nonsteal_turnover`

```text
Offensive non-steal TO% = 100 * max(team TO - opponent steals, 0) / possessions
Defensive non-steal TO% = 100 * max(opponent TO - team steals, 0) / possessions
```

This separates dead-ball and uncredited turnovers from steals. Examples include
travels, offensive fouls, shot-clock violations, and passes out of bounds.

- Lower offensive non-steal TO% is better.
- Higher defensive non-steal TO% is treated as better.

**Matchup meaning:** it helps distinguish general ball security from specific
vulnerability to steals. A strength-vs-weakness edge can indicate an offense
that avoids unforced mistakes against a defense that creates few non-steal
turnovers, or the reverse.

**Caution:** subtracting steals is a box-score approximation; turnover and steal
bookkeeping do not always map perfectly one to one.

## Miscellaneous Creation And Style Factors

### Assist Rate

**Feature family:** `assist`

```text
Offensive assist rate = 100 * team assists / team FGM
Defensive assist rate allowed = 100 * opponent assists / opponent FGM
```

- Higher offensive assist rate is treated as stronger creation.
- Lower opponent assist rate is treated as stronger defense.
- It describes how often made baskets are assisted, not total passing quality.

**Matchup meaning:** strength vs strength compares a high-assist offense with a
defense that suppresses assisted baskets. Strength vs weakness can identify a
passing offense facing a defense prone to losing structure.

**Caution:** assist rate is strongly affected by scorekeeper conventions,
offensive system, and shot mix. Isolation-heavy teams can be excellent offenses
with modest assist rates.

### Three-Point Attempt Rate

**Feature family:** `three_point_rate`

```text
Offensive 3PA rate = 100 * team 3PA / team FGA
Defensive 3PA rate allowed = 100 * opponent 3PA / opponent FGA
```

Three-point attempt rate is treated as style, not as inherently good or bad. Its
within-season percentile measures how perimeter-oriented a team or its opponent
shot profile is.

- High offensive rate means a larger share of shots are threes.
- High defensive rate means opponents take a larger share of shots from three.
- `three_point_rate_environment` measures the overall likelihood that a matchup
  will be three-point heavy.
- `three_point_leverage_edge` increases the impact of a three-point efficiency
  advantage when the matchup is expected to generate high three-point volume.

**Caution:** volume creates both scoring upside and single-game variance. A high
attempt rate is not automatically a strength.

## Per-Factor Engineered Matchup Features

For every strength-based factor above, let:

- `A_off` = Team A offensive strength percentile.
- `A_def` = Team A defensive strength percentile.
- `B_off` = Team B offensive strength percentile.
- `B_def` = Team B defensive strength percentile.

The model creates the following features for each factor.

### Raw Offensive And Defensive Differences

```text
raw offensive difference = direction * (Team A raw offense - Team B raw offense)
raw defensive difference = direction * (Team A raw defense - Team B raw defense)
```

These features preserve the original percentage-point distance between teams.
The direction is reversed for lower-is-better rates, so a positive value always
favors Team A. They complement percentile features: percentiles describe
relative standing within a season, while raw differences preserve whether the
underlying gap is small or large.

### Offensive Strength Difference

```text
A_off - B_off
```

Compares the teams' offensive ability in the same factor without considering
the opposing defenses.

### Defensive Strength Difference

```text
A_def - B_def
```

Compares the teams' defensive ability in the same factor.

### Overall Strength Difference

```text
(A_off + A_def - B_off - B_def) / 2
```

Summarizes two-way strength in that factor.

### Net Matchup Edge

```text
(A_off - B_def) - (B_off - A_def)
```

Compares Team A's offensive matchup with Team B's offensive matchup. Positive
values indicate the factor-level matchup favors Team A on balance.

### Net Strength Vs Strength

```text
A_off * B_def - B_off * A_def
```

Emphasizes games in which strong offense meets strong defense, then compares
that collision with the reverse side of the matchup.

### Net Strength Vs Weakness

```text
A_off * (1 - B_def) - B_off * (1 - A_def)
```

Emphasizes opportunities for a strong offense to attack a weak defense.

### Net Weakness Vs Strength

```text
(1 - A_off) * B_def - (1 - B_off) * A_def
```

Emphasizes cases in which an offensive weakness runs into an opposing defensive
strength.

## Three-Point Style Features

Because three-point attempt rate is a neutral style variable, it uses different
transformations:

- `off_style_diff`: difference between offensive 3PA-rate percentiles.
- `def_style_diff`: difference between defensive 3PA-rate-allowed percentiles.
- `expected_diff`: difference in each team's expected three-point environment.
- `environment`: average of both teams' offensive and defensive 3PA-rate
  percentiles; high values indicate a three-point-heavy matchup.

## Composite Features

### Four Factor Edge Mean

The average net matchup edge across eFG%, turnover percentage, offensive
rebounding percentage, and free-throw rate. It estimates broad matchup quality.

### Four Factor Edge Minimum And Maximum

The weakest and strongest of the four net matchup edges. The minimum captures a
potential vulnerability; the maximum captures a standout advantage.

### Four Factor Edge Standard Deviation

The dispersion of the four matchup edges. A high value means the matchup is
uneven, with pronounced strengths and weaknesses rather than a uniform edge.

### Four Factor Positive Edge Count

The number of positive Four Factor edges minus the number of negative edges. It
ranges from `-4` to `4` and measures breadth of matchup advantages.

### Possession Creation Edge

The average turnover and offensive-rebounding net edges. It summarizes which
team is more likely to create extra shot opportunities.

### Shooting Pressure Edge

The average eFG% and free-throw-rate net edges. It combines shot efficiency with
the ability to pressure the defense and reach the line.

### Perimeter Edge

The average eFG% and three-point-percentage net edges. It summarizes shooting
value with an emphasis on perimeter conversion.

### Interior Edge

The average two-point, offensive-rebounding, free-throw-rate, and block-rate net
edges. It represents paint scoring, second chances, foul pressure, and rim
protection together.

### Ball Security Edge

The average overall turnover, steal, and non-steal-turnover net edges. It
separates possession protection and disruption into related components.

### Four Factor Balance Difference

For each team, take its weakest average offense-defense percentile among the
Four Factors. The feature is Team A's weakest value minus Team B's weakest
value. It rewards teams without an obvious Four Factors deficiency.

### Four Factor Breadth Difference

Counts how many Four Factors have an average offense-defense percentile of at
least `0.67` for each team, then subtracts Team B's count from Team A's.

### Strength Vs Strength Composite

The average net strength-vs-strength term across all Four Factors and
miscellaneous strength factors.

### Weakness Exploitation Composite

The average net strength-vs-weakness term across all Four Factors and
miscellaneous strength factors.

### Three-Point Leverage Edge

```text
three-point net matchup edge * three-point-rate environment
```

This gives more weight to a perimeter-efficiency advantage when both teams'
profiles suggest that many shots will come from three.

## Interpreting Importance

The individual chart reports the increase in held-out log loss after shuffling
one feature within each validation season. The grouped chart shuffles all
features belonging to a basketball concept together.

- Larger positive importance means the fitted model relied more on the feature.
- Near-zero importance means the feature added little unique held-out signal.
- Negative importance can occur when noise or correlation makes a feature
  slightly harmful on a particular sample.
- Correlated features share predictive credit, so exact individual rankings can
  understate the importance of a broader concept.
- Importance is not causality. It does not prove that changing a factor by a
  specified amount will cause a corresponding change in win probability.
- Stability across held-out seasons and agreement across model families are more
  trustworthy than a single importance estimate.
