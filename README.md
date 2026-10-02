# Football Season-to-Season Performance Projection

A hands-on machine learning project exploring whether a football player's
G+A per 90 (goals + assists per 90 minutes) in one season can be predicted
from their stats in the prior complete season — built to learn the real
ML workflow (data → features → training → evaluation → inference), not to
build a production-grade predictor.

Dataset: ["Football Players Stats"](https://www.kaggle.com/) by
hubertsidorowicz (Kaggle, FBref-sourced), Big 5 European leagues,
seasons 2024-25, 2025-26, and 2026-27 (2026-27 in progress).

---

## Project Goal

This followed a Week 2 data-analytics project on the same dataset. The
goal here was different: build a genuine supervised ML pipeline — a model
that generalizes to players it has never seen — rather than describing
patterns in data already collected. The target question: **does a
player's prior-season performance meaningfully predict their next
season's rate, and what does it actually feel like to build, break, and
fix a model that tries to answer that?**

## Design Decisions (and what was rejected)

Two framings were considered and deliberately rejected before settling on
the final approach:

- **Predicting an unseen future season (2027-28)** — rejected. There is
  no ground truth to validate against, ever, on this project's timeline.
- **Fabricating synthetic "partial-season" snapshots** (simulating what a
  complete historical season would have looked like partway through, to
  train a current-season projector) — rejected after design review. Any
  fixed-fraction or formula-based simulation risks the label being
  trivially recoverable from the fabricated feature (a leakage trap
  disguised as a modeling choice), and the realism of invented noise can
  never be verified against real partial-season data.

**Final design:** use each player's real, complete **2024-25** season to
predict their real, complete **2025-26** season. Every value used, on
both sides, is real — nothing simulated. The trained model is then
applied to the real 2025-26 → 2026-27 transition as a genuine current
application, checked against the real (partial) 2026-27 data as a live
sanity check, not a formal evaluation.

## Data Pipeline

1. **Granularity check**: confirmed via direct inspection (not assumed
   from the Kaggle description) that the dataset is season-level player
   aggregates only — no match/gameweek-level data exists.
2. **Cleaning**: manual age/nation corrections applied before any
   grouping (grouping silently drops NaN keys). Goalkeepers excluded;
   dual-position outfielders retained.
3. **Player-level collapsing**: a player can have multiple rows in one
   season after a mid-season transfer. Stats are **summed across all
   squad rows** before computing per-90 rates — using only the
   highest-minutes ("canonical") squad's totals would silently discard
   a transfer player's output at their other club.
4. **Identity**: `Player + Nation`, since the dataset has no stable
   player ID.
5. **Supervised table**: inner join of 2024-25 (features) and 2025-26
   (label) on `Player + Nation`. Players missing from either season are
   **excluded**, never assigned a fabricated or zero target.
6. **Minimum-minutes filter**: `Min >= 450` applied to **both** seasons,
   since per-90 rates from very little playing time are unreliable as
   either a feature or a label.
7. **Position encoding**: positions include dual-role strings (`DF,MF`).
   Standard one-hot would treat `DF,MF` as a category unrelated to pure
   `DF` or `MF`. Instead, three **multi-label binary flags** (`DF`, `MF`,
   `FW`) let a dual-position player's prediction draw on both pure-role
   populations. Verified: no player had zero or all three flags set, so
   no dummy-variable-trap handling was needed (the flags don't sum to a
   constant across rows).

**Final dataset:** 1,231 players, one row each. Features:
`Gls_per90_24`, `Ast_per90_24`, `GA_per90_24`, `Age_24`, `Min_24`, `DF`,
`MF`, `FW`. Target: `GA_per90_25`. `Player`/`Nation` retained for
reference only, never fed to any model.

## Methodology Mistake — and the Fix

A real leakage mistake was made and corrected mid-project, left in this
writeup deliberately because it's the most instructive part of the week:

A Decision Tree was swept across `max_depth` values (3, 5, 7, 10,
unlimited), and **`max_depth=5` was selected because it had the best
MAE on the held-out test set.** This is invalid — using test performance
to choose between models means the test set has influenced a modeling
decision, so the resulting "test" MAE (0.1137) is not an honest estimate
of real-world performance.

**Fix:** the training set was split into a train/validation portion
(and later evaluated via 5-fold cross-validation) to select
hyperparameters using *only* training data. The honestly-selected depth
(4) was then retrained on the full training set and evaluated on the
untouched test set **exactly once**. The corrected test MAE (0.1199) was
in fact *worse* than the leaked number — direct evidence that the
original selection process was giving an overly optimistic result.

## Models and Results

| Model | Test MAE | Notes |
|---|---|---|
| Baseline (training mean) | 0.1653 | Predicts every player as the training-set average |
| Linear Regression | 0.1163 | ~30% MAE reduction vs. baseline |
| Decision Tree (unlimited depth) | 0.1604 (test) / ~0 (train) | Severe overfitting — memorized training players |
| Decision Tree (`max_depth=4`, honestly selected) | 0.1199 | Selected via train/validation split + 5-fold CV |
| **Random Forest** (`n_estimators=200`, `max_depth=6`, honestly tuned) | **0.1140** | Best of the models tested |

Hyperparameters for every model were selected using only training/
validation data or cross-validation; the test set was evaluated exactly
once per final model.

Linear Regression coefficients and the Random Forest's feature
importances **agreed**: `GA_per90_24` (prior-season rate) was by far the
strongest predictor in both.

## Real Application: Projecting 2026-27

The final Random Forest (retrained on all 1,231 historical players) was
applied to real 2025-26 → 2026-27 data:

- 2,392 outfield players appear in 2026-27 so far.
- **516** have no 2025-26 row (promoted, transferred in from outside the
  Big 5) — the model structurally cannot predict for them; this is a
  data-availability limit, not a bug.
- **653** have a 2025-26 row but under 450 minutes, and were excluded by
  the project's own minutes threshold.
- **1,223** are usable. Of those, **158** currently have enough 2026-27
  minutes (≥450) for a live sanity check against partial real results.

This comparison is **not a formal test** — 2026-27 is still in progress,
so these are partial-season actuals, not final ground truth.

**Closest predictions** (within ~0.01 of actual): Oihan Sancet, Arsène
Kouassi, Marcos Llorente, Jan Paul van Hecke, Pedri.

**Largest misses** — notably, all five were **underpredictions** of
breakout performances: Raphinha, Sergio Camello, Pascal Groß, Lamine
Yamal, Kylian Mbappé. This is a known structural limitation of tree-based
models: a Random Forest's predictions are an average over values seen in
training, so it cannot output a value more extreme than anything in its
training data — it cannot "see" a breakout coming that has no precedent
in the 2024-25 → 2025-26 transition it learned from.

A separate check (Héctor Fort, age 17, 585 minutes; Christoph
Baumgartner, age 24, 1,610 minutes — two of the largest Linear Regression
errors from the original evaluation) confirmed these misses weren't
simply a low-minutes artifact: both profiles are very different, yet
both were badly mispredicted. The features used here — prior rate, age,
position, minutes — cannot see a breakout or decline coming; that's an
honest boundary of season-level aggregate stats, not a pipeline defect.

## Limitations

- No match-level data exists in this dataset, so no in-season form,
  opposition difficulty, or injury information is available.
- The model cannot predict for any player without a usable prior season
  (transfers into the Big 5, promotions, debutants).
- 2026-27 comparisons are against a partial season and will shift as the
  season progresses.
- Tree-based models cannot extrapolate beyond their training range,
  making them structurally weak at predicting breakout performances.

## What This Project Was For

The goal was never a production-grade predictor — it was to go through
the complete real ML workflow once, including making and fixing a real
mistake: defining a target, engineering features, building a baseline,
training and comparing multiple models, deliberately inducing and
diagnosing overfitting, correctly separating validation from testing, and
applying a finished model to genuinely new data with an honest account of
where and why it fails.