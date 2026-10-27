# Phase 5b: feature selection (fast mode, 200 bootstrap resamples)

Decided on validation only: the simplest set whose validation error is not clearly worse than the best set's (95% interval of the gain vs the best reaches 0). Test columns are shown, never used.

**Applied** (`FEATURES`): sales `attributes + photo`, price `attributes only`, by the declared rule: a block leaves only if it helps in neither check, the second being the same-month ranking on 242 (`forecast_report.md`), where the photo helps the sales ranking.

## sales: best on validation `attributes only`, chosen `attributes only`

|  | val_spearman_same_month | val_mae_log | val_within_2x | test_spearman_same_month | test_mae_log | test_within_2x | val_gain_vs_best_lo | val_gain_vs_best_hi |
|---|---|---|---|---|---|---|---|---|
| attributes only | 0.466 | 0.498 | 74.2% | 0.508 | 0.514 | 76.5% | 0.000 | 0.000 |
| attributes + photo | 0.457 | 0.511 | 74.1% | 0.502 | 0.524 | 75.7% | -0.029 | 0.005 |
| without photo and description | 0.472 | 0.524 | 72.2% | 0.492 | 0.539 | 72.9% | -0.039 | -0.014 |
| without photo | 0.461 | 0.524 | 73.1% | 0.478 | 0.540 | 73.9% | -0.041 | -0.014 |
| without launch month | 0.463 | 0.511 | 75.3% | 0.495 | 0.528 | 75.1% | -0.030 | 0.006 |
| all features | 0.451 | 0.526 | 75.1% | 0.483 | 0.536 | 74.4% | -0.046 | -0.010 |

## price: best on validation `attributes + photo`, chosen `attributes only`

|  | val_mae_eur | val_bias | val_within_1_point | test_mae_eur | test_bias | test_within_1_point | val_gain_vs_best_lo | val_gain_vs_best_hi |
|---|---|---|---|---|---|---|---|---|
| attributes only | 2.178 | -1.9% | 75.2% | 2.656 | -4.4% | 77.9% | -0.051 | 0.025 |
| attributes + photo | 2.166 | -2.3% | 75.0% | 2.623 | -4.0% | 77.8% | 0.000 | 0.000 |
| without photo and description | 2.333 | -5.4% | 70.9% | 2.611 | -4.4% | 78.4% | -0.223 | -0.115 |
| without photo | 2.407 | -5.6% | 69.9% | 2.605 | -4.5% | 77.7% | -0.305 | -0.175 |
| without launch month | 2.190 | -2.4% | 74.8% | 2.596 | -3.9% | 78.2% | -0.053 | 0.011 |
| all features | 2.339 | -5.5% | 70.2% | 2.629 | -4.7% | 77.4% | -0.238 | -0.116 |

## price: corrections for the rise of markups (232 + 241 → 242, halves of the new models)

Adopted only if the gain on half A excludes 0; half B shown, never used.

|  | A_mae_eur | A_bias | A_gain_lo | A_gain_hi | B_mae_eur | B_bias | adopted |
|---|---|---|---|---|---|---|---|
| none (chosen model) | 2.398 | -4.1% | 0.000 | 0.000 | 2.586 | -4.2% | False |
| season index (new season = latest) | 2.386 | -3.7% | -0.023 | 0.045 | 2.598 | -4.0% | False |
| season 241 weighted ×2 | 2.419 | -4.0% | -0.078 | 0.031 | 2.554 | -4.1% | False |
| season 241 weighted ×3 | 2.433 | -3.9% | -0.108 | 0.036 | 2.589 | -3.9% | False |
