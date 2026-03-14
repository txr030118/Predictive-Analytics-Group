# Hotel Booking Cancellation Prediction

Predicts whether a hotel booking will be cancelled using the
[Hotel Booking Demand dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
(119,390 bookings, 2015–2017).

## Workflow

```
1. EDA → 2. Feature Engineering → 3. Preprocessing →
4. Temporal Train/Test Split → 5. Cross-Validation →
6. Model Evaluation → 7. Feature Importance → 8. Verification
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Exclude `reservation_status` & `reservation_status_date` | Direct target leakage |
| Temporal 80/20 split | Mirrors production deployment; avoids temporal leakage |
| CV on training data only | Prevents test-set contamination |
| Permutation importance (10 repeats) | Model-agnostic; directly measures AUC impact |
| Stratified k-fold (k=5) | Preserves class ratio across folds |

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 (macro) |
|---|---|---|---|
| Logistic Regression | 0.910 ± 0.003 | 0.859 | 0.729 |
| Random Forest | 0.936 ± 0.001 | 0.871 | 0.768 |
| **Hist Gradient Boosting** | **0.948 ± 0.001** | **0.883** | **0.779** |

### Top 5 Cancellation Risk Factors

1. **Non-Refundable Deposit** — strongest single signal (Δ AUC = 0.066)
2. **Online Travel Agency channel** — OTA bookings cancel at higher rates
3. **Country = Portugal (PRT)** — domestic bookings show distinct patterns
4. **Total Special Requests** — *negative* predictor; engaged guests cancel less
5. **Lead Time** — longer booking horizon → higher cancellation probability

## Verification Checks

All 5 checks passed:
- ✓ No leakage columns in features
- ✓ Temporal split valid
- ✓ No index overlap between train and test
- ✓ CV on training data only
- ✓ Best test ROC-AUC ≥ 0.70

## Reproduce

```bash
pip install -r requirements.txt
python hotel_cancellation_prediction.py
```

All outputs are written to `outputs/`:

| File | Description |
|---|---|
| `eda_overview.png` | 6-panel EDA chart |
| `model_comparison.png` | ROC + PR curves for all models |
| `feature_importance.png` | Top-20 features by permutation importance |
| `confusion_matrix.png` | Best model confusion matrix |
| `feature_importance.csv` | Full feature importance table |
| `results_summary.json` | Machine-readable metrics |

## Limitations

- `agent` (14% missing) and `country` (0.4% missing) are median/mode imputed.
- Hyperparameters are sensible defaults, not Bayesian-optimised.
- Trained on two specific hotels; generalisation to other properties is uncertain.
- The non-refundable deposit finding is counter-intuitive and may reflect
  booking platform behaviour rather than customer intent.
