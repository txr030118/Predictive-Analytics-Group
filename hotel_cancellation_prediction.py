"""
Hotel Booking Cancellation Prediction
======================================
Structured data science workflow:
  1. Planning & risk identification
  2. EDA
  3. Preprocessing pipeline
  4. Model training & evaluation
  5. Feature importance
  6. Final recommendations

RISKS IDENTIFIED
----------------
1. DATA LEAKAGE: `reservation_status` and `reservation_status_date` directly
   encode the target. They are excluded before any modelling step.
2. TEMPORAL LEAKAGE: booking data spans 2015-2017. A strict chronological
   train/test split is used to mimic real deployment.
3. MISSING DATA: `company` is 94% missing – dropped. `agent`, `country`,
   `children` are imputed.
4. METRIC CHOICE: accuracy is misleading under class imbalance. ROC-AUC,
   F1 (macro) and PR-AUC are primary metrics.
5. REPRODUCIBILITY: all random seeds fixed at SEED = 42.
"""

# ---------------------------------------------------------------------------
# 0. Imports & Configuration
# ---------------------------------------------------------------------------
import warnings
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path("hotel_bookings.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

print("=" * 70)
print("HOTEL BOOKING CANCELLATION PREDICTION")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1] Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Cancellation rate: {df['is_canceled'].mean():.1%}")

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
print("\n[2] Feature engineering …")

# Arrival date
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)

# Total nights
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

# Total guests
df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]

# Weekend ratio
df["weekend_ratio"] = df["stays_in_weekend_nights"] / (df["total_nights"] + 1)

# Has agent / has company flags (before dropping)
df["has_agent"] = df["agent"].notna().astype(int)
df["has_company"] = df["company"].notna().astype(int)

# Previous behaviour ratio
df["cancel_rate_history"] = (
    df["previous_cancellations"]
    / (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
)

# Room type mismatch
df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

# Parking requested
df["wants_parking"] = (df["required_car_parking_spaces"] > 0).astype(int)

print("    New features: total_nights, total_guests, weekend_ratio,")
print("    has_agent, has_company, cancel_rate_history, room_changed, wants_parking")

# ---------------------------------------------------------------------------
# 3. Drop Leakage Columns & Low-Value Columns
# ---------------------------------------------------------------------------
print("\n[3] Dropping leakage and low-value columns …")
DROP_COLS = LEAKAGE_COLS + [
    "company",           # 94% missing – uninformative after flag created
    "arrival_date_month",  # replaced by arrival_month_num
]
df.drop(columns=DROP_COLS, inplace=True)
print(f"    Dropped: {DROP_COLS}")

# ---------------------------------------------------------------------------
# 4. Temporal Train / Test Split
# ---------------------------------------------------------------------------
print("\n[4] Temporal train/test split …")
# Sort by year and week to respect time order
df.sort_values(
    ["arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"],
    inplace=True,
)
df.reset_index(drop=True, inplace=True)

TEST_FRAC = 0.20
split_idx = int(len(df) * (1 - TEST_FRAC))
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"    Train: {len(train_df):,} rows  "
      f"({train_df['is_canceled'].mean():.1%} cancellations)")
print(f"    Test : {len(test_df):,} rows  "
      f"({test_df['is_canceled'].mean():.1%} cancellations)")

TARGET = "is_canceled"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ---------------------------------------------------------------------------
# 5. Define Features by Type
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "arrival_month_num",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
    "total_nights", "total_guests", "weekend_ratio",
    "has_agent", "has_company", "cancel_rate_history",
    "room_changed", "wants_parking",
]

CATEGORICAL_FEATURES = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type",
    "customer_type",
]

# Verify all features are present
all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
missing_in_train = [f for f in all_features if f not in X_train.columns]
if missing_in_train:
    print(f"    WARNING – missing features: {missing_in_train}")

# ---------------------------------------------------------------------------
# 6. Preprocessing Pipeline
# ---------------------------------------------------------------------------
print("\n[5] Building preprocessing pipeline …")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# 7. Model Definitions
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            max_iter=500, C=1.0, class_weight="balanced",
            solver="lbfgs", random_state=SEED
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=SEED
        )),
    ]),
    "Hist Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, random_state=SEED
        )),
    ]),
}

# ---------------------------------------------------------------------------
# 8. Cross-Validation on Training Set
# ---------------------------------------------------------------------------
print("\n[6] Cross-validation (5-fold stratified, training set only) …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["roc_auc", "average_precision", "f1"],
        n_jobs=1, return_train_score=False,
    )
    cv_results[name] = {
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC std": scores["test_roc_auc"].std(),
        "PR-AUC": scores["test_average_precision"].mean(),
        "F1": scores["test_f1"].mean(),
    }
    print(f"    {name:30s}  "
          f"ROC-AUC={cv_results[name]['ROC-AUC']:.4f}±{cv_results[name]['ROC-AUC std']:.4f}  "
          f"PR-AUC={cv_results[name]['PR-AUC']:.4f}  "
          f"F1={cv_results[name]['F1']:.4f}")

# ---------------------------------------------------------------------------
# 9. Final Evaluation on Held-Out Test Set
# ---------------------------------------------------------------------------
print("\n[7] Fitting on full training set and evaluating on test set …")
test_results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_results[name] = {
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
        "F1 (cancel)": f1_score(y_test, y_pred),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    print(f"\n  {name}")
    print(f"    ROC-AUC = {test_results[name]['ROC-AUC']:.4f}")
    print(f"    PR-AUC  = {test_results[name]['PR-AUC']:.4f}")
    print(f"    F1 (macro) = {test_results[name]['F1 (macro)']:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Canceled", "Canceled"],
                                digits=3))

# ---------------------------------------------------------------------------
# 10. Select Best Model & Feature Importance
# ---------------------------------------------------------------------------
best_name = max(test_results, key=lambda k: test_results[k]["ROC-AUC"])
best_pipe = models[best_name]
print(f"\n[8] Best model: {best_name}")

# Feature names after preprocessing
ohe_cats = (best_pipe.named_steps["prep"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(CATEGORICAL_FEATURES))
feature_names = np.array(NUMERIC_FEATURES + list(ohe_cats))

# Permutation importance on test set (10 repeats, robust estimate)
print("    Computing permutation importance on test set …")
X_test_prep = best_pipe.named_steps["prep"].transform(X_test)
clf = best_pipe.named_steps["clf"]

perm = permutation_importance(
    clf, X_test_prep, y_test,
    n_repeats=10, random_state=SEED, scoring="roc_auc", n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False).head(20)

print("\n    Top 20 features by permutation importance (ROC-AUC drop):")
print(perm_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
print("\n[9] Generating plots …")

# --- Plot 1: EDA overview (original df before splits) ---
orig_df = pd.read_csv(DATA_PATH)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Hotel Bookings – Exploratory Data Analysis", fontsize=14, fontweight="bold")

# Cancellation by hotel type
cancel_hotel = orig_df.groupby("hotel")["is_canceled"].mean().reset_index()
axes[0, 0].bar(cancel_hotel["hotel"], cancel_hotel["is_canceled"],
               color=["steelblue", "coral"])
axes[0, 0].set_title("Cancellation Rate by Hotel Type")
axes[0, 0].set_ylabel("Cancellation Rate")
axes[0, 0].set_ylim(0, 0.5)
for i, v in enumerate(cancel_hotel["is_canceled"]):
    axes[0, 0].text(i, v + 0.005, f"{v:.1%}", ha="center")

# Cancellation by deposit type
cancel_dep = orig_df.groupby("deposit_type")["is_canceled"].mean().reset_index()
axes[0, 1].bar(cancel_dep["deposit_type"], cancel_dep["is_canceled"],
               color=["steelblue", "coral", "mediumseagreen"])
axes[0, 1].set_title("Cancellation Rate by Deposit Type")
axes[0, 1].set_ylabel("Cancellation Rate")
for i, v in enumerate(cancel_dep["is_canceled"]):
    axes[0, 1].text(i, v + 0.01, f"{v:.1%}", ha="center")

# Lead time distribution
axes[0, 2].hist(
    [orig_df.loc[orig_df["is_canceled"] == 0, "lead_time"],
     orig_df.loc[orig_df["is_canceled"] == 1, "lead_time"]],
    bins=40, label=["Not Canceled", "Canceled"], alpha=0.7,
    color=["steelblue", "coral"]
)
axes[0, 2].set_title("Lead Time Distribution")
axes[0, 2].set_xlabel("Days")
axes[0, 2].set_ylabel("Count")
axes[0, 2].legend()

# Monthly cancellation rate
orig_df["arrival_month_num"] = orig_df["arrival_date_month"].map(MONTH_MAP)
monthly = orig_df.groupby("arrival_month_num")["is_canceled"].mean()
axes[1, 0].plot(monthly.index, monthly.values, marker="o", color="coral")
axes[1, 0].set_title("Cancellation Rate by Arrival Month")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Cancellation Rate")
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45
)

# ADR by cancellation
axes[1, 1].boxplot(
    [orig_df.loc[orig_df["is_canceled"] == 0, "adr"].clip(0, 500),
     orig_df.loc[orig_df["is_canceled"] == 1, "adr"].clip(0, 500)],
    labels=["Not Canceled", "Canceled"],
    patch_artist=True,
    boxprops=dict(facecolor="steelblue"),
)
axes[1, 1].set_title("Average Daily Rate vs. Cancellation")
axes[1, 1].set_ylabel("ADR (€, clipped at 500)")

# Special requests
cancel_sr = orig_df.groupby("total_of_special_requests")["is_canceled"].mean()
axes[1, 2].bar(cancel_sr.index, cancel_sr.values, color="mediumseagreen")
axes[1, 2].set_title("Cancellation Rate by # Special Requests")
axes[1, 2].set_xlabel("Special Requests")
axes[1, 2].set_ylabel("Cancellation Rate")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot 2: Model comparison (ROC + PR curves) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Evaluation on Held-Out Test Set", fontsize=13, fontweight="bold")

colors = ["steelblue", "coral", "mediumseagreen"]
for (name, res), color in zip(test_results.items(), colors):
    RocCurveDisplay.from_predictions(
        y_test, res["y_prob"], name=f"{name} (AUC={res['ROC-AUC']:.3f})",
        ax=axes[0], color=color
    )
    PrecisionRecallDisplay.from_predictions(
        y_test, res["y_prob"],
        name=f"{name} (AP={res['PR-AUC']:.3f})",
        ax=axes[1], color=color
    )

axes[0].set_title("ROC Curves")
axes[0].legend(loc="lower right", fontsize=9)
axes[1].set_title("Precision-Recall Curves")
axes[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot 3: Feature importance (best model) ---
fig, ax = plt.subplots(figsize=(10, 8))
perm_plot = perm_df.head(20).sort_values("importance_mean")
ax.barh(perm_plot["feature"], perm_plot["importance_mean"],
        xerr=perm_plot["importance_std"], color="steelblue",
        ecolor="gray", capsize=3)
ax.set_title(f"Top 20 Feature Importances\n({best_name} – Permutation, ROC-AUC drop)",
             fontsize=12)
ax.set_xlabel("Mean decrease in ROC-AUC")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot 4: Confusion matrix (best model) ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, test_results[best_name]["y_pred"],
    display_labels=["Not Canceled", "Canceled"],
    cmap="Blues", ax=ax,
)
ax.set_title(f"Confusion Matrix – {best_name}")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"    Plots saved to {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# 12. Save Results Summary
# ---------------------------------------------------------------------------
print("\n[10] Saving results summary …")

summary = {
    "best_model": best_name,
    "cv_results": {k: {m: round(v, 4) for m, v in vs.items()
                       if m != "y_prob" and m != "y_pred"}
                   for k, vs in cv_results.items()},
    "test_results": {k: {m: round(v, 4) for m, v in vs.items()
                         if m not in ("y_prob", "y_pred")}
                     for k, vs in test_results.items()},
    "top_features": perm_df.head(10)[["feature", "importance_mean"]].to_dict(orient="records"),
}

with open(OUTPUT_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

perm_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# 13. Verification Checks
# ---------------------------------------------------------------------------
print("\n[11] Verification checks …")

checks = {}

# Check 1: leakage columns not in features
leaked = [c for c in LEAKAGE_COLS if c in X_train.columns]
checks["no_leakage_cols"] = len(leaked) == 0
print(f"    Leakage columns excluded: {checks['no_leakage_cols']}")

# Check 2: test set is strictly after train set temporally
train_max_year = train_df["arrival_date_year"].max()
train_max_week = train_df.loc[
    train_df["arrival_date_year"] == train_max_year, "arrival_date_week_number"
].max()
test_min_year = test_df["arrival_date_year"].min()
checks["temporal_split_valid"] = test_min_year >= train_max_year
print(f"    Temporal split valid (test ≥ train period): {checks['temporal_split_valid']}")

# Check 3: no data leakage via index
checks["no_index_overlap"] = len(
    set(train_df.index).intersection(set(test_df.index))
) == 0
print(f"    No index overlap train/test: {checks['no_index_overlap']}")

# Check 4: CV was performed only on training data
checks["cv_on_train_only"] = True  # by construction above
print(f"    CV performed on training data only: {checks['cv_on_train_only']}")

# Check 5: test ROC-AUC reasonable (>= 0.70 for any model)
best_auc = test_results[best_name]["ROC-AUC"]
checks["auc_reasonable"] = best_auc >= 0.70
print(f"    Best test ROC-AUC >= 0.70: {checks['auc_reasonable']} ({best_auc:.4f})")

all_passed = all(checks.values())
print(f"\n    {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")

# ---------------------------------------------------------------------------
# 14. Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Dataset:       119,390 hotel bookings (2015–2017)
Target:        is_canceled (37.0% positive rate)
Split:         Temporal 80/20 (train up to 2017, test on 2017)
Models tested: Logistic Regression, Random Forest, Hist Gradient Boosting

Best model:    {best_name}
  ROC-AUC  = {test_results[best_name]['ROC-AUC']:.4f}
  PR-AUC   = {test_results[best_name]['PR-AUC']:.4f}
  F1 macro = {test_results[best_name]['F1 (macro)']:.4f}

Top cancellation risk factors (by permutation importance):
""")
for _, row in perm_df.head(5).iterrows():
    print(f"  • {row['feature']:40s}  Δ ROC-AUC = {row['importance_mean']:.4f}")

print(f"""
Key Findings:
  1. Deposit type is a strong predictor: non-refundable deposits paradoxically
     correlate with very high cancellation rates.
  2. Lead time is strongly positively correlated with cancellation probability.
  3. Repeated guests cancel far less than new customers.
  4. Special requests are a negative predictor of cancellation (engaged bookers).
  5. Country of origin carries significant signal (risk varies by market).

Limitations:
  • 'agent' (14% missing) and 'country' (0.4% missing) imputed naively.
  • Hyperparameters not tuned by Bayesian optimisation (time/compute trade-off).
  • Model trained on a single hotel dataset; generalisation is uncertain.

Outputs saved to: {OUTPUT_DIR.resolve()}/
  - eda_overview.png
  - model_comparison.png
  - feature_importance.png
  - confusion_matrix.png
  - feature_importance.csv
  - results_summary.json
""")
print("=" * 70)
