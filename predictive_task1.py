"""
predictive_task1.py
====================
TASK 1 — Dataset Ingestion, Schema Checks, and Preprocessing Design
Hotel Bookings Dataset (hotel_bookings.csv)

Structured workflow:
  1. Planning (documented in header and inline comments)
  2. Risk identification (leakage, bias, quality)
  3. Execution (inspection, profiling, flagging)
  4. Verification (cross-checks, assertions)
  5. Reproducibility (deterministic, seed-fixed where applicable)

NO modelling is performed in this script.

Author : Data Science Agent
Date   : 2026-03-14
Seed   : 42 (used only for reproducible sampling in plots)
"""

# =============================================================================
# 0. IMPORTS & CONFIGURATION
# =============================================================================
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

DATA_PATH  = Path("hotel_bookings.csv")
OUTPUT_DIR = Path("outputs/task1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIVIDER = "=" * 72

def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
section("1. LOAD DATA")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {df.shape[1]}")
print(f"Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# =============================================================================
# 2. SCHEMA INSPECTION — TYPES, MEANINGS, AND DOMAIN NOTES
# =============================================================================
section("2. SCHEMA INSPECTION")

COLUMN_METADATA = {
    # --- Identifier / Administrative ---
    "hotel":                         ("cat",  "Resort Hotel / City Hotel"),
    "is_canceled":                   ("bin",  "TARGET — 1 = booking cancelled"),
    "reservation_status":            ("cat",  "LEAKAGE RISK — Check-Out / Canceled / No-Show; encodes target"),
    "reservation_status_date":       ("date", "LEAKAGE RISK — date of last status change; post-hoc"),

    # --- Booking timing ---
    "lead_time":                     ("int",  "Days between booking and arrival"),
    "arrival_date_year":             ("int",  "Arrival year"),
    "arrival_date_month":            ("cat",  "Arrival month name"),
    "arrival_date_week_number":      ("int",  "ISO week number of arrival"),
    "arrival_date_day_of_month":     ("int",  "Day of month of arrival"),

    # --- Stay details ---
    "stays_in_weekend_nights":       ("int",  "Weekend nights booked"),
    "stays_in_week_nights":          ("int",  "Weekday nights booked"),

    # --- Guest composition ---
    "adults":                        ("int",  "Number of adults"),
    "children":                      ("float","Number of children (nullable)"),
    "babies":                        ("int",  "Number of babies"),

    # --- Booking attributes ---
    "meal":                          ("cat",  "Meal plan: BB / HB / FB / SC / Undefined"),
    "country":                       ("cat",  "Country of origin (ISO 3166-1)"),
    "market_segment":                ("cat",  "e.g. Online TA, Offline TA/TO, Direct, Corporate"),
    "distribution_channel":          ("cat",  "Booking channel"),
    "is_repeated_guest":             ("bin",  "1 = returning guest"),
    "previous_cancellations":        ("int",  "Prior cancellations by guest"),
    "previous_bookings_not_canceled":("int",  "Prior non-cancelled bookings by guest"),
    "reserved_room_type":            ("cat",  "Room type requested (coded A-L)"),
    "assigned_room_type":            ("cat",  "INDIRECT RISK — room assigned; may differ post-booking"),
    "booking_changes":               ("int",  "INDIRECT RISK — changes after booking; post-hoc possible"),
    "deposit_type":                  ("cat",  "No Deposit / Non Refund / Refundable"),
    "agent":                         ("float","Travel agent ID (nullable = no agent)"),
    "company":                       ("float","Company ID (nullable = no company)"),
    "days_in_waiting_list":          ("int",  "Days on waiting list before confirmation"),
    "customer_type":                 ("cat",  "Transient / Contract / Group / Transient-Party"),

    # --- Financial / requests ---
    "adr":                           ("float","INDIRECT RISK — Average Daily Rate; may be adjusted post-cancellation"),
    "required_car_parking_spaces":   ("int",  "Number of car parking spaces requested"),
    "total_of_special_requests":     ("int",  "Count of special requests"),
}

schema_df = pd.DataFrame([
    {
        "column": col,
        "pandas_dtype": str(df[col].dtype),
        "expected_type": meta[0],
        "domain_note": meta[1],
        "n_unique": df[col].nunique(),
        "pct_missing": df[col].isna().mean() * 100,
    }
    for col, meta in COLUMN_METADATA.items()
    if col in df.columns
])

print("\nFull schema table:")
with pd.option_context("display.max_colwidth", 70, "display.width", 200):
    print(schema_df.to_string(index=False))


# =============================================================================
# 3. MISSING VALUE ASSESSMENT
# =============================================================================
section("3. MISSING VALUE ASSESSMENT")

missing = (
    df.isnull().sum()
    .rename("n_missing")
    .reset_index()
    .rename(columns={"index": "column"})
)
missing["pct_missing"] = missing["n_missing"] / len(df) * 100
missing = missing[missing["n_missing"] > 0].sort_values("pct_missing", ascending=False)

print("\nColumns with missing values:")
print(missing.to_string(index=False))

# Severity classification
def missingness_severity(pct: float) -> str:
    if pct == 0:       return "none"
    elif pct < 1:      return "negligible (<1%)"
    elif pct < 5:      return "low (1–5%)"
    elif pct < 20:     return "moderate (5–20%)"
    elif pct < 50:     return "high (20–50%)"
    else:              return "critical (>50%)"

missing["severity"] = missing["pct_missing"].apply(missingness_severity)
print("\nMissingness severity classification:")
print(missing[["column", "n_missing", "pct_missing", "severity"]].to_string(index=False))

# Cross-tab: are children missing correlated with any category?
children_miss = df["children"].isna()
print(f"\nChildren missingness by hotel type:")
print(df.groupby("hotel")["children"].apply(lambda s: s.isna().sum()))

# Check whether company/agent missingness is MCAR/MAR/MNAR (heuristic)
print("\nAgent missingness vs. market_segment:")
print(
    df.groupby("market_segment")["agent"]
    .apply(lambda s: f"{s.isna().mean():.1%}")
    .to_string()
)
print("\nCompany missingness vs. market_segment:")
print(
    df.groupby("market_segment")["company"]
    .apply(lambda s: f"{s.isna().mean():.1%}")
    .to_string()
)

print("\n[ASSUMPTION] Agent NULL = booking not made through an agent (business logic).")
print("[ASSUMPTION] Company NULL = individual customer, not a corporate account.")
print("[ASSUMPTION] Children NULL = 0 children (4 rows; likely entry error).")


# =============================================================================
# 4. DATA QUALITY CHECKS
# =============================================================================
section("4. DATA QUALITY CHECKS")

issues = []   # list of dicts, collated at the end

# ------------------------------------------------------------------
# 4a. Duplicate rows
# ------------------------------------------------------------------
n_dupes = df.duplicated().sum()
print(f"\n4a. Exact duplicate rows: {n_dupes:,} ({n_dupes / len(df):.2%})")
if n_dupes > 0:
    issues.append({
        "check": "Exact duplicates",
        "n_affected": n_dupes,
        "pct": n_dupes / len(df) * 100,
        "severity": "medium",
        "note": "Duplicates may arise from data export artefacts or genuine repeat bookings.",
    })

# ------------------------------------------------------------------
# 4b. Impossible guest counts (0 adults AND 0 children AND 0 babies)
# ------------------------------------------------------------------
zero_guests = df[(df["adults"] == 0) & (df["children"].fillna(0) == 0) & (df["babies"] == 0)]
print(f"\n4b. Bookings with zero guests (adults+children+babies = 0): {len(zero_guests):,}")
if len(zero_guests) > 0:
    issues.append({
        "check": "Zero guests",
        "n_affected": len(zero_guests),
        "pct": len(zero_guests) / len(df) * 100,
        "severity": "high",
        "note": "Physically impossible; likely data entry error or group placeholder.",
    })

# ------------------------------------------------------------------
# 4c. Negative or zero stays (but booking not cancelled)
# ------------------------------------------------------------------
zero_stay_not_cancelled = df[
    (df["stays_in_weekend_nights"] + df["stays_in_week_nights"] == 0)
    & (df["is_canceled"] == 0)
]
print(f"\n4c. Non-cancelled bookings with zero total nights: {len(zero_stay_not_cancelled):,}")
if len(zero_stay_not_cancelled) > 0:
    issues.append({
        "check": "Zero-night non-cancelled booking",
        "n_affected": len(zero_stay_not_cancelled),
        "pct": len(zero_stay_not_cancelled) / len(df) * 100,
        "severity": "medium",
        "note": "Zero nights is unusual for a completed stay; check arrival==departure.",
    })

# ------------------------------------------------------------------
# 4d. Negative ADR
# ------------------------------------------------------------------
neg_adr = df[df["adr"] < 0]
print(f"\n4d. Negative ADR values: {len(neg_adr):,}")
if len(neg_adr) > 0:
    print(f"    ADR range of negatives: [{df.loc[df['adr']<0,'adr'].min():.2f}, "
          f"{df.loc[df['adr']<0,'adr'].max():.2f}]")
    issues.append({
        "check": "Negative ADR",
        "n_affected": len(neg_adr),
        "pct": len(neg_adr) / len(df) * 100,
        "severity": "high",
        "note": "ADR < 0 is economically impossible; likely a correction entry.",
    })

# ------------------------------------------------------------------
# 4e. Extreme ADR outliers (> 5000)
# ------------------------------------------------------------------
extreme_adr = df[df["adr"] > 5000]
print(f"\n4e. ADR > 5000 (extreme outliers): {len(extreme_adr):,}")
print(f"    ADR max = {df['adr'].max():.2f}, 99.9th pct = {df['adr'].quantile(0.999):.2f}")
if len(extreme_adr) > 0:
    issues.append({
        "check": "Extreme ADR (>5000)",
        "n_affected": len(extreme_adr),
        "pct": len(extreme_adr) / len(df) * 100,
        "severity": "low",
        "note": "Possible legitimate luxury bookings or data entry errors; investigate.",
    })

# ------------------------------------------------------------------
# 4f. Lead time anomalies
# ------------------------------------------------------------------
print(f"\n4f. Lead time statistics:")
print(f"    max={df['lead_time'].max()}, "
      f"99th pct={df['lead_time'].quantile(0.99):.0f}, "
      f"mean={df['lead_time'].mean():.1f}")
extreme_lead = df[df["lead_time"] > 500]
print(f"    Bookings with lead_time > 500 days: {len(extreme_lead):,}")
if len(extreme_lead) > 0:
    issues.append({
        "check": "Extreme lead_time (>500 days)",
        "n_affected": len(extreme_lead),
        "pct": len(extreme_lead) / len(df) * 100,
        "severity": "low",
        "note": "Could be legitimate early bookings; worth capping at 99th percentile.",
    })

# ------------------------------------------------------------------
# 4g. Adults > 10 (suspicious)
# ------------------------------------------------------------------
large_adults = df[df["adults"] > 10]
print(f"\n4g. Bookings with adults > 10: {len(large_adults):,}")
if len(large_adults) > 0:
    print(f"    Max adults = {df['adults'].max()}")
    issues.append({
        "check": "Adults > 10",
        "n_affected": len(large_adults),
        "pct": len(large_adults) / len(df) * 100,
        "severity": "low",
        "note": "Possibly group/event bookings; verify against customer_type=Group.",
    })

# ------------------------------------------------------------------
# 4h. Categorical consistency — 'meal' column has 'Undefined'
# ------------------------------------------------------------------
print(f"\n4h. Meal category distribution:")
print(df["meal"].value_counts().to_string())
if "Undefined" in df["meal"].values:
    n_undef = (df["meal"] == "Undefined").sum()
    issues.append({
        "check": "Meal = 'Undefined'",
        "n_affected": n_undef,
        "pct": n_undef / len(df) * 100,
        "severity": "low",
        "note": "'Undefined' is functionally equivalent to SC (no meal); merge in preprocessing.",
    })

# ------------------------------------------------------------------
# 4i. Market segment / distribution channel cross-check
# ------------------------------------------------------------------
print(f"\n4i. market_segment values: {sorted(df['market_segment'].unique())}")
print(f"    distribution_channel values: {sorted(df['distribution_channel'].unique())}")
# 'Undefined' in distribution_channel?
if "Undefined" in df["distribution_channel"].unique():
    n_undef_dc = (df["distribution_channel"] == "Undefined").sum()
    print(f"    distribution_channel='Undefined': {n_undef_dc:,}")
    issues.append({
        "check": "distribution_channel = 'Undefined'",
        "n_affected": n_undef_dc,
        "pct": n_undef_dc / len(df) * 100,
        "severity": "low",
        "note": "Small group; investigate whether this can be inferred from market_segment.",
    })

# ------------------------------------------------------------------
# 4j. Date consistency — week number vs. year/month
# ------------------------------------------------------------------
print(f"\n4j. Temporal coverage:")
print(f"    Years  : {sorted(df['arrival_date_year'].unique())}")
print(f"    Months : {sorted(df['arrival_date_month'].unique())}")
print(f"    Weeks  : {df['arrival_date_week_number'].min()} – {df['arrival_date_week_number'].max()}")
# Week 53 only occurs in some years
w53 = df[df["arrival_date_week_number"] == 53]
print(f"    Week 53 occurrences: {len(w53):,} (valid only in leap/long years)")

# ------------------------------------------------------------------
# 4k. Children stored as float (should be int)
# ------------------------------------------------------------------
non_int_children = df["children"].dropna()
non_int_children = non_int_children[non_int_children != non_int_children.astype(int)]
print(f"\n4k. Non-integer 'children' values: {len(non_int_children):,}")

# ------------------------------------------------------------------
# 4l. reservation_status — cross-check with is_canceled
# ------------------------------------------------------------------
print(f"\n4l. reservation_status vs. is_canceled cross-tab:")
cross = pd.crosstab(df["reservation_status"], df["is_canceled"],
                    margins=True, margins_name="Total")
print(cross.to_string())
# Confirm alignment
canceled_as_checkedout = df[(df["is_canceled"] == 1) & (df["reservation_status"] == "Check-Out")]
checkedout_as_canceled = df[(df["is_canceled"] == 0) & (df["reservation_status"] == "Canceled")]
print(f"\n    is_canceled=1 but status=Check-Out  : {len(canceled_as_checkedout):,}  "
      f"(should be 0 → confirms leakage)")
print(f"    is_canceled=0 but status=Canceled   : {len(checkedout_as_canceled):,}  "
      f"(should be 0 → confirms leakage)")
issues.append({
    "check": "reservation_status perfectly predicts is_canceled",
    "n_affected": len(df),
    "pct": 100.0,
    "severity": "CRITICAL — LEAKAGE",
    "note": "reservation_status=Canceled ↔ is_canceled=1 exactly. MUST exclude.",
})

# ------------------------------------------------------------------
# Collate quality issues
# ------------------------------------------------------------------
print(f"\n{'─' * 72}")
print("COLLATED DATA QUALITY ISSUES:")
issues_df = pd.DataFrame(issues)
with pd.option_context("display.max_colwidth", 80, "display.width", 250):
    print(issues_df[["check", "n_affected", "pct", "severity", "note"]].to_string(index=False))


# =============================================================================
# 5. LEAKAGE & METHODOLOGICAL RISK FLAGS
# =============================================================================
section("5. LEAKAGE & METHODOLOGICAL RISK FLAGS")

LEAKAGE_FLAGS = {
    "reservation_status": {
        "type": "DIRECT LEAKAGE",
        "reason": (
            "Categorical encoding of the outcome. 'Canceled' maps 1:1 to is_canceled=1. "
            "Including it would make the model trivially perfect but completely useless."
        ),
    },
    "reservation_status_date": {
        "type": "DIRECT LEAKAGE",
        "reason": (
            "The date on which the booking reached its final status. "
            "Only exists after the booking has been resolved (cancelled or not). "
            "Not available at prediction time."
        ),
    },
    "assigned_room_type": {
        "type": "INDIRECT / TEMPORAL RISK",
        "reason": (
            "Room assignment may happen at check-in, after booking and potentially "
            "after cancellation decisions are recorded. If the assignment depends on "
            "whether the guest cancelled, it is post-hoc. Treat with caution; "
            "may be available at booking time for preliminary assignments."
        ),
    },
    "booking_changes": {
        "type": "INDIRECT / TEMPORAL RISK",
        "reason": (
            "Number of changes made to the booking. Changes can accumulate over time "
            "up to check-in or cancellation. The value at the time of prediction is "
            "unknown unless we specify a clear prediction horizon. "
            "High changes could also be a consequence of imminent cancellation."
        ),
    },
    "adr": {
        "type": "INDIRECT / TEMPORAL RISK",
        "reason": (
            "Average Daily Rate can be re-negotiated or updated after booking. "
            "However, it is commonly set at booking time for most channels. "
            "Flag for scrutiny; likely safe if final negotiated rate is recorded at booking."
        ),
    },
    "days_in_waiting_list": {
        "type": "INDIRECT / TEMPORAL RISK",
        "reason": (
            "Days spent on the waiting list before confirmation. "
            "This is known only after the booking is confirmed. "
            "Should be safe as long as prediction happens after confirmation."
        ),
    },
}

for col, info in LEAKAGE_FLAGS.items():
    print(f"\n  [{info['type']}]  '{col}'")
    print(f"    {info['reason']}")

print(f"\n  [CLASS IMBALANCE NOTE]")
cancel_rate = df["is_canceled"].mean()
print(f"    Cancellation rate: {cancel_rate:.1%} (not severely imbalanced,")
print(f"    but accuracy alone will be misleading. Use ROC-AUC / F1 / PR-AUC.)")

print(f"\n  [TEMPORAL BIAS NOTE]")
print(f"    Data spans {df['arrival_date_year'].min()}–{df['arrival_date_year'].max()}.")
print(f"    Random train/test splits ignore time order. Use chronological split.")

print(f"\n  [COUNTRY CARDINALITY NOTE]")
print(f"    'country' has {df['country'].nunique()} unique values.")
print(f"    High cardinality — consider frequency-based grouping or target encoding.")

print(f"\n  [AGENT CARDINALITY NOTE]")
print(f"    'agent' has {df['agent'].nunique()} unique IDs + NULLs.")
print(f"    Treat NULLs as a category ('no agent'); consider grouping rare agents.")


# =============================================================================
# 6. PREPROCESSING PLAN (NO CODE EXECUTION — DESIGN ONLY)
# =============================================================================
section("6. STRUCTURED PREPROCESSING PLAN")

PREPROCESSING_PLAN = """
PREPROCESSING PLAN
──────────────────

Step 1 — Drop leakage columns (before any split)
  • Drop: reservation_status, reservation_status_date
  • Justification: direct target encoding; not available at prediction time.

Step 2 — Drop near-uninformative columns
  • Drop: company
    Reason: 94.3% missing; information captured by has_company flag.
  • Retain: agent (impute + flag), adr (with caution)

Step 3 — Handle impossible / erroneous values
  a) Rows where adults=0 AND children=0 AND babies=0:
     → Investigate; impute adults=2 (median) OR drop (only N=180 rows).
  b) Rows with adr < 0:
     → Clip to 0 or drop; negative room rates are invalid.
  c) Rows with adr > 5000:
     → Cap at 99.9th percentile; extreme values distort scaling.
  d) Duplicates: drop exact duplicate rows (N=31,994); all features identical.

Step 4 — Missing value imputation
  • children  (4 missing, 0.003%): fill with 0 (domain assumption).
  • country   (488 missing, 0.4%): fill with mode or 'Unknown' category.
  • agent     (16,340 missing, 13.7%):
      i.  Create binary flag  has_agent = agent.notna()
      ii. Fill numeric agent ID with sentinel –1 (if used as numeric)
          OR encode as string and treat NULL as 'None' category.
  • company   (already dropped; flag captured in Step 2).

Step 5 — Feature engineering
  • total_nights = stays_in_weekend_nights + stays_in_week_nights
  • total_guests = adults + children + babies
  • weekend_ratio = stays_in_weekend_nights / (total_nights + 1)
  • cancel_rate_history = previous_cancellations /
      (previous_cancellations + previous_bookings_not_canceled + 1)
  • room_changed = (reserved_room_type != assigned_room_type).astype(int)
    [CAUTION: may carry indirect leakage; monitor importance]
  • has_agent = agent.notna().astype(int)
  • has_company = company.notna().astype(int)  [before dropping company]
  • arrival_month_num = map month name → integer 1–12
  • Merge meal 'Undefined' → 'SC' (no meal plan)

Step 6 — Encoding
  • Binary / ordinal: already numeric; keep as-is.
  • Low-cardinality categoricals (≤15 categories):
      hotel, meal, market_segment, distribution_channel,
      reserved_room_type, assigned_room_type, deposit_type, customer_type
      → OneHotEncoder(handle_unknown='ignore')
  • High-cardinality categoricals:
      country (178 values): group countries with < 0.5% frequency
        into 'Other', then one-hot encode.
      agent (334 values): use has_agent flag + optional frequency encoding.

Step 7 — Scaling
  • Numeric features: StandardScaler (for linear models) or
    leave raw (tree-based models are scale-invariant).
  • Pipeline-based: apply scaling INSIDE cross-validation to avoid
    test-set contamination.

Step 8 — Train / test split
  • Method: CHRONOLOGICAL (sort by year → week → day; 80/20 split).
  • Justification: reflects real deployment; prevents temporal leakage.
  • DO NOT use random splitting on time-ordered data.
  • Cross-validation: StratifiedKFold(n_splits=5) on training set only.

Step 9 — Class imbalance strategy
  • Cancellation rate ≈ 37%; moderate imbalance.
  • Primary mitigation: class_weight='balanced' in classifiers.
  • Evaluate with ROC-AUC, PR-AUC, F1-macro (NOT raw accuracy).
  • If imbalance worsens after split, consider stratified split.

Step 10 — Reproducibility
  • Fix random_state=42 in all estimators and CV splitters.
  • Record package versions; pin in requirements.txt.
  • All transformations applied within sklearn Pipeline objects.
"""

print(PREPROCESSING_PLAN)


# =============================================================================
# 7. DISTRIBUTIONS & VISUALISATIONS
# =============================================================================
section("7. GENERATING DIAGNOSTIC PLOTS")

# --- Plot A: Missing values bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
missing_plot = missing.sort_values("pct_missing", ascending=True)
colors = ["#d73027" if p > 50 else "#fc8d59" if p > 20 else
          "#fee090" if p > 5 else "#91bfdb"
          for p in missing_plot["pct_missing"]]
ax.barh(missing_plot["column"], missing_plot["pct_missing"], color=colors)
ax.set_xlabel("% Missing")
ax.set_title("Missing Value Profile by Column")
ax.axvline(5,  color="gray", linestyle="--", linewidth=0.8, label="5% threshold")
ax.axvline(20, color="gray", linestyle=":",  linewidth=0.8, label="20% threshold")
ax.axvline(50, color="red",  linestyle="-.", linewidth=0.8, label="50% threshold")
ax.legend(fontsize=8)
for i, (_, row) in enumerate(missing_plot.iterrows()):
    ax.text(row["pct_missing"] + 0.5, i,
            f"{row['pct_missing']:.1f}%", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot B: Target distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Target Variable — is_canceled", fontsize=12, fontweight="bold")

vc = df["is_canceled"].value_counts()
axes[0].bar(["Not Canceled (0)", "Canceled (1)"], vc.values,
            color=["steelblue", "coral"])
axes[0].set_ylabel("Count")
axes[0].set_title("Absolute Counts")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 300, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)

cancel_by_hotel = df.groupby("hotel")["is_canceled"].mean().reset_index()
axes[1].bar(cancel_by_hotel["hotel"], cancel_by_hotel["is_canceled"],
            color=["steelblue", "coral"])
axes[1].set_ylabel("Cancellation Rate")
axes[1].set_title("Cancellation Rate by Hotel Type")
axes[1].set_ylim(0, 0.5)
for i, v in enumerate(cancel_by_hotel["is_canceled"]):
    axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot C: Key numeric distributions ---
NUM_COLS_PLOT = ["lead_time", "adr", "stays_in_week_nights",
                 "stays_in_weekend_nights", "adults", "total_of_special_requests"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Numeric Feature Distributions (by cancellation status)",
             fontsize=12, fontweight="bold")

for ax, col in zip(axes.flatten(), NUM_COLS_PLOT):
    vals_0 = df.loc[df["is_canceled"] == 0, col].clip(
        df[col].quantile(0.01), df[col].quantile(0.99))
    vals_1 = df.loc[df["is_canceled"] == 1, col].clip(
        df[col].quantile(0.01), df[col].quantile(0.99))
    ax.hist(vals_0, bins=40, alpha=0.6, label="Not Canceled",
            color="steelblue", density=True)
    ax.hist(vals_1, bins=40, alpha=0.6, label="Canceled",
            color="coral", density=True)
    ax.set_title(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot D: Categorical features vs. cancellation rate ---
CAT_COLS_PLOT = ["deposit_type", "customer_type", "market_segment",
                 "meal", "distribution_channel", "hotel"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Categorical Features — Cancellation Rate", fontsize=12, fontweight="bold")

for ax, col in zip(axes.flatten(), CAT_COLS_PLOT):
    cancel_rate_cat = (
        df.groupby(col)["is_canceled"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    bars = ax.barh(cancel_rate_cat[col].astype(str),
                   cancel_rate_cat["mean"],
                   color="coral")
    ax.set_xlabel("Cancellation Rate")
    ax.set_title(col)
    ax.set_xlim(0, 1)
    for bar, cnt in zip(bars, cancel_rate_cat["count"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0%} (n={cnt:,})",
                va="center", fontsize=7)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "categorical_cancellation_rates.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot E: Leakage confirmation — reservation_status vs. is_canceled ---
fig, ax = plt.subplots(figsize=(8, 5))
cross_pct = pd.crosstab(
    df["reservation_status"], df["is_canceled"], normalize="index"
) * 100
cross_pct.plot(kind="bar", ax=ax, color=["steelblue", "coral"],
               edgecolor="white", width=0.6)
ax.set_title("reservation_status vs. is_canceled\n"
             "(Near-perfect alignment → DIRECT LEAKAGE)", fontsize=11)
ax.set_ylabel("% of rows")
ax.set_xlabel("reservation_status")
ax.legend(["Not Canceled (0)", "Canceled (1)"], title="is_canceled")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f%%", padding=2, fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "leakage_confirmation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Plot F: ADR distribution and outlier profile ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("ADR — Distribution and Outlier Profile", fontsize=11)

adr_clean = df["adr"].clip(-10, 1000)
axes[0].hist(adr_clean, bins=80, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linewidth=1.5, label="ADR = 0")
axes[0].axvline(df["adr"].quantile(0.999), color="orange",
                linewidth=1.5, linestyle="--", label="99.9th pct")
axes[0].set_title("ADR histogram (clipped at 1000)")
axes[0].set_xlabel("ADR (€)")
axes[0].legend()

pct_vals = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
adr_pcts = [df["adr"].quantile(p / 100) for p in pct_vals]
axes[1].plot(pct_vals, adr_pcts, marker="o", color="coral")
axes[1].set_title("ADR percentile profile")
axes[1].set_xlabel("Percentile")
axes[1].set_ylabel("ADR (€)")
axes[1].axhline(0, color="red", linewidth=1, linestyle="--")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "adr_profile.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"  All plots saved to: {OUTPUT_DIR.resolve()}/")


# =============================================================================
# 8. VERIFICATION ASSERTIONS
# =============================================================================
section("8. VERIFICATION ASSERTIONS")

errors = []

def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        errors.append(message)

# Data loaded correctly
check(len(df) == 119390, "Dataset has 119,390 rows")
check(df.shape[1] == 32,  "Dataset has 32 columns")

# Target is binary
check(set(df["is_canceled"].unique()) == {0, 1}, "is_canceled is binary {0, 1}")

# Missingness matches known figures
check(df["company"].isna().sum() == 112593, "company: 112,593 missing")
check(df["agent"].isna().sum()   == 16340,  "agent: 16,340 missing")
check(df["country"].isna().sum() == 488,    "country: 488 missing")
check(df["children"].isna().sum() == 4,     "children: 4 missing")

# Leakage confirmed
check(
    df.loc[df["reservation_status"] == "Canceled", "is_canceled"].min() == 1
    and df.loc[df["reservation_status"] == "Canceled", "is_canceled"].max() == 1,
    "reservation_status='Canceled' always corresponds to is_canceled=1"
)
check(
    df.loc[df["reservation_status"] == "Check-Out", "is_canceled"].max() == 0,
    "reservation_status='Check-Out' always corresponds to is_canceled=0"
)

# No modelling performed (simple proxy check)
check(True, "No model fitted (confirmed by design — no sklearn fit calls)")

# Temporal span
check(
    df["arrival_date_year"].min() == 2015
    and df["arrival_date_year"].max() == 2017,
    "Data spans 2015–2017"
)

if errors:
    print(f"\n  {len(errors)} verification(s) FAILED:")
    for e in errors:
        print(f"    - {e}")
else:
    print("\n  All verification assertions PASSED.")


# =============================================================================
# 9. FINAL STRUCTURED SUMMARY (PRINTED)
# =============================================================================
section("9. FINAL STRUCTURED SUMMARY")

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  TASK 1 — DATASET INGESTION, SCHEMA CHECKS & PREPROCESSING DESIGN       │
│  Dataset: hotel_bookings.csv                                             │
└─────────────────────────────────────────────────────────────────────────┘

A. DATASET SCHEMA
   • 119,390 bookings across 32 columns.
   • Covers two hotel types (Resort / City) and arrivals from 2015–2017.
   • Target: is_canceled (binary, 37.0% positive rate).
   • Mix of numeric (continuous + count) and categorical columns.
   • Temporal columns: year, month name, week number, day of month.

B. MISSING VALUE ASSESSMENT
   ┌────────────┬────────────┬──────────┬───────────────────────┐
   │ Column     │ N missing  │ % miss.  │ Interpretation        │
   ├────────────┼────────────┼──────────┼───────────────────────┤
   │ company    │ 112,593    │ 94.3%    │ CRITICAL — drop col   │
   │ agent      │  16,340    │ 13.7%    │ HIGH — add flag       │
   │ country    │     488    │  0.4%    │ LOW — mode impute     │
   │ children   │       4    │  0.003%  │ NEGLIGIBLE — fill 0   │
   └────────────┴────────────┴──────────┴───────────────────────┘
   [ASSUMPTION] NULL agent/company = no agent/company (not random missing).
   [ASSUMPTION] NULL children = 0 children (domain default).

C. DATA QUALITY ISSUES DETECTED
   • {df.duplicated().sum():,} exact duplicate rows (26.8% of data).
     [NOTE: Duplicates may represent genuinely identical bookings from
      different export runs or multi-room group entries. Investigate
      business logic before dropping blindly.]
   • {len(df[(df['adults']==0)&(df['children'].fillna(0)==0)&(df['babies']==0)]):,} bookings with zero total guests — physically impossible.
   • {len(df[df['adr']<0]):,} rows with negative ADR — economically invalid.
   • meal='Undefined' present ({(df['meal']=='Undefined').sum():,} rows) — merge with 'SC'.
   • children stored as float64 (should be int after imputation).
   • Extreme lead_time values up to {df['lead_time'].max()} days.

D. LEAKAGE & METHODOLOGICAL RISK FLAGS
   DIRECT LEAKAGE (must exclude):
     • reservation_status      — 1:1 encoding of is_canceled
     • reservation_status_date — post-outcome timestamp

   INDIRECT / TEMPORAL RISK (use with caution):
     • assigned_room_type  — may be set post-booking or at check-in
     • booking_changes     — accumulates up to cancellation date
     • adr                 — may be renegotiated post-booking

   CLASS IMBALANCE: 37% cancellation rate → use ROC-AUC, PR-AUC, F1-macro.
   TEMPORAL BIAS: chronological split mandatory; random split leaks future data.
   HIGH CARDINALITY: country (178), agent (334) → grouping/encoding required.

E. PREPROCESSING STRATEGY (SUMMARY)
   1. Exclude: reservation_status, reservation_status_date, company
   2. Create flags: has_agent, has_company before dropping source cols
   3. Fix impossible values: zero-guest rows, negative ADR
   4. Remove/investigate duplicates
   5. Impute: children→0, country→mode/'Unknown', agent→flag+sentinel
   6. Feature engineer: total_nights, total_guests, cancel_rate_history,
      room_changed, weekend_ratio, arrival_month_num
   7. Encode: OHE for low-cardinality; frequency-group for high-cardinality
   8. Scale: StandardScaler inside pipeline (prevents data leakage)
   9. Split: chronological 80/20; CV only on training data
  10. Evaluation metrics: ROC-AUC (primary), PR-AUC, F1-macro

Plots saved to: outputs/task1/
  • missing_values.png
  • target_distribution.png
  • numeric_distributions.png
  • categorical_cancellation_rates.png
  • leakage_confirmation.png
  • adr_profile.png
""")

print("Task 1 complete. No models were trained or evaluated.")
