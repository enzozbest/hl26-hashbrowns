import asyncio

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from hashbrowns.config import settings
from hashbrowns.ibex.client import IbexClient
from hashbrowns.ml.encoder import TARGET_NAMES, FEATURE_NAMES, encode_batch

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


async def query_db():
    async with IbexClient(settings) as client:
        return await client.applications(
            date_from="2025-01-01", date_to="2025-12-31", council_ids=[366]
        )


responses = asyncio.run(query_db())

X, y_raw = encode_batch(responses)
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = [TARGET_NAMES[i] for i in le.classes_]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# If FEATURE_NAMES isn't exported from your encoder, replace with generic names:
try:
    feature_names = FEATURE_NAMES
except Exception:
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

print(f"Analysing {len(feature_names)} features across {len(y)} samples\n")

# ---------------------------------------------------------------------------
# 1. Correlation with target
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. CORRELATION WITH TARGET (point-biserial / Pearson)")
print("=" * 60)

correlations = np.array([np.corrcoef(X_train_s[:, i], y_train)[0, 1] for i in range(X_train_s.shape[1])])
corr_df = pd.DataFrame({
    "feature": feature_names,
    "correlation": correlations,
    "abs_correlation": np.abs(correlations),
}).sort_values("abs_correlation", ascending=False)

print(corr_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Random Forest feature importance (Gini)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. RANDOM FOREST — GINI IMPORTANCE")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_s, y_train)

gini_df = pd.DataFrame({
    "feature": feature_names,
    "gini_importance": rf.feature_importances_,
}).sort_values("gini_importance", ascending=False)

print(gini_df.to_string(index=False))

rf_acc = rf.score(X_test_s, y_test)
print(f"\nRF test accuracy: {rf_acc:.3f}")

# ---------------------------------------------------------------------------
# 3. Permutation importance (model-agnostic, more reliable)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. PERMUTATION IMPORTANCE (on test set)")
print("=" * 60)

perm = permutation_importance(
    rf, X_test_s, y_test,
    n_repeats=20,
    random_state=42,
    scoring="f1_macro",
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

print(perm_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Features with near-zero variance
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. LOW VARIANCE FEATURES (candidates for removal)")
print("=" * 60)

variances = np.var(X_train, axis=0)
var_df = pd.DataFrame({
    "feature": feature_names,
    "variance": variances,
}).sort_values("variance", ascending=True)

low_var = var_df[var_df["variance"] < 0.01]
if len(low_var) > 0:
    print(f"Found {len(low_var)} near-zero variance features:")
    print(low_var.to_string(index=False))
else:
    print("No near-zero variance features found.")

# ---------------------------------------------------------------------------
# 5. Feature-feature correlation (redundancy)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. HIGHLY CORRELATED FEATURE PAIRS (redundancy, |r| > 0.8)")
print("=" * 60)

corr_matrix = np.corrcoef(X_train_s.T)
pairs = []
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        r = corr_matrix[i, j]
        if abs(r) > 0.8:
            pairs.append((feature_names[i], feature_names[j], r))

if pairs:
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])
    pairs_df = pairs_df.sort_values("correlation", key=abs, ascending=False)
    print(pairs_df.to_string(index=False))
else:
    print("No highly correlated pairs found.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY — RANKED BY PERMUTATION IMPORTANCE")
print("=" * 60)

summary = perm_df.merge(corr_df[["feature", "correlation"]], on="feature")
summary = summary.merge(gini_df, on="feature")
summary = summary.sort_values("importance_mean", ascending=False)
summary["verdict"] = summary.apply(
    lambda r: "DROP" if r["importance_mean"] < 0.001 and abs(r["correlation"]) < 0.05 else "KEEP",
    axis=1,
)

print(summary.to_string(index=False))

n_drop = (summary["verdict"] == "DROP").sum()
print(f"\nSuggested to drop: {n_drop}/{len(feature_names)} features")
print("Features to keep:", summary[summary["verdict"] == "KEEP"]["feature"].tolist())