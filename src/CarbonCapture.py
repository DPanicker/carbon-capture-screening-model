"""
Carbon Capture Candidate Classifier
====================================
End-to-end ML pipeline that classifies Alberta's industrial 
facilities as CCS (Carbon Capture & Storage) or CU (Carbon 
Utilization) candidates using Canada's GHGRP emissions data.

Author: [Your Name]
Date:   [Date]
Course: AI Pathways Technical Track — Capstone
"""

# ── Imports ─────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ── Configuration ───────────────────────────────────────
DATA_PATH       = os.path.join("src", "Capstone Dataset.csv")
TIER_THRESHOLD  = 100_000
CO2_THRESHOLD   = 0.85
RANDOM_STATE    = 42
OUTPUT_DIR      = "images"
MODEL_DIR       = "models"
OUTPUT_DIR_DOCS = "docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCS, exist_ok=True)



# ── Step 1: Load & Rename ──────────────────────────────
def load_and_rename(path):
    """Load raw GHGRP dataset and rename bilingual columns."""
    df = pd.read_csv(path)

    cols_to_keep = [
        "GHGRP ID No. / No d'identification du PDGES",
        "Reference Year / Année de référence",
        "Facility Name / Nom de l'installation",
        "Facility Province or Territory / Province ou territoire de l'installation",
        "Facility NAICS Code / Code SCIAN de l'installation",
        "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais",
        "CO2 (tonnes)",
        "CH4 (tonnes CO2e / tonnes éq. CO2)",
        "N2O (tonnes CO2e / tonnes éq. CO2)",
        "Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2)",
    ]

    rename_map = {
        "GHGRP ID No. / No d'identification du PDGES": "facility_id",
        "Reference Year / Année de référence": "year",
        "Facility Name / Nom de l'installation": "facility_name",
        "Facility Province or Territory / Province ou territoire de l'installation": "province",
        "Facility NAICS Code / Code SCIAN de l'installation": "naics_code",
        "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais": "naics_desc",
        "CO2 (tonnes)": "co2",
        "CH4 (tonnes CO2e / tonnes éq. CO2)": "ch4_co2e",
        "N2O (tonnes CO2e / tonnes éq. CO2)": "n2o_co2e",
        "Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2)": "total_emissions",
    }

    return df[cols_to_keep].rename(columns=rename_map)


# ── Step 2: Filter & Clean ─────────────────────────────
def filter_alberta(df):
    """Filter for Alberta, convert types, check duplicates."""
    df_ab = df[df["province"] == "Alberta"].copy()

    # Convert types
    df_ab["year"] = pd.to_numeric(df_ab["year"], errors="coerce").astype("Int64")
    
    for col in ["co2", "ch4_co2e", "n2o_co2e", "total_emissions"]:
        df_ab[col] = pd.to_numeric(df_ab[col], errors="coerce").astype("Float64")
    
    df_ab["naics_code"] = pd.to_numeric(df_ab["naics_code"], errors="coerce").astype("Int64")

    # Check duplicates
    exact_dupes = df_ab.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dupes}")

    return df_ab


# ── Step 3: Aggregate to Facility Level ────────────────
def aggregate_facilities(df_ab):
    """
    Collapse yearly records into one row per facility.
    Uses mean (not sum) because facilities report for 
    different numbers of years (1-20).
    """
    facility = df_ab.groupby("facility_id", as_index=False).agg(
        facility_name=("facility_name", "first"),
        naics_code=("naics_code", "first"),
        naics_desc=("naics_desc", "first"),
        avg_annual_emissions=("total_emissions", "mean"),
        years_reported=("year", "nunique"),
        avg_co2=("co2", "mean"),
        avg_ch4_co2e=("ch4_co2e", "mean"),
        avg_n2o_co2e=("n2o_co2e", "mean"),
    )

    # Calculate gas composition shares
    facility["co2_share"] = facility["avg_co2"] / facility["avg_annual_emissions"]
    facility["ch4_share"] = facility["avg_ch4_co2e"] / facility["avg_annual_emissions"]
    facility["n2o_share"] = facility["avg_n2o_co2e"] / facility["avg_annual_emissions"]

    return facility


# ── Step 4: Label Facilities ───────────────────────────
def label_facilities(facility):
    """
    Apply emission band thresholding and CCS/CU priority labels.
    co2_share is used ONLY for labeling, never as a model feature.
    """
    # Emission band
    facility["emission_band"] = np.where(
        facility["avg_annual_emissions"] >= TIER_THRESHOLD,
        "Above Threshold",
        "Below Threshold"
    )

    print("\n🏭 Emission Band Distribution:")
    print(facility["emission_band"].value_counts())

    # Filter to major emitters only
    above = facility[facility["emission_band"] == "Above Threshold"].copy()

    # Priority label
    above["priority"] = np.where(
        above["co2_share"] >= CO2_THRESHOLD,
        "CCS Candidate",
        "Potential CU Candidate"
    )

    return above


# ── Step 5: Sector-Level Summary ───────────────────────
def sector_summary(above):
    """Roll up facility labels to NAICS sector level."""
    above["naics_sector"] = above["naics_code"].astype(str).str[:2]

    sector = above.groupby("naics_sector", as_index=False).agg(
        naics_desc=("naics_desc", "first"),
        facilities=("facility_id", "count"),
        avg_sector_emissions=("avg_annual_emissions", "mean"),
        ccs_candidates=("priority", lambda x: (x == "CCS Candidate").sum()),
    )

    print("\n🏭 Sector-Level Policy Report:")
    print(sector[["naics_sector", "naics_desc", "facilities", "ccs_candidates"]])

    return sector


# ── Step 6: Feature Engineering ────────────────────────
def prepare_features(above):
    """
    Build model features. Deliberately excludes co2_share 
    to prevent data leakage (labels derived from co2_share).
    """
    above["log_emissions"] = np.log1p(above["avg_annual_emissions"])
    above["naics_sector"] = above["naics_code"].astype(str).str[:2]

    le_naics = LabelEncoder()
    above["naics_sector_encoded"] = le_naics.fit_transform(above["naics_sector"])

    feature_cols = ["log_emissions", "naics_sector_encoded", "years_reported"]
    X = above[feature_cols]
    y = above["priority"]

    print(f"\nFacilities in model:  {len(X)}")
    print(f"Features used:        {X.shape[1]}")
    print(f"Feature columns:      {X.columns.tolist()}")
    print(f"\nClass distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))

    return X, y, le_naics


# ── Step 7: Train & Evaluate ──────────────────────────
def train_and_evaluate(X, y):
    """
    Decision Tree with stratified 5-fold cross-validation.
    Chosen for interpretability over raw performance.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=2,
        min_samples_split=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Cross-validated predictions
    y_pred_cv = cross_val_predict(tree, X, y, cv=cv)

    # Classification Report
    labels = ["CCS Candidate", "Potential CU Candidate"]
    print("\nClassification Report (cross-validated predictions):")
    print(classification_report(y, y_pred_cv, target_names=labels))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_cv, labels=labels)
    print("Confusion Matrix (cross-validated predictions):")
    print(cm)
    print("Rows = Actual | Columns = Predicted")
    print(f"Labels: {labels}")

    # Fit final model on full data for visualization
    tree.fit(X, y)

    return tree, cm


# ── Step 8: Visualize & Save ──────────────────────────
def save_outputs(tree, X, cm):
    """Save decision tree plot, confusion matrix, and model."""
    labels = ["CCS Candidate", "Potential CU Candidate"]

    # Decision Tree Plot
    plt.figure(figsize=(18, 8))
    plot_tree(
        tree,
        feature_names=list(X.columns),
        class_names=labels,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree — CCS vs Potential CU Candidate", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree.png"), dpi=150)
    plt.close()
    print(f"\n✅ Saved: {OUTPUT_DIR}/decision_tree.png")

    # Feature Importance
    importance = pd.Series(tree.feature_importances_, index=X.columns)
    print("\n📊 Feature Importance:")
    print(importance.sort_values(ascending=False).to_string())

    # Save Model
    model_path = os.path.join(MODEL_DIR, "decision_tree_ccs.joblib")
    joblib.dump(tree, model_path)
    print(f"✅ Saved: {model_path}")


# ── Main Pipeline ─────────────────────────────────────
def main():
    """Execute the full ML pipeline."""
    print("=" * 55)
    print("🏭 CARBON CAPTURE CANDIDATE CLASSIFIER")
    print("=" * 55)

    # Pipeline execution
    df_clean = load_and_rename(DATA_PATH)
    df_ab = filter_alberta(df_clean)
    facility = aggregate_facilities(df_ab)
    above = label_facilities(facility)
    sector = sector_summary(above)
    X, y, le_naics = prepare_features(above)
    tree, cm = train_and_evaluate(X, y)
    save_outputs(tree, X, cm)

    # ── Use sector: Save sector report to CSV ──
    sector_path = os.path.join(OUTPUT_DIR_DOCS, "sector_report.csv")
    sector.to_csv(sector_path, index=False)
    print(f"✅ Saved: {sector_path}")

    # ── Use le_naics: Save encoder for deployment ──
    encoder_path = os.path.join(MODEL_DIR, "label_encoder_naics.joblib")
    joblib.dump(le_naics, encoder_path)
    print(f"✅ Saved: {encoder_path}")

    print("\n" + "=" * 55)
    print("✅ PIPELINE COMPLETE")
    print("=" * 55)

if __name__ == "__main__":
    main()