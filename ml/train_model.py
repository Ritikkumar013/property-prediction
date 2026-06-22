"""
Property Price Predictor - XGBoost Model Training
Trains on synthetic Pune residential property data and exports
learned weights/parameters as JSON for use in the Node.js backend.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

np.random.seed(42)

# ── Pune corridor config (mirrors backend data) ──────────────────────────────

CORRIDORS = {
    "dehu-solapur": {
        "base_price_sqft": 5800,
        "demand": 1.05,
        "locations": {
            "Dehu Road": 0.85, "Pimpri-Chinchwad": 0.95, "Hinjewadi": 1.15,
            "Baner": 1.25, "Kothrud": 1.20, "Shivajinagar": 1.10,
            "Koregaon Park": 1.30, "Hadapsar": 0.90, "Solapur Road": 0.80,
        },
    },
    "kolhapur-nashik": {
        "base_price_sqft": 4200,
        "demand": 0.95,
        "locations": {
            "Katraj": 0.88, "Dhankawadi": 0.82, "Sinhagad Road": 0.95,
            "Warje": 1.05, "Khadki": 0.92, "Aundh": 1.15,
            "Vishrantwadi": 0.90, "Moshi": 0.78, "Nashik Road": 0.75,
        },
    },
}

AMENITY_LIST = [
    "gym", "swimming_pool", "clubhouse", "garden", "playground",
    "security", "power_backup", "lift", "parking", "cctv",
]

FURNISHING_TYPES = ["unfurnished", "semi-furnished", "furnished"]

BHK_SQFT_RANGES = {
    1: (350, 600),
    2: (650, 1100),
    3: (1000, 1800),
    4: (1500, 2800),
    5: (2200, 4000),
}


def generate_synthetic_data(n_samples=10000):
    """Generate realistic Pune property data for training."""
    records = []

    for _ in range(n_samples):
        corridor_key = np.random.choice(list(CORRIDORS.keys()))
        corridor = CORRIDORS[corridor_key]

        location_name = np.random.choice(list(corridor["locations"].keys()))
        location_mult = corridor["locations"][location_name]

        bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.35, 0.30, 0.18, 0.07])
        sqft_min, sqft_max = BHK_SQFT_RANGES[bhk]
        sqft = np.random.randint(sqft_min, sqft_max + 1)

        bathrooms = min(bhk, max(1, bhk - np.random.choice([0, 1], p=[0.6, 0.4])))
        total_floors = np.random.choice([5, 7, 10, 12, 15, 20, 25])
        floor = np.random.randint(0, total_floors + 1)
        property_age = np.random.choice(
            [0, 1, 2, 3, 5, 7, 10, 15, 20, 25],
            p=[0.15, 0.12, 0.12, 0.10, 0.12, 0.10, 0.10, 0.08, 0.06, 0.05],
        )
        furnishing = np.random.choice(FURNISHING_TYPES, p=[0.45, 0.35, 0.20])
        parking = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])

        n_amenities = np.random.randint(0, len(AMENITY_LIST) + 1)
        amenities = list(np.random.choice(AMENITY_LIST, size=n_amenities, replace=False))

        # ── Calculate realistic price ──
        base = corridor["base_price_sqft"] * sqft * location_mult

        bhk_factors = {1: 0.75, 2: 1.0, 3: 1.25, 4: 1.55, 5: 1.85}
        base *= bhk_factors[bhk]

        # Floor premium
        if floor <= 0:
            base *= 0.95
        elif floor / total_floors <= 0.3:
            base *= 1.0
        elif floor / total_floors <= 0.6:
            base *= 1.03
        elif floor / total_floors <= 0.85:
            base *= 1.06
        else:
            base *= 1.04

        # Age depreciation
        age_factors = {0: 1.10, 1: 1.05, 2: 1.05, 3: 1.0, 5: 1.0,
                       7: 0.92, 10: 0.92, 15: 0.82, 20: 0.82, 25: 0.70}
        base *= age_factors.get(property_age, 0.85)

        # Bathroom factor
        if bathrooms >= bhk:
            base *= 1.04
        else:
            base *= 1.0

        # Furnishing
        furn_mult = {"unfurnished": 1.0, "semi-furnished": 1.08, "furnished": 1.18}
        base *= furn_mult[furnishing]

        base *= (1 + parking * 0.02)
        base *= corridor["demand"]

        # Amenity bonus
        amenity_weights = {
            "gym": 0.020, "swimming_pool": 0.035, "clubhouse": 0.025,
            "garden": 0.015, "playground": 0.010, "security": 0.020,
            "power_backup": 0.015, "lift": 0.020, "parking": 0.015, "cctv": 0.010,
        }
        amenity_bonus = sum(amenity_weights.get(a, 0) for a in amenities)
        base *= (1 + amenity_bonus)

        # Add realistic noise (±8%)
        noise = np.random.uniform(0.92, 1.08)
        price_lakhs = round((base * noise) / 100000, 2)

        records.append({
            "corridor": corridor_key,
            "bhk": bhk,
            "sqft": sqft,
            "bathrooms": bathrooms,
            "floor": floor,
            "total_floors": total_floors,
            "property_age": property_age,
            "furnishing": furnishing,
            "parking": parking,
            "amenity_count": len(amenities),
            "has_gym": int("gym" in amenities),
            "has_pool": int("swimming_pool" in amenities),
            "has_clubhouse": int("clubhouse" in amenities),
            "has_garden": int("garden" in amenities),
            "has_playground": int("playground" in amenities),
            "has_security": int("security" in amenities),
            "has_power_backup": int("power_backup" in amenities),
            "has_lift": int("lift" in amenities),
            "has_parking": int("parking" in amenities),
            "has_cctv": int("cctv" in amenities),
            "floor_ratio": round(floor / max(total_floors, 1), 4),
            "price_lakhs": price_lakhs,
        })

    return pd.DataFrame(records)


def train_and_export():
    print("=" * 60)
    print("Property Price Predictor - XGBoost Training")
    print("=" * 60)
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n[1/5] Generating 10,000 synthetic Pune property samples...")
    df = generate_synthetic_data(10000)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Price range: Rs {df['price_lakhs'].min():.2f}L - Rs {df['price_lakhs'].max():.2f}L")
    print(f"  Mean price:  Rs {df['price_lakhs'].mean():.2f}L")

    # ── Encode categorical features ──
    print("\n[2/5] Encoding features...")
    le_corridor = LabelEncoder()
    df["corridor_enc"] = le_corridor.fit_transform(df["corridor"])

    le_furnishing = LabelEncoder()
    df["furnishing_enc"] = le_furnishing.fit_transform(df["furnishing"])

    feature_cols = [
        "corridor_enc", "bhk", "sqft", "bathrooms", "floor", "total_floors",
        "property_age", "furnishing_enc", "parking", "amenity_count",
        "has_gym", "has_pool", "has_clubhouse", "has_garden", "has_playground",
        "has_security", "has_power_backup", "has_lift", "has_parking",
        "has_cctv", "floor_ratio",
    ]

    X = df[feature_cols]
    y = df["price_lakhs"]

    # ── Split ──
    print("\n[3/5] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ── Train XGBoost ──
    print("\n[4/5] Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE:  Rs {mae:.2f} Lakhs")
    print(f"  R2:   {r2:.4f}")

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance (top 10):")
    for feat, imp in sorted_imp[:10]:
        bar = "#" * int(imp * 50)
        print(f"    {feat:20s} {imp:.4f} {bar}")

    # ── Export model weights ──
    print("\n[5/5] Exporting model to JSON...")

    # Save the raw XGBoost model as JSON (tree dump)
    model_json_path = os.path.join(os.path.dirname(__file__), "xgboost_model.json")
    model.save_model(model_json_path)

    # Export metadata + encodings for the JS backend
    export_data = {
        "model_info": {
            "algorithm": "XGBoost (XGBRegressor)",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "training_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
        },
        "feature_columns": feature_cols,
        "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
        "encodings": {
            "corridor": dict(zip(
                le_corridor.classes_.tolist(),
                le_corridor.transform(le_corridor.classes_).tolist(),
            )),
            "furnishing": dict(zip(
                le_furnishing.classes_.tolist(),
                le_furnishing.transform(le_furnishing.classes_).tolist(),
            )),
        },
        "training_stats": {
            "price_mean": round(float(y.mean()), 4),
            "price_std": round(float(y.std()), 4),
            "price_min": round(float(y.min()), 4),
            "price_max": round(float(y.max()), 4),
        },
        "corridor_base_prices": {
            k: v["base_price_sqft"] for k, v in CORRIDORS.items()
        },
    }

    weights_path = os.path.join(os.path.dirname(__file__), "model_weights.json")
    with open(weights_path, "w") as f:
        json.dump(export_data, f, indent=2)

    # ── Also export prediction calibration data ──
    # Generate predictions for standard property configs so JS can interpolate
    calibration_data = []
    for corridor_key in CORRIDORS:
        corridor_enc = le_corridor.transform([corridor_key])[0]
        for bhk in [1, 2, 3, 4, 5]:
            for sqft in [500, 800, 1000, 1200, 1500, 2000, 2500, 3000]:
                for furn in FURNISHING_TYPES:
                    furn_enc = le_furnishing.transform([furn])[0]
                    row = pd.DataFrame([{
                        "corridor_enc": corridor_enc,
                        "bhk": bhk,
                        "sqft": sqft,
                        "bathrooms": max(1, bhk - 1),
                        "floor": 5,
                        "total_floors": 10,
                        "property_age": 2,
                        "furnishing_enc": furn_enc,
                        "parking": 1,
                        "amenity_count": 3,
                        "has_gym": 1, "has_pool": 0, "has_clubhouse": 0,
                        "has_garden": 0, "has_playground": 0, "has_security": 1,
                        "has_power_backup": 1, "has_lift": 0, "has_parking": 0,
                        "has_cctv": 0,
                        "floor_ratio": 0.5,
                    }])
                    pred = model.predict(row)[0]
                    calibration_data.append({
                        "corridor": corridor_key,
                        "bhk": bhk,
                        "sqft": sqft,
                        "furnishing": furn,
                        "predicted_price": round(float(pred), 2),
                    })

    calibration_path = os.path.join(os.path.dirname(__file__), "calibration_data.json")
    with open(calibration_path, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\n  Model saved:       {model_json_path}")
    print(f"  Weights exported:  {weights_path}")
    print(f"  Calibration data:  {calibration_path}")
    print(f"  Calibration points: {len(calibration_data)}")
    print(f"\n{'=' * 60}")
    print("Training complete! Files ready for Node.js backend.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_and_export()
