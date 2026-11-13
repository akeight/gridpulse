
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_power.parquet"
)


def load_hourly_data():
    """Load the processed hourly electricity dataset."""

    df = pd.read_parquet(DATA_PATH)

    df = df.sort_values("timestamp").set_index("timestamp")

    return df


def build_features():
    """Build shared features for electricity demand forecasting."""

    df = load_hourly_data()

    # Preserve hourly timestamps and exclude incomplete readings
    power = df["mean_power_kw"].where(
        df["valid_minutes"] == 60
    )

    features = pd.DataFrame(index=power.index)

    # Target variable
    features["target_kw"] = power

    # Calendar features
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month

    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    # Historical demand features
    features["lag_24"] = power.shift(24)
    features["lag_168"] = power.shift(168)

    # Leakage-safe historical rolling average
    features["rolling_mean_24"] = (
        power.shift(24).rolling(24).mean()
    )

    return features


if __name__ == "__main__":
    features = build_features()

    print("Feature dataset shape:", features.shape)
    print("\nFeature preview:")
    print(features.tail())

    print("\nMissing values:")
    print(features.isna().sum())
