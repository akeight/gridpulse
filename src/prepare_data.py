
"""Convert UCI household minute readings into hourly mean active power (kW)."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/household_power_consumption.txt"
DEST = ROOT / "data/processed/hourly_power.parquet"


def main():
    if not SOURCE.exists():
        raise SystemExit(
            f"Download the UCI dataset and place the extracted .txt file at {SOURCE}"
        )

    # Load raw minute-level readings.
    df = pd.read_csv(
        SOURCE,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Parse timestamps and active power.
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce",
    )

    missing_power = df["Global_active_power"].isna().sum()
    missing_timestamps = df["timestamp"].isna().sum()
    duplicate_timestamps = df["timestamp"].dropna().duplicated().sum()

    # Remove invalid observations before hourly aggregation.
    clean = (
        df.dropna(subset=["timestamp", "Global_active_power"])
        .sort_values("timestamp")
    )

    # Detect duplicates among usable readings before resampling.
    clean_duplicates = clean["timestamp"].duplicated().sum()
    if clean_duplicates:
        raise ValueError(
            f"Found {clean_duplicates:,} duplicate valid timestamps. "
            "Review the source data before aggregating."
        )

    # Aggregate minute-level readings into hourly mean power.
    hourly = (
        clean.set_index("timestamp")["Global_active_power"]
        .resample("h")
        .agg(["mean", "count"])
        .rename(
            columns={
                "mean": "mean_power_kw",
                "count": "valid_minutes",
            }
        )
        .reset_index()
    )

    # Preserve incomplete and missing hours for later analysis.
    incomplete_hours = (hourly["valid_minutes"] < 60).sum()
    missing_hours = (hourly["valid_minutes"] == 0).sum()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(DEST, index=False)

    print(f"Raw rows: {len(df):,}")
    print(f"Missing power values: {missing_power:,}")
    print(f"Invalid timestamps: {missing_timestamps:,}")
    print(f"Duplicate timestamps in raw data: {duplicate_timestamps:,}")
    print(f"Hourly rows: {len(hourly):,}")
    print(f"Incomplete hours (<60 valid minutes): {incomplete_hours:,}")
    print(f"Fully missing hours: {missing_hours:,}")
    print(f"Saved to: {DEST}")

    print("\nFirst five hourly observations:")
    print(hourly.head())


if __name__ == "__main__":
    main()

