from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "soil_ph",
    "soil_moisture_pct",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "irrigation_mm",
    "sunlight_hours",
    "humidity_pct",
]

TARGET_COLUMN = "yield_ton_ha"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the crop dataset."""
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df
