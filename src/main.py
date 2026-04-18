from pathlib import Path

from .data_utils import load_dataset
from .optimize import optimize_inputs
from .train import train_yield_model


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "sample_crop_data.csv"

    df = load_dataset(data_path)
    artifacts = train_yield_model(df)

    print("=== Model Performance ===")
    print(f"MAE: {artifacts.mae:.3f} ton/ha")
    print(f"R2:  {artifacts.r2:.3f}")

    # Example base scenario (replace with real farm condition).
    current_scenario = {
        "rainfall_mm": 650,
        "temperature_c": 27.5,
        "soil_ph": 6.4,
        "soil_moisture_pct": 28,
        "nitrogen_kg_ha": 90,
        "phosphorus_kg_ha": 45,
        "potassium_kg_ha": 40,
        "irrigation_mm": 180,
        "sunlight_hours": 7.2,
        "humidity_pct": 68,
    }

    result = optimize_inputs(artifacts.model, current_scenario)

    print("\n=== Optimization Result ===")
    print(f"Baseline predicted yield:  {result.baseline_yield:.3f} ton/ha")
    print(f"Optimized predicted yield: {result.optimized_yield:.3f} ton/ha")
    print(f"Estimated gain:            {result.improvement_ton_ha:.3f} ton/ha")
    print(f"Gain percentage:           {result.improvement_pct:.2f}%")
    print("Recommended controllable inputs:")
    for key, value in result.optimized_inputs.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
