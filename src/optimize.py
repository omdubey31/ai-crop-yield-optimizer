from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd


@dataclass
class OptimizationResult:
    baseline_yield: float
    optimized_yield: float
    optimized_inputs: dict
    improvement_ton_ha: float
    improvement_pct: float


def optimize_inputs(model, base_features: dict) -> OptimizationResult:
    """
    Brute-force search over practical agronomic ranges for controllable inputs.
    """
    baseline_df = pd.DataFrame([base_features])
    baseline_yield = float(model.predict(baseline_df)[0])

    nitrogen_range = range(50, 201, 10)
    phosphorus_range = range(20, 121, 10)
    potassium_range = range(20, 121, 10)
    irrigation_range = range(100, 501, 20)

    best_features = dict(base_features)
    best_yield = baseline_yield

    for n, p, k, irr in product(
        nitrogen_range, phosphorus_range, potassium_range, irrigation_range
    ):
        trial = dict(base_features)
        trial["nitrogen_kg_ha"] = n
        trial["phosphorus_kg_ha"] = p
        trial["potassium_kg_ha"] = k
        trial["irrigation_mm"] = irr

        trial_df = pd.DataFrame([trial])
        predicted = float(model.predict(trial_df)[0])

        if predicted > best_yield:
            best_yield = predicted
            best_features = trial

    improvement = best_yield - baseline_yield
    improvement_pct = (improvement / baseline_yield * 100.0) if baseline_yield else 0.0

    return OptimizationResult(
        baseline_yield=baseline_yield,
        optimized_yield=best_yield,
        optimized_inputs={
            "nitrogen_kg_ha": best_features["nitrogen_kg_ha"],
            "phosphorus_kg_ha": best_features["phosphorus_kg_ha"],
            "potassium_kg_ha": best_features["potassium_kg_ha"],
            "irrigation_mm": best_features["irrigation_mm"],
        },
        improvement_ton_ha=improvement,
        improvement_pct=improvement_pct,
    )
