import streamlit as st
import pandas as pd
from src.data_utils import load_dataset
from src.train import train_yield_model
from src.optimize import optimize_inputs

st.set_page_config(page_title="AI Crop Yield Optimizer")
st.title("AI-Powered Crop Yield Prediction and Optimization")

df = load_dataset("data/sample_crop_data.csv")
artifacts = train_yield_model(df)

rainfall_mm = st.number_input("Rainfall (mm)", 100.0, 2000.0, 650.0)
temperature_c = st.number_input("Temperature (C)", 5.0, 50.0, 27.5)
soil_ph = st.number_input("Soil pH", 3.0, 10.0, 6.4)
soil_moisture_pct = st.number_input("Soil Moisture (%)", 1.0, 100.0, 28.0)
nitrogen_kg_ha = st.number_input("Nitrogen (kg/ha)", 0.0, 400.0, 90.0)
phosphorus_kg_ha = st.number_input("Phosphorus (kg/ha)", 0.0, 300.0, 45.0)
potassium_kg_ha = st.number_input("Potassium (kg/ha)", 0.0, 300.0, 40.0)
irrigation_mm = st.number_input("Irrigation (mm)", 0.0, 1000.0, 180.0)
sunlight_hours = st.number_input("Sunlight Hours", 1.0, 15.0, 7.2)
humidity_pct = st.number_input("Humidity (%)", 1.0, 100.0, 68.0)

scenario = {
    "rainfall_mm": rainfall_mm,
    "temperature_c": temperature_c,
    "soil_ph": soil_ph,
    "soil_moisture_pct": soil_moisture_pct,
    "nitrogen_kg_ha": nitrogen_kg_ha,
    "phosphorus_kg_ha": phosphorus_kg_ha,
    "potassium_kg_ha": potassium_kg_ha,
    "irrigation_mm": irrigation_mm,
    "sunlight_hours": sunlight_hours,
    "humidity_pct": humidity_pct,
}

baseline = float(artifacts.model.predict(pd.DataFrame([scenario]))[0])
result = optimize_inputs(artifacts.model, scenario)

st.write(f"Baseline Yield: {baseline:.3f} ton/ha")
st.write(f"Optimized Yield: {result.optimized_yield:.3f} ton/ha")
st.write(f"Gain: {result.improvement_ton_ha:.3f} ton/ha ({result.improvement_pct:.2f}%)")
st.json(result.optimized_inputs)
