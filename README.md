# AI-Powered Crop Yield Prediction and Optimization

This project predicts crop yield from agronomic and weather features, then suggests practical input settings to maximize expected yield.

## What this project includes

- Supervised ML yield prediction (`RandomForestRegressor`)
- Model evaluation (MAE and R2)
- Scenario optimization for controllable inputs:
  - nitrogen (`kg/ha`)
  - phosphorus (`kg/ha`)
  - potassium (`kg/ha`)
  - irrigation (`mm`)
- Synthetic sample dataset so you can run immediately

## Project structure

```
ai-crop-yield-optimizer/
  data/
    sample_crop_data.csv
  src/
    data_utils.py
    train.py
    optimize.py
    main.py
  requirements.txt
  README.md
```

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

- Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the `ai-crop-yield-optimizer` folder:

```bash
python -m src.main
```

You will see:
- model metrics
- current farm scenario predicted yield
- optimized recommendation and estimated yield gain

## Notes

- The sample dataset is synthetic and for demonstration.
- In a real deployment, replace `data/sample_crop_data.csv` with your historical farm data and retrain.
- You can extend features with satellite indices (NDVI), pest pressure, and forecast weather.
