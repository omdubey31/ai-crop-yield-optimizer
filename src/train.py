from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .data_utils import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class TrainingArtifacts:
    model: RandomForestRegressor
    mae: float
    r2: float


def train_yield_model(df) -> TrainingArtifacts:
    """Train and evaluate a random forest model for crop yield."""
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return TrainingArtifacts(model=model, mae=mae, r2=r2)
