"""
train_model.py
Run this once to train and save the model: python train_model.py
"""

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

np.random.seed(42)
n = 1000

sqft        = np.random.randint(500, 5000, n)
bedrooms    = np.random.randint(1, 7, n)
bathrooms   = np.random.randint(1, 5, n)
age         = np.random.randint(0, 80, n)
garage      = np.random.randint(0, 4, n)
floors      = np.random.randint(1, 4, n)
pool        = np.random.randint(0, 2, n)
location    = np.random.randint(0, 3, n)   # 0=suburban, 1=urban, 2=rural

price = (
    sqft       * 120
    + bedrooms * 8_000
    + bathrooms * 6_000
    - age      * 1_200
    + garage   * 5_000
    + floors   * 4_000
    + pool     * 15_000
    + location * 20_000
    + np.random.normal(0, 15_000, n)
)
price = np.clip(price, 50_000, 2_000_000)

X = np.column_stack([sqft, bedrooms, bathrooms, age, garage, floors, pool, location])
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                          learning_rate=0.05, random_state=42)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE      : ${mean_absolute_error(y_test, y_pred):,.0f}")

joblib.dump(pipeline, "house_price_model.joblib")
print("Model saved → house_price_model.joblib")
