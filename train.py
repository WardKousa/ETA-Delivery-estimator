# train.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def make_synthetic(n=8000, random_state=42):
    rng = np.random.RandomState(random_state)
    distance = rng.uniform(1, 200, size=n)  # km
    num_stops = rng.poisson(5, size=n) + 1
    pickup_hour = rng.randint(6, 20, size=n)
    is_rush = (((pickup_hour >= 7) & (pickup_hour <= 9)) | ((pickup_hour >= 16) & (pickup_hour <= 18))).astype(int)
    rain = rng.binomial(1, 0.15, size=n)
    hub_dwell = rng.exponential(scale=10, size=n)  # minutes

    travel_min = distance * 1.5
    stops_min = num_stops * rng.uniform(3, 8, size=n)
    rush_penalty = is_rush * rng.uniform(5, 20, size=n)
    rain_penalty = rain * rng.uniform(2, 10, size=n)
    dwell = hub_dwell
    noise = rng.normal(0, 10, size=n)

    eta = travel_min + stops_min + rush_penalty + rain_penalty + dwell + noise

    df = pd.DataFrame({
        'distance_km': distance,
        'num_stops': num_stops,
        'pickup_hour': pickup_hour,
        'is_rush': is_rush,
        'rain': rain,
        'hub_dwell': dwell,
        'eta_min': eta
    })
    return df

def main():
    print("Generating synthetic dataset...")
    df = make_synthetic(8000)
    df.to_csv("data/synthetic.csv", index=False)

    X = df.drop(columns=["eta_min"])
    y = df["eta_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("Training Linear Regression (baseline)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    ypred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, ypred_lr)
    print(f"Linear MAE: {mae_lr:.2f} minutes")

    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    ypred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, ypred_rf)
    print(f"RandomForest MAE: {mae_rf:.2f} minutes")

    # Global explainability: permutation importance on RF
    print("Computing permutation importances (may take a moment)...")
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = dict(zip(X.columns, perm.importances_mean.tolist()))

    # save artifacts
    joblib.dump(lr, "models/linear.pkl")
    joblib.dump(rf, "models/rf.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump({"mae_lr": mae_lr, "mae_rf": mae_rf, "importances": importances}, f, indent=2)

    print("Saved models in models/. Data saved in data/synthetic.csv")
    print("Done.")

if __name__ == "__main__":
    main()
