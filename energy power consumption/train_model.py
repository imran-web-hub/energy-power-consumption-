# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

df = pd.read_csv("Energy_consumption.csv")
print(df.head())
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nColumns in Dataset:")
print(df.columns.tolist())

df = pd.get_dummies(df, columns=["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"], drop_first=True)


X = df.drop(["Timestamp", "EnergyConsumption"], axis=1)

y = df["EnergyConsumption"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")



y_pred = model.predict(X_test)

print("\nPrediction Completed Successfully!")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

print(X.columns.tolist())

joblib.dump(model, "model/energy_model.pkl")
joblib.dump(X.columns.tolist(), "model/model_columns.pkl")

print("\nModel Saved Successfully!")