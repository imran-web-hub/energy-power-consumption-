import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model/energy_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

st.title("Energy Consumption Prediction")


st.write("here is our project description")

st.write("This project predicts energy consumption of a building using Machine Learning.")
st.write("It uses factors like Temperature, Humidity, Occupancy, HVAC Usage, and Lighting Usage.")
st.write("The model used is Random Forest Regressor, trained on historical energy data.")
st.write("Tools used: Python, Pandas, Scikit-learn, Streamlit.")
st.write("This dashboard allows users to enter input values and get instant energy predictions.")

df = pd.read_csv("Energy_consumption.csv")
st.write("Dataset Preview:")
st.dataframe(df.head())

st.write("Dataset Summary:")
st.write(df.describe())

st.write("Number of Rows and Columns:", df.shape)

st.write("Column Names:")
st.write(df.columns.tolist())

st.write("Missing Values in Each Column:")
st.write(df.isnull().sum())

import matplotlib.pyplot as plt
st.write("Temperature vs Energy Consumption:")

fig, ax = plt.subplots()
ax.scatter(df["Temperature"], df["EnergyConsumption"])
ax.set_xlabel("Temperature")
ax.set_ylabel("Energy Consumption")

st.pyplot(fig)

st.write("Humidity vs Energy Consumption:")

fig2, ax2 = plt.subplots()
ax2.scatter(df["Humidity"], df["EnergyConsumption"])
ax2.set_xlabel("Humidity")
ax2.set_ylabel("Energy Consumption")

st.pyplot(fig2)

st.write("Occupancy Distribution:")

fig3, ax3 = plt.subplots()
df["Occupancy"].value_counts().sort_index().plot(kind="bar", ax=ax3)
ax3.set_xlabel("Occupancy")
ax3.set_ylabel("Count")

st.pyplot(fig3)

import seaborn as sns
st.write("Correlation Heatmap:")

fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax4)

st.pyplot(fig4)

st.write("Enter values to predict Energy Consumption:")

temperature = st.number_input("Temperature", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
square_footage = st.number_input("Square Footage", value=1000.0)
occupancy = st.number_input("Occupancy (people count)", value=1)
renewable_energy = st.number_input("Renewable Energy", value=0.0)

hvac_usage = st.selectbox("HVAC Usage", ["On", "Off"])
lighting_usage = st.selectbox("Lighting Usage", ["On", "Off"])
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
holiday = st.selectbox("Holiday", ["Yes", "No"])

if st.button("Predict Energy Consumption"):
    input_dict = {
        "Temperature": temperature,
        "Humidity": humidity,
        "SquareFootage": square_footage,
        "Occupancy": occupancy,
        "RenewableEnergy": renewable_energy,
        "HVACUsage": hvac_usage,
        "LightingUsage": lighting_usage,
        "DayOfWeek": day_of_week,
        "Holiday": holiday
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)
    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f}")

    st.write("---")
st.write("Model Performance:")

st.write("---")
st.write("Model Performance:")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# recreate the same train-test split used during training
df_encoded = pd.get_dummies(df.drop("Timestamp", axis=1), columns=["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"], drop_first=True)
X_full = df_encoded.drop("EnergyConsumption", axis=1)
y_full = df_encoded["EnergyConsumption"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_test)

st.write("Mean Absolute Error (MAE):", round(mae, 2))
st.write("Mean Squared Error (MSE):", round(mse, 2))
st.write("Root Mean Squared Error (RMSE):", round(rmse, 2))
st.write("R2 Score:", round(r2, 2))