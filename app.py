import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="AutoInsights",
    layout="centered"
)

REG_PATH = "models/best_regressor.joblib"
CLS_PATH = "models/best_classifier.joblib"
LEADERBOARD_PATH = "reports/leaderboards.json"

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    reg = joblib.load(REG_PATH)
    cls = joblib.load(CLS_PATH)
    with open(LEADERBOARD_PATH, "r") as f:
        leaderboard = json.load(f)
    return reg, cls, leaderboard

regressor, classifier, leaderboard = load_artifacts()

# -------------------------------------------------
# Helpers to select best models
# -------------------------------------------------
def get_best_regression_model(results):
    best_name, best_r2 = None, -float("inf")
    for name, info in results.items():
        if info.get("R2", -float("inf")) > best_r2:
            best_r2 = info["R2"]
            best_name = name
    return best_name, results[best_name]

def get_best_classification_model(results):
    best_name, best_acc = None, -float("inf")
    for name, info in results.items():
        if info.get("Accuracy", -float("inf")) > best_acc:
            best_acc = info["Accuracy"]
            best_name = name
    return best_name, results[best_name]

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Models & Metrics"])

# =================================================
# PAGE 1 — PREDICTION
# =================================================
if page == "Predict":
    st.title("AutoInsights")
    st.write("Predict vehicle MSRP and performance category")

    with st.form("vehicle_form"):
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Model Year", 1980, 2025, 2018)
            engine_hp = st.number_input("Engine HP", 50, 1500, 200)
            engine_cylinders = st.number_input("Engine Cylinders", 2, 16, 4)
            number_of_doors = st.selectbox("Number of Doors", [2, 4])
            engine_fuel_type = st.selectbox(
                "Engine Fuel Type",
                ["regular unleaded", "premium unleaded", "diesel", "electric", "flex-fuel"]
            )

        with col2:
            city_mpg = st.number_input("City MPG", 5, 60, 22)
            highway_mpg = st.number_input("Highway MPG", 5, 60, 30)
            transmission_type = st.selectbox(
                "Transmission Type",
                ["AUTOMATIC", "MANUAL", "AUTOMATED_MANUAL", "DIRECT_DRIVE"]
            )
            driven_wheels = st.selectbox(
                "Driven Wheels",
                ["front wheel drive", "rear wheel drive", "all wheel drive", "four wheel drive"]
            )

        make = st.text_input("Make", "Toyota")
        model = st.text_input("Model", "Camry")
        market_category = st.text_input("Market Category", "Sedan")
        vehicle_size = st.selectbox("Vehicle Size", ["Compact", "Midsize", "Large"])
        vehicle_style = st.selectbox(
            "Vehicle Style",
            ["Sedan", "SUV", "Coupe", "Convertible", "Hatchback", "Pickup", "Minivan", "Wagon"]
        )

        submit = st.form_submit_button("Predict")

    if submit:
        age = datetime.now().year - year

        X_reg = pd.DataFrame([{
            "age": age,
            "engine_hp": engine_hp,
            "engine_cylinders": engine_cylinders,
            "highway_mpg": highway_mpg,
            "city_mpg": city_mpg,
            "number_of_doors": number_of_doors,
            "engine_fuel_type": engine_fuel_type,
            "transmission_type": transmission_type,
            "driven_wheels": driven_wheels
        }])

        X_cls = pd.DataFrame([{
            "age": age,
            "engine_cylinders": engine_cylinders,
            "highway_mpg": highway_mpg,
            "city_mpg": city_mpg,
            "number_of_doors": number_of_doors,
            "make": make,
            "model": model,
            "engine_fuel_type": engine_fuel_type,
            "transmission_type": transmission_type,
            "driven_wheels": driven_wheels,
            "market_category": market_category,
            "vehicle_size": vehicle_size,
            "vehicle_style": vehicle_style
        }])

        try:
            log_price = regressor.predict(X_reg)[0]
            price = float(np.expm1(log_price))
            perf = classifier.predict(X_cls)[0]

            st.success("Prediction successful")
            st.metric("Estimated MSRP", f"${price:,.0f}")
            st.metric("Performance Category", perf)

        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)

# =================================================
# PAGE 2 — MODELS & METRICS
# =================================================
else:
    st.title("Models and Metrics")

    # ---------------- Regression ----------------
    st.header("Regression Model Used")

    reg_name, reg_info = get_best_regression_model(
        leaderboard["Regression Results"]
    )

    reg_metrics_df = pd.DataFrame([{
        "Model Family": reg_name,
        "Regressor": reg_info["Regressor"],
        "R²": round(reg_info["R2"], 3),
        "RMSE": round(reg_info["RMSE"], 2),
        "MAE": round(reg_info["MAE"], 2),
        "Kernel": reg_info["Kernel"]
    }])

    st.subheader("Performance Metrics")
    st.table(reg_metrics_df)

    reg_params_df = (
        pd.DataFrame.from_dict(reg_info["Best_Params"], orient="index")
        .reset_index()
        .rename(columns={"index": "Parameter", 0: "Value"})
    )

    st.subheader("Model Hyperparameters")
    st.table(reg_params_df)

    # ---------------- Classification ----------------
    st.header("Classification Model Used")

    cls_name, cls_info = get_best_classification_model(
        leaderboard["Classification Results"]
    )

    cls_metrics_df = pd.DataFrame([{
        "Model Family": cls_name,
        "Classifier": cls_info["Classifier"],
        "Accuracy": round(cls_info["Accuracy"], 3),
        "F1 (Macro)": round(cls_info["F1_macro"], 3),
        "Kernel": cls_info["Kernel"]
    }])

    st.subheader("Performance Metrics")
    st.table(cls_metrics_df)

    cls_params_df = (
        pd.DataFrame.from_dict(cls_info["Best_Params"], orient="index")
        .reset_index()
        .rename(columns={"index": "Parameter", 0: "Value"})
    )

    st.subheader("Model Hyperparameters")
    st.table(cls_params_df)
