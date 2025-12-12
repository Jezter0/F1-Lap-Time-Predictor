import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly
import pickle
import json

from flask import Flask, render_template, request
from f1_data_loader import load_race_data, load_2025_dropdown
from helper import prepare_inputs_infer, get_driver_info
from positional_encoding import PositionalEncoding

# Load model + scaler
x_scaler = joblib.load("static/model/X_scaler.pkl")
y_scaler = joblib.load("static/model/y_scaler.pkl")
with open("static/model/id_mappings.pkl", "rb") as f:
    vocab = pickle.load(f)

races_2025, drivers_2025 = load_2025_dropdown()

app = Flask(__name__)


@app.route("/")
def index():
    drivers_info, map = get_driver_info()
    print(drivers_info)
    return render_template("index.html", races=races_2025, drivers=drivers_2025, drivers_info=drivers_info, driver_code_map=map)


@app.route("/predict", methods=["POST"])
def predict():
    race = request.form["race"]
    driver = request.form["driver"]
    model_choice = request.form["model_choice"]

    if model_choice == "lstm":
        model_path = "static/model/F1_laptime_model.keras"
    elif model_choice == "bilstm":
        model_path = "static/model/F1_laptime_model_bilstm.keras"
    elif model_choice == "gru":
        model_path = "static/model/F1_laptime_model_GRU.keras"
    elif model_choice == "transformer":
        model_path = "static/model/F1_laptime_model_transformer.keras"
        model = tf.keras.models.load_model("static/model/F1_laptime_model_transformer.keras",
                custom_objects={"PositionalEncoding": PositionalEncoding})
    
    if model_choice != "transformer":
        model = tf.keras.models.load_model(model_path, compile=False)


    df_race = load_race_data(2025, race)
    df_driver = df_race[df_race["Driver"] == driver].sort_values("LapNumber")

    # preserve original indices
    df_driver["orig_index"] = df_driver.index

    # filtering
    df_driver["Stint"] = df_driver["Stint"].astype(int)
    df_clean = df_driver[
        (df_driver["pit_flag"] == 0) &
        (df_driver["yellow_flag"] == 0) &
        (df_driver["sc_flag"] == 0) &
        (df_driver["vsc_flag"] == 0)
    ]
    df_clean = df_clean[df_clean["lap_time"] < df_clean["lap_time"].quantile(0.99)]
    df_clean = df_clean[(df_clean["lap_time"] > 40) & (df_clean["lap_time"] < 110)]
    df_clean = df_clean.reset_index(drop=True)

    print(df_clean[["LapNumber", "Stint"]].head(20))
    print(df_clean["Stint"].unique())
    print(len(df_clean))
    # model inputs
    X_num, Xd, Xt, indices = prepare_inputs_infer(df_clean, x_scaler, vocab)
    print("X_num shape:", X_num.shape)
    print("Xd shape:", Xd.shape)
    print("Xt shape:", Xt.shape)

    # prediction
    y_pred_scaled = model.predict(
        {
            "num_input": X_num,
            "driver_input": Xd,
            "team_input": Xt
        }
    ).flatten()

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # correct alignment: use df_clean
    y_true = df_clean.loc[indices, "lap_time"].values
    laps = df_clean.loc[indices, "LapNumber"].values
    laps_list = laps.tolist()
    y_true_list = y_true.tolist()
    y_pred_list = y_pred.tolist()

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps_list, y=y_true_list, mode="lines", name="True"))
    fig.add_trace(go.Scatter(x=laps_list, y=y_pred_list, mode="lines", name="Predicted"))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(graphJSON[:200])

    race_meta = df_clean.iloc[0]

    return render_template(
        "results.html",
        graphJSON=graphJSON,
        race_info=race_meta,
        driver=driver
    )
