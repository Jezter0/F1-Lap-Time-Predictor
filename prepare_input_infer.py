import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

WINDOW_SIZE = 6
NUMERIC_COLS = ["LapNumber","s1","s2","s3","TyreLife","AirTemp","TrackTemp","Rainfall"]

# These must be identical to training mappings
driver_map = joblib.load("driver_map.pkl")
team_map = joblib.load("team_map.pkl")
scaler = joblib.load("X_scaler.pkl")


def map_with_unknown(val, mapping, unknown_idx):
    return mapping.get(val, unknown_idx)


def make_windows(df):
    df = df.sort_values("LapNumber")

    X_num_list = []
    Xd_list = []
    Xt_list = []

    laps = df["LapNumber"].values
    num_mat = df[NUMERIC_COLS].values

    for i in range(len(df) - WINDOW_SIZE):
        win_laps = laps[i : i + WINDOW_SIZE + 1]

        # ensure consecutive laps
        if np.max(np.diff(win_laps)) <= 2:
            window = num_mat[i : i + WINDOW_SIZE + 1]
            if np.any(np.isnan(window)):
                continue

            X_num_list.append(window[:WINDOW_SIZE])
            Xd_list.append(df["driver_id"].iloc[i + WINDOW_SIZE])
            Xt_list.append(df["team_id"].iloc[i + WINDOW_SIZE])

    if len(X_num_list) == 0:
        return np.empty((0,WINDOW_SIZE,len(NUMERIC_COLS))), np.array([]), np.array([])

    return (
        np.stack(X_num_list),
        np.array(Xd_list),
        np.array(Xt_list)
    )


def prepare_inputs(df):
    """Prepares only inputs needed for prediction."""
    
    # Encode driver/team with saved maps
    unknown_driver = len(driver_map)
    unknown_team = len(team_map)

    df["driver_id"] = df["Driver"].apply(lambda x: map_with_unknown(x, driver_map, unknown_driver))
    df["team_id"] = df["Team"].apply(lambda x: map_with_unknown(x, team_map, unknown_team))

    # Scale numerics
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Build windows
    return make_windows(df)