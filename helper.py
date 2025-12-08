import numpy as np

NUMERIC_COLS = ["LapNumber", "s1", "s2", "s3", "TyreLife", "AirTemp", "TrackTemp", "Rainfall"]

def prepare_inputs_infer(df, x_scaler, vocab, window_size=6):

    df = df.copy()

    df["driver_id"] = df["Driver"].map(lambda x: vocab["driver_map"].get(x, vocab["driver_unk"]))
    df["team_id"]   = df["Team"].map(lambda x: vocab["team_map"].get(x, vocab["team_unk"]))

    df[NUMERIC_COLS] = x_scaler.transform(df[NUMERIC_COLS].to_numpy())

    groups = df.groupby(["race", "driver_id", "Stint"], sort=False)

    X_num_list = []
    Xd_list = []
    Xt_list = []
    idx_list = []

    for _, g in groups:
        g = g.sort_values("LapNumber")
        laps = g["LapNumber"].values
        num_mat = g[NUMERIC_COLS].values

        for i in range(len(g) - window_size):
            window_laps = laps[i:i+window_size+1]
            if np.max(np.diff(window_laps)) > 2:
                continue

            X_num_list.append(num_mat[i:i+window_size])
            Xd_list.append(g["driver_id"].iloc[i])
            Xt_list.append(g["team_id"].iloc[i])
            idx_list.append(g.index[i+window_size])

    if len(X_num_list) == 0:
        return (np.empty((0,window_size,len(NUMERIC_COLS))),
                np.empty((0,)), np.empty((0,)),
                np.empty((0,), dtype=int))

    return (
        np.stack(X_num_list),
        np.array(Xd_list),
        np.array(Xt_list),
        np.array(idx_list)
    )

def get_driver_info():
    drivers_info = {
    "Pierre Gasly": {
        "number": 10,
        "team": "Alpine",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-52 Pierre Gasly - F1 Driver for Alpine.png",
        "team_color": "#00BFFF"
    },
    "Franco Colapinto": {
        "number": 43,
        "team": "Alpine",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-56 Franco Colapinto - F1 Driver for Alpine.png",
        "team_color": "#00BFFF"
    },
    "Fernando Alonso": {
        "number": 14,
        "team": "Aston Martin",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-28 Fernando Alonso - F1 Driver for Aston Martin.png",
        "team_color": "#006F62"
    },
    "Lance Stroll": {
        "number": 18,
        "team": "Aston Martin",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-24 Lance Stroll - F1 Driver for Aston Martin.png",
        "team_color": "#006F62"
    },
    "Charles Leclerc": {
        "number": 16,
        "team": "Ferrari",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-43 Charles Leclerc - F1 Driver for Ferrari.png",
        "team_color": "#FF2800"
    },
    "Lewis Hamilton": {
        "number": 44,
        "team": "Ferrari",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-47 Lewis Hamilton - F1 Driver for Ferrari.png",
        "team_color": "#FF2800"
    },
    "Oliver Bearman": {
        "number": 87,
        "team": "Haas",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-35 Oliver Bearman - F1 Driver for Haas.png",
        "team_color": "#FFFFFF"
    },
    "Esteban Ocon": {
        "number": 31,
        "team": "Haas",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-32 Esteban Ocon - F1 Driver for Haas.png",
        "team_color": "#FFFFFF"
    },
    "Lando Norris": {
        "number": 4,
        "team": "McLaren",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-09 Lando Norris - F1 Driver for McLaren.png",
        "team_color": "#FF8700"
    },
    "Oscar Piastri": {
        "number": 81,
        "team": "McLaren",
        "photo": "static/images/Screenshot 2025-11-30 at 19-22-54 Oscar Piastri - F1 Driver for McLaren.png",
        "team_color": "#FF8700"
    },
    "George Russell": {
        "number": 63,
        "team": "Mercedes",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-16 George Russell - F1 Driver for Mercedes.png",
        "team_color": "#00D2BE"
    },
    "Andrea Kimi Antonelli": {
        "number": 12,
        "team": "Mercedes",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-21 Kimi Antonelli - F1 Driver for Mercedes.png",
        "team_color": "#00D2BE"
    },
    "Max Verstappen": {
        "number": 1,
        "team": "Red Bull",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-35 Max Verstappen - F1 Driver for Red Bull Racing.png",
        "team_color": "#0600EF"
    },
    "Liam Lawson": {
        "number": 30,
        "team": "Racing Bulls",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-16 Liam Lawson - F1 Driver for Racing Bulls.png",
        "team_color": "#4B4B4B"
    },
    "Yuki Tsunoda": {
        "number": 22,
        "team": "Red Bull",
        "photo": "static/images/Screenshot 2025-11-30 at 19-23-39 Yuki Tsunoda - F1 Driver for Red Bull Racing.png",
        "team_color": "#0600EF"
    },
    "Isack Hadjar": {
        "number": 20,
        "team": "Racing Bulls",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-20 Isack Hadjar - F1 Driver for Racing Bulls.png",
        "team_color": "#4B4B4B"
    },
    "Nico Hulkenberg": {
        "number": 27,
        "team": "Haas",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-45 Nico Hulkenberg - F1 Driver for Stake F1 Team Kick Sauber.png",
        "team_color": "#FFFFFF"
    },
    "Gabriel Bortoleto": {
        "number": 5,
        "team": "Haas",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-49 Gabriel Bortoleto - F1 Driver for Stake F1 Team Kick Sauber.png",
        "team_color": "#FFFFFF"
    },
    "Alex Albon": {
        "number": 23,
        "team": "Williams",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-07 Alexander Albon - F1 Driver for Williams.png",
        "team_color": "#005AFF"
    },
    "Carlos Sainz": {
        "number": 55,
        "team": "Williams",
        "photo": "static/images/Screenshot 2025-11-30 at 19-24-12 Carlos Sainz - F1 Driver for Williams.png",
        "team_color": "#005AFF"
    }
    }

    driver_code_map = {
    "VER": "Max Verstappen",
    "HAM": "Lewis Hamilton",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "RUS": "George Russell",
    "ANT": "Andrea Kimi Antonelli",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "COL": "Franco Colapinto",
    "ALB": "Alex Albon",
    "BEA": "Oliver Bearman",
    "OCO": "Esteban Ocon",
    "LAW": "Liam Lawson",
    "TSU": "Yuki Tsunoda",
    "HAD": "Isack Hadjar",
    "HUL": "Nico Hülkenberg",
    "BOR": "Gabriel Bortoleto"
    }
    return drivers_info, driver_code_map

