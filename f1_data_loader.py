import os
import fastf1
import pandas as pd
from fastf1.events import get_event_schedule

CACHE_DIR = "static/f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
CACHE_DIR_PROCESSED = "static/processed_races"
os.makedirs(CACHE_DIR_PROCESSED, exist_ok=True)


def td_to_sec(td):
    """Convert Timedelta to seconds."""
    return td.total_seconds() if isinstance(td, pd.Timedelta) else None


def process_session(year, event_name):
    try:
        session = fastf1.get_session(year, event_name, "R")
        session.load()
    except Exception as e:
        print(f"FAILED SESSION {event_name}: {e}")
        return None

    laps = session.laps.copy()
    weather = session.weather_data.copy()
    status = session.track_status.copy()

    # timestamps
    laps = laps.rename(columns={"Time": "LapTS"})
    status = status.rename(columns={"Time": "StatusTS"})
    weather = weather.rename(columns={"Time": "WeatherTS"})

    laps = laps.sort_values("LapTS")
    status = status.sort_values("StatusTS")
    weather = weather.sort_values("WeatherTS")

    # weather forward fill
    weather = weather[["WeatherTS", "AirTemp", "TrackTemp", "Rainfall"]].ffill()

    laps = pd.merge_asof(
        laps,
        weather,
        left_on="LapTS",
        right_on="WeatherTS",
        direction="backward",
        tolerance=pd.Timedelta("45s")
    )

    # lap times
    laps["lap_time"] = laps["LapTime"].apply(td_to_sec)
    laps["s1"] = laps["Sector1Time"].apply(td_to_sec)
    laps["s2"] = laps["Sector2Time"].apply(td_to_sec)
    laps["s3"] = laps["Sector3Time"].apply(td_to_sec)

    # fix missing s1
    laps["s1"] = laps.apply(
        lambda r: r["lap_time"] - r["s2"] - r["s3"]
        if pd.isna(r["s1"]) and r["lap_time"] and r["s2"] and r["s3"]
        else r["s1"],
        axis=1
    )

    # pit flag
    laps["pit_flag"] = ((laps["PitInTime"].notna()) | (laps["PitOutTime"].notna())).astype(int)

    status["Status"] = status["Status"].astype(int)
    status = pd.merge_asof(
        status.sort_values("StatusTS"),
        laps[["LapNumber", "LapTS"]].sort_values("LapTS"),
        left_on="StatusTS",
        right_on="LapTS",
        direction="backward"
    )

    lap_status_mode = (
        status.groupby("LapNumber")["Status"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else 1)
    )

    laps = laps.merge(lap_status_mode, on="LapNumber", how="left")

    laps["yellow_flag"] = (laps["Status"] == 2).astype(int)
    laps["sc_flag"]     = (laps["Status"] == 4).astype(int)
    laps["red_flag"]    = (laps["Status"] == 5).astype(int)
    laps["vsc_flag"]    = ((laps["Status"] == 6) | (laps["Status"] == 7)).astype(int)

    out = laps[[
        "Driver", "Team", "LapNumber",
        "lap_time", "s1", "s2", "s3",
        "Compound", "TyreLife", "Stint",
        "pit_flag",
        "AirTemp", "TrackTemp", "Rainfall",
        "yellow_flag", "sc_flag", "vsc_flag", "red_flag"
    ]].copy()

    out.insert(0, "race", event_name)

    # -----------------------------
    # 🔥 FINAL CLEANING STEP
    # -----------------------------

    # Remove rows with missing sector times or lap time
    out = out.dropna(subset=["lap_time", "s1", "s2", "s3"])

    # Weather ffill + bfill to clean missing values
    weather_cols = ["AirTemp", "TrackTemp", "Rainfall"]
    out[weather_cols] = out[weather_cols].fillna(method="ffill")
    out[weather_cols] = out[weather_cols].fillna(method="bfill")

    return out


def load_race_data(year, race_name):
    """
    Loads race laps from local CSV cache if available.
    Otherwise processes using FastF1, saves CSV, then returns dataframe.
    """
    safe_name = race_name.replace(" ", "_")
    cache_file = f"{CACHE_DIR_PROCESSED}/{year}_{safe_name}.csv"

    # 1. If cached file exists → load it
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached race: {race_name}")
            return pd.read_csv(cache_file)
        except Exception as e:
            print(f"Error loading cached CSV for {race_name}: {e}")
            # continue and regenerate
    
    # 2. Otherwise: load via FastF1
    print(f"Fetching from FastF1: {race_name}")
    df = process_session(year, race_name)

    if df is None:
        print(f"Failed to load race (FastF1): {race_name}")
        return None

    # 3. Save processed data
    try:
        df.to_csv(cache_file, index=False)
        print(f"Saved processed race to: {cache_file}")
    except Exception as e:
        print(f"Could not save cache for {race_name}: {e}")

    return df


def list_races(year=2025):
    """Return a list of official F1 race names for the season."""
    schedule = get_event_schedule(year)
    return schedule["EventName"].tolist()

def load_2025_dropdown():
    year = 2025
    all_races = list_races(year)
    good_races = []
    drivers = set()

    for race in all_races:
        if race == 'Qatar Grand Prix':
            break
        df = load_race_data(year, race)
        if df is None:
            print(f"Skipping race (failed): {race}")
            continue

        good_races.append(race)

        if "Driver" in df.columns:
            drivers.update(df["Driver"].unique())

    return sorted(good_races), sorted(list(drivers))