import os
import fastf1
import pandas as pd
from tqdm import tqdm

CACHE_DIR = "f1_cache"
YEAR = 2024

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)


def td_to_sec(td):
    return td.total_seconds() if isinstance(td, pd.Timedelta) else None


def process_session(event_name):
    try:
        session = fastf1.get_session(YEAR, event_name, "R")
        session.load()
    except Exception as e:
        print(f"FAILED SESSION {event_name}: {e}")
        return None

    laps = session.laps.copy()
    weather = session.weather_data.copy()
    status = session.track_status.copy()

    # --- timestamps ---
    laps = laps.rename(columns={"Time": "LapTS"})
    status = status.rename(columns={"Time": "StatusTS"})
    weather = weather.rename(columns={"Time": "WeatherTS"})

    laps = laps.sort_values("LapTS")
    status = status.sort_values("StatusTS")
    weather = weather.sort_values("WeatherTS")

    # --- Weather merge ---
    weather = weather[["WeatherTS", "AirTemp", "TrackTemp", "Rainfall"]].ffill()

    laps = pd.merge_asof(
        laps,
        weather,
        left_on="LapTS",
        right_on="WeatherTS",
        direction="backward",
        tolerance=pd.Timedelta("45s")
    )

    # --- Lap times ---
    laps["lap_time"] = laps["LapTime"].apply(td_to_sec)
    laps["s1"] = laps["Sector1Time"].apply(td_to_sec)
    laps["s2"] = laps["Sector2Time"].apply(td_to_sec)
    laps["s3"] = laps["Sector3Time"].apply(td_to_sec)

    # Fix missing S1
    laps["s1"] = laps.apply(
        lambda r: r["lap_time"] - r["s2"] - r["s3"]
        if pd.isna(r["s1"]) and r["lap_time"] and r["s2"] and r["s3"]
        else r["s1"],
        axis=1
    )

    # --- Pit flag ---
    laps["pit_flag"] = ((laps["PitInTime"].notna()) | (laps["PitOutTime"].notna())).astype(int)

    # --- Map status to laps ---
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
    laps = laps.rename(columns={"Status_y": "Status"})

    # --- Correct track flags ---
    laps["yellow_flag"] = (laps["Status"] == 2).astype(int)
    laps["sc_flag"]     = (laps["Status"] == 4).astype(int)
    laps["red_flag"]    = (laps["Status"] == 5).astype(int)
    laps["vsc_flag"]    = ((laps["Status"] == 6) | (laps["Status"] == 7)).astype(int)

    # --- Output ---
    out = laps[[
        "Driver", "Team", "LapNumber",
        "lap_time", "s1", "s2", "s3",
        "Compound", "TyreLife", "Stint",
        "pit_flag",
        "AirTemp", "TrackTemp", "Rainfall",
        "yellow_flag", "sc_flag", "vsc_flag", "red_flag"
    ]].copy()

    out.insert(0, "race", event_name)
    return out


# ===== RUN =====
all_data = []
for ev in tqdm(schedule["EventName"], desc="Processing 2024"):
    df = process_session(ev)
    if df is not None:
        all_data.append(df)

final = pd.concat(all_data, ignore_index=True)
final = final.dropna(subset=["lap_time"])
final.to_csv("f1_2024_laps_clean.csv", index=False)

print("Saved f1_2024_laps_clean.csv", len(final))