"""
PR 6 — Training data generation script.

Fetches 2 years of historical weather for 5 German cities from Open-Meteo,
computes PVLIB solar generation labels, and writes a parquet file.

Usage:
    python scripts/generate_training_data.py             # full run → data/berlin_solar_2yr.parquet
    python scripts/generate_training_data.py --dry-run   # 7 days only, validates schema, no write

Logs go to stdout and to logs/generate_training_data.log (created automatically).
If something fails, share the full log file with Claude for diagnosis.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pvlib
import requests

# ── Logging setup ──────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "generate_training_data.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CITIES = [
    {"name": "Berlin",    "lat": 52.52, "lon": 13.42, "altitude": 34},
    {"name": "Hamburg",   "lat": 53.55, "lon":  9.99, "altitude": 14},
    {"name": "Cologne",   "lat": 50.94, "lon":  6.96, "altitude": 37},
    {"name": "Munich",    "lat": 48.14, "lon": 11.58, "altitude": 519},
    {"name": "Frankfurt", "lat": 50.11, "lon":  8.68, "altitude": 112},
]

PANEL_TILT = 30.0       # degrees
PANEL_AZIMUTH = 180.0   # south-facing
PANEL_CAPACITY_KW = 5000.0  # 5 MW reference system

FULL_START = "2015-01-01"
# Open-Meteo archive has a ~5-day lag. Compute end date dynamically so
# re-running the script always fetches the most recent available data.
OPEN_METEO_ARCHIVE_LAG_DAYS = 5
FULL_END = (date.today() - timedelta(days=OPEN_METEO_ARCHIVE_LAG_DAYS)).isoformat()

DRY_START = "2024-01-01"
DRY_DAYS  = 7

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_VARIABLES = [
    "shortwave_radiation",
    "temperature_2m",
    "windspeed_10m",
    "cloudcover",
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "solar_training.parquet"

# ── Open-Meteo fetch ───────────────────────────────────────────────────────────

def fetch_weather(city: dict, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo archive API for one city."""
    log.info(
        "[%s] Fetching weather %s → %s (lat=%.2f, lon=%.2f)",
        city["name"], start, end, city["lat"], city["lon"],
    )
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(OPEN_METEO_VARIABLES),
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    log.info("[%s] HTTP %s — %s", city["name"], resp.status_code, resp.url)
    resp.raise_for_status()

    payload = resp.json()
    hourly = payload.get("hourly", {})

    times = hourly.get("time", [])
    if not times:
        raise ValueError(f"[{city['name']}] Empty 'hourly.time' in response. Full response: {payload}")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(times, utc=True),
        "ghi":         hourly["shortwave_radiation"],
        "temperature": hourly["temperature_2m"],
        "wind_speed":  hourly["windspeed_10m"],
        "cloud_cover": [v / 100.0 for v in hourly["cloudcover"]],  # % → 0–1
    })

    log.info(
        "[%s] Fetched %d rows | ghi range [%.1f, %.1f] W/m²",
        city["name"], len(df), df["ghi"].min(), df["ghi"].max(),
    )

    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning("[%s] Null values found: %s", city["name"], null_counts[null_counts > 0].to_dict())
    else:
        log.info("[%s] No null values.", city["name"])

    return df


# ── PVLIB solar generation labels ─────────────────────────────────────────────

def compute_generation(city: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add clear_sky_index and generation_kwh columns using PVLIB PVWatts model.

    Uses PVWatts (simple efficiency-based model) rather than the full ModelChain
    to avoid module/inverter database dependencies.
    """
    log.info("[%s] Running PVLIB PVWatts (tilt=%.0f°, azimuth=%.0f°, capacity=%.0f kW) ...",
             city["name"], PANEL_TILT, PANEL_AZIMUTH, PANEL_CAPACITY_KW)

    location = pvlib.location.Location(
        latitude=city["lat"],
        longitude=city["lon"],
        altitude=city["altitude"],
        tz="UTC",
    )

    times_index = pd.DatetimeIndex(df["timestamp"])

    # Clear-sky irradiance for clear_sky_index feature
    clear_sky = location.get_clearsky(times_index)
    df = df.copy()
    df["ghi_clearsky"] = clear_sky["ghi"].values
    df["clear_sky_index"] = df.apply(
        lambda r: r["ghi"] / r["ghi_clearsky"] if r["ghi_clearsky"] > 0 else 0.0,
        axis=1,
    )
    log.info("[%s] Clear-sky computed | clearsky GHI range [%.1f, %.1f]",
             city["name"], df["ghi_clearsky"].min(), df["ghi_clearsky"].max())

    # Solar position for plane-of-array irradiance
    solar_pos = pvlib.solarposition.get_solarposition(times_index, city["lat"], city["lon"])

    # DHI/DNI decomposition from GHI using Erbs model
    dni_dhi = pvlib.irradiance.erbs(df["ghi"].values, solar_pos["zenith"].values, times_index)
    dni = dni_dhi["dni"].fillna(0.0).clip(lower=0.0)
    dhi = dni_dhi["dhi"].fillna(0.0).clip(lower=0.0)

    # Plane-of-array (POA) irradiance on tilted surface
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_TILT,
        surface_azimuth=PANEL_AZIMUTH,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=dni,
        ghi=df["ghi"].values,
        dhi=dhi,
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    # Cell temperature (Sandia open-rack glass-glass coefficients)
    temp_cell = pvlib.temperature.sapm_cell(
        poa_global=poa_global,
        temp_air=df["temperature"].values,
        wind_speed=df["wind_speed"].values,
        a=-3.56, b=-0.075, deltaT=3,
    )

    # PVWatts DC model: P_dc = E_poa * capacity * (1 + gamma*(T_cell - 25))
    # gamma = -0.004 /°C (typical crystalline silicon temperature coefficient)
    pdc0_w = PANEL_CAPACITY_KW * 1000  # rated DC capacity in W
    dc_power_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global,
        temp_cell=temp_cell,
        pdc0=pdc0_w,
        gamma_pdc=-0.004,
    ).fillna(0.0).clip(lower=0.0)

    # PVWatts AC model: apply inverter efficiency (~0.96 typical)
    # pvlib ≥0.10: pvwatts_ac moved to pvlib.inverter.pvwatts
    ac_power_w = pvlib.inverter.pvwatts(
        pdc=dc_power_w,
        pdc0=pdc0_w,
        eta_inv_nom=0.96,
    ).fillna(0.0).clip(lower=0.0)

    df["generation_kwh"] = (ac_power_w / 1000.0).values  # W → kWh per hour

    non_zero = (df["generation_kwh"] > 0).sum()
    log.info(
        "[%s] PVWatts done | generation range [%.2f, %.2f] kWh | non-zero hours: %d / %d",
        city["name"],
        df["generation_kwh"].min(),
        df["generation_kwh"].max(),
        non_zero,
        len(df),
    )

    if df["generation_kwh"].isna().any():
        n_nan = df["generation_kwh"].isna().sum()
        log.warning("[%s] %d NaN values in generation_kwh — filling with 0.0", city["name"], n_nan)
        df["generation_kwh"] = df["generation_kwh"].fillna(0.0)

    return df


# ── Feature columns ───────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ghi_rolling_3h and ghi_delta_1h (match FEATURE_COLUMNS in app/features.py)."""
    df = df.copy()
    df["ghi_rolling_3h"] = df["ghi"].rolling(window=3, min_periods=1).mean()
    df["ghi_delta_1h"] = df["ghi"].diff().fillna(0.0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    return df


# ── Schema validation ──────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "timestamp", "ghi", "temperature", "wind_speed", "cloud_cover",
    "panel_tilt", "panel_azimuth", "hour_of_day", "day_of_year",
    "clear_sky_index", "ghi_rolling_3h", "ghi_delta_1h",
    "latitude", "longitude",
    "generation_kwh",  # label
]


def validate_schema(df: pd.DataFrame, city_name: str) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{city_name}] Missing columns: {missing}")
    log.info("[%s] Schema OK — all %d expected columns present.", city_name, len(EXPECTED_COLUMNS))

    # Sanity: no negative generation
    neg = (df["generation_kwh"] < 0).sum()
    if neg > 0:
        raise ValueError(f"[{city_name}] {neg} rows have negative generation_kwh — check PVLIB model.")

    # Sanity: GHI should be non-negative
    neg_ghi = (df["ghi"] < 0).sum()
    if neg_ghi > 0:
        log.warning("[%s] %d rows have negative GHI (Open-Meteo artefact) — clipping to 0.", city_name, neg_ghi)


# ── Main ───────────────────────────────────────────────────────────────────────

def process_city(city: dict, start: str, end: str) -> pd.DataFrame:
    df = fetch_weather(city, start, end)
    df = compute_generation(city, df)
    df = add_rolling_features(df)
    df = add_time_features(df)

    # Clip negative GHI artefacts
    df["ghi"] = df["ghi"].clip(lower=0.0)

    # Add static columns
    df["panel_tilt"] = PANEL_TILT
    df["panel_azimuth"] = PANEL_AZIMUTH
    df["latitude"] = city["lat"]
    df["longitude"] = city["lon"]

    validate_schema(df, city["name"])
    return df[EXPECTED_COLUMNS]


def main(dry_run: bool, force: bool) -> None:
    log.info("=" * 70)
    log.info("generate_training_data.py  |  dry_run=%s  force=%s", dry_run, force)
    log.info("=" * 70)

    # Skip if parquet already exists and --force not passed
    if not dry_run and not force and OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        log.info(
            "Parquet already exists at %s (%.1f MB). Skipping fetch and processing.",
            OUTPUT_PATH, size_mb,
        )
        log.info("To regenerate from scratch, run with --force.")
        return

    if dry_run:
        year_ranges = [(DRY_START, (date.fromisoformat(DRY_START) + timedelta(days=DRY_DAYS - 1)).isoformat())]
        cities = CITIES[:1]  # Berlin only for smoke test
        log.info("DRY RUN: %s → %s, city: %s only", year_ranges[0][0], year_ranges[0][1], cities[0]["name"])
    else:
        # Split into per-year chunks: smaller requests = lower timeout risk,
        # cheaper to retry a single year if one fails.
        cities = CITIES
        start_year = date.fromisoformat(FULL_START).year
        end_date = date.fromisoformat(FULL_END)
        year_ranges = []
        for y in range(start_year, end_date.year + 1):
            y_start = f"{y}-01-01"
            y_end = min(date(y, 12, 31), end_date).isoformat()
            year_ranges.append((y_start, y_end))
        total_years = len(year_ranges)
        log.info(
            "FULL RUN: %s → %s | %d cities × %d annual chunks = %d fetches",
            FULL_START, FULL_END, len(cities), total_years, len(cities) * total_years,
        )

    all_dfs: list[pd.DataFrame] = []

    for i, city in enumerate(cities, 1):
        log.info("── City %d/%d: %s ────────────────────────────────────", i, len(cities), city["name"])
        city_dfs: list[pd.DataFrame] = []
        for y_start, y_end in year_ranges:
            try:
                df = process_city(city, y_start, y_end)
                city_dfs.append(df)
                log.info("[%s] ✓ %s → %s: %d rows", city["name"], y_start, y_end, len(df))
            except Exception:
                log.error("[%s] FAILED for %s → %s — traceback below:", city["name"], y_start, y_end)
                log.error(traceback.format_exc())
                log.error("Aborting. Fix the error above and re-run.")
                sys.exit(1)
        city_total = sum(len(d) for d in city_dfs)
        all_dfs.extend(city_dfs)
        log.info("[%s] All years done — %d rows total", city["name"], city_total)

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Combined dataset: %d rows × %d columns", *combined.shape)
    log.info("Cities in dataset: %s", combined.groupby("latitude")["latitude"].count().to_dict())
    log.info(
        "generation_kwh stats: min=%.2f  mean=%.2f  max=%.2f  zeros=%d",
        combined["generation_kwh"].min(),
        combined["generation_kwh"].mean(),
        combined["generation_kwh"].max(),
        (combined["generation_kwh"] == 0).sum(),
    )

    if dry_run:
        log.info("DRY RUN complete — schema valid, no file written.")
        log.info("Re-run without --dry-run to generate the full dataset.")
    else:
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        log.info("Written → %s  (%.1f MB)", OUTPUT_PATH, size_mb)
        log.info("DONE ✓")

    log.info("Log saved to: %s", LOG_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate REIP solar training data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch 7 days for Berlin only, validate schema, do not write parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite existing parquet even if it already exists.",
    )
    args = parser.parse_args()

    try:
        main(dry_run=args.dry_run, force=args.force)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        log.critical("Unhandled exception — traceback:")
        log.critical(traceback.format_exc())
        sys.exit(1)
