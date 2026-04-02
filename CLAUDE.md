# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`reip-ml` is the Python ML inference service of the REIP (Renewable Energy Intelligence Platform) — a three-service system that helps solar/wind operators reduce imbalance penalties via 24h generation forecasts and market-calibrated risk scoring.

The other two sibling repos must be cloned **side by side** in the same parent directory (`projects/renewable-energy/code/`):
- `reip-core` — Go backend + REST API (port 8080); owns docker-compose and boots all services
- `reip-dashboard` — React + Vite dashboard (port 5173)

`reip-ml` runs on **port 8001** and is called by `reip-core` for ML inference.

## Commands

```bash
# Dev workflow (run from reip-core — it owns docker-compose)
make up              # Start full stack including reip-ml
make logs            # Tail all service logs
docker compose logs -f reip-ml  # Single service logs

# Tests (run inside reip-ml directory)
pytest tests/ -v                          # All tests
pytest tests/test_features.py -v         # Single file
pytest tests/ -v --cov=app --cov-report=term-missing  # With coverage

# Lint
ruff check app/ tests/
ruff format app/ tests/

# Training scripts (offline, not part of the server)
python scripts/generate_training_data.py --dry-run   # Smoke test (7 days, no write)
python scripts/generate_training_data.py             # Full 2-year fetch → parquet
python scripts/train_solar.py                        # Train XGBoost models → joblib
python scripts/train_solar.py --check-only           # Validate existing models
```

## Architecture

### Service contract with reip-core

`reip-core` calls `POST /predict` when new weather data arrives (every hour via ingestion scheduler). Results are stored in `reip-core`'s `forecast_cache` Postgres table.

**`POST /predict` request:**
```json
{
  "site_id": "uuid",
  "features": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "ghi": 650.2,
      "temperature": 18.4,
      "wind_speed": 3.1,
      "cloud_cover": 0.2,
      "hour_of_day": 10,
      "day_of_year": 81,
      "panel_tilt": 30.0,
      "panel_azimuth": 180.0
    }
  ]
}
```

**`POST /predict` response:**
```json
{
  "site_id": "uuid",
  "predictions": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "p10_kwh": 710.0,
      "p50_kwh": 842.5,
      "p90_kwh": 960.0,
      "confidence_score": 0.87,
      "model_version": "solar_generic_v1"
    }
  ]
}
```

### Key modules

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory; `lifespan` loads model registry at startup |
| `app/config.py` | `Settings` via pydantic-settings; reads `MODEL_DIR`, `LOG_LEVEL`, `ENV` |
| `app/schemas.py` | Pydantic v2 request/response models — the HTTP contract |
| `app/features.py` | `FEATURE_COLUMNS` constant + pure feature engineering functions |
| `app/model_registry.py` | Load-once `ModelRegistry`; `predict(X)` returns `(p10, p50, p90)` arrays |
| `app/routers/health.py` | `GET /health` |
| `app/routers/predict.py` | `POST /predict` — wires features → registry → response |
| `scripts/generate_training_data.py` | Offline: Open-Meteo fetch + PVLIB labels → parquet |
| `scripts/train_solar.py` | Offline: XGBoost quantile training → `.joblib` |

### Data flow

```
reip-core (hourly ingestion)
        │
        │  POST /predict
        ▼
  reip-ml (FastAPI)
        │
        ├── features.build_feature_matrix()  [pure Python]
        │        └── PVLIB clear-sky GHI
        │
        └── ModelRegistry.predict(X)          [XGBoost quantile]
                 ├── p10_kwh  (α=0.1)
                 ├── p50_kwh  (α=0.5)
                 └── p90_kwh  (α=0.9)
        │
        ▼
  PredictResponse  →  reip-core stores in forecast_cache
```

### Model design

- **Algorithm:** XGBoost `reg:quantileerror` — 3 separate models for P10/P50/P90
- **Training data:** 2 years of Open-Meteo historical weather for Berlin → fed through PVLIB `ModelChain` (fixed tilt=30, azimuth=180) to generate synthetic `generation_kwh` labels
- **Feature columns (11):** `ghi`, `temperature`, `cloud_cover`, `wind_speed`, `panel_tilt`, `panel_azimuth`, `hour_of_day`, `day_of_year`, `clear_sky_index`, `ghi_rolling_3h`, `ghi_delta_1h`
- **`FEATURE_COLUMNS`** is the single source of truth — defined in `app/features.py`, imported by both training script and inference handler. Column order must never drift.
- **Model files:** `models/solar/generic_v1_p{10,50,90}.joblib` (gitignored)
- **Model loading:** Once at startup via `lifespan`, stored on `app.state.registry`

### Testing without model files

CI never needs real `.joblib` files. `tests/conftest.py` provides a `stub_registry` fixture (`MagicMock(spec=ModelRegistry)`) returning deterministic `(710.0, 842.5, 960.0)`. All integration tests use this fixture — tests run in under a second.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `/app/models` | Directory containing `solar/generic_v1_p*.joblib` |
| `LOG_LEVEL` | `info` | Uvicorn log level |
| `ENV` | `development` | `development` or `production` |

## Engineering Principles

### TDD — red first, always

Write a failing test **before** any implementation code. No production code exists without a failing test that justified it. This applies at every layer: feature functions, registry loading, HTTP handlers.

```
Write failing test → Run (watch fail) → Implement minimum to pass → Run (green) → Refactor → Repeat
```

Skipping the failing-test step is not acceptable, even to sketch something out.

### Incremental PRs — never big-bang

Each PR must: do **one thing** (one module, one endpoint, one script), be reviewable in under 30 minutes, and leave CI green. Break any task that feels too large into smaller PRs before building.

### Plan before coding

For non-trivial features, a design note exists before the first line of code. If the design is unclear, ask — don't assume and implement.

### Pure functions first

Feature engineering logic lives in `app/features.py` as stateless, deterministic functions with no I/O. Test these first — they are the easiest to TDD and the most likely source of subtle bugs.

## CI (GitHub Actions)

On every push/PR to `main`:
1. `ruff check app/ tests/`
2. `pytest tests/ -v --cov=app`
3. No model files needed — stub registry handles all tests
