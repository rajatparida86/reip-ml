# reip-ml Build Plan

This file tracks the PR-by-PR implementation sequence for the reip-ml service.
Check off each item as its PR is merged to `main`.

---

## PR sequence

### PR 1 — Project scaffold `[x]`
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`
- `Dockerfile` (multi-stage production), `Dockerfile.dev` (hot-reload)
- `.gitignore` (`*.joblib`, `__pycache__`, `.venv`, `data/`)
- Empty `app/__init__.py`, `app/main.py` (`app = FastAPI()`), `tests/__init__.py`, `tests/conftest.py`
- `models/solar/.gitkeep`
- **Failing test first:** `tests/test_health.py::test_import` — `from app.main import app` fails until `app = FastAPI()` exists
- Acceptance: `pytest tests/` green, `docker build -f Dockerfile .` succeeds

### PR 2 — Schemas + `GET /health` `[x]`
- `app/schemas.py` — `FeatureRow`, `PredictRequest`, `Prediction`, `PredictResponse`, `HealthResponse`
- `app/config.py` — `Settings(BaseSettings)` reading `MODEL_DIR` (default `/app/models`), `LOG_LEVEL`, `ENV`
- `app/routers/health.py` — `GET /health` → `{"status": "ok", "model_version": "solar_generic_v1"}`
- **Failing tests first:** `test_health_returns_200`, `test_health_body` — fail until router wired
- Acceptance: `pytest tests/test_health.py` green

### PR 3 — Feature engineering (pure functions) `[x]`
- `app/features.py`:
  - `FEATURE_COLUMNS: list[str]` — 11 columns, single source of truth shared with training script
  - `compute_clear_sky_index(ghi, ghi_clearsky) -> float` — safe division (returns 0.0 when clearsky=0)
  - `compute_rolling_ghi_3h(ghi_series) -> list[float]` — 3-element rolling mean
  - `compute_ghi_delta_1h(ghi_series) -> list[float]` — diff from previous step, 0.0 at index 0
  - `build_feature_matrix(rows, clear_sky_series) -> np.ndarray` — returns shape `(n, 11)`, columns in `FEATURE_COLUMNS` order
- **Failing tests first:** all in `tests/test_features.py` — clear sky index (normal + zero denominator), rolling (single element, three elements), delta (first=0, subsequent diffs), feature matrix shape
- Acceptance: `pytest tests/test_features.py` green (zero ML imports needed)

### PR 4 — Model registry `[x]`
- `app/model_registry.py`:
  - `ModelRegistry` class — holds p10/p50/p90 XGBoost models + `version` string
  - `ModelRegistry.predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `load_registry(model_dir: str) -> ModelRegistry` — raises `FileNotFoundError` if any `.joblib` missing
- `app/main.py` — updated with `lifespan` context manager; registry stored on `app.state.registry`
- `tests/conftest.py` — `stub_registry` fixture: `MagicMock(spec=ModelRegistry)` returning `(710.0, 842.5, 960.0)` for any input
- **Failing tests first:** `test_load_registry_missing_dir`, `test_load_registry_missing_file`
- Acceptance: `pytest tests/test_model_registry.py` green (no real `.joblib` files needed)

### PR 5 — `POST /predict` endpoint `[x]`
- `app/routers/predict.py` — full handler flow:
  1. Validate `PredictRequest` via Pydantic
  2. Compute PVLIB clear-sky GHI (fixed Berlin location: 52.52°N, 13.42°E — v1 simplification)
  3. `features.build_feature_matrix(rows, clear_sky_series)` → `X`
  4. `request.app.state.registry.predict(X)` → `(p10, p50, p90)`
  5. `confidence_score = 1 - (p90 - p10) / (p90 + 1e-9)` per row
  6. Return `PredictResponse`
- **Failing tests first:** `test_predict_returns_200`, `test_predict_response_shape`, `test_predict_timestamp_preserved`, `test_predict_missing_features_returns_422`, `test_predict_empty_features_returns_422`
- All tests use `stub_registry` fixture — no real models needed
- Acceptance: `pytest tests/test_predict.py` green

### PR 6 — Training data generation script `[x]`
- `scripts/generate_training_data.py`:
  - Fetches 2 years Open-Meteo **historical** weather for Berlin (`/archive` endpoint, 2022-01-01 to 2023-12-31)
  - `pvlib.location.Location(52.52, 13.42, altitude=34)` + `get_clearsky()` for `clear_sky_index`
  - `pvlib.ModelChain` (fixed tilt=30, azimuth=180) → hourly `generation_kwh` labels
  - Writes `data/berlin_solar_2yr.parquet` (gitignored)
  - `--dry-run` flag: 7-day fetch, schema validation, no write to disk
- Acceptance: `python scripts/generate_training_data.py --dry-run` exits 0

### PR 7 — Training script + model serialisation `[x]`
- `scripts/train_solar.py`:
  - Loads `data/berlin_solar_2yr.parquet`
  - Imports `FEATURE_COLUMNS` from `app.features` (single source of truth — never redefine)
  - Chronological 80/20 train/val split (no random shuffle — prevents data leakage)
  - Trains 3 `XGBRegressor(objective="reg:quantileerror", quantile_alpha=α)` for α ∈ {0.1, 0.5, 0.9}
  - Saves to `models/solar/generic_v1_p10.joblib`, `generic_v1_p50.joblib`, `generic_v1_p90.joblib`
  - `--check-only` flag: loads existing models, runs 5-row predict, validates output shape
- Acceptance: `pytest tests/` still green; `python scripts/train_solar.py --check-only` exits 0 with real models present

### PR 8 — docker-compose integration (reip-core repo) `[ ]`
- Changes in **reip-core** (not this repo):
  - Uncomment `reip-ml` service block in `docker-compose.yml`
  - Uncomment `reip-models:` volume
  - Add `reip-ml` to `reip-core` `depends_on`
- Acceptance: `make up` (from reip-core) → `curl http://localhost:8001/health` returns 200

### PR 9 — CI pipeline `[ ]`
- `.github/workflows/ci.yml`:
  ```yaml
  - uses: actions/setup-python@v5
    with: { python-version: "3.11", cache: pip }
  - run: pip install -r requirements-dev.txt
  - run: ruff check app/ tests/
  - run: pytest tests/ -v --cov=app --cov-report=term-missing
  ```
- No `.joblib` files needed — stub registry handles all tests
- Acceptance: green badge on `main`

---

## Implementation notes (deviations from original plan)

- **13 features, not 11:** `latitude` and `longitude` added to `FEATURE_COLUMNS` in PR 3. `PredictRequest` also takes `latitude`/`longitude` so PVLIB clear-sky runs per site location, not hardcoded Berlin.
- **5 German cities in training data (PR 6):** Berlin, Hamburg, Cologne, Munich, Frankfurt — ~87,600 rows total. Improves generalization across German sites.
- **PVWatts model (PR 6):** Used `pvlib.pvsystem.pvwatts_dc` + `pvlib.inverter.pvwatts` rather than `ModelChain` to avoid module/inverter database dependencies and Python version compatibility issues.
- **`--dry-run` verified working:** Smoke test passes against live Open-Meteo API (Jan 2022, Berlin, 168 rows).

---

## Key architectural decisions (rationale)

| Decision | Why |
|---|---|
| `FEATURE_COLUMNS` in `app/features.py` | Single source of truth imported by both training and inference — prevents column order drift |
| `lifespan` not `@app.on_event` | FastAPI idiomatic since v0.93; deprecated pattern avoided |
| `stub_registry` in conftest | Real models (~5–20 MB each) must not be in the repo; CI runs in <1s with no disk I/O |
| Fixed Berlin location for v1 | Per-site clear-sky computation deferred to v2 when per-site models are introduced |
| Chronological train/val split | Random split leaks future data into training — always split time-series by time |
| `confidence_score = 1 - (p90-p10)/(p90+1e-9)` | v1 heuristic matching `forecast_cache.confidence_score` column in reip-core |
