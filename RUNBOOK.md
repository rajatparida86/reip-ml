# reip-ml Offline Runbook

Steps to run outside a Claude session. Do these in order — each step depends on the previous one succeeding.

---

## Prerequisites

You need to be in the `reip-ml` directory for all commands:

```bash
cd ~/github.com/rajatparida86/claude-workspace/projects/renewable-energy/code/reip-ml
```

**macOS only — install OpenMP before training:**
```bash
brew install libomp
```
XGBoost on macOS requires this. Without it, the training script fails with `libxgboost.dylib could not be loaded`.

---

## Step 1 — Set up the Python environment (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

**What this does:** Creates an isolated Python environment and installs all dependencies (FastAPI, XGBoost, PVLIB, pandas, pytest, etc.).

**Expected output:** Lots of pip install lines ending with `Successfully installed ...`. No errors.

**If it fails:**
- `python3: command not found` → install Python 3.11+: `brew install python@3.11`
- `pip: command not found` → try `python3 -m pip install -r requirements-dev.txt`
- Any other error → paste the terminal output to Claude

**Next time** (environment already created): just run `source .venv/bin/activate` to activate it.

---

## Step 2 — Verify tests pass (sanity check before training)

```bash
source .venv/bin/activate   # skip if already active
pytest tests/ -v
```

**Expected output:**
```
28 passed in ~1s
```

**If any tests fail:** Paste the output to Claude before proceeding. Do not run the training scripts if tests are failing.

---

## Step 3 — Generate training data

```bash
source .venv/bin/activate
python scripts/generate_training_data.py
```

**What this does:**
- Calls the Open-Meteo free weather API to fetch 11+ years of hourly weather (2015→today) for 5 German cities: Berlin, Hamburg, Cologne, Munich, Frankfurt
- Runs PVLIB physics model to compute how much solar energy a 5 MW panel would have generated each hour
- Saves ~492,600 rows to `data/solar_training.parquet`

**How long:** ~20–40 minutes (~55 API calls, one per city per year, with retry/backoff). You need internet access.

**Expected output (last few lines):**
```
2026-xx-xx xx:xx:xx  INFO      Combined dataset: 492600 rows × 15 columns
2026-xx-xx xx:xx:xx  INFO      generation_kwh stats: min=0.00  mean=...  max=...
2026-xx-xx xx:xx:xx  INFO      Written → .../data/solar_training.parquet  (xx.x MB)
2026-xx-xx xx:xx:xx  INFO      DONE ✓
```

**Log file:** `logs/generate_training_data.log` — this is the file to share with Claude if anything goes wrong.

**If it fails:**
- The script will print `FAILED — traceback below:` with the exact error before exiting
- Common issues:
  - **No internet / Open-Meteo timeout:** The log will show `requests.exceptions.ConnectionError` or HTTP non-200. Check your internet connection and re-run.
  - **Missing column in parquet:** The log will show `Missing columns: [...]`. This means the script changed but the parquet is stale — delete `data/` and re-run.
  - **Any other error:** Copy the full `logs/generate_training_data.log` file content and paste to Claude.

**Smoke test first (optional):** Before the full 2-year run, you can test just 7 days for Berlin with no file written:
```bash
python scripts/generate_training_data.py --dry-run
```
This completes in ~5 seconds. If `--dry-run` fails, fix it before running the full script.

---

## Step 4 — Train the models

```bash
source .venv/bin/activate
python scripts/train_solar.py
```

**What this does:**
- Loads `data/solar_training.parquet`
- Splits data chronologically: first 80% for training, last 20% for validation (never shuffled — this prevents data leakage)
- Trains 3 XGBoost models, one each for P10 / P50 / P90 quantiles
- Logs accuracy metrics (MAE, RMSE, skill score vs. baseline)
- Saves 3 model files: `models/solar/generic_v1_p10.joblib`, `generic_v1_p50.joblib`, `generic_v1_p90.joblib`
- Automatically runs a smoke test after saving

**How long:** ~5–10 minutes on CPU (394,080 training rows). No GPU needed.

**Expected output (key lines to look for):**
```
INFO  Loaded 492600 rows × 15 columns
INFO  Chronological split: train=394080 rows (80%), val=98520 rows (20%)
INFO  [P10] Val MAE=...  RMSE=...  Coverage=0.xxx (target=0.1)
INFO  [P50] Val MAE=...  RMSE=...  Coverage=0.xxx (target=0.5)
INFO  [P90] Val MAE=...  RMSE=...  Coverage=0.9xx (target=0.9)
INFO  Skill score (P50): 1.000  (target ≥ 0.20 over baseline)
INFO  Skill score target MET. ✓
INFO  [P10] Saved → models/solar/generic_v1_p10.joblib (~510 KB)
INFO  [P50] Saved → models/solar/generic_v1_p50.joblib (~505 KB)
INFO  [P90] Saved → models/solar/generic_v1_p90.joblib (~513 KB)
INFO  CHECK-ONLY passed. ✓
INFO  TRAINING COMPLETE ✓
```

**Log file:** `logs/train_solar.log` — share with Claude if anything goes wrong. Log appends across runs so you won't lose training metrics by running `--check-only` afterwards.

**Warnings to ignore:**
- `Coverage X.XXX is more than 10pp away from target` on P10/P50 — this is an artifact of 50% nighttime zeros in the data (the models are correctly predicting ~0 at night). P90 coverage at ~0.95 is the reliable signal.
- `Quantile ordering violated on some rows` in the smoke test — the smoke test uses random garbage inputs, not real solar data. Normal for XGBoost quantile regression; clipped at inference time.

**Warnings to pay attention to:**
- `Skill score X.XXX is below the 0.20 target` — the model is barely better than a naive baseline. Paste the full log to Claude for diagnosis.

**If it fails:**
- `FileNotFoundError: Training data not found` → Step 3 didn't complete. Re-run Step 3 first.
- `ModuleNotFoundError: No module named 'app.features'` → You're not in the right directory. `cd` to `reip-ml/` and try again.
- Any other error → paste `logs/train_solar.log` to Claude.

---

## Step 5 — Verify models work

```bash
source .venv/bin/activate
python scripts/train_solar.py --check-only
```

**What this does:** Loads the 3 saved `.joblib` files, runs a 5-row synthetic predict on each, checks output shapes, checks P10 ≤ P50 ≤ P90 ordering.

**Expected output:**
```
INFO  All 3 models loaded successfully.
INFO  [P10] Output shape: (5,)  values: [...]
INFO  [P50] Output shape: (5,)  values: [...]
INFO  [P90] Output shape: (5,)  values: [...]
INFO  Quantile ordering P10 ≤ P50 ≤ P90 holds on smoke data. ✓
INFO  CHECK-ONLY passed. ✓
```

**If it fails:** The `.joblib` files may be corrupt or the wrong version. Re-run Step 4.

---

## Step 6 — Run the service locally (optional — to manually test the API)

```bash
source .venv/bin/activate
MODEL_DIR=models uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

In a second terminal, test the endpoints:

```bash
# Health check
curl http://localhost:8001/health

# Expected:
# {"status":"ok","model_version":"solar_generic_v1"}

# Predict (single hour, Berlin)
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "test-site-001",
    "latitude": 52.52,
    "longitude": 13.42,
    "features": [{
      "timestamp": "2024-06-21T12:00:00Z",
      "ghi": 750.0,
      "temperature": 22.0,
      "wind_speed": 3.5,
      "cloud_cover": 0.1,
      "hour_of_day": 12,
      "day_of_year": 173,
      "panel_tilt": 30.0,
      "panel_azimuth": 180.0
    }]
  }' | python3 -m json.tool

# Expected: JSON with p10_kwh, p50_kwh, p90_kwh, confidence_score
```

**Note:** If the models are not loaded (Step 3–5 not done), `/predict` returns `{"detail":"Model registry not loaded"}` with HTTP 503. `/health` always works regardless.

---

## Step 7 — Tell Claude to continue (next session)

Once Steps 3–5 complete successfully, start a new Claude session and say:

> "Training data generation and model training completed successfully. Ready to do PR 8 (docker-compose integration) and PR 9 (GitHub Actions CI)."

If anything failed in Steps 3–5, say:

> "Step X failed. Here is the log:" and paste the relevant log file (`logs/generate_training_data.log` or `logs/train_solar.log`).

---

## Quick reference — all commands in order

```bash
cd ~/github.com/rajatparida86/claude-workspace/projects/renewable-energy/code/reip-ml

# One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Verify
pytest tests/ -v

# Generate data (needs internet, ~20-40 min, 2015→today, 5 cities)
python scripts/generate_training_data.py

# Train models (~5-10 min)
python scripts/train_solar.py

# Verify models
python scripts/train_solar.py --check-only

# Optional: run the service
MODEL_DIR=models uvicorn app.main:app --port 8001
```

---

## What the 3 model files mean

After training, you'll have:

| File | What it predicts |
|---|---|
| `models/solar/generic_v1_p10.joblib` | Pessimistic floor — 90% of actual outcomes will be at or above this |
| `models/solar/generic_v1_p50.joblib` | Best single-point estimate (median) |
| `models/solar/generic_v1_p90.joblib` | Optimistic ceiling — only 10% of actual outcomes will exceed this |

These files are gitignored (too large for git, and they're reproducible by re-running the training script). Keep them in `models/solar/` — the service loads them at startup.
