# ParkIQ — AI-Driven Parking Congestion Intelligence

> Transforming raw parking violation data into real-time enforcement intelligence for Bengaluru traffic police.

---

## What is ParkIQ?

ParkIQ is an end-to-end AI system that ingests parking violation records and produces actionable intelligence — which junctions to patrol, when to send officers, which challans will stick, and what the ROI of each enforcement action is.

Built on **298,450 violation records** across **168 named junctions** and **55 police stations** in Bengaluru (Nov 2023 – Apr 2024).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (~5 minutes)
python run_pipeline.py

# 3. Launch the dashboard
streamlit run dashboard.py

# 4. Evaluate both ML models
python evaluate_models.py
```

---

## System Architecture

```
Raw CSV (298K records)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Module 0 — Ingest & CIS Scoring            │
│  CIS = proximity × vehicle_weight × time    │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────────┐
│  Module 1     │    │  Module 2          │
│  GRU Junction │    │  Spillover Chain   │
│  Stress       │    │  Graph (NetworkX)  │
└──────┬────────┘    └────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Module 3 — Enforcement Priority Engine     │
│  Multi-officer 2-opt TSP patrol routing     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Module 4 — Attribution Engine (Core ML)    │
│  Model A: LightGBM Poisson count forecaster │
│  Model B: LightGBM challan validator        │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
┌────────────┐ ┌────────┐ ┌──────────────────┐
│ Module 5   │ │Module 6│ │ Module 7         │
│ Policy Sim │ │Anomaly │ │ Revenue & ROI    │
│            │ │+Hotspot│ │ + Shift Schedule │
└─────┬──────┘ └───┬────┘ └────────┬─────────┘
      └────────────┴───────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Streamlit       │
        │  Dashboard       │
        │  (11 pages)      │
        └──────────────────┘
```

---

## Modules

### Module 0 — Data Ingest & CIS Scoring
Loads and cleans the raw CSV. Computes the **Causal Impact Score (CIS)** for every violation record:

```
CIS = proximity_score × vehicle_weight × time_multiplier
```

- `proximity_score`: 1.5 if near junction, 1.2 if near road crossing, 1.3 if on main road
- `vehicle_weight`: 0.3 (cycle) → 3.0 (tanker), based on congestion contribution
- `time_multiplier`: 1.5× during peak hours (8–10am, 5–8pm), 1.0× otherwise

**Output:** `outputs/scored.parquet` (298,450 records, CIS mean=1.39, max=10.0)

---

### Module 1 — Junction Stress Predictor (GRU)
Trains a **PyTorch GRU** per junction on hourly violation time-series. Predicts next-hour stress (0–1). Falls back to exponential smoothing for junctions with sparse data.

**Output:** `outputs/junction_stress.parquet` (168 junctions)

---

### Module 2 — Spillover Chain Graph
Builds a **directed NetworkX graph** where edges represent congestion propagation between junctions within 600m. Computes **PageRank centrality** to identify which junctions amplify downstream congestion the most.

**Output:** `outputs/spillover_graph.json` (169 nodes, 207 edges)

---

### Module 3 — Enforcement Priority Engine
Computes a composite priority score per junction:

```
Priority = 0.40 × CIS_mean
         + 0.28 × LSTM_stress
         + 0.17 × spillover_centrality
         + 0.10 × recency_score
         + 0.05 × validation_quality
```

Then runs **2-opt TSP optimisation** to generate patrol routes for 3 officers across the top 15 priority zones.

**Output:** `outputs/enforcement_priorities.csv`

---

### Module 4 — Attribution Engine (Core ML)

Two genuinely independent models, both leak-free and independently validated:

#### Model A — Hourly Violation Count Forecaster
- **Algorithm:** LightGBM with Poisson objective (correct for count data)
- **Task:** Predict how many violations will occur at junction J in the next hour
- **Features:** lag-1h, lag-2h, lag-24h, lag-48h, lag-168h, rolling means (3/6/24/48h), rolling std, EWM, junction×hour target encoding, cyclical hour encoding (sin/cos), day-of-month, trend
- **Split:** Strict time-based — train on Nov 2023–Mar 2024, test on Mar–Apr 2024
- **Leak check:** All lags shifted ≥1 step from target hour. Junction×hour TE computed from training data only
- **Tuning:** Optuna (10 trials, TPE sampler)
- **Results:**

  | Metric | Score | Baseline comparison |
  |--------|-------|-------------------|
  | MAE    | 0.166 | Naive-last: 2.93 → **84% better** |
  | RMSE   | 0.924 | — |
  | R²     | 0.920 | Naive-mean R² = 0.0 |
  | CV R² (4-fold) | 0.919 ± 0.011 | Stable, not overfitting |

#### Model B — Challan Validation Predictor
- **Algorithm:** LightGBM binary classifier + Isotonic calibration
- **Task:** Will this challan be confirmed/validated? (drives revenue forecasting)
- **Training:** November–March only — April excluded because `validated=False` in April is a **data pipeline artifact** (challans not yet reviewed at extraction time), not ground truth rejection
- **Inference:** Full dataset including April — every record gets a `validation_proba`
- **Features:** Police station, vehicle type, offence code, center code, junction, GPS coordinates, hour, weekday, weekend, station×hour target encoding (smoothed, train-only)
- **Explicitly excluded:** `prox_score`, `near_junction` (corr=0.98 with prox_score — same thing), `near_crossing`, `main_road`, `time_mult`, `cis`, `vehicle_weight`, `month` (data pipeline timing artifact)
- **Results:**

  | Metric | Score |
  |--------|-------|
  | ROC-AUC | 0.803 |
  | PR-AUC  | 0.956 |
  | F1 (validated) | 0.937 |
  | Brier score | 0.073 (well-calibrated) |
  | CV AUC std (5-fold) | 0.0009 (rock-stable) |

**Output:** `outputs/attribution.parquet` — every record gets `validation_proba`, `predicted_count`, `congestion_pct`, `top_shap_feature`

---

### Module 5 — Policy Simulation Lab
What-if engine: simulate restricting parking in zone X by Y% → compute downstream CIS reduction across the city. Pre-computes simulations for top 5 junctions × 4 restriction levels.

**Output:** `outputs/policy_report.json`

---

### Module 6 — Anomaly Detection + Repeat Hotspot Analysis

**Anomaly Detection:** Isolation Forest on CIS + spatial + temporal features. Flags top 5% most anomalous records — unusual vehicle-location-time combinations that don't fit normal patterns.

**Repeat Hotspot Analysis:** Grids the city into ~100m cells. Each cell is scored:
```
hotspot_score = 0.40 × norm(total_violations)
              + 0.30 × norm(avg_CIS)
              + 0.30 × persistence
```
where `persistence = weeks_active / total_weeks_in_dataset`.

Tiers use **percentile-based thresholds** (not fixed cutoffs) so classification is always meaningful regardless of dataset period:

| Tier | Percentile | Count |
|------|-----------|-------|
| Critical | Top 5% | 392 cells |
| High | 80th–95th | 1,172 cells |
| Medium | 50th–80th | 2,347 cells |
| Low | Bottom 50% | 3,903 cells |

**Output:** `outputs/anomaly_scores.parquet`, `outputs/repeat_locations.csv`

---

### Module 7 — Revenue & ROI Intelligence
Estimates per-junction enforcement economics:

```
Expected Revenue  = Σ fine_amount × collection_rate
Congestion Saving = violations × avg_vehicles_affected × delay_hours × ₹120/vh
Patrol Cost       = patrol_visits × 1.5hr × ₹250/hr
ROI               = (Revenue + Saving) / Patrol Cost
```

Also generates a **weekly officer shift schedule** from predicted stress windows.

**Dataset-level results:**
- Expected fine revenue: **₹6.59 Cr**
- Congestion savings: **₹2.17 Cr**
- Overall ROI: **75.4×**

**Output:** `outputs/roi_report.csv`, `outputs/shift_schedule.json`

---

## Dashboard Pages

| Page | What you see |
|------|-------------|
| 📊 Overview | KPIs, hourly/weekday violation patterns, CIS heatmap, validation probability distribution |
| 🗺️ Zone Heatmap | CIS density map / enforcement priority map / anomaly map (3 tabs) |
| 🔮 Junction Stress | GRU predicted stress vs historical mean, tier donut, bubble chart |
| 🚨 Anomaly Detection | Score distribution, anomalies by hour, top anomalous junctions, scatter |
| 📍 Repeat Hotspots | Tier map (Critical/High/Medium/Low), persistence vs score chart |
| 📈 Congestion Attribution | SHAP plots, per-record driver breakdown, predicted count by hour |
| 🚔 Enforcement Routing | Multi-officer patrol map, priority score component breakdown |
| 💰 Revenue & ROI | Fine revenue stack, ROI × Priority strategic quadrant |
| 🗓️ Shift Scheduler | Weekly heatmap, per-day shift detail |
| 🧪 Policy Simulator | Pre-computed what-if restriction simulations |
| 🔗 Spillover Graph | PageRank centrality map, top propagation nodes |
| 🤖 Model Performance | Full eval metrics, MAE-by-bucket, threshold sensitivity, CV stability |

---

## Model Evaluation

Run the standalone evaluation script to reproduce all metrics in the terminal:

```bash
python evaluate_models.py
```

This trains both models from scratch, runs Optuna tuning, evaluates on held-out test sets, and runs 5-fold cross-validation. Takes ~10 minutes.

**Key design decisions documented in the script:**
- Why April is excluded from Model B training (pipeline lag artifact)
- Why `near_junction` is excluded from Model B (corr=0.98 with `prox_score`)
- Why `month` is excluded from Model B (learns data pipeline timing, not real signal)
- Why Poisson objective is used for Model A (count data, non-negative)
- Why isotonic calibration is applied to Model B (raw probabilities were miscalibrated)

---

## Data Leakage — What Was Fixed

Previous versions had critical leakage issues. All fixed in v5:

| Version | Leakage | Fix |
|---------|---------|-----|
| v1 | Predicted CIS from its own formula components (R²=0.9999) | Replaced with genuine forecasting task |
| v2 | `near_junction` has corr=0.98 with `prox_score` | Excluded from all models |
| v3 | `junc_hour_hist_avg` computed over full dataset incl. test rows | Recomputed from training data only |
| v4 | `month` feature in Model B — April artifact inflated AUC | Excluded month; April removed from training |

---

## Tech Stack

**Backend / ML**
- Python 3.11
- LightGBM — gradient boosting (Poisson + binary)
- PyTorch — GRU stress predictor
- scikit-learn — Isolation Forest, isotonic calibration, cross-validation
- Optuna — hyperparameter search (TPE sampler)
- SHAP — feature attribution
- NetworkX — spillover graph + PageRank
- GeoPandas / Shapely — spatial operations

**Frontend**
- Streamlit — dashboard
- Plotly — all charts and maps (Mapbox dark theme)

**Data**
- Pandas + PyArrow (Parquet)
- 298,450 violation records, Nov 2023 – Apr 2024, Bengaluru

---

## File Structure

```
parkiq/
├── config.py                   # paths, constants, fine schedule
├── run_pipeline.py             # runs all 8 modules in sequence
├── evaluate_models.py          # standalone model evaluation script
├── dashboard.py                # Streamlit dashboard (11 pages)
│
├── module0_ingest.py           # data loading + CIS scoring
├── module1_junction_stress.py  # GRU stress predictor
├── module2_spillover.py        # spillover chain graph
├── module3_enforcement.py      # priority engine + TSP routing
├── module4_attribution.py      # Model A + Model B (core ML)
├── module5_policy_sim.py       # policy simulation
├── module6_anomaly_repeat.py   # anomaly detection + hotspots
├── module7_roi.py              # revenue, ROI, shift schedule
│
├── data/
│   └── cleaned_dataset.csv     # raw input (298K records)
│
├── outputs/                    # generated by pipeline
│   ├── scored.parquet
│   ├── junction_stress.parquet
│   ├── spillover_graph.json
│   ├── enforcement_priorities.csv
│   ├── attribution.parquet
│   ├── policy_report.json
│   ├── anomaly_scores.parquet
│   ├── repeat_locations.csv
│   ├── roi_report.csv
│   ├── shift_schedule.json
│   └── shap_summary.png
│
└── requirements.txt
```

---

## Requirements

```
streamlit>=1.35
plotly>=5.22
pandas>=2.0
numpy>=1.26
lightgbm>=4.0
optuna>=3.0
shap>=0.45
torch>=2.2
scikit-learn>=1.4
geopandas>=0.14
networkx>=3.2
pyarrow>=15.0
shapely>=2.0
```

Install with:
```bash
pip install -r requirements.txt
```