# DriftWatch

Open-source ML data quality and drift monitoring platform.

Detects when production data distributions diverge from training data — before model performance degrades silently.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![React](https://img.shields.io/badge/React-18-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![Tests](https://img.shields.io/badge/Tests-57%20passing-brightgreen)

## The Problem

ML models degrade silently. Training data looked one way in January. By June, production data looks completely different. The model never gets told. Accuracy drops, revenue follows. Nobody notices until it's too late.

## What DriftWatch Does

- Runs statistical drift tests (PSI, KS, Chi-Squared, Jensen-Shannon) against stored baselines
- Creates alerts when features exceed drift thresholds
- Sends Slack notifications when drift is detected
- Visualizes drift timelines per feature in a React dashboard
- Exposes a pip-installable SDK for one-line integration

## Architecture

Data Ingestion → Drift Engine → PostgreSQL → Alert Engine → React Dashboard
↑ ↓
REST API Slack Notifications
Python SDK


## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Pydantic v2 |
| Drift Detection | NumPy + SciPy (built from scratch) |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Task Queue | Celery + Redis (architecture ready) |
| Frontend | React 18 + TypeScript + Recharts + Tailwind |
| SDK | pip install driftwatch |
| CI/CD | GitHub Actions |
| Deployment | Render + Vercel |

## Statistical Tests

All implemented from scratch — no Evidently wrapper.

- **PSI (Population Stability Index)** — measures magnitude of distribution shift. PSI > 0.2 triggers alert.
- **KS Test (Kolmogorov-Smirnov)** — statistical significance of numerical feature drift. p < 0.05 triggers alert.
- **Chi-Squared** — categorical feature drift detection.
- **Jensen-Shannon Divergence** — symmetric, bounded (0-1) divergence measure.

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16
- Node.js 20+

### Backend

```bash
git clone https://github.com/YOUR_USERNAME/driftwatch.git
cd driftwatch

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
cp .env.example .env  # configure DATABASE_URL

python -m alembic upgrade head
uvicorn api.main:app --reload
```

API docs at `http://localhost:8000/docs`

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard at `http://localhost:5173`

### SDK

```bash
pip install driftwatch
```

```python
from driftwatch import DriftClient
import pandas as pd

client = DriftClient(api_url="http://localhost:8000")

model = client.register_model(
    name="my-fraud-model",
    feature_names=["age", "income", "region"],
)

client.log_baseline(model_id=model.id, dataframe=training_df)

report = client.log(model_id=model.id, dataframe=production_df)
print(report["drifted_features"])
```

## API Endpoints

POST /api/v1/models Register a model
GET /api/v1/models List all models
POST /api/v1/models/{id}/baseline Upload training baseline
POST /api/v1/models/{id}/runs Submit production batch → drift report
GET /api/v1/models/{id}/runs List run history
GET /api/v1/runs/{id} Get drift report
GET /api/v1/alerts List alerts
PATCH /api/v1/alerts/{id}/acknowledge Acknowledge alert
GET /health Health check



## Running Tests

```bash
pytest tests/ -v
```

57 tests across PSI, KS, Chi-Squared, and Jensen-Shannon detectors.

## Comparable Tools

| Tool | Type | Gap |
|---|---|---|
| Evidently AI | Python library | No UI, no real-time API |
| WhyLabs | Closed SaaS | Not free, not self-hosted |
| Arize AI | Enterprise SaaS | Expensive, closed source |
| **DriftWatch** | Open-source platform | Self-hosted, full UI, free |

## License

MIT

