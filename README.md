# DriftWatch

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018-blue)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-lightgrey)](https://postgresql.org)
[![Tests](https://img.shields.io/badge/Tests-57%20passing-brightgreen)](https://github.com/smaharx/driftwatch/tree/main/tests)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

DriftWatch is an open-source ML data quality and drift monitoring platform that helps teams detect when production data distributions diverge from the data used to train their models.

The platform combines statistical drift detection, persistent baselines and run history, alerting, a React dashboard, a REST API, and a Python SDK so data and ML teams can identify data drift before it silently impacts model performance.

---

## Live Demo

| Service | URL |
|---|---|
| API | https://driftwatch-production-e733.up.railway.app |
| API Docs | https://driftwatch-production-e733.up.railway.app/docs |
| Dashboard | https://driftwatch-nine.vercel.app |

---

## Overview

Machine learning models can degrade without an obvious failure in the application itself. A model may be trained on one population or feature distribution and later receive production data that looks significantly different. Without continuous monitoring, that change can remain invisible until model quality or business outcomes decline.

DriftWatch addresses this problem by storing training baselines, evaluating production batches against those baselines, recording drift reports, and surfacing actionable alerts through a web dashboard and Slack notifications.

---

## Key Features

- Statistical drift detection using PSI, Kolmogorov-Smirnov, Chi-Squared, and Jensen-Shannon divergence — all implemented from scratch
- Baseline management for training and reference datasets
- Production run logging with historical drift reports
- Configurable feature-level drift alerts with severity scoring (low, medium, high)
- Slack notifications when configured drift thresholds are exceeded
- React dashboard for monitoring drift trends and feature timelines
- REST API built with FastAPI and Pydantic v2
- Python SDK for simple model registration, baseline logging, and production monitoring
- PostgreSQL persistence with SQLAlchemy and Alembic migrations
- Automated test coverage across the drift detection algorithms and platform behavior

---

## System Architecture

The application is organized into separate components so the statistical engine, API, persistence layer, alerting, and dashboard can evolve independently:

1. **Data Ingestion Layer** — receives model metadata, baseline datasets, and production batches
2. **Drift Detection Layer** — evaluates feature distributions using statistical tests implemented in the project
3. **Data Persistence Layer** — stores models, baselines, monitoring runs, drift reports, and alerts in PostgreSQL
4. **Alerting Layer** — creates alerts and supports Slack notifications when drift thresholds are crossed
5. **API Layer** — exposes monitoring functionality through a FastAPI REST API
6. **Presentation Layer** — React dashboard for inspecting models, drift history, feature-level trends, and alerts
7. **Client Layer** — Python SDK for integrating DriftWatch into existing ML pipelines

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11+ |
| Backend API | FastAPI + Pydantic v2 |
| Drift Detection | NumPy + SciPy (built from scratch) |
| Database | PostgreSQL 16 |
| ORM & Migrations | SQLAlchemy + Alembic |
| Task Processing | Synchronous (Celery + Redis architecture ready) |
| Frontend | React 18 + TypeScript |
| Data Visualization | Recharts |
| Styling | Tailwind CSS |
| SDK | Python package (`driftwatch`) |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Deployment | Railway (API) + Vercel (Dashboard) |

---

## Statistical Tests

DriftWatch implements its core statistical detectors directly rather than wrapping an external monitoring framework.

| Test | Purpose | Default Trigger |
|---|---|---|
| Population Stability Index (PSI) | Measures the magnitude of distribution shift | PSI > 0.20 |
| Kolmogorov-Smirnov (KS) | Tests statistical differences between numerical distributions | p < 0.05 |
| Chi-Squared | Detects drift in categorical feature distributions | p < 0.05 |
| Jensen-Shannon Divergence | Measures symmetric, bounded divergence (0–1) between distributions | JSD > 0.15 |

---

## Project Structure

```text
driftwatch/
│
├── api/                    # FastAPI application, routes, and schemas
├── core/                   # Core drift detection and domain logic
│   └── detectors/          # PSI, KS, Chi-Squared, Jensen-Shannon
├── dashboard/              # React + TypeScript frontend
├── db/                     # Database models and Alembic migrations
├── docs/                   # Project documentation
├── experiments/            # Drift experiments and evaluation scripts
├── sdk/                    # Python SDK (pip install driftwatch)
├── services/               # Drift analyzer and notification services
├── tests/                  # Automated test suite (57 tests)
├── workers/                # Celery async task workers
├── .env.example            # Environment template
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Local service orchestration
├── pyproject.toml          # Python project and test configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/smaharx/driftwatch.git
cd driftwatch
```

Set up a Python virtual environment and install the dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

1. Copy `.env.example` to `.env`
2. Configure `DATABASE_URL`, `REDIS_URL`, and `SLACK_WEBHOOK_URL`
3. Ensure PostgreSQL is running before applying migrations

Run migrations:

```bash
python -m alembic upgrade head
```

---

## Running the Application

### Backend

```bash
uvicorn api.main:app --reload
```

API at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard at `http://localhost:5173`

### Docker

```bash
docker compose up --build
```

---

## Python SDK

```bash
pip install driftwatch
```

```python
from driftwatch import DriftClient

client = DriftClient(api_url="https://driftwatch-production-e733.up.railway.app")

model = client.register_model(
    name="fraud-detector-v1",
    feature_names=["age", "income", "region"],
)

client.log_baseline(model_id=model.id, dataframe=training_df)

report = client.log(model_id=model.id, dataframe=production_df)
print(report["drifted_features"])
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/models` | Register a model |
| GET | `/api/v1/models` | List registered models |
| POST | `/api/v1/models/{id}/baseline` | Upload a training baseline |
| POST | `/api/v1/models/{id}/runs` | Submit production batch → drift report |
| GET | `/api/v1/models/{id}/runs` | List run history |
| GET | `/api/v1/runs/{id}` | Retrieve a drift report |
| GET | `/api/v1/alerts` | List alerts |
| PATCH | `/api/v1/alerts/{id}/acknowledge` | Acknowledge an alert |
| GET | `/health` | Health check |

Full interactive documentation at `/docs`.

---

## Running Tests

```bash
pytest tests/ -v
```

57 tests covering PSI, KS, Chi-Squared, and Jensen-Shannon detectors — edge cases, error handling, symmetry properties, and threshold behavior.

---

## Comparable Tools

| Tool | Type | Gap |
|---|---|---|
| Evidently AI | Python library | No UI, no real-time API |
| WhyLabs | Closed SaaS | Not self-hosted, free tier limited |
| Arize AI | Enterprise SaaS | Expensive, closed source |
| NannyML | Python library | No dashboard, complex setup |
| **DriftWatch** | Open-source platform | Self-hosted, full UI, free, pip SDK |

---

## Future Improvements

- Full async processing via Celery + Redis (architecture written, workers in workers/ directory)
- Expanded alerting integrations beyond Slack
- Additional drift metrics and data quality checks
- Improved SDK documentation and integration examples
- Stronger deployment tooling for self-hosted installations

---

## Authors

Shahzaib Mahar ([smaharx](https://github.com/smaharx))

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.