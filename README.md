# DriftWatch

Open-source data quality and drift monitoring for ML pipelines — catch silent model degradation before it hits production.

DriftWatch computes statistical drift between a reference dataset and live production data, flags degradation early, and surfaces it on a dashboard — without wrapping an existing drift library. Every detector is implemented from scratch.

## Why

Most drift-monitoring tools either lock you into a vendor platform or hide the statistics behind a black box. DriftWatch is built to be:
- **Transparent** — every test is implemented from first principles (NumPy/SciPy), not imported from a black-box package
- **Self-hostable** — Docker Compose up and you're running, no SaaS dependency
- **Lightweight to integrate** — a minimal pip-installable SDK, not a heavyweight agent

## Status

🚧 Early development. Not yet installable or deployable.

- [x] Repo scaffolded, MIT licensed, `develop` branch model with conventional commit prefixes
- [x] Statistical detectors implemented from scratch: PSI, KS test, Chi-Squared, Jensen-Shannon divergence
- [ ] FastAPI backend (in progress — `feat/week2-fastapi-backend`)
  - [x] SQLAlchemy models with cascade deletes, indexes, PostgreSQL JSON columns
  - [x] Alembic migrations configured
  - [ ] API routes
  - [ ] DB session dependency wiring
- [ ] Celery/Redis async worker pipeline
- [ ] React + Recharts dashboard
- [ ] CI/CD (GitHub Actions → Render + Vercel)
- [ ] Pip-installable SDK

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  React +    │◄────►│   FastAPI    │◄────►│   PostgreSQL     │
│  Recharts   │      │   Backend    │      │                  │
│  Dashboard  │      └──────┬───────┘      └─────────────────┘
└─────────────┘             │
                             ▼
                     ┌──────────────┐
                     │ Celery/Redis │
                     │ Async Workers│──── PSI / KS / Chi² / JS divergence
                     └──────────────┘
```

## Detectors

| Test | Use case |
|---|---|
| Population Stability Index (PSI) | Numerical feature distribution shift |
| Kolmogorov–Smirnov (KS) | Continuous distribution comparison |
| Chi-Squared | Categorical feature distribution shift |
| Jensen-Shannon Divergence | Symmetric distribution distance |

## Tech Stack

- **Backend:** FastAPI, SQLAlchemy, Alembic, Pydantic Settings
- **Workers:** Celery, Redis
- **Database:** PostgreSQL
- **Frontend:** React, Recharts
- **Infra:** Docker Compose, GitHub Actions, Render, Vercel

## Local Development

```bash
git clone https://github.com/<your-username>/driftwatch.git
cd driftwatch
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

_(Docker Compose setup and full run instructions coming once the backend routes and worker pipeline are wired up.)_

## Roadmap

See the current branch (`feat/week2-fastapi-backend`) for active work. Near-term:
1. Finish FastAPI routes + DB session dependency
2. Run first Alembic migration against local Postgres
3. Wire Celery worker to invoke detectors on scheduled/triggered jobs
4. Ship minimal dashboard reading from the API

## License

MIT