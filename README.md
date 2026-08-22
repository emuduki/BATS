# BATS — Binary Options AI Trading System

BATS is an automated binary options trading platform leveraging technical indicators, regime detection, machine learning models, multi-agent decision systems, and real-time execution interfaces.

---

## System Architecture Overview

```
BATS/
│
├── backend/            # FastAPI REST API, Websockets, Database & Redis handling
├── trading/            # Market data, strategies, indicators, risk management, execution, backtesting
├── ai/                 # Machine learning models (PyTorch/Scikit-Learn), feature engineering, training
├── agents/             # Multi-agent architecture (Technical, Regime, Prediction, Decision agents)
├── frontend/           # Next.js / React + TypeScript real-time trading dashboard
├── docker-compose.yml  # Docker multi-container deployment
└── README.md
```

---

## Quick Start (Docker Compose)

1. **Clone & Configure Environment**:
   ```bash
   cp .env.example .env
   ```

2. **Launch All Services**:
   ```bash
   docker-compose up --build
   ```

3. **Check API Health**:
   ```bash
   curl http://localhost:8000/health
   ```

   Response:
   ```json
   {
     "status": "running",
     "database": "connected",
     "redis": "connected"
   }
   ```

---

## API Endpoints (Phase 0)

- `GET /health`: Health status of API, PostgreSQL connection, and Redis cache.
- `GET /api/v1/signals`: Fetch recent trading signals.
- `POST /api/v1/signals`: Submit/Generate a standardized trading signal.

---

## Standardized Signal Format

Every agent and strategy module in BATS communicates using a standardized signal payload:

```json
{
  "symbol": "R_100",
  "direction": "UP",
  "confidence": 0.87,
  "duration": 60,
  "duration_unit": "seconds",
  "strategy": "ema_rsi",
  "timestamp": "2026-08-22T10:50:00Z"
}
```
