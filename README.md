# ⚡ Energy Trading DataOps / MLOps Pipeline

> **U-CS 101 Final Year Project**
> End-to-end automation of data pipelines from ingestion to a Deep Reinforcement Learning trading agent for energy markets.
>
> **Project dashboard:** https://glynncarson.github.io/energy-trading-mlops/ (test results, defects, trading results and pipeline status, rebuilt on every push)

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION (Airflow)                       │
│                                                                       │
│  [Ingest] → [Validate] → [Transform] → [Train DRL Agent]            │
│                                              ↓                        │
│                                        [Backtest]                     │
│                                              ↓                        │
│                                        [Notify]                       │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   EIA API    │   │   AWS S3     │   │  AWS Glue    │   │   MLflow     │
│  (raw data)  │──▶│  (data lake) │──▶│  (catalog)   │   │  (tracking)  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                  ▲
┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  Terraform   │   │  Docker /    │   │  GitHub      │───────────┘
│ (infra IaC)  │   │  ECR         │   │  Actions CI  │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## 🗂 Project Structure

```
energy-trading-mlops/
├── src/
│   ├── ingestion/
│   │   └── ingest.py          # EIA API + synthetic data ingestion → S3
│   ├── transformation/
│   │   └── transform.py       # Cleaning, feature engineering, validation, splits
│   ├── ml/
│   │   └── drl/
│   │       ├── env.py         # Gymnasium trading environment
│   │       ├── train_drl.py   # PPO / SAC / TD3 / A2C training with MLflow tracking
│   │       ├── backtest.py    # Backtest + benchmark comparison + report dashboard
│   │       └── drl_tasks.py   # Airflow task wrappers for the DRL pipeline
│   ├── pipeline/
│   │   └── dag.py             # Apache Airflow DAG (full pipeline)
│   └── utils/
│       └── helpers.py         # Config loading, S3 I/O, logging
├── config/
│   ├── pipeline.yaml          # Pipeline config (schedule, thresholds, S3 paths)
│   └── drl.yaml               # DRL environment, action space, and algorithm config
├── terraform/
│   ├── main.tf                # S3, Glue, IAM, ECR, CloudWatch, SNS
│   ├── variables.tf
│   └── outputs.tf
├── tests/
│   ├── test_ingestion.py
│   ├── test_transformation.py
│   └── test_drl.py
├── .github/
│   └── workflows/
│       ├── ci.yml             # Lint → Test → Smoke → Docker build → TF validate
│       └── deploy.yml         # Push to ECR → Terraform apply → Deploy DAG
├── Dockerfile                 # Multi-stage production image
├── docker-compose.yml         # Local dev stack (Airflow + MLflow + LocalStack)
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Terraform >= 1.6
- AWS CLI (for cloud deployment)

### 1. Clone and configure
```bash
git clone https://github.com/your-org/energy-trading-mlops.git
cd energy-trading-mlops

cp .env.example .env
# Edit .env — at minimum, set EIA_API_KEY or leave source=synthetic in pipeline.yaml
```

### 2. Start local stack
```bash
docker-compose up -d localstack mlflow-server airflow-webserver airflow-scheduler
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 (admin/admin) |
| MLflow UI | http://localhost:5000 |
| LocalStack | http://localhost:4566 |

### 3. Run the pipeline locally (without Airflow)
```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Ingest synthetic data
python -m src.ingestion.ingest

# Transform
python -m src.transformation.transform

# Train the DRL trading agent (requires MLflow server running)
python -m src.ml.drl.train_drl

# Backtest the trained agent against Buy & Hold / Random baselines
python -m src.ml.drl.backtest
```

### 4. Run tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## ☁️ Cloud Deployment (AWS)

### 1. Provision infrastructure
```bash
cd terraform
terraform init
terraform plan -var="alert_email=your@email.com"
terraform apply
```

### 2. Push to main branch
The GitHub Actions CI/CD pipeline will automatically:
1. Run all tests and linting
2. Build and push Docker image to ECR
3. Apply Terraform changes
4. Deploy the Airflow DAG to MWAA

---

## 🤖 DRL Trading Agent Details

| Component | Detail |
|---|---|
| Algorithm | PPO (configurable: SAC, TD3, A2C) via Stable-Baselines3 |
| Environment | Custom Gymnasium env simulating hourly energy trading |
| Observation | 48 engineered market features (lags, rolling stats, RSI, MACD, Bollinger Bands, cyclical time encodings) + portfolio state [cash ratio, position ratio, unrealised PnL] |
| Action | Continuous `Box([-1, 1])` target position, or `Discrete(3)` Hold/Buy/Sell |
| Reward | Portfolio PnL change minus transaction costs (optionally rolling Sharpe ratio) |
| Episode | 720-hour (30-day) rolling window, randomly sampled during training |
| Tracking | MLflow (hyperparameters, rollout metrics, portfolio metrics, model artifacts) |
| Evaluation | Backtest vs. Buy & Hold and Random Agent baselines — Sharpe, max drawdown, Calmar ratio, win rate, VaR, profit factor |
| Retraining | Scheduled daily via Airflow |

---

## 📊 Pipeline Stages

| Stage | Task | Output |
|---|---|---|
| **Ingest** | Pull hourly energy data from EIA API or generate synthetic | Raw Parquet on S3 |
| **Transform** | Clean, validate, engineer 50+ features | Train/Val/Test Parquet splits on S3 |
| **Train DRL Agent** | PPO/SAC/TD3/A2C training in a custom trading environment, tracked in MLflow | Trained agent checkpoint |
| **Backtest** | Walk the agent through held-out data, compare vs. Buy & Hold / Random baselines | Performance dashboard (PNG) on S3 |
| **Notify** | Email summary via SNS | Stakeholder notification |

---

## 🔧 Configuration

All pipeline behaviour is controlled via YAML — no code changes needed:

```yaml
# config/pipeline.yaml
ingestion:
  source: synthetic          # Change to eia_api for live data
  lookback_days: 365
```

```yaml
# config/drl.yaml
training:
  algorithm: PPO              # PPO | SAC | TD3 | A2C
  total_timesteps: 500_000

environment:
  initial_cash: 100_000.0
  max_position_mwh: 100.0
  stop_loss_pct: 0.30
```

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| DRL | Stable-Baselines3, Gymnasium, PyTorch |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow |
| Infrastructure | Terraform, AWS (S3, Glue, ECR, CloudWatch, SNS) |
| CI/CD | GitHub Actions |
| Containerization | Docker, Docker Compose |
| Data Validation | Custom pandas-based validation rules |
| Config | YAML |
| Testing | pytest, moto (mock AWS) |

---

## 📄 License

MIT © 2024 Energy Trading MLOps Project
