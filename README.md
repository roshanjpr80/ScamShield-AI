# 🛡️ ScamShield AI - Advanced Fraud Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-blue.svg)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-roshanjpr80%2FScamShield--AI-black.svg)](https://github.com/roshanjpr80/ScamShield-AI)

**Enterprise-grade multi-modal fraud detection platform** showcasing advanced ML/DL, NLP, CV, GenAI, and Agentic AI capabilities for fraud prevention and scam detection.

**🔗 GitHub Repository:** https://github.com/roshanjpr80/ScamShield-AI

## 🎯 Project Overview

ScamShield AI is a comprehensive fraud detection system that combines multiple AI techniques to detect various types of scams and fraudulent activities in real-time:

- **UPI Payment Fraud Detection** (Multi-agent LangGraph workflow)
- **Text-based Scam Detection** (DistilBERT + ensemble models)
- **Transaction Pattern Analysis** (XGBoost + sklearn ensembles)
- **Document Fraud Detection** (CV forensics + OCR)
- **Risk Scoring Engine** (Rule-based + ML hybrid)
- **MLOps Pipeline** (MLflow tracking + drift monitoring)
- **Interactive Dashboard** (Streamlit analytics)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (for deployment)
- PostgreSQL 14+
- Redis 7+

### Installation

```bash
# Clone repository
git clone https://github.com/roshanjpr80/ScamShield-AI.git
cd scamshield-ai

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Run API server
python -m uvicorn upishield_core.api.main:app --reload --port 8000

# Run Frontend dashboard
cd frontend && npm run dev
```

### Docker Deployment

```bash
docker-compose up -d
```

Access services:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend (Next.js): http://localhost:3000
- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9090

## 📁 Project Structure

```
scamshield-ai/
├── upishield_core/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py          # App entry point
│   │   └── routes/          # REST API endpoints
│   │       ├── health.py
│   │       ├── scoring.py
│   │       ├── detection.py
│   │       ├── models.py
│   │       ├── analytics.py
│   │       ├── alerts.py
│   │       ├── cases.py
│   │       └── ai_assistant.py
│   ├── cli/                  # Typer CLI commands
│   ├── models/               # ML model wrappers
│   ├── services/             # Business logic
│   ├── database/             # SQLAlchemy ORM
│   ├── auth/                 # JWT authentication
│   ├── schemas/              # Pydantic models
│   ├── core/                 # Configuration
│   └── tests/                # Pytest suite (51 tests)
├── frontend/                 # Next.js 14 Dashboard
│   ├── src/app/             # App Router pages
│   ├── src/components/      # React components
│   ├── src/lib/             # API client, utilities
│   └── package.json
├── models/                   # ONNX model files
├── data/                     # Generated datasets
├── prometheus/               # Prometheus config
├── grafana/                  # Grafana dashboards
├── alembic/                  # Database migrations
├── deploy/                   # Deployment configs
│   ├── docker-compose.yml
│   ├── render.yaml
│   └── kubernetes/
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── render.yaml
└── README.md
```

## 🧠 AI/ML Features

### Multi-Modal Detection
- **Text Analysis**: Detects phishing, smishing, vishing patterns using fine-tuned DistilBERT
- **Transaction Analysis**: XGBoost ensemble with 26 engineered features
- **Document Forensics**: ELA, noise analysis, copy-move detection using OpenCV
- **Fusion Model**: Combines all modalities for comprehensive risk assessment

### Agentic AI Workflow
- **FeatureExtraction Agent**: Processes raw input into structured features
- **PatternRecognition Agent**: Identifies known fraud patterns
- **ModelEnsemble Agent**: Runs predictions across multiple models
- **DecisionConsensus Agent**: Combines model predictions with voting
- **ReportGeneration Agent**: Creates human-readable reports

### MLOps Pipeline
- Automated training with MLflow experiment tracking
- Model versioning and A/B testing
- Real-time drift detection (KS test, JS divergence, Elliptic Envelope)
- Automatic retraining triggers
- Feature store management

### Observability
- Prometheus metrics collection
- OpenTelemetry distributed tracing
- Structured JSON logging
- Performance benchmarking

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /ready | Readiness probe |
| GET | /metrics | Prometheus metrics |
| POST | /api/v1/auth/login | User login |
| POST | /api/v1/auth/refresh | Refresh token |
| GET | /api/v1/auth/me | Current user info |
| POST | /api/v1/score | Score transaction |
| POST | /api/v1/score/batch | Batch score transactions |
| GET | /api/v1/transactions | List transactions |
| GET | /api/v1/transactions/{id} | Get transaction details |
| POST | /api/v1/detect/sms | SMS scam detection |
| POST | /api/v1/detect/job-scam | Job scam detection |
| GET | /api/v1/detections/sms | List SMS detections |
| GET | /api/v1/detections/job | List job detections |
| GET | /api/v1/models | List loaded models |
| GET | /api/v1/models/{name} | Model details |
| POST | /api/v1/models/reload | Reload models |
| GET | /api/v1/analytics/summary | Dashboard summary |
| GET | /api/v1/analytics/trends | Fraud trends |
| GET | /api/v1/analytics/risk-distribution | Risk score histogram |
| GET | /api/v1/alerts | List alerts |
| GET | /api/v1/alerts/{id} | Alert details |
| PATCH | /api/v1/alerts/{id} | Update alert status |
| GET | /api/v1/cases | List cases |
| POST | /api/v1/cases | Create case |
| GET | /api/v1/cases/{id} | Case details |
| POST | /api/v1/ai/query | AI assistant query |
| POST | /api/v1/ai/generate-report | Generate fraud report |

## 📊 Dashboard Pages (Next.js 14)

1. **Dashboard Overview** - Key metrics, fraud trends, recent alerts, risk distribution charts
2. **Transactions** - Searchable table, transaction details, score breakdown, export
3. **SMS Detection** - Paste SMS, instant verdict, factor analysis, explanation
4. **Job Scam Detection** - Paste job description, risk analysis, red flags
5. **Alerts Management** - Filterable feed, severity badges, acknowledge/assign actions
6. **Cases** - Investigation workflow, timeline, evidence, notes
7. **Models** - Model performance monitoring, version info, reload trigger
8. **Analytics** - Trends, geographic analysis, time series, model performance
9. **Settings** - User profile, API keys, thresholds, system config

## 🧪 Running Tests

```bash
# Run all tests (51 tests)
pytest upishield_core/tests/ -v

# Run specific test categories
pytest upishield_core/tests/test_features.py -v
pytest upishield_core/tests/test_ensemble.py -v
pytest upishield_core/tests/test_decision_engine.py -v

# With coverage
pytest upishield_core/tests/ --cov=upishield_core --cov-report=html
```

## 🚢 Deployment

### Local Development (Docker Compose)
```bash
docker-compose up -d
# API: http://localhost:8000
# Frontend: http://localhost:3000
# Grafana: http://localhost:3001
```

### Local Development (Manual)
```bash
# Terminal 1 - Backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn upishield_core.api.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm install && npm run dev
```

### Production (Render.com - One Click)
1. Push to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) → New + → Blueprint
3. Select repository, click Apply
4. Set `SECRET_KEY` in API service Environment
5. Done! Auto-deploys on every push

### Production (Docker)
```bash
docker build -t scamshield-api .
docker build -t scamshield-frontend ./frontend
```

### Cloud Deployment Options
- **Render.com** - Blueprint deploy (configured)
- **AWS ECS/Fargate** - Container deployment
- **Google Cloud Run** - Serverless containers
- **Azure Container Apps** - Managed containers

## 📈 Skills Demonstrated

This project showcases expertise in:

### Programming Languages
- ✅ Python (advanced)
- ✅ SQL (PostgreSQL, SQLAlchemy)
- ✅ Bash scripting
- ✅ YAML/Docker

### Machine Learning
- ✅ PyTorch neural networks
- ✅ TensorFlow/Keras
- ✅ XGBoost ensemble methods
- ✅ HuggingFace Transformers
- ✅ scikit-learn pipelines
- ✅ Feature engineering

### Deep Learning & NLP
- ✅ DistilBERT fine-tuning
- ✅ Transformer architectures
- ✅ Text classification
- ✅ Sequence modeling
- ✅ Tokenization strategies

### Computer Vision
- ✅ OpenCV image processing
- ✅ Document forensics
- ✅ Copy-move detection
- ✅ Error Level Analysis
- ✅ Noise pattern analysis
- ✅ OCR integration

### Generative AI & Agentic Systems
- ✅ LangChain LLM orchestration
- ✅ LangGraph agent workflows
- ✅ Prompt engineering
- ✅ Function calling
- ✅ Multi-agent systems
- ✅ RAG implementations

### Backend Development
- ✅ FastAPI async APIs
- ✅ RESTful design
- ✅ WebSocket support
- ✅ JWT authentication
- ✅ Rate limiting
- ✅ Input validation

### Database & ORMs
- ✅ SQLAlchemy async
- ✅ Alembic migrations
- ✅ PostgreSQL optimization
- ✅ Redis caching
- ✅ ETL pipelines

### MLOps & Observability
- ✅ MLflow experiment tracking
- ✅ Model versioning
- ✅ Drift detection
- ✅ Automated retraining
- ✅ Prometheus metrics
- ✅ OpenTelemetry tracing

### DevOps & Deployment
- ✅ Docker/containerization
- ✅ Docker Compose
- ✅ CI/CD pipelines
- ✅ Kubernetes basics
- ✅ Cloud deployment (AWS/GCP/Azure)
- ✅ Infrastructure as Code

### Frontend & Visualization
- ✅ Next.js 14 (App Router, React 18)
- ✅ TanStack Query + Table
- ✅ Recharts interactive charts
- ✅ Tailwind CSS responsive UI
- ✅ Real-time data visualization

### Security & Compliance
- ✅ OWASP security practices
- ✅ Data encryption
- ✅ Audit logging
- ✅ Privacy preservation

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [ONNX Runtime](https://onnxruntime.ai/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Render.com Blueprint](https://render.com/docs/blueprint-spec)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Author

Built with ❤️ for career showcase and learning purposes.

---

**Keywords**: Fraud Detection, ML/AI, Python, FastAPI, Next.js, React, ONNX Runtime, XGBoost, Transformers, MLOps, Prometheus, Grafana, Docker, Render.com, Portfolio Project
