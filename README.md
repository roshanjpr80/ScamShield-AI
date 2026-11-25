# 🛡️ ScamShield AI - Advanced Fraud Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-blue.svg)](https://streamlit.io/)

**Enterprise-grade multi-modal fraud detection platform** showcasing advanced ML/DL, NLP, CV, GenAI, and Agentic AI capabilities for fraud prevention and scam detection.

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
git clone https://github.com/yourusername/scamshield-ai.git
cd scamshield-ai

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Run API server
python -m uvicorn scamshield.api.main:app --reload --port 8000

# Run Streamlit dashboard
streamlit run scamshield/frontend/streamlit_app.py
```

### Docker Deployment

```bash
docker-compose up -d
```

Access services:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

## 📁 Project Structure

```
scamshield-ai/
├── scamshield/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py          # App entry point
│   │   └── routes/          # REST API endpoints
│   │       ├── auth.py      # Authentication (JWT)
│   │       ├── transactions.py
│   │       ├── detections.py
│   │       ├── alerts.py
│   │       ├── models.py    # MLOps endpoints
│   │       ├── analytics.py
│   │       └── scam_library.py
│   ├── core/                 # Core ML/DL models
│   │   ├── text_detector.py        # DistilBERT + ensemble
│   │   ├── transaction_detector.py # XGBoost + sklearn
│   │   ├── document_detector.py    # CV forensics
│   │   └── inference_pipeline.py   # 5-agent workflow
│   ├── services/             # Business logic
│   │   ├── upi_parser.py         # SMS parser
│   │   └── risk_scorer.py        # Risk scoring engine
│   ├── mlops/                # ML operations
│   │   ├── training_pipeline.py  # Training automation
│   │   ├── mlflow_integration.py # Experiment tracking
│   │   └── drift_detection.py    # Drift monitoring
│   ├── frontend/             # Streamlit dashboards
│   │   └── streamlit_app.py     # Interactive UI
│   ├── database/             # SQLAlchemy ORM
│   ├── schemas/              # Pydantic models
│   ├── data/                 # Data generation
│   └── observability/        # Metrics & logging
├── tests/                    # Comprehensive test suite
├── deploy/                   # Deployment configs
│   ├── docker-compose.yml
│   ├── render.yaml
│   └── kubernetes/
├── alembic/                  # Database migrations
├── requirements.txt
├── pyproject.toml
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
| POST | /api/v1/auth/login | User login |
| POST | /api/v1/auth/register | User registration |
| GET | /api/v1/transactions | List transactions |
| POST | /api/v1/detections/text | Text scam detection |
| POST | /api/v1/detections/transaction | Transaction fraud detection |
| POST | /api/v1/detections/multimodal | Multi-modal detection |
| GET | /api/v1/alerts | Get fraud alerts |
| POST | /api/v1/models/train | Trigger model training |
| GET | /api/v1/analytics/overview | Dashboard overview |
| GET | /api/v1/scam-library | Scam knowledge base |

## 📊 Dashboard Pages

1. **Dashboard Overview** - Key metrics, fraud trends, recent alerts
2. **Transactions** - CRUD operations, filtering, bulk actions
3. **Fraud Detection** - Text, transaction, document, multi-modal inputs
4. **Alerts Management** - Alert triage, bulk updates, investigations
5. **Models (MLOps)** - Version management, training jobs, drift monitoring
6. **Analytics** - Trends, geographic analysis, time series, performance
7. **Scam Library** - Knowledge base with search and add functionality
8. **Settings** - Profile, thresholds, system configuration

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_upi_parser.py -v
pytest tests/test_risk_scorer.py -v
pytest tests/test_data_generator.py -v

# With coverage
pytest --cov=scamshield --cov-report=html
```

## 🚢 Deployment

### Local Development
```bash
python -m uvicorn scamshield.api.main:app --reload
streamlit run scamshield/frontend/streamlit_app.py
```

### Production (Docker)
```bash
docker-compose -f deploy/docker-compose.prod.yml up -d
```

### Cloud Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions:
- Render.com (one-click deploy)
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Apps

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
- ✅ Streamlit dashboards
- ✅ Plotly interactive charts
- ✅ Responsive UI design
- ✅ Real-time data visualization

### Security & Compliance
- ✅ OWASP security practices
- ✅ Data encryption
- ✅ Audit logging
- ✅ Privacy preservation

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Agents](https://langchain-ai.github.io/langgraph/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [MLflow Guide](https://mlflow.org/docs/latest/)
- [Streamlit Documentation](https://docs.streamlit.io/)

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

**Keywords**: Fraud Detection, ML/AI, Python, FastAPI, PyTorch, Transformers, LangChain, LangGraph, MLOps, Streamlit, Computer Vision, NLP, Generative AI, Agentic AI, Portfolio Project
