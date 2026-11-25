'use client'

import { useState, useEffect } from 'react'
import { Search, ChevronDown, ChevronRight, Copy, Check, ExternalLink, Github, BookOpen, Code, Terminal, Shield, Zap, Brain, Network, Server, Database, Settings, ArrowRight, Menu, X, Sun, Moon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'

interface DocSection {
  id: string
  title: string
  icon: React.ReactNode
  children: DocPage[]
}

interface DocPage {
  id: string
  title: string
  description: string
  content: string
}

const docSections: DocSection[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: <BookOpen className="h-5 w-5" />,
    children: [
      {
        id: 'introduction',
        title: 'Introduction',
        description: 'Overview of ScamShield AI platform',
        content: `# Introduction to ScamShield AI

ScamShield AI is a production-grade fraud detection platform designed for real-time UPI transaction monitoring and scam prevention. Built with modern ML/AI technologies, it achieves sub-50ms inference latency using ONNX Runtime with INT8 quantization and ensemble inference.

## Key Features

- **Real-time Scoring**: Sub-50ms latency for fraud detection
- **Ensemble Models**: XGBoost + LightGBM + Transformer + GNN
- **ONNX Runtime**: INT8 quantized models for production deployment
- **Scalable Architecture**: Kubernetes-native with horizontal autoscaling
- **Comprehensive Monitoring**: Prometheus/Grafana + OpenTelemetry tracing
- **Model Registry**: Versioning, A/B testing, canary deployments

## Architecture Overview

\`\`\`
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  API Gateway │────▶│  Feature    │
│  (Mobile/   │     │  (Kong/Envoy)│     │  Store      │
│   Web)      │     └─────────────┘     └──────┬──────┘
└─────────────┘                                  │
                                                 ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Decision   │◀───│  Ensemble   │◀───│  Feature    │
│  Engine     │     │  Inference  │     │  Computer   │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Alerting   │     │  Case Mgmt  │
│  & Notify   │     │  & Review   │
└─────────────┘     └─────────────┘
\`\`\`

## Quick Start

1. Clone the repository
2. Install dependencies: \`make install\`
3. Start services: \`make dev\`
4. Access dashboard at \`http://localhost:3000\`
5. API docs at \`http://localhost:8000/docs\`
`,
      },
      {
        id: 'installation',
        title: 'Installation',
        description: 'Set up ScamShield AI locally or in production',
        content: `# Installation Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Kubernetes (for production)
- PostgreSQL 15+
- Redis 7+

## Local Development

### 1. Clone Repository

\`\`\`bash
git clone https://github.com/your-org/scamshield-ai.git
cd scamshield-ai
\`\`\`

### 2. Backend Setup

\`\`\`bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your configuration
alembic upgrade head
uvicorn app.main:app --reload --port 8000
\`\`\`

### 3. Frontend Setup

\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`

### 4. Using Docker Compose

\`\`\`bash
docker-compose -f docker-compose.dev.yml up -d
\`\`\`

This starts:
- PostgreSQL on port 5432
- Redis on port 6379
- Backend API on port 8000
- Frontend on port 3000
- Prometheus on port 9090
- Grafana on port 3001

## Production Deployment

### Kubernetes (Recommended)

\`\`\`bash
# Install with Kustomize
kubectl apply -k k8s/overlays/prod

# Or with Helm
helm install scamshield ./helm/scamshield -n scamshield --create-namespace
\`\`\`

### Docker Swarm

\`\`\`bash
docker stack deploy -c docker-compose.prod.yml scamshield
\`\`\`

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| \`DATABASE_URL\` | PostgreSQL connection string | Required |
| \`REDIS_URL\` | Redis connection string | Required |
| \`MODEL_REGISTRY_PATH\` | S3/GCS path for model artifacts | Required |
| \`JWT_SECRET\` | JWT signing secret | Required |
| \`LOG_LEVEL\` | Logging level | INFO |
| \`ENABLE_TRACING\` | Enable OpenTelemetry | true |

See \`.env.example\` for complete list.
`,
      },
      {
        id: 'architecture',
        title: 'Architecture',
        description: 'System architecture and design decisions',
        content: `# System Architecture

## High-Level Components

### 1. API Gateway (Kong/Envoy)
- Rate limiting, authentication, request routing
- mTLS for service-to-service communication
- Request/response transformation

### 2. Feature Store (Redis + PostgreSQL)
- **Online Store**: Redis for real-time feature retrieval (<5ms)
- **Offline Store**: PostgreSQL for training data and batch features
- Feature versioning and lineage tracking

### 3. Feature Computer
- Real-time feature computation from raw transactions
- Behavioral, device, network, and velocity features
- Incremental updates with configurable windows

### 4. Ensemble Inference Engine
- Parallel model execution using ThreadPoolExecutor
- ONNX Runtime with INT8 quantization
- Weighted voting with uncertainty estimation

### 4. Decision Engine
- Rule-based decision logic (velocity, amount, device, graph)
- ML score integration with business rules
- Explainable decisions with feature attribution

### 5. Alerting & Notification
- Multi-channel alerting (Slack, PagerDuty, Email, SMS, Webhook)
- Alert deduplication and grouping
- Escalation policies with on-call rotation

### 6. Case Management
- Investigation workflow with status tracking
- Evidence collection and annotation
- Collaboration tools for fraud analysts

## Data Flow

\`\`\`
Transaction Request
       │
       ▼
┌──────────────────┐
│  Feature Store   │◀── Cached Features
│  (Redis)         │
└────────┬─────────┘
         │ Cache Miss
         ▼
┌──────────────────┐
│ Feature Computer │
│  (Real-time)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Ensemble Models  │
│  (Parallel)      │
│  • XGBoost       │
│  • LightGBM      │
│  • Transformer   │
│  • GNN           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Decision Engine  │
│  • ML Score      │
│  • Rule Engine   │
│  • Threat Intel  │
└────────┬─────────┘
         │
         ▼
   Decision Response
\`\`\`

## Technology Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI (Python 3.11+) |
| ORM | SQLAlchemy 2.0 (Async) |
| Validation | Pydantic v2 |
| ML Runtime | ONNX Runtime 1.16+ |
| Models | XGBoost, LightGBM, PyTorch |
| Feature Store | Redis + PostgreSQL |
| Message Queue | Redis Streams / Kafka |
| Monitoring | Prometheus + Grafana |
| Tracing | OpenTelemetry + Jaeger |
| Container | Docker + Kubernetes |
| CI/CD | GitHub Actions |
| Frontend | Next.js 14 + React 18 |
| Styling | Tailwind CSS |
| State | Zustand + TanStack Query |
| Charts | Recharts |

## Performance Targets

| Metric | Target |
|--------|--------|
| P50 Latency | < 20ms |
| P99 Latency | < 50ms |
| Throughput | > 10,000 req/s |
| Availability | 99.99% |
| Model Accuracy | > 97% (Ensemble) |
| False Positive Rate | < 0.5% |
`,
      },
    ],
  },
  {
    id: 'api-reference',
    title: 'API Reference',
    icon: <Code className="h-5 w-5" />,
    children: [
      {
        id: 'authentication',
        title: 'Authentication',
        description: 'API authentication and authorization',
        content: `# Authentication

## API Key Authentication

All API endpoints require authentication via API key.

### Header Authentication

\`\`\`http
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Query Parameter (Legacy)

\`\`\`http
GET /api/v1/score?api_key=sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

## Scopes

| Scope | Description |
|-------|-------------|
| \`score:read\` | Read scoring endpoints |
| \`score:write\` | Submit transactions for scoring |
| \`models:read\` | Read model registry |
| \`models:write\` | Manage model versions |
| \`alerts:read\` | Read alerts and rules |
| \`alerts:write\` | Manage alerts and rules |
| \`cases:read\` | Read investigation cases |
| \`cases:write\` | Manage investigation cases |
| \`admin\` | Administrative access |

## Rate Limits

| Tier | Requests/Second | Burst |
|------|-----------------|-------|
| Free | 10 | 50 |
| Pro | 100 | 500 |
| Enterprise | 10,000 | 50,000 |

Rate limit headers:
- \`X-RateLimit-Limit\`: Request limit
- \`X-RateLimit-Remaining\`: Remaining requests
- \`X-RateLimit-Reset\`: Unix timestamp when limit resets

## Error Responses

\`\`\`json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired API key",
    "details": {}
  }
}
\`\`\`

Common error codes:
- \`UNAUTHORIZED\` - Invalid credentials
- \`FORBIDDEN\` - Insufficient permissions
- \`RATE_LIMITED\` - Rate limit exceeded
- \`VALIDATION_ERROR\` - Request validation failed
- \`NOT_FOUND\` - Resource not found
- \`INTERNAL_ERROR\` - Server error
`,
      },
      {
        id: 'scoring-api',
        title: 'Scoring API',
        description: 'Real-time fraud scoring endpoints',
        content: `# Scoring API

## Score Transaction

Submit a transaction for real-time fraud scoring.

### Request

\`\`\`http
POST /api/v1/score
Content-Type: application/json
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

\`\`\`json
{
  "transaction_id": "txn_abc123",
  "payer_vpa": "user@upi",
  "payee_vpa": "merchant@upi",
  "amount": 5000.00,
  "currency": "INR",
  "timestamp": "2024-01-15T10:30:00Z",
  "device_id": "dev_xyz789",
  "ip_address": "203.0.113.45",
  "metadata": {
    "app_version": "2.1.0",
    "os": "android",
    "network_type": "wifi"
  }
}
\`\`\`

### Response

\`\`\`json
{
  "transaction_id": "txn_abc123",
  "decision": "APPROVE",
  "risk_score": 0.12,
  "confidence": 0.95,
  "ml_score": 0.08,
  "rule_score": 0.15,
  "threat_score": 0.05,
  "processing_time_ms": 18,
  "model_version": "ensemble_v3.1.0",
  "features": {
    "velocity_5m": 2,
    "amount_zscore": 0.5,
    "device_trust_score": 0.92,
    "merchant_risk_score": 0.15
  },
  "rules_triggered": [],
  "threat_intel": {
    "payer_reputation": "clean",
    "payee_reputation": "clean",
    "ip_reputation": "clean"
  },
  "explanation": {
    "top_factors": [
      {"feature": "device_trust_score", "impact": -0.15, "direction": "protective"},
      {"feature": "velocity_5m", "impact": 0.02, "direction": "risk"}
    ]
  }
}
\`\`\`

### Decision Values

| Decision | Description |
|----------|-------------|
| \`APPROVE\` | Low risk, auto-approve |
| \`REVIEW\` | Medium risk, manual review required |
| \`DECLINE\` | High risk, block transaction |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| \`decision\` | string | Final decision |
| \`risk_score\` | float | Combined risk score (0-1) |
| \`confidence\` | float | Model confidence (0-1) |
| \`ml_score\` | float | ML ensemble score |
| \`rule_score\` | float | Rule engine score |
| \`threat_score\` | float | Threat intelligence score |
| \`processing_time_ms\` | int | Total processing time |
| \`model_version\` | string | Model version used |
| \`features\` | object | Key computed features |
| \`rules_triggered\` | array | List of triggered rules |
| \`threat_intel\` | object | Threat intelligence data |
| \`explanation\` | object | Explainable AI output |

## Batch Scoring

Score multiple transactions in a single request.

### Request

\`\`\`http
POST /api/v1/score/batch
Content-Type: application/json
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

\`\`\`json
{
  "transactions": [
    { "transaction_id": "txn_001", "payer_vpa": "user1@upi", "payee_vpa": "merchant@upi", "amount": 1000 },
    { "transaction_id": "txn_002", "payer_vpa": "user2@upi", "payee_vpa": "merchant@upi", "amount": 50000 }
  ]
}
\`\`\`

### Response

\`\`\`json
{
  "results": [
    { "transaction_id": "txn_001", "decision": "APPROVE", "risk_score": 0.05, "processing_time_ms": 12 },
    { "transaction_id": "txn_002", "decision": "DECLINE", "risk_score": 0.92, "processing_time_ms": 15 }
  ],
  "batch_id": "batch_abc123",
  "total_processing_time_ms": 28
}
\`\`\`

## Get Score History

Retrieve scoring history for a transaction or user.

\`\`\`http
GET /api/v1/score/history?payer_vpa=user@upi&limit=100
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`
`,
      },
      {
        id: 'models-api',
        title: 'Models API',
        description: 'Model registry and management endpoints',
        content: `# Models API

## List Models

\`\`\`http
GET /api/v1/models
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Response

\`\`\`json
{
  "models": [
    {
      "id": "model_001",
      "name": "UPI Fraud XGBoost Classifier",
      "type": "xgboost",
      "description": "Gradient boosted decision trees...",
      "current_version": "v2.3.1",
      "tags": ["production", "low-latency"],
      "owner": "ml-team",
      "created_at": "2023-08-15T10:00:00Z",
      "updated_at": "2024-01-10T14:30:00Z"
    }
  ],
  "total": 5
}
\`\`\`

## Get Model Details

\`\`\`http
GET /api/v1/models/{model_id}
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

## List Model Versions

\`\`\`http
GET /api/v1/models/{model_id}/versions
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Response

\`\`\`json
{
  "versions": [
    {
      "id": "ver_001",
      "version": "v2.3.1",
      "status": "production",
      "accuracy": 0.962,
      "precision": 0.941,
      "recall": 0.923,
      "f1": 0.932,
      "latency_ms": 8,
      "training_date": "2024-01-05T08:00:00Z",
      "deployed_date": "2024-01-10T14:30:00Z",
      "artifacts": {
        "onnx_path": "s3://models/xgboost/v2.3.1/model.onnx",
        "config_path": "s3://models/xgboost/v2.3.1/config.json",
        "metrics_path": "s3://models/xgboost/v2.3.1/metrics.json"
      },
      "metadata": {
        "training_data_size": 2500000,
        "features": 50,
        "hyperparameters": { "n_estimators": 500, "max_depth": 8 },
        "git_commit": "a1b2c3d4",
        "trained_by": "ml-pipeline"
      }
    }
  ]
}
\`\`\`

## Promote Version

\`\`\`http
POST /api/v1/models/{model_id}/versions/{version_id}/promote
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "environment": "production",
  "strategy": "blue_green",
  "canary_percentage": 0
}
\`\`\`

## Rollback Version

\`\`\`http
POST /api/v1/models/{model_id}/versions/{version_id}/rollback
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

## Register New Model

\`\`\`http
POST /api/v1/models
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "name": "New Fraud Model",
  "type": "xgboost",
  "description": "Experimental model for testing",
  "tags": ["experimental"],
  "version": "v1.0.0",
  "artifacts": {
    "onnx_path": "s3://models/new_model/v1.0.0/model.onnx",
    "config_path": "s3://models/new_model/v1.0.0/config.json"
  },
  "metadata": {
    "training_data_size": 100000,
    "features": 50,
    "hyperparameters": {}
  }
}
\`\`\`
`,
      },
      {
        id: 'alerts-api',
        title: 'Alerts API',
        description: 'Alert management and notification endpoints',
        content: `# Alerts API

## List Alerts

\`\`\`http
GET /api/v1/alerts?status=firing&severity=critical&limit=50
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| \`status\` | string | Filter: firing, acknowledged, resolved, silenced |
| \`severity\` | string | Filter: critical, high, medium, low |
| \`assignee\` | string | Filter by assignee email |
| \`rule_id\` | string | Filter by rule ID |
| \`since\` | datetime | Filter alerts after timestamp |
| \`limit\` | int | Max results (default: 50, max: 500) |
| \`offset\` | int | Pagination offset |

### Response

\`\`\`json
{
  "alerts": [
    {
      "id": "alert_001",
      "rule_id": "rule_001",
      "rule_name": "High Fraud Rate Alert",
      "severity": "critical",
      "title": "Fraud rate exceeded 0.5% threshold",
      "message": "Current fraud rate: 0.67%...",
      "status": "firing",
      "fired_at": "2024-01-15T10:25:00Z",
      "acknowledged_at": null,
      "resolved_at": null,
      "assignee": null,
      "metadata": {
        "current_value": 0.0067,
        "threshold": 0.005
      }
    }
  ],
  "total": 142
}
\`\`\`

## Acknowledge Alert

\`\`\`http
POST /api/v1/alerts/{alert_id}/acknowledge
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "assignee": "analyst@company.com",
  "note": "Investigating potential coordinated attack"
}
\`\`\`

## Resolve Alert

\`\`\`http
POST /api/v1/alerts/{alert_id}/resolve
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "resolution": "false_positive",
  "note": "Verified legitimate bulk payments",
  "action_taken": "User whitelisted for 24h"
}
\`\`\`

## Silence Alert

\`\`\`http
POST /api/v1/alerts/{alert_id}/silence
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "duration_minutes": 60,
  "reason": "Known maintenance window"
}
\`\`\`

## Alert Rules

### List Rules

\`\`\`http
GET /api/v1/alerts/rules
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Create Rule

\`\`\`http
POST /api/v1/alerts/rules
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "name": "Custom Velocity Alert",
  "description": "Alert on high velocity for specific user segment",
  "condition": "user_velocity_5m > 100 AND user_segment = 'premium'",
  "severity": "high",
  "enabled": true,
  "channels": ["slack", "email"],
  "threshold": 100,
  "window": "5m",
  "cooldown": 300
}
\`\`\`

### Update Rule

\`\`\`http
PATCH /api/v1/alerts/rules/{rule_id}
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "enabled": false,
  "threshold": 150
}
\`\`\`

### Test Rule

\`\`\`http
POST /api/v1/alerts/rules/{rule_id}/test
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "sample_data": { "user_velocity_5m": 150, "user_segment": "premium" }
}
\`\`\`

## Notification Channels

### List Channels

\`\`\`http
GET /api/v1/alerts/channels
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Test Channel

\`\`\`http
POST /api/v1/alerts/channels/{channel_id}/test
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`
`,
      },
      {
        id: 'cases-api',
        title: 'Cases API',
        description: 'Investigation case management endpoints',
        content: `# Cases API

## List Cases

\`\`\`http
GET /api/v1/cases?status=open&assignee=analyst@company.com
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

### Response

\`\`\`json
{
  "cases": [
    {
      "id": "case_001",
      "title": "Coordinated Fraud Attack - IP Range 203.0.113.0/24",
      "description": "Multiple high-value transactions from suspicious IP range",
      "status": "in_progress",
      "priority": "critical",
      "assignee": "fraud-lead@company.com",
      "created_at": "2024-01-14T16:30:00Z",
      "updated_at": "2024-01-15T10:00:00Z",
      "alert_ids": ["alert_005"],
      "transaction_ids": ["txn_001", "txn_002", "txn_003"],
      "tags": ["coordinated-attack", "ip-block"]
    }
  ],
  "total": 23
}
\`\`\`

## Create Case

\`\`\`http
POST /api/v1/cases
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "title": "Suspicious Activity - User user123@upi",
  "description": "Unusual transaction pattern detected",
  "priority": "high",
  "assignee": "analyst@company.com",
  "alert_ids": ["alert_003"],
  "transaction_ids": ["txn_abc123"],
  "tags": ["velocity", "new-device"]
}
\`\`\`

## Get Case Details

\`\`\`http
GET /api/v1/cases/{case_id}
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
\`\`\`

## Update Case

\`\`\`http
PATCH /api/v1/cases/{case_id}
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "status": "in_progress",
  "assignee": "senior-analyst@company.com",
  "priority": "critical"
}
\`\`\`

## Add Case Note

\`\`\`http
POST /api/v1/cases/{case_id}/notes
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "content": "Reviewed transaction history. User verified identity via video KYC.",
  "type": "investigation"
}
\`\`\`

## Add Evidence

\`\`\`http
POST /api/v1/cases/{case_id}/evidence
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "type": "screenshot",
  "title": "Transaction Pattern Analysis",
  "description": "Chart showing velocity spike",
  "url": "https://storage.company.com/evidence/case_001/chart.png",
  "metadata": { "chart_type": "velocity_timeline" }
}
\`\`\`

## Close Case

\`\`\`http
POST /api/v1/cases/{case_id}/close
Authorization: Bearer sk_live_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json

{
  "resolution": "fraud_confirmed",
  "summary": "Coordinated attack confirmed. 47 fraudulent transactions blocked. IP range added to blocklist.",
  "action_taken": [
    "IP range 203.0.113.0/24 blocked at WAF",
    "Affected users notified",
    "Merchants compensated"
  ]
}
\`\`\`
`,
      },
    ],
  },
  {
    id: 'guides',
    title: 'Guides & Tutorials',
    icon: <Terminal className="h-5 w-5" />,
    children: [
      {
        id: 'feature-engineering',
        title: 'Feature Engineering Guide',
        description: 'Understanding and customizing features',
        content: `# Feature Engineering Guide

## Feature Categories

### 1. Behavioral Features
Computed from user's historical transaction patterns.

| Feature | Description | Window |
|---------|-------------|--------|
| \`avg_txn_amount_7d\` | Average transaction amount | 7 days |
| \`avg_txn_amount_30d\` | Average transaction amount | 30 days |
| \`txn_count_7d\` | Transaction count | 7 days |
| \`txn_count_30d\` | Transaction count | 30 days |
| \`unique_merchants_7d\` | Unique merchants transacted with | 7 days |
| \`unique_merchants_30d\` | Unique merchants transacted with | 30 days |
| \`avg_txn_interval_hours\` | Average time between transactions | 30 days |
| \`night_txn_ratio\` | Ratio of night transactions (10PM-6AM) | 30 days |
| \`weekend_txn_ratio\` | Ratio of weekend transactions | 30 days |
| \`favorite_merchant_category\` | Most frequent merchant category | 30 days |

### 2. Device Features
Device fingerprinting and trust signals.

| Feature | Description |
|---------|-------------|
| \`device_trust_score\` | Composite trust score (0-1) |
| \`is_rooted\` | Device is rooted/jailbroken |
| \`is_emulator\` | Running in emulator |
| \`has_vpn\` | VPN/proxy detected |
| \`os_version\` | Operating system version |
| \`app_version\` | App version |
| \`screen_resolution\` | Screen resolution |
| \`device_age_days\` | Days since first seen |
| \`device_model_risk\` | Model-based risk score |

### 3. Network Features
Network and connection analysis.

| Feature | Description |
|---------|-------------|
| \`ip_reputation_score\` | IP reputation (0-1) |
| \`asn_reputation\` | ASN reputation score |
| \`country_risk\` | Country risk level |
| \`is_tor\` | Tor exit node detected |
| \`is_proxy\` | Proxy detected |
| \`connection_type\` | wifi/cellular/ethernet |
| \`carrier_risk\` | Carrier risk score |

### 4. Velocity Features
Transaction frequency and pattern analysis.

| Feature | Description | Window |
|---------|-------------|--------|
| \`velocity_1m\` | Transactions per minute | 1 min |
| \`velocity_5m\` | Transactions per 5 minutes | 5 min |
| \`velocity_1h\` | Transactions per hour | 1 hour |
| \`velocity_24h\` | Transactions per day | 24 hours |
| \`velocity_amount_5m\` | Amount per 5 minutes | 5 min |
| \`unique_payees_5m\` | Unique payees in 5 min | 5 min |
| \`burst_score\` | Burst transaction indicator | 1 hour |

### 5. Graph Features
Entity relationship analysis via GNN.

| Feature | Description |
|---------|-------------|
| \`pagerank_score\` | PageRank in transaction graph |
| \`community_id\` | Community cluster ID |
| \`clustering_coefficient\` | Local clustering coefficient |
| \`degree_centrality\` | Node degree centrality |
| \`betweenness_centrality\` | Betweenness centrality |
| \`shared_device_count\` | Users sharing same device |
| \`shared_ip_count\` | Users sharing same IP |

## Custom Features

### Adding New Features

1. Define feature in \`upishield_core/features/definitions.py\`

\`\`\`python
from upishield_core.features.base import Feature, FeatureType

class CustomVelocityFeature(Feature):
    name = "custom_velocity_10m"
    description = "Transaction velocity over 10 minutes"
    feature_type = FeatureType.VELOCITY
    window = "10m"
    dependencies = ["transaction_events"]

    async def compute(self, context: FeatureContext) -> float:
        events = await context.get_events("transaction_events", window="10m")
        return len(events) / 10.0  # per minute
\`\`\`

2. Register in feature registry

\`\`\`python
from upishield_core.features.registry import feature_registry

feature_registry.register(CustomVelocityFeature)
\`\`\`

3. Add to feature vector in \`compute.py\`

\`\`\`python
FEATURE_VECTOR = [
    # ... existing features
    "custom_velocity_10m",
]
\`\`\`

### Feature Validation

All features must pass validation:

\`\`\`python
# In tests/test_features.py
def test_custom_velocity_feature():
    feature = CustomVelocityFeature()
    result = await feature.compute(mock_context)
    assert 0 <= result <= 1000  # Reasonable bounds
\`\`\`

## Feature Importance

Current model feature importance (Ensemble v3.1.0):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | device_trust_score | 0.187 | Device |
| 2 | velocity_5m | 0.156 | Velocity |
| 3 | ip_reputation_score | 0.134 | Network |
| 4 | avg_txn_amount_7d | 0.112 | Behavioral |
| 5 | merchant_risk_score | 0.098 | Merchant |
| 6 | pagerank_score | 0.087 | Graph |
| 7 | txn_count_7d | 0.076 | Behavioral |
| 8 | night_txn_ratio | 0.054 | Behavioral |
| 9 | is_rooted | 0.048 | Device |
| 10 | unique_payees_5m | 0.042 | Velocity |

## Feature Monitoring

Monitor feature drift in production:

\`\`\`bash
# Check feature distributions
python -m upishield_core.features.monitor --feature velocity_5m --days 7

# Alert on drift
python -m upishield_core.features.monitor --alert-threshold 0.1
\`\`\`
`,
      },
      {
        id: 'model-training',
        title: 'Model Training Pipeline',
        description: 'Training, evaluation, and deployment of models',
        content: `# Model Training Pipeline

## Overview

The training pipeline uses MLflow for experiment tracking and Kubeflow Pipelines for orchestration.

## Pipeline Stages

\`\`\`
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Data        │──▶│ Feature     │──▶│ Training    │──▶│ Evaluation  │──▶│ Deployment  │
│ Extraction  │   │ Engineering │   │ (Distributed)│   │ & Validation│   │ (Staging)   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
\`\`\`

### 1. Data Extraction

\`\`\`bash
# Extract training data from feature store
python -m training.pipelines.extract \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output s3://training-data/fraud_2023.parquet
\`\`\`

### 2. Feature Engineering

\`\`\`bash
# Generate feature matrix
python -m training.pipelines.features \
  --input s3://training-data/fraud_2023.parquet \
  --feature-config config/features.yaml \
  --output s3://training-data/features_2023.parquet
\`\`\`

### 3. Model Training

#### XGBoost

\`\`\`bash
python -m training.models.xgboost_train \
  --data s3://training-data/features_2023.parquet \
  --params config/xgboost_params.yaml \
  --output s3://models/xgboost/v2.3.1/ \
  --mlflow-experiment "fraud-xgboost"
\`\`\`

#### LightGBM

\`\`\`bash
python -m training.models.lightgbm_train \
  --data s3://training-data/features_2023.parquet \
  --params config/lightgbm_params.yaml \
  --output s3://models/lightgbm/v1.8.2/ \
  --mlflow-experiment "fraud-lightgbm"
\`\`\`

#### Transformer

\`\`\`bash
python -m training.models.transformer_train \
  --data s3://training-data/sequences_2023.parquet \
  --params config/transformer_params.yaml \
  --output s3://models/transformer/v1.2.0/ \
  --mlflow-experiment "fraud-transformer" \
  --gpus 4
\`\`\`

#### GNN

\`\`\`bash
python -m training.models.gnn_train \
  --graph s3://training-data/graph_2023.pt \
  --params config/gnn_params.yaml \
  --output s3://models/gnn/v1.0.5/ \
  --mlflow-experiment "fraud-gnn"
\`\`\`

### 4. Model Export to ONNX

\`\`\`bash
# Convert to ONNX with INT8 quantization
python -m training.export.onnx_export \
  --model-path s3://models/xgboost/v2.3.1/model.pkl \
  --output s3://models/xgboost/v2.3.1/model.onnx \
  --quantize int8 \
  --opset 17
\`\`\`

### 5. Evaluation & Validation

\`\`\`bash
# Run evaluation suite
python -m training.eval.evaluate \
  --model s3://models/xgboost/v2.3.1/model.onnx \
  --test-data s3://training-data/test_2023.parquet \
  --metrics accuracy,precision,recall,f1,auc,latency \
  --threshold config/eval_thresholds.yaml
\`\`\`

### 6. Deployment

\`\`\`bash
# Deploy to staging
kubectl apply -k k8s/overlays/staging

# Promote to production (after validation)
python -m training.deploy.promote \
  --model xgboost \
  --version v2.3.1 \
  --environment production \
  --strategy blue_green
\`\`\`

## Hyperparameter Configuration

### XGBoost Parameters

\`\`\`yaml
# config/xgboost_params.yaml
n_estimators: 500
max_depth: 8
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 3
gamma: 0.1
reg_alpha: 0.1
reg_lambda: 1.0
objective: "binary:logistic"
eval_metric: "auc"
n_jobs: -1
random_state: 42
\`\`\`

### Transformer Parameters

\`\`\`yaml
# config/transformer_params.yaml
d_model: 256
n_heads: 8
n_layers: 6
dropout: 0.1
seq_len: 20
vocab_size: 50000
batch_size: 256
learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 1000
max_epochs: 50
early_stopping_patience: 5
\`\`\`

## Experiment Tracking

All experiments tracked in MLflow:

\`\`\`bash
# Start MLflow UI
mlflow ui --port 5000

# Compare runs
python -m training.utils.compare_runs \
  --experiment fraud-xgboost \
  --metric f1 \
  --top-k 5
\`\`\`

## Model Validation Checklist

Before promoting to production:

- [ ] Accuracy > 95% (or improvement over current)
- [ ] Precision > 93% (for fraud class)
- [ ] Recall > 90% (for fraud class)
- [ ] F1 > 92%
- [ ] P99 Latency < 20ms (single model) / < 50ms (ensemble)
- [ ] No significant drift in feature distributions
- [ ] Passes adversarial test suite
- [ ] Passes fairness checks (demographic parity)
- [ ] ONNX export successful with INT8 quantization
- [ ] Load test passed at 2x expected traffic
`,
      },
      {
        id: 'deployment-guide',
        title: 'Deployment Guide',
        description: 'Production deployment strategies and operations',
        content: `# Deployment Guide

## Environments

| Environment | Purpose | URL | Replicas |
|-------------|---------|-----|----------|
| Development | Local development | localhost:8000 | 1 |
| Staging | Pre-production testing | staging.api.scamshield.ai | 2-5 |
| Production | Live traffic | api.scamshield.ai | 3-20 (HPA) |

## Kubernetes Deployment

### Base Configuration

\`\`\`yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scamshield-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scamshield-api
  template:
    metadata:
      labels:
        app: scamshield-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: api
        image: scamshield/api:v3.1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        envFrom:
        - configMapRef:
            name: scamshield-config
        - secretRef:
            name: scamshield-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
\`\`\`

### Horizontal Pod Autoscaler

\`\`\`yaml
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: scamshield-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: scamshield-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
\`\`\`

### Production Overlay

\`\`\`yaml
# k8s/overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
patches:
- patch: |-
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: scamshield-api
    spec:
      replicas: 5
  target:
    kind: Deployment
    name: scamshield-api
images:
- name: scamshield/api
  newTag: v3.1.0-prod
configMapGenerator:
- name: scamshield-config
  behavior: merge
  literals:
  - LOG_LEVEL=INFO
  - ENABLE_TRACING=true
  - ENVIRONMENT=production
\`\`\`

## Deployment Strategies

### Blue-Green Deployment

\`\`\`bash
# Deploy new version to green environment
kubectl apply -k k8s/overlays/green

# Run smoke tests
python -m tests.smoke --env green

# Switch traffic
kubectl patch service scamshield-api -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for 10 minutes, then decommission blue
\`\`\`

### Canary Deployment

\`\`\`bash
# Deploy canary (5% traffic)
kubectl apply -f k8s/canary.yaml

# Gradually increase traffic
for i in 10 25 50 100; do
  kubectl patch virtualservice scamshield -p "{\"spec\":[{\"route\":[{\"destination\":{\"host\":\"scamshield-api\",\"subset\":\"v3.1.0\"},\"weight\":$i},{\"destination\":{\"host\":\"scamshield-api\",\"subset\":\"v3.0.2\"},\"weight\":$((100-i))}}]}"
  sleep 300
  # Check metrics
  python -m tests.canary_check --threshold 0.01
done
\`\`\`

### Rollback

\`\`\`bash
# Quick rollback
kubectl rollout undo deployment/scamshield-api

# Or specific version
kubectl rollout undo deployment/scamshield-api --to-revision=5
\`\`\`

## Database Migrations

\`\`\`bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Add new column"

# Rollback migration
alembic downgrade -1
\`\`\`

## Secrets Management

### Using Kubernetes Secrets

\`\`\`bash
# Create secret
kubectl create secret generic scamshield-secrets \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=JWT_SECRET="..." \
  --from-literal=REDIS_URL="redis://..." \
  -n scamshield
\`\`\`

### Using External Secrets Operator (AWS Secrets Manager)

\`\`\`yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: scamshield-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: ClusterSecretStore
  target:
    name: scamshield-secrets
  data:
  - secretKey: DATABASE_URL
    remoteRef:
      key: prod/scamshield/database
      property: url
  - secretKey: JWT_SECRET
    remoteRef:
      key: prod/scamshield/jwt
      property: secret
\`\`\`

## Monitoring Setup

### Prometheus Rules

\`\`\`yaml
# k8s/monitoring/prometheus-rules.yaml
groups:
- name: scamshield-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "P99 latency exceeds 50ms"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Error rate exceeds 1%"

  - alert: ModelAccuracyDrop
    expr: model_accuracy < 0.95
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy below 95%"
\`\`\`

### Grafana Dashboards

Import dashboards from \`grafana/dashboards/\`:

- \`scamshield-overview.json\` - System overview
- \`scamshield-scoring.json\` - Scoring performance
- \`scamshield-models.json\` - Model metrics
- \`scamshield-business.json\` - Business KPIs

## Disaster Recovery

### Backup Strategy

| Component | Frequency | Retention |
|-----------|-----------|-----------|
| PostgreSQL | Daily + WAL | 30 days |
| Redis | Hourly RDB | 7 days |
| Model Artifacts | On deploy | Permanent |
| ConfigMaps/Secrets | On change | 90 days |

### Recovery Procedures

\`\`\`bash
# Restore database
pg_restore -d scamshield backup_20240115.dump

# Restore Redis
redis-cli -r /path/to/dump.rdb

# Verify model artifacts
aws s3 ls s3://models/ --recursive
\`\`\`

## Security Hardening

- Network policies for pod-to-pod communication
- mTLS with Istio/Linkerd
- Pod security standards (restricted)
- Image signing with cosign
- Runtime security with Falco
- Regular vulnerability scanning (Trivy)
`,
      },
    ],
  },
  {
    id: 'operations',
    title: 'Operations',
    icon: <Settings className="h-5 w-5" />,
    children: [
      {
        id: 'monitoring',
        title: 'Monitoring & Observability',
        description: 'Metrics, logging, and tracing setup',
        content: `# Monitoring & Observability

## Metrics (Prometheus)

### Key Metrics

#### Request Metrics
- \`http_requests_total\` - Total requests by method, path, status
- \`http_request_duration_seconds\` - Request latency histogram
- \`http_request_size_bytes\` - Request size distribution
- \`http_response_size_bytes\` - Response size distribution

#### Business Metrics
- \`fraud_transactions_total\` - Fraud transactions by decision
- \`fraud_rate\` - Current fraud rate (5m window)
- \`amount_protected_total\` - Total amount protected
- \`active_users\` - Currently active users

#### Model Metrics
- \`model_inference_duration_seconds\` - Model inference latency
- \`model_predictions_total\` - Predictions by model, outcome
- \`model_accuracy\` - Current model accuracy (24h window)
- \`ensemble_agreement\` - Model agreement rate

#### System Metrics
- \`feature_store_latency_seconds\` - Feature store latency
- \`feature_store_hit_rate\` - Cache hit rate
- \`queue_depth\` - Processing queue depth
- \`active_connections\` - Active WebSocket connections

### Query Examples

\`\`\`promql
# P99 latency over 5m
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Fraud rate by decision
rate(fraud_transactions_total[5m])

# Model accuracy trend
model_accuracy[1h]

# Feature store hit rate
rate(feature_store_hits_total[5m]) / rate(feature_store_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
\`\`\`

## Logging (Structured JSON)

### Log Format

\`\`\`json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "upishield_core.decision.engine",
  "trace_id": "abc123def456",
  "span_id": "789ghi012",
  "message": "Transaction scored",
  "fields": {
    "transaction_id": "txn_abc123",
    "decision": "APPROVE",
    "risk_score": 0.12,
    "processing_time_ms": 18
  }
}
\`\`\`

### Log Levels

| Level | Use Case |
|-------|----------|
| \`DEBUG\` | Detailed diagnostic information |
| \`INFO\` | General operational events |
| \`WARNING\` | Unexpected but handled situations |
| \`ERROR\` | Failed operations requiring attention |
| \`CRITICAL\` | System-wide issues |

### Log Aggregation

Logs shipped to:
- **Loki** (development) - \`http://localhost:3100\`
- **Elasticsearch** (production) - \`https://logs.company.com\`

Query with LogQL:
\`\`\`logql
{app="scamshield-api"} |= "ERROR" | json | level="ERROR"
{app="scamshield-api"} | json | decision="DECLINE" | risk_score > 0.8
\`\`\`

## Distributed Tracing (OpenTelemetry)

### Trace Context Propagation

\`\`\`http
# Request headers
traceparent: 00-abc123def4567890abc123def456789-789ghi012jkl3456-01
tracestate: vendor=scamshield
\`\`\`

### Key Spans

| Span Name | Description |
|-----------|-------------|
| \`score_transaction\` | End-to-end scoring |
| \`fetch_features\` | Feature store lookup |
| \`compute_features\` | Real-time feature computation |
| \`ensemble_inference\` | Parallel model inference |
| \`decision_engine\` | Rule evaluation & decision |
| \`publish_result\` | Result publishing |

### Trace Analysis

\`\`\`bash
# Find slow traces
jaeger-query --service scamshield-api --min-duration 50ms

# Trace by transaction ID
jaeger-query --tag transaction_id=txn_abc123
\`\`\`

## Alerting Rules

### Critical Alerts

\`\`\`yaml
# P99 latency > 50ms for 5m
- alert: ScoringLatencyHigh
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{handler="/score"}[5m])) > 0.05
  for: 5m
  labels:
    severity: critical
    team: platform
  annotations:
    summary: "Scoring API P99 latency > 50ms"
    runbook: "https://wiki.scamshield.ai/runbooks/high-latency"

# Error rate > 1%
- alert: ScoringErrorRateHigh
  expr: rate(http_requests_total{handler="/score",status=~"5.."}[5m]) > 0.01
  for: 2m
  labels:
    severity: critical
    team: platform
  annotations:
    summary: "Scoring API error rate > 1%"

# Feature store unavailable
- alert: FeatureStoreDown
  expr: up{job="feature-store"} == 0
  for: 1m
  labels:
    severity: critical
    team: data
  annotations:
    summary: "Feature store is down"
\`\`\`

### Warning Alerts

\`\`\`yaml
# Model accuracy dropping
- alert: ModelAccuracyDegrading
  expr: model_accuracy < 0.96
  for: 15m
  labels:
    severity: warning
    team: ml
  annotations:
    summary: "Model accuracy below 96%"

# Queue backing up
- alert: ProcessingQueueBacklog
  expr: queue_depth > 1000
  for: 5m
  labels:
    severity: warning
    team: platform
  annotations:
    summary: "Processing queue has >1000 items"

# Low cache hit rate
- alert: FeatureStoreCacheMiss
  expr: rate(feature_store_misses_total[5m]) / rate(feature_store_requests_total[5m]) > 0.3
  for: 10m
  labels:
    severity: warning
    team: data
  annotations:
    summary: "Feature store cache hit rate < 70%"
\`\`\`

## Runbooks

Common incident runbooks:

1. **High Latency** - \`/runbooks/high-latency.md\`
2. **High Error Rate** - \`/runbooks/high-error-rate.md\`
3. **Model Accuracy Drop** - \`/runbooks/model-accuracy-drop.md\`
4. **Feature Store Down** - \`/runbooks/feature-store-down.md\`
5. **Queue Backlog** - \`/runbooks/queue-backlog.md\`
6. **Database Connection Pool Exhausted** - \`/runbooks/db-pool-exhausted.md\`

## SLOs (Service Level Objectives)

| SLI | SLO | Measurement Window |
|-----|-----|-------------------|
| Availability | 99.99% | 30 days |
| Latency (P99) | < 50ms | 30 days |
| Error Rate | < 0.1% | 30 days |
| Scoring Throughput | > 10k req/s | 5 min |
| Model Accuracy | > 97% | 24 hours |
| Feature Store Latency (P99) | < 10ms | 30 days |
`,
      },
      {
        id: 'incident-response',
        title: 'Incident Response',
        description: 'Incident response procedures and escalation',
        content: `# Incident Response

## Severity Levels

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| **SEV-1** (Critical) | Service unavailable, data loss, security breach | 15 min | Immediate - Page on-call + Manager |
| **SEV-2** (High) | Major feature degraded, high error rate | 30 min | Page on-call |
| **SEV-3** (Medium) | Minor feature degraded, performance issue | 2 hours | Assign to team |
| **SEV-4** (Low) | Cosmetic issue, documentation | Next business day | Track in backlog |

## Incident Lifecycle

\`\`\`
DETECTED → ACKNOWLEDGED → INVESTIGATING → MITIGATED → RESOLVED → POSTMORTEM
    │           │              │            │          │           │
    ▼           ▼              ▼            ▼          ▼           ▼
  Alert      On-call        Root cause   Workaround  Fix deployed  Blameless
  fires      responds       analysis     applied      & verified   retrospective
\`\`\`

## Response Procedures

### SEV-1: Critical Incident

1. **Detection** (Automated)
   - PagerDuty alerts on-call engineer
   - Slack #incidents channel notified
   - Status page updated automatically

2. **Acknowledgment** (0-15 min)
   - On-call acknowledges in PagerDuty
   - Joins incident bridge (Zoom/Teams)
   - Posts initial status in #incidents

3. **Investigation** (15-60 min)
   - Check dashboards: Grafana, Jaeger, Logs
   - Identify blast radius
   - Determine root cause category:
     - Deployment issue → Rollback
     - Infrastructure → Scale/Restart
     - Dependency → Circuit breaker/Failover
     - Code bug → Hotfix

4. **Mitigation** (30-120 min)
   - Apply workaround
   - Communicate to stakeholders
   - Update status page every 15 min

5. **Resolution** (Variable)
   - Deploy fix
   - Verify recovery
   - Monitor for 30 min

6. **Postmortem** (Within 48 hours)
   - Blameless retrospective
   - Action items with owners
   - Share learnings

### Communication Templates

#### Initial Status (Slack)

\`\`\`
🚨 INCIDENT DECLARED - SEV-1
**Service**: ScamShield API
**Impact**: Scoring API returning 503 errors
**Started**: 2024-01-15 10:30 UTC
**On-call**: @engineer-name
**Bridge**: [Zoom Link]
**Status**: Investigating - checking recent deployment
\`\`\`

#### Update (Every 15 min)

\`\`\`
🔄 INCIDENT UPDATE - SEV-1
**Service**: ScamShield API
**Impact**: Scoring API returning 503 errors (100% errors)
**Root Cause**: Identified - database connection pool exhaustion from v3.1.0 deployment
**Mitigation**: Rolling back to v3.0.2 (in progress, 50% complete)
**ETA**: 10:45 UTC
**Next Update**: 10:40 UTC
\`\`\`

#### Resolution

\`\`\`
✅ INCIDENT RESOLVED - SEV-1
**Service**: ScamShield API
**Duration**: 25 minutes (10:30 - 10:55 UTC)
**Root Cause**: Database connection pool misconfiguration in v3.1.0
**Resolution**: Rolled back to v3.0.2, connections stabilized
**Action Items**:
- Fix connection pool config (PR #1234, @engineer)
- Add pre-deploy connection pool validation (PR #1235, @engineer)
- Improve deployment smoke tests (PR #1236, @engineer)
**Postmortem**: Scheduled 2024-01-17 14:00 UTC
\`\`\`

## Escalation Contacts

| Role | Primary | Secondary |
|------|---------|-----------|
| Platform On-call | @platform-oncall | @platform-lead |
| ML On-call | @ml-oncall | @ml-lead |
| Data On-call | @data-oncall | @data-lead |
| Engineering Manager | @eng-manager | @vp-engineering |
| Security | @security-oncall | @ciso |

## Runbook Index

| Scenario | Runbook | Owner |
|----------|---------|-------|
| API High Latency | \`runbooks/high-latency.md\` | Platform |
| API High Error Rate | \`runbooks/high-error-rate.md\` | Platform |
| Model Accuracy Drop | \`runbooks/model-accuracy-drop.md\` | ML |
| Feature Store Down | \`runbooks/feature-store-down.md\` | Data |
| Database Issues | \`runbooks/database-issues.md\` | Platform |
| Redis Issues | \`runbooks/redis-issues.md\` | Platform |
| Kafka Issues | \`runbooks/kafka-issues.md\` | Data |
| Security Breach | \`runbooks/security-breach.md\` | Security |
| Capacity Exhaustion | \`runbooks/capacity-exhaustion.md\` | Platform |
| Deployment Failure | \`runbooks/deployment-failure.md\` | Platform |

## Postmortem Template

\`\`\`markdown
# Incident Postmortem: [INC-XXXX]

## Summary
- **Date**: YYYY-MM-DD
- **Duration**: XX minutes
- **Severity**: SEV-X
- **Services Affected**: [list]
- **Impact**: [user-facing impact]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert fired |
| HH:MM | On-call acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |

## Root Cause
[Technical explanation of what happened]

## Contributing Factors
- [Factor 1]
- [Factor 2]

## Resolution
[What fixed the issue]

## Action Items
| Action | Owner | Due Date | Tracking |
|--------|-------|----------|----------|
| [Action 1] | @owner | YYYY-MM-DD | JIRA-XXXX |
| [Action 2] | @owner | YYYY-MM-DD | JIRA-XXXX |

## Lessons Learned
- [Lesson 1]
- [Lesson 2]
\`\`\`

## War Room Guidelines

1. **Single incident commander** - Clear decision authority
2. **Time-boxed updates** - Every 15 min for SEV-1, 30 min for SEV-2
3. **Shared document** - Collaborative timeline in Google Docs/Notion
4. **No blame culture** - Focus on systems, not individuals
5. **Stakeholder communication** - Dedicated communicator for external updates
`,
      },
    ],
  },
  {
    id: 'reference',
    title: 'Reference',
    icon: <Shield className="h-5 w-5" />,
    children: [
      {
        id: 'configuration',
        title: 'Configuration Reference',
        description: 'Complete configuration options',
        content: `# Configuration Reference

## Environment Variables

### Core Application

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`APP_NAME\` | string | No | "ScamShield AI" | Application name |
| \`ENVIRONMENT\` | string | No | "development" | Environment: development, staging, production |
| \`LOG_LEVEL\` | string | No | "INFO" | Log level: DEBUG, INFO, WARNING, ERROR |
| \`API_PREFIX\` | string | No | "/api/v1" | API route prefix |
| \`ENABLE_DOCS\` | bool | No | true | Enable OpenAPI docs |
| \`ENABLE_TRACING\` | bool | No | true | Enable OpenTelemetry tracing |

### Database

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`DATABASE_URL\` | string | Yes | - | PostgreSQL connection URL |
| \`DATABASE_POOL_SIZE\` | int | No | 20 | Connection pool size |
| \`DATABASE_MAX_OVERFLOW\` | int | No | 10 | Max overflow connections |
| \`DATABASE_POOL_TIMEOUT\` | int | No | 30 | Pool timeout (seconds) |
| \`DATABASE_ECHO\` | bool | No | false | Echo SQL queries |

### Redis

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`REDIS_URL\` | string | Yes | - | Redis connection URL |
| \`REDIS_POOL_SIZE\` | int | No | 10 | Connection pool size |
| \`REDIS_TIMEOUT\` | int | No | 5 | Connection timeout (seconds) |
| \`REDIS_KEY_PREFIX\` | string | No | "scamshield:" | Key prefix for namespacing |

### Authentication

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`JWT_SECRET\` | string | Yes | - | JWT signing secret (min 32 chars) |
| \`JWT_ALGORITHM\` | string | No | "HS256" | JWT algorithm |
| \`JWT_EXPIRE_MINUTES\` | int | No | 60 | Access token expiry |
| \`REFRESH_TOKEN_EXPIRE_DAYS\` | int | No | 30 | Refresh token expiry |
| \`API_KEY_PREFIX\` | string | No | "sk_" | API key prefix |

### Model Registry

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`MODEL_REGISTRY_PATH\` | string | Yes | - | S3/GCS path for model artifacts |
| \`MODEL_CACHE_DIR\` | string | No | "/tmp/models" | Local model cache directory |
| \`ENSEMBLE_WEIGHTS\` | JSON | No | See below | Ensemble model weights |
| \`UNCERTAINTY_THRESHOLD\` | float | No | 0.15 | Uncertainty threshold for review |

Default Ensemble Weights:
\`\`\`json
{
  "xgboost": 0.30,
  "lightgbm": 0.25,
  "transformer": 0.25,
  "gnn": 0.20
}
\`\`\`

### Feature Store

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`FEATURE_STORE_TTL\` | int | No | 3600 | Feature cache TTL (seconds) |
| \`FEATURE_COMPUTE_TIMEOUT\` | int | No | 100 | Feature computation timeout (ms) |
| \`VELOCITY_WINDOWS\` | JSON | No | See below | Velocity window configs |

Default Velocity Windows:
\`\`\`json
{
  "1m": 60,
  "5m": 300,
  "1h": 3600,
  "24h": 86400
}
\`\`\`

### Decision Engine

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`DECISION_THRESHOLDS\` | JSON | No | See below | Decision thresholds |
| \`HIGH_VALUE_THRESHOLD\` | float | No | 50000 | High value transaction threshold |
| \`VELOCITY_THRESHOLDS\` | JSON | No | See below | Velocity thresholds |
| \`ENABLE_THREAT_INTEL\` | bool | No | true | Enable threat intelligence |

Default Decision Thresholds:
\`\`\`json
{
  "approve": 0.3,
  "review": 0.7,
  "decline": 0.7
}
\`\`\`

Default Velocity Thresholds:
\`\`\`json
{
  "5m": 20,
  "1h": 100,
  "24h": 500
}
\`\`\`

### Monitoring

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`PROMETHEUS_PORT\` | int | No | 9090 | Prometheus metrics port |
| \`OTEL_EXPORTER_OTLP_ENDPOINT\` | string | No | - | OTLP endpoint for tracing |
| \`OTEL_SERVICE_NAME\` | string | No | "scamshield-api" | Service name for tracing |
| \`JAEGER_AGENT_HOST\` | string | No | "localhost" | Jaeger agent host |
| \`JAEGER_AGENT_PORT\` | int | No | 6831 | Jaeger agent port |

### External Services

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| \`THREAT_INTEL_API_KEY\` | string | No | - | Threat intelligence API key |
| \`THREAT_INTEL_API_URL\` | string | No | - | Threat intelligence API URL |
| \`SLACK_WEBHOOK_URL\` | string | No | - | Slack webhook for alerts |
| \`PAGERDUTY_INTEGRATION_KEY\` | string | No | - | PagerDuty integration key |
| \`TWILIO_ACCOUNT_SID\` | string | No | - | Twilio account SID |
| \`TWILIO_AUTH_TOKEN\` | string | No | - | Twilio auth token |
| \`TWILIO_FROM_NUMBER\` | string | No | - | Twilio from number |

## Configuration Files

### config/features.yaml

\`\`\`yaml
# Feature computation configuration
behavioral:
  windows: [7d, 30d]
  aggregations: [mean, count, std, min, max]
  merchant_categories: true

device:
  trust_factors:
    - is_rooted: -0.3
    - is_emulator: -0.4
    - has_vpn: -0.2
    - os_version_age: -0.1
  model_risk_enabled: true

network:
  ip_reputation_enabled: true
  asn_reputation_enabled: true
  tor_detection: true
  proxy_detection: true

velocity:
  windows: [1m, 5m, 1h, 24h]
  burst_threshold: 10
  amount_velocity: true

graph:
  pagerank_enabled: true
  community_detection: true
  max_hops: 3
\`\`\`

### config/rules.yaml

\`\`\`yaml
# Rule engine configuration
rules:
  - id: "VEL_001"
    name: "High Velocity 5m"
    condition: "velocity_5m > 20"
    score: 0.3
    action: "review"

  - id: "VEL_002"
    name: "High Velocity 1h"
    condition: "velocity_1h > 100"
    score: 0.4
    action: "review"

  - id: "AMT_001"
    name: "High Value Transaction"
    condition: "amount > 50000 AND amount_zscore > 3"
    score: 0.25
    action: "review"

  - id: "DEV_001"
    name: "Rooted Device"
    condition: "is_rooted == true"
    score: 0.4
    action: "decline"

  - id: "NET_001"
    name: "Tor/Proxy Detected"
    condition: "is_tor == true OR is_proxy == true"
    score: 0.35
    action: "review"

  - id: "GRF_001"
    name: "High Risk Graph Cluster"
    condition: "pagerank_score > 0.8 AND community_risk > 0.7"
    score: 0.3
    action: "review"

rule_weights:
  velocity: 1.0
  amount: 1.0
  device: 1.2
  network: 1.1
  graph: 0.9
\`\`\`

### config/model_thresholds.yaml

\`\`\`yaml
# Model evaluation thresholds
minimum_thresholds:
  accuracy: 0.95
  precision: 0.92
  recall: 0.90
  f1: 0.91
  auc: 0.97
  latency_p99_ms: 20

production_thresholds:
  accuracy: 0.96
  precision: 0.93
  recall: 0.91
  f1: 0.92
  auc: 0.98
  latency_p99_ms: 15

drift_thresholds:
  psi_threshold: 0.1
  ks_threshold: 0.05
  feature_drift_threshold: 0.1
\`\`\`
`,
      },
      {
        id: 'error-codes',
        title: 'Error Codes',
        description: 'Complete error code reference',
        content: `# Error Codes

## HTTP Status Codes

| Code | Name | Description |
|------|------|-------------|
| 200 | OK | Success |
| 201 | Created | Resource created |
| 204 | No Content | Success, no response body |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Application Error Codes

### Authentication Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`AUTH_INVALID_CREDENTIALS\` | 401 | Invalid API key or credentials | Verify API key |
| \`AUTH_TOKEN_EXPIRED\` | 401 | Authentication token has expired | Refresh token |
| \`AUTH_TOKEN_INVALID\` | 401 | Invalid or malformed token | Re-authenticate |
| \`AUTH_INSUFFICIENT_SCOPE\` | 403 | Token lacks required scope | Request appropriate scope |
| \`AUTH_ACCOUNT_DISABLED\` | 403 | Account has been disabled | Contact support |
| \`AUTH_RATE_LIMITED\` | 429 | Too many authentication attempts | Wait and retry |

### Validation Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`VALIDATION_ERROR\` | 422 | Request validation failed | Check request format |
| \`VALIDATION_MISSING_FIELD\` | 422 | Required field missing | Add required field |
| \`VALIDATION_INVALID_FORMAT\` | 422 | Field format invalid | Correct format |
| \`VALIDATION_OUT_OF_RANGE\` | 422 | Value out of allowed range | Use valid range |
| \`VALIDATION_DUPLICATE\` | 409 | Duplicate resource | Use unique identifier |

### Scoring Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`SCORING_FEATURE_STORE_UNAVAILABLE\` | 503 | Feature store temporarily unavailable | Retry with backoff |
| \`SCORING_MODEL_UNAVAILABLE\` | 503 | ML model temporarily unavailable | Retry with backoff |
| \`SCORING_TIMEOUT\` | 504 | Scoring request timed out | Retry with longer timeout |
| \`SCORING_INVALID_TRANSACTION\` | 422 | Transaction data invalid | Fix transaction data |
| \`SCORING_FEATURE_COMPUTE_FAILED\` | 500 | Feature computation failed | Contact support |

### Model Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`MODEL_NOT_FOUND\` | 404 | Model not found | Verify model ID |
| \`MODEL_VERSION_NOT_FOUND\` | 404 | Model version not found | Verify version |
| \`MODEL_DEPLOYMENT_FAILED\` | 500 | Model deployment failed | Check deployment logs |
| \`MODEL_PROMOTION_FAILED\` | 500 | Model promotion failed | Check target environment |
| \`MODEL_ARTIFACT_MISSING\` | 500 | Model artifact not found | Re-upload artifact |

### Alert Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`ALERT_NOT_FOUND\` | 404 | Alert not found | Verify alert ID |
| \`ALERT_ALREADY_ACKNOWLEDGED\` | 409 | Alert already acknowledged | No action needed |
| \`ALERT_ALREADY_RESOLVED\` | 409 | Alert already resolved | No action needed |
| \`ALERT_RULE_NOT_FOUND\` | 404 | Alert rule not found | Verify rule ID |
| \`ALERT_CHANNEL_TEST_FAILED\` | 500 | Notification channel test failed | Check channel config |

### Case Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`CASE_NOT_FOUND\` | 404 | Case not found | Verify case ID |
| \`CASE_ALREADY_CLOSED\` | 409 | Case already closed | No action needed |
| \`CASE_INVALID_TRANSITION\` | 422 | Invalid status transition | Use valid transition |

### Rate Limiting Errors

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|------------|
| \`RATE_LIMIT_EXCEEDED\` | 429 | Rate limit exceeded | Implement backoff |
| \`RATE_LIMIT_QUOTA_EXHAUSTED\` | 429 | Monthly quota exhausted | Upgrade plan |

## Error Response Format

\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field_errors": [
        {
          "field": "amount",
          "code": "VALIDATION_OUT_OF_RANGE",
          "message": "Amount must be greater than 0"
        }
      ]
    },
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
\`\`\`

## Retry Guidelines

| Error Category | Retry | Backoff | Max Retries |
|----------------|-------|---------|-------------|
| 5xx Server Errors | Yes | Exponential (1s, 2s, 4s, 8s...) | 3 |
| 429 Rate Limited | Yes | Respect \`Retry-After\` header | 5 |
| Network Errors | Yes | Exponential | 3 |
| 4xx Client Errors | No | - | 0 |
| \`SCORING_TIMEOUT\` | Yes | Exponential | 2 |

## Client SDK Error Handling

\`\`\`python
from scamshield import ScamShieldClient
from scamshield.exceptions import (
    AuthenticationError,
    ValidationError,
    ScoringError,
    RateLimitError,
    ServerError
)

client = ScamShieldClient(api_key="sk_live_...")

try:
    result = client.score_transaction(transaction)
except AuthenticationError as e:
    # Invalid API key
    logger.error(f"Auth failed: {e}")
except ValidationError as e:
    # Invalid request data
    logger.error(f"Validation failed: {e.details}")
except ScoringError as e:
    # Scoring service issue
    if e.retryable:
        # Retry with backoff
        pass
except RateLimitError as e:
    # Rate limited
    wait_time = e.retry_after or 60
    time.sleep(wait_time)
except ServerError as e:
    # 5xx error
    logger.error(f"Server error: {e}")
\`\`\`
`,
      },
      {
        id: 'changelog',
        title: 'Changelog',
        description: 'Version history and release notes',
        content: `# Changelog

## v3.1.0 (2024-01-15) - Current Production

### 🚀 New Features
- **Ensemble v3.1.0**: Updated weights (XGBoost: 0.30, LightGBM: 0.25, Transformer: 0.25, GNN: 0.20)
- **Transformer v1.2.0**: Improved attention mechanism, 15% accuracy boost
- **Real-time Feature Store**: Redis-backed with 99.9% hit rate
- **Graph Neural Network**: Entity resolution for synthetic identity detection
- **Explainable AI**: SHAP-based feature attribution in scoring response

### 🐛 Bug Fixes
- Fixed feature store connection leak under high load
- Resolved race condition in velocity computation
- Fixed model version rollback not clearing cache
- Corrected threat intelligence cache invalidation

### ⚡ Performance
- P99 latency reduced from 35ms to 23ms
- Throughput increased to 15,000 req/s
- Memory usage reduced by 30%
- Cold start time < 2s

### 🔧 Operations
- Added OpenTelemetry distributed tracing
- New Prometheus alerts for model drift
- Improved deployment rollback automation
- Enhanced runbook documentation

---

## v3.0.2 (2023-12-25)

### 🚀 New Features
- **Ensemble v3.0.2**: Production ensemble with weighted voting
- **Model Registry**: Full versioning, promotion, rollback
- **Alert Rules Engine**: Configurable rules with multi-channel notifications
- **Case Management**: Investigation workflow with evidence tracking

### 🐛 Bug Fixes
- Fixed batch scoring memory leak
- Resolved WebSocket connection drops
- Corrected fraud rate calculation in dashboard

### ⚡ Performance
- P99 latency: 25ms
- Throughput: 10,000 req/s

---

## v2.3.1 (2023-11-15)

### 🚀 New Features
- **XGBoost v2.3.1**: 500 estimators, max_depth=8
- **LightGBM v1.8.2**: Native categorical support
- **Feature Store v2**: Online/offline separation
- **Velocity Features**: 1m, 5m, 1h, 24h windows

### 🐛 Bug Fixes
- Fixed feature computation for new users
- Resolved device fingerprinting edge cases

---

## v2.2.5 (2023-10-15)

### 🚀 New Features
- **XGBoost v2.2.5**: Initial production model
- **Real-time Scoring API**: Sub-50ms target
- **Basic Alerting**: Slack + Email notifications
- **Docker Deployment**: Containerized services

### 🐛 Bug Fixes
- Fixed database connection pooling
- Resolved async task cancellation issues

---

## v1.0.0 (2023-08-01) - Initial Release

### 🚀 New Features
- **Core Architecture**: FastAPI + SQLAlchemy + Pydantic
- **Basic Fraud Detection**: Rule-based engine
- **PostgreSQL + Redis**: Data layer
- **Prometheus Monitoring**: Basic metrics
- **GitHub Actions CI/CD**: Automated testing

---

## Upcoming Releases

### v3.2.0 (Planned Q1 2024)
- **AutoML Pipeline**: Automated model retraining
- **Feature Store v3**: Feature lineage & validation
- **Advanced Threat Intel**: Real-time threat feeds
- **Multi-region Deployment**: Active-active setup
- **GraphQL API**: Flexible query interface

### v4.0.0 (Planned Q2 2024)
- **LLM-powered Investigation**: Natural language case analysis
- **Federated Learning**: Privacy-preserving model training
- **Real-time Model Updates**: Online learning capability
- **Advanced Visualization**: 3D graph exploration
`,
      },
    ],
  },
]

const pageContents: Record<string, string> = {
  introduction: docSections[0].children[0].content,
  installation: docSections[0].children[1].content,
  architecture: docSections[0].children[2].content,
  authentication: docSections[1].children[0].content,
  'scoring-api': docSections[1].children[1].content,
  'models-api': docSections[1].children[2].content,
  'alerts-api': docSections[1].children[3].content,
  'cases-api': docSections[1].children[4].content,
  'feature-engineering': docSections[2].children[0].content,
  'model-training': docSections[2].children[1].content,
  'deployment-guide': docSections[2].children[2].content,
  monitoring: docSections[3].children[0].content,
  'incident-response': docSections[3].children[1].content,
  configuration: docSections[4].children[0].content,
  'error-codes': docSections[4].children[1].content,
  changelog: docSections[4].children[2].content,
}

function renderMarkdown(content: string) {
  // Simple markdown rendering for demo
  return content
    .split('\n')
    .map(line => {
      if (line.startsWith('### ')) return <h3 className="text-lg font-semibold text-gray-900 mt-6 mb-3">{line.slice(4)}</h3>
      if (line.startsWith('## ')) return <h2 className="text-xl font-bold text-gray-900 mt-8 mb-4 pb-2 border-b border-gray-200">{line.slice(3)}</h2>
      if (line.startsWith('# ')) return <h1 className="text-2xl font-bold text-gray-900 mt-8 mb-4">{line.slice(2)}</h1>
      if (line.startsWith('|') && line.endsWith('|')) return <div className="overflow-x-auto my-4"><table className="table"><tbody><tr>{line.split('|').slice(1, -1).map((cell, i) => <td key={i} className="px-3 py-2 text-sm">{cell.trim()}</td>)}</tr></tbody></table></div>
      if (line.startsWith('```')) return <div className="my-4" />
      if (line.startsWith('`') && line.endsWith('`')) return <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono">{line.slice(1, -1)}</code>
      if (line.startsWith('- ') || line.startsWith('* ')) return <li className="ml-4 mb-1">{line.slice(2)}</li>
      if (line.match(/^\d+\. /)) return <li className="ml-4 mb-1">{line}</li>
      if (line.trim() === '') return <br key="br" />
      return <p className="text-gray-700 mb-3">{line}</p>
    })
}

export default function DocsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [selectedPage, setSelectedPage] = useState<string>('introduction')
  const [searchQuery, setSearchQuery] = useState('')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const filteredSections = docSections.map(section => ({
    ...section,
    children: section.children.filter(page =>
      page.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      page.description.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })).filter(section => section.children.length > 0)

  const currentContent = pageContents[selectedPage] || pageContents.introduction

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar onCollapseChange={setSidebarCollapsed} collapsed={sidebarCollapsed} />
      <Header sidebarCollapsed={sidebarCollapsed} />

      <main className={cn('pt-16 transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-64')}>
        <div className="flex flex-col lg:flex-row h-[calc(100vh-4rem)]">
          {/* Sidebar Navigation */}
          <aside className={cn('w-72 lg:w-80 bg-white border-r border-gray-200 flex flex-col hidden lg:flex', mobileMenuOpen && 'fixed inset-y-0 left-0 z-50')}>
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">Documentation</h2>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="lg:hidden p-2 text-gray-500 hover:text-gray-700"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="p-4 border-b border-gray-200">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search docs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input pl-10"
                />
              </div>
            </div>

            <nav className="flex-1 overflow-y-auto p-4 space-y-4">
              {filteredSections.map((section) => (
                <div key={section.id}>
                  <h3 className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-2">
                    {section.icon}
                    {section.title}
                  </h3>
                  <ul className="space-y-1">
                    {section.children.map((page) => (
                      <li key={page.id}>
                        <button
                          onClick={() => { setSelectedPage(page.id); setMobileMenuOpen(false) }}
                          className={cn(
                            'w-full text-left px-3 py-2 rounded-lg text-sm transition-colors',
                            selectedPage === page.id
                              ? 'bg-primary-50 text-primary-700 font-medium'
                              : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                          )}
                        >
                          <div className="font-medium">{page.title}</div>
                          <div className="text-xs text-gray-500">{page.description}</div>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </nav>

            <div className="p-4 border-t border-gray-200">
              <a
                href="https://github.com/your-org/scamshield-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-gray-600 hover:text-primary-600"
              >
                <Github className="h-4 w-4" />
                View on GitHub
              </a>
            </div>
          </aside>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(true)}
            className="lg:hidden fixed bottom-4 right-4 z-40 p-3 bg-primary-600 text-white rounded-full shadow-lg"
          >
            <Menu className="h-6 w-6" />
          </button>

          {/* Overlay for mobile */}
          {mobileMenuOpen && (
            <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setMobileMenuOpen(false)} />
          )}

          {/* Content Area */}
          <article className="flex-1 overflow-y-auto p-6 lg:p-8 max-w-4xl mx-auto w-full">
            <div className="prose prose-gray max-w-none">
              <div className="mb-8">
                <nav className="flex items-center gap-2 text-sm text-gray-500 mb-4">
                  <span>Documentation</span>
                  <ChevronRight className="h-4 w-4" />
                  <span className="capitalize">{selectedPage.replace(/-/g, ' ')}</span>
                </nav>
                <h1 className="text-3xl font-bold text-gray-900">
                  {docSections.flatMap(s => s.children).find(p => p.id === selectedPage)?.title || 'Documentation'}
                </h1>
                <p className="text-gray-500 mt-2">
                  {docSections.flatMap(s => s.children).find(p => p.id === selectedPage)?.description}
                </p>
              </div>

              <div className="space-y-4">
                {renderMarkdown(currentContent)}
              </div>
            </div>
          </article>
        </div>
      </main>
    </div>
  )
}