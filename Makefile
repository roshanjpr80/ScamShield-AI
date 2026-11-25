# ScamShield AI - Makefile

.PHONY: help install dev prod test lint format clean docker docker-up docker-down migrate reset-db status logs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt
	pre-commit install

install-dev: ## Install dev dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

install-prod: ## Install production dependencies
	pip install -r requirements-api.txt
	pip install gunicorn uvicorn[standard]

dev: ## Run API in development mode
	python -m uvicorn scamshield.api.main:app --reload --port 8000 --log-level debug

dev-full: ## Run API + Streamlit in development
	make dev & streamlit run scamshield/frontend/streamlit_app.py --server.port 8501

prod: ## Run API in production mode
	gunicorn scamshield.api.main:app \
		-w 4 \
		-k uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--access-logfile - \
		--error-logfile -

test: ## Run tests
	pytest tests/ -v --tb=short

test-all: ## Run all tests with coverage
	pytest tests/ -v --cov=scamshield --cov-report=html --cov-report=term-missing

test-watch: ## Run tests on file change
	pytest-watch -- -- pytest tests/ -v --tb=short

lint: ## Run linters
	ruff check scamshield/ tests/
	mypy scamshield/
	flake8 scamshield/ tests/

format: ## Format code
	black scamshield/ tests/
	ruff check --fix scamshield/ tests/

check: lint format ## Run all checks
	@echo "All checks passed!"

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov/ dist/ build/ *.egg-info
	rm -rf models/* data/processed/* data/generated/*
	@echo "Cleanup complete."

# Docker commands
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## View service logs
	docker-compose logs -f api

docker-restart: ## Restart all services
	docker-compose restart

docker-clean: ## Remove all containers and volumes
	docker-compose down -v --remove-orphans

# Database migrations
migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration
	alembic revision --autogenerate -m "$(message)"

migrate-status: ## Check migration status
	alembic history

reset-db: ## Reset database (WARNING: deletes all data)
	alembic downgrade base && alembic upgrade head
	@echo "Database reset complete."

# ML Model management
train-model: ## Train a model
	python -m scamshield.mlops.training_pipeline train --type text_detector --version v1.0.0

check-drift: ## Check for model drift
	python -m scamshield.mlops.drift_detection check --model-type text_detector --version v1.0.0

deploy-model: ## Deploy a model version
	python -m scamshield.api.routes.models deploy --model-type text_detector --version v1.0.0

# Monitoring
monitor: ## Open monitoring dashboard
	open http://localhost:9090

grafana: ## Open Grafana dashboard
	open http://localhost:3000

mlflow: ## Open MLflow tracking UI
	open http://localhost:5000

# Status commands
status: ## Show system status
	@echo "=== ScamShield AI Status ==="
	@curl -s http://localhost:8000/health 2>/dev/null | jq . || echo "API not running"
	@echo ""
	@docker ps --filter "name=scamshield" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/source docs/build/html

docs-serve: ## Serve documentation locally
	sphinx-autobuild docs/source docs/build/html

# Utility
seed-data: ## Seed initial data
	python -m scamshield.data.fraud_data_generator seed

generate-test-data: ## Generate test dataset
	python -m scamshield.data.fraud_data_generator generate --count 10000 --output ./data/test_dataset.csv

setup-hooks: ## Setup git hooks
	pre-commit install
	git config core.hooksPath .hooks

upgrade-deps: ## Upgrade all dependencies
	pip list --outdated --format=json | jq -r '.[].name' | xargs pip install -U

update-submodules: ## Update submodules
	git submodule update --init --recursive

run-examples: ## Run example scripts
	python examples/text_detection_demo.py
	python examples/transaction_detection_demo.py
