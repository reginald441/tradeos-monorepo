# =============================================================================
# TradeOS Makefile
# Common commands for development and deployment
# =============================================================================

.PHONY: help build up down logs migrate shell test clean deploy

# =============================================================================
# DEFAULT TARGET
# =============================================================================
.DEFAULT_GOAL := help

# =============================================================================
# COLORS
# =============================================================================
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "$(BLUE)TradeOS - Available Commands$(NC)"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# BUILD COMMANDS
# =============================================================================
build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build --parallel

build-no-cache: ## Build all Docker images without cache
	@echo "$(BLUE)Building Docker images (no cache)...$(NC)"
	docker-compose build --no-cache --parallel

build-backend: ## Build backend Docker image
	@echo "$(BLUE)Building backend image...$(NC)"
	docker-compose build backend

build-frontend: ## Build frontend Docker image
	@echo "$(BLUE)Building frontend image...$(NC)"
	docker-compose build frontend

# =============================================================================
# RUN COMMANDS
# =============================================================================
up: ## Start all services in detached mode
	@echo "$(GREEN)Starting TradeOS services...$(NC)"
	docker-compose up -d

up-logs: ## Start all services and attach to logs
	@echo "$(GREEN)Starting TradeOS services with logs...$(NC)"
	docker-compose up

up-prod: ## Start services in production mode
	@echo "$(GREEN)Starting TradeOS in production mode...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

down: ## Stop all services
	@echo "$(YELLOW)Stopping TradeOS services...$(NC)"
	docker-compose down

down-volumes: ## Stop all services and remove volumes
	@echo "$(RED)Stopping services and removing volumes...$(NC)"
	docker-compose down -v

restart: down up ## Restart all services

restart-backend: ## Restart backend service
	@echo "$(YELLOW)Restarting backend...$(NC)"
	docker-compose restart backend

restart-frontend: ## Restart frontend service
	@echo "$(YELLOW)Restarting frontend...$(NC)"
	docker-compose restart frontend

# =============================================================================
# LOG COMMANDS
# =============================================================================
logs: ## View logs from all services
	docker-compose logs -f

logs-backend: ## View backend logs
	docker-compose logs -f backend

logs-frontend: ## View frontend logs
	docker-compose logs -f frontend

logs-nginx: ## View nginx logs
	docker-compose logs -f nginx

logs-db: ## View database logs
	docker-compose logs -f postgres

logs-redis: ## View redis logs
	docker-compose logs -f redis

# =============================================================================
# DATABASE COMMANDS
# =============================================================================
db-init: ## Initialize database (create extensions, schemas, run migrations)
	@echo "$(BLUE)Initializing database...$(NC)"
	docker-compose exec backend python scripts/init_db.py

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose exec backend alembic upgrade head

db-migrate-down: ## Rollback last migration
	@echo "$(YELLOW)Rolling back last migration...$(NC)"
	docker-compose exec backend alembic downgrade -1

db-migrate-create: ## Create new migration (usage: make db-migrate-create msg="description")
	@echo "$(BLUE)Creating new migration...$(NC)"
	docker-compose exec backend alembic revision --autogenerate -m "$(msg)"

db-seed: ## Seed database with test data
	@echo "$(BLUE)Seeding database...$(NC)"
	docker-compose exec backend python scripts/seed_data.py

db-reset: ## Reset database (drop and recreate)
	@echo "$(RED)Resetting database...$(NC)"
	docker-compose down -v
	docker-compose up -d postgres
	@sleep 5
	$(MAKE) db-init
	$(MAKE) db-seed

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U tradeos -d tradeos

db-backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(NC)"
	@mkdir -p backups
	docker-compose exec postgres pg_dump -U tradeos tradeos > backups/tradeos_$$(date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (usage: make db-restore file=backups/tradeos_xxx.sql)
	@echo "$(BLUE)Restoring database from $(file)...$(NC)"
	docker-compose exec -T postgres psql -U tradeos -d tradeos < $(file)

# =============================================================================
# SHELL COMMANDS
# =============================================================================
shell-backend: ## Open shell in backend container
	docker-compose exec backend /bin/sh

shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

shell-db: ## Open shell in database container
	docker-compose exec postgres /bin/sh

shell-redis: ## Open redis CLI
	docker-compose exec redis redis-cli

# =============================================================================
# TEST COMMANDS
# =============================================================================
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	docker-compose exec backend pytest

test-backend: ## Run backend tests
	@echo "$(BLUE)Running backend tests...$(NC)"
	docker-compose exec backend pytest app/tests

test-frontend: ## Run frontend tests
	@echo "$(BLUE)Running frontend tests...$(NC)"
	docker-compose exec frontend npm test

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	docker-compose exec backend pytest --cov=app --cov-report=html

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running E2E tests...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit

# =============================================================================
# LINT COMMANDS
# =============================================================================
lint: lint-backend lint-frontend ## Run all linters

lint-backend: ## Run backend linters
	@echo "$(BLUE)Linting backend...$(NC)"
	docker-compose exec backend ruff check app
	docker-compose exec backend black --check app
	docker-compose exec backend mypy app

lint-frontend: ## Run frontend linters
	@echo "$(BLUE)Linting frontend...$(NC)"
	docker-compose exec frontend npm run lint

format: ## Format all code
	@echo "$(BLUE)Formatting code...$(NC)"
	docker-compose exec backend black app
	docker-compose exec backend ruff check --fix app
	docker-compose exec frontend npm run format

# =============================================================================
# CLEANUP COMMANDS
# =============================================================================
clean: ## Remove stopped containers and unused images
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker system prune -f

clean-all: ## Remove all containers, images, and volumes
	@echo "$(RED)Removing all Docker resources...$(NC)"
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -af --volumes

clean-logs: ## Clean log files
	@echo "$(YELLOW)Cleaning log files...$(NC)"
	docker-compose exec backend sh -c 'rm -f /app/logs/*.log'

# =============================================================================
# DEPLOYMENT COMMANDS
# =============================================================================
deploy: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(NC)"
	./scripts/deploy.sh

deploy-staging: ## Deploy to staging
	@echo "$(GREEN)Deploying to staging...$(NC)"
	./scripts/deploy.sh staging

# =============================================================================
# MONITORING COMMANDS
# =============================================================================
monitor: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

metrics: ## Show Prometheus metrics
	@curl -s http://localhost:9090/api/v1/targets | jq

# =============================================================================
# UTILITY COMMANDS
# =============================================================================
ps: ## Show running containers
	docker-compose ps

status: ## Show service status
	@echo "$(BLUE)TradeOS Service Status$(NC)"
	@echo "======================="
	@docker-compose ps
	@echo ""
	@echo "$(BLUE)Resource Usage$(NC)"
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

health: ## Check service health
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost/health || echo "$(RED)Health check failed$(NC)"
	@curl -s http://localhost:8000/health || echo "$(RED)Backend health check failed$(NC)"

update: ## Update all images and restart
	@echo "$(BLUE)Updating images...$(NC)"
	docker-compose pull
	docker-compose up -d

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================
dev-setup: ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	cp .env.example .env
	@echo "$(YELLOW)Please edit .env file with your configuration$(NC)"
	$(MAKE) build
	$(MAKE) up
	@sleep 10
	$(MAKE) db-init
	$(MAKE) db-seed

dev-logs: ## Development mode with logs
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# =============================================================================
# BACKUP COMMANDS
# =============================================================================
backup: ## Create full backup
	@echo "$(BLUE)Creating full backup...$(NC)"
	@mkdir -p backups
	$(MAKE) db-backup
	docker run --rm -v tradeos_postgres_data:/data -v $$(pwd)/backups:/backup alpine tar czf /backup/postgres_data_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# =============================================================================
# CERTIFICATE COMMANDS
# =============================================================================
certs-generate: ## Generate self-signed SSL certificates
	@echo "$(BLUE)Generating SSL certificates...$(NC)"
	@mkdir -p nginx/ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/ssl/tradeos.key \
		-out nginx/ssl/tradeos.crt \
		-subj "/C=US/ST=State/L=City/O=TradeOS/CN=localhost"

certs-letsencrypt: ## Obtain Let's Encrypt certificates
	@echo "$(BLUE)Obtaining Let's Encrypt certificates...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.letsencrypt.yml up certbot
