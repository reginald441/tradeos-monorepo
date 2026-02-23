#!/bin/bash
# =============================================================================
# TradeOS Production Deployment Script
# Handles deployment to production and staging environments
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="tradeos"
ENVIRONMENT="${1:-production}"
BACKUP_BEFORE_DEPLOY=true
HEALTH_CHECK_TIMEOUT=120

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed"
        exit 1
    fi
}

load_env() {
    if [ -f "$PROJECT_DIR/.env" ]; then
        log_info "Loading environment variables..."
        set -a
        source "$PROJECT_DIR/.env"
        set +a
    else
        log_warning ".env file not found, using defaults"
    fi
}

# =============================================================================
# PRE-DEPLOYMENT CHECKS
# =============================================================================
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check required commands
    check_command docker
    check_command docker-compose
    check_command curl
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_error ".env file not found. Please create one from .env.example"
        exit 1
    fi
    
    # Check SSL certificates for production
    if [ "$ENVIRONMENT" = "production" ]; then
        if [ ! -f "$PROJECT_DIR/nginx/ssl/tradeos.crt" ] || [ ! -f "$PROJECT_DIR/nginx/ssl/tradeos.key" ]; then
            log_warning "SSL certificates not found. Generating self-signed certificates..."
            mkdir -p "$PROJECT_DIR/nginx/ssl"
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout "$PROJECT_DIR/nginx/ssl/tradeos.key" \
                -out "$PROJECT_DIR/nginx/ssl/tradeos.crt" \
                -subj "/C=US/ST=State/L=City/O=TradeOS/CN=localhost"
        fi
    fi
    
    log_success "Pre-deployment checks passed"
}

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================
create_backup() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        log_info "Creating pre-deployment backup..."
        
        BACKUP_DIR="$PROJECT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup database
        if docker-compose ps | grep -q "postgres"; then
            log_info "Backing up database..."
            docker-compose exec -T postgres pg_dump -U "${POSTGRES_USER:-tradeos}" "${POSTGRES_DB:-tradeos}" > "$BACKUP_DIR/database.sql" || {
                log_warning "Database backup failed, continuing..."
            }
        fi
        
        # Backup environment file
        cp "$PROJECT_DIR/.env" "$BACKUP_DIR/.env"
        
        log_success "Backup created at $BACKUP_DIR"
    fi
}

# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================
pull_latest_images() {
    log_info "Pulling latest images..."
    docker-compose pull
}

build_images() {
    log_info "Building Docker images..."
    docker-compose build --parallel --no-cache
}

stop_services() {
    log_info "Stopping existing services..."
    docker-compose down --remove-orphans
}

start_services() {
    log_info "Starting services..."
    docker-compose up -d
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    log_info "Waiting for database..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-tradeos}" &> /dev/null; then
            break
        fi
        sleep 2
    done
    
    # Run migrations
    docker-compose exec -T backend alembic upgrade head || {
        log_warning "Migration failed, attempting to continue..."
    }
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================
health_check() {
    log_info "Performing health checks..."
    
    local start_time=$(date +%s)
    local services=("nginx" "backend" "postgres" "redis")
    local all_healthy=true
    
    for service in "${services[@]}"; do
        log_info "Checking $service..."
        
        local healthy=false
        local attempts=0
        local max_attempts=30
        
        while [ $attempts -lt $max_attempts ]; do
            if docker-compose ps "$service" | grep -q "healthy"; then
                healthy=true
                break
            fi
            
            attempts=$((attempts + 1))
            sleep 2
        done
        
        if [ "$healthy" = false ]; then
            log_error "$service is not healthy"
            all_healthy=false
        else
            log_success "$service is healthy"
        fi
    done
    
    # Check application endpoints
    log_info "Checking application endpoints..."
    
    # Wait for services to be ready
    sleep 5
    
    # Check main endpoint
    if curl -sf http://localhost/health &> /dev/null; then
        log_success "Main endpoint is responding"
    else
        log_error "Main endpoint is not responding"
        all_healthy=false
    fi
    
    # Check API endpoint
    if curl -sf http://localhost:8000/health &> /dev/null; then
        log_success "API endpoint is responding"
    else
        log_error "API endpoint is not responding"
        all_healthy=false
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$all_healthy" = true ]; then
        log_success "All health checks passed in ${duration}s"
        return 0
    else
        log_error "Some health checks failed after ${duration}s"
        return 1
    fi
}

# =============================================================================
# ROLLBACK FUNCTION
# =============================================================================
rollback() {
    log_error "Deployment failed! Initiating rollback..."
    
    # Stop current services
    docker-compose down
    
    # Restore from backup if available
    if [ -d "$PROJECT_DIR/backups" ]; then
        local latest_backup=$(ls -t "$PROJECT_DIR/backups" | head -1)
        if [ -n "$latest_backup" ]; then
            log_info "Restoring from backup: $latest_backup"
            # Restore database
            if [ -f "$PROJECT_DIR/backups/$latest_backup/database.sql" ]; then
                docker-compose up -d postgres
                sleep 5
                docker-compose exec -T postgres psql -U "${POSTGRES_USER:-tradeos}" "${POSTGRES_DB:-tradeos}" < "$PROJECT_DIR/backups/$latest_backup/database.sql"
            fi
        fi
    fi
    
    # Restart with previous images
    docker-compose up -d
    
    log_warning "Rollback completed"
}

# =============================================================================
# POST-DEPLOYMENT
# =============================================================================
post_deployment() {
    log_info "Running post-deployment tasks..."
    
    # Clean up old images
    log_info "Cleaning up old Docker images..."
    docker image prune -af --filter "until=168h" &> /dev/null || true
    
    # Clean up old backups (keep last 10)
    if [ -d "$PROJECT_DIR/backups" ]; then
        log_info "Cleaning up old backups..."
        ls -t "$PROJECT_DIR/backups" | tail -n +11 | xargs -r rm -rf
    fi
    
    # Show deployment summary
    echo ""
    log_success "Deployment completed successfully!"
    echo ""
    echo "========================================"
    echo "Service URLs:"
    echo "  - Application: https://localhost"
    echo "  - API: http://localhost:8000"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo "========================================"
}

# =============================================================================
# MONITORING SETUP
# =============================================================================
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Grafana datasources directory
    mkdir -p "$PROJECT_DIR/monitoring/grafana/datasources"
    
    # Create Prometheus datasource config
    cat > "$PROJECT_DIR/monitoring/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF
    
    log_success "Monitoring configured"
}

# =============================================================================
# MAIN DEPLOYMENT FLOW
# =============================================================================
main() {
    echo "========================================"
    echo "  TradeOS Deployment Script"
    echo "  Environment: $ENVIRONMENT"
    echo "========================================"
    echo ""
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Load environment variables
    load_env
    
    # Pre-deployment checks
    pre_deployment_checks
    
    # Create backup
    create_backup
    
    # Setup monitoring
    setup_monitoring
    
    # Stop existing services
    stop_services
    
    # Build and start services
    if [ "$ENVIRONMENT" = "production" ]; then
        pull_latest_images
    else
        build_images
    fi
    
    # Start services
    start_services
    
    # Run migrations
    run_migrations
    
    # Health checks
    if ! health_check; then
        rollback
        exit 1
    fi
    
    # Post-deployment tasks
    post_deployment
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
