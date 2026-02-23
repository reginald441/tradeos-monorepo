"""
TradeOS Main Application
FastAPI application with all routers, middleware, and configurations.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from database.connection import close_database, init_database
from middleware.logging import (
    ErrorLoggingMiddleware,
    RequestLoggingMiddleware,
    StructuredLoggingMiddleware,
)
from middleware.rate_limit import (
    AUTH_RATE_LIMIT,
    MARKET_DATA_RATE_LIMIT,
    TRADE_RATE_LIMIT,
    RateLimitMiddleware,
)

# Configure structured logging
def setup_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        format=settings.logging.format,
        level=getattr(logging, settings.logging.level.upper()),
        stream=sys.stdout,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.json_format else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Setup logging on module load
setup_logging()
logger = structlog.get_logger("tradeos.app")


# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan events.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting TradeOS",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    logger.info("TradeOS started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TradeOS")
    
    try:
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))
    
    logger.info("TradeOS shutdown complete")


# ============================================================================
# Exception Handlers
# ============================================================================

async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    """Handle validation exceptions."""
    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=exc.errors() if hasattr(exc, "errors") else str(exc),
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
        },
    )


async def http_exception_handler(request: Request, exc) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "detail": exc.detail,
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.exception(
        "Unhandled exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred" if not settings.debug else str(exc),
        },
    )


# ============================================================================
# Health Check Router
# ============================================================================

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns:
        dict: Health status information.
    """
    from database.connection import check_database_health
    
    db_health = await check_database_health()
    
    return {
        "status": "healthy" if db_health["connected"] else "unhealthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": db_health,
    }


@health_router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> dict:
    """
    Readiness check endpoint for Kubernetes.
    
    Returns:
        dict: Readiness status.
    """
    return {
        "ready": True,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
    }


@health_router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> dict:
    """
    Liveness check endpoint for Kubernetes.
    
    Returns:
        dict: Liveness status.
    """
    return {
        "alive": True,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
    }


# ============================================================================
# API Version Router
# ============================================================================

api_v1_router = APIRouter(prefix="/api/v1")


# Import actual routers from routers module
from routers import (
    auth_router,
    trading_router,
    strategies_router,
    risk_router,
    backtest_router,
    market_router,
    user_router,
    admin_router,
    billing_router,
)

# Include all routers in API v1
api_v1_router.include_router(auth_router)
api_v1_router.include_router(user_router)
api_v1_router.include_router(trading_router)
api_v1_router.include_router(strategies_router)
api_v1_router.include_router(risk_router)
api_v1_router.include_router(backtest_router)
api_v1_router.include_router(market_router)
api_v1_router.include_router(billing_router)
api_v1_router.include_router(admin_router)


# ============================================================================
# Application Factory
# ============================================================================

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application.
    """
    app = FastAPI(
        title=settings.app_name,
        description="TradeOS - Advanced Algorithmic Trading Platform",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # ==========================================================================
    # Middleware (order matters - first added = first executed)
    # ==========================================================================
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Structured logging context
    app.add_middleware(StructuredLoggingMiddleware)
    
    # Request/response logging
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths=["/health", "/ready", "/live", "/metrics", "/docs", "/redoc", "/openapi.json"],
        log_request_body=settings.debug,
        log_response_body=False,
    )
    
    # Error logging
    app.add_middleware(ErrorLoggingMiddleware, include_traceback=settings.debug)
    
    # Rate limiting
    path_configs = {
        "/api/v1/auth": AUTH_RATE_LIMIT,
        "/api/v1/trades": TRADE_RATE_LIMIT,
        "/api/v1/orders": TRADE_RATE_LIMIT,
        "/api/v1/market": MARKET_DATA_RATE_LIMIT,
    }
    app.add_middleware(
        RateLimitMiddleware,
        path_configs=path_configs,
        skip_paths=["/health", "/ready", "/live", "/docs", "/redoc", "/openapi.json"],
    )
    
    # ==========================================================================
    # Exception Handlers
    # ==========================================================================
    
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # ==========================================================================
    # Routers
    # ==========================================================================
    
    # Health check routes (no prefix)
    app.include_router(health_router)
    
    # API v1 routes
    app.include_router(api_v1_router)
    
    # ==========================================================================
    # Root Endpoint
    # ==========================================================================
    
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """
        Root endpoint.
        
        Returns:
            dict: Application information.
        """
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Advanced Algorithmic Trading Platform",
            "documentation": "/docs" if settings.debug else None,
            "health": "/health",
            "api": "/api/v1",
        }
    
    return app


# ============================================================================
# Create Application Instance
# ============================================================================

app = create_application()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.logging.level.lower(),
    )
