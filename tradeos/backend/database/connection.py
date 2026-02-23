"""
TradeOS Database Connection Management
SQLAlchemy 2.0 async database engine and session management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()

# Global engine and session factory (initialized on first use)
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.
    
    Returns:
        AsyncEngine: The SQLAlchemy async engine instance.
    """
    global _engine
    
    if _engine is None:
        _engine = create_async_engine(
            settings.database.async_url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
            future=True,
        )
        
        # Add event listeners for connection monitoring
        @event.listens_for(_engine.sync_engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("New database connection established")
        
        @event.listens_for(_engine.sync_engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Database connection checked out from pool")
        
        logger.info("Database engine initialized")
    
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session maker.
    
    Returns:
        async_sessionmaker: The session maker configured with the engine.
    """
    global _async_session_maker
    
    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Prevent expired object errors
            autoflush=False,         # Manual flush control
            autocommit=False,        # Explicit transaction control
        )
        logger.info("Session maker initialized")
    
    return _async_session_maker


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions in FastAPI endpoints.
    
    Yields:
        AsyncSession: A database session that is automatically closed.
    
    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Use this for background tasks or when you need manual session control.
    
    Example:
        async with get_db_context() as db:
            result = await db.execute(query)
    
    Yields:
        AsyncSession: A database session.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database context error: {e}")
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called during application startup.
    """
    engine = get_engine()
    
    try:
        # Import all models to ensure they're registered with Base
        from database import models  # noqa: F401
        
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database() -> None:
    """
    Close the database engine and cleanup resources.
    
    This should be called during application shutdown.
    """
    global _engine, _async_session_maker
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("Database engine disposed")


async def check_database_health() -> dict:
    """
    Check database connectivity and return health status.
    
    Returns:
        dict: Health check results with status and details.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            start_time = __import__('time').time()
            result = await conn.execute(text("SELECT 1"))
            await result.scalar()
            latency_ms = (__import__('time').time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "connected": True,
                "latency_ms": round(latency_ms, 2),
                "pool_size": settings.database.pool_size,
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }


async def execute_raw_query(query: str, params: Optional[dict] = None) -> list:
    """
    Execute a raw SQL query and return results.
    
    Args:
        query: The SQL query string.
        params: Optional query parameters.
    
    Returns:
        list: Query results as a list of dictionaries.
    """
    async with get_db_context() as session:
        result = await session.execute(text(query), params or {})
        rows = result.mappings().all()
        return [dict(row) for row in rows]


class DatabaseTransaction:
    """
    Context manager for database transactions with savepoint support.
    
    Example:
        async with DatabaseTransaction() as tx:
            tx.session.add(entity)
            # Automatically commits on exit, rolls back on exception
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self._provided_session = session
        self.session: Optional[AsyncSession] = None
        self._own_session = session is None
    
    async def __aenter__(self) -> "DatabaseTransaction":
        if self._provided_session:
            self.session = self._provided_session
        else:
            session_maker = get_session_maker()
            self.session = session_maker()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.session:
            return
        
        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            if self._own_session:
                await self.session.close()


# Convenience exports
__all__ = [
    "Base",
    "get_engine",
    "get_session_maker",
    "get_db_session",
    "get_db_context",
    "init_database",
    "close_database",
    "check_database_health",
    "execute_raw_query",
    "DatabaseTransaction",
]
