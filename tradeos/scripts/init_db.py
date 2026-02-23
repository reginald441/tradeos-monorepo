#!/usr/bin/env python3
# =============================================================================
# TradeOS Database Initialization Script
# Creates database, extensions, and initial schema
# =============================================================================

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://tradeos:tradeos@localhost:5432/tradeos")
DATABASE_NAME = os.getenv("POSTGRES_DB", "tradeos")
POSTGRES_USER = os.getenv("POSTGRES_USER", "tradeos")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "tradeos")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# =============================================================================
# SQL SCRIPTS
# =============================================================================
CREATE_EXTENSIONS_SQL = """
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Verify extensions
SELECT * FROM pg_extension WHERE extname IN ('uuid-ossp', 'pgcrypto', 'timescaledb');
"""

CREATE_SCHEMA_SQL = """
-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set schema comments
COMMENT ON SCHEMA trading IS 'Trading-related tables and functions';
COMMENT ON SCHEMA analytics IS 'Analytics and reporting tables';
COMMENT ON SCHEMA audit IS 'Audit logging tables';
"""

CREATE_ROLES_SQL = """
-- Create application roles
DO $$
BEGIN
    -- Read-only role for reporting
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'tradeos_readonly') THEN
        CREATE ROLE tradeos_readonly NOLOGIN;
    END IF;
    
    -- Application role
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'tradeos_app') THEN
        CREATE ROLE tradeos_app NOLOGIN;
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA trading, analytics, audit TO tradeos_app;
GRANT USAGE ON SCHEMA analytics TO tradeos_readonly;
"""

CREATE_FUNCTIONS_SQL = """
-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create audit logging function
CREATE OR REPLACE FUNCTION audit.log_change()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit.change_log (
        table_name,
        operation,
        old_data,
        new_data,
        changed_by,
        changed_at
    ) VALUES (
        TG_TABLE_NAME,
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        current_user,
        CURRENT_TIMESTAMP
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
"""

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================
async def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    logger.info(f"Checking if database '{DATABASE_NAME}' exists...")
    
    # Connect to default postgres database
    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=int(POSTGRES_PORT),
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database="postgres"
    )
    
    try:
        # Check if database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            DATABASE_NAME
        )
        
        if not result:
            logger.info(f"Creating database '{DATABASE_NAME}'...")
            await conn.execute(f'CREATE DATABASE "{DATABASE_NAME}"')
            logger.info(f"Database '{DATABASE_NAME}' created successfully")
        else:
            logger.info(f"Database '{DATABASE_NAME}' already exists")
    finally:
        await conn.close()


async def execute_sql_statements(engine, sql_statements: str, description: str):
    """Execute a block of SQL statements."""
    logger.info(f"Executing: {description}...")
    
    async with engine.connect() as conn:
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql_statements.split(';') if s.strip()]
        
        for statement in statements:
            if statement and not statement.startswith('--'):
                try:
                    await conn.execute(text(statement))
                    await conn.commit()
                except Exception as e:
                    # Ignore errors for "already exists" cases
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Statement failed (may already exist): {e}")
        
        logger.info(f"{description} completed")


async def init_timescaledb():
    """Initialize TimescaleDB hypertables."""
    logger.info("Setting up TimescaleDB hypertables...")
    
    engine = create_async_engine(DATABASE_URL)
    
    hypertable_sql = """
    -- Create hypertables for time-series data
    -- These will be created after the main tables are set up by migrations
    
    -- Example: Convert trades table to hypertable if it exists
    DO $$
    BEGIN
        IF EXISTS (SELECT FROM information_schema.tables 
                   WHERE table_schema = 'trading' AND table_name = 'trades') THEN
            PERFORM create_hypertable('trading.trades', 'created_at', 
                if_not_exists => TRUE,
                chunk_time_interval => INTERVAL '1 day');
        END IF;
        
        IF EXISTS (SELECT FROM information_schema.tables 
                   WHERE table_schema = 'trading' AND table_name = 'market_data') THEN
            PERFORM create_hypertable('trading.market_data', 'timestamp', 
                if_not_exists => TRUE,
                chunk_time_interval => INTERVAL '1 hour');
        END IF;
    END
    $$;
    """
    
    try:
        async with engine.connect() as conn:
            await conn.execute(text(hypertable_sql))
            await conn.commit()
        logger.info("TimescaleDB hypertables configured")
    except Exception as e:
        logger.warning(f"TimescaleDB setup (may already be configured): {e}")
    finally:
        await engine.dispose()


async def run_migrations():
    """Run Alembic migrations."""
    logger.info("Running database migrations...")
    
    import subprocess
    
    backend_dir = Path(__file__).parent.parent / "backend"
    
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=backend_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info("Migrations completed successfully")
    else:
        logger.error(f"Migration failed: {result.stderr}")
        raise RuntimeError("Database migration failed")


async def init_database():
    """Initialize the complete database setup."""
    logger.info("=" * 60)
    logger.info("TradeOS Database Initialization")
    logger.info("=" * 60)
    
    # Step 1: Create database if not exists
    await create_database_if_not_exists()
    
    # Step 2: Create engine for the target database
    engine = create_async_engine(DATABASE_URL)
    
    try:
        # Step 3: Create extensions
        await execute_sql_statements(engine, CREATE_EXTENSIONS_SQL, "Creating extensions")
        
        # Step 4: Create schemas
        await execute_sql_statements(engine, CREATE_SCHEMA_SQL, "Creating schemas")
        
        # Step 5: Create roles
        await execute_sql_statements(engine, CREATE_ROLES_SQL, "Creating roles")
        
        # Step 6: Create functions
        await execute_sql_statements(engine, CREATE_FUNCTIONS_SQL, "Creating functions")
        
    finally:
        await engine.dispose()
    
    # Step 7: Run migrations
    await run_migrations()
    
    # Step 8: Setup TimescaleDB
    await init_timescaledb()
    
    logger.info("=" * 60)
    logger.info("Database initialization completed successfully!")
    logger.info("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(init_database())
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
