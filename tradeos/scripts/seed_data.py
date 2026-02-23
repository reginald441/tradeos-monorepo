#!/usr/bin/env python3
# =============================================================================
# TradeOS Database Seeding Script
# Populates database with test data for development and testing
# =============================================================================

import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://tradeos:tradeos@localhost:5432/tradeos")

# =============================================================================
# SEED DATA
# =============================================================================
USERS_DATA = [
    {
        "id": str(uuid4()),
        "email": "admin@tradeos.io",
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "first_name": "System",
        "last_name": "Administrator",
        "role": "admin",
        "is_active": True,
        "is_verified": True,
    },
    {
        "id": str(uuid4()),
        "email": "trader@tradeos.io",
        "username": "trader",
        "password_hash": hashlib.sha256("trader123".encode()).hexdigest(),
        "first_name": "John",
        "last_name": "Trader",
        "role": "trader",
        "is_active": True,
        "is_verified": True,
    },
    {
        "id": str(uuid4()),
        "email": "analyst@tradeos.io",
        "username": "analyst",
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "first_name": "Jane",
        "last_name": "Analyst",
        "role": "analyst",
        "is_active": True,
        "is_verified": True,
    },
]

ACCOUNTS_DATA = [
    {
        "id": str(uuid4()),
        "name": "Primary Trading Account",
        "broker": "Interactive Brokers",
        "account_type": "margin",
        "currency": "USD",
        "balance": Decimal("100000.00"),
        "buying_power": Decimal("200000.00"),
    },
    {
        "id": str(uuid4()),
        "name": "Retirement Account",
        "broker": "Fidelity",
        "account_type": "ira",
        "currency": "USD",
        "balance": Decimal("50000.00"),
        "buying_power": Decimal("50000.00"),
    },
]

PORTFOLIOS_DATA = [
    {
        "id": str(uuid4()),
        "name": "Growth Portfolio",
        "description": "High growth stock portfolio",
        "strategy": "growth",
        "risk_level": "high",
    },
    {
        "id": str(uuid4()),
        "name": "Dividend Portfolio",
        "description": "Dividend income portfolio",
        "strategy": "income",
        "risk_level": "medium",
    },
    {
        "id": str(uuid4()),
        "name": "Index Fund Portfolio",
        "description": "Low cost index fund portfolio",
        "strategy": "passive",
        "risk_level": "low",
    },
]

ASSETS_DATA = [
    {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "asset_type": "stock", "exchange": "NASDAQ"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "exchange": "NYSE"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "asset_type": "etf", "exchange": "NASDAQ"},
    {"symbol": "BTC", "name": "Bitcoin", "asset_type": "crypto", "exchange": "BINANCE"},
    {"symbol": "ETH", "name": "Ethereum", "asset_type": "crypto", "exchange": "BINANCE"},
]

TRADES_DATA = [
    {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": Decimal("100"),
        "price": Decimal("175.50"),
        "asset_type": "stock",
    },
    {
        "symbol": "MSFT",
        "side": "buy",
        "quantity": Decimal("50"),
        "price": Decimal("380.25"),
        "asset_type": "stock",
    },
    {
        "symbol": "TSLA",
        "side": "sell",
        "quantity": Decimal("25"),
        "price": Decimal("240.00"),
        "asset_type": "stock",
    },
    {
        "symbol": "BTC",
        "side": "buy",
        "quantity": Decimal("0.5"),
        "price": Decimal("65000.00"),
        "asset_type": "crypto",
    },
]

# =============================================================================
# SEEDING FUNCTIONS
# =============================================================================
async def seed_users(session: AsyncSession):
    """Seed users table."""
    logger.info("Seeding users...")
    
    for user_data in USERS_DATA:
        # Check if user exists
        result = await session.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": user_data["email"]}
        )
        if result.scalar():
            logger.info(f"User {user_data['email']} already exists, skipping")
            continue
        
        await session.execute(text("""
            INSERT INTO users (id, email, username, password_hash, first_name, last_name, 
                             role, is_active, is_verified, created_at, updated_at)
            VALUES (:id, :email, :username, :password_hash, :first_name, :last_name,
                    :role, :is_active, :is_verified, NOW(), NOW())
        """), user_data)
    
    await session.commit()
    logger.info(f"Seeded {len(USERS_DATA)} users")


async def seed_accounts(session: AsyncSession, user_id: str):
    """Seed accounts table."""
    logger.info("Seeding accounts...")
    
    for account_data in ACCOUNTS_DATA:
        # Check if account exists
        result = await session.execute(
            text("SELECT id FROM accounts WHERE name = :name"),
            {"name": account_data["name"]}
        )
        if result.scalar():
            logger.info(f"Account {account_data['name']} already exists, skipping")
            continue
        
        account_data["user_id"] = user_id
        account_data["created_at"] = datetime.utcnow()
        account_data["updated_at"] = datetime.utcnow()
        
        await session.execute(text("""
            INSERT INTO accounts (id, user_id, name, broker, account_type, currency, 
                                balance, buying_power, created_at, updated_at)
            VALUES (:id, :user_id, :name, :broker, :account_type, :currency,
                    :balance, :buying_power, :created_at, :updated_at)
        """), account_data)
    
    await session.commit()
    logger.info(f"Seeded {len(ACCOUNTS_DATA)} accounts")


async def seed_portfolios(session: AsyncSession, user_id: str):
    """Seed portfolios table."""
    logger.info("Seeding portfolios...")
    
    for portfolio_data in PORTFOLIOS_DATA:
        # Check if portfolio exists
        result = await session.execute(
            text("SELECT id FROM portfolios WHERE name = :name"),
            {"name": portfolio_data["name"]}
        )
        if result.scalar():
            logger.info(f"Portfolio {portfolio_data['name']} already exists, skipping")
            continue
        
        portfolio_data["user_id"] = user_id
        portfolio_data["created_at"] = datetime.utcnow()
        portfolio_data["updated_at"] = datetime.utcnow()
        
        await session.execute(text("""
            INSERT INTO portfolios (id, user_id, name, description, strategy, risk_level,
                                  created_at, updated_at)
            VALUES (:id, :user_id, :name, :description, :strategy, :risk_level,
                    :created_at, :updated_at)
        """), portfolio_data)
    
    await session.commit()
    logger.info(f"Seeded {len(PORTFOLIOS_DATA)} portfolios")


async def seed_assets(session: AsyncSession):
    """Seed assets table."""
    logger.info("Seeding assets...")
    
    for asset_data in ASSETS_DATA:
        # Check if asset exists
        result = await session.execute(
            text("SELECT id FROM assets WHERE symbol = :symbol"),
            {"symbol": asset_data["symbol"]}
        )
        if result.scalar():
            logger.info(f"Asset {asset_data['symbol']} already exists, skipping")
            continue
        
        asset_data["id"] = str(uuid4())
        asset_data["created_at"] = datetime.utcnow()
        asset_data["updated_at"] = datetime.utcnow()
        
        await session.execute(text("""
            INSERT INTO assets (id, symbol, name, asset_type, exchange, created_at, updated_at)
            VALUES (:id, :symbol, :name, :asset_type, :exchange, :created_at, :updated_at)
        """), asset_data)
    
    await session.commit()
    logger.info(f"Seeded {len(ASSETS_DATA)} assets")


async def seed_trades(session: AsyncSession, user_id: str, account_id: str):
    """Seed trades table."""
    logger.info("Seeding trades...")
    
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i, trade_data in enumerate(TRADES_DATA):
        trade_data["id"] = str(uuid4())
        trade_data["user_id"] = user_id
        trade_data["account_id"] = account_id
        trade_data["status"] = "filled"
        trade_data["created_at"] = base_time + timedelta(days=i * 2)
        trade_data["updated_at"] = trade_data["created_at"]
        trade_data["executed_at"] = trade_data["created_at"]
        trade_data["total_amount"] = trade_data["quantity"] * trade_data["price"]
        trade_data["commission"] = Decimal("5.00")
        
        await session.execute(text("""
            INSERT INTO trades (id, user_id, account_id, symbol, side, quantity, price,
                              asset_type, status, total_amount, commission, 
                              created_at, updated_at, executed_at)
            VALUES (:id, :user_id, :account_id, :symbol, :side, :quantity, :price,
                    :asset_type, :status, :total_amount, :commission,
                    :created_at, :updated_at, :executed_at)
        """), trade_data)
    
    await session.commit()
    logger.info(f"Seeded {len(TRADES_DATA)} trades")


async def seed_market_data(session: AsyncSession):
    """Seed market data for charts."""
    logger.info("Seeding market data...")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]
    base_prices = {"AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0, "SPY": 450.0}
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    for symbol in symbols:
        current_price = base_prices[symbol]
        current_time = start_time
        
        while current_time < end_time:
            # Generate OHLC data with random variation
            import random
            variation = random.uniform(-0.02, 0.02)
            open_price = current_price * (1 + variation)
            high_price = open_price * (1 + abs(random.uniform(0, 0.01)))
            low_price = open_price * (1 - abs(random.uniform(0, 0.01)))
            close_price = (high_price + low_price) / 2
            volume = random.randint(1000000, 10000000)
            
            await session.execute(text("""
                INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                ON CONFLICT (symbol, timestamp) DO NOTHING
            """), {
                "symbol": symbol,
                "timestamp": current_time,
                "open": Decimal(str(open_price)),
                "high": Decimal(str(high_price)),
                "low": Decimal(str(low_price)),
                "close": Decimal(str(close_price)),
                "volume": volume
            })
            
            current_price = close_price
            current_time += timedelta(hours=1)
    
    await session.commit()
    logger.info(f"Seeded market data for {len(symbols)} symbols")


async def seed_database():
    """Main seeding function."""
    logger.info("=" * 60)
    logger.info("TradeOS Database Seeding")
    logger.info("=" * 60)
    
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            # Seed users
            await seed_users(session)
            
            # Get first user ID for relationships
            result = await session.execute(text("SELECT id FROM users WHERE username = 'admin' LIMIT 1"))
            admin_user_id = result.scalar()
            
            if admin_user_id:
                # Seed accounts
                await seed_accounts(session, admin_user_id)
                
                # Get first account ID
                result = await session.execute(text("SELECT id FROM accounts LIMIT 1"))
                account_id = result.scalar()
                
                # Seed portfolios
                await seed_portfolios(session, admin_user_id)
                
                # Seed trades if account exists
                if account_id:
                    await seed_trades(session, admin_user_id, account_id)
            
            # Seed assets
            await seed_assets(session)
            
            # Seed market data
            await seed_market_data(session)
            
            logger.info("=" * 60)
            logger.info("Database seeding completed successfully!")
            logger.info("=" * 60)
            logger.info("\nTest Accounts:")
            logger.info("  admin@tradeos.io / admin123")
            logger.info("  trader@tradeos.io / trader123")
            logger.info("  analyst@tradeos.io / analyst123")
            
        except Exception as e:
            logger.error(f"Seeding failed: {e}")
            await session.rollback()
            raise
    
    await engine.dispose()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(seed_database())
    except KeyboardInterrupt:
        logger.info("Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        sys.exit(1)
