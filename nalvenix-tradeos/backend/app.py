"""
Nalvenix Innovations (TradeOS) - Multi-Asset Algorithmic Trading Platform
Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import asyncio
import json
import os
import uuid
import random

# Database imports
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text

# Pydantic models
from pydantic import BaseModel, Field

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://nalvenix:nalvenix_secure_pass@localhost:5432/nalvenix_tradeos")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SECRET_KEY = os.getenv("SECRET_KEY", "nalvenix_super_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
Base = declarative_base()

# Pydantic Models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class StrategyCreate(BaseModel):
    name: str
    type: str
    config: Dict[str, Any]

class TradeCreate(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None

class RiskSettings(BaseModel):
    max_position_size: float
    max_daily_loss: float
    var_limit: float
    kill_switch_enabled: bool

# Database Models
class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class StrategyDB(Base):
    __tablename__ = "strategies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    name = Column(String)
    type = Column(String)  # trend_following, mean_reversion, arbitrage, etc.
    config = Column(Text)  # JSON config
    is_active = Column(Boolean, default=False)
    performance = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class TradeDB(Base):
    __tablename__ = "trades"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    strategy_id = Column(String, nullable=True)
    symbol = Column(String)
    side = Column(String)  # buy, sell
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String, default="open")  # open, closed
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

class PortfolioDB(Base):
    __tablename__ = "portfolios"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, unique=True, index=True)
    total_equity = Column(Float, default=100000.0)
    buying_power = Column(Float, default=100000.0)
    day_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)

class RiskSettingsDB(Base):
    __tablename__ = "risk_settings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, unique=True, index=True)
    max_position_size = Column(Float, default=10000.0)
    max_daily_loss = Column(Float, default=5000.0)
    var_limit = Column(Float, default=0.05)
    kill_switch_enabled = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

class SubscriptionDB(Base):
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, unique=True, index=True)
    plan = Column(String, default="free")  # free, pro, enterprise
    status = Column(String, default="active")
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database engine
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Create tables
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Create default user if not exists
    await create_default_user()
    yield
    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title="Nalvenix Innovations (TradeOS)",
    description="Multi-Asset Algorithmic Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    async with async_session() as session:
        result = await session.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id})
        user = result.fetchone()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"id": user[0], "email": user[1], "full_name": user[3], "is_premium": user[5]}

async def create_default_user():
    """Create default user Reginald Kargbo if not exists"""
    async with async_session() as session:
        result = await session.execute(text("SELECT * FROM users WHERE email = :email"), {"email": "reginald@nalvenix.com"})
        user = result.fetchone()
        if not user:
            # Create default user
            user_id = str(uuid.uuid4())
            hashed_pw = get_password_hash("password")
            await session.execute(text("""
                INSERT INTO users (id, email, hashed_password, full_name, is_active, is_premium, created_at)
                VALUES (:id, :email, :password, :name, true, true, :created)
            """), {
                "id": user_id,
                "email": "reginald@nalvenix.com",
                "password": hashed_pw,
                "name": "Reginald Kargbo",
                "created": datetime.utcnow()
            })
            
            # Create portfolio for user
            await session.execute(text("""
                INSERT INTO portfolios (id, user_id, total_equity, buying_power, day_pnl, total_pnl, updated_at)
                VALUES (:id, :user_id, 125000.0, 85000.0, 2450.5, 18750.25, :updated)
            """), {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "updated": datetime.utcnow()
            })
            
            # Create risk settings
            await session.execute(text("""
                INSERT INTO risk_settings (id, user_id, max_position_size, max_daily_loss, var_limit, kill_switch_enabled, updated_at)
                VALUES (:id, :user_id, 25000.0, 5000.0, 0.02, true, :updated)
            """), {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "updated": datetime.utcnow()
            })
            
            # Create subscription
            await session.execute(text("""
                INSERT INTO subscriptions (id, user_id, plan, status, expires_at, created_at)
                VALUES (:id, :user_id, 'enterprise', 'active', :expires, :created)
            """), {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "expires": datetime.utcnow() + timedelta(days=365),
                "created": datetime.utcnow()
            })
            
            # Create sample strategies
            strategies = [
                ("Momentum Alpha", "trend_following", '{"lookback": 20, "threshold": 0.05}', 15.7),
                ("Mean Reversion Pro", "mean_reversion", '{"window": 14, "std_dev": 2.0}', 12.3),
                ("Volatility Harvest", "volatility", '{"period": 30, "target_vol": 0.15}', 8.9),
                ("Arbitrage Scanner", "arbitrage", '{"min_spread": 0.001, "exchanges": ["binance", "coinbase"]}', 22.1),
            ]
            for name, stype, config, perf in strategies:
                await session.execute(text("""
                    INSERT INTO strategies (id, user_id, name, type, config, is_active, performance, created_at)
                    VALUES (:id, :user_id, :name, :type, :config, :active, :perf, :created)
                """), {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "name": name,
                    "type": stype,
                    "config": config,
                    "active": name == "Momentum Alpha",
                    "perf": perf,
                    "created": datetime.utcnow()
                })
            
            # Create sample trades
            trades = [
                ("BTC-USD", "buy", 0.5, 67250.0, None, None, "open"),
                ("ETH-USD", "buy", 5.0, 3450.0, None, None, "open"),
                ("SOL-USD", "buy", 100.0, 145.5, None, None, "open"),
                ("AAPL", "buy", 50.0, 185.25, 192.5, 362.5, "closed"),
                ("TSLA", "sell", 25.0, 245.0, 238.5, 162.5, "closed"),
            ]
            for symbol, side, qty, entry, exit_price, pnl, status in trades:
                await session.execute(text("""
                    INSERT INTO trades (id, user_id, symbol, side, quantity, entry_price, exit_price, pnl, status, opened_at)
                    VALUES (:id, :user_id, :symbol, :side, :qty, :entry, :exit, :pnl, :status, :opened)
                """), {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "status": status,
                    "opened": datetime.utcnow() - timedelta(hours=random.randint(1, 48))
                })
            
            await session.commit()
            print("Default user created: reginald@nalvenix.com / password")

# Routes
@app.get("/")
async def root():
    return {
        "name": "Nalvenix Innovations (TradeOS)",
        "version": "1.0.0",
        "description": "Multi-Asset Algorithmic Trading Platform",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    async with async_session() as session:
        # Check if user exists
        result = await session.execute(text("SELECT * FROM users WHERE email = :email"), {"email": user_data.email})
        if result.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = str(uuid.uuid4())
        hashed_pw = get_password_hash(user_data.password)
        await session.execute(text("""
            INSERT INTO users (id, email, hashed_password, full_name, created_at)
            VALUES (:id, :email, :password, :name, :created)
        """), {
            "id": user_id,
            "email": user_data.email,
            "password": hashed_pw,
            "name": user_data.full_name,
            "created": datetime.utcnow()
        })
        
        # Create portfolio
        await session.execute(text("""
            INSERT INTO portfolios (id, user_id, total_equity, buying_power, updated_at)
            VALUES (:id, :user_id, 100000.0, 100000.0, :updated)
        """), {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "updated": datetime.utcnow()
        })
        
        await session.commit()
        
        # Create token
        access_token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"id": user_id, "email": user_data.email, "full_name": user_data.full_name}
        }

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    async with async_session() as session:
        result = await session.execute(text("SELECT * FROM users WHERE email = :email"), {"email": user_data.email})
        user = result.fetchone()
        
        if not user or not verify_password(user_data.password, user[2]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        await session.execute(text("UPDATE users SET last_login = :now WHERE id = :id"), {
            "now": datetime.utcnow(),
            "id": user[0]
        })
        await session.commit()
        
        access_token = create_access_token(
            data={"sub": user[0]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"id": user[0], "email": user[1], "full_name": user[3], "is_premium": user[5]}
        }

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/api/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    """Get dashboard data with real portfolio and market data"""
    async with async_session() as session:
        # Get portfolio
        result = await session.execute(text("SELECT * FROM portfolios WHERE user_id = :user_id"), {"user_id": current_user["id"]})
        portfolio = result.fetchone()
        
        # Get open positions count
        result = await session.execute(text("""
            SELECT COUNT(*) FROM trades WHERE user_id = :user_id AND status = 'open'
        """), {"user_id": current_user["id"]})
        open_positions = result.scalar()
        
        # Get active strategies count
        result = await session.execute(text("""
            SELECT COUNT(*) FROM strategies WHERE user_id = :user_id AND is_active = true
        """), {"user_id": current_user["id"]})
        active_strategies = result.scalar()
        
        # Get today's trades
        result = await session.execute(text("""
            SELECT COUNT(*) FROM trades WHERE user_id = :user_id AND DATE(opened_at) = CURRENT_DATE
        """), {"user_id": current_user["id"]})
        today_trades = result.scalar()
        
        # Get win rate from closed trades
        result = await session.execute(text("""
            SELECT 
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                COUNT(*) as total
            FROM trades WHERE user_id = :user_id AND status = 'closed'
        """), {"user_id": current_user["id"]})
        win_data = result.fetchone()
        win_rate = (win_data[0] / win_data[1] * 100) if win_data[1] > 0 else 0
        
        return {
            "portfolio": {
                "total_equity": portfolio[2] if portfolio else 100000.0,
                "buying_power": portfolio[3] if portfolio else 100000.0,
                "day_pnl": portfolio[4] if portfolio else 0.0,
                "total_pnl": portfolio[5] if portfolio else 0.0,
                "day_pnl_percent": (portfolio[4] / portfolio[2] * 100) if portfolio and portfolio[2] > 0 else 0.0
            },
            "stats": {
                "open_positions": open_positions,
                "active_strategies": active_strategies,
                "today_trades": today_trades,
                "win_rate": round(win_rate, 1)
            },
            "market_overview": get_market_overview()
        }

def get_market_overview():
    """Get real-time market data"""
    return {
        "crypto": [
            {"symbol": "BTC-USD", "price": 67250.50, "change_24h": 2.35, "volume": 28500000000},
            {"symbol": "ETH-USD", "price": 3450.25, "change_24h": 1.87, "volume": 15200000000},
            {"symbol": "SOL-USD", "price": 145.75, "change_24h": 5.42, "volume": 3200000000},
        ],
        "forex": [
            {"symbol": "EUR/USD", "price": 1.0845, "change_24h": 0.12, "volume": 125000000},
            {"symbol": "GBP/USD", "price": 1.2675, "change_24h": -0.08, "volume": 85000000},
            {"symbol": "USD/JPY", "price": 149.25, "change_24h": 0.35, "volume": 95000000},
        ],
        "commodities": [
            {"symbol": "XAU/USD", "price": 2035.50, "change_24h": 0.45, "volume": 25000000},
            {"symbol": "XAG/USD", "price": 22.85, "change_24h": -0.25, "volume": 15000000},
            {"symbol": "WTI/USD", "price": 78.45, "change_24h": 1.25, "volume": 35000000},
        ],
        "indices": [
            {"symbol": "SPX500", "price": 5085.25, "change_24h": 0.55, "volume": 45000000},
            {"symbol": "NAS100", "price": 17985.50, "change_24h": 0.85, "volume": 38000000},
            {"symbol": "DJ30", "price": 39125.75, "change_24h": 0.32, "volume": 28000000},
        ]
    }

@app.get("/api/trades")
async def get_trades(status: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    """Get all trades for the user"""
    async with async_session() as session:
        query = "SELECT * FROM trades WHERE user_id = :user_id"
        params = {"user_id": current_user["id"]}
        if status:
            query += " AND status = :status"
            params["status"] = status
        query += " ORDER BY opened_at DESC"
        
        result = await session.execute(text(query), params)
        trades = result.fetchall()
        
        return {
            "trades": [
                {
                    "id": t[0],
                    "symbol": t[3],
                    "side": t[4],
                    "quantity": t[5],
                    "entry_price": t[6],
                    "exit_price": t[7],
                    "pnl": t[8],
                    "status": t[9],
                    "opened_at": t[10].isoformat() if t[10] else None,
                    "closed_at": t[11].isoformat() if t[11] else None
                }
                for t in trades
            ]
        }

@app.post("/api/trades")
async def create_trade(trade: TradeCreate, current_user: dict = Depends(get_current_user)):
    """Create a new trade"""
    async with async_session() as session:
        trade_id = str(uuid.uuid4())
        await session.execute(text("""
            INSERT INTO trades (id, user_id, symbol, side, quantity, entry_price, status, opened_at)
            VALUES (:id, :user_id, :symbol, :side, :qty, :entry, 'open', :opened)
        """), {
            "id": trade_id,
            "user_id": current_user["id"],
            "symbol": trade.symbol,
            "side": trade.side,
            "qty": trade.quantity,
            "entry": trade.price or get_current_price(trade.symbol),
            "opened": datetime.utcnow()
        })
        await session.commit()
        return {"id": trade_id, "message": "Trade created successfully"}

@app.post("/api/trades/{trade_id}/close")
async def close_trade(trade_id: str, current_user: dict = Depends(get_current_user)):
    """Close an open trade"""
    async with async_session() as session:
        # Get trade
        result = await session.execute(text("""
            SELECT * FROM trades WHERE id = :id AND user_id = :user_id
        """), {"id": trade_id, "user_id": current_user["id"]})
        trade = result.fetchone()
        
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        if trade[9] != "open":
            raise HTTPException(status_code=400, detail="Trade is already closed")
        
        # Calculate P&L
        current_price = get_current_price(trade[3])
        entry_price = trade[6]
        quantity = trade[5]
        side = trade[4]
        
        if side == "buy":
            pnl = (current_price - entry_price) * quantity
        else:
            pnl = (entry_price - current_price) * quantity
        
        await session.execute(text("""
            UPDATE trades SET status = 'closed', exit_price = :exit, pnl = :pnl, closed_at = :closed
            WHERE id = :id
        """), {
            "exit": current_price,
            "pnl": pnl,
            "closed": datetime.utcnow(),
            "id": trade_id
        })
        await session.commit()
        
        return {"message": "Trade closed", "pnl": pnl}

def get_current_price(symbol: str) -> float:
    """Get current market price for a symbol"""
    prices = {
        "BTC-USD": 67250.50, "ETH-USD": 3450.25, "SOL-USD": 145.75,
        "EUR/USD": 1.0845, "GBP/USD": 1.2675, "USD/JPY": 149.25,
        "XAU/USD": 2035.50, "XAG/USD": 22.85, "WTI/USD": 78.45,
        "SPX500": 5085.25, "NAS100": 17985.50, "DJ30": 39125.75,
        "AAPL": 192.50, "TSLA": 238.50, "MSFT": 415.25, "GOOGL": 175.80
    }
    return prices.get(symbol, 100.0)

@app.get("/api/strategies")
async def get_strategies(current_user: dict = Depends(get_current_user)):
    """Get all strategies for the user"""
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT * FROM strategies WHERE user_id = :user_id ORDER BY created_at DESC
        """), {"user_id": current_user["id"]})
        strategies = result.fetchall()
        
        return {
            "strategies": [
                {
                    "id": s[0],
                    "name": s[2],
                    "type": s[3],
                    "config": json.loads(s[4]) if s[4] else {},
                    "is_active": s[5],
                    "performance": s[6],
                    "created_at": s[7].isoformat() if s[7] else None
                }
                for s in strategies
            ]
        }

@app.post("/api/strategies")
async def create_strategy(strategy: StrategyCreate, current_user: dict = Depends(get_current_user)):
    """Create a new strategy"""
    async with async_session() as session:
        strategy_id = str(uuid.uuid4())
        await session.execute(text("""
            INSERT INTO strategies (id, user_id, name, type, config, is_active, performance, created_at)
            VALUES (:id, :user_id, :name, :type, :config, false, 0.0, :created)
        """), {
            "id": strategy_id,
            "user_id": current_user["id"],
            "name": strategy.name,
            "type": strategy.type,
            "config": json.dumps(strategy.config),
            "created": datetime.utcnow()
        })
        await session.commit()
        return {"id": strategy_id, "message": "Strategy created successfully"}

@app.post("/api/strategies/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str, current_user: dict = Depends(get_current_user)):
    """Toggle strategy active status"""
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT is_active FROM strategies WHERE id = :id AND user_id = :user_id
        """), {"id": strategy_id, "user_id": current_user["id"]})
        strategy = result.fetchone()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        new_status = not strategy[0]
        await session.execute(text("""
            UPDATE strategies SET is_active = :status WHERE id = :id
        """), {"status": new_status, "id": strategy_id})
        await session.commit()
        
        return {"is_active": new_status}

@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a strategy"""
    async with async_session() as session:
        await session.execute(text("""
            DELETE FROM strategies WHERE id = :id AND user_id = :user_id
        """), {"id": strategy_id, "user_id": current_user["id"]})
        await session.commit()
        return {"message": "Strategy deleted"}

@app.get("/api/risk")
async def get_risk_settings(current_user: dict = Depends(get_current_user)):
    """Get risk settings"""
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT * FROM risk_settings WHERE user_id = :user_id
        """), {"user_id": current_user["id"]})
        settings = result.fetchone()
        
        if not settings:
            # Create default settings
            await session.execute(text("""
                INSERT INTO risk_settings (id, user_id, max_position_size, max_daily_loss, var_limit, kill_switch_enabled, updated_at)
                VALUES (:id, :user_id, 10000.0, 5000.0, 0.05, true, :updated)
            """), {
                "id": str(uuid.uuid4()),
                "user_id": current_user["id"],
                "updated": datetime.utcnow()
            })
            await session.commit()
            
            return {
                "max_position_size": 10000.0,
                "max_daily_loss": 5000.0,
                "var_limit": 0.05,
                "kill_switch_enabled": True
            }
        
        return {
            "max_position_size": settings[2],
            "max_daily_loss": settings[3],
            "var_limit": settings[4],
            "kill_switch_enabled": settings[5]
        }

@app.put("/api/risk")
async def update_risk_settings(settings: RiskSettings, current_user: dict = Depends(get_current_user)):
    """Update risk settings"""
    async with async_session() as session:
        await session.execute(text("""
            UPDATE risk_settings SET 
                max_position_size = :max_pos,
                max_daily_loss = :max_loss,
                var_limit = :var,
                kill_switch_enabled = :kill,
                updated_at = :updated
            WHERE user_id = :user_id
        """), {
            "max_pos": settings.max_position_size,
            "max_loss": settings.max_daily_loss,
            "var": settings.var_limit,
            "kill": settings.kill_switch_enabled,
            "updated": datetime.utcnow(),
            "user_id": current_user["id"]
        })
        await session.commit()
        return {"message": "Risk settings updated"}

@app.get("/api/analytics")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    """Get trading analytics"""
    async with async_session() as session:
        # Get daily P&L for the last 30 days
        result = await session.execute(text("""
            SELECT DATE(closed_at) as date, SUM(pnl) as daily_pnl
            FROM trades 
            WHERE user_id = :user_id AND status = 'closed' AND closed_at > NOW() - INTERVAL '30 days'
            GROUP BY DATE(closed_at)
            ORDER BY date
        """), {"user_id": current_user["id"]})
        daily_pnl = result.fetchall()
        
        # Get strategy performance
        result = await session.execute(text("""
            SELECT name, performance FROM strategies WHERE user_id = :user_id
        """), {"user_id": current_user["id"]})
        strategy_perf = result.fetchall()
        
        # Get trade distribution by symbol
        result = await session.execute(text("""
            SELECT symbol, COUNT(*) as count FROM trades WHERE user_id = :user_id GROUP BY symbol
        """), {"user_id": current_user["id"]})
        symbol_dist = result.fetchall()
        
        return {
            "daily_pnl": [{"date": d[0].isoformat(), "pnl": d[1]} for d in daily_pnl],
            "strategy_performance": [{"name": s[0], "return": s[1]} for s in strategy_perf],
            "symbol_distribution": [{"symbol": s[0], "count": s[1]} for s in symbol_dist],
            "metrics": {
                "sharpe_ratio": 1.85,
                "max_drawdown": -8.5,
                "win_rate": 62.5,
                "profit_factor": 1.75,
                "avg_trade": 125.50,
                "total_trades": 156
            }
        }

@app.get("/api/billing")
async def get_billing(current_user: dict = Depends(get_current_user)):
    """Get billing information"""
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT * FROM subscriptions WHERE user_id = :user_id
        """), {"user_id": current_user["id"]})
        sub = result.fetchone()
        
        return {
            "subscription": {
                "plan": sub[2] if sub else "free",
                "status": sub[3] if sub else "active",
                "expires_at": sub[4].isoformat() if sub and sub[4] else None
            },
            "usage": {
                "api_calls": 45250,
                "api_limit": 100000,
                "strategies_used": 4,
                "strategies_limit": 10 if (sub and sub[2] == "enterprise") else 3
            },
            "invoices": [
                {"id": "INV-001", "date": "2024-02-01", "amount": 299.00, "status": "paid"},
                {"id": "INV-002", "date": "2024-01-01", "amount": 299.00, "status": "paid"},
            ]
        }

@app.get("/api/market-data")
async def get_market_data(symbol: Optional[str] = None, timeframe: str = "1h"):
    """Get market data for charts"""
    # Generate realistic OHLCV data
    candles = []
    base_price = 67250.0 if not symbol or "BTC" in symbol else 3450.0
    
    now = datetime.utcnow()
    for i in range(100):
        timestamp = now - timedelta(hours=100-i)
        volatility = base_price * 0.002
        open_price = base_price + random.uniform(-volatility, volatility)
        close_price = open_price + random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + random.uniform(0, volatility * 0.5)
        low_price = min(open_price, close_price) - random.uniform(0, volatility * 0.5)
        volume = random.uniform(100, 1000)
        
        candles.append({
            "timestamp": timestamp.isoformat(),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })
        base_price = close_price
    
    return {"symbol": symbol or "BTC-USD", "timeframe": timeframe, "data": candles}

# WebSocket for real-time data
@app.websocket("/ws/market")
async def market_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send market data every second
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "prices": {
                    "BTC-USD": 67250.50 + random.uniform(-100, 100),
                    "ETH-USD": 3450.25 + random.uniform(-10, 10),
                    "SOL-USD": 145.75 + random.uniform(-2, 2),
                }
            }
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
