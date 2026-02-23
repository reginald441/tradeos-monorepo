"""
TradeOS API Routers

All API route handlers for the TradeOS platform.
"""

from .auth import router as auth_router
from .trading import router as trading_router
from .strategies import router as strategies_router
from .risk import router as risk_router
from .backtest import router as backtest_router
from .market import router as market_router
from .user import router as user_router
from .admin import router as admin_router
from .billing import router as billing_router

__all__ = [
    "auth_router",
    "trading_router",
    "strategies_router",
    "risk_router",
    "backtest_router",
    "market_router",
    "user_router",
    "admin_router",
    "billing_router",
]
