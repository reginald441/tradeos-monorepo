"""
TradeOS Execution Engine
========================
Complete order execution system for algorithmic trading.

Modules:
    - base_exchange: Abstract base class for all exchanges
    - exchanges: Concrete exchange implementations (Binance, Coinbase, Kraken)
    - forex: Forex broker integrations (MT5, cTrader)
    - models: Order models and types
    - order_manager: Order lifecycle management
    - paper_trading: Simulated trading environment
    - slippage_model: Slippage estimation and tracking
    - latency_tracker: Execution latency monitoring
    - config: Exchange configuration management
"""

from .base_exchange import (
    BaseExchange,
    ExchangeFactory,
    Balance,
    Ticker,
    OrderBook,
    OrderBookLevel,
    Position,
    ExchangeError,
    AuthenticationError,
    InsufficientFundsError,
    InvalidOrderError,
    RateLimitError,
    NetworkError,
    OrderNotFoundError,
)

from .models.order import (
    Order,
    OrderRequest,
    OrderFill,
    OrderBatch,
    BracketOrder,
    OrderType,
    OrderSide,
    OrderState,
    TimeInForce,
    PositionSide,
    OrderEvent,
    OrderEventType,
)

from .order_manager import (
    OrderManager,
    OrderSubmission,
    OrderSubmissionStatus,
    RetryPolicy,
)

from .paper_trading import PaperTradingExchange

from .slippage_model import (
    SlippageModel,
    SlippageRecord,
    SlippageEstimate,
    MarketConditions,
    global_slippage_model,
)

from .latency_tracker import (
    LatencyTracker,
    LatencyMeasurement,
    LatencyStats,
    LatencyType,
    AsyncLatencyContext,
    measure_latency,
    global_latency_tracker,
)

from .config.exchange_config import (
    ExchangeConfig,
    ExchangeConfigManager,
    ConnectionConfig,
    RateLimitConfig,
    TradingFees,
    SymbolConfig,
    config_manager,
    get_exchange_config,
)

# Exchange implementations
from .exchanges.binance_execution import BinanceExchange, BinanceFuturesExchange
from .exchanges.coinbase_execution import CoinbaseExchange
from .exchanges.kraken_execution import KrakenExchange

# Forex bridges
from .forex.mt5_bridge import MT5Bridge
from .forex.ctrader_bridge import cTraderBridge

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseExchange",
    "ExchangeFactory",
    "Balance",
    "Ticker",
    "OrderBook",
    "OrderBookLevel",
    "Position",
    
    # Exceptions
    "ExchangeError",
    "AuthenticationError",
    "InsufficientFundsError",
    "InvalidOrderError",
    "RateLimitError",
    "NetworkError",
    "OrderNotFoundError",
    
    # Order models
    "Order",
    "OrderRequest",
    "OrderFill",
    "OrderBatch",
    "BracketOrder",
    "OrderType",
    "OrderSide",
    "OrderState",
    "TimeInForce",
    "PositionSide",
    "OrderEvent",
    "OrderEventType",
    
    # Order management
    "OrderManager",
    "OrderSubmission",
    "OrderSubmissionStatus",
    "RetryPolicy",
    
    # Paper trading
    "PaperTradingExchange",
    
    # Slippage
    "SlippageModel",
    "SlippageRecord",
    "SlippageEstimate",
    "MarketConditions",
    "global_slippage_model",
    
    # Latency tracking
    "LatencyTracker",
    "LatencyMeasurement",
    "LatencyStats",
    "LatencyType",
    "AsyncLatencyContext",
    "measure_latency",
    "global_latency_tracker",
    
    # Configuration
    "ExchangeConfig",
    "ExchangeConfigManager",
    "ConnectionConfig",
    "RateLimitConfig",
    "TradingFees",
    "SymbolConfig",
    "config_manager",
    "get_exchange_config",
    
    # Exchanges
    "BinanceExchange",
    "BinanceFuturesExchange",
    "CoinbaseExchange",
    "KrakenExchange",
    
    # Forex
    "MT5Bridge",
    "cTraderBridge",
]
