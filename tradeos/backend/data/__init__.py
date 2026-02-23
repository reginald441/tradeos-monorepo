"""
TradeOS Market Intelligence Engine - Data Layer

This package provides comprehensive market data handling capabilities:
- WebSocket connections with auto-reconnection
- Exchange API clients (Binance, Coinbase, Forex)
- OHLC candle aggregation
- Data normalization
- Time-series storage (TimescaleDB)
- Redis caching
- Data validation
- Market microstructure analysis
- Unified price feeds
- Historical data import
"""

from .websocket_manager import (
    WebSocketManager,
    WebSocketConnection,
    WebSocketConfig,
    ConnectionState,
    FeedType,
    get_websocket_manager,
    create_default_manager,
)

from .config.symbols import (
    Symbol,
    AssetClass,
    MarketType,
    get_symbol,
    get_symbols_by_asset_class,
    get_symbols_by_exchange,
    get_exchange_symbol,
    get_all_symbol_codes,
    get_timeframe_seconds,
    TIMEFRAMES,
    CRYPTO_SYMBOLS,
    FOREX_SYMBOLS,
    COMMODITY_SYMBOLS,
    INDEX_SYMBOLS,
)

from .exchanges.binance_client import (
    BinanceClient,
    BinanceRESTClient,
    BinanceWebSocketClient,
    BinanceConfig,
    BinanceMarket,
    create_spot_client,
    create_futures_client,
    get_spot_client,
    get_futures_client,
)

from .exchanges.coinbase_client import (
    CoinbaseClient,
    CoinbaseRESTClient,
    CoinbaseWebSocketClient,
    CoinbaseConfig,
    create_client as create_coinbase_client,
    get_client as get_coinbase_client,
)

from .exchanges.forex_client import (
    ForexClient,
    ForexConfig,
    ForexProvider,
    create_client as create_forex_client,
    get_client as get_forex_client,
)

from .aggregators.ohlc_builder import (
    OHLCBuilder,
    CandleBuilder,
    MultiTimeframeBuilder,
    RealtimeOHLCManager,
    CandleAggregator,
    OHLCV,
    Tick,
    create_builder,
    create_realtime_manager,
    get_builder,
)

from .normalizers.data_normalizer import (
    DataNormalizer,
    ExchangeType,
    DataType,
    NormalizedTrade,
    NormalizedOrderbook,
    NormalizedTicker,
    NormalizedOHLC,
    NormalizedFundingRate,
    NormalizedLiquidation,
    create_normalizer,
    get_normalizer,
)

from .storage.timescale_store import (
    TimescaleStore,
    TimescaleConfig,
    create_store,
    get_store,
)

from .storage.redis_cache import (
    RedisCache,
    RedisConfig,
    CacheEntry,
    CacheStats,
    get_cache,
    init_cache,
    cached,
)

from .processors.validator import (
    DataValidator,
    ValidationPipeline,
    ValidationRules,
    ValidationLevel,
    ValidationResult,
    ValidationError,
    create_validator,
    create_pipeline,
    get_validator,
)

from .processors.microstructure import (
    MicrostructureProcessor,
    SpreadAnalyzer,
    VolumeProfiler,
    OrderFlowAnalyzer,
    LiquidityAnalyzer,
    PriceImpactAnalyzer,
    MicrostructureSignal,
    create_processor,
    get_processor,
)

from .feeds.price_feed import (
    UnifiedPriceFeed,
    PriceFeedAggregator,
    AggregatedPrice,
    PricePoint,
    AggregationMethod,
    create_feed,
    get_feed,
)

from .historical.importer import (
    HistoricalDataImporter,
    CSVImporter,
    APIImporter,
    ImportConfig,
    ImportProgress,
    ImportFormat,
    create_importer,
    get_importer,
)

__version__ = "1.0.0"

__all__ = [
    # WebSocket
    "WebSocketManager",
    "WebSocketConnection",
    "WebSocketConfig",
    "ConnectionState",
    "FeedType",
    "get_websocket_manager",
    "create_default_manager",
    
    # Symbols
    "Symbol",
    "AssetClass",
    "MarketType",
    "get_symbol",
    "get_symbols_by_asset_class",
    "get_symbols_by_exchange",
    "get_exchange_symbol",
    "get_all_symbol_codes",
    "get_timeframe_seconds",
    "TIMEFRAMES",
    
    # Exchanges
    "BinanceClient",
    "BinanceRESTClient",
    "BinanceWebSocketClient",
    "BinanceConfig",
    "BinanceMarket",
    "create_spot_client",
    "create_futures_client",
    "get_spot_client",
    "get_futures_client",
    "CoinbaseClient",
    "CoinbaseRESTClient",
    "CoinbaseWebSocketClient",
    "CoinbaseConfig",
    "create_coinbase_client",
    "get_coinbase_client",
    "ForexClient",
    "ForexConfig",
    "ForexProvider",
    "create_forex_client",
    "get_forex_client",
    
    # Aggregators
    "OHLCBuilder",
    "CandleBuilder",
    "MultiTimeframeBuilder",
    "RealtimeOHLCManager",
    "CandleAggregator",
    "OHLCV",
    "Tick",
    "create_builder",
    "create_realtime_manager",
    "get_builder",
    
    # Normalizers
    "DataNormalizer",
    "ExchangeType",
    "DataType",
    "NormalizedTrade",
    "NormalizedOrderbook",
    "NormalizedTicker",
    "NormalizedOHLC",
    "NormalizedFundingRate",
    "NormalizedLiquidation",
    "create_normalizer",
    "get_normalizer",
    
    # Storage
    "TimescaleStore",
    "TimescaleConfig",
    "create_store",
    "get_store",
    "RedisCache",
    "RedisConfig",
    "CacheEntry",
    "CacheStats",
    "get_cache",
    "init_cache",
    "cached",
    
    # Processors
    "DataValidator",
    "ValidationPipeline",
    "ValidationRules",
    "ValidationLevel",
    "ValidationResult",
    "ValidationError",
    "create_validator",
    "create_pipeline",
    "get_validator",
    "MicrostructureProcessor",
    "SpreadAnalyzer",
    "VolumeProfiler",
    "OrderFlowAnalyzer",
    "LiquidityAnalyzer",
    "PriceImpactAnalyzer",
    "MicrostructureSignal",
    "create_processor",
    "get_processor",
    
    # Feeds
    "UnifiedPriceFeed",
    "PriceFeedAggregator",
    "AggregatedPrice",
    "PricePoint",
    "AggregationMethod",
    "create_feed",
    "get_feed",
    
    # Historical
    "HistoricalDataImporter",
    "CSVImporter",
    "APIImporter",
    "ImportConfig",
    "ImportProgress",
    "ImportFormat",
    "create_importer",
    "get_importer",
]
