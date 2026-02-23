"""
TradeOS Exchange Clients

Exchange-specific API clients for market data.
"""

from .binance_client import (
    BinanceClient,
    BinanceRESTClient,
    BinanceWebSocketClient,
    BinanceConfig,
    BinanceMarket,
    BinanceStreamType,
    BinanceAPIError,
    create_spot_client,
    create_futures_client,
    get_spot_client,
    get_futures_client,
)

from .coinbase_client import (
    CoinbaseClient,
    CoinbaseRESTClient,
    CoinbaseWebSocketClient,
    CoinbaseConfig,
    CoinbaseAPIError,
    create_client as create_coinbase_client,
    get_client as get_coinbase_client,
)

from .forex_client import (
    ForexClient,
    ForexConfig,
    ForexProvider,
    ForexRateAPI,
    ExchangeRateAPI,
    AlphaVantageAPI,
    OpenExchangeRatesAPI,
    ForexAPIError,
    create_client as create_forex_client,
    get_client as get_forex_client,
)

__all__ = [
    # Binance
    "BinanceClient",
    "BinanceRESTClient",
    "BinanceWebSocketClient",
    "BinanceConfig",
    "BinanceMarket",
    "BinanceStreamType",
    "BinanceAPIError",
    "create_spot_client",
    "create_futures_client",
    "get_spot_client",
    "get_futures_client",
    
    # Coinbase
    "CoinbaseClient",
    "CoinbaseRESTClient",
    "CoinbaseWebSocketClient",
    "CoinbaseConfig",
    "CoinbaseAPIError",
    "create_coinbase_client",
    "get_coinbase_client",
    
    # Forex
    "ForexClient",
    "ForexConfig",
    "ForexProvider",
    "ForexRateAPI",
    "ExchangeRateAPI",
    "AlphaVantageAPI",
    "OpenExchangeRatesAPI",
    "ForexAPIError",
    "create_forex_client",
    "get_forex_client",
]
