"""
TradeOS Exchange Configuration
==============================
Configuration management for exchange connections.
Supports API keys, rate limits, and connection settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set


@dataclass
class RateLimitConfig:
    """Rate limit configuration for exchanges."""
    requests_per_second: int = 10
    requests_per_minute: int = 600
    requests_per_hour: int = 10000
    burst_size: int = 20
    retry_after_seconds: float = 1.0
    max_retries: int = 3
    backoff_multiplier: float = 2.0


@dataclass
class ConnectionConfig:
    """Connection configuration."""
    base_url: str = ""
    ws_url: str = ""
    api_key: str = ""
    api_secret: str = ""
    passphrase: Optional[str] = None  # For Coinbase Pro, etc.
    timeout_seconds: float = 30.0
    connect_timeout: float = 10.0
    max_connections: int = 10
    keepalive: bool = True
    verify_ssl: bool = True
    proxy_url: Optional[str] = None
    
    # Testnet settings
    use_testnet: bool = False
    testnet_base_url: Optional[str] = None
    testnet_ws_url: Optional[str] = None
    
    @property
    def effective_base_url(self) -> str:
        """Get the effective base URL (testnet or production)."""
        if self.use_testnet and self.testnet_base_url:
            return self.testnet_base_url
        return self.base_url
    
    @property
    def effective_ws_url(self) -> str:
        """Get the effective WebSocket URL."""
        if self.use_testnet and self.testnet_ws_url:
            return self.testnet_ws_url
        return self.ws_url


@dataclass
class TradingFees:
    """Trading fee structure."""
    maker_fee: Decimal = field(default_factory=lambda: Decimal("0.001"))  # 0.1%
    taker_fee: Decimal = field(default_factory=lambda: Decimal("0.001"))  # 0.1%
    
    # Tiered fees based on volume
    volume_tiers: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_fee_for_volume(self, volume_30d: Decimal, is_maker: bool = False) -> Decimal:
        """Get fee rate based on 30-day volume."""
        applicable_tier = None
        for tier in sorted(self.volume_tiers, key=lambda x: Decimal(x["min_volume"]), reverse=True):
            if volume_30d >= Decimal(tier["min_volume"]):
                applicable_tier = tier
                break
        
        if applicable_tier:
            return Decimal(applicable_tier["maker_fee"] if is_maker else applicable_tier["taker_fee"])
        return self.maker_fee if is_maker else self.taker_fee


@dataclass
class SymbolConfig:
    """Configuration for a trading symbol."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: Decimal = field(default_factory=lambda: Decimal("0.0001"))
    max_quantity: Decimal = field(default_factory=lambda: Decimal("1000000"))
    quantity_step: Decimal = field(default_factory=lambda: Decimal("0.0001"))
    min_price: Decimal = field(default_factory=lambda: Decimal("0.00000001"))
    max_price: Decimal = field(default_factory=lambda: Decimal("1000000"))
    price_tick: Decimal = field(default_factory=lambda: Decimal("0.01"))
    min_notional: Decimal = field(default_factory=lambda: Decimal("10"))
    max_leverage: Optional[Decimal] = None
    is_tradable: bool = True
    
    # Contract specifications (for derivatives)
    contract_size: Optional[Decimal] = None
    settlement_asset: Optional[str] = None


@dataclass
class ExchangeConfig:
    """Complete configuration for an exchange."""
    name: str
    exchange_type: str = "spot"  # spot, futures, options, forex
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    fees: TradingFees = field(default_factory=TradingFees)
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)
    enabled: bool = True
    
    # Supported features
    supports_spot: bool = True
    supports_margin: bool = False
    supports_futures: bool = False
    supports_options: bool = False
    supports_websocket: bool = True
    supports_testnet: bool = True
    
    # Default settings
    default_time_in_force: str = "GTC"
    max_open_orders: int = 200
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a symbol."""
        return self.symbols.get(symbol)
    
    def add_symbol(self, config: SymbolConfig):
        """Add symbol configuration."""
        self.symbols[config.symbol] = config


class ExchangeConfigManager:
    """Manager for all exchange configurations."""
    
    def __init__(self):
        self._configs: Dict[str, ExchangeConfig] = {}
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Binance configuration
        binance_config = ExchangeConfig(
            name="binance",
            exchange_type="spot",
            connection=ConnectionConfig(
                base_url="https://api.binance.com",
                ws_url="wss://stream.binance.com:9443/ws",
                api_key=os.getenv("BINANCE_API_KEY", ""),
                api_secret=os.getenv("BINANCE_API_SECRET", ""),
                testnet_base_url="https://testnet.binance.vision",
                testnet_ws_url="wss://testnet.binance.vision/ws",
                use_testnet=os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true",
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=20,
                requests_per_minute=1200,
                burst_size=50,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0.001"),
                taker_fee=Decimal("0.001"),
            ),
            supports_spot=True,
            supports_margin=True,
            supports_futures=True,
            supports_testnet=True,
        )
        
        # Binance Futures configuration
        binance_futures_config = ExchangeConfig(
            name="binance_futures",
            exchange_type="futures",
            connection=ConnectionConfig(
                base_url="https://fapi.binance.com",
                ws_url="wss://fstream.binance.com/ws",
                api_key=os.getenv("BINANCE_FUTURES_API_KEY", ""),
                api_secret=os.getenv("BINANCE_FUTURES_API_SECRET", ""),
                testnet_base_url="https://testnet.binancefuture.com",
                testnet_ws_url="wss://stream.binancefuture.com/ws",
                use_testnet=os.getenv("BINANCE_FUTURES_USE_TESTNET", "false").lower() == "true",
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=20,
                requests_per_minute=1200,
                burst_size=50,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0.0002"),
                taker_fee=Decimal("0.0004"),
            ),
            supports_futures=True,
            supports_testnet=True,
        )
        
        # Coinbase configuration
        coinbase_config = ExchangeConfig(
            name="coinbase",
            exchange_type="spot",
            connection=ConnectionConfig(
                base_url="https://api.exchange.coinbase.com",
                ws_url="wss://ws-feed.exchange.coinbase.com",
                api_key=os.getenv("COINBASE_API_KEY", ""),
                api_secret=os.getenv("COINBASE_API_SECRET", ""),
                passphrase=os.getenv("COINBASE_PASSPHRASE", ""),
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=10,
                requests_per_minute=300,
                burst_size=15,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0.005"),
                taker_fee=Decimal("0.006"),
            ),
            supports_spot=True,
        )
        
        # Kraken configuration
        kraken_config = ExchangeConfig(
            name="kraken",
            exchange_type="spot",
            connection=ConnectionConfig(
                base_url="https://api.kraken.com",
                ws_url="wss://ws.kraken.com",
                api_key=os.getenv("KRAKEN_API_KEY", ""),
                api_secret=os.getenv("KRAKEN_API_SECRET", ""),
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=1,
                requests_per_minute=60,
                burst_size=5,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0.0016"),
                taker_fee=Decimal("0.0026"),
            ),
            supports_spot=True,
            supports_futures=True,
        )
        
        # MT5 Bridge configuration
        mt5_config = ExchangeConfig(
            name="mt5",
            exchange_type="forex",
            connection=ConnectionConfig(
                base_url="localhost",
                ws_url="tcp://localhost:5555",  # ZeroMQ
                timeout_seconds=60.0,
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=100,
                burst_size=100,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0"),
                taker_fee=Decimal("0"),
            ),
            supports_spot=True,
        )
        
        # cTrader configuration
        ctrader_config = ExchangeConfig(
            name="ctrader",
            exchange_type="forex",
            connection=ConnectionConfig(
                base_url="https://api.ctrader.com",
                ws_url="wss://api.ctrader.com/ws",
                api_key=os.getenv("CTRADER_API_KEY", ""),
                api_secret=os.getenv("CTRADER_API_SECRET", ""),
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=10,
                requests_per_minute=600,
                burst_size=20,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0"),
                taker_fee=Decimal("0"),
            ),
            supports_spot=True,
        )
        
        # Paper trading configuration
        paper_config = ExchangeConfig(
            name="paper",
            exchange_type="paper",
            connection=ConnectionConfig(
                base_url="internal",
                timeout_seconds=1.0,
            ),
            rate_limits=RateLimitConfig(
                requests_per_second=10000,
                burst_size=10000,
            ),
            fees=TradingFees(
                maker_fee=Decimal("0.001"),
                taker_fee=Decimal("0.001"),
            ),
            supports_spot=True,
            supports_margin=True,
            supports_futures=True,
        )
        
        self._configs = {
            "binance": binance_config,
            "binance_futures": binance_futures_config,
            "coinbase": coinbase_config,
            "kraken": kraken_config,
            "mt5": mt5_config,
            "ctrader": ctrader_config,
            "paper": paper_config,
        }
    
    def get_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for an exchange."""
        return self._configs.get(exchange_name)
    
    def add_config(self, config: ExchangeConfig):
        """Add or update exchange configuration."""
        self._configs[config.name] = config
    
    def remove_config(self, exchange_name: str):
        """Remove exchange configuration."""
        if exchange_name in self._configs:
            del self._configs[exchange_name]
    
    def list_exchanges(self) -> List[str]:
        """List all configured exchanges."""
        return list(self._configs.keys())
    
    def get_enabled_exchanges(self) -> List[str]:
        """List enabled exchanges."""
        return [name for name, config in self._configs.items() if config.enabled]
    
    def update_api_credentials(self, exchange_name: str, api_key: str, 
                               api_secret: str, passphrase: Optional[str] = None):
        """Update API credentials for an exchange."""
        config = self._configs.get(exchange_name)
        if config:
            config.connection.api_key = api_key
            config.connection.api_secret = api_secret
            if passphrase:
                config.connection.passphrase = passphrase


# Global configuration manager instance
config_manager = ExchangeConfigManager()


def get_exchange_config(exchange_name: str) -> Optional[ExchangeConfig]:
    """Get configuration for an exchange."""
    return config_manager.get_config(exchange_name)
