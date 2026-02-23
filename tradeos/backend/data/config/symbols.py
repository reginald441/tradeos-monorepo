"""
TradeOS Symbols Configuration
Centralized symbol definitions for all supported trading instruments.

This module defines:
- Cryptocurrency pairs (spot and futures)
- Forex pairs
- Commodities (gold, silver, oil)
- Indices
- Stock symbols

Each symbol includes metadata for proper handling across exchanges.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal


class AssetClass(Enum):
    """Asset classification."""
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    STOCK = "stock"
    FUTURES = "futures"
    OPTION = "option"


class MarketType(Enum):
    """Market type classification."""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTIONS = "options"


@dataclass(frozen=True)
class Symbol:
    """Trading symbol definition."""
    base: str
    quote: str
    asset_class: AssetClass
    market_type: MarketType
    
    # Exchange-specific formats
    binance_spot: str = ""
    binance_futures: str = ""
    coinbase: str = ""
    
    # Trading parameters
    price_precision: int = 2
    quantity_precision: int = 8
    min_quantity: Decimal = Decimal("0.00000001")
    max_quantity: Decimal = Decimal("999999999")
    min_notional: Decimal = Decimal("10")
    tick_size: Decimal = Decimal("0.01")
    
    # Metadata
    is_active: bool = True
    supported_exchanges: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Set default exchange formats if not provided
        object.__setattr__(
            self, 'binance_spot',
            self.binance_spot or f"{self.base}{self.quote}".upper()
        )
        object.__setattr__(
            self, 'binance_futures',
            self.binance_futures or f"{self.base}{self.quote}_PERP".upper()
        )
        object.__setattr__(
            self, 'coinbase',
            self.coinbase or f"{self.base}-{self.quote}".upper()
        )
    
    @property
    def symbol(self) -> str:
        """Get standard symbol format."""
        return f"{self.base}{self.quote}".upper()
    
    @property
    def symbol_lower(self) -> str:
        """Get lowercase symbol format."""
        return f"{self.base}{self.quote}".lower()
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return f"{self.base}/{self.quote}"
    
    def get_exchange_format(self, exchange: str) -> str:
        """Get exchange-specific symbol format."""
        formats = {
            'binance_spot': self.binance_spot,
            'binance_futures': self.binance_futures,
            'binance': self.binance_spot,
            'coinbase': self.coinbase,
        }
        return formats.get(exchange.lower(), self.symbol)


# =============================================================================
# CRYPTOCURRENCY SYMBOLS
# =============================================================================

CRYPTO_SYMBOLS: Dict[str, Symbol] = {
    # Bitcoin pairs
    "BTCUSDT": Symbol(
        base="BTC",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=5,
        min_quantity=Decimal("0.00001"),
        max_quantity=Decimal("9000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance", "coinbase", "kraken", "bybit"]
    ),
    "BTCUSD": Symbol(
        base="BTC",
        quote="USD",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=5,
        min_quantity=Decimal("0.00001"),
        max_quantity=Decimal("9000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["coinbase", "kraken", "gemini"]
    ),
    "BTCUSDC": Symbol(
        base="BTC",
        quote="USDC",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=5,
        min_quantity=Decimal("0.00001"),
        max_quantity=Decimal("9000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance", "coinbase"]
    ),
    
    # Ethereum pairs
    "ETHUSDT": Symbol(
        base="ETH",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=4,
        min_quantity=Decimal("0.0001"),
        max_quantity=Decimal("90000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance", "coinbase", "kraken", "bybit"]
    ),
    "ETHUSD": Symbol(
        base="ETH",
        quote="USD",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=4,
        min_quantity=Decimal("0.0001"),
        max_quantity=Decimal("90000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["coinbase", "kraken", "gemini"]
    ),
    "ETHBTC": Symbol(
        base="ETH",
        quote="BTC",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=6,
        quantity_precision=4,
        min_quantity=Decimal("0.0001"),
        max_quantity=Decimal("90000"),
        min_notional=Decimal("0.0001"),
        tick_size=Decimal("0.000001"),
        supported_exchanges=["binance", "coinbase", "kraken"]
    ),
    
    # Altcoins
    "SOLUSDT": Symbol(
        base="SOL",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("1000000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance", "coinbase", "bybit"]
    ),
    "ADAUSDT": Symbol(
        base="ADA",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("90000000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["binance", "coinbase"]
    ),
    "XRPUSDT": Symbol(
        base="XRP",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=4,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("90000000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.0001"),
        supported_exchanges=["binance", "coinbase", "kraken"]
    ),
    "DOTUSDT": Symbol(
        base="DOT",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("900000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance", "coinbase", "kraken"]
    ),
    "LINKUSDT": Symbol(
        base="LINK",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("900000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance", "coinbase", "kraken"]
    ),
    "MATICUSDT": Symbol(
        base="MATIC",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=4,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("10000000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.0001"),
        supported_exchanges=["binance", "coinbase"]
    ),
    "AVAXUSDT": Symbol(
        base="AVAX",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance", "coinbase"]
    ),
    "UNIUSDT": Symbol(
        base="UNI",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance", "coinbase"]
    ),
    "LTCUSDT": Symbol(
        base="LTC",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=3,
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance", "coinbase", "kraken"]
    ),
    "BCHUSDT": Symbol(
        base="BCH",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=3,
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance", "coinbase"]
    ),
    
    # Futures perpetual contracts
    "BTCUSDT_PERP": Symbol(
        base="BTC",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL,
        binance_futures="BTCUSDT",
        price_precision=2,
        quantity_precision=3,
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("1000"),
        min_notional=Decimal("5"),
        tick_size=Decimal("0.10"),
        supported_exchanges=["binance_futures", "bybit", "okx"]
    ),
    "ETHUSDT_PERP": Symbol(
        base="ETH",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL,
        binance_futures="ETHUSDT",
        price_precision=2,
        quantity_precision=3,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("5"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["binance_futures", "bybit", "okx"]
    ),
    "SOLUSDT_PERP": Symbol(
        base="SOL",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL,
        binance_futures="SOLUSDT",
        price_precision=3,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("5"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["binance_futures", "bybit"]
    ),
}


# =============================================================================
# FOREX SYMBOLS
# =============================================================================

FOREX_SYMBOLS: Dict[str, Symbol] = {
    "EURUSD": Symbol(
        base="EUR",
        quote="USD",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"]
    ),
    "GBPUSD": Symbol(
        base="GBP",
        quote="USD",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"]
    ),
    "USDJPY": Symbol(
        base="USD",
        quote="JPY",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"]
    ),
    "USDCHF": Symbol(
        base="USD",
        quote="CHF",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com"]
    ),
    "AUDUSD": Symbol(
        base="AUD",
        quote="USD",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"]
    ),
    "USDCAD": Symbol(
        base="USD",
        quote="CAD",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"]
    ),
    "NZDUSD": Symbol(
        base="NZD",
        quote="USD",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com"]
    ),
    "EURGBP": Symbol(
        base="EUR",
        quote="GBP",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=5,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.00001"),
        supported_exchanges=["oanda", "forex_com"]
    ),
    "EURJPY": Symbol(
        base="EUR",
        quote="JPY",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["oanda", "forex_com"]
    ),
    "GBPJPY": Symbol(
        base="GBP",
        quote="JPY",
        asset_class=AssetClass.FOREX,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("1000"),
        max_quantity=Decimal("100000000"),
        min_notional=Decimal("1000"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["oanda", "forex_com"]
    ),
}


# =============================================================================
# COMMODITY SYMBOLS
# =============================================================================

COMMODITY_SYMBOLS: Dict[str, Symbol] = {
    "XAUUSD": Symbol(
        base="XAU",
        quote="USD",
        asset_class=AssetClass.COMMODITY,
        market_type=MarketType.SPOT,
        price_precision=2,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"],
        aliases=["GOLD", "GOLDUSD"]
    ),
    "XAGUSD": Symbol(
        base="XAG",
        quote="USD",
        asset_class=AssetClass.COMMODITY,
        market_type=MarketType.SPOT,
        price_precision=3,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.001"),
        supported_exchanges=["oanda", "forex_com", "interactive_brokers"],
        aliases=["SILVER", "SILVERUSD"]
    ),
    "USOIL": Symbol(
        base="USOIL",
        quote="USD",
        asset_class=AssetClass.COMMODITY,
        market_type=MarketType.FUTURES,
        price_precision=2,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["WTI", "CL", "OIL"]
    ),
    "UKOIL": Symbol(
        base="UKOIL",
        quote="USD",
        asset_class=AssetClass.COMMODITY,
        market_type=MarketType.FUTURES,
        price_precision=2,
        quantity_precision=0,
        min_quantity=Decimal("1"),
        max_quantity=Decimal("100000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.01"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["BRENT", "BZ"]
    ),
}


# =============================================================================
# INDEX SYMBOLS
# =============================================================================

INDEX_SYMBOLS: Dict[str, Symbol] = {
    "US30": Symbol(
        base="US30",
        quote="USD",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=0,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["DJ30", "DOW", "DOWJONES"]
    ),
    "SPX500": Symbol(
        base="SPX500",
        quote="USD",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=1,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["SP500", "SPX", "S&P500"]
    ),
    "NAS100": Symbol(
        base="NAS100",
        quote="USD",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=1,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("0.1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["NDX", "NASDAQ", "NASDAQ100"]
    ),
    "GER30": Symbol(
        base="GER30",
        quote="EUR",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=0,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["DAX", "DE30"]
    ),
    "UK100": Symbol(
        base="UK100",
        quote="GBP",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=0,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["FTSE", "FTSE100"]
    ),
    "JPN225": Symbol(
        base="JPN225",
        quote="JPY",
        asset_class=AssetClass.INDEX,
        market_type=MarketType.SPOT,
        price_precision=0,
        quantity_precision=2,
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("10000"),
        min_notional=Decimal("1"),
        tick_size=Decimal("1"),
        supported_exchanges=["oanda", "interactive_brokers"],
        aliases=["NIKKEI", "NIKKEI225"]
    ),
}


# =============================================================================
# COMBINED SYMBOL REGISTRY
# =============================================================================

ALL_SYMBOLS: Dict[str, Symbol] = {
    **CRYPTO_SYMBOLS,
    **FOREX_SYMBOLS,
    **COMMODITY_SYMBOLS,
    **INDEX_SYMBOLS,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_symbol(symbol_code: str) -> Optional[Symbol]:
    """Get a symbol by its code."""
    # Try direct lookup
    symbol = ALL_SYMBOLS.get(symbol_code.upper())
    if symbol:
        return symbol
    
    # Try alias lookup
    for sym in ALL_SYMBOLS.values():
        if symbol_code.upper() in [a.upper() for a in sym.aliases]:
            return sym
    
    return None


def get_symbols_by_asset_class(asset_class: AssetClass) -> Dict[str, Symbol]:
    """Get all symbols for a specific asset class."""
    return {
        code: sym for code, sym in ALL_SYMBOLS.items()
        if sym.asset_class == asset_class
    }


def get_symbols_by_exchange(exchange: str) -> Dict[str, Symbol]:
    """Get all symbols supported by a specific exchange."""
    return {
        code: sym for code, sym in ALL_SYMBOLS.items()
        if exchange.lower() in [e.lower() for e in sym.supported_exchanges]
    }


def get_active_symbols() -> Dict[str, Symbol]:
    """Get all active symbols."""
    return {
        code: sym for code, sym in ALL_SYMBOLS.items()
        if sym.is_active
    }


def get_exchange_symbol(symbol_code: str, exchange: str) -> Optional[str]:
    """Get exchange-specific symbol format."""
    symbol = get_symbol(symbol_code)
    if symbol:
        return symbol.get_exchange_format(exchange)
    return None


def get_all_symbol_codes() -> List[str]:
    """Get list of all symbol codes."""
    return list(ALL_SYMBOLS.keys())


def get_crypto_symbols() -> Dict[str, Symbol]:
    """Get all cryptocurrency symbols."""
    return CRYPTO_SYMBOLS.copy()


def get_forex_symbols() -> Dict[str, Symbol]:
    """Get all forex symbols."""
    return FOREX_SYMBOLS.copy()


def get_commodity_symbols() -> Dict[str, Symbol]:
    """Get all commodity symbols."""
    return COMMODITY_SYMBOLS.copy()


def get_index_symbols() -> Dict[str, Symbol]:
    """Get all index symbols."""
    return INDEX_SYMBOLS.copy()


def is_valid_symbol(symbol_code: str) -> bool:
    """Check if a symbol code is valid."""
    return get_symbol(symbol_code) is not None


def get_symbol_pairs(asset_class: Optional[AssetClass] = None) -> List[Tuple[str, str]]:
    """Get list of (base, quote) pairs."""
    symbols = ALL_SYMBOLS.values()
    if asset_class:
        symbols = [s for s in symbols if s.asset_class == asset_class]
    return [(s.base, s.quote) for s in symbols]


# Timeframe definitions
TIMEFRAMES = {
    "1s": {"seconds": 1, "label": "1 Second"},
    "5s": {"seconds": 5, "label": "5 Seconds"},
    "15s": {"seconds": 15, "label": "15 Seconds"},
    "30s": {"seconds": 30, "label": "30 Seconds"},
    "1m": {"seconds": 60, "label": "1 Minute"},
    "3m": {"seconds": 180, "label": "3 Minutes"},
    "5m": {"seconds": 300, "label": "5 Minutes"},
    "15m": {"seconds": 900, "label": "15 Minutes"},
    "30m": {"seconds": 1800, "label": "30 Minutes"},
    "1h": {"seconds": 3600, "label": "1 Hour"},
    "2h": {"seconds": 7200, "label": "2 Hours"},
    "4h": {"seconds": 14400, "label": "4 Hours"},
    "6h": {"seconds": 21600, "label": "6 Hours"},
    "8h": {"seconds": 28800, "label": "8 Hours"},
    "12h": {"seconds": 43200, "label": "12 Hours"},
    "1d": {"seconds": 86400, "label": "1 Day"},
    "3d": {"seconds": 259200, "label": "3 Days"},
    "1w": {"seconds": 604800, "label": "1 Week"},
    "1M": {"seconds": 2592000, "label": "1 Month"},
}


def get_timeframe_seconds(timeframe: str) -> int:
    """Get timeframe in seconds."""
    return TIMEFRAMES.get(timeframe, {}).get("seconds", 60)


def get_timeframe_label(timeframe: str) -> str:
    """Get timeframe human-readable label."""
    return TIMEFRAMES.get(timeframe, {}).get("label", timeframe)


if __name__ == "__main__":
    # Print summary of available symbols
    print("=" * 60)
    print("TradeOS Symbol Configuration")
    print("=" * 60)
    print(f"\nTotal Symbols: {len(ALL_SYMBOLS)}")
    print(f"  - Cryptocurrencies: {len(CRYPTO_SYMBOLS)}")
    print(f"  - Forex Pairs: {len(FOREX_SYMBOLS)}")
    print(f"  - Commodities: {len(COMMODITY_SYMBOLS)}")
    print(f"  - Indices: {len(INDEX_SYMBOLS)}")
    
    print("\n" + "=" * 60)
    print("Sample Symbols:")
    print("=" * 60)
    
    for code in ["BTCUSDT", "ETHUSDT", "EURUSD", "XAUUSD", "US30"]:
        sym = get_symbol(code)
        if sym:
            print(f"\n{code}:")
            print(f"  Display: {sym.display_name}")
            print(f"  Asset Class: {sym.asset_class.value}")
            print(f"  Market Type: {sym.market_type.value}")
            print(f"  Price Precision: {sym.price_precision}")
            print(f"  Exchanges: {', '.join(sym.supported_exchanges)}")
