"""
TradeOS Forex Client
Forex and commodity data client using multiple free/paid data sources.

Supported Providers:
- Alpha Vantage (free tier available)
- ForexRateAPI (free tier available)
- ExchangeRate-API (free tier available)
- OpenExchangeRates (free tier available)
- Yahoo Finance (unofficial)

Features:
- Real-time and historical forex rates
- Multiple provider fallback
- Rate limiting
- Caching
- Error handling and retry logic
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)


class ForexProvider(Enum):
    """Supported forex data providers."""
    ALPHA_VANTAGE = "alpha_vantage"
    FOREX_RATE_API = "forex_rate_api"
    EXCHANGE_RATE_API = "exchange_rate_api"
    OPEN_EXCHANGE_RATES = "open_exchange_rates"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class ForexConfig:
    """Forex client configuration."""
    # Provider selection
    primary_provider: ForexProvider = ForexProvider.FOREX_RATE_API
    fallback_providers: List[ForexProvider] = None
    
    # API keys
    alpha_vantage_api_key: Optional[str] = None
    open_exchange_rates_app_id: Optional[str] = None
    exchange_rate_api_key: Optional[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 5  # Conservative for free tiers
    
    # Caching
    cache_ttl_seconds: int = 60  # Cache forex rates for 60 seconds
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = [
                ForexProvider.EXCHANGE_RATE_API,
                ForexProvider.ALPHA_VANTAGE
            ]


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 5):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class ForexRateAPI:
    """ForexRateAPI client (free tier available)."""
    
    BASE_URL = "https://api.forexrateapi.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def get_latest_rates(
        self,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get latest exchange rates."""
        session = await self._get_session()
        
        params = {"base": base}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/latest?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "error" in data:
                raise ForexAPIError(data["error"]["message"])
            
            return {
                "base": data.get("base"),
                "timestamp": data.get("timestamp"),
                "rates": data.get("rates", {})
            }
    
    async def get_historical_rates(
        self,
        date: str,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get historical exchange rates for a specific date."""
        session = await self._get_session()
        
        params = {"base": base}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/{date}?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "error" in data:
                raise ForexAPIError(data["error"]["message"])
            
            return {
                "base": data.get("base"),
                "date": data.get("date"),
                "rates": data.get("rates", {})
            }
    
    async def convert(
        self,
        from_currency: str,
        to_currency: str,
        amount: float
    ) -> Dict[str, Any]:
        """Convert currency."""
        session = await self._get_session()
        
        params = {
            "from": from_currency,
            "to": to_currency,
            "amount": amount
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/convert?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "error" in data:
                raise ForexAPIError(data["error"]["message"])
            
            return data
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ExchangeRateAPI:
    """ExchangeRate-API client (free tier available)."""
    
    BASE_URL = "https://api.exchangerate-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def get_latest_rates(self, base: str = "USD") -> Dict[str, Any]:
        """Get latest exchange rates."""
        session = await self._get_session()
        
        url = f"{self.BASE_URL}/latest/{base}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if response.status != 200:
                raise ForexAPIError(data.get("error", "Unknown error"))
            
            return {
                "base": data.get("base"),
                "date": data.get("date"),
                "timestamp": data.get("time_last_updated"),
                "rates": data.get("rates", {})
            }
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class AlphaVantageAPI:
    """Alpha Vantage API client (free tier: 5 calls per minute)."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(requests_per_minute=5)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with rate limiting."""
        await self._rate_limiter.acquire()
        
        session = await self._get_session()
        params["apikey"] = self.api_key
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "Error Message" in data:
                raise ForexAPIError(data["Error Message"])
            
            if "Note" in data:
                raise ForexAPIError(f"Rate limit: {data['Note']}")
            
            return data
    
    async def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str
    ) -> Dict[str, Any]:
        """Get real-time exchange rate."""
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        
        data = await self._request(params)
        
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        return {
            "from": rate_data.get("1. From_Currency Code"),
            "to": rate_data.get("3. To_Currency Code"),
            "rate": float(rate_data.get("5. Exchange Rate", 0)),
            "timestamp": rate_data.get("6. Last Refreshed"),
            "timezone": rate_data.get("7. Time Zone"),
            "bid": float(rate_data.get("8. Bid Price", 0)),
            "ask": float(rate_data.get("9. Ask Price", 0)),
        }
    
    async def get_fx_intraday(
        self,
        from_symbol: str,
        to_symbol: str,
        interval: str = "5min"
    ) -> Dict[str, Any]:
        """Get intraday forex data."""
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "outputsize": "compact"
        }
        
        data = await self._request(params)
        
        time_series_key = f"Time Series FX ({interval})"
        time_series = data.get(time_series_key, {})
        
        candles = []
        for timestamp, values in time_series.items():
            candles.append({
                "timestamp": timestamp,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
            })
        
        return {
            "from": data.get("Meta Data", {}).get("2. From Symbol"),
            "to": data.get("Meta Data", {}).get("3. To Symbol"),
            "interval": interval,
            "candles": candles
        }
    
    async def get_fx_daily(
        self,
        from_symbol: str,
        to_symbol: str
    ) -> Dict[str, Any]:
        """Get daily forex data."""
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": "compact"
        }
        
        data = await self._request(params)
        
        time_series = data.get("Time Series FX (Daily)", {})
        
        candles = []
        for date, values in time_series.items():
            candles.append({
                "date": date,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
            })
        
        return {
            "from": data.get("Meta Data", {}).get("2. From Symbol"),
            "to": data.get("Meta Data", {}).get("3. To Symbol"),
            "candles": candles
        }
    
    async def get_digital_currency_daily(
        self,
        symbol: str,
        market: str = "USD"
    ) -> Dict[str, Any]:
        """Get daily cryptocurrency data."""
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market
        }
        
        data = await self._request(params)
        
        time_series_key = f"Time Series (Digital Currency Daily)"
        time_series = data.get(time_series_key, {})
        
        candles = []
        for date, values in time_series.items():
            candles.append({
                "date": date,
                "open": float(values.get(f"1a. open ({market})", 0)),
                "high": float(values.get(f"2a. high ({market})", 0)),
                "low": float(values.get(f"3a. low ({market})", 0)),
                "close": float(values.get(f"4a. close ({market})", 0)),
                "volume": float(values.get(f"5. volume", 0)),
                "market_cap": float(values.get(f"6. market cap (USD)", 0)),
            })
        
        return {
            "symbol": data.get("Meta Data", {}).get("2. Digital Currency Code"),
            "market": market,
            "candles": candles
        }
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class OpenExchangeRatesAPI:
    """OpenExchangeRates API client."""
    
    BASE_URL = "https://openexchangerates.org/api"
    
    def __init__(self, app_id: str):
        self.app_id = app_id
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def get_latest_rates(
        self,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get latest exchange rates."""
        session = await self._get_session()
        
        params = {"app_id": self.app_id}
        if base != "USD":
            params["base"] = base
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        url = f"{self.BASE_URL}/latest.json?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "error" in data:
                raise ForexAPIError(data["error"]["message"])
            
            return {
                "base": data.get("base"),
                "timestamp": data.get("timestamp"),
                "rates": data.get("rates", {})
            }
    
    async def get_historical_rates(
        self,
        date: str,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get historical exchange rates."""
        session = await self._get_session()
        
        params = {"app_id": self.app_id}
        if base != "USD":
            params["base"] = base
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        url = f"{self.BASE_URL}/historical/{date}.json?{urlencode(params)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            if "error" in data:
                raise ForexAPIError(data["error"]["message"])
            
            return {
                "base": data.get("base"),
                "date": date,
                "timestamp": data.get("timestamp"),
                "rates": data.get("rates", {})
            }
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ForexAPIError(Exception):
    """Forex API error."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"Forex API Error: {message}")


class ForexClient:
    """Unified Forex client with multiple provider support."""
    
    def __init__(self, config: Optional[ForexConfig] = None):
        self.config = config or ForexConfig()
        
        # Initialize providers
        self._providers: Dict[ForexProvider, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        self._init_providers()
    
    def _init_providers(self):
        """Initialize API clients for configured providers."""
        # Primary provider
        if self.config.primary_provider == ForexProvider.FOREX_RATE_API:
            self._providers[ForexProvider.FOREX_RATE_API] = ForexRateAPI()
        elif self.config.primary_provider == ForexProvider.EXCHANGE_RATE_API:
            self._providers[ForexProvider.EXCHANGE_RATE_API] = ExchangeRateAPI()
        elif self.config.primary_provider == ForexProvider.ALPHA_VANTAGE:
            if self.config.alpha_vantage_api_key:
                self._providers[ForexProvider.ALPHA_VANTAGE] = AlphaVantageAPI(
                    self.config.alpha_vantage_api_key
                )
        elif self.config.primary_provider == ForexProvider.OPEN_EXCHANGE_RATES:
            if self.config.open_exchange_rates_app_id:
                self._providers[ForexProvider.OPEN_EXCHANGE_RATES] = OpenExchangeRatesAPI(
                    self.config.open_exchange_rates_app_id
                )
        
        # Fallback providers
        for provider in self.config.fallback_providers:
            if provider not in self._providers:
                if provider == ForexProvider.FOREX_RATE_API:
                    self._providers[provider] = ForexRateAPI()
                elif provider == ForexProvider.EXCHANGE_RATE_API:
                    self._providers[provider] = ExchangeRateAPI()
                elif provider == ForexProvider.ALPHA_VANTAGE:
                    if self.config.alpha_vantage_api_key:
                        self._providers[provider] = AlphaVantageAPI(
                            self.config.alpha_vantage_api_key
                        )
                elif provider == ForexProvider.OPEN_EXCHANGE_RATES:
                    if self.config.open_exchange_rates_app_id:
                        self._providers[provider] = OpenExchangeRatesAPI(
                            self.config.open_exchange_rates_app_id
                        )
    
    def _get_cache_key(self, method: str, *args) -> str:
        """Generate cache key."""
        return f"{method}:{':'.join(str(a) for a in args)}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                return self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Any):
        """Cache data."""
        self._cache[key] = data
        self._cache_timestamps[key] = time.time()
    
    async def _try_providers(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Try multiple providers until one succeeds."""
        providers = [self.config.primary_provider] + self.config.fallback_providers
        
        last_error = None
        for provider in providers:
            if provider not in self._providers:
                continue
            
            try:
                provider_client = self._providers[provider]
                method = getattr(provider_client, method_name)
                return await method(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                last_error = e
                continue
        
        if last_error:
            raise last_error
        raise ForexAPIError("All providers failed")
    
    async def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str
    ) -> Dict[str, Any]:
        """Get exchange rate between two currencies."""
        cache_key = self._get_cache_key("rate", from_currency, to_currency)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Try Alpha Vantage first for detailed rate
        if ForexProvider.ALPHA_VANTAGE in self._providers:
            try:
                result = await self._providers[ForexProvider.ALPHA_VANTAGE].get_exchange_rate(
                    from_currency, to_currency
                )
                self._set_cached(cache_key, result)
                return result
            except Exception as e:
                logger.warning(f"Alpha Vantage failed: {e}")
        
        # Fallback to other providers
        rates_data = await self._try_providers("get_latest_rates", base=from_currency)
        
        rate = rates_data.get("rates", {}).get(to_currency)
        if rate is None:
            raise ForexAPIError(f"Rate not found for {from_currency}/{to_currency}")
        
        result = {
            "from": from_currency,
            "to": to_currency,
            "rate": float(rate),
            "timestamp": rates_data.get("timestamp"),
        }
        
        self._set_cached(cache_key, result)
        return result
    
    async def get_latest_rates(
        self,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get latest exchange rates."""
        cache_key = self._get_cache_key("rates", base, symbols)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = await self._try_providers("get_latest_rates", base=base, symbols=symbols)
        self._set_cached(cache_key, result)
        return result
    
    async def convert(
        self,
        from_currency: str,
        to_currency: str,
        amount: float
    ) -> Dict[str, Any]:
        """Convert amount between currencies."""
        rate_data = await self.get_exchange_rate(from_currency, to_currency)
        rate = rate_data.get("rate", 0)
        
        return {
            "from": from_currency,
            "to": to_currency,
            "amount": amount,
            "rate": rate,
            "result": amount * rate,
            "timestamp": rate_data.get("timestamp"),
        }
    
    async def get_historical_rates(
        self,
        date: str,
        base: str = "USD",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get historical exchange rates."""
        return await self._try_providers("get_historical_rates", date=date, base=base, symbols=symbols)
    
    async def get_fx_intraday(
        self,
        from_symbol: str,
        to_symbol: str,
        interval: str = "5min"
    ) -> Dict[str, Any]:
        """Get intraday forex data (Alpha Vantage only)."""
        if ForexProvider.ALPHA_VANTAGE not in self._providers:
            raise ForexAPIError("Alpha Vantage provider required for intraday data")
        
        return await self._providers[ForexProvider.ALPHA_VANTAGE].get_fx_intraday(
            from_symbol, to_symbol, interval
        )
    
    async def get_fx_daily(
        self,
        from_symbol: str,
        to_symbol: str
    ) -> Dict[str, Any]:
        """Get daily forex data (Alpha Vantage only)."""
        if ForexProvider.ALPHA_VANTAGE not in self._providers:
            raise ForexAPIError("Alpha Vantage provider required for daily data")
        
        return await self._providers[ForexProvider.ALPHA_VANTAGE].get_fx_daily(
            from_symbol, to_symbol
        )
    
    async def get_crypto_daily(
        self,
        symbol: str,
        market: str = "USD"
    ) -> Dict[str, Any]:
        """Get daily cryptocurrency data (Alpha Vantage only)."""
        if ForexProvider.ALPHA_VANTAGE not in self._providers:
            raise ForexAPIError("Alpha Vantage provider required for crypto data")
        
        return await self._providers[ForexProvider.ALPHA_VANTAGE].get_digital_currency_daily(
            symbol, market
        )
    
    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()


# Factory functions
def create_client(
    alpha_vantage_api_key: Optional[str] = None,
    open_exchange_rates_app_id: Optional[str] = None,
    primary_provider: ForexProvider = ForexProvider.FOREX_RATE_API
) -> ForexClient:
    """Create a Forex client."""
    config = ForexConfig(
        primary_provider=primary_provider,
        alpha_vantage_api_key=alpha_vantage_api_key,
        open_exchange_rates_app_id=open_exchange_rates_app_id
    )
    return ForexClient(config)


# Singleton instance
_client_instance: Optional[ForexClient] = None


def get_client() -> ForexClient:
    """Get singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = create_client()
    return _client_instance


if __name__ == "__main__":
    async def test_forex_client():
        """Test Forex client."""
        # Create client with free providers
        client = create_client(primary_provider=ForexProvider.EXCHANGE_RATE_API)
        
        try:
            # Test exchange rate
            print("Testing exchange rate...")
            rate = await client.get_exchange_rate("EUR", "USD")
            print(f"EUR/USD: {rate}")
            
            # Test latest rates
            print("\nTesting latest rates...")
            rates = await client.get_latest_rates("USD", ["EUR", "GBP", "JPY"])
            print(f"USD rates: {rates}")
            
            # Test conversion
            print("\nTesting conversion...")
            conversion = await client.convert("USD", "EUR", 100)
            print(f"100 USD to EUR: {conversion}")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await client.close()
    
    asyncio.run(test_forex_client())
