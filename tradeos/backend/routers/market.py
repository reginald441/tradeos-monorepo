"""
Market Data Router

Handles market data retrieval, price feeds, and chart data.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime, timedelta
from decimal import Decimal

from ..dependencies.auth import get_current_user, get_current_user_ws
from ..database.models import User, MarketData
from ..database.connection import get_db_session
from ..data.feeds.price_feed import UnifiedPriceFeed
from ..data.config.symbols import SUPPORTED_SYMBOLS

router = APIRouter(prefix="/market", tags=["Market Data"])


# Request/Response Models
class OHLCVData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TickerData(BaseModel):
    symbol: str
    price: float
    change_24h: float
    change_pct_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookResponse(BaseModel):
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime


class SymbolInfo(BaseModel):
    symbol: str
    name: str
    type: str
    exchange: str
    min_quantity: float
    max_quantity: float
    quantity_step: float
    price_precision: int


# Initialize price feed
price_feed = UnifiedPriceFeed()


@router.get("/symbols", response_model=List[SymbolInfo])
async def get_supported_symbols(
    asset_type: Optional[Literal["crypto", "forex", "commodity", "index"]] = None
):
    """Get list of supported trading symbols."""
    symbols = []
    
    for symbol, info in SUPPORTED_SYMBOLS.items():
        if asset_type and info.get("type") != asset_type:
            continue
        
        symbols.append(SymbolInfo(
            symbol=symbol,
            name=info.get("name", symbol),
            type=info.get("type", "crypto"),
            exchange=info.get("exchange", "Multiple"),
            min_quantity=info.get("min_quantity", 0.001),
            max_quantity=info.get("max_quantity", 1000000),
            quantity_step=info.get("quantity_step", 0.001),
            price_precision=info.get("price_precision", 2)
        ))
    
    return symbols


@router.get("/ohlcv/{symbol}", response_model=List[OHLCVData])
async def get_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    current_user: User = Depends(get_current_user)
):
    """Get OHLCV candlestick data for a symbol."""
    async with get_db_session() as session:
        from sqlalchemy import select, and_
        
        query = select(MarketData).where(
            and_(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            )
        )
        
        if start_time:
            query = query.where(MarketData.timestamp >= start_time)
        if end_time:
            query = query.where(MarketData.timestamp <= end_time)
        
        query = query.order_by(MarketData.timestamp.desc()).limit(limit)
        
        result = await session.execute(query)
        data = result.scalars().all()
        
        if not data:
            # Return mock data if no data in database
            return _generate_mock_ohlcv(symbol, timeframe, limit)
        
        return [
            OHLCVData(
                timestamp=d.timestamp,
                open=float(d.open),
                high=float(d.high),
                low=float(d.low),
                close=float(d.close),
                volume=float(d.volume)
            )
            for d in reversed(data)
        ]


@router.get("/ticker/{symbol}", response_model=TickerData)
async def get_ticker(symbol: str):
    """Get current ticker data for a symbol."""
    try:
        # Try to get from price feed
        ticker = await price_feed.get_ticker(symbol)
        
        return TickerData(
            symbol=symbol,
            price=ticker.get("price", 0),
            change_24h=ticker.get("change_24h", 0),
            change_pct_24h=ticker.get("change_pct_24h", 0),
            volume_24h=ticker.get("volume_24h", 0),
            high_24h=ticker.get("high_24h", 0),
            low_24h=ticker.get("low_24h", 0),
            bid=ticker.get("bid", 0),
            ask=ticker.get("ask", 0),
            spread=ticker.get("spread", 0),
            timestamp=datetime.utcnow()
        )
    except Exception:
        # Return mock ticker
        return _generate_mock_ticker(symbol)


@router.get("/orderbook/{symbol}", response_model=OrderBookResponse)
async def get_orderbook(
    symbol: str,
    depth: int = 10
):
    """Get order book for a symbol."""
    try:
        orderbook = await price_feed.get_orderbook(symbol, depth=depth)
        
        return OrderBookResponse(
            symbol=symbol,
            bids=[OrderBookLevel(price=b[0], quantity=b[1]) for b in orderbook.get("bids", [])],
            asks=[OrderBookLevel(price=a[0], quantity=a[1]) for a in orderbook.get("asks", [])],
            timestamp=datetime.utcnow()
        )
    except Exception:
        # Return mock orderbook
        return _generate_mock_orderbook(symbol, depth)


@router.get("/prices")
async def get_multiple_prices(symbols: str):
    """Get current prices for multiple symbols (comma-separated)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    prices = {}
    for symbol in symbol_list:
        try:
            ticker = await price_feed.get_ticker(symbol)
            prices[symbol] = {
                "price": ticker.get("price", 0),
                "change_24h": ticker.get("change_pct_24h", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception:
            # Mock price
            import random
            base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            prices[symbol] = {
                "price": base_price * (1 + random.uniform(-0.02, 0.02)),
                "change_24h": random.uniform(-5, 5),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    return prices


@router.websocket("/ws/price/{symbol}")
async def price_websocket(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time price updates."""
    await websocket.accept()
    
    try:
        # Subscribe to price feed
        await price_feed.subscribe_symbol(symbol)
        
        while True:
            # Get latest price
            ticker = await price_feed.get_ticker(symbol)
            
            await websocket.send_json({
                "symbol": symbol,
                "price": ticker.get("price", 0),
                "bid": ticker.get("bid", 0),
                "ask": ticker.get("ask", 0),
                "volume_24h": ticker.get("volume_24h", 0),
                "change_24h": ticker.get("change_pct_24h", 0),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Wait before next update
            import asyncio
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        await price_feed.unsubscribe_symbol(symbol)
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


@router.get("/search")
async def search_symbols(query: str, limit: int = 10):
    """Search for symbols by name or ticker."""
    query = query.upper()
    
    results = []
    for symbol, info in SUPPORTED_SYMBOLS.items():
        name = info.get("name", "").upper()
        
        if query in symbol or query in name:
            results.append({
                "symbol": symbol,
                "name": info.get("name", symbol),
                "type": info.get("type", "crypto"),
                "exchange": info.get("exchange", "Multiple")
            })
        
        if len(results) >= limit:
            break
    
    return results


@router.get("/history/{symbol}")
async def get_price_history(
    symbol: str,
    period: Literal["1d", "7d", "30d", "90d", "1y"] = "30d",
    current_user: User = Depends(get_current_user)
):
    """Get price history for charting."""
    # Calculate date range
    end_time = datetime.utcnow()
    period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}
    start_time = end_time - timedelta(days=period_days.get(period, 30))
    
    # Get OHLCV data
    ohlcv = await get_ohlcv(
        symbol=symbol,
        timeframe="1h" if period in ["1d", "7d"] else "1d",
        limit=1000,
        start_time=start_time,
        end_time=end_time,
        current_user=current_user
    )
    
    return {
        "symbol": symbol,
        "period": period,
        "data": [d.dict() for d in ohlcv]
    }


# Helper functions for mock data
def _generate_mock_ohlcv(symbol: str, timeframe: str, limit: int) -> List[OHLCVData]:
    """Generate mock OHLCV data."""
    import random
    
    # Base price based on symbol
    if "BTC" in symbol:
        base_price = 65000
    elif "ETH" in symbol:
        base_price = 3500
    elif "SOL" in symbol:
        base_price = 150
    elif "EUR" in symbol:
        base_price = 1.08
    elif "XAU" in symbol:
        base_price = 2050
    else:
        base_price = 100
    
    data = []
    current_price = base_price
    
    for i in range(limit):
        timestamp = datetime.utcnow() - timedelta(
            minutes=limit-i if timeframe == "1m" else
            hours=limit-i if timeframe == "1h" else
            days=limit-i
        )
        
        # Random price movement
        change = random.uniform(-0.02, 0.02)
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = random.uniform(100, 10000)
        
        data.append(OHLCVData(
            timestamp=timestamp,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 2)
        ))
        
        current_price = close_price
    
    return data


def _generate_mock_ticker(symbol: str) -> TickerData:
    """Generate mock ticker data."""
    import random
    
    if "BTC" in symbol:
        base_price = 65000
    elif "ETH" in symbol:
        base_price = 3500
    elif "SOL" in symbol:
        base_price = 150
    elif "EUR" in symbol:
        base_price = 1.08
    elif "XAU" in symbol:
        base_price = 2050
    else:
        base_price = 100
    
    price = base_price * (1 + random.uniform(-0.02, 0.02))
    change_pct = random.uniform(-5, 5)
    change = price * change_pct / 100
    
    return TickerData(
        symbol=symbol,
        price=round(price, 2),
        change_24h=round(change, 2),
        change_pct_24h=round(change_pct, 2),
        volume_24h=round(random.uniform(1000000, 100000000), 2),
        high_24h=round(price * 1.05, 2),
        low_24h=round(price * 0.95, 2),
        bid=round(price * 0.999, 2),
        ask=round(price * 1.001, 2),
        spread=round(price * 0.002, 2),
        timestamp=datetime.utcnow()
    )


def _generate_mock_orderbook(symbol: str, depth: int) -> OrderBookResponse:
    """Generate mock orderbook data."""
    import random
    
    if "BTC" in symbol:
        base_price = 65000
    elif "ETH" in symbol:
        base_price = 3500
    elif "SOL" in symbol:
        base_price = 150
    else:
        base_price = 100
    
    bids = []
    asks = []
    
    for i in range(depth):
        bid_price = base_price * (1 - 0.001 * (i + 1))
        ask_price = base_price * (1 + 0.001 * (i + 1))
        
        bids.append(OrderBookLevel(
            price=round(bid_price, 2),
            quantity=round(random.uniform(0.1, 10), 4)
        ))
        
        asks.append(OrderBookLevel(
            price=round(ask_price, 2),
            quantity=round(random.uniform(0.1, 10), 4)
        ))
    
    return OrderBookResponse(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow()
    )
