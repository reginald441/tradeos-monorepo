"""
TradeOS TimescaleDB Store
Time-series data storage using TimescaleDB/PostgreSQL.

Features:
- Async PostgreSQL operations
- TimescaleDB hypertables for time-series data
- Efficient bulk inserts
- Query optimization
- Data retention policies
- Compression support
- Partitioning by time and symbol
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)


@dataclass
class TimescaleConfig:
    """TimescaleDB connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "tradeos"
    user: str = "tradeos"
    password: str = ""
    
    # Connection pool settings
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: float = 60.0
    
    # Hypertable settings
    chunk_time_interval: str = "1 day"
    compression_after: str = "7 days"
    retention_after: str = "90 days"
    
    def get_dsn(self) -> str:
        """Get connection DSN."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class TimescaleStore:
    """TimescaleDB store for time-series market data."""
    
    # Table schemas
    TABLE_TRADES = "trades"
    TABLE_ORDERBOOKS = "orderbooks"
    TABLE_OHLC = "ohlc"
    TABLE_TICKERS = "tickers"
    TABLE_FUNDING_RATES = "funding_rates"
    TABLE_LIQUIDATIONS = "liquidations"
    TABLE_MARK_PRICES = "mark_prices"
    
    def __init__(self, config: Optional[TimescaleConfig] = None):
        self.config = config or TimescaleConfig()
        self._pool: Optional[Pool] = None
        self._initialized = False
    
    async def connect(self) -> None:
        """Establish database connection pool."""
        try:
            logger.info(f"Connecting to TimescaleDB at {self.config.host}:{self.config.port}")
            
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            logger.info("Successfully connected to TimescaleDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from TimescaleDB")
    
    async def initialize(self) -> None:
        """Initialize database schema and hypertables."""
        if self._initialized:
            return
        
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            # Create extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Create tables and hypertables
            await self._create_trades_table(conn)
            await self._create_ohlc_table(conn)
            await self._create_tickers_table(conn)
            await self._create_orderbooks_table(conn)
            await self._create_funding_rates_table(conn)
            await self._create_liquidations_table(conn)
            await self._create_mark_prices_table(conn)
            
            # Create indexes
            await self._create_indexes(conn)
        
        self._initialized = True
        logger.info("Database schema initialized")
    
    async def _create_trades_table(self, conn: Connection) -> None:
        """Create trades table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_TRADES} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                trade_id TEXT,
                price DOUBLE PRECISION NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                quote_quantity DOUBLE PRECISION,
                side TEXT NOT NULL,
                is_buyer_maker BOOLEAN DEFAULT FALSE,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange, trade_id)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_TRADES}', 'time', 
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_ohlc_table(self, conn: Connection) -> None:
        """Create OHLC table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_OHLC} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                quote_volume DOUBLE PRECISION,
                trades INTEGER,
                taker_buy_volume DOUBLE PRECISION,
                taker_buy_quote_volume DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                closed BOOLEAN DEFAULT TRUE,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange, timeframe)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_OHLC}', 'time',
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_tickers_table(self, conn: Connection) -> None:
        """Create tickers table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_TICKERS} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                quote_volume DOUBLE PRECISION,
                change DOUBLE PRECISION,
                change_percent DOUBLE PRECISION,
                bid DOUBLE PRECISION,
                ask DOUBLE PRECISION,
                bid_volume DOUBLE PRECISION,
                ask_volume DOUBLE PRECISION,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_TICKERS}', 'time',
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_orderbooks_table(self, conn: Connection) -> None:
        """Create orderbooks table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_ORDERBOOKS} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                last_update_id BIGINT,
                bids JSONB NOT NULL,
                asks JSONB NOT NULL,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_ORDERBOOKS}', 'time',
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_funding_rates_table(self, conn: Connection) -> None:
        """Create funding rates table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_FUNDING_RATES} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                funding_rate DOUBLE PRECISION NOT NULL,
                mark_price DOUBLE PRECISION,
                index_price DOUBLE PRECISION,
                estimated_rate DOUBLE PRECISION,
                next_funding_time TIMESTAMPTZ,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_FUNDING_RATES}', 'time',
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_liquidations_table(self, conn: Connection) -> None:
        """Create liquidations table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_LIQUIDATIONS} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                liquidation_id TEXT,
                price DOUBLE PRECISION NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                side TEXT NOT NULL,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange, liquidation_id)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_LIQUIDATIONS}', 'time',
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_mark_prices_table(self, conn: Connection) -> None:
        """Create mark prices table and hypertable."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_MARK_PRICES} (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                mark_price DOUBLE PRECISION NOT NULL,
                index_price DOUBLE PRECISION,
                funding_rate DOUBLE PRECISION,
                raw_data JSONB,
                PRIMARY KEY (time, symbol, exchange)
            );
        """)
        
        await conn.execute(f"""
            SELECT create_hypertable('{self.TABLE_MARK_PRICES}', 'time',
                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                if_not_exists => TRUE
            );
        """)
    
    async def _create_indexes(self, conn: Connection) -> None:
        """Create additional indexes for performance."""
        # Trades indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
            ON {self.TABLE_TRADES} (symbol, time DESC);
        """)
        
        # OHLC indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timeframe_time 
            ON {self.TABLE_OHLC} (symbol, timeframe, time DESC);
        """)
        
        # Ticker indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_tickers_symbol_time 
            ON {self.TABLE_TICKERS} (symbol, time DESC);
        """)
        
        logger.info("Indexes created")
    
    # =========================================================================
    # Insert Operations
    # =========================================================================
    
    async def insert_trade(self, trade: Dict[str, Any]) -> bool:
        """Insert a single trade."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.TABLE_TRADES} (
                        time, symbol, exchange, trade_id, price, quantity,
                        quote_quantity, side, is_buyer_maker, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (time, symbol, exchange, trade_id) DO NOTHING
                """,
                    datetime.fromtimestamp(trade["timestamp"] / 1000),
                    trade["symbol"],
                    trade["exchange"],
                    trade.get("trade_id"),
                    trade["price"],
                    trade["quantity"],
                    trade.get("quote_quantity"),
                    trade["side"],
                    trade.get("is_buyer_maker", False),
                    json.dumps(trade.get("raw_data", {}))
                )
            return True
        except Exception as e:
            logger.error(f"Error inserting trade: {e}")
            return False
    
    async def insert_trades(self, trades: List[Dict[str, Any]]) -> int:
        """Bulk insert trades."""
        if not trades:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                # Prepare data
                records = [
                    (
                        datetime.fromtimestamp(t["timestamp"] / 1000),
                        t["symbol"],
                        t["exchange"],
                        t.get("trade_id"),
                        t["price"],
                        t["quantity"],
                        t.get("quote_quantity"),
                        t["side"],
                        t.get("is_buyer_maker", False),
                        json.dumps(t.get("raw_data", {}))
                    )
                    for t in trades
                ]
                
                await conn.copy_records_to_table(
                    self.TABLE_TRADES,
                    records=records,
                    columns=[
                        "time", "symbol", "exchange", "trade_id", "price",
                        "quantity", "quote_quantity", "side", "is_buyer_maker", "raw_data"
                    ]
                )
            
            return len(trades)
        except Exception as e:
            logger.error(f"Error bulk inserting trades: {e}")
            return 0
    
    async def insert_ohlc(self, candle: Dict[str, Any]) -> bool:
        """Insert a single OHLC candle."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.TABLE_OHLC} (
                        time, symbol, exchange, timeframe, open, high, low, close,
                        volume, quote_volume, trades, taker_buy_volume,
                        taker_buy_quote_volume, vwap, closed, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (time, symbol, exchange, timeframe) 
                    DO UPDATE SET
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trades = EXCLUDED.trades,
                        taker_buy_volume = EXCLUDED.taker_buy_volume,
                        taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
                        vwap = EXCLUDED.vwap,
                        closed = EXCLUDED.closed
                """,
                    datetime.fromtimestamp(candle["timestamp"] / 1000),
                    candle["symbol"],
                    candle["exchange"],
                    candle["timeframe"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                    candle.get("quote_volume"),
                    candle.get("trades"),
                    candle.get("taker_buy_volume"),
                    candle.get("taker_buy_quote_volume"),
                    candle.get("vwap"),
                    candle.get("closed", True),
                    json.dumps(candle.get("raw_data", {}))
                )
            return True
        except Exception as e:
            logger.error(f"Error inserting OHLC: {e}")
            return False
    
    async def insert_ohlc_batch(self, candles: List[Dict[str, Any]]) -> int:
        """Bulk insert OHLC candles."""
        if not candles:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                records = [
                    (
                        datetime.fromtimestamp(c["timestamp"] / 1000),
                        c["symbol"],
                        c["exchange"],
                        c["timeframe"],
                        c["open"],
                        c["high"],
                        c["low"],
                        c["close"],
                        c["volume"],
                        c.get("quote_volume"),
                        c.get("trades"),
                        c.get("taker_buy_volume"),
                        c.get("taker_buy_quote_volume"),
                        c.get("vwap"),
                        c.get("closed", True),
                        json.dumps(c.get("raw_data", {}))
                    )
                    for c in candles
                ]
                
                await conn.copy_records_to_table(
                    self.TABLE_OHLC,
                    records=records,
                    columns=[
                        "time", "symbol", "exchange", "timeframe", "open", "high",
                        "low", "close", "volume", "quote_volume", "trades",
                        "taker_buy_volume", "taker_buy_quote_volume", "vwap", "closed", "raw_data"
                    ]
                )
            
            return len(candles)
        except Exception as e:
            logger.error(f"Error bulk inserting OHLC: {e}")
            return 0
    
    async def insert_ticker(self, ticker: Dict[str, Any]) -> bool:
        """Insert a ticker update."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.TABLE_TICKERS} (
                        time, symbol, exchange, price, open, high, low, close,
                        volume, quote_volume, change, change_percent, bid, ask,
                        bid_volume, ask_volume, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    datetime.fromtimestamp(ticker["timestamp"] / 1000),
                    ticker["symbol"],
                    ticker["exchange"],
                    ticker["price"],
                    ticker.get("open"),
                    ticker.get("high"),
                    ticker.get("low"),
                    ticker.get("close"),
                    ticker.get("volume"),
                    ticker.get("quote_volume"),
                    ticker.get("change"),
                    ticker.get("change_percent"),
                    ticker.get("bid"),
                    ticker.get("ask"),
                    ticker.get("bid_volume"),
                    ticker.get("ask_volume"),
                    json.dumps(ticker.get("raw_data", {}))
                )
            return True
        except Exception as e:
            logger.error(f"Error inserting ticker: {e}")
            return False
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    async def get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get trades for a symbol."""
        try:
            async with self._pool.acquire() as conn:
                query = f"""
                    SELECT * FROM {self.TABLE_TRADES}
                    WHERE symbol = $1
                """
                params = [symbol]
                
                if start_time:
                    query += f" AND time >= ${len(params) + 1}"
                    params.append(start_time)
                
                if end_time:
                    query += f" AND time <= ${len(params) + 1}"
                    params.append(end_time)
                
                query += f" ORDER BY time DESC LIMIT ${len(params) + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": int(row["time"].timestamp() * 1000),
                        "symbol": row["symbol"],
                        "exchange": row["exchange"],
                        "trade_id": row["trade_id"],
                        "price": row["price"],
                        "quantity": row["quantity"],
                        "quote_quantity": row["quote_quantity"],
                        "side": row["side"],
                        "is_buyer_maker": row["is_buyer_maker"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    async def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get OHLC candles for a symbol."""
        try:
            async with self._pool.acquire() as conn:
                query = f"""
                    SELECT * FROM {self.TABLE_OHLC}
                    WHERE symbol = $1 AND timeframe = $2
                """
                params = [symbol, timeframe]
                
                if start_time:
                    query += f" AND time >= ${len(params) + 1}"
                    params.append(start_time)
                
                if end_time:
                    query += f" AND time <= ${len(params) + 1}"
                    params.append(end_time)
                
                query += f" ORDER BY time DESC LIMIT ${len(params) + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": int(row["time"].timestamp() * 1000),
                        "symbol": row["symbol"],
                        "exchange": row["exchange"],
                        "timeframe": row["timeframe"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "quote_volume": row["quote_volume"],
                        "trades": row["trades"],
                        "vwap": row["vwap"],
                        "closed": row["closed"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting OHLC: {e}")
            return []
    
    async def get_latest_ohlc(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest OHLC candle."""
        candles = await self.get_ohlc(symbol, timeframe, limit=1)
        return candles[0] if candles else None
    
    async def get_ticker_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get ticker history for a symbol."""
        try:
            async with self._pool.acquire() as conn:
                query = f"""
                    SELECT * FROM {self.TABLE_TICKERS}
                    WHERE symbol = $1
                """
                params = [symbol]
                
                if start_time:
                    query += f" AND time >= ${len(params) + 1}"
                    params.append(start_time)
                
                if end_time:
                    query += f" AND time <= ${len(params) + 1}"
                    params.append(end_time)
                
                query += f" ORDER BY time DESC LIMIT ${len(params) + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": int(row["time"].timestamp() * 1000),
                        "symbol": row["symbol"],
                        "exchange": row["exchange"],
                        "price": row["price"],
                        "volume": row["volume"],
                        "change_percent": row["change_percent"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting ticker history: {e}")
            return []
    
    # =========================================================================
    # Maintenance Operations
    # =========================================================================
    
    async def setup_compression(self) -> None:
        """Setup compression policy for hypertables."""
        try:
            async with self._pool.acquire() as conn:
                for table in [self.TABLE_TRADES, self.TABLE_OHLC, self.TABLE_TICKERS]:
                    try:
                        await conn.execute(f"""
                            ALTER TABLE {table} SET (
                                timescaledb.compress,
                                timescaledb.compress_segmentby = 'symbol, exchange'
                            );
                        """)
                        
                        await conn.execute(f"""
                            SELECT add_compression_policy('{table}', 
                                INTERVAL '{self.config.compression_after}',
                                if_not_exists => TRUE
                            );
                        """)
                        
                        logger.info(f"Compression setup for {table}")
                    except Exception as e:
                        logger.warning(f"Compression setup failed for {table}: {e}")
        except Exception as e:
            logger.error(f"Error setting up compression: {e}")
    
    async def setup_retention(self) -> None:
        """Setup retention policy for hypertables."""
        try:
            async with self._pool.acquire() as conn:
                for table in [self.TABLE_TRADES, self.TABLE_ORDERBOOKS, self.TABLE_TICKERS]:
                    try:
                        await conn.execute(f"""
                            SELECT add_retention_policy('{table}',
                                INTERVAL '{self.config.retention_after}',
                                if_not_exists => TRUE
                            );
                        """)
                        
                        logger.info(f"Retention policy setup for {table}")
                    except Exception as e:
                        logger.warning(f"Retention policy setup failed for {table}: {e}")
        except Exception as e:
            logger.error(f"Error setting up retention: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            async with self._pool.acquire() as conn:
                # Get hypertable sizes
                sizes = await conn.fetch("""
                    SELECT hypertable_name, 
                           pg_size_pretty(total_bytes) as size,
                           total_bytes
                    FROM hypertable_detailed_size
                    ORDER BY total_bytes DESC;
                """)
                
                # Get chunk counts
                chunks = await conn.fetch("""
                    SELECT hypertable_name, COUNT(*) as chunk_count
                    FROM timescaledb_information.chunks
                    GROUP BY hypertable_name;
                """)
                
                return {
                    "hypertables": [
                        {
                            "name": row["hypertable_name"],
                            "size": row["size"],
                            "size_bytes": row["total_bytes"],
                        }
                        for row in sizes
                    ],
                    "chunks": {
                        row["hypertable_name"]: row["chunk_count"]
                        for row in chunks
                    }
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Factory functions
def create_store(config: Optional[TimescaleConfig] = None) -> TimescaleStore:
    """Create a TimescaleDB store."""
    return TimescaleStore(config)


# Singleton instance
_store_instance: Optional[TimescaleStore] = None


def get_store() -> TimescaleStore:
    """Get singleton store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = create_store()
    return _store_instance


if __name__ == "__main__":
    async def test_store():
        """Test TimescaleDB store."""
        store = create_store()
        
        try:
            await store.connect()
            await store.initialize()
            
            # Insert test trade
            trade = {
                "timestamp": int(time.time() * 1000),
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "trade_id": "12345",
                "price": 50000.0,
                "quantity": 0.1,
                "quote_quantity": 5000.0,
                "side": "buy",
                "is_buyer_maker": False,
            }
            
            success = await store.insert_trade(trade)
            print(f"Trade inserted: {success}")
            
            # Query trades
            trades = await store.get_trades("BTCUSDT", limit=10)
            print(f"Trades: {trades}")
            
            # Get stats
            stats = await store.get_stats()
            print(f"Stats: {stats}")
            
        finally:
            await store.disconnect()
    
    asyncio.run(test_store())
