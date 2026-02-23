"""
TradeOS Historical Data Importer
Import historical market data from various sources.

Features:
- CSV file import
- API-based import
- Batch processing
- Data validation
- Progress tracking
- Resume capability
- Multiple format support
"""

import asyncio
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from pathlib import Path
import aiohttp
import aiofiles

from ..config.symbols import get_symbol, TIMEFRAMES, get_timeframe_seconds
from ..processors.validator import DataValidator, ValidationResult
from ..storage.timescale_store import TimescaleStore, get_store

logger = logging.getLogger(__name__)


class ImportFormat(Enum):
    """Supported import formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    BINANCE_CSV = "binance_csv"
    COINBASE_CSV = "coinbase_csv"


@dataclass
class ImportConfig:
    """Import configuration."""
    batch_size: int = 1000
    validate_data: bool = True
    skip_errors: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    progress_interval: int = 1000
    
    # CSV settings
    csv_delimiter: str = ","
    csv_quotechar: str = '"'
    csv_has_header: bool = True


@dataclass
class ImportProgress:
    """Import progress tracking."""
    total_records: int = 0
    processed_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def records_per_second(self) -> float:
        """Get processing rate."""
        if not self.start_time:
            return 0.0
        
        elapsed = (self.end_time or datetime.now()) - self.start_time
        seconds = elapsed.total_seconds()
        
        if seconds == 0:
            return 0.0
        
        return self.processed_records / seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "progress_percent": self.progress_percent,
            "records_per_second": self.records_per_second,
            "errors": len(self.errors),
        }


class CSVImporter:
    """Import data from CSV files."""
    
    # Column mappings for different formats
    COLUMN_MAPPINGS = {
        "standard": {
            "timestamp": ["timestamp", "time", "date", "datetime", "ts"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c"],
            "volume": ["volume", "vol", "v"],
            "quote_volume": ["quote_volume", "quotevol", "quote_vol", "q"],
        },
        "binance": {
            "timestamp": ["open_time", "Open time"],
            "open": ["open", "Open"],
            "high": ["high", "High"],
            "low": ["low", "Low"],
            "close": ["close", "Close"],
            "volume": ["volume", "Volume"],
            "quote_volume": ["quote_asset_volume", "Quote asset volume"],
            "trades": ["number_of_trades", "Number of trades"],
        },
        "coinbase": {
            "timestamp": ["time", "timestamp"],
            "low": ["low"],
            "high": ["high"],
            "open": ["open"],
            "close": ["close"],
            "volume": ["volume"],
        }
    }
    
    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or ImportConfig()
        self.validator = DataValidator()
    
    async def import_ohlc_csv(
        self,
        file_path: str,
        symbol: str,
        timeframe: str,
        exchange: str = "unknown",
        format_type: str = "standard",
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportProgress:
        """Import OHLC data from CSV file."""
        progress = ImportProgress()
        progress.start_time = datetime.now()
        
        # Count total records
        progress.total_records = await self._count_lines(file_path)
        
        column_mapping = self.COLUMN_MAPPINGS.get(format_type, self.COLUMN_MAPPINGS["standard"])
        
        batch = []
        
        async with aiofiles.open(file_path, mode='r', newline='') as f:
            # Read header
            first_line = await f.readline()
            
            # Detect if has header
            has_header = self.config.csv_has_header
            if has_header:
                headers = first_line.strip().split(self.config.csv_delimiter)
                field_mapping = self._map_fields(headers, column_mapping)
            else:
                # Reset file pointer
                await f.seek(0)
                field_mapping = {k: i for i, k in enumerate(column_mapping.keys())}
            
            # Process records
            async for line in f:
                try:
                    row = line.strip().split(self.config.csv_delimiter)
                    
                    # Parse record
                    record = self._parse_ohlc_row(row, field_mapping, symbol, timeframe, exchange)
                    
                    if record:
                        # Validate
                        if self.config.validate_data:
                            result = self.validator.validate_ohlc(record)
                            if result.is_valid and result.cleaned_data:
                                batch.append(result.cleaned_data)
                                progress.valid_records += 1
                            else:
                                progress.invalid_records += 1
                        else:
                            batch.append(record)
                            progress.valid_records += 1
                    
                    progress.processed_records += 1
                    
                    # Insert batch
                    if len(batch) >= self.config.batch_size:
                        await self._insert_ohlc_batch(batch)
                        batch = []
                    
                    # Progress callback
                    if progress_callback and progress.processed_records % self.config.progress_interval == 0:
                        progress_callback(progress)
                
                except Exception as e:
                    progress.errors.append(str(e))
                    if not self.config.skip_errors:
                        raise
        
        # Insert remaining batch
        if batch:
            await self._insert_ohlc_batch(batch)
        
        progress.end_time = datetime.now()
        
        if progress_callback:
            progress_callback(progress)
        
        logger.info(f"Import complete: {progress.to_dict()}")
        return progress
    
    async def import_trades_csv(
        self,
        file_path: str,
        symbol: str,
        exchange: str = "unknown",
        format_type: str = "standard",
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportProgress:
        """Import trade data from CSV file."""
        progress = ImportProgress()
        progress.start_time = datetime.now()
        
        progress.total_records = await self._count_lines(file_path)
        
        trade_columns = {
            "standard": {
                "timestamp": ["timestamp", "time", "date"],
                "price": ["price", "p"],
                "quantity": ["quantity", "qty", "amount", "size"],
                "side": ["side", "type"],
                "trade_id": ["trade_id", "id", "tid"],
            }
        }
        
        column_mapping = trade_columns.get(format_type, trade_columns["standard"])
        
        batch = []
        
        async with aiofiles.open(file_path, mode='r', newline='') as f:
            first_line = await f.readline()
            
            has_header = self.config.csv_has_header
            if has_header:
                headers = first_line.strip().split(self.config.csv_delimiter)
                field_mapping = self._map_fields(headers, column_mapping)
            else:
                await f.seek(0)
                field_mapping = {k: i for i, k in enumerate(column_mapping.keys())}
            
            async for line in f:
                try:
                    row = line.strip().split(self.config.csv_delimiter)
                    
                    record = self._parse_trade_row(row, field_mapping, symbol, exchange)
                    
                    if record:
                        if self.config.validate_data:
                            result = self.validator.validate_trade(record)
                            if result.is_valid and result.cleaned_data:
                                batch.append(result.cleaned_data)
                                progress.valid_records += 1
                            else:
                                progress.invalid_records += 1
                        else:
                            batch.append(record)
                            progress.valid_records += 1
                    
                    progress.processed_records += 1
                    
                    if len(batch) >= self.config.batch_size:
                        await self._insert_trades_batch(batch)
                        batch = []
                    
                    if progress_callback and progress.processed_records % self.config.progress_interval == 0:
                        progress_callback(progress)
                
                except Exception as e:
                    progress.errors.append(str(e))
                    if not self.config.skip_errors:
                        raise
        
        if batch:
            await self._insert_trades_batch(batch)
        
        progress.end_time = datetime.now()
        
        if progress_callback:
            progress_callback(progress)
        
        logger.info(f"Trade import complete: {progress.to_dict()}")
        return progress
    
    async def _count_lines(self, file_path: str) -> int:
        """Count lines in file."""
        count = 0
        async with aiofiles.open(file_path, mode='r') as f:
            async for _ in f:
                count += 1
        return count - (1 if self.config.csv_has_header else 0)
    
    def _map_fields(
        self,
        headers: List[str],
        column_mapping: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Map CSV headers to field names."""
        mapping = {}
        headers_lower = [h.lower().strip() for h in headers]
        
        for field, possible_names in column_mapping.items():
            for name in possible_names:
                if name.lower() in headers_lower:
                    mapping[field] = headers_lower.index(name.lower())
                    break
        
        return mapping
    
    def _parse_ohlc_row(
        self,
        row: List[str],
        field_mapping: Dict[str, int],
        symbol: str,
        timeframe: str,
        exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Parse OHLC row from CSV."""
        try:
            timestamp = self._parse_timestamp(row[field_mapping.get("timestamp", 0)])
            
            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "open": float(row[field_mapping.get("open", 1)]),
                "high": float(row[field_mapping.get("high", 2)]),
                "low": float(row[field_mapping.get("low", 3)]),
                "close": float(row[field_mapping.get("close", 4)]),
                "volume": float(row[field_mapping.get("volume", 5)]),
                "quote_volume": float(row[field_mapping.get("quote_volume", 6)]) if "quote_volume" in field_mapping else 0,
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing row: {e}")
            return None
    
    def _parse_trade_row(
        self,
        row: List[str],
        field_mapping: Dict[str, int],
        symbol: str,
        exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Parse trade row from CSV."""
        try:
            timestamp = self._parse_timestamp(row[field_mapping.get("timestamp", 0)])
            
            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "exchange": exchange,
                "price": float(row[field_mapping.get("price", 1)]),
                "quantity": float(row[field_mapping.get("quantity", 2)]),
                "side": row[field_mapping.get("side", 3)].lower() if "side" in field_mapping else "buy",
                "trade_id": row[field_mapping.get("trade_id", 4)] if "trade_id" in field_mapping else None,
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing trade row: {e}")
            return None
    
    def _parse_timestamp(self, value: str) -> int:
        """Parse timestamp from various formats."""
        # Try integer timestamp (milliseconds)
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try ISO format
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        
        raise ValueError(f"Cannot parse timestamp: {value}")
    
    async def _insert_ohlc_batch(self, batch: List[Dict[str, Any]]):
        """Insert OHLC batch to database."""
        store = get_store()
        await store.insert_ohlc_batch(batch)
    
    async def _insert_trades_batch(self, batch: List[Dict[str, Any]]):
        """Insert trades batch to database."""
        store = get_store()
        await store.insert_trades(batch)


class APIImporter:
    """Import data from APIs."""
    
    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or ImportConfig()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def import_from_binance_api(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportProgress:
        """Import historical data from Binance API."""
        from ..exchanges.binance_client import create_spot_client
        
        progress = ImportProgress()
        progress.start_time = datetime.now()
        
        client = create_spot_client()
        
        try:
            # Calculate number of requests needed
            interval_seconds = get_timeframe_seconds(timeframe)
            total_seconds = (end_time - start_time).total_seconds()
            expected_candles = int(total_seconds / interval_seconds)
            progress.total_records = expected_candles
            
            current_start = start_time
            all_candles = []
            
            while current_start < end_time:
                # Fetch candles
                start_ms = int(current_start.timestamp() * 1000)
                end_ms = int(min(
                    (current_start + timedelta(days=1)).timestamp() * 1000,
                    end_time.timestamp() * 1000
                ))
                
                candles = await client.rest.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_ms,
                    end_time=end_ms,
                    limit=1000
                )
                
                if not candles:
                    break
                
                # Format candles
                for c in candles:
                    all_candles.append({
                        "timestamp": c[0],
                        "symbol": symbol,
                        "exchange": "binance",
                        "timeframe": timeframe,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                        "quote_volume": float(c[7]),
                        "trades": c[8],
                        "taker_buy_volume": float(c[9]),
                        "taker_buy_quote_volume": float(c[10]),
                    })
                
                progress.processed_records += len(candles)
                progress.valid_records += len(candles)
                
                if progress_callback:
                    progress_callback(progress)
                
                # Move to next batch
                current_start = datetime.fromtimestamp(candles[-1][6] / 1000)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Insert all candles
            if all_candles:
                store = get_store()
                await store.insert_ohlc_batch(all_candles)
            
        finally:
            await client.stop()
        
        progress.end_time = datetime.now()
        
        if progress_callback:
            progress_callback(progress)
        
        return progress
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class HistoricalDataImporter:
    """Main historical data importer."""
    
    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or ImportConfig()
        self.csv_importer = CSVImporter(self.config)
        self.api_importer = APIImporter(self.config)
    
    async def import_csv(
        self,
        file_path: str,
        symbol: str,
        data_type: str = "ohlc",
        timeframe: str = "1m",
        exchange: str = "unknown",
        format_type: str = "standard",
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportProgress:
        """Import data from CSV file."""
        if data_type == "ohlc":
            return await self.csv_importer.import_ohlc_csv(
                file_path, symbol, timeframe, exchange, format_type, progress_callback
            )
        elif data_type == "trades":
            return await self.csv_importer.import_trades_csv(
                file_path, symbol, exchange, format_type, progress_callback
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    async def import_from_api(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportProgress:
        """Import data from API."""
        if source.lower() == "binance":
            return await self.api_importer.import_from_binance_api(
                symbol, timeframe, start_time, end_time, progress_callback
            )
        else:
            raise ValueError(f"Unsupported API source: {source}")
    
    async def close(self):
        """Close importers."""
        await self.api_importer.close()


# Factory functions
def create_importer(config: Optional[ImportConfig] = None) -> HistoricalDataImporter:
    """Create a historical data importer."""
    return HistoricalDataImporter(config)


# Singleton instance
_importer_instance: Optional[HistoricalDataImporter] = None


def get_importer() -> HistoricalDataImporter:
    """Get singleton importer instance."""
    global _importer_instance
    if _importer_instance is None:
        _importer_instance = create_importer()
    return _importer_instance


if __name__ == "__main__":
    async def test_importer():
        """Test importer."""
        importer = create_importer()
        
        # Create test CSV
        test_csv = "/tmp/test_ohlc.csv"
        with open(test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            
            base_time = int(time.time() * 1000)
            for i in range(100):
                writer.writerow([
                    base_time + i * 60000,
                    50000 + i,
                    50100 + i,
                    49900 + i,
                    50050 + i,
                    10 + i
                ])
        
        def on_progress(progress):
            print(f"Progress: {progress.progress_percent:.1f}%")
        
        # Import CSV
        result = await importer.import_csv(
            test_csv,
            symbol="BTCUSDT",
            data_type="ohlc",
            timeframe="1m",
            exchange="test",
            progress_callback=on_progress
        )
        
        print(f"Import result: {result.to_dict()}")
        
        # Cleanup
        os.remove(test_csv)
        await importer.close()
    
    asyncio.run(test_importer())
