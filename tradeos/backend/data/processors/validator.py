"""
TradeOS Data Validator
Data validation and cleaning pipeline for market data.

Features:
- Schema validation
- Range validation
- Outlier detection
- Data type checking
- Missing data handling
- Duplicate detection
- Statistical validation
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from collections import defaultdict
import statistics
import time

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Reject any invalid data
    MODERATE = "moderate"  # Allow minor issues with warnings
    LENIENT = "lenient"    # Allow most issues, just log


class ValidationErrorType(Enum):
    """Types of validation errors."""
    MISSING_FIELD = "missing_field"
    INVALID_TYPE = "invalid_type"
    OUT_OF_RANGE = "out_of_range"
    NEGATIVE_VALUE = "negative_value"
    ZERO_VALUE = "zero_value"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    TIMESTAMP_INVALID = "timestamp_invalid"
    PRICE_INVALID = "price_invalid"
    QUANTITY_INVALID = "quantity_invalid"


@dataclass
class ValidationError:
    """Validation error details."""
    error_type: ValidationErrorType
    field: str
    message: str
    value: Any = None
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    data: Dict[str, Any]
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    cleaned_data: Optional[Dict[str, Any]] = None
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return any(e.severity == "error" for e in self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


@dataclass
class ValidationRules:
    """Validation rules configuration."""
    # Price rules
    min_price: float = 0.0
    max_price: float = 1e12
    price_precision: int = 8
    
    # Quantity rules
    min_quantity: float = 0.0
    max_quantity: float = 1e12
    quantity_precision: int = 8
    
    # Timestamp rules
    max_timestamp_future_ms: int = 60000  # 1 minute
    max_timestamp_past_ms: int = 86400000  # 1 day
    
    # Outlier detection
    outlier_std_threshold: float = 5.0
    outlier_window_size: int = 100
    
    # Duplicate detection
    duplicate_window_ms: int = 1000
    
    # Validation level
    level: ValidationLevel = ValidationLevel.MODERATE


class DataValidator:
    """Market data validator."""
    
    def __init__(self, rules: Optional[ValidationRules] = None):
        self.rules = rules or ValidationRules()
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._seen_ids: Dict[str, Set[str]] = defaultdict(set)
        self._seen_timestamps: Dict[str, List[int]] = defaultdict(list)
    
    def validate_trade(self, trade: Dict[str, Any]) -> ValidationResult:
        """Validate trade data."""
        errors = []
        warnings = []
        cleaned = trade.copy()
        
        # Required fields
        required = ["symbol", "timestamp", "price", "quantity", "side"]
        for field in required:
            if field not in trade or trade[field] is None:
                errors.append(ValidationError(
                    ValidationErrorType.MISSING_FIELD,
                    field,
                    f"Required field '{field}' is missing"
                ))
        
        if errors:
            return ValidationResult(False, trade, errors, warnings)
        
        # Validate symbol
        if not isinstance(trade.get("symbol"), str):
            errors.append(ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "symbol",
                "Symbol must be a string"
            ))
        
        # Validate timestamp
        ts_result = self._validate_timestamp(trade.get("timestamp"))
        if ts_result:
            errors.append(ts_result)
        else:
            cleaned["timestamp"] = int(trade["timestamp"])
        
        # Validate price
        price_result = self._validate_price(trade.get("price"))
        if price_result:
            errors.append(price_result)
        else:
            cleaned["price"] = float(trade["price"])
        
        # Validate quantity
        qty_result = self._validate_quantity(trade.get("quantity"))
        if qty_result:
            errors.append(qty_result)
        else:
            cleaned["quantity"] = float(trade["quantity"])
        
        # Validate side
        side = trade.get("side", "").lower()
        if side not in ["buy", "sell", "b", "s"]:
            errors.append(ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "side",
                "Side must be 'buy' or 'sell'"
            ))
        else:
            cleaned["side"] = "buy" if side in ["buy", "b"] else "sell"
        
        # Check for outliers
        symbol = trade.get("symbol", "")
        if symbol and "price" in cleaned:
            outlier_error = self._check_outlier(symbol, cleaned["price"])
            if outlier_error:
                if self.rules.level == ValidationLevel.STRICT:
                    errors.append(outlier_error)
                else:
                    warnings.append(outlier_error)
        
        # Check for duplicates
        trade_id = trade.get("trade_id")
        if trade_id:
            dup_error = self._check_duplicate(symbol, str(trade_id))
            if dup_error:
                errors.append(dup_error)
        
        # Update history
        if symbol and "price" in cleaned:
            self._update_price_history(symbol, cleaned["price"])
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            data=trade,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned if is_valid else None
        )
    
    def validate_ohlc(self, ohlc: Dict[str, Any]) -> ValidationResult:
        """Validate OHLC candle data."""
        errors = []
        warnings = []
        cleaned = ohlc.copy()
        
        # Required fields
        required = ["symbol", "timestamp", "timeframe", "open", "high", "low", "close", "volume"]
        for field in required:
            if field not in ohlc or ohlc[field] is None:
                errors.append(ValidationError(
                    ValidationErrorType.MISSING_FIELD,
                    field,
                    f"Required field '{field}' is missing"
                ))
        
        if errors:
            return ValidationResult(False, ohlc, errors, warnings)
        
        # Validate OHLC logic
        o = float(ohlc.get("open", 0))
        h = float(ohlc.get("high", 0))
        l = float(ohlc.get("low", 0))
        c = float(ohlc.get("close", 0))
        
        # High must be >= all other prices
        if h < o or h < l or h < c:
            errors.append(ValidationError(
                ValidationErrorType.PRICE_INVALID,
                "high",
                f"High ({h}) must be >= open ({o}), low ({l}), and close ({c})"
            ))
        
        # Low must be <= all other prices
        if l > o or l > h or l > c:
            errors.append(ValidationError(
                ValidationErrorType.PRICE_INVALID,
                "low",
                f"Low ({l}) must be <= open ({o}), high ({h}), and close ({c})"
            ))
        
        # Validate prices
        for field in ["open", "high", "low", "close"]:
            result = self._validate_price(ohlc.get(field))
            if result:
                errors.append(result)
            else:
                cleaned[field] = float(ohlc[field])
        
        # Validate volume
        vol_result = self._validate_quantity(ohlc.get("volume"))
        if vol_result:
            errors.append(vol_result)
        else:
            cleaned["volume"] = float(ohlc["volume"])
        
        # Validate timestamp
        ts_result = self._validate_timestamp(ohlc.get("timestamp"))
        if ts_result:
            errors.append(ts_result)
        else:
            cleaned["timestamp"] = int(ohlc["timestamp"])
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            data=ohlc,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned if is_valid else None
        )
    
    def validate_orderbook(self, orderbook: Dict[str, Any]) -> ValidationResult:
        """Validate orderbook data."""
        errors = []
        warnings = []
        cleaned = orderbook.copy()
        
        # Required fields
        required = ["symbol", "bids", "asks"]
        for field in required:
            if field not in orderbook:
                errors.append(ValidationError(
                    ValidationErrorType.MISSING_FIELD,
                    field,
                    f"Required field '{field}' is missing"
                ))
        
        if errors:
            return ValidationResult(False, orderbook, errors, warnings)
        
        # Validate bids and asks
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not isinstance(bids, list):
            errors.append(ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "bids",
                "Bids must be a list"
            ))
        
        if not isinstance(asks, list):
            errors.append(ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "asks",
                "Asks must be a list"
            ))
        
        # Validate bid/ask format and check spread
        if isinstance(bids, list) and isinstance(asks, list):
            if bids and asks:
                try:
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    
                    if best_bid >= best_ask:
                        errors.append(ValidationError(
                            ValidationErrorType.PRICE_INVALID,
                            "spread",
                            f"Best bid ({best_bid}) >= best ask ({best_ask})"
                        ))
                    
                    # Check for crossed book
                    for bid in bids:
                        for ask in asks:
                            if float(bid[0]) >= float(ask[0]):
                                warnings.append(ValidationError(
                                    ValidationErrorType.PRICE_INVALID,
                                    "crossed_book",
                                    f"Crossed book: bid {bid[0]} >= ask {ask[0]}",
                                    severity="warning"
                                ))
                                break
                
                except (IndexError, ValueError) as e:
                    errors.append(ValidationError(
                        ValidationErrorType.INVALID_TYPE,
                        "orderbook",
                        f"Invalid orderbook format: {e}"
                    ))
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            data=orderbook,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned if is_valid else None
        )
    
    def validate_ticker(self, ticker: Dict[str, Any]) -> ValidationResult:
        """Validate ticker data."""
        errors = []
        warnings = []
        cleaned = ticker.copy()
        
        # Required fields
        if "price" not in ticker and "close" not in ticker:
            errors.append(ValidationError(
                ValidationErrorType.MISSING_FIELD,
                "price",
                "Either 'price' or 'close' is required"
            ))
        
        if errors:
            return ValidationResult(False, ticker, errors, warnings)
        
        # Validate price
        price = ticker.get("price") or ticker.get("close")
        price_result = self._validate_price(price)
        if price_result:
            errors.append(price_result)
        else:
            cleaned["price"] = float(price)
        
        # Validate optional fields
        for field in ["open", "high", "low", "volume", "bid", "ask"]:
            if field in ticker and ticker[field] is not None:
                try:
                    val = float(ticker[field])
                    if val < 0:
                        warnings.append(ValidationError(
                            ValidationErrorType.NEGATIVE_VALUE,
                            field,
                            f"{field} is negative: {val}",
                            severity="warning"
                        ))
                    cleaned[field] = val
                except (ValueError, TypeError):
                    warnings.append(ValidationError(
                        ValidationErrorType.INVALID_TYPE,
                        field,
                        f"{field} is not a valid number",
                        severity="warning"
                    ))
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            data=ticker,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned if is_valid else None
        )
    
    def _validate_timestamp(self, timestamp: Any) -> Optional[ValidationError]:
        """Validate timestamp."""
        if timestamp is None:
            return ValidationError(
                ValidationErrorType.MISSING_FIELD,
                "timestamp",
                "Timestamp is required"
            )
        
        try:
            ts = int(timestamp)
        except (ValueError, TypeError):
            return ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "timestamp",
                "Timestamp must be an integer"
            )
        
        now = int(time.time() * 1000)
        
        # Check if timestamp is in the future
        if ts > now + self.rules.max_timestamp_future_ms:
            return ValidationError(
                ValidationErrorType.TIMESTAMP_INVALID,
                "timestamp",
                f"Timestamp is in the future: {ts} > {now}"
            )
        
        # Check if timestamp is too old
        if ts < now - self.rules.max_timestamp_past_ms:
            return ValidationError(
                ValidationErrorType.TIMESTAMP_INVALID,
                "timestamp",
                f"Timestamp is too old: {ts} < {now - self.rules.max_timestamp_past_ms}"
            )
        
        return None
    
    def _validate_price(self, price: Any) -> Optional[ValidationError]:
        """Validate price."""
        if price is None:
            return ValidationError(
                ValidationErrorType.MISSING_FIELD,
                "price",
                "Price is required"
            )
        
        try:
            p = float(price)
        except (ValueError, TypeError):
            return ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "price",
                "Price must be a number"
            )
        
        if p < 0:
            return ValidationError(
                ValidationErrorType.NEGATIVE_VALUE,
                "price",
                f"Price is negative: {p}"
            )
        
        if p == 0:
            return ValidationError(
                ValidationErrorType.ZERO_VALUE,
                "price",
                "Price is zero"
            )
        
        if p < self.rules.min_price:
            return ValidationError(
                ValidationErrorType.OUT_OF_RANGE,
                "price",
                f"Price below minimum: {p} < {self.rules.min_price}"
            )
        
        if p > self.rules.max_price:
            return ValidationError(
                ValidationErrorType.OUT_OF_RANGE,
                "price",
                f"Price above maximum: {p} > {self.rules.max_price}"
            )
        
        return None
    
    def _validate_quantity(self, quantity: Any) -> Optional[ValidationError]:
        """Validate quantity."""
        if quantity is None:
            return ValidationError(
                ValidationErrorType.MISSING_FIELD,
                "quantity",
                "Quantity is required"
            )
        
        try:
            q = float(quantity)
        except (ValueError, TypeError):
            return ValidationError(
                ValidationErrorType.INVALID_TYPE,
                "quantity",
                "Quantity must be a number"
            )
        
        if q < 0:
            return ValidationError(
                ValidationErrorType.NEGATIVE_VALUE,
                "quantity",
                f"Quantity is negative: {q}"
            )
        
        if q > self.rules.max_quantity:
            return ValidationError(
                ValidationErrorType.OUT_OF_RANGE,
                "quantity",
                f"Quantity above maximum: {q} > {self.rules.max_quantity}"
            )
        
        return None
    
    def _check_outlier(self, symbol: str, price: float) -> Optional[ValidationError]:
        """Check if price is an outlier."""
        history = self._price_history.get(symbol, [])
        
        if len(history) < 10:
            return None
        
        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 0
        
        if std == 0:
            return None
        
        z_score = abs(price - mean) / std
        
        if z_score > self.rules.outlier_std_threshold:
            return ValidationError(
                ValidationErrorType.OUTLIER,
                "price",
                f"Price {price} is an outlier (z-score: {z_score:.2f})",
                value=price,
                severity="warning"
            )
        
        return None
    
    def _check_duplicate(self, symbol: str, trade_id: str) -> Optional[ValidationError]:
        """Check for duplicate trade ID."""
        if symbol not in self._seen_ids:
            self._seen_ids[symbol] = set()
        
        if trade_id in self._seen_ids[symbol]:
            return ValidationError(
                ValidationErrorType.DUPLICATE,
                "trade_id",
                f"Duplicate trade ID: {trade_id}",
                severity="error"
            )
        
        self._seen_ids[symbol].add(trade_id)
        
        # Limit set size
        if len(self._seen_ids[symbol]) > 10000:
            self._seen_ids[symbol] = set(list(self._seen_ids[symbol])[-5000:])
        
        return None
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for outlier detection."""
        self._price_history[symbol].append(price)
        
        # Limit history size
        if len(self._price_history[symbol]) > self.rules.outlier_window_size:
            self._price_history[symbol] = self._price_history[symbol][-self.rules.outlier_window_size:]
    
    def clear_history(self, symbol: Optional[str] = None):
        """Clear price history."""
        if symbol:
            self._price_history.pop(symbol, None)
            self._seen_ids.pop(symbol, None)
        else:
            self._price_history.clear()
            self._seen_ids.clear()


class ValidationPipeline:
    """Pipeline for processing and validating data streams."""
    
    def __init__(self, rules: Optional[ValidationRules] = None):
        self.validator = DataValidator(rules)
        self._processors: List[Callable[[Dict], Dict]] = []
        self._error_handlers: List[Callable[[ValidationResult], None]] = []
        self._stats = {
            "processed": 0,
            "valid": 0,
            "invalid": 0,
            "errors": defaultdict(int),
        }
    
    def add_processor(self, processor: Callable[[Dict], Dict]):
        """Add a data processor to the pipeline."""
        self._processors.append(processor)
    
    def add_error_handler(self, handler: Callable[[ValidationResult], None]):
        """Add an error handler."""
        self._error_handlers.append(handler)
    
    def process_trade(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate a trade through the pipeline."""
        self._stats["processed"] += 1
        
        # Run pre-processors
        data = trade.copy()
        for processor in self._processors:
            try:
                data = processor(data)
            except Exception as e:
                logger.error(f"Processor error: {e}")
        
        # Validate
        result = self.validator.validate_trade(data)
        
        if result.is_valid:
            self._stats["valid"] += 1
            return result.cleaned_data
        else:
            self._stats["invalid"] += 1
            for error in result.errors:
                self._stats["errors"][error.error_type.value] += 1
            
            # Call error handlers
            for handler in self._error_handlers:
                try:
                    handler(result)
                except Exception as e:
                    logger.error(f"Error handler error: {e}")
            
            return None
    
    def process_ohlc(self, ohlc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate OHLC through the pipeline."""
        self._stats["processed"] += 1
        
        result = self.validator.validate_ohlc(ohlc)
        
        if result.is_valid:
            self._stats["valid"] += 1
            return result.cleaned_data
        else:
            self._stats["invalid"] += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "error_rate": self._stats["invalid"] / max(self._stats["processed"], 1),
            "valid_rate": self._stats["valid"] / max(self._stats["processed"], 1),
        }


# Factory functions
def create_validator(rules: Optional[ValidationRules] = None) -> DataValidator:
    """Create a data validator."""
    return DataValidator(rules)


def create_pipeline(rules: Optional[ValidationRules] = None) -> ValidationPipeline:
    """Create a validation pipeline."""
    return ValidationPipeline(rules)


# Singleton instance
_validator_instance: Optional[DataValidator] = None


def get_validator() -> DataValidator:
    """Get singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = create_validator()
    return _validator_instance


if __name__ == "__main__":
    def test_validator():
        """Test data validator."""
        validator = create_validator()
        
        # Valid trade
        valid_trade = {
            "symbol": "BTCUSDT",
            "timestamp": int(time.time() * 1000),
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
            "trade_id": "12345"
        }
        
        result = validator.validate_trade(valid_trade)
        print(f"Valid trade result: {result.is_valid}, errors: {len(result.errors)}")
        
        # Invalid trade (negative price)
        invalid_trade = {
            "symbol": "BTCUSDT",
            "timestamp": int(time.time() * 1000),
            "price": -100,
            "quantity": 0.1,
            "side": "buy"
        }
        
        result = validator.validate_trade(invalid_trade)
        print(f"Invalid trade result: {result.is_valid}, errors: {[e.message for e in result.errors]}")
        
        # Invalid OHLC (high < low)
        invalid_ohlc = {
            "symbol": "BTCUSDT",
            "timestamp": int(time.time() * 1000),
            "timeframe": "1m",
            "open": 50000,
            "high": 49900,  # Invalid: high < low
            "low": 50100,
            "close": 50050,
            "volume": 10
        }
        
        result = validator.validate_ohlc(invalid_ohlc)
        print(f"Invalid OHLC result: {result.is_valid}, errors: {[e.message for e in result.errors]}")
    
    test_validator()
