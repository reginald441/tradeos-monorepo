"""
TradeOS Utility Helpers
Common utility functions for datetime operations, math helpers, and more.
"""

import decimal
import hashlib
import json
import math
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union

import pytz


# ============================================================================
# DateTime Helpers
# ============================================================================

def utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        datetime: Current UTC datetime with timezone info.
    """
    return datetime.now(timezone.utc)


def utc_timestamp() -> float:
    """
    Get current UTC timestamp.
    
    Returns:
        float: Current UTC timestamp.
    """
    return utc_now().timestamp()


def utc_timestamp_ms() -> int:
    """
    Get current UTC timestamp in milliseconds.
    
    Returns:
        int: Current UTC timestamp in milliseconds.
    """
    return int(utc_timestamp() * 1000)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.
    
    Args:
        dt: Datetime to convert.
    
    Returns:
        datetime: UTC datetime.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def from_timestamp(timestamp: Union[int, float], unit: str = "s") -> datetime:
    """
    Convert timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp.
        unit: Time unit ('s' for seconds, 'ms' for milliseconds, 'us' for microseconds).
    
    Returns:
        datetime: UTC datetime.
    """
    if unit == "ms":
        timestamp = timestamp / 1000
    elif unit == "us":
        timestamp = timestamp / 1_000_000
    
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def to_timestamp(dt: datetime, unit: str = "s") -> Union[int, float]:
    """
    Convert datetime to timestamp.
    
    Args:
        dt: Datetime to convert.
        unit: Time unit ('s' for seconds, 'ms' for milliseconds, 'us' for microseconds).
    
    Returns:
        Union[int, float]: Unix timestamp.
    """
    ts = dt.timestamp()
    if unit == "ms":
        return int(ts * 1000)
    elif unit == "us":
        return int(ts * 1_000_000)
    return ts


def floor_datetime(dt: datetime, interval: timedelta) -> datetime:
    """
    Floor datetime to the nearest interval.
    
    Args:
        dt: Datetime to floor.
        interval: Interval to floor to.
    
    Returns:
        datetime: Floored datetime.
    """
    seconds = int(interval.total_seconds())
    timestamp = int(dt.timestamp())
    floored_timestamp = (timestamp // seconds) * seconds
    return datetime.fromtimestamp(floored_timestamp, tz=dt.tzinfo or timezone.utc)


def ceil_datetime(dt: datetime, interval: timedelta) -> datetime:
    """
    Ceil datetime to the nearest interval.
    
    Args:
        dt: Datetime to ceil.
        interval: Interval to ceil to.
    
    Returns:
        datetime: Ceiled datetime.
    """
    seconds = int(interval.total_seconds())
    timestamp = int(dt.timestamp())
    ceiled_timestamp = ((timestamp + seconds - 1) // seconds) * seconds
    return datetime.fromtimestamp(ceiled_timestamp, tz=dt.tzinfo or timezone.utc)


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Args:
        dt: Past datetime.
    
    Returns:
        str: Human-readable time ago (e.g., "2 hours ago").
    """
    now = utc_now()
    diff = now - to_utc(dt)
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    else:
        months = int(seconds / 2592000)
        return f"{months} month{'s' if months > 1 else ''} ago"


def parse_datetime(date_string: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """
    Parse datetime from string with multiple format support.
    
    Args:
        date_string: Date string to parse.
        formats: List of formats to try.
    
    Returns:
        Optional[datetime]: Parsed datetime or None if parsing fails.
    """
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
        ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    return None


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime to format.
        fmt: Format string.
    
    Returns:
        str: Formatted datetime string.
    """
    return dt.strftime(fmt)


def get_timeframe_delta(timeframe: str) -> timedelta:
    """
    Get timedelta for a trading timeframe.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d').
    
    Returns:
        timedelta: Corresponding timedelta.
    """
    multipliers = {
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "M": 2592000,
    }
    
    match = re.match(r"(\d+)([mhdwM])", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    return timedelta(seconds=value * multipliers[unit])


# ============================================================================
# Math Helpers
# ============================================================================

def round_decimal(value: Union[str, float, Decimal], precision: int = 8) -> Decimal:
    """
    Round a value to specified decimal precision.
    
    Args:
        value: Value to round.
        precision: Number of decimal places.
    
    Returns:
        Decimal: Rounded decimal value.
    """
    if isinstance(value, str):
        value = Decimal(value)
    elif isinstance(value, float):
        value = Decimal(str(value))
    
    quantize_str = "0." + "0" * precision
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def safe_divide(numerator: Union[int, float, Decimal], 
                denominator: Union[int, float, Decimal], 
                default: Union[int, float, Decimal] = 0) -> Decimal:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator.
        denominator: Denominator.
        default: Default value if division by zero.
    
    Returns:
        Decimal: Division result or default.
    """
    try:
        if Decimal(str(denominator)) == 0:
            return Decimal(str(default))
        return Decimal(str(numerator)) / Decimal(str(denominator))
    except (decimal.InvalidOperation, ValueError):
        return Decimal(str(default))


def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value.
        new_value: New value.
    
    Returns:
        Decimal: Percentage change.
    """
    if old_value == 0:
        return Decimal("0")
    return ((new_value - old_value) / old_value) * 100


def calculate_pnl(entry_price: Decimal, exit_price: Decimal, 
                  quantity: Decimal, side: str = "long") -> Decimal:
    """
    Calculate profit/loss for a trade.
    
    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Position quantity.
        side: Position side ('long' or 'short').
    
    Returns:
        Decimal: Profit/loss amount.
    """
    price_diff = exit_price - entry_price
    
    if side.lower() == "short":
        price_diff = -price_diff
    
    return price_diff * quantity


def calculate_pnl_pct(entry_price: Decimal, exit_price: Decimal, 
                      side: str = "long", leverage: int = 1) -> Decimal:
    """
    Calculate profit/loss percentage for a trade.
    
    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        side: Position side ('long' or 'short').
        leverage: Position leverage.
    
    Returns:
        Decimal: Profit/loss percentage.
    """
    if entry_price == 0:
        return Decimal("0")
    
    price_diff = exit_price - entry_price
    
    if side.lower() == "short":
        price_diff = -price_diff
    
    return (price_diff / entry_price) * 100 * leverage


def calculate_average_price(quantities: List[Decimal], prices: List[Decimal]) -> Decimal:
    """
    Calculate volume-weighted average price.
    
    Args:
        quantities: List of quantities.
        prices: List of prices.
    
    Returns:
        Decimal: Volume-weighted average price.
    """
    if len(quantities) != len(prices) or len(quantities) == 0:
        return Decimal("0")
    
    total_value = sum(q * p for q, p in zip(quantities, prices))
    total_quantity = sum(quantities)
    
    if total_quantity == 0:
        return Decimal("0")
    
    return total_value / total_quantity


def calculate_sharpe_ratio(returns: List[Decimal], 
                           risk_free_rate: Decimal = Decimal("0")) -> Decimal:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: List of returns.
        risk_free_rate: Risk-free rate.
    
    Returns:
        Decimal: Sharpe ratio.
    """
    if len(returns) < 2:
        return Decimal("0")
    
    returns_decimal = [Decimal(str(r)) for r in returns]
    excess_returns = [r - risk_free_rate for r in returns_decimal]
    
    mean_return = sum(excess_returns) / len(excess_returns)
    
    variance = sum((r - mean_return) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
    std_dev = variance.sqrt()
    
    if std_dev == 0:
        return Decimal("0")
    
    return mean_return / std_dev


def calculate_drawdown(equity_curve: List[Decimal]) -> Tuple[Decimal, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values.
    
    Returns:
        Tuple[Decimal, int, int]: (max_drawdown, start_index, end_index)
    """
    if len(equity_curve) < 2:
        return Decimal("0"), 0, 0
    
    max_drawdown = Decimal("0")
    peak = equity_curve[0]
    peak_index = 0
    dd_start = 0
    dd_end = 0
    
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_index = i
        
        drawdown = (peak - value) / peak if peak > 0 else Decimal("0")
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            dd_start = peak_index
            dd_end = i
    
    return max_drawdown, dd_start, dd_end


def truncate_decimal(value: Decimal, precision: int) -> Decimal:
    """
    Truncate decimal to specified precision (not round).
    
    Args:
        value: Value to truncate.
        precision: Number of decimal places.
    
    Returns:
        Decimal: Truncated value.
    """
    str_value = str(value)
    if "." in str_value:
        integer_part, decimal_part = str_value.split(".")
        truncated = f"{integer_part}.{decimal_part[:precision]}"
        return Decimal(truncated)
    return value


# ============================================================================
# String Helpers
# ============================================================================

def generate_uuid() -> str:
    """
    Generate a UUID4 string.
    
    Returns:
        str: UUID4 string.
    """
    return str(uuid.uuid4())


def generate_short_id(length: int = 12) -> str:
    """
    Generate a short unique identifier.
    
    Args:
        length: Length of the identifier.
    
    Returns:
        str: Short unique identifier.
    """
    import secrets
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def hash_string(input_string: str, algorithm: str = "sha256") -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        input_string: String to hash.
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5').
    
    Returns:
        str: Hashed string.
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_string.encode("utf-8"))
    return hash_obj.hexdigest()


def mask_string(input_string: str, visible_chars: int = 4, mask_char: str = "*") -> str:
    """
    Mask a string showing only first and last visible_chars.
    
    Args:
        input_string: String to mask.
        visible_chars: Number of visible characters at start and end.
        mask_char: Character to use for masking.
    
    Returns:
        str: Masked string.
    """
    if len(input_string) <= visible_chars * 2:
        return mask_char * len(input_string)
    
    return input_string[:visible_chars] + mask_char * (len(input_string) - visible_chars * 2) + input_string[-visible_chars:]


def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify.
    
    Returns:
        str: URL-friendly slug.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-")


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to max_length with suffix.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.
    
    Returns:
        str: Truncated string.
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# Data Helpers
# ============================================================================

def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely load JSON string, returning default on error.
    
    Args:
        data: JSON string.
        default: Default value if parsing fails.
    
    Returns:
        Any: Parsed JSON or default.
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Safely dump data to JSON string, returning default on error.
    
    Args:
        data: Data to serialize.
        default: Default value if serialization fails.
    
    Returns:
        str: JSON string or default.
    """
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        input_list: List to chunk.
        chunk_size: Size of each chunk.
    
    Returns:
        List[List[Any]]: List of chunks.
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten.
    
    Returns:
        List[Any]: Flattened list.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary.
        dict2: Dictionary to merge.
    
    Returns:
        Dict: Merged dictionary.
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def remove_none_values(data: Dict) -> Dict:
    """
    Remove None values from dictionary.
    
    Args:
        data: Dictionary to clean.
    
    Returns:
        Dict: Dictionary without None values.
    """
    return {k: v for k, v in data.items() if v is not None}


# ============================================================================
# Trading Helpers
# ============================================================================

def format_symbol(symbol: str, separator: str = "") -> str:
    """
    Format trading symbol consistently.
    
    Args:
        symbol: Symbol to format.
        separator: Separator to use (e.g., '/', '-', '').
    
    Returns:
        str: Formatted symbol.
    """
    # Remove existing separators
    clean = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    
    # Common base-quote pairs
    common_quotes = ["USDT", "USD", "BTC", "ETH", "BNB", "USDC", "BUSD"]
    
    for quote in common_quotes:
        if clean.endswith(quote):
            base = clean[:-len(quote)]
            if separator:
                return f"{base}{separator}{quote}"
            return clean
    
    return clean


def parse_symbol(symbol: str) -> Tuple[str, str]:
    """
    Parse symbol into base and quote assets.
    
    Args:
        symbol: Trading symbol.
    
    Returns:
        Tuple[str, str]: (base, quote)
    """
    clean = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    
    common_quotes = ["USDT", "USD", "BTC", "ETH", "BNB", "USDC", "BUSD", "TUSD"]
    
    for quote in common_quotes:
        if clean.endswith(quote):
            base = clean[:-len(quote)]
            return base, quote
    
    # Default: split in half (for 6-8 char symbols like BTCUSDT)
    mid = len(clean) // 2
    return clean[:mid], clean[mid:]


def calculate_position_size(
    account_balance: Decimal,
    risk_percent: Decimal,
    entry_price: Decimal,
    stop_loss: Decimal
) -> Decimal:
    """
    Calculate position size based on risk parameters.
    
    Args:
        account_balance: Account balance.
        risk_percent: Risk percentage per trade.
        entry_price: Entry price.
        stop_loss: Stop loss price.
    
    Returns:
        Decimal: Position size.
    """
    if entry_price == stop_loss:
        return Decimal("0")
    
    risk_amount = account_balance * (risk_percent / 100)
    price_risk = abs(entry_price - stop_loss)
    
    return risk_amount / price_risk


def calculate_liquidation_price(
    entry_price: Decimal,
    margin: Decimal,
    position_size: Decimal,
    side: str = "long",
    maintenance_margin: Decimal = Decimal("0.005")
) -> Decimal:
    """
    Calculate liquidation price for leveraged position.
    
    Args:
        entry_price: Entry price.
        margin: Margin amount.
        position_size: Position size.
        side: Position side ('long' or 'short').
        maintenance_margin: Maintenance margin rate.
    
    Returns:
        Decimal: Liquidation price.
    """
    if position_size == 0:
        return Decimal("0")
    
    leverage = (entry_price * position_size) / margin
    
    if side.lower() == "long":
        liq_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
    else:
        liq_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
    
    return max(Decimal("0"), liq_price)


# ============================================================================
# Export all helpers
# ============================================================================

__all__ = [
    # DateTime helpers
    "utc_now",
    "utc_timestamp",
    "utc_timestamp_ms",
    "to_utc",
    "from_timestamp",
    "to_timestamp",
    "floor_datetime",
    "ceil_datetime",
    "time_ago",
    "parse_datetime",
    "format_datetime",
    "get_timeframe_delta",
    # Math helpers
    "round_decimal",
    "safe_divide",
    "calculate_percentage_change",
    "calculate_pnl",
    "calculate_pnl_pct",
    "calculate_average_price",
    "calculate_sharpe_ratio",
    "calculate_drawdown",
    "truncate_decimal",
    # String helpers
    "generate_uuid",
    "generate_short_id",
    "hash_string",
    "mask_string",
    "slugify",
    "truncate_string",
    # Data helpers
    "safe_json_loads",
    "safe_json_dumps",
    "chunk_list",
    "flatten_list",
    "deep_merge",
    "remove_none_values",
    # Trading helpers
    "format_symbol",
    "parse_symbol",
    "calculate_position_size",
    "calculate_liquidation_price",
]
