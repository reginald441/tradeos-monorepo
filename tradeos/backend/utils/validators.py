"""
TradeOS Input Validation Utilities
Validation functions for user inputs, trading parameters, and more.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple, Union

import validators
from email_validator import EmailNotValidError, validate_email

from config.settings import settings
from database.models import (
    ApiKeyPermission,
    OrderType,
    PositionSide,
    StrategyType,
    SubscriptionTier,
    TimeFrame,
    TradeSide,
    UserRole,
)


# ============================================================================
# Email Validation
# ============================================================================

def validate_email_address(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_email or error_message)
    """
    try:
        info = validate_email(email, check_deliverability=False)
        return True, info.normalized
    except EmailNotValidError as e:
        return False, str(e)


def is_valid_email(email: str) -> bool:
    """
    Quick check if email format is valid.
    
    Args:
        email: Email to check.
    
    Returns:
        bool: True if valid.
    """
    is_valid, _ = validate_email_address(email)
    return is_valid


# ============================================================================
# Password Validation
# ============================================================================

def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    min_length = settings.security.password_min_length
    
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one special character"
    
    # Check for common patterns
    common_patterns = ["password", "123456", "qwerty", "admin", "letmein"]
    if any(pattern in password.lower() for pattern in common_patterns):
        return False, "Password contains common patterns"
    
    return True, None


def is_strong_password(password: str) -> bool:
    """
    Quick check if password is strong.
    
    Args:
        password: Password to check.
    
    Returns:
        bool: True if strong.
    """
    is_valid, _ = validate_password(password)
    return is_valid


# ============================================================================
# Trading Symbol Validation
# ============================================================================

def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Symbol to validate (e.g., 'BTCUSDT', 'BTC/USDT').
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_symbol or error_message)
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    # Remove separators and normalize
    clean = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    
    # Check length
    if len(clean) < 4 or len(clean) > 20:
        return False, "Symbol must be between 4 and 20 characters"
    
    # Check characters (alphanumeric only)
    if not re.match(r"^[A-Z0-9]+$", clean):
        return False, "Symbol must contain only alphanumeric characters"
    
    return True, clean


def is_valid_symbol(symbol: str) -> bool:
    """
    Quick check if symbol format is valid.
    
    Args:
        symbol: Symbol to check.
    
    Returns:
        bool: True if valid.
    """
    is_valid, _ = validate_symbol(symbol)
    return is_valid


def validate_symbols(symbols: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate multiple symbols.
    
    Args:
        symbols: List of symbols to validate.
    
    Returns:
        Tuple[bool, List[str], List[str]]: (all_valid, valid_symbols, invalid_symbols)
    """
    valid = []
    invalid = []
    
    for symbol in symbols:
        is_valid, normalized = validate_symbol(symbol)
        if is_valid:
            valid.append(normalized)
        else:
            invalid.append(symbol)
    
    return len(invalid) == 0, valid, invalid


# ============================================================================
# Price and Quantity Validation
# ============================================================================

def validate_price(price: Union[str, float, Decimal], 
                   min_price: Optional[Decimal] = None,
                   max_price: Optional[Decimal] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate price value.
    
    Args:
        price: Price to validate.
        min_price: Optional minimum price.
        max_price: Optional maximum price.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        price_dec = Decimal(str(price))
    except (InvalidOperation, ValueError):
        return False, "Invalid price format"
    
    if price_dec <= 0:
        return False, "Price must be greater than zero"
    
    if min_price is not None and price_dec < min_price:
        return False, f"Price must be at least {min_price}"
    
    if max_price is not None and price_dec > max_price:
        return False, f"Price must not exceed {max_price}"
    
    return True, None


def validate_quantity(quantity: Union[str, float, Decimal],
                      min_qty: Optional[Decimal] = None,
                      max_qty: Optional[Decimal] = None,
                      step_size: Optional[Decimal] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate quantity value.
    
    Args:
        quantity: Quantity to validate.
        min_qty: Optional minimum quantity.
        max_qty: Optional maximum quantity.
        step_size: Optional step size for validation.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        qty_dec = Decimal(str(quantity))
    except (InvalidOperation, ValueError):
        return False, "Invalid quantity format"
    
    if qty_dec <= 0:
        return False, "Quantity must be greater than zero"
    
    if min_qty is not None and qty_dec < min_qty:
        return False, f"Quantity must be at least {min_qty}"
    
    if max_qty is not None and qty_dec > max_qty:
        return False, f"Quantity must not exceed {max_qty}"
    
    if step_size is not None and step_size > 0:
        remainder = qty_dec % step_size
        if remainder != 0:
            return False, f"Quantity must be a multiple of {step_size}"
    
    return True, None


def validate_notional_value(price: Decimal, 
                            quantity: Decimal,
                            min_notional: Decimal = Decimal("10")) -> Tuple[bool, Optional[str]]:
    """
    Validate notional value of a trade.
    
    Args:
        price: Price per unit.
        quantity: Quantity.
        min_notional: Minimum notional value.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    notional = price * quantity
    
    if notional < min_notional:
        return False, f"Notional value ({notional}) is below minimum ({min_notional})"
    
    return True, None


# ============================================================================
# TimeFrame Validation
# ============================================================================

def validate_timeframe(timeframe: str) -> Tuple[bool, Optional[str]]:
    """
    Validate trading timeframe.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d').
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_timeframe or error_message)
    """
    valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    
    tf = timeframe.lower().strip()
    
    if tf in valid_timeframes:
        return True, tf
    
    return False, f"Invalid timeframe. Valid options: {', '.join(valid_timeframes)}"


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Quick check if timeframe is valid.
    
    Args:
        timeframe: Timeframe to check.
    
    Returns:
        bool: True if valid.
    """
    is_valid, _ = validate_timeframe(timeframe)
    return is_valid


def validate_timeframes(timeframes: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate multiple timeframes.
    
    Args:
        timeframes: List of timeframes to validate.
    
    Returns:
        Tuple[bool, List[str], List[str]]: (all_valid, valid_tfs, invalid_tfs)
    """
    valid = []
    invalid = []
    
    for tf in timeframes:
        is_valid, normalized = validate_timeframe(tf)
        if is_valid:
            valid.append(normalized)
        else:
            invalid.append(tf)
    
    return len(invalid) == 0, valid, invalid


# ============================================================================
# Enum Validation
# ============================================================================

def validate_trade_side(side: str) -> Tuple[bool, Optional[str]]:
    """
    Validate trade side.
    
    Args:
        side: Trade side to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_side or error_message)
    """
    side_lower = side.lower().strip()
    valid_sides = [s.value for s in TradeSide]
    
    if side_lower in valid_sides:
        return True, side_lower
    
    return False, f"Invalid trade side. Valid options: {', '.join(valid_sides)}"


def validate_position_side(side: str) -> Tuple[bool, Optional[str]]:
    """
    Validate position side.
    
    Args:
        side: Position side to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_side or error_message)
    """
    side_lower = side.lower().strip()
    valid_sides = [s.value for s in PositionSide]
    
    if side_lower in valid_sides:
        return True, side_lower
    
    return False, f"Invalid position side. Valid options: {', '.join(valid_sides)}"


def validate_order_type(order_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate order type.
    
    Args:
        order_type: Order type to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_type or error_message)
    """
    ot_lower = order_type.lower().strip()
    valid_types = [t.value for t in OrderType]
    
    if ot_lower in valid_types:
        return True, ot_lower
    
    return False, f"Invalid order type. Valid options: {', '.join(valid_types)}"


def validate_strategy_type(strategy_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate strategy type.
    
    Args:
        strategy_type: Strategy type to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_type or error_message)
    """
    st_lower = strategy_type.lower().strip()
    valid_types = [t.value for t in StrategyType]
    
    if st_lower in valid_types:
        return True, st_lower
    
    return False, f"Invalid strategy type. Valid options: {', '.join(valid_types)}"


def validate_user_role(role: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user role.
    
    Args:
        role: User role to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_role or error_message)
    """
    role_lower = role.lower().strip()
    valid_roles = [r.value for r in UserRole]
    
    if role_lower in valid_roles:
        return True, role_lower
    
    return False, f"Invalid user role. Valid options: {', '.join(valid_roles)}"


def validate_subscription_tier(tier: str) -> Tuple[bool, Optional[str]]:
    """
    Validate subscription tier.
    
    Args:
        tier: Subscription tier to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, normalized_tier or error_message)
    """
    tier_lower = tier.lower().strip()
    valid_tiers = [t.value for t in SubscriptionTier]
    
    if tier_lower in valid_tiers:
        return True, tier_lower
    
    return False, f"Invalid subscription tier. Valid options: {', '.join(valid_tiers)}"


def validate_api_permissions(permissions: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate API key permissions.
    
    Args:
        permissions: List of permissions to validate.
    
    Returns:
        Tuple[bool, List[str], List[str]]: (all_valid, valid_perms, invalid_perms)
    """
    valid_perms = [p.value for p in ApiKeyPermission]
    valid = []
    invalid = []
    
    for perm in permissions:
        perm_lower = perm.lower().strip()
        if perm_lower in valid_perms:
            valid.append(perm_lower)
        else:
            invalid.append(perm)
    
    return len(invalid) == 0, valid, invalid


# ============================================================================
# Risk Parameter Validation
# ============================================================================

def validate_risk_percentage(value: Union[str, float, Decimal],
                             field_name: str = "Risk percentage",
                             min_val: Decimal = Decimal("0.01"),
                             max_val: Decimal = Decimal("100")) -> Tuple[bool, Optional[str]]:
    """
    Validate risk percentage value.
    
    Args:
        value: Risk percentage to validate.
        field_name: Name of the field for error messages.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        val_dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False, f"Invalid {field_name} format"
    
    if val_dec < min_val:
        return False, f"{field_name} must be at least {min_val}%"
    
    if val_dec > max_val:
        return False, f"{field_name} must not exceed {max_val}%"
    
    return True, None


def validate_max_drawdown(value: Union[str, float, Decimal]) -> Tuple[bool, Optional[str]]:
    """
    Validate maximum drawdown percentage.
    
    Args:
        value: Max drawdown to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    return validate_risk_percentage(
        value,
        field_name="Maximum drawdown",
        min_val=Decimal("0.1"),
        max_val=Decimal("100")
    )


def validate_leverage(leverage: int, max_leverage: int = 125) -> Tuple[bool, Optional[str]]:
    """
    Validate leverage value.
    
    Args:
        leverage: Leverage to validate.
        max_leverage: Maximum allowed leverage.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        lev = int(leverage)
    except (ValueError, TypeError):
        return False, "Invalid leverage format"
    
    if lev < 1:
        return False, "Leverage must be at least 1x"
    
    if lev > max_leverage:
        return False, f"Leverage must not exceed {max_leverage}x"
    
    return True, None


# ============================================================================
# String Validation
# ============================================================================

def validate_string_length(value: str,
                           min_length: int = 0,
                           max_length: int = 255,
                           field_name: str = "Value") -> Tuple[bool, Optional[str]]:
    """
    Validate string length.
    
    Args:
        value: String to validate.
        min_length: Minimum length.
        max_length: Maximum length.
        field_name: Name of the field for error messages.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if value is None:
        return False, f"{field_name} cannot be None"
    
    if len(value) < min_length:
        return False, f"{field_name} must be at least {min_length} characters"
    
    if len(value) > max_length:
        return False, f"{field_name} must not exceed {max_length} characters"
    
    return True, None


def validate_alphanumeric(value: str,
                          allow_underscore: bool = True,
                          allow_hyphen: bool = False,
                          field_name: str = "Value") -> Tuple[bool, Optional[str]]:
    """
    Validate alphanumeric string.
    
    Args:
        value: String to validate.
        allow_underscore: Whether to allow underscores.
        allow_hyphen: Whether to allow hyphens.
        field_name: Name of the field for error messages.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    pattern = r"^[a-zA-Z0-9"
    if allow_underscore:
        pattern += "_"
    if allow_hyphen:
        pattern += "-"
    pattern += r"]+$"
    
    if not re.match(pattern, value):
        allowed = "alphanumeric characters"
        if allow_underscore:
            allowed += ", underscores"
        if allow_hyphen:
            allowed += ", hyphens"
        return False, f"{field_name} must contain only {allowed}"
    
    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format.
    
    Args:
        url: URL to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if validators.url(url):
        return True, None
    
    return False, "Invalid URL format"


def validate_uuid(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate UUID format.
    
    Args:
        value: UUID string to validate.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    if re.match(uuid_pattern, value.lower()):
        return True, None
    
    return False, "Invalid UUID format"


# ============================================================================
# Date Range Validation
# ============================================================================

def validate_date_range(start_date, end_date, 
                        max_days: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date.
        end_date: End date.
        max_days: Maximum allowed range in days.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    from datetime import datetime
    
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    if start_date == end_date:
        return False, "Start date and end date cannot be the same"
    
    if max_days is not None:
        delta = (end_date - start_date).days
        if delta > max_days:
            return False, f"Date range must not exceed {max_days} days"
    
    if end_date > datetime.utcnow():
        return False, "End date cannot be in the future"
    
    return True, None


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_strategy_config(config: dict, strategy_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate strategy configuration.
    
    Args:
        config: Strategy configuration dictionary.
        strategy_type: Type of strategy.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Configuration must be a dictionary"
    
    # Common required fields
    if "symbols" not in config or not config["symbols"]:
        return False, "Configuration must include 'symbols'"
    
    if "timeframes" not in config or not config["timeframes"]:
        return False, "Configuration must include 'timeframes'"
    
    # Validate symbols
    for symbol in config["symbols"]:
        is_valid, error = validate_symbol(symbol)
        if not is_valid:
            return False, f"Invalid symbol in config: {error}"
    
    # Validate timeframes
    for tf in config["timeframes"]:
        is_valid, error = validate_timeframe(tf)
        if not is_valid:
            return False, f"Invalid timeframe in config: {error}"
    
    return True, None


# ============================================================================
# Export all validators
# ============================================================================

__all__ = [
    # Email validation
    "validate_email_address",
    "is_valid_email",
    # Password validation
    "validate_password",
    "is_strong_password",
    # Symbol validation
    "validate_symbol",
    "is_valid_symbol",
    "validate_symbols",
    # Price and quantity validation
    "validate_price",
    "validate_quantity",
    "validate_notional_value",
    # Timeframe validation
    "validate_timeframe",
    "is_valid_timeframe",
    "validate_timeframes",
    # Enum validation
    "validate_trade_side",
    "validate_position_side",
    "validate_order_type",
    "validate_strategy_type",
    "validate_user_role",
    "validate_subscription_tier",
    "validate_api_permissions",
    # Risk validation
    "validate_risk_percentage",
    "validate_max_drawdown",
    "validate_leverage",
    # String validation
    "validate_string_length",
    "validate_alphanumeric",
    "validate_url",
    "validate_uuid",
    # Date range validation
    "validate_date_range",
    # Configuration validation
    "validate_strategy_config",
]
