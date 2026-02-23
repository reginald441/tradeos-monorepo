"""
Risk Limits Configuration for TradeOS

Default risk limits by subscription tier.
All limits are designed to protect capital while allowing
appropriate trading flexibility based on user tier.
"""

from decimal import Decimal
from typing import Dict
from ..models.risk_profile import RiskLimits, SubscriptionTier


# ============================================================================
# FREE TIER - Most Restrictive
# ============================================================================
FREE_TIER_LIMITS = RiskLimits(
    # Position sizing - very conservative
    max_position_size_pct=Decimal("0.05"),      # 5% max per position
    max_risk_per_trade_pct=Decimal("0.01"),     # 1% risk per trade
    min_position_size=Decimal("10.0"),
    
    # Portfolio exposure - limited
    max_total_exposure_pct=Decimal("0.50"),     # 50% total exposure
    max_single_asset_exposure_pct=Decimal("0.10"),  # 10% per asset
    max_sector_exposure_pct=Decimal("0.25"),    # 25% per sector
    max_leverage=Decimal("1.0"),                # No leverage
    max_margin_usage_pct=Decimal("0.50"),       # 50% margin
    
    # Drawdown limits - strict
    max_daily_loss_pct=Decimal("0.03"),         # 3% daily loss
    max_weekly_loss_pct=Decimal("0.07"),        # 7% weekly loss
    max_drawdown_pct=Decimal("0.10"),           # 10% max drawdown
    circuit_breaker_drawdown_pct=Decimal("0.15"),  # 15% hard stop
    
    # Correlation limits
    max_correlation_exposure=Decimal("0.60"),
    max_portfolio_heat=Decimal("0.30"),
    
    # VaR limits
    max_daily_var_pct=Decimal("0.03"),
    var_confidence_level=Decimal("0.95"),
    
    # Kill switch triggers - very sensitive
    kill_switch_daily_loss_pct=Decimal("0.05"),
    kill_switch_drawdown_pct=Decimal("0.20"),
    
    # Trading halts
    trading_halt_cooldown_minutes=60,
    auto_resume_after_halt=False,
)

# ============================================================================
# BASIC TIER - Moderate Restrictions
# ============================================================================
BASIC_TIER_LIMITS = RiskLimits(
    # Position sizing
    max_position_size_pct=Decimal("0.08"),
    max_risk_per_trade_pct=Decimal("0.015"),
    min_position_size=Decimal("10.0"),
    
    # Portfolio exposure
    max_total_exposure_pct=Decimal("0.75"),
    max_single_asset_exposure_pct=Decimal("0.15"),
    max_sector_exposure_pct=Decimal("0.35"),
    max_leverage=Decimal("1.5"),
    max_margin_usage_pct=Decimal("0.60"),
    
    # Drawdown limits
    max_daily_loss_pct=Decimal("0.04"),
    max_weekly_loss_pct=Decimal("0.08"),
    max_drawdown_pct=Decimal("0.15"),
    circuit_breaker_drawdown_pct=Decimal("0.20"),
    
    # Correlation limits
    max_correlation_exposure=Decimal("0.65"),
    max_portfolio_heat=Decimal("0.40"),
    
    # VaR limits
    max_daily_var_pct=Decimal("0.04"),
    var_confidence_level=Decimal("0.95"),
    
    # Kill switch triggers
    kill_switch_daily_loss_pct=Decimal("0.07"),
    kill_switch_drawdown_pct=Decimal("0.25"),
    
    # Trading halts
    trading_halt_cooldown_minutes=45,
    auto_resume_after_halt=False,
)

# ============================================================================
# PRO TIER - Standard Professional Limits
# ============================================================================
PRO_TIER_LIMITS = RiskLimits(
    # Position sizing
    max_position_size_pct=Decimal("0.12"),
    max_risk_per_trade_pct=Decimal("0.02"),
    min_position_size=Decimal("10.0"),
    
    # Portfolio exposure
    max_total_exposure_pct=Decimal("1.0"),
    max_single_asset_exposure_pct=Decimal("0.20"),
    max_sector_exposure_pct=Decimal("0.40"),
    max_leverage=Decimal("2.0"),
    max_margin_usage_pct=Decimal("0.70"),
    
    # Drawdown limits
    max_daily_loss_pct=Decimal("0.05"),
    max_weekly_loss_pct=Decimal("0.10"),
    max_drawdown_pct=Decimal("0.20"),
    circuit_breaker_drawdown_pct=Decimal("0.25"),
    
    # Correlation limits
    max_correlation_exposure=Decimal("0.70"),
    max_portfolio_heat=Decimal("0.50"),
    
    # VaR limits
    max_daily_var_pct=Decimal("0.05"),
    var_confidence_level=Decimal("0.95"),
    
    # Kill switch triggers
    kill_switch_daily_loss_pct=Decimal("0.10"),
    kill_switch_drawdown_pct=Decimal("0.30"),
    
    # Trading halts
    trading_halt_cooldown_minutes=30,
    auto_resume_after_halt=True,
)

# ============================================================================
# ENTERPRISE TIER - Advanced Limits
# ============================================================================
ENTERPRISE_TIER_LIMITS = RiskLimits(
    # Position sizing
    max_position_size_pct=Decimal("0.15"),
    max_risk_per_trade_pct=Decimal("0.025"),
    min_position_size=Decimal("10.0"),
    
    # Portfolio exposure
    max_total_exposure_pct=Decimal("1.25"),
    max_single_asset_exposure_pct=Decimal("0.25"),
    max_sector_exposure_pct=Decimal("0.50"),
    max_leverage=Decimal("3.0"),
    max_margin_usage_pct=Decimal("0.75"),
    
    # Drawdown limits
    max_daily_loss_pct=Decimal("0.06"),
    max_weekly_loss_pct=Decimal("0.12"),
    max_drawdown_pct=Decimal("0.25"),
    circuit_breaker_drawdown_pct=Decimal("0.30"),
    
    # Correlation limits
    max_correlation_exposure=Decimal("0.75"),
    max_portfolio_heat=Decimal("0.60"),
    
    # VaR limits
    max_daily_var_pct=Decimal("0.06"),
    var_confidence_level=Decimal("0.99"),
    
    # Kill switch triggers
    kill_switch_daily_loss_pct=Decimal("0.12"),
    kill_switch_drawdown_pct=Decimal("0.35"),
    
    # Trading halts
    trading_halt_cooldown_minutes=20,
    auto_resume_after_halt=True,
)

# ============================================================================
# INSTITUTIONAL TIER - Maximum Flexibility
# ============================================================================
INSTITUTIONAL_TIER_LIMITS = RiskLimits(
    # Position sizing
    max_position_size_pct=Decimal("0.20"),
    max_risk_per_trade_pct=Decimal("0.03"),
    min_position_size=Decimal("100.0"),
    
    # Portfolio exposure
    max_total_exposure_pct=Decimal("2.0"),
    max_single_asset_exposure_pct=Decimal("0.30"),
    max_sector_exposure_pct=Decimal("0.60"),
    max_leverage=Decimal("5.0"),
    max_margin_usage_pct=Decimal("0.85"),
    
    # Drawdown limits
    max_daily_loss_pct=Decimal("0.08"),
    max_weekly_loss_pct=Decimal("0.15"),
    max_drawdown_pct=Decimal("0.30"),
    circuit_breaker_drawdown_pct=Decimal("0.40"),
    
    # Correlation limits
    max_correlation_exposure=Decimal("0.80"),
    max_portfolio_heat=Decimal("0.70"),
    
    # VaR limits
    max_daily_var_pct=Decimal("0.08"),
    var_confidence_level=Decimal("0.99"),
    
    # Kill switch triggers
    kill_switch_daily_loss_pct=Decimal("0.15"),
    kill_switch_drawdown_pct=Decimal("0.45"),
    
    # Trading halts
    trading_halt_cooldown_minutes=15,
    auto_resume_after_halt=True,
)


# ============================================================================
# TIER MAPPING
# ============================================================================
TIER_LIMITS: Dict[SubscriptionTier, RiskLimits] = {
    SubscriptionTier.FREE: FREE_TIER_LIMITS,
    SubscriptionTier.BASIC: BASIC_TIER_LIMITS,
    SubscriptionTier.PRO: PRO_TIER_LIMITS,
    SubscriptionTier.ENTERPRISE: ENTERPRISE_TIER_LIMITS,
    SubscriptionTier.INSTITUTIONAL: INSTITUTIONAL_TIER_LIMITS,
}


def get_risk_limits(tier: SubscriptionTier) -> RiskLimits:
    """
    Get risk limits for a subscription tier.
    
    Args:
        tier: Subscription tier
        
    Returns:
        RiskLimits for the tier
    """
    return TIER_LIMITS.get(tier, PRO_TIER_LIMITS)


def get_tier_from_string(tier_str: str) -> SubscriptionTier:
    """
    Convert string to SubscriptionTier enum.
    
    Args:
        tier_str: Tier name as string
        
    Returns:
        SubscriptionTier enum value
    """
    tier_map = {
        "free": SubscriptionTier.FREE,
        "basic": SubscriptionTier.BASIC,
        "pro": SubscriptionTier.PRO,
        "enterprise": SubscriptionTier.ENTERPRISE,
        "institutional": SubscriptionTier.INSTITUTIONAL,
    }
    return tier_map.get(tier_str.lower(), SubscriptionTier.PRO)


# ============================================================================
# GLOBAL DEFAULTS
# ============================================================================
DEFAULT_RISK_LIMITS = PRO_TIER_LIMITS

# Circuit breaker cooldown periods (in seconds)
CIRCUIT_BREAKER_COOLDOWNS = {
    "daily_loss": 3600,      # 1 hour
    "weekly_loss": 86400,    # 24 hours
    "drawdown": 1800,        # 30 minutes
    "var_breach": 900,       # 15 minutes
    "exposure": 600,         # 10 minutes
}

# Alert thresholds (percentage of limit before alert)
ALERT_THRESHOLDS = {
    "position_size": Decimal("0.80"),
    "exposure": Decimal("0.85"),
    "drawdown": Decimal("0.80"),
    "daily_loss": Decimal("0.75"),
    "var": Decimal("0.90"),
    "correlation": Decimal("0.85"),
    "leverage": Decimal("0.80"),
    "margin": Decimal("0.85"),
}

# Maximum consecutive violations before kill switch
MAX_CONSECUTIVE_VIOLATIONS = {
    "position_size": 3,
    "exposure": 2,
    "drawdown": 1,
    "daily_loss": 1,
    "var": 2,
    "leverage": 2,
}
