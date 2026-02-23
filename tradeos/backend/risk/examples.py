"""
TradeOS Risk Engine - Usage Examples

This file demonstrates how to use the Risk Engine for various scenarios.
"""

from decimal import Decimal
from datetime import datetime

# Import the Risk Engine
from tradeos.backend.risk import (
    create_risk_engine,
    RiskEngine,
    TradeRequest,
    PortfolioState,
    Position,
    SubscriptionTier,
    PositionSizingMethod,
)


def example_1_basic_initialization():
    """Example 1: Basic Risk Engine initialization"""
    print("=" * 60)
    print("Example 1: Basic Risk Engine Initialization")
    print("=" * 60)
    
    # Create and initialize the risk engine
    engine = create_risk_engine(
        tier="pro",
        position_sizing_method="risk_per_trade",
    )
    engine.initialize()
    
    print(f"Risk Engine Status: {engine.get_status()}")
    print()


def example_2_trade_validation():
    """Example 2: Validating a trade"""
    print("=" * 60)
    print("Example 2: Trade Validation")
    print("=" * 60)
    
    # Create engine
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Create portfolio state
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("50000"),
        buying_power=Decimal("100000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
    )
    
    # Create a trade request
    trade = TradeRequest(
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        price=Decimal("150"),
        stop_loss=Decimal("140"),
        order_type="limit",
    )
    
    # Validate the trade
    result = engine.validate_trade(trade, portfolio)
    
    print(f"Trade: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
    print(f"Is Valid: {result.is_valid}")
    print(f"Risk Level: {result.risk_level.name}")
    
    if not result.is_valid:
        print(f"Rejection Reason: {result.rejection_reason}")
    
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    print()


def example_3_position_sizing():
    """Example 3: Position sizing calculation"""
    print("=" * 60)
    print("Example 3: Position Sizing")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("100000"),
        buying_power=Decimal("100000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
    )
    
    # Calculate position size with stop loss
    sizing = engine.calculate_position_size(
        symbol="AAPL",
        entry_price=Decimal("150"),
        portfolio_state=portfolio,
        stop_loss=Decimal("140"),
    )
    
    print(f"Recommended Quantity: {sizing.recommended_quantity}")
    print(f"Recommended Value: ${sizing.recommended_value:,.2f}")
    print(f"Risk Amount: ${sizing.risk_amount:,.2f}")
    print(f"Risk %: {sizing.risk_pct:.2%}")
    print(f"Method: {sizing.method.value}")
    print(f"Confidence: {sizing.confidence_score:.0%}")
    
    if sizing.warnings:
        print(f"Warnings: {sizing.warnings}")
    
    print()


def example_4_risk_report():
    """Example 4: Generate risk report"""
    print("=" * 60)
    print("Example 4: Risk Report Generation")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Create portfolio with positions
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_entry_price=Decimal("145"),
            current_price=Decimal("150"),
            side="long",
            unrealized_pnl=Decimal("500"),
        ),
        "MSFT": Position(
            symbol="MSFT",
            quantity=Decimal("50"),
            avg_entry_price=Decimal("280"),
            current_price=Decimal("290"),
            side="long",
            unrealized_pnl=Decimal("500"),
        ),
    }
    
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("35000"),
        buying_power=Decimal("35000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions=positions,
    )
    
    # Update engine with portfolio state
    engine.update_portfolio_state(portfolio)
    
    # Generate risk report
    report = engine.generate_risk_report(portfolio)
    
    print(f"Report Timestamp: {report.timestamp}")
    print(f"Overall Risk Level: {report.risk_level.name}")
    print(f"Portfolio Equity: ${report.portfolio_state.total_equity:,.2f}")
    print(f"Total Exposure: ${report.portfolio_state.total_exposure:,.2f}")
    print(f"Leverage: {report.portfolio_state.leverage:.2f}x")
    
    if report.alerts:
        print(f"Alerts: {report.alerts}")
    
    if report.recommendations:
        print(f"Recommendations: {report.recommendations}")
    
    print()


def example_5_drawdown_protection():
    """Example 5: Drawdown protection"""
    print("=" * 60)
    print("Example 5: Drawdown Protection")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Portfolio in drawdown
    portfolio = PortfolioState(
        total_equity=Decimal("85000"),  # Down from 100k peak
        cash_balance=Decimal("85000"),
        buying_power=Decimal("85000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
        peak_equity=Decimal("100000"),
    )
    
    # Update engine
    updates = engine.update_portfolio_state(portfolio)
    
    print(f"Can Trade: {engine.can_trade()}")
    print(f"Drawdown Updates: {updates.get('drawdown', {})}")
    
    # Get drawdown report
    if engine.drawdown_controller:
        report = engine.drawdown_controller.get_drawdown_report()
        print(f"Current Drawdown: {report['current_drawdown_pct']:.2%}")
        print(f"Max Drawdown: {report['max_drawdown_pct']:.2%}")
    
    print()


def example_6_kill_switch():
    """Example 6: Kill switch usage"""
    print("=" * 60)
    print("Example 6: Kill Switch")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Check initial status
    print(f"Initial Can Trade: {engine.can_trade()}")
    print(f"Kill Switch Status: {engine.get_kill_switch_status()}")
    
    # Trigger kill switch manually
    engine.trigger_kill_switch(
        reason="Manual emergency stop",
        user_id="admin"
    )
    
    print(f"After Trigger - Can Trade: {engine.can_trade()}")
    print(f"Kill Switch Status: {engine.get_kill_switch_status()}")
    
    # Release kill switch
    engine.release_kill_switch(user_id="admin", force=True)
    
    print(f"After Release - Can Trade: {engine.can_trade()}")
    print()


def example_7_exposure_limits():
    """Example 7: Exposure limit validation"""
    print("=" * 60)
    print("Example 7: Exposure Limits")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Register sector metadata
    engine.exposure_manager.register_symbol_metadata(
        symbol="AAPL",
        sector="technology",
        region="us",
    )
    engine.exposure_manager.register_symbol_metadata(
        symbol="MSFT",
        sector="technology",
        region="us",
    )
    
    # Portfolio with concentrated position
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=Decimal("500"),
            avg_entry_price=Decimal("150"),
            current_price=Decimal("150"),
            side="long",
        ),
    }
    
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("25000"),
        buying_power=Decimal("25000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions=positions,
    )
    
    # Update and check exposure
    updates = engine.update_portfolio_state(portfolio)
    
    print(f"Exposure Alerts: {updates.get('exposure', {}).get('alerts', [])}")
    
    # Try to add more to same sector
    trade = TradeRequest(
        symbol="MSFT",
        side="buy",
        quantity=Decimal("200"),
        price=Decimal("300"),
    )
    
    result = engine.validate_trade(trade, portfolio)
    print(f"Trade Valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Rejection: {result.rejection_reason}")
    
    print()


def example_8_validation_statistics():
    """Example 8: Validation statistics"""
    print("=" * 60)
    print("Example 8: Validation Statistics")
    print("=" * 60)
    
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("100000"),
        buying_power=Decimal("100000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
    )
    
    # Validate several trades
    trades = [
        TradeRequest(symbol="AAPL", side="buy", quantity=Decimal("10"), price=Decimal("150")),
        TradeRequest(symbol="GOOGL", side="buy", quantity=Decimal("5"), price=Decimal("2800")),
        TradeRequest(symbol="TSLA", side="buy", quantity=Decimal("1000"), price=Decimal("800")),  # Too large
    ]
    
    for trade in trades:
        result = engine.validate_trade(trade, portfolio)
        print(f"{trade.symbol}: {'PASS' if result.is_valid else 'FAIL'} - {result.rejection_reason or 'OK'}")
    
    # Get statistics
    stats = engine.get_validation_stats()
    print(f"\nValidation Statistics:")
    print(f"  Total: {stats['total_validations']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Pass Rate: {stats['pass_rate']:.1%}")
    print(f"  Rejection Reasons: {stats['rejection_reasons']}")
    
    print()


def example_9_compare_position_sizing_methods():
    """Example 9: Compare different position sizing methods"""
    print("=" * 60)
    print("Example 9: Compare Position Sizing Methods")
    print("=" * 60)
    
    from tradeos.backend.risk import PositionSizer, PositionSizingParams
    
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("100000"),
        buying_power=Decimal("100000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
    )
    
    # Compare all methods
    sizer = PositionSizer(tier=SubscriptionTier.PRO)
    
    results = sizer.compare_methods(
        symbol="AAPL",
        entry_price=Decimal("150"),
        portfolio_state=portfolio,
        stop_loss=Decimal("140"),
    )
    
    print(f"{'Method':<25} {'Quantity':<12} {'Risk %':<10} {'Confidence'}")
    print("-" * 60)
    
    for method, sizing in results.items():
        print(f"{method.value:<25} {str(sizing.recommended_quantity):<12} "
              f"{sizing.risk_pct:<10.2%} {sizing.confidence_score:.0%}")
    
    print()


def example_10_full_workflow():
    """Example 10: Complete trading workflow with risk management"""
    print("=" * 60)
    print("Example 10: Complete Trading Workflow")
    print("=" * 60)
    
    # Initialize engine
    engine = create_risk_engine(tier="pro")
    engine.initialize()
    
    # Register callbacks
    def on_risk_alert(alert_type, data):
        print(f"  [ALERT] {alert_type}: {data}")
    
    def on_trading_halt(reason):
        print(f"  [HALT] Trading halted: {reason}")
    
    engine.on_risk_alert(on_risk_alert)
    engine.on_trading_halt(on_trading_halt)
    
    # Initial portfolio
    portfolio = PortfolioState(
        total_equity=Decimal("100000"),
        cash_balance=Decimal("100000"),
        buying_power=Decimal("100000"),
        margin_used=Decimal("0"),
        margin_available=Decimal("100000"),
        positions={},
    )
    
    print("Step 1: Calculate position size for AAPL")
    sizing = engine.calculate_position_size(
        symbol="AAPL",
        entry_price=Decimal("150"),
        portfolio_state=portfolio,
        stop_loss=Decimal("140"),
    )
    print(f"  Recommended: {sizing.recommended_quantity} shares")
    
    print("\nStep 2: Create and validate trade")
    trade = TradeRequest(
        symbol="AAPL",
        side="buy",
        quantity=sizing.recommended_quantity,
        price=Decimal("150"),
        stop_loss=Decimal("140"),
    )
    
    result = engine.validate_trade(trade, portfolio)
    print(f"  Validation: {'PASS' if result.is_valid else 'FAIL'}")
    
    if result.is_valid:
        print("\nStep 3: Execute trade (simulated)")
        # Update portfolio with new position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=trade.quantity,
            avg_entry_price=trade.price,
            current_price=trade.price,
            side="long",
        )
        portfolio.cash_balance -= trade.quantity * trade.price
        
        print("\nStep 4: Update risk metrics")
        updates = engine.update_portfolio_state(portfolio)
        
        print("\nStep 5: Generate risk report")
        report = engine.generate_risk_report(portfolio)
        print(f"  Risk Level: {report.risk_level.name}")
        print(f"  Exposure: ${report.portfolio_state.total_exposure:,.2f}")
        print(f"  Leverage: {report.portfolio_state.leverage:.2f}x")
    
    print()


if __name__ == "__main__":
    """Run all examples"""
    print("\n" + "=" * 60)
    print("TradeOS Risk Engine - Usage Examples")
    print("=" * 60 + "\n")
    
    example_1_basic_initialization()
    example_2_trade_validation()
    example_3_position_sizing()
    example_4_risk_report()
    example_5_drawdown_protection()
    example_6_kill_switch()
    example_7_exposure_limits()
    example_8_validation_statistics()
    example_9_compare_position_sizing_methods()
    example_10_full_workflow()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
