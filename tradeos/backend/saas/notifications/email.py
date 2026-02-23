"""
TradeOS Email Notifications
===========================
Email service for user notifications and alerts.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from jinja2 import Template

from ..config.saas_config import get_email_config, EmailConfig


class EmailTemplate(Enum):
    """Available email templates"""
    WELCOME = "welcome"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    PASSWORD_CHANGED = "password_changed"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    SUBSCRIPTION_CANCELED = "subscription_canceled"
    PAYMENT_SUCCEEDED = "payment_succeeded"
    PAYMENT_FAILED = "payment_failed"
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"
    RISK_ALERT = "risk_alert"
    STRATEGY_ALERT = "strategy_alert"
    QUOTA_WARNING = "quota_warning"
    SECURITY_ALERT = "security_alert"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


@dataclass
class EmailMessage:
    """Email message"""
    to_email: str
    to_name: Optional[str]
    subject: str
    html_content: str
    text_content: Optional[str] = None
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


# Email templates (HTML)
EMAIL_TEMPLATES = {
    EmailTemplate.WELCOME: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #1a73e8; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .button { display: inline-block; background: #1a73e8; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to TradeOS!</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>Welcome to TradeOS - your algorithmic trading platform. We're excited to have you on board!</p>
            <p>With TradeOS, you can:</p>
            <ul>
                <li>Create and backtest trading strategies</li>
                <li>Execute trades automatically</li>
                <li>Analyze performance with advanced metrics</li>
                <li>Connect to multiple exchanges</li>
            </ul>
            <a href="{{ dashboard_url }}" class="button">Go to Dashboard</a>
            <p>If you have any questions, our support team is here to help.</p>
        </div>
        <div class="footer">
            <p>© {{ year }} TradeOS. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.EMAIL_VERIFICATION: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #1a73e8; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; text-align: center; }
        .button { display: inline-block; background: #1a73e8; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .code { font-size: 24px; font-weight: bold; letter-spacing: 4px; 
                background: #e8f0fe; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Verify Your Email</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>Please verify your email address to complete your registration.</p>
            <a href="{{ verification_url }}" class="button">Verify Email</a>
            <p>Or use this verification code:</p>
            <div class="code">{{ verification_code }}</div>
            <p>This link will expire in 24 hours.</p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.PASSWORD_RESET: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #1a73e8; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; text-align: center; }
        .button { display: inline-block; background: #1a73e8; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .warning { background: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>We received a request to reset your password.</p>
            <a href="{{ reset_url }}" class="button">Reset Password</a>
            <div class="warning">
                <p><strong>Didn't request this?</strong></p>
                <p>If you didn't request a password reset, you can safely ignore this email.</p>
            </div>
            <p>This link will expire in 1 hour.</p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.SUBSCRIPTION_CONFIRMED: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #28a745; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .plan-box { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; 
                    border: 2px solid #28a745; }
        .feature-list { list-style: none; padding: 0; }
        .feature-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
        .feature-list li:before { content: "✓ "; color: #28a745; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Subscription Confirmed!</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>Your subscription to <strong>{{ plan_name }}</strong> has been confirmed.</p>
            <div class="plan-box">
                <h3>{{ plan_name }}</h3>
                <p><strong>Amount:</strong> ${{ amount }}/{{ interval }}</p>
                <p><strong>Start Date:</strong> {{ start_date }}</p>
                <p><strong>Next Billing:</strong> {{ next_billing }}</p>
            </div>
            <p>You now have access to:</p>
            <ul class="feature-list">
                {% for feature in features %}
                <li>{{ feature }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.PAYMENT_SUCCEEDED: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #28a745; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .invoice-box { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .total { font-size: 24px; font-weight: bold; color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Payment Successful</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>Thank you for your payment. Your invoice details:</p>
            <div class="invoice-box">
                <p><strong>Invoice #:</strong> {{ invoice_number }}</p>
                <p><strong>Date:</strong> {{ date }}</p>
                <p><strong>Description:</strong> {{ description }}</p>
                <p class="total">Total: ${{ amount }}</p>
            </div>
            <p><a href="{{ invoice_url }}">View Invoice</a></p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.PAYMENT_FAILED: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .alert { background: #f8d7da; padding: 15px; margin: 20px 0; border-left: 4px solid #dc3545; }
        .button { display: inline-block; background: #dc3545; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Payment Failed</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <div class="alert">
                <p><strong>We couldn't process your payment.</strong></p>
                <p>{{ error_message }}</p>
            </div>
            <p>Please update your payment method to avoid service interruption.</p>
            <a href="{{ billing_url }}" class="button">Update Payment Method</a>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.TRADE_EXECUTED: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #1a73e8; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .trade-box { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .buy { color: #28a745; }
        .sell { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trade Executed</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>Your strategy <strong>{{ strategy_name }}</strong> has executed a trade:</p>
            <div class="trade-box">
                <p><strong>Symbol:</strong> {{ symbol }}</p>
                <p><strong>Side:</strong> <span class="{{ side.lower() }}">{{ side }}</span></p>
                <p><strong>Quantity:</strong> {{ quantity }}</p>
                <p><strong>Price:</strong> ${{ price }}</p>
                <p><strong>Total:</strong> ${{ total }}</p>
                <p><strong>Time:</strong> {{ timestamp }}</p>
            </div>
            <p><a href="{{ trades_url }}">View All Trades</a></p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.RISK_ALERT: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .alert-box { background: #f8d7da; padding: 20px; margin: 20px 0; 
                     border-left: 4px solid #dc3545; border-radius: 4px; }
        .metric { display: flex; justify-content: space-between; padding: 10px 0; 
                  border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Risk Alert</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <div class="alert-box">
                <h3>{{ alert_title }}</h3>
                <p>{{ alert_description }}</p>
            </div>
            <p><strong>Strategy:</strong> {{ strategy_name }}</p>
            <div class="metrics">
                <div class="metric">
                    <span>Current Drawdown:</span>
                    <span>{{ drawdown }}%</span>
                </div>
                <div class="metric">
                    <span>Daily P&L:</span>
                    <span>${{ daily_pnl }}</span>
                </div>
                <div class="metric">
                    <span>Position Size:</span>
                    <span>${{ position_size }}</span>
                </div>
            </div>
            <p><a href="{{ strategy_url }}">Manage Strategy</a></p>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.QUOTA_WARNING: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #ffc107; color: #333; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .usage-bar { background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; }
        .usage-fill { background: #ffc107; height: 100%; }
        .button { display: inline-block; background: #1a73e8; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Usage Alert</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <p>You're approaching your {{ resource_name }} limit for your {{ tier }} plan.</p>
            <div class="usage-bar">
                <div class="usage-fill" style="width: {{ percent_used }}%;"></div>
            </div>
            <p>{{ used }} / {{ limit }} {{ resource_name }} used ({{ percent_used }}%)</p>
            <a href="{{ upgrade_url }}" class="button">Upgrade Plan</a>
        </div>
    </div>
</body>
</html>
""",
    
    EmailTemplate.SECURITY_ALERT: """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { background: #f9f9f9; padding: 30px; }
        .alert-box { background: #f8d7da; padding: 20px; margin: 20px 0; 
                     border-left: 4px solid #dc3545; border-radius: 4px; }
        .button { display: inline-block; background: #dc3545; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Security Alert</h1>
        </div>
        <div class="content">
            <h2>Hi {{ name }},</h2>
            <div class="alert-box">
                <h3>{{ alert_type }}</h3>
                <p>{{ alert_description }}</p>
            </div>
            <p><strong>Time:</strong> {{ timestamp }}</p>
            <p><strong>IP Address:</strong> {{ ip_address }}</p>
            <p><strong>Location:</strong> {{ location }}</p>
            <p>If this wasn't you, please secure your account immediately.</p>
            <a href="{{ security_url }}" class="button">Review Activity</a>
        </div>
    </div>
</body>
</html>
""",
}


class EmailService:
    """
    Email service for TradeOS notifications.
    
    Supports:
    - SMTP email sending
    - Template-based emails
    - Multiple notification types
    """
    
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or get_email_config()
        self.templates = {
            t: Template(html) for t, html in EMAIL_TEMPLATES.items()
        }
    
    def _create_smtp_connection(self):
        """Create SMTP connection"""
        context = ssl.create_default_context()
        
        if self.config.use_tls:
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            server.starttls(context=context)
        else:
            server = smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port, context=context)
        
        server.login(self.config.smtp_username, self.config.smtp_password)
        return server
    
    def render_template(
        self,
        template: EmailTemplate,
        **context
    ) -> str:
        """Render email template with context"""
        tmpl = self.templates.get(template)
        if not tmpl:
            raise ValueError(f"Unknown template: {template}")
        
        # Add common context
        context.setdefault("year", datetime.now().year)
        context.setdefault("app_name", "TradeOS")
        context.setdefault("support_email", "support@tradeos.io")
        
        return tmpl.render(**context)
    
    async def send_email(
        self,
        message: EmailMessage
    ) -> bool:
        """
        Send an email.
        
        Args:
            message: Email message to send
            
        Returns:
            True if sent successfully
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = message.subject
            msg["From"] = f"{message.from_name or self.config.from_name} <{message.from_email or self.config.from_email}>"
            msg["To"] = f"{message.to_name or ''} <{message.to_email}>".strip()
            
            if message.reply_to:
                msg["Reply-To"] = message.reply_to
            
            # Attach HTML content
            html_part = MIMEText(message.html_content, "html")
            msg.attach(html_part)
            
            # Attach text content if provided
            if message.text_content:
                text_part = MIMEText(message.text_content, "plain")
                msg.attach(text_part)
            
            # Send
            with self._create_smtp_connection() as server:
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
    
    async def send_templated_email(
        self,
        to_email: str,
        to_name: Optional[str],
        template: EmailTemplate,
        subject: str,
        **context
    ) -> bool:
        """
        Send a templated email.
        
        Args:
            to_email: Recipient email
            to_name: Recipient name
            template: Email template
            subject: Email subject
            **context: Template context variables
            
        Returns:
            True if sent successfully
        """
        html_content = self.render_template(template, **context)
        
        message = EmailMessage(
            to_email=to_email,
            to_name=to_name,
            subject=subject,
            html_content=html_content
        )
        
        return await self.send_email(message)
    
    # ============== Specific Notification Methods ==============
    
    async def send_welcome_email(
        self,
        email: str,
        name: str,
        dashboard_url: str = "https://app.tradeos.io/dashboard"
    ) -> bool:
        """Send welcome email to new users"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.WELCOME,
            subject="Welcome to TradeOS!",
            name=name,
            dashboard_url=dashboard_url
        )
    
    async def send_verification_email(
        self,
        email: str,
        name: str,
        verification_token: str,
        verification_url: Optional[str] = None
    ) -> bool:
        """Send email verification email"""
        if not verification_url:
            verification_url = f"https://app.tradeos.io/verify-email?token={verification_token}"
        
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.EMAIL_VERIFICATION,
            subject="Verify Your Email Address",
            name=name,
            verification_url=verification_url,
            verification_code=verification_token[:8]
        )
    
    async def send_password_reset_email(
        self,
        email: str,
        name: str,
        reset_token: str,
        reset_url: Optional[str] = None
    ) -> bool:
        """Send password reset email"""
        if not reset_url:
            reset_url = f"https://app.tradeos.io/reset-password?token={reset_token}"
        
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.PASSWORD_RESET,
            subject="Password Reset Request",
            name=name,
            reset_url=reset_url
        )
    
    async def send_subscription_confirmation(
        self,
        email: str,
        name: str,
        plan_name: str,
        amount: str,
        interval: str,
        start_date: str,
        next_billing: str,
        features: List[str]
    ) -> bool:
        """Send subscription confirmation email"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.SUBSCRIPTION_CONFIRMED,
            subject="Your Subscription is Confirmed!",
            name=name,
            plan_name=plan_name,
            amount=amount,
            interval=interval,
            start_date=start_date,
            next_billing=next_billing,
            features=features
        )
    
    async def send_payment_succeeded(
        self,
        email: str,
        name: str,
        invoice_number: str,
        amount: str,
        description: str,
        invoice_url: str
    ) -> bool:
        """Send payment success email"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.PAYMENT_SUCCEEDED,
            subject="Payment Successful",
            name=name,
            invoice_number=invoice_number,
            amount=amount,
            description=description,
            date=datetime.now().strftime("%Y-%m-%d"),
            invoice_url=invoice_url
        )
    
    async def send_payment_failed(
        self,
        email: str,
        name: str,
        error_message: str,
        billing_url: str = "https://app.tradeos.io/billing"
    ) -> bool:
        """Send payment failure email"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.PAYMENT_FAILED,
            subject="Payment Failed - Action Required",
            name=name,
            error_message=error_message,
            billing_url=billing_url
        )
    
    async def send_trade_execution_notification(
        self,
        email: str,
        name: str,
        strategy_name: str,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        total: str,
        timestamp: str,
        trades_url: str
    ) -> bool:
        """Send trade execution notification"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.TRADE_EXECUTED,
            subject=f"Trade Executed: {symbol} {side}",
            name=name,
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            total=total,
            timestamp=timestamp,
            trades_url=trades_url
        )
    
    async def send_risk_alert(
        self,
        email: str,
        name: str,
        strategy_name: str,
        alert_title: str,
        alert_description: str,
        drawdown: str,
        daily_pnl: str,
        position_size: str,
        strategy_url: str
    ) -> bool:
        """Send risk limit alert"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.RISK_ALERT,
            subject=f"Risk Alert: {strategy_name}",
            name=name,
            strategy_name=strategy_name,
            alert_title=alert_title,
            alert_description=alert_description,
            drawdown=drawdown,
            daily_pnl=daily_pnl,
            position_size=position_size,
            strategy_url=strategy_url
        )
    
    async def send_quota_warning(
        self,
        email: str,
        name: str,
        resource_name: str,
        tier: str,
        used: int,
        limit: int,
        percent_used: float,
        upgrade_url: str = "https://app.tradeos.io/billing/upgrade"
    ) -> bool:
        """Send quota warning email"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.QUOTA_WARNING,
            subject=f"Usage Alert: {resource_name} at {int(percent_used)}%",
            name=name,
            resource_name=resource_name,
            tier=tier,
            used=used,
            limit=limit,
            percent_used=percent_used,
            upgrade_url=upgrade_url
        )
    
    async def send_security_alert(
        self,
        email: str,
        name: str,
        alert_type: str,
        alert_description: str,
        ip_address: str,
        location: str,
        security_url: str = "https://app.tradeos.io/security"
    ) -> bool:
        """Send security alert email"""
        return await self.send_templated_email(
            to_email=email,
            to_name=name,
            template=EmailTemplate.SECURITY_ALERT,
            subject=f"Security Alert: {alert_type}",
            name=name,
            alert_type=alert_type,
            alert_description=alert_description,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            ip_address=ip_address,
            location=location,
            security_url=security_url
        )


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create global email service"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
