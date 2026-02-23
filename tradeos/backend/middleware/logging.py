"""
TradeOS Logging Middleware
Request/response logging middleware with structured logging support.
"""

import logging
import time
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import settings

# Configure logging
logger = logging.getLogger("tradeos.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request details, processing time, and response status.
    Supports structured JSON logging for production environments.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 10000
    ):
        """
        Initialize the logging middleware.
        
        Args:
            app: The ASGI application.
            exclude_paths: List of paths to exclude from logging.
            log_request_body: Whether to log request bodies.
            log_response_body: Whether to log response bodies.
            max_body_size: Maximum body size to log.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static/",
            "/favicon.ico"
        ]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
    
    def _should_log(self, path: str) -> bool:
        """Check if the path should be logged."""
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return False
        return True
    
    def _sanitize_headers(self, headers: dict) -> dict:
        """Sanitize sensitive headers."""
        sensitive_headers = {
            "authorization", "x-api-key", "cookie", "x-csrf-token",
            "x-auth-token", "api-key", "password", "secret"
        }
        
        sanitized = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_headers):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body if logging is enabled."""
        if not self.log_request_body:
            return None
        
        try:
            body = await request.body()
            if len(body) > self.max_body_size:
                return f"<Body too large: {len(body)} bytes>"
            return body.decode("utf-8", errors="replace") if body else None
        except Exception as e:
            return f"<Error reading body: {e}>"
    
    def _build_log_data(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
        request_body: Optional[str] = None,
        response_body: Optional[str] = None
    ) -> dict:
        """Build structured log data."""
        log_data = {
            "event": "http_request",
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 3),
            "content_length": response.headers.get("content-length"),
        }
        
        # Add user info if available
        if hasattr(request.state, "user_id"):
            log_data["user_id"] = str(request.state.user_id)
        
        # Add headers in debug mode
        if settings.debug:
            log_data["headers"] = self._sanitize_headers(dict(request.headers))
        
        # Add bodies if enabled
        if request_body:
            log_data["request_body"] = request_body[:self.max_body_size]
        if response_body:
            log_data["response_body"] = response_body[:self.max_body_size]
        
        return log_data
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: The incoming request.
            call_next: The next middleware/endpoint in the chain.
        
        Returns:
            Response: The response from the next handler.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Check if we should log this request
        if not self._should_log(request.url.path):
            return await call_next(request)
        
        # Record start time
        start_time = time.perf_counter()
        
        # Get request body if needed
        request_body = await self._get_request_body(request)
        
        # Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 3),
                    "error": str(e)
                }
            )
            raise
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Get response body if needed
        response_body = None
        if self.log_response_body:
            # Note: Reading response body in middleware can be tricky
            # This is a simplified version
            pass
        
        # Build and log
        log_data = self._build_log_data(
            request, response, duration_ms, request_body, response_body
        )
        
        # Log based on status code
        if response.status_code >= 500:
            logger.error("Server error", extra=log_data)
        elif response.status_code >= 400:
            logger.warning("Client error", extra=log_data)
        elif duration_ms > 1000:  # Slow request
            logger.warning("Slow request", extra=log_data)
        else:
            logger.info("Request completed", extra=log_data)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds structured logging context to each request.
    
    Adds request-scoped context for correlation across log entries.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add logging context and process request."""
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Add to logging context
        import structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            request_path=request.url.path,
            request_method=request.method
        )
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed error logging.
    
    Captures and logs detailed error information for debugging.
    """
    
    def __init__(self, app: ASGIApp, include_traceback: bool = True):
        super().__init__(app)
        self.include_traceback = include_traceback
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log any errors."""
        try:
            return await call_next(request)
        except Exception as e:
            import traceback
            
            error_data = {
                "event": "unhandled_exception",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            
            if self.include_traceback:
                error_data["traceback"] = traceback.format_exc()
            
            logger.critical("Unhandled exception", extra=error_data)
            raise


# Factory function for creating logging middleware with default settings
def create_logging_middleware(
    exclude_paths: Optional[list] = None,
    log_request_body: bool = False,
    log_response_body: bool = False
) -> type:
    """
    Create a configured logging middleware class.
    
    Args:
        exclude_paths: Paths to exclude from logging.
        log_request_body: Whether to log request bodies.
        log_response_body: Whether to log response bodies.
    
    Returns:
        Configured middleware class.
    """
    class ConfiguredLoggingMiddleware(RequestLoggingMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(
                app,
                exclude_paths=exclude_paths,
                log_request_body=log_request_body,
                log_response_body=log_response_body
            )
    
    return ConfiguredLoggingMiddleware


__all__ = [
    "RequestLoggingMiddleware",
    "StructuredLoggingMiddleware",
    "ErrorLoggingMiddleware",
    "create_logging_middleware",
]
