"""
TradeOS Dependencies Package
"""

from dependencies.auth import (
    authenticate_user_or_api_key,
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_token_from_request,
    get_api_key_user,
    get_current_active_user,
    get_current_user,
    get_current_verified_user,
    get_optional_user,
    hash_password,
    http_bearer,
    is_password_strong,
    oauth2_scheme,
    pwd_context,
    require_admin,
    require_api_permission,
    require_role,
    verify_password,
)

__all__ = [
    # Password utilities
    "verify_password",
    "hash_password",
    "is_password_strong",
    "pwd_context",
    # Token utilities
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "extract_token_from_request",
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "require_role",
    "require_admin",
    "get_api_key_user",
    "authenticate_user_or_api_key",
    "require_api_permission",
    "get_optional_user",
    # Schemes
    "oauth2_scheme",
    "http_bearer",
]
