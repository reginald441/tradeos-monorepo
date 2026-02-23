"""
TradeOS OAuth2 Integration
==========================
Google and GitHub OAuth authentication handlers.
"""

import secrets
import hashlib
import base64
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from ..config.saas_config import get_oauth_config, OAuthConfig


@dataclass
class OAuthUserInfo:
    """Standardized OAuth user information"""
    provider: str
    provider_user_id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    username: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class OAuthTokens:
    """OAuth tokens from provider"""
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: str = "bearer"
    scope: Optional[str] = None


class OAuthStateManager:
    """
    Manages OAuth state parameter for CSRF protection.
    Uses PKCE (Proof Key for Code Exchange) for enhanced security.
    """
    
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}
    
    def generate_state(self, provider: str, redirect_after: Optional[str] = None) -> str:
        """Generate a secure state parameter"""
        state = secrets.token_urlsafe(32)
        
        # Generate PKCE code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        self._states[state] = {
            "provider": provider,
            "code_verifier": code_verifier,
            "code_challenge": code_challenge,
            "redirect_after": redirect_after or "/",
            "created_at": __import__('time').time()
        }
        
        return state
    
    def validate_state(self, state: str, provider: str) -> Optional[str]:
        """
        Validate state parameter and return code verifier.
        
        Returns:
            code_verifier if valid, None otherwise
        """
        data = self._states.pop(state, None)
        if not data:
            return None
        
        # Check expiration (10 minutes)
        if __import__('time').time() - data["created_at"] > 600:
            return None
        
        # Verify provider matches
        if data["provider"] != provider:
            return None
        
        return data.get("code_verifier")
    
    def get_redirect_after(self, state: str) -> str:
        """Get redirect URL after OAuth completion"""
        data = self._states.get(state, {})
        return data.get("redirect_after", "/")


class GoogleOAuthHandler:
    """Google OAuth2 authentication handler"""
    
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    SCOPES = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ]
    
    def __init__(self, config: Optional[OAuthConfig] = None):
        self.config = config or get_oauth_config()
        self.state_manager = OAuthStateManager()
    
    def get_authorization_url(
        self,
        redirect_after: Optional[str] = None,
        additional_scopes: Optional[list] = None
    ) -> str:
        """
        Generate Google OAuth authorization URL.
        
        Returns:
            Authorization URL to redirect user to
        """
        state = self.state_manager.generate_state("google", redirect_after)
        
        scopes = self.SCOPES.copy()
        if additional_scopes:
            scopes.extend(additional_scopes)
        
        params = {
            "client_id": self.config.google_client_id,
            "redirect_uri": self.config.google_redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": self.state_manager._states[state]["code_challenge"],
            "code_challenge_method": "S256"
        }
        
        return f"{self.AUTH_URL}?{urlencode(params)}"
    
    async def exchange_code(
        self,
        code: str,
        state: str
    ) -> tuple[OAuthTokens, OAuthUserInfo]:
        """
        Exchange authorization code for tokens and user info.
        
        Args:
            code: Authorization code from callback
            state: State parameter from callback
            
        Returns:
            Tuple of (OAuthTokens, OAuthUserInfo)
            
        Raises:
            HTTPException: If code exchange fails
        """
        # Validate state and get code verifier
        code_verifier = self.state_manager.validate_state(state, "google")
        if not code_verifier:
            raise HTTPException(status_code=400, detail="Invalid or expired state parameter")
        
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.config.google_client_id,
                    "client_secret": self.config.google_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": self.config.google_redirect_uri,
                    "code_verifier": code_verifier
                }
            )
            
            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to exchange code: {token_response.text}"
                )
            
            token_data = token_response.json()
            tokens = OAuthTokens(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_in=token_data.get("expires_in"),
                token_type=token_data.get("token_type", "bearer"),
                scope=token_data.get("scope")
            )
            
            # Get user info
            user_response = await client.get(
                self.USERINFO_URL,
                headers={"Authorization": f"Bearer {tokens.access_token}"}
            )
            
            if user_response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to fetch user info"
                )
            
            user_data = user_response.json()
            user_info = OAuthUserInfo(
                provider="google",
                provider_user_id=user_data["id"],
                email=user_data["email"],
                name=user_data.get("name"),
                avatar_url=user_data.get("picture"),
                raw_data=user_data
            )
            
            return tokens, user_info
    
    async def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """Refresh access token using refresh token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.config.google_client_id,
                    "client_secret": self.config.google_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to refresh token"
                )
            
            data = response.json()
            return OAuthTokens(
                access_token=data["access_token"],
                expires_in=data.get("expires_in"),
                token_type=data.get("token_type", "bearer"),
                scope=data.get("scope")
            )


class GitHubOAuthHandler:
    """GitHub OAuth2 authentication handler"""
    
    AUTH_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    API_URL = "https://api.github.com"
    
    SCOPES = ["read:user", "user:email"]
    
    def __init__(self, config: Optional[OAuthConfig] = None):
        self.config = config or get_oauth_config()
        self.state_manager = OAuthStateManager()
    
    def get_authorization_url(
        self,
        redirect_after: Optional[str] = None,
        additional_scopes: Optional[list] = None
    ) -> str:
        """Generate GitHub OAuth authorization URL"""
        state = self.state_manager.generate_state("github", redirect_after)
        
        scopes = self.SCOPES.copy()
        if additional_scopes:
            scopes.extend(additional_scopes)
        
        params = {
            "client_id": self.config.github_client_id,
            "redirect_uri": self.config.github_redirect_uri,
            "scope": " ".join(scopes),
            "state": state
        }
        
        return f"{self.AUTH_URL}?{urlencode(params)}"
    
    async def exchange_code(
        self,
        code: str,
        state: str
    ) -> tuple[OAuthTokens, OAuthUserInfo]:
        """Exchange authorization code for tokens and user info"""
        # Validate state
        code_verifier = self.state_manager.validate_state(state, "github")
        if not code_verifier:
            raise HTTPException(status_code=400, detail="Invalid or expired state parameter")
        
        async with httpx.AsyncClient() as client:
            # Exchange code for token
            token_response = await client.post(
                self.TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": self.config.github_client_id,
                    "client_secret": self.config.github_client_secret,
                    "code": code,
                    "redirect_uri": self.config.github_redirect_uri
                }
            )
            
            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to exchange code"
                )
            
            token_data = token_response.json()
            
            if "error" in token_data:
                raise HTTPException(
                    status_code=400,
                    detail=token_data.get("error_description", "OAuth error")
                )
            
            tokens = OAuthTokens(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "bearer"),
                scope=token_data.get("scope")
            )
            
            # Get user info
            user_response = await client.get(
                f"{self.API_URL}/user",
                headers={
                    "Authorization": f"Bearer {tokens.access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if user_response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to fetch user info"
                )
            
            user_data = user_response.json()
            
            # Get primary email if not public
            email = user_data.get("email")
            if not email:
                emails_response = await client.get(
                    f"{self.API_URL}/user/emails",
                    headers={
                        "Authorization": f"Bearer {tokens.access_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                )
                if emails_response.status_code == 200:
                    emails = emails_response.json()
                    primary = next(
                        (e for e in emails if e.get("primary") and e.get("verified")),
                        None
                    )
                    if primary:
                        email = primary["email"]
            
            if not email:
                raise HTTPException(
                    status_code=400,
                    detail="Email is required but not available"
                )
            
            user_info = OAuthUserInfo(
                provider="github",
                provider_user_id=str(user_data["id"]),
                email=email,
                name=user_data.get("name") or user_data.get("login"),
                avatar_url=user_data.get("avatar_url"),
                username=user_data.get("login"),
                raw_data=user_data
            )
            
            return tokens, user_info


class OAuthManager:
    """
    Central OAuth manager for all providers.
    """
    
    def __init__(self, config: Optional[OAuthConfig] = None):
        self.config = config or get_oauth_config()
        self.google = GoogleOAuthHandler(self.config)
        self.github = GitHubOAuthHandler(self.config)
    
    def get_login_url(
        self,
        provider: str,
        redirect_after: Optional[str] = None
    ) -> str:
        """
        Get OAuth login URL for specified provider.
        
        Args:
            provider: OAuth provider (google, github)
            redirect_after: URL to redirect after successful auth
            
        Returns:
            Authorization URL
        """
        if provider == "google":
            return self.google.get_authorization_url(redirect_after)
        elif provider == "github":
            return self.github.get_authorization_url(redirect_after)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    async def handle_callback(
        self,
        provider: str,
        code: str,
        state: str,
        user_handler: Optional[Callable[[OAuthUserInfo, OAuthTokens], Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback.
        
        Args:
            provider: OAuth provider
            code: Authorization code
            state: State parameter
            user_handler: Optional callback to handle user creation/login
            
        Returns:
            Dict with user info and tokens
        """
        if provider == "google":
            tokens, user_info = await self.google.exchange_code(code, state)
        elif provider == "github":
            tokens, user_info = await self.github.exchange_code(code, state)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        
        result = {
            "user_info": user_info,
            "tokens": tokens,
            "redirect_after": self.google.state_manager.get_redirect_after(state) if provider == "google" 
                             else self.github.state_manager.get_redirect_after(state)
        }
        
        if user_handler:
            result["user"] = await user_handler(user_info, tokens)
        
        return result


# Global OAuth manager instance
_oauth_manager: Optional[OAuthManager] = None


def get_oauth_manager() -> OAuthManager:
    """Get or create global OAuth manager"""
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = OAuthManager()
    return _oauth_manager


# FastAPI route helpers
async def oauth_login(
    provider: str,
    redirect_after: Optional[str] = None
) -> RedirectResponse:
    """
    Initiate OAuth login flow.
    
    Usage in FastAPI:
        @app.get("/auth/{provider}/login")
        async def login(provider: str, redirect_after: Optional[str] = None):
            return await oauth_login(provider, redirect_after)
    """
    manager = get_oauth_manager()
    auth_url = manager.get_login_url(provider, redirect_after)
    return RedirectResponse(url=auth_url)


async def oauth_callback(
    provider: str,
    request: Request,
    user_handler: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Handle OAuth callback.
    
    Usage in FastAPI:
        @app.get("/auth/{provider}/callback")
        async def callback(provider: str, request: Request):
            return await oauth_callback(provider, request)
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")
    
    manager = get_oauth_manager()
    return await manager.handle_callback(provider, code, state, user_handler)
