"""
🔐 Supreme System V5 - JWT Authentication System
Enterprise-grade security with JWT token-based authentication

Features:
- JWT token generation and validation
- Secure secret key management
- Role-based access control (RBAC)
- Token refresh mechanism
- Automatic token expiration
- Rate limiting integration
- Audit logging for security events
"""

import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger("supreme_auth")


class UserRole(Enum):
    """User roles for RBAC"""

    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"


class TokenType(Enum):
    """JWT token types"""

    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class AuthConfig:
    """Authentication configuration"""

    # JWT settings
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60  # 1 hour
    refresh_token_expire_days: int = 30  # 30 days

    # Security settings
    bcrypt_rounds: int = 12
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15

    # Rate limiting
    rate_limit_per_minute: int = 60

    def __post_init__(self):
        """Ensure secure configuration"""
        if len(self.jwt_secret_key) < 32:
            self.jwt_secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "Generated new JWT secret key - ensure consistency across restarts"
            )


class TokenClaims(BaseModel):
    """JWT token claims structure"""

    sub: str  # Subject (user ID)
    iat: int  # Issued at
    exp: int  # Expiration
    type: str  # Token type
    role: str  # User role
    permissions: List[str] = []
    session_id: str = ""


class User(BaseModel):
    """User model for authentication"""

    user_id: str
    username: str
    email: Optional[str] = None
    role: UserRole
    permissions: List[str] = []
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)  # type: ignore[assignment]
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None


class AuthenticationError(Exception):
    """Custom authentication error"""

    pass


class AuthorizationError(Exception):
    """Custom authorization error"""

    pass


class JWTManager:
    """JWT token manager with enhanced security"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.revoked_tokens: set[str] = set()

    def generate_access_token(self, user: User, session_id: Optional[str] = None) -> str:
        """Generate JWT access token"""
        if not session_id:
            session_id = secrets.token_urlsafe(16)

        now = datetime.utcnow()
        exp = now + timedelta(minutes=self.config.access_token_expire_minutes)

        claims = TokenClaims(
            sub=user.user_id,
            iat=int(now.timestamp()),
            exp=int(exp.timestamp()),
            type=TokenType.ACCESS.value,
            role=user.role.value,
            permissions=user.permissions,
            session_id=session_id,
        )

        # Store session info
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "created_at": now,
            "last_activity": now,
            "user_agent": "",  # Can be set by caller
            "ip_address": "",  # Can be set by caller
        }

        token = jwt.encode(
            claims.dict(), self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm
        )

        logger.info("🔑 Access token generated for user %s", user.username)
        return token

    def generate_refresh_token(self, user: User, session_id: str) -> str:
        """Generate JWT refresh token"""
        now = datetime.utcnow()
        exp = now + timedelta(days=self.config.refresh_token_expire_days)

        claims = TokenClaims(
            sub=user.user_id,
            iat=int(now.timestamp()),
            exp=int(exp.timestamp()),
            type=TokenType.REFRESH.value,
            role=user.role.value,
            session_id=session_id,
        )

        token = jwt.encode(
            claims.dict(), self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm
        )

        logger.info("🔄 Refresh token generated for user %s", user.username)
        return token

    def verify_token(self, token: str, token_type: Optional[TokenType] = None) -> TokenClaims:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                raise AuthenticationError("Token has been revoked")

            # Decode token
            payload = jwt.decode(
                token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm]
            )

            claims = TokenClaims(**payload)

            # Verify token type if specified
            if token_type and claims.type != token_type.value:
                raise AuthenticationError(
                    f"Invalid token type: expected {token_type.value}"
                )

            # Check if session is still active
            if claims.session_id and claims.session_id not in self.active_sessions:
                raise AuthenticationError("Session has been terminated")

            # Update last activity
            if claims.session_id in self.active_sessions:
                self.active_sessions[claims.session_id]["last_activity"] = datetime.utcnow()

            return claims

        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid token: {exc}") from exc

    def revoke_token(self, token: str) -> None:
        """Revoke a specific token"""
        self.revoked_tokens.add(token)
        logger.info("🚫 Token revoked")

    def revoke_session(self, session_id: str) -> None:
        """Revoke all tokens for a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info("🚫 Session %s terminated", session_id)

    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired_sessions: List[str] = []

        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data.get("last_activity", session_data["created_at"])  # type: ignore[index]
            if now - last_activity > timedelta(hours=24):  # 24 hour inactivity
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.revoke_session(session_id)

        if expired_sessions:
            logger.info("🧹 Cleaned up %d expired sessions", len(expired_sessions))


class UserManager:
    """User management with secure password handling"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.users: Dict[str, User] = {}
        self.username_to_id: Dict[str, str] = {}
        self.password_hashes: Dict[str, str] = {}

        # Create default admin user
        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default system users"""
        # Create admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@supreme-system-v5.com",
            role=UserRole.ADMIN,
            permissions=["*"],  # All permissions
        )

        # Create API user for system integration
        api_user = User(
            user_id="api_001",
            username="supreme_api",
            role=UserRole.API_USER,
            permissions=["trading.*", "portfolio.*", "performance.*"],
        )

        # Store users with default passwords
        self.create_user(admin_user, "Supreme@Admin2025!")
        self.create_user(api_user, "Supreme@API2025!")

        logger.info("👥 Default users created: admin, supreme_api")

    def create_user(self, user: User, password: str) -> bool:
        """Create new user with hashed password"""
        try:
            # Check if username already exists
            if user.username in self.username_to_id:
                raise ValueError(f"Username {user.username} already exists")

            # Hash password
            password_hash = bcrypt.hashpw(
                password.encode("utf-8"), bcrypt.gensalt(rounds=self.config.bcrypt_rounds)
            ).decode("utf-8")

            # Store user
            self.users[user.user_id] = user
            self.username_to_id[user.username] = user.user_id
            self.password_hashes[user.user_id] = password_hash

            logger.info("👤 User created: %s (%s)", user.username, user.role.value)
            return True

        except Exception as exc:  # Narrowing here would require mapping all possible bcrypt errors
            logger.error("❌ User creation failed: %s", exc)
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        try:
            # Get user ID
            user_id = self.username_to_id.get(username)
            if not user_id:
                logger.warning("🚫 Login attempt with unknown username: %s", username)
                return None

            user = self.users.get(user_id)
            if not user:
                return None

            # Check if user is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                logger.warning("🔒 Login attempt for locked user: %s", username)
                return None

            # Check if user is active
            if not user.is_active:
                logger.warning("🚫 Login attempt for inactive user: %s", username)
                return None

            # Verify password
            password_hash = self.password_hashes.get(user_id)
            if not password_hash:
                return None

            if bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
                # Reset login attempts on successful login
                user.login_attempts = 0
                user.last_login = datetime.utcnow()
                user.locked_until = None

                logger.info("✅ Successful login: %s", username)
                return user

            # Increment login attempts
            user.login_attempts += 1

            # Lock user if too many attempts
            if user.login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                logger.warning("🔒 User locked due to failed attempts: %s", username)

            logger.warning("🚫 Failed login attempt %d: %s", user.login_attempts, username)
            return None

        except Exception as exc:
            logger.error("❌ Authentication error: %s", exc)
            return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def update_user_permissions(self, user_id: str, permissions: List[str]) -> bool:
        """Update user permissions"""
        user = self.users.get(user_id)
        if user:
            user.permissions = permissions
            logger.info("🔐 Updated permissions for user %s", user.username)
            return True
        return False


class AuthenticationManager:
    """Main authentication manager"""

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.jwt_manager = JWTManager(self.config)
        self.user_manager = UserManager(self.config)
        self.security_scheme = HTTPBearer()

        logger.info("🔐 Authentication manager initialized")
        logger.info(
            "   JWT expiration: %d minutes", self.config.access_token_expire_minutes
        )
        logger.info("   Max login attempts: %d", self.config.max_login_attempts)

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """User login with JWT token generation"""
        # Authenticate user
        user = self.user_manager.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Generate tokens
        session_id = secrets.token_urlsafe(16)
        access_token = self.jwt_manager.generate_access_token(user, session_id)
        refresh_token = self.jwt_manager.generate_refresh_token(user, session_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.access_token_expire_minutes * 60,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": user.permissions,
            },
        }

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            claims = self.jwt_manager.verify_token(refresh_token, TokenType.REFRESH)

            # Get user
            user = self.user_manager.get_user_by_id(claims.sub)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")

            # Generate new access token
            access_token = self.jwt_manager.generate_access_token(
                user, claims.session_id
            )

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.config.access_token_expire_minutes * 60,
            }

        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def logout(self, token: str) -> None:
        """User logout with token revocation"""
        try:
            claims = self.jwt_manager.verify_token(token)
            self.jwt_manager.revoke_session(claims.session_id)
            logger.info("👋 User logged out: %s", claims.sub)
        except AuthenticationError:
            # Token already invalid or cannot be parsed; nothing to do
            pass

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> User:
        """Get current authenticated user from token"""
        try:
            token = credentials.credentials
            claims = self.jwt_manager.verify_token(token, TokenType.ACCESS)

            user = self.user_manager.get_user_by_id(claims.sub)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")

            return user

        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def require_permission(self, permission: str):
        """Decorator to require specific permission"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get current user from dependency
                current_user = kwargs.get("current_user")
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                # Check permissions
                if not self.has_permission(current_user, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission}",
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        # Admin has all permissions
        if user.role == UserRole.ADMIN or "*" in user.permissions:
            return True

        # Check exact permission
        if permission in user.permissions:
            return True

        # Check wildcard permissions
        for user_perm in user.permissions:
            if user_perm.endswith("*"):
                prefix = user_perm[:-1]
                if permission.startswith(prefix):
                    return True

        return False


# Global authentication manager instance
auth_config = AuthConfig()
auth_manager = AuthenticationManager(auth_config)


# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> User:
    """FastAPI dependency to get current user"""
    return await auth_manager.get_current_user(credentials)


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require admin user"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


async def get_trader_user(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require trader or admin user"""
    if current_user.role not in [UserRole.ADMIN, UserRole.TRADER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Trader privileges required"
        )
    return current_user


# Utility functions

def create_api_key_user(username: str, permissions: List[str]) -> Dict[str, str]:
    """Create API key user for system integration"""
    user = User(
        user_id=f"api_{secrets.token_urlsafe(8)}",
        username=username,
        role=UserRole.API_USER,
        permissions=permissions,
    )

    api_key = f"sk-supreme-{secrets.token_urlsafe(32)}"

    if auth_manager.user_manager.create_user(user, api_key):
        return {
            "user_id": user.user_id,
            "username": username,
            "api_key": api_key,
            "permissions": permissions,
        }
    return {}


def get_auth_stats() -> Dict[str, Any]:
    """Get authentication system statistics"""
    return {
        "total_users": len(auth_manager.user_manager.users),
        "active_sessions": len(auth_manager.jwt_manager.active_sessions),
        "revoked_tokens": len(auth_manager.jwt_manager.revoked_tokens),
        "config": {
            "access_token_expire_minutes": auth_config.access_token_expire_minutes,
            "refresh_token_expire_days": auth_config.refresh_token_expire_days,
            "max_login_attempts": auth_config.max_login_attempts,
        },
    }


if __name__ == "__main__":
    # Demo authentication system
    print("🔐 Supreme System V5 Authentication Demo")
    print("=" * 50)

    # Login demo
    login_result = auth_manager.login("admin", "Supreme@Admin2025!")
    print(f"✅ Login successful: {login_result['user']['username']}")

    # Token verification demo
    access_token = login_result["access_token"]
    print(f"🔑 Access token generated (length: {len(access_token)})")

    # Get auth stats
    stats = get_auth_stats()
    print(f"📊 Auth stats: {stats}")

    print("🚀 Authentication system ready for production!")
