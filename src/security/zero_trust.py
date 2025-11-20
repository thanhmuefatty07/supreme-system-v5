"""
Zero Trust Security Implementation - BeyondCorp Pattern

Implements Google's BeyondCorp security model for trading systems.
Every access request is authenticated, authorized, and continuously evaluated.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import jwt
import pyotp
import bcrypt
import hashlib
import ipaddress
from enum import Enum

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for zero trust."""
    DENIED = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


@dataclass
class UserContext:
    """User security context for zero trust evaluation."""
    user_id: str
    roles: List[str]
    device_fingerprint: str
    ip_address: str
    risk_score: float
    last_auth_time: datetime
    mfa_verified: bool = False
    session_id: str = ""


@dataclass
class AccessDecision:
    """Access control decision result."""
    allowed: bool
    access_level: AccessLevel
    confidence: float
    risk_factors: List[str]
    required_actions: List[str]
    expires_at: datetime
    reason: str


class ZeroTrustManager:
    """
    Zero Trust Security Manager.
    
    Implements continuous authentication and authorization:
    - No implicit trust based on network location
    - Every access request is verified
    - Context-aware access decisions
    - Continuous risk evaluation
    """

    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", self._generate_secure_secret())
        self.totp_window = 1  # 30-second windows
        self.max_session_duration = timedelta(hours=8)
        self.risk_threshold = 0.7
        
        # Load access policies
        self.policies = self._load_access_policies()
        
        # Initialize audit logging
        self.audit_events = []
        
        logger.info("🔐 Zero Trust Security Manager initialized")

    def _generate_secure_secret(self) -> str:
        """Generate secure JWT secret."""
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _load_access_policies(self) -> Dict[str, Any]:
        """Load zero trust access policies."""
        return {
            "admin": {
                "allowed_ips": ["192.168.1.0/24", "10.0.0.0/8"],
                "required_mfa": True,
                "max_risk_score": 0.3,
                "time_restrictions": {"start": "09:00", "end": "18:00"},
                "resources": ["*"]
            },
            "trader": {
                "allowed_ips": ["*"],  # Any IP but with MFA
                "required_mfa": True,
                "max_risk_score": 0.5,
                "geofencing": True,
                "resources": ["trading", "positions", "orders"]
            },
            "monitor": {
                "allowed_ips": ["*"],
                "required_mfa": False,
                "max_risk_score": 0.8,
                "read_only": True,
                "resources": ["dashboard", "metrics"]
            }
        }

    def authenticate(
        self,
        username: str,
        password: str,
        device_fingerprint: str,
        ip_address: str,
        mfa_token: Optional[str] = None
    ) -> Optional[str]:
        """
        Authenticate user with zero trust evaluation.
        
        Args:
            username: Username
            password: Password
            device_fingerprint: Unique device identifier
            ip_address: Client IP address
            mfa_token: Optional MFA token
            
        Returns:
            JWT token or None if authentication failed
        """
        # Step 1: Verify credentials
        if not self._verify_credentials(username, password):
            self._audit_log("AUTH_FAILED", username, ip_address, "invalid_credentials")
            return None
        
        # Step 2: Calculate initial risk score
        risk_score = self._calculate_risk_score(username, ip_address, device_fingerprint)
        
        # Step 3: Check if MFA required
        user_roles = self._get_user_roles(username)
        primary_role = user_roles[0] if user_roles else "monitor"
        policy = self.policies.get(primary_role, self.policies["monitor"])
        
        mfa_verified = False
        if policy["required_mfa"]:
            if not mfa_token:
                self._audit_log("AUTH_PENDING_MFA", username, ip_address, "mfa_required")
                return "MFA_REQUIRED"
            
            if not self.verify_mfa(username, mfa_token):
                self._audit_log("AUTH_FAILED", username, ip_address, "invalid_mfa")
                return None
            
            mfa_verified = True
            risk_score -= 0.2  # MFA reduces risk
        
        # Step 4: Generate user context
        user_context = UserContext(
            user_id=username,
            roles=user_roles,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            risk_score=max(0.0, risk_score),
            last_auth_time=datetime.now(),
            mfa_verified=mfa_verified,
            session_id=self._generate_session_id()
        )
        
        # Step 5: Create JWT token
        token = jwt.encode({
            "user_id": user_context.user_id,
            "roles": user_context.roles,
            "device_fingerprint": user_context.device_fingerprint,
            "ip_address": user_context.ip_address,
            "risk_score": user_context.risk_score,
            "mfa_verified": user_context.mfa_verified,
            "session_id": user_context.session_id,
            "iat": datetime.now().timestamp(),
            "exp": (datetime.now() + self.max_session_duration).timestamp()
        }, self.jwt_secret, algorithm="HS256")
        
        self._audit_log("AUTH_SUCCESS", username, ip_address, 
                       f"session:{user_context.session_id},mfa:{mfa_verified}")
        
        return token

    def authorize(
        self,
        token: str,
        resource: str,
        action: str,
        current_ip: str
    ) -> AccessDecision:
        """
        Authorize request with zero trust continuous evaluation.
        
        Args:
            token: JWT authentication token
            resource: Resource being accessed
            action: Action being performed (read/write/execute)
            current_ip: Current client IP
            
        Returns:
            AccessDecision with authorization result
        """
        try:
            # Decode and validate JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Reconstruct user context
            user_context = UserContext(
                user_id=payload["user_id"],
                roles=payload["roles"],
                device_fingerprint=payload["device_fingerprint"],
                ip_address=payload["ip_address"],
                risk_score=payload["risk_score"],
                last_auth_time=datetime.fromtimestamp(payload["iat"]),
                mfa_verified=payload.get("mfa_verified", False),
                session_id=payload.get("session_id", "")
            )
            
            # Continuous risk evaluation
            current_risk = self._evaluate_continuous_risk(user_context, current_ip)
            
            # Make access decision
            decision = self._make_access_decision(
                user_context, resource, action, current_risk
            )
            
            # Audit logging
            self._audit_log(
                "ACCESS_DECISION",
                user_context.user_id,
                current_ip,
                f"resource:{resource},action:{action},allowed:{decision.allowed},risk:{current_risk:.2f}"
            )
            
            return decision
            
        except jwt.ExpiredSignatureError:
            self._audit_log("TOKEN_EXPIRED", "unknown", current_ip, "jwt_expired")
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                ["token_expired"], ["re_authenticate"],
                datetime.now(), "Token expired"
            )
            
        except jwt.InvalidTokenError:
            self._audit_log("INVALID_TOKEN", "unknown", current_ip, "jwt_invalid")
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                ["invalid_token"], ["re_authenticate"],
                datetime.now(), "Invalid token"
            )

    def setup_mfa(self, username: str) -> Dict[str, str]:
        """
        Setup TOTP MFA for user.
        
        Args:
            username: Username to setup MFA
            
        Returns:
            Dictionary with secret and provisioning URI
        """
        # Generate random secret
        secret = pyotp.random_base32()
        
        # Store hashed secret (implementation needed)
        hashed_secret = bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()
        self._store_mfa_secret(username, hashed_secret)
        
        # Generate provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name="Supreme System V5"
        )
        
        self._audit_log("MFA_SETUP", username, "internal", "mfa_configured")
        
        return {
            "secret": secret,
            "provisioning_uri": provisioning_uri,
            "qr_code_url": f"https://chart.googleapis.com/chart?chs=200x200&cht=qr&chl={provisioning_uri}"
        }

    def verify_mfa(self, username: str, token: str) -> bool:
        """
        Verify MFA token.
        
        Args:
            username: Username
            token: 6-digit TOTP token
            
        Returns:
            True if token is valid
        """
        secret = self._retrieve_mfa_secret(username)
        if not secret:
            logger.warning(f"MFA secret not found for {username}")
            return False
        
        totp = pyotp.TOTP(secret)
        is_valid = totp.verify(token, valid_window=self.totp_window)
        
        self._audit_log(
            "MFA_VERIFICATION",
            username,
            "internal",
            f"result:{is_valid}"
        )
        
        return is_valid

    def _calculate_risk_score(
        self,
        username: str,
        ip_address: str,
        device_fingerprint: str
    ) -> float:
        """Calculate initial risk score."""
        risk_score = 0.0
        
        # IP reputation check
        if self._is_suspicious_ip(ip_address):
            risk_score += 0.4
            logger.warning(f"Suspicious IP detected: {ip_address}")
        
        # Known device check
        if self._is_known_device(username, device_fingerprint):
            risk_score -= 0.2
        else:
            risk_score += 0.3
            logger.info(f"New device detected for {username}")
        
        # Time-based risk
        if self._is_business_hours():
            risk_score -= 0.1
        else:
            risk_score += 0.2
        
        # Location-based risk (geofencing)
        if not self._is_known_location(ip_address):
            risk_score += 0.3
            logger.warning(f"Unknown location: {ip_address}")
        
        return max(0.0, min(1.0, risk_score))

    def _evaluate_continuous_risk(
        self,
        user_context: UserContext,
        current_ip: str
    ) -> float:
        """Continuous risk evaluation during session."""
        risk_score = user_context.risk_score
        
        # IP change detection
        if current_ip != user_context.ip_address:
            risk_score += 0.3
            logger.warning(
                f"IP changed for {user_context.user_id}: "
                f"{user_context.ip_address} → {current_ip}"
            )
        
        # Session age check
        session_age = datetime.now() - user_context.last_auth_time
        if session_age > timedelta(hours=4):
            risk_score += 0.2
        
        # Activity pattern anomaly detection
        if self._detect_anomalous_activity(user_context):
            risk_score += 0.4
            logger.warning(f"Anomalous activity detected for {user_context.user_id}")
        
        return max(0.0, min(1.0, risk_score))

    def _make_access_decision(
        self,
        user_context: UserContext,
        resource: str,
        action: str,
        risk_score: float
    ) -> AccessDecision:
        """Make final access decision with zero trust evaluation."""
        
        # Get primary role policy
        primary_role = user_context.roles[0] if user_context.roles else "monitor"
        policy = self.policies.get(primary_role, self.policies["monitor"])
        
        risk_factors = []
        required_actions = []
        
        # Risk threshold check
        if risk_score > policy["max_risk_score"]:
            risk_factors.append(f"risk_score_high:{risk_score:.2f}")
            required_actions.append("step_up_authentication")
            
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                risk_factors, required_actions,
                datetime.now() + timedelta(minutes=5),
                f"Risk score {risk_score:.2f} exceeds threshold {policy['max_risk_score']}"
            )
        
        # IP restriction check
        if not self._check_ip_allowed(user_context.ip_address, policy["allowed_ips"]):
            risk_factors.append("ip_not_allowed")
            required_actions.append("use_vpn")
            
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                risk_factors, required_actions,
                datetime.now() + timedelta(minutes=1),
                f"IP {user_context.ip_address} not in allowed list"
            )
        
        # MFA requirement check
        if policy["required_mfa"] and not user_context.mfa_verified:
            risk_factors.append("mfa_not_verified")
            required_actions.append("complete_mfa")
            
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                risk_factors, required_actions,
                datetime.now() + timedelta(minutes=5),
                "MFA verification required"
            )
        
        # Role-based access control
        if not self._check_resource_access(user_context.roles, resource, policy):
            risk_factors.append("insufficient_permissions")
            required_actions.append("request_elevation")
            
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                risk_factors, required_actions,
                datetime.now() + timedelta(minutes=1),
                f"Insufficient permissions for resource: {resource}"
            )
        
        # Time-based restrictions
        if not self._check_time_restrictions(policy):
            risk_factors.append("outside_business_hours")
            required_actions.append("schedule_request")
            
            return AccessDecision(
                False, AccessLevel.DENIED, 0.0,
                risk_factors, required_actions,
                datetime.now() + timedelta(hours=12),
                "Access outside permitted hours"
            )
        
        # Action-based access level
        access_level = self._determine_access_level(action, policy)
        
        # Grant access
        return AccessDecision(
            True, access_level,
            1.0 - risk_score,
            [], [],
            datetime.now() + timedelta(minutes=30),
            "Access granted"
        )

    def _check_ip_allowed(self, ip: str, allowed_patterns: List[str]) -> bool:
        """Check if IP is allowed by policy."""
        if "*" in allowed_patterns:
            return True
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for pattern in allowed_patterns:
                try:
                    network = ipaddress.ip_network(pattern, strict=False)
                    if ip_obj in network:
                        return True
                except ValueError:
                    if pattern == ip:
                        return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return False
        
        return False

    def _check_resource_access(
        self,
        roles: List[str],
        resource: str,
        policy: Dict
    ) -> bool:
        """Check if roles have access to resource."""
        allowed_resources = policy.get("resources", [])
        
        if "*" in allowed_resources:
            return True
        
        # Check if resource matches allowed patterns
        for allowed in allowed_resources:
            if resource.startswith(allowed):
                return True
        
        return False

    def _check_time_restrictions(self, policy: Dict) -> bool:
        """Check time-based access restrictions."""
        if "time_restrictions" not in policy:
            return True
        
        restrictions = policy["time_restrictions"]
        current_time = datetime.now().time()
        
        start_time = datetime.strptime(restrictions["start"], "%H:%M").time()
        end_time = datetime.strptime(restrictions["end"], "%H:%M").time()
        
        return start_time <= current_time <= end_time

    def _determine_access_level(self, action: str, policy: Dict) -> AccessLevel:
        """Determine access level based on action and policy."""
        if policy.get("read_only", False):
            return AccessLevel.READ
        
        if action in ["read", "view", "list"]:
            return AccessLevel.READ
        elif action in ["write", "create", "update"]:
            return AccessLevel.WRITE
        elif action in ["delete", "admin", "configure"]:
            return AccessLevel.ADMIN
        
        return AccessLevel.READ

    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify username and password."""
        # TODO: Implement database lookup
        # For now, placeholder
        stored_hash = self._get_password_hash(username)
        if not stored_hash:
            return False
        
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def _get_user_roles(self, username: str) -> List[str]:
        """Get user roles from database."""
        # TODO: Implement database lookup
        # Placeholder role mapping
        role_mapping = {
            "admin": ["admin"],
            "trader": ["trader"],
            "monitor": ["monitor"]
        }
        return role_mapping.get(username, ["monitor"])

    def _calculate_risk_score(
        self,
        username: str,
        ip_address: str,
        device_fingerprint: str
    ) -> float:
        """Calculate comprehensive risk score."""
        risk_score = 0.0
        
        # IP reputation
        if self._is_suspicious_ip(ip_address):
            risk_score += 0.4
        
        # Device trust
        if not self._is_known_device(username, device_fingerprint):
            risk_score += 0.3
        else:
            risk_score -= 0.2
        
        # Time context
        if not self._is_business_hours():
            risk_score += 0.2
        
        # Login history
        if self._has_recent_failed_attempts(username):
            risk_score += 0.3
        
        return max(0.0, min(1.0, risk_score))

    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check IP reputation."""
        # TODO: Integrate with IP reputation service
        # Placeholder: check if IP is from known VPN/proxy ranges
        suspicious_ranges = [
            "tor_exit_nodes",  # Tor network
            "known_vpn_ranges"  # VPN services
        ]
        return False  # Placeholder

    def _is_known_device(self, username: str, device_fingerprint: str) -> bool:
        """Check if device is known for this user."""
        # TODO: Implement device registry lookup
        return False  # Conservative default

    def _is_known_location(self, ip: str) -> bool:
        """Check if IP is from known location."""
        # TODO: Implement geolocation check
        return True  # Placeholder

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        current_hour = datetime.now().hour
        return 9 <= current_hour <= 18

    def _detect_anomalous_activity(self, user_context: UserContext) -> bool:
        """Detect anomalous user activity patterns."""
        # TODO: Implement ML-based anomaly detection
        return False  # Placeholder

    def _has_recent_failed_attempts(self, username: str) -> bool:
        """Check for recent failed authentication attempts."""
        # TODO: Implement failed attempt tracking
        return False  # Placeholder

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    def _get_password_hash(self, username: str) -> Optional[str]:
        """Get stored password hash for user."""
        # TODO: Implement database lookup
        return None

    def _store_mfa_secret(self, username: str, secret: str):
        """Store MFA secret securely."""
        # TODO: Implement secure storage
        pass

    def _retrieve_mfa_secret(self, username: str) -> Optional[str]:
        """Retrieve MFA secret securely."""
        # TODO: Implement secure retrieval
        return None

    def _audit_log(self, event_type: str, user_id: str, ip_address: str, details: str):
        """Comprehensive audit logging."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details,
            "severity": self._calculate_event_severity(event_type)
        }
        
        # Store audit event
        self.audit_events.append(audit_entry)
        
        # Log to system logger
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")
        
        # Send to SIEM (if configured)
        self._send_to_siem(audit_entry)

    def _calculate_event_severity(self, event_type: str) -> str:
        """Calculate event severity for alerting."""
        severity_map = {
            "AUTH_FAILED": "HIGH",
            "INVALID_TOKEN": "HIGH",
            "ACCESS_DENIED": "MEDIUM",
            "IP_CHANGE": "MEDIUM",
            "TOKEN_EXPIRED": "LOW",
            "AUTH_SUCCESS": "INFO",
            "ACCESS_GRANTED": "INFO"
        }
        return severity_map.get(event_type, "INFO")

    def _send_to_siem(self, audit_entry: Dict):
        """Send audit entry to SIEM system."""
        # TODO: Implement SIEM integration
        # Examples: Datadog, Splunk, ELK, Azure Sentinel
        pass

    def get_audit_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve audit logs with filtering."""
        filtered_logs = self.audit_events
        
        if start_time:
            filtered_logs = [
                log for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) >= start_time
            ]
        
        if end_time:
            filtered_logs = [
                log for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) <= end_time
            ]
        
        if user_id:
            filtered_logs = [
                log for log in filtered_logs
                if log["user_id"] == user_id
            ]
        
        if event_type:
            filtered_logs = [
                log for log in filtered_logs
                if log["event_type"] == event_type
            ]
        
        return filtered_logs


# Global instance
_zero_trust_manager: Optional[ZeroTrustManager] = None


def get_zero_trust_manager() -> ZeroTrustManager:
    """Get or create global zero trust manager instance."""
    global _zero_trust_manager
    if _zero_trust_manager is None:
        _zero_trust_manager = ZeroTrustManager()
    return _zero_trust_manager


# Backward compatibility alias
# Some code may import ZeroTrustSecurity instead of ZeroTrustManager
ZeroTrustSecurity = ZeroTrustManager