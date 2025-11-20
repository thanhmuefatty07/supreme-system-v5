#!/usr/bin/env python3
"""
Enterprise Security Integration Module

Integrates Zero Trust Security and Quantum Cryptography into the main application.
Provides unified security interfaces for production deployment.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .zero_trust import ZeroTrustSecurity, AccessLevel
from .quantum_crypto import QuantumCryptography

logger = logging.getLogger(__name__)


class EnterpriseSecurityManager:
    """
    Unified enterprise security manager integrating all security components.

    Features:
    - Zero Trust access control
    - Quantum-safe cryptography
    - Security monitoring and alerting
    - Production-ready security policies
    """

    def __init__(self):
        """Initialize enterprise security manager."""
        try:
            self.zero_trust = ZeroTrustSecurity()
            logger.info("✅ Zero Trust Security initialized")
        except Exception as e:
            logger.error(f"❌ Zero Trust Security failed: {e}")
            self.zero_trust = None

        try:
            self.quantum_crypto = QuantumCryptography()
            logger.info("✅ Quantum Cryptography initialized")
        except Exception as e:
            logger.error(f"❌ Quantum Cryptography failed: {e}")
            self.quantum_crypto = None

        self.security_events = []
        self.is_initialized = self.zero_trust is not None or self.quantum_crypto is not None

        if self.is_initialized:
            logger.info("🚀 Enterprise Security Manager initialized successfully")
        else:
            logger.warning("⚠️ Enterprise Security Manager partially initialized")

    def authenticate_user(self, username: str, password: str, device_fingerprint: str = None,
                         ip_address: str = None) -> Dict[str, Any]:
        """
        Authenticate user with zero trust security.

        Args:
            username: User identifier
            password: User password
            device_fingerprint: Device fingerprint for trust evaluation
            ip_address: Client IP address

        Returns:
            Authentication result with access token if successful
        """
        if not self.zero_trust:
            return {
                'authenticated': False,
                'reason': 'Zero Trust Security not available',
                'access_level': AccessLevel.DENIED
            }

        try:
            # Authenticate with zero trust
            auth_result = self.zero_trust.authenticate_user(
                username=username,
                password=password,
                device_fingerprint=device_fingerprint,
                ip_address=ip_address
            )

            # Log security event
            self._log_security_event('authentication', {
                'username': username,
                'ip_address': ip_address,
                'success': auth_result.get('authenticated', False)
            })

            return auth_result

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                'authenticated': False,
                'reason': f'Authentication failed: {str(e)}',
                'access_level': AccessLevel.DENIED
            }

    def authorize_request(self, token: str, resource: str, action: str,
                         current_ip: str = None) -> Dict[str, Any]:
        """
        Authorize request with continuous evaluation.

        Args:
            token: JWT access token
            resource: Resource being accessed
            action: Action being performed
            current_ip: Current client IP

        Returns:
            Authorization result
        """
        if not self.zero_trust:
            return {
                'authorized': False,
                'reason': 'Zero Trust Security not available'
            }

        try:
            # Authorize with zero trust
            auth_result = self.zero_trust.authorize_request(
                token=token,
                resource=resource,
                action=action,
                current_ip=current_ip
            )

            # Log security event
            self._log_security_event('authorization', {
                'resource': resource,
                'action': action,
                'ip_address': current_ip,
                'authorized': auth_result.get('authorized', False)
            })

            return auth_result

        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return {
                'authorized': False,
                'reason': f'Authorization failed: {str(e)}'
            }

    def encrypt_data(self, data: bytes, context: str = "trading_data") -> Dict[str, Any]:
        """
        Encrypt data using quantum-safe cryptography.

        Args:
            data: Data to encrypt
            context: Encryption context for key derivation

        Returns:
            Encrypted data with metadata
        """
        if not self.quantum_crypto:
            return {
                'encrypted': False,
                'reason': 'Quantum Cryptography not available',
                'data': data  # Return original data
            }

        try:
            # Encrypt with quantum-safe algorithm
            encrypted_data, metadata = self.quantum_crypto.encrypt_data(data, context)

            # Log security event
            self._log_security_event('encryption', {
                'context': context,
                'algorithm': metadata.get('algorithm', 'unknown'),
                'success': True
            })

            return {
                'encrypted': True,
                'data': encrypted_data,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return {
                'encrypted': False,
                'reason': f'Encryption failed: {str(e)}',
                'data': data
            }

    def decrypt_data(self, encrypted_data: bytes, metadata: Dict[str, Any],
                    context: str = "trading_data") -> Dict[str, Any]:
        """
        Decrypt data using quantum-safe cryptography.

        Args:
            encrypted_data: Data to decrypt
            metadata: Encryption metadata
            context: Decryption context

        Returns:
            Decrypted data
        """
        if not self.quantum_crypto:
            return {
                'decrypted': False,
                'reason': 'Quantum Cryptography not available',
                'data': encrypted_data
            }

        try:
            # Decrypt with quantum-safe algorithm
            decrypted_data = self.quantum_crypto.decrypt_data(encrypted_data, metadata, context)

            # Log security event
            self._log_security_event('decryption', {
                'context': context,
                'algorithm': metadata.get('algorithm', 'unknown'),
                'success': True
            })

            return {
                'decrypted': True,
                'data': decrypted_data,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return {
                'decrypted': False,
                'reason': f'Decryption failed: {str(e)}',
                'data': encrypted_data
            }

    def sign_message(self, message: bytes, key_id: str = "default") -> Dict[str, Any]:
        """
        Sign message with quantum-safe digital signature.

        Args:
            message: Message to sign
            key_id: Key identifier for signing

        Returns:
            Digital signature with metadata
        """
        if not self.quantum_crypto:
            return {
                'signed': False,
                'reason': 'Quantum Cryptography not available'
            }

        try:
            # Sign with quantum-safe algorithm
            signature, metadata = self.quantum_crypto.sign_message(message, key_id)

            # Log security event
            self._log_security_event('signing', {
                'key_id': key_id,
                'algorithm': metadata.get('algorithm', 'unknown'),
                'success': True
            })

            return {
                'signed': True,
                'signature': signature,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Signing error: {e}")
            return {
                'signed': False,
                'reason': f'Signing failed: {str(e)}'
            }

    def verify_signature(self, message: bytes, signature: bytes,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify quantum-safe digital signature.

        Args:
            message: Original message
            signature: Digital signature
            metadata: Signature metadata

        Returns:
            Verification result
        """
        if not self.quantum_crypto:
            return {
                'verified': False,
                'reason': 'Quantum Cryptography not available'
            }

        try:
            # Verify with quantum-safe algorithm
            is_valid = self.quantum_crypto.verify_signature(message, signature, metadata)

            # Log security event
            self._log_security_event('verification', {
                'algorithm': metadata.get('algorithm', 'unknown'),
                'valid': is_valid
            })

            return {
                'verified': is_valid,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                'verified': False,
                'reason': f'Verification failed: {str(e)}'
            }

    def get_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status.

        Returns:
            Security system status and health metrics
        """
        return {
            'initialized': self.is_initialized,
            'zero_trust_available': self.zero_trust is not None,
            'quantum_crypto_available': self.quantum_crypto is not None,
            'security_events_count': len(self.security_events),
            'last_security_event': self.security_events[-1] if self.security_events else None,
            'timestamp': datetime.now().isoformat()
        }

    def get_security_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive security report.

        Returns:
            Detailed security system report
        """
        report = {
            'system_status': self.get_security_status(),
            'security_events': self.security_events[-100:],  # Last 100 events
            'component_status': {}
        }

        # Zero Trust status
        if self.zero_trust:
            report['component_status']['zero_trust'] = {
                'status': 'operational',
                'active_sessions': getattr(self.zero_trust, '_active_sessions', 0),
                'failed_auth_attempts': getattr(self.zero_trust, '_failed_attempts', 0)
            }
        else:
            report['component_status']['zero_trust'] = {
                'status': 'unavailable',
                'reason': 'OQS library not installed'
            }

        # Quantum Crypto status
        if self.quantum_crypto:
            report['component_status']['quantum_crypto'] = {
                'status': 'operational',
                'algorithms_available': getattr(self.quantum_crypto, '_available_algorithms', []),
                'keys_generated': getattr(self.quantum_crypto, '_keys_generated', 0)
            }
        else:
            report['component_status']['quantum_crypto'] = {
                'status': 'unavailable',
                'reason': 'OQS library not installed'
            }

        return report

    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log security event for monitoring.

        Args:
            event_type: Type of security event
            details: Event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }

        self.security_events.append(event)

        # Keep only last 1000 events to prevent memory issues
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        logger.info(f"🔐 Security Event: {event_type} - {details}")


# Global enterprise security manager instance
enterprise_security = EnterpriseSecurityManager()


def get_enterprise_security() -> EnterpriseSecurityManager:
    """
    Get the global enterprise security manager instance.

    Returns:
        EnterpriseSecurityManager instance
    """
    return enterprise_security


def init_enterprise_security():
    """
    Initialize enterprise security components.

    Call this function during application startup to ensure
    all security components are properly initialized.
    """
    global enterprise_security
    enterprise_security = EnterpriseSecurityManager()

    if enterprise_security.is_initialized:
        logger.info("🚀 Enterprise Security initialized successfully")
        return True
    else:
        logger.warning("⚠️ Enterprise Security initialization failed")
        return False


# Example usage and integration points
"""
Integration Examples:

1. FastAPI Integration:
```python
from fastapi import Depends, HTTPException
from src.security.integration import get_enterprise_security

@app.post("/api/trade")
async def execute_trade(
    trade_request: TradeRequest,
    security: EnterpriseSecurityManager = Depends(get_enterprise_security)
):
    # Authenticate user
    auth_result = security.authenticate_user(
        username=trade_request.username,
        password=trade_request.password,
        device_fingerprint=request.headers.get('X-Device-Fingerprint'),
        ip_address=request.client.host
    )

    if not auth_result['authenticated']:
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Authorize action
    authz_result = security.authorize_request(
        token=auth_result['token'],
        resource="trading",
        action="execute_trade",
        current_ip=request.client.host
    )

    if not authz_result['authorized']:
        raise HTTPException(status_code=403, detail="Authorization failed")

    # Execute trade with encrypted data
    # ... trading logic ...
```

2. Data Encryption Integration:
```python
from src.security.integration import get_enterprise_security

security = get_enterprise_security()

# Encrypt sensitive trading data
sensitive_data = b"account_balance=1000000"
encrypted = security.encrypt_data(sensitive_data, context="trading_data")

# Store encrypted data
save_to_database(encrypted['data'], encrypted['metadata'])

# Decrypt when needed
decrypted = security.decrypt_data(
    encrypted_data=retrieved_data,
    metadata=retrieved_metadata,
    context="trading_data"
)
```

3. Digital Signature Integration:
```python
# Sign trading order
order_data = b"BUY BTCUSDT 1.0 @ 50000"
signature = security.sign_message(order_data, key_id="trading_key")

# Verify signature
is_valid = security.verify_signature(
    message=order_data,
    signature=signature['signature'],
    metadata=signature['metadata']
)
```
"""
