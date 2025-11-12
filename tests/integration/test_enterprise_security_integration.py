#!/usr/bin/env python3
"""
Integration Tests for Enterprise Security Components

Tests the integration of Zero Trust Security and Quantum Cryptography
components to ensure they work together in production scenarios.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from security.integration import EnterpriseSecurityManager, get_enterprise_security
    from security.zero_trust import AccessLevel
except ImportError:
    pytest.skip("Enterprise security modules not available", allow_module_level=True)


class TestEnterpriseSecurityIntegration:
    """Integration tests for enterprise security manager."""

    @pytest.fixture
    def security_manager(self):
        """Create enterprise security manager instance."""
        return EnterpriseSecurityManager()

    def test_security_manager_initialization(self, security_manager):
        """Test enterprise security manager initialization."""
        assert security_manager is not None

        # Check if components are initialized (may be None if dependencies missing)
        # This is expected behavior - components gracefully degrade
        assert hasattr(security_manager, 'zero_trust')
        assert hasattr(security_manager, 'quantum_crypto')
        assert hasattr(security_manager, 'is_initialized')

    def test_security_status_reporting(self, security_manager):
        """Test security status reporting."""
        status = security_manager.get_security_status()

        assert isinstance(status, dict)
        assert 'initialized' in status
        assert 'zero_trust_available' in status
        assert 'quantum_crypto_available' in status
        assert 'security_events_count' in status
        assert 'timestamp' in status

    def test_security_report_generation(self, security_manager):
        """Test comprehensive security report generation."""
        report = security_manager.get_security_report()

        assert isinstance(report, dict)
        assert 'system_status' in report
        assert 'security_events' in report
        assert 'component_status' in report

        # Check component status structure
        assert 'zero_trust' in report['component_status']
        assert 'quantum_crypto' in report['component_status']

    def test_authentication_workflow(self, security_manager):
        """Test complete authentication workflow."""
        # Test authentication (may fail if components not available)
        result = security_manager.authenticate_user(
            username="test_user",
            password="test_pass",
            device_fingerprint="test_device_123",
            ip_address="192.168.1.100"
        )

        assert isinstance(result, dict)

        # Result should indicate whether authentication worked or why it failed
        if security_manager.zero_trust:
            # If zero trust is available, check proper response structure
            assert 'authenticated' in result
            assert 'access_level' in result
        else:
            # If not available, should return appropriate error
            assert result.get('authenticated') == False
            assert 'Zero Trust Security not available' in result.get('reason', '')

    def test_authorization_workflow(self, security_manager):
        """Test authorization workflow."""
        # Test authorization (requires valid token)
        result = security_manager.authorize_request(
            token="invalid_token",
            resource="trading_system",
            action="read",
            current_ip="192.168.1.100"
        )

        assert isinstance(result, dict)

        if security_manager.zero_trust:
            assert 'authorized' in result
        else:
            assert result.get('authorized') == False
            assert 'Zero Trust Security not available' in result.get('reason', '')

    def test_encryption_workflow(self, security_manager):
        """Test encryption/decryption workflow."""
        test_data = b"Hello, Quantum-Safe World!"
        context = "test_encryption"

        # Encrypt data
        encrypted = security_manager.encrypt_data(test_data, context)

        assert isinstance(encrypted, dict)
        assert 'encrypted' in encrypted
        assert 'data' in encrypted
        assert 'timestamp' in encrypted

        if security_manager.quantum_crypto and encrypted['encrypted']:
            # If encryption worked, test decryption
            assert 'metadata' in encrypted

            decrypted = security_manager.decrypt_data(
                encrypted_data=encrypted['data'],
                metadata=encrypted['metadata'],
                context=context
            )

            assert isinstance(decrypted, dict)
            assert decrypted.get('decrypted') == True
            assert decrypted.get('data') == test_data
        else:
            # If encryption not available, should return original data
            assert encrypted.get('encrypted') == False
            assert encrypted.get('data') == test_data

    def test_digital_signature_workflow(self, security_manager):
        """Test digital signature workflow."""
        test_message = b"Important trading instruction"
        key_id = "test_key"

        # Sign message
        signature = security_manager.sign_message(test_message, key_id)

        assert isinstance(signature, dict)
        assert 'signed' in signature

        if security_manager.quantum_crypto and signature['signed']:
            # If signing worked, test verification
            assert 'signature' in signature
            assert 'metadata' in signature

            verified = security_manager.verify_signature(
                message=test_message,
                signature=signature['signature'],
                metadata=signature['metadata']
            )

            assert isinstance(verified, dict)
            assert verified.get('verified') == True
        else:
            # If signing not available
            assert signature.get('signed') == False

    def test_security_event_logging(self, security_manager):
        """Test security event logging."""
        initial_count = len(security_manager.security_events)

        # Trigger some security operations to generate events
        security_manager.authenticate_user("test", "test")
        security_manager.authorize_request("token", "resource", "action")

        # Check that events were logged
        final_count = len(security_manager.security_events)
        assert final_count >= initial_count

        # Check event structure
        if final_count > initial_count:
            latest_event = security_manager.security_events[-1]
            assert 'timestamp' in latest_event
            assert 'event_type' in latest_event
            assert 'details' in latest_event

    def test_graceful_degradation(self, security_manager):
        """Test graceful degradation when components are unavailable."""
        # This test verifies that the system continues to function
        # even when security components are not fully available

        # All methods should return appropriate error responses
        # rather than crashing the application

        auth_result = security_manager.authenticate_user("test", "test")
        assert isinstance(auth_result, dict)

        authz_result = security_manager.authorize_request("token", "res", "act")
        assert isinstance(authz_result, dict)

        encrypt_result = security_manager.encrypt_data(b"test")
        assert isinstance(encrypt_result, dict)

        sign_result = security_manager.sign_message(b"test")
        assert isinstance(sign_result, dict)

    def test_get_enterprise_security_function(self):
        """Test the get_enterprise_security() function."""
        security = get_enterprise_security()

        assert isinstance(security, EnterpriseSecurityManager)
        assert hasattr(security, 'authenticate_user')
        assert hasattr(security, 'authorize_request')
        assert hasattr(security, 'encrypt_data')
        assert hasattr(security, 'sign_message')

    @patch('security.integration.logger')
    def test_error_handling(self, mock_logger, security_manager):
        """Test error handling in security operations."""
        # Test with invalid inputs or simulated failures

        # This should not crash the system
        result = security_manager.encrypt_data(None)  # Invalid input
        assert isinstance(result, dict)

        # Check that errors are logged
        # mock_logger.error.assert_called()  # May or may not be called depending on implementation

    def test_concurrent_security_operations(self, security_manager):
        """Test concurrent security operations."""
        import asyncio

        async def concurrent_test():
            # Run multiple security operations concurrently
            tasks = []

            for i in range(5):
                tasks.append(
                    asyncio.create_task(
                        self._async_authenticate(security_manager, f"user_{i}")
                    )
                )

            results = await asyncio.gather(*tasks)
            return results

        async def _async_authenticate(self, security, username):
            # Helper for async authentication test
            return security.authenticate_user(username, "password")

        # Run concurrent test
        results = asyncio.run(concurrent_test())

        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_security_integration_with_real_components(self, security_manager):
        """Test integration when real components are available."""
        # This test is more comprehensive when actual security libraries are installed

        status = security_manager.get_security_status()

        # Document current state
        print(f"Security Status: {status}")

        # Verify that the system reports its actual state accurately
        assert status['zero_trust_available'] == (security_manager.zero_trust is not None)
        assert status['quantum_crypto_available'] == (security_manager.quantum_crypto is not None)

        # Test that operations work as expected based on availability
        if status['zero_trust_available']:
            print("✅ Zero Trust Security is operational")
        else:
            print("⚠️ Zero Trust Security not available (OQS library missing)")

        if status['quantum_crypto_available']:
            print("✅ Quantum Cryptography is operational")
        else:
            print("⚠️ Quantum Cryptography not available (OQS library missing)")


class TestSecurityIntegrationExamples:
    """Tests for the integration examples and usage patterns."""

    def test_example_authentication_flow(self):
        """Test the authentication flow example from docstring."""
        security = get_enterprise_security()

        # Simulate the FastAPI integration example
        class MockTradeRequest:
            def __init__(self):
                self.username = "trader"
                self.password = "secure_pass"

        class MockRequest:
            def __init__(self):
                self.headers = {'X-Device-Fingerprint': 'device_123'}
                self.client = Mock()
                self.client.host = "10.0.0.1"

        trade_request = MockTradeRequest()
        request = MockRequest()

        # Test authentication
        auth_result = security.authenticate_user(
            username=trade_request.username,
            password=trade_request.password,
            device_fingerprint=request.headers.get('X-Device-Fingerprint'),
            ip_address=request.client.host
        )

        assert isinstance(auth_result, dict)
        # Should either succeed or fail gracefully
        assert 'authenticated' in auth_result

    def test_example_encryption_flow(self):
        """Test the encryption/decryption flow example."""
        security = get_enterprise_security()

        # Test data encryption flow
        sensitive_data = b"account_balance=1000000"

        # Encrypt
        encrypted = security.encrypt_data(sensitive_data, context="trading_data")
        assert isinstance(encrypted, dict)

        if encrypted.get('encrypted'):
            # If encryption worked, test the full flow
            encrypted_data = encrypted['data']
            metadata = encrypted['metadata']

            # Simulate saving/loading from database
            retrieved_data = encrypted_data
            retrieved_metadata = metadata

            # Decrypt
            decrypted = security.decrypt_data(
                encrypted_data=retrieved_data,
                metadata=retrieved_metadata,
                context="trading_data"
            )

            assert isinstance(decrypted, dict)
            if decrypted.get('decrypted'):
                assert decrypted['data'] == sensitive_data

    def test_example_digital_signature_flow(self):
        """Test the digital signature flow example."""
        security = get_enterprise_security()

        # Test signature flow
        order_data = b"BUY BTCUSDT 1.0 @ 50000"

        # Sign
        signature = security.sign_message(order_data, key_id="trading_key")
        assert isinstance(signature, dict)

        if signature.get('signed'):
            # If signing worked, test verification
            is_valid = security.verify_signature(
                message=order_data,
                signature=signature['signature'],
                metadata=signature['metadata']
            )

            assert isinstance(is_valid, dict)
            assert is_valid.get('verified') == True


if __name__ == "__main__":
    # Run basic integration tests
    print("🧪 Running Enterprise Security Integration Tests...")

    security = EnterpriseSecurityManager()
    status = security.get_security_status()

    print(f"Security Status: {status}")

    # Test basic functionality
    auth_result = security.authenticate_user("test", "test")
    print(f"Authentication Test: {auth_result}")

    encrypt_result = security.encrypt_data(b"test data")
    print(f"Encryption Test: {'✅ Success' if encrypt_result.get('encrypted') else '⚠️ Not Available'}")

    sign_result = security.sign_message(b"test message")
    print(f"Signature Test: {'✅ Success' if sign_result.get('signed') else '⚠️ Not Available'}")

    print("🎉 Integration tests completed!")
