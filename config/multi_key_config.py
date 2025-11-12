#!/usr/bin/env python3
"""
Multi-Key Configuration for Enterprise AI Optimizer

This file contains the configuration for multi-API key rotation
to avoid Gemini quota limits and ensure reliable operation.
"""

import os
from typing import List, Optional


class MultiKeyConfig:
    """Configuration for multi-API key enterprise operation."""

    # ============================================================================
    # GEMINI API KEYS - ADD YOUR KEYS HERE (5-100 keys recommended)
    # ============================================================================
    # Create multiple Google Cloud projects and generate API keys
    # Visit: https://makersuite.google.com/app/apikey
    #
    # WARNING: Never commit real API keys to version control
    # Use environment variables or secure secret management

    GEMINI_KEYS: List[str] = [
        # Add your actual Gemini API keys here
        # Example format:
        # "AIzaSyBH8mRSlNVKQoRi5uCrEJikTJlqhRhPA-g",
        # "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z",
        # ... add more keys

        # PLACEHOLDER - Replace with real keys
        os.getenv("GEMINI_KEY_1", ""),
        os.getenv("GEMINI_KEY_2", ""),
        os.getenv("GEMINI_KEY_3", ""),
        os.getenv("GEMINI_KEY_4", ""),
        os.getenv("GEMINI_KEY_5", ""),
    ]

    # Remove empty strings from keys list
    GEMINI_KEYS = [key for key in GEMINI_KEYS if key.strip()]

    # ============================================================================
    # FALLBACK PROVIDERS
    # ============================================================================

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY")

    # ============================================================================
    # OPTIMIZER SETTINGS
    # ============================================================================

    # Batch processing (reduce to avoid quota spam)
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "3"))
    MAX_CONCURRENT_BATCHES: int = int(os.getenv("MAX_CONCURRENT_BATCHES", "2"))

    # Coverage target
    TARGET_COVERAGE: float = float(os.getenv("TARGET_COVERAGE", "85.0"))

    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAYS: List[int] = [90, 120, 180]  # Progressive delays

    # ============================================================================
    # MONITORING & ALERTING
    # ============================================================================

    ALERT_WEBHOOK_URL: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    QUOTA_DASHBOARD_URL: Optional[str] = os.getenv("QUOTA_DASHBOARD_URL")

    # Quota thresholds for alerts
    QUOTA_ALERT_THRESHOLD: float = 0.8  # Alert when 80% of quota used
    KEY_ERROR_THRESHOLD: int = 5  # Alert after 5 consecutive errors

    # ============================================================================
    # VALIDATION
    # ============================================================================

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that the configuration is properly set up."""
        errors = []

        if len(cls.GEMINI_KEYS) < 3:
            errors.append(f"Only {len(cls.GEMINI_KEYS)} Gemini keys configured. Recommended: 5-100 keys.")

        if not cls.OPENAI_API_KEY and not cls.CLAUDE_API_KEY:
            errors.append("No fallback providers configured (OpenAI/Claude).")

        if cls.BATCH_SIZE > 5:
            errors.append(f"Batch size {cls.BATCH_SIZE} is too high. Recommended: 3-5.")

        if errors:
            print("⚠️ CONFIGURATION WARNINGS:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✅ Multi-key configuration validated successfully!")
        print(f"  🔑 {len(cls.GEMINI_KEYS)} Gemini keys configured")
        print(f"  📦 Batch size: {cls.BATCH_SIZE}")
        print(f"  🔄 Fallback providers: OpenAI={'✅' if cls.OPENAI_API_KEY else '❌'}, Claude={'✅' if cls.CLAUDE_API_KEY else '❌'}")

        return True

    @classmethod
    def get_quota_report_template(cls) -> dict:
        """Get template for quota monitoring reports."""
        return {
            "total_keys": len(cls.GEMINI_KEYS),
            "active_keys": len([k for k in cls.GEMINI_KEYS if k.strip()]),
            "batch_size": cls.BATCH_SIZE,
            "max_concurrent": cls.MAX_CONCURRENT_BATCHES,
            "fallback_available": {
                "openai": bool(cls.OPENAI_API_KEY),
                "claude": bool(cls.CLAUDE_API_KEY)
            },
            "alerting_enabled": bool(cls.ALERT_WEBHOOK_URL),
            "monitoring_enabled": bool(cls.QUOTA_DASHBOARD_URL)
        }


# Validate configuration on import
if __name__ != "__main__":
    MultiKeyConfig.validate_config()
