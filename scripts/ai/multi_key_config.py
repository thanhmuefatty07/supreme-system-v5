#!/usr/bin/env python3
"""
Multi-Key Configuration for Supreme System V5
Enterprise-Grade API Key Management

🔑 MULTI-KEY ROTATION:
- 6 Gemini API keys configured
- Round-robin rotation to avoid quota limits
- 90 requests/minute total throughput (6 × 15 RPM)
- Zero cost with FREE tier

🔒 SECURITY:
- Keys configured directly for immediate use
- Automatic validation on load
- Quota monitoring and alerts

🎯 PERFORMANCE:
- Total capacity: 90 RPM across all keys
- 6,000,000 tokens/minute combined
- Zero quota errors with proper rotation
"""

import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiKeyConfig:
    """
    Enterprise Multi-Key Configuration
    
    Manages 6 Gemini API keys for high-throughput, quota-free operation.
    Supports scaling to 100+ keys for maximum scalability.
    """
    
    # ===================================================================
    # 🔑 GEMINI API KEYS - Loaded from Environment Variables
    # ===================================================================
    # SECURITY: Never hardcode API keys in source code!
    # Keys are loaded from environment variables for security.
    
    GEMINI_KEYS: List[str] = [
        # Load from environment variables (filter out empty values)
        key for key in [
            os.getenv("GEMINI_KEY_1", ""),
            os.getenv("GEMINI_KEY_2", ""),
            os.getenv("GEMINI_KEY_3", ""),
            os.getenv("GEMINI_KEY_4", ""),
            os.getenv("GEMINI_KEY_5", ""),
            os.getenv("GEMINI_KEY_6", ""),
        ] if key and key.startswith("AIzaSy") and len(key) > 30
    ]
    
    # ===================================================================
    # 📊 QUOTA CAPACITY CALCULATION
    # ===================================================================
    
    # Per-key limits (Gemini FREE tier)
    RPM_PER_KEY: int = 15  # Requests per minute per key
    TPM_PER_KEY: int = 1_000_000  # Tokens per minute per key
    
    # Total capacity with 6 keys
    TOTAL_RPM: int = 90  # 6 keys × 15 RPM = 90 requests/minute
    TOTAL_TPM: int = 6_000_000  # 6 keys × 1M TPM = 6M tokens/minute
    
    # Daily capacity
    DAILY_REQUESTS: int = 90 * 60 * 24  # 129,600 requests/day
    DAILY_TOKENS: int = 6_000_000 * 60 * 24  # 8.64 billion tokens/day
    
    # ===================================================================
    # 🔄 FALLBACK PROVIDER KEYS (OPTIONAL)
    # ===================================================================
    
    # OpenAI API key for fallback (when all Gemini keys exhausted)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Claude API key for secondary fallback
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY")
    
    # ===================================================================
    # ⚙️ BATCH PROCESSING CONFIGURATION
    # ===================================================================
    
    # Batch size: Number of gaps to process per batch
    # With 6 keys, we can safely use 3-5 per batch
    BATCH_SIZE: int = 3
    
    # Max concurrent batches: How many batches to process in parallel
    # With 90 RPM, we can run 2-3 concurrent batches safely
    MAX_CONCURRENT_BATCHES: int = 2
    
    # Delay between requests (seconds)
    # With round-robin, we can be more aggressive: 4s delay = 15 req/min per key
    REQUEST_DELAY: float = 4.0
    
    # Retry delays for 429 errors (seconds)
    # If we still hit 429 with 6 keys, wait longer
    RETRY_DELAYS: List[int] = [90, 120, 180]
    
    # ===================================================================
    # 🎯 COVERAGE TARGETS
    # ===================================================================
    
    # Target coverage percentage
    TARGET_COVERAGE: float = 85.0
    
    # Maximum iterations for optimization
    MAX_ITERATIONS: int = 10
    
    # ===================================================================
    # 📊 MONITORING & ALERTS
    # ===================================================================
    
    # Webhook for quota alerts
    ALERT_WEBHOOK_URL: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    
    # Quota dashboard URL (Grafana/custom)
    QUOTA_DASHBOARD_URL: Optional[str] = os.getenv("QUOTA_DASHBOARD_URL")
    
    # Alert thresholds
    QUOTA_ALERT_THRESHOLD: float = 0.8  # Alert when 80% of quota used
    KEY_ERROR_THRESHOLD: int = 5  # Alert after 5 consecutive errors per key
    
    # ===================================================================
    # ✅ VALIDATION & UTILITIES
    # ===================================================================
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration and log status.
        
        Returns:
            bool: True if configuration is valid
        """
        
        # Check Gemini keys
        valid_keys = [k for k in cls.GEMINI_KEYS if k and k.startswith("AIzaSy") and len(k) > 30]
        invalid_keys = len(cls.GEMINI_KEYS) - len(valid_keys)
        
        logger.info("🔑 Multi-Key Configuration Validation:")
        logger.info("="*60)
        logger.info(f"✅ Valid Gemini keys: {len(valid_keys)}")
        
        if invalid_keys > 0:
            logger.warning(f"⚠️ Invalid keys found: {invalid_keys}")
        
        if len(valid_keys) == 0:
            logger.error("❌ NO VALID GEMINI KEYS! Optimizer cannot run.")
            logger.error("Configure keys in config/multi_key_config.py or set environment variables.")
            return False
        
        # Calculate total capacity
        total_rpm = len(valid_keys) * cls.RPM_PER_KEY
        total_tpm = len(valid_keys) * cls.TPM_PER_KEY
        
        logger.info(f"📊 Total capacity: {total_rpm} RPM, {total_tpm:,} TPM")
        logger.info(f"📊 Daily capacity: {total_rpm * 60 * 24:,} requests, {total_tpm * 60 * 24:,} tokens")
        logger.info(f"💼 Batch config: {cls.BATCH_SIZE} per batch, {cls.MAX_CONCURRENT_BATCHES} concurrent")
        
        # Check fallback providers
        if cls.OPENAI_API_KEY:
            logger.info("✅ OpenAI fallback: AVAILABLE")
        else:
            logger.info("ℹ️ OpenAI fallback: NOT CONFIGURED (optional)")
        
        if cls.CLAUDE_API_KEY:
            logger.info("✅ Claude fallback: AVAILABLE")
        else:
            logger.info("ℹ️ Claude fallback: NOT CONFIGURED (optional)")
        
        logger.info("="*60)
        
        return len(valid_keys) > 0
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get configuration summary for reporting.
        
        Returns:
            dict: Configuration summary
        """
        valid_keys = [k for k in cls.GEMINI_KEYS if k and k.startswith("AIzaSy") and len(k) > 30]
        
        return {
            "gemini_keys_count": len(valid_keys),
            "total_rpm": len(valid_keys) * cls.RPM_PER_KEY,
            "total_tpm": len(valid_keys) * cls.TPM_PER_KEY,
            "daily_capacity_requests": len(valid_keys) * cls.RPM_PER_KEY * 60 * 24,
            "daily_capacity_tokens": len(valid_keys) * cls.TPM_PER_KEY * 60 * 24,
            "batch_size": cls.BATCH_SIZE,
            "max_concurrent_batches": cls.MAX_CONCURRENT_BATCHES,
            "request_delay": cls.REQUEST_DELAY,
            "target_coverage": cls.TARGET_COVERAGE,
            "max_iterations": cls.MAX_ITERATIONS,
            "openai_available": bool(cls.OPENAI_API_KEY),
            "claude_available": bool(cls.CLAUDE_API_KEY),
            "monitoring_enabled": bool(cls.ALERT_WEBHOOK_URL),
        }


# ===================================================================
# 🚀 AUTO-VALIDATION ON IMPORT
# ===================================================================

if __name__ != "__main__":
    # Validate configuration when module is imported
    if not MultiKeyConfig.validate_config():
        logger.warning("⚠️ Multi-key configuration has warnings. Review config/multi_key_config.py")


# ===================================================================
# 🧪 TESTING & VALIDATION CLI
# ===================================================================

if __name__ == "__main__":
    print("🔑 Multi-Key Configuration Test")
    print("="*70)
    
    # Validate configuration
    is_valid = MultiKeyConfig.validate_config()
    
    # Print detailed summary
    print("\n📊 Configuration Summary:")
    print("="*70)
    summary = MultiKeyConfig.get_config_summary()
    
    print(f"\n🔑 API Keys:")
    print(f"  Gemini keys: {summary['gemini_keys_count']}")
    print(f"  OpenAI available: {'✅' if summary['openai_available'] else '❌'}")
    print(f"  Claude available: {'✅' if summary['claude_available'] else '❌'}")
    
    print(f"\n📊 Throughput Capacity:")
    print(f"  Requests/minute: {summary['total_rpm']}")
    print(f"  Tokens/minute: {summary['total_tpm']:,}")
    print(f"  Daily requests: {summary['daily_capacity_requests']:,}")
    print(f"  Daily tokens: {summary['daily_capacity_tokens']:,}")
    
    print(f"\n💼 Processing Configuration:")
    print(f"  Batch size: {summary['batch_size']}")
    print(f"  Concurrent batches: {summary['max_concurrent_batches']}")
    print(f"  Request delay: {summary['request_delay']}s")
    
    print(f"\n🎯 Optimization Targets:")
    print(f"  Target coverage: {summary['target_coverage']}%")
    print(f"  Max iterations: {summary['max_iterations']}")
    
    print(f"\n🚨 Monitoring:")
    print(f"  Alerting: {'✅' if summary['monitoring_enabled'] else '❌'}")
    
    print("\n" + "="*70)
    if is_valid:
        print("✅ CONFIGURATION VALID - READY FOR QUOTA-FREE OPERATION!")
        print("🚀 Run: bash RUN_OPTIMIZER.sh")
    else:
        print("❌ CONFIGURATION INVALID - REVIEW ERRORS ABOVE")
    print("="*70)
