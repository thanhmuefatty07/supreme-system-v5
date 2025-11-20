#!/usr/bin/env python3
"""
Enterprise AI Coverage Optimizer Runner

🚀 ADVANCED ENTERPRISE FEATURES:
- Multi-API Key Round-Robin (5-100 keys) - No quota limits
- Auto-retry with exponential backoff (90-180s) for rate limits
- Batch/stream processing for high-throughput test generation
- Intelligent fallback to OpenAI/Claude when quota exhausted
- Comprehensive quota monitoring and alerting
- Optimized prompts for maximum efficiency

Usage:
    python scripts/enterprise_optimizer.py --gemini-keys "key1" "key2" "key3" --openai-key "sk-..." --claude-key "sk-ant-..."

Environment Variables:
    GEMINI_KEYS: Comma-separated list of Gemini API keys
    OPENAI_API_KEY: OpenAI API key for fallback
    CLAUDE_API_KEY: Claude API key for fallback
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.gemini_coverage_optimizer import GeminiCoverageOptimizer


def load_keys_from_env() -> tuple[List[str], Optional[str], Optional[str]]:
    """Load API keys from environment variables."""
    gemini_keys = []
    openai_key = None
    claude_key = None

    # Load Gemini keys
    gemini_keys_str = os.getenv("GEMINI_KEYS", "")
    if gemini_keys_str:
        gemini_keys = [key.strip() for key in gemini_keys_str.split(",") if key.strip()]

    # Load fallback keys
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")

    return gemini_keys, openai_key, claude_key


def validate_keys(gemini_keys: List[str], openai_key: Optional[str], claude_key: Optional[str]) -> bool:
    """Validate that we have sufficient keys for enterprise operation."""
    if len(gemini_keys) < 3:
        print("⚠️ WARNING: Less than 3 Gemini keys provided. Recommended: 5-100 keys for enterprise operation.")
        print("   This may result in quota limits. Consider adding more keys.")

    if not any([openai_key, claude_key]):
        print("⚠️ WARNING: No fallback providers configured (OpenAI/Claude).")
        print("   If all Gemini keys hit quota limits, optimization may fail.")

    if not gemini_keys:
        print("❌ ERROR: No Gemini API keys provided!")
        return False

    return True


async def run_enterprise_optimization(
    gemini_keys: List[str],
    openai_key: Optional[str] = None,
    claude_key: Optional[str] = None,
    source_dir: str = "src",
    target_coverage: float = 85.0,
    max_iterations: int = 10,
    batch_size: int = 3,
    max_concurrent_batches: int = 2,
    verbose: bool = True
) -> dict:
    """
    Run enterprise-grade AI coverage optimization.

    Args:
        gemini_keys: List of Gemini API keys for round-robin
        openai_key: OpenAI API key for fallback
        claude_key: Claude API key for fallback
        source_dir: Source code directory to analyze
        target_coverage: Target coverage percentage (0-100)
        max_iterations: Maximum optimization iterations
        batch_size: Requests per batch (3-5 recommended)
        max_concurrent_batches: Max concurrent batch processing
        verbose: Enable verbose logging

    Returns:
        Comprehensive optimization results
    """

    print("🚀 ENTERPRISE AI COVERAGE OPTIMIZER")
    print("=" * 60)
    print(f"🎯 Target Coverage: {target_coverage}%")
    print(f"🔑 Gemini Keys: {len(gemini_keys)}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"⚡ Max Concurrent: {max_concurrent_batches}")
    print(f"🔄 Fallback OpenAI: {'✅' if openai_key else '❌'}")
    print(f"🔄 Fallback Claude: {'✅' if claude_key else '❌'}")
    print()

    # Initialize enterprise optimizer
    optimizer = GeminiCoverageOptimizer(
        gemini_keys=gemini_keys,
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        batch_size=batch_size,
        max_concurrent_batches=max_concurrent_batches
    )

    # Run optimization
    result = await optimizer.optimize_coverage(
        source_dir=source_dir,
        target_coverage=target_coverage / 100.0,
        max_iterations=max_iterations
    )

    return result


def print_enterprise_report(result: dict):
    """Print comprehensive enterprise optimization report."""
    print("\n🏆 ENTERPRISE OPTIMIZATION COMPLETE")
    print("=" * 80)

    # Coverage results
    print("📊 COVERAGE RESULTS:")
    print(f"  🎯 Target: {85.0}%")  # Default target
    print(f"  📈 Final: {result['final_coverage']*100:.1f}%")
    print(f"  ✅ Achieved: {'YES' if result['target_achieved'] else 'NO'}")
    print(f"  📏 Improvement: {(result['final_coverage']-result['initial_coverage'])*100:.1f}%")
    print()

    # Performance metrics
    print("⚡ PERFORMANCE METRICS:")
    print(f"  🧪 Tests Generated: {result['tests_generated']}")
    print(f"  🔄 Total Requests: {result['total_requests']}")
    print(f"  ⏱️ Total Time: {result['elapsed_seconds']:.1f}s")
    print(f"  📈 Requests/sec: {result['requests_per_second']:.2f}")
    print()

    # Quota monitoring
    print("🔑 QUOTA MONITORING:")
    print(f"  🔄 Multi-Key Rotation: {'YES' if result['multi_key_rotation'] else 'NO'}")
    print(f"  🔄 Fallback Used: {'YES' if result['fallback_used'] else 'NO'}")
    print(f"  ⚠️ Total Errors: {result['total_errors']}")
    print(f"  📊 Error Rate: {result['error_rate']*100:.1f}%")
    print()

    # Enterprise features
    print("🏗️ ENTERPRISE FEATURES:")
    print(f"  📦 Batch Processing: {'YES' if result['batch_processing_enabled'] else 'NO'}")
    print(f"  🔄 Auto Retry: {'YES' if result['auto_retry_enabled'] else 'NO'}")
    print(f"  🛡️ Quota-Free: {'YES' if result['quota_free_operation'] else 'NO'}")
    print()

    # Quota details
    if result['quota_reports']:
        final_quota = result['quota_reports'][-1]
        print("📈 KEY USAGE DETAILS:")
        for key_name, stats in final_quota.get('gemini_keys', {}).items():
            status = "🔴" if stats['errors'] > 0 else "🟢"
            print(f"  {status} {key_name}: {stats['requests']} req, {stats['errors']} err ({stats['error_rate']*100:.1f}%)")
        print()

    # Recommendations
    if not result['target_achieved']:
        print("💡 RECOMMENDATIONS:")
        if result['error_rate'] > 0.1:
            print("  - Add more Gemini API keys to reduce quota pressure")
        if not result['fallback_used'] and result['error_rate'] > 0.05:
            print("  - Configure OpenAI/Claude fallback keys")
        if result['requests_per_second'] < 1.0:
            print("  - Increase batch_size or max_concurrent_batches")
        print()

    print("✅ Enterprise optimization complete!")


async def main():
    """Main enterprise optimizer entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise AI Coverage Optimizer - Quota-Free Operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with multiple keys
  python scripts/enterprise_optimizer.py --gemini-keys key1 key2 key3

  # Full enterprise setup
  python scripts/enterprise_optimizer.py \\
    --gemini-keys key1 key2 key3 key4 key5 \\
    --openai-key sk-your-openai-key \\
    --claude-key sk-ant-your-claude-key \\
    --target-coverage 85 \\
    --batch-size 3 \\
    --max-concurrent 2

  # Using environment variables
  export GEMINI_KEYS="key1,key2,key3"
  export OPENAI_API_KEY="sk-..."
  python scripts/enterprise_optimizer.py

Enterprise Features:
  🚀 Multi-key round-robin (5-100 keys recommended)
  🔄 Auto-retry with 90-180s exponential backoff
  📦 Intelligent batch processing (3-5 req/batch)
  🔄 OpenAI/Claude fallback when quota exhausted
  📊 Comprehensive quota monitoring & alerting
  🎯 Optimized prompts for maximum efficiency
        """
    )

    parser.add_argument(
        "--gemini-keys", nargs="+",
        help="Gemini API keys (5-100 recommended for enterprise)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key for fallback when Gemini quota exhausted"
    )
    parser.add_argument(
        "--claude-key",
        help="Claude API key for fallback when Gemini quota exhausted"
    )
    parser.add_argument(
        "--source-dir", default="src",
        help="Source code directory to analyze (default: src)"
    )
    parser.add_argument(
        "--target-coverage", type=float, default=85.0,
        help="Target coverage percentage (default: 85.0)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum optimization iterations (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Requests per batch, 3-5 recommended (default: 3)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max concurrent batches (default: 2)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Enable verbose logging (default: True)"
    )

    args = parser.parse_args()

    # Load keys from environment if not provided
    gemini_keys = args.gemini_keys
    openai_key = args.openai_key
    claude_key = args.claude_key

    if not gemini_keys or not openai_key or not claude_key:
        env_gemini, env_openai, env_claude = load_keys_from_env()
        gemini_keys = gemini_keys or env_gemini
        openai_key = openai_key or env_openai
        claude_key = claude_key or env_claude

    # Validate configuration
    if not validate_keys(gemini_keys, openai_key, claude_key):
        return 1

    # Run enterprise optimization
    try:
        result = await run_enterprise_optimization(
            gemini_keys=gemini_keys,
            openai_key=openai_key,
            claude_key=claude_key,
            source_dir=args.source_dir,
            target_coverage=args.target_coverage,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            max_concurrent_batches=args.max_concurrent,
            verbose=args.verbose
        )

        # Print comprehensive report
        print_enterprise_report(result)

        # Return success/failure code
        return 0 if result['target_achieved'] else 1

    except Exception as e:
        print(f"❌ Enterprise optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
