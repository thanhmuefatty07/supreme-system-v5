#!/usr/bin/env python3
"""
Entry point for running coverage optimizer as a module.

Usage:
    python -m src.ai.coverage_optimizer --source-dir src --target-coverage 80
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ai.coverage_optimizer import AICoverageOptimizer


def main():
    """Main entry point for coverage optimizer."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Coverage Optimizer for Supreme System V5'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='src',
        help='Source directory to analyze (default: src)'
    )
    parser.add_argument(
        '--target-coverage',
        type=float,
        default=80.0,
        help='Target coverage percentage (default: 80.0)'
    )
    parser.add_argument(
        '--ai-provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic', 'google'],
        help='AI provider to use (default: openai)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/unit',
        help='Output directory for generated tests (default: tests/unit)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum optimization iterations (default: 5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--openai-api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key and args.ai_provider == 'openai':
        print("⚠️ Warning: No OpenAI API key provided. Some features may not work.")
        print("Set OPENAI_API_KEY environment variable or use --openai-api-key")

    # Create optimizer instance
    optimizer = AICoverageOptimizer(openai_api_key=api_key)

    # Run optimization
    try:
        result = asyncio.run(
            optimizer.achieve_80_percent_coverage(args.source_dir)
        )
        
        print("\n" + "="*60)
        print("COVERAGE OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Initial Coverage: {result['initial_coverage']:.2f}%")
        print(f"Final Coverage: {result['final_coverage']:.2f}%")
        print(f"Improvement: +{result['coverage_improvement']:.2f}%")
        print(f"Target Achieved: {'✅ YES' if result['target_achieved'] else '❌ NO'}")
        print("="*60 + "\n")

        if result['target_achieved']:
            sys.exit(0)
        else:
            print(f"⚠️ Target coverage ({args.target_coverage}%) not achieved")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
