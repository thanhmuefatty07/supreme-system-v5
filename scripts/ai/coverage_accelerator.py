#!/usr/bin/env python3
"""
AI-POWERED COVERAGE ACCELERATOR
Multi-key Gemini parallel processing for 80% coverage in 3 days
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CoverageTarget:
    """Coverage improvement target"""
    file_path: str
    current_coverage: float
    target_coverage: float
    priority: int  # 1-10, higher = more important

@dataclass
class TestGenerationTask:
    """AI test generation task"""
    file_path: str
    code_content: str
    existing_tests: str
    coverage_gaps: List[str]
    priority: int

class MultiKeyGeminiManager:
    """Manages multiple Gemini API keys for maximum throughput"""

    def __init__(self):
        self.keys = self._load_api_keys()
        self.key_usage = {i: 0 for i in range(len(self.keys))}
        self.rate_limits = {i: 14 for i in range(len(self.keys))}  # 14 RPM per key
        self.last_used = {i: 0 for i in range(len(self.keys))}

    def _load_api_keys(self) -> List[str]:
        """Load all available Gemini API keys"""
        keys = []
        for i in range(1, 7):  # 6 keys
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                keys.append(key)
        return keys

    def get_available_key(self) -> tuple[int, str]:
        """Get the best available key based on rate limits"""
        if not self.keys:
            raise ValueError("No Gemini API keys found in environment")

        current_time = time.time()
        min_wait = float('inf')
        best_key_idx = 0

        for idx, last_used in self.last_used.items():
            time_since_last = current_time - last_used
            rpm_limit = self.rate_limits[idx]

            # Calculate wait time for next request (60 seconds / RPM)
            wait_time = max(0, (60 / rpm_limit) - time_since_last)

            if wait_time < min_wait:
                min_wait = wait_time
                best_key_idx = idx

        if min_wait > 0:
            time.sleep(min_wait)

        self.last_used[best_key_idx] = time.time()
        self.key_usage[best_key_idx] += 1

        return best_key_idx, self.keys[best_key_idx]

    async def generate_with_key(self, prompt: str, key_idx: int) -> str:
        """Generate content using specific key"""
        api_key = self.keys[key_idx]

        # Configure Gemini with specific key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error with key {key_idx}: {e}")
            return ""

class AICoverageAccelerator:
    """Main AI-powered coverage acceleration engine"""

    def __init__(self):
        self.key_manager = MultiKeyGeminiManager()
        self.executor = ThreadPoolExecutor(max_workers=6)  # Match number of keys
        self.coverage_targets: List[CoverageTarget] = []

    def analyze_current_coverage(self) -> List[CoverageTarget]:
        """Analyze current coverage and identify improvement targets"""
        import glob

        # Find coverage.json file
        coverage_files = glob.glob('**/coverage.json', recursive=True)
        if not coverage_files:
            logger.error("coverage.json not found. Run pytest with coverage first.")
            return []

        coverage_file = coverage_files[0]
        logger.info(f"Found coverage file: {coverage_file}")

        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading coverage file: {e}")
            return []

        targets = []
        files = coverage_data.get('files', {})

        for file_path, file_data in files.items():
            summary = file_data.get('summary', {})
            current_pct = summary.get('percent_covered', 0)

            # Skip files already at high coverage
            if current_pct >= 80:
                continue

            # Calculate priority based on file importance and coverage gap
            priority = self._calculate_file_priority(file_path, current_pct)

            targets.append(CoverageTarget(
                file_path=file_path,
                current_coverage=current_pct,
                target_coverage=min(90, current_pct + 20),  # Aim for +20% improvement
                priority=priority
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda x: x.priority, reverse=True)
        return targets

    def _calculate_file_priority(self, file_path: str, current_pct: float) -> int:
        """Calculate priority score for a file"""
        priority = 5  # Base priority

        # Core modules get higher priority
        if 'core' in file_path or 'main' in file_path:
            priority += 3

        # API modules get high priority
        if 'client' in file_path or 'api' in file_path:
            priority += 2

        # Utility modules get medium priority
        if 'utils' in file_path or 'helpers' in file_path:
            priority += 1

        # Lower coverage gets higher priority
        if current_pct < 50:
            priority += 2
        elif current_pct < 30:
            priority += 3

        return min(priority, 10)  # Cap at 10

    def read_file_content(self, file_path: str) -> str:
        """Read file content safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

    def find_existing_tests(self, file_path: str) -> str:
        """Find existing tests for the target file"""
        # Convert src path to test path
        test_path = file_path.replace('src/', 'tests/')
        test_path = test_path.replace('.py', '_test.py')

        # Try unit test path first
        unit_test_path = test_path.replace('tests/', 'tests/unit/')

        for test_file in [unit_test_path, test_path]:
            if os.path.exists(test_file):
                return self.read_file_content(test_file)

        return "# No existing tests found"

    def create_test_generation_prompt(self, task: TestGenerationTask) -> str:
        """Create comprehensive AI prompt for test generation"""
        return f"""
You are an expert Python testing engineer specializing in high-coverage test suites.

TARGET FILE: {task.file_path}
CURRENT COVERAGE: ~{task.priority * 10}% (estimated)
TARGET COVERAGE: 80%+

EXISTING CODE:
```python
{task.code_content}
```

EXISTING TESTS (if any):
```python
{task.existing_tests}
```

REQUIREMENTS:
1. Generate comprehensive pytest tests to achieve 80%+ coverage
2. Cover all functions, classes, and edge cases
3. Use proper mocking for external dependencies
4. Include error handling and boundary tests
5. Follow pytest best practices
6. Add docstrings and comments

OUTPUT FORMAT:
```python
import pytest
from unittest.mock import MagicMock
from {task.file_path.replace('src/', '').replace('.py', '').replace('/', '.')} import *

# Test class/functions here
```

Focus on the most impactful tests that will give the biggest coverage increase.
Generate only the test code, no explanations.
"""

    async def generate_tests_for_file(self, task: TestGenerationTask) -> str:
        """Generate tests for a single file using AI"""
        prompt = self.create_test_generation_prompt(task)

        try:
            key_idx, api_key = self.key_manager.get_available_key()
            logger.info(f"Using Gemini key {key_idx} for {task.file_path}")

            # Generate with timeout
            test_code = await asyncio.wait_for(
                self.key_manager.generate_with_key(prompt, key_idx),
                timeout=30
            )

            # Clean up the response
            if test_code.startswith('```python'):
                test_code = test_code.split('```python')[1]
            if test_code.endswith('```'):
                test_code = test_code.rsplit('```', 1)[0]

            return test_code.strip()

        except Exception as e:
            logger.error(f"Failed to generate tests for {task.file_path}: {e}")
            return ""

    def save_generated_test(self, file_path: str, test_code: str):
        """Save generated test to appropriate location"""
        # Determine test file path
        test_file_path = file_path.replace('src/', 'tests/unit/')
        test_file_path = test_file_path.replace('.py', '_test.py')

        # Ensure directory exists
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

        # Check if test file exists
        if os.path.exists(test_file_path):
            # Append to existing file
            with open(test_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n# AI-generated tests for additional coverage\n{test_code}\n")
        else:
            # Create new test file
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(f'"""AI-generated tests for {file_path}"""\n\n{test_code}\n')

        logger.info(f"Saved AI-generated tests to {test_file_path}")

    async def accelerate_coverage(self, target_files: Optional[List[str]] = None):
        """Main coverage acceleration workflow"""
        logger.info("🚀 Starting AI-powered coverage acceleration...")

        # Analyze current coverage
        if not target_files:
            targets = self.analyze_current_coverage()
        else:
            targets = [CoverageTarget(f, 0, 80, 5) for f in target_files]

        if not targets:
            logger.warning("No coverage targets found")
            return

        logger.info(f"🎯 Found {len(targets)} coverage improvement targets")

        # Create test generation tasks
        tasks = []
        for target in targets[:10]:  # Process top 10 files first
            code_content = self.read_file_content(target.file_path)
            existing_tests = self.find_existing_tests(target.file_path)

            if not code_content:
                continue

            task = TestGenerationTask(
                file_path=target.file_path,
                code_content=code_content,
                existing_tests=existing_tests,
                coverage_gaps=[],  # Could be enhanced to analyze specific gaps
                priority=target.priority
            )
            tasks.append(task)

        logger.info(f"🤖 Generating tests for {len(tasks)} files using {len(self.key_manager.keys)} API keys...")

        # Generate tests in parallel
        generated_tests = {}
        semaphore = asyncio.Semaphore(6)  # Limit concurrent requests

        async def generate_with_semaphore(task):
            async with semaphore:
                test_code = await self.generate_tests_for_file(task)
                return task.file_path, test_code

        # Run parallel generation
        generation_tasks = [generate_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Generation task failed: {result}")
                continue

            file_path, test_code = result
            if test_code:
                generated_tests[file_path] = test_code
                self.save_generated_test(file_path, test_code)

        logger.info(f"✅ Generated tests for {len(generated_tests)} files")

        # Run coverage again to measure improvement
        logger.info("📊 Measuring coverage improvement...")
        self.run_coverage_measurement()

    def run_coverage_measurement(self):
        """Run pytest coverage measurement"""
        import subprocess

        cmd = [
            "python", "-m", "pytest",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=json",
            "-x", "--tb=no", "-q"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            logger.info("Coverage measurement completed")

            # Parse new coverage
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    data = json.load(f)
                    totals = data.get('totals', {})
                    new_coverage = totals.get('percent_covered', 0)
                    logger.info(f"🎯 New coverage: {new_coverage}%")

        except subprocess.TimeoutExpired:
            logger.error("Coverage measurement timed out")
        except Exception as e:
            logger.error(f"Coverage measurement failed: {e}")

async def main():
    """Main execution function"""
    import sys

    accelerator = AICoverageAccelerator()

    # Check if API keys are available
    if not accelerator.key_manager.keys:
        logger.error("❌ No Gemini API keys found. Set GEMINI_API_KEY_1 through GEMINI_API_KEY_6")
        return

    logger.info(f"🔑 Found {len(accelerator.key_manager.keys)} Gemini API keys")

    # Get target files from command line arguments
    target_files = sys.argv[1:] if len(sys.argv) > 1 else None

    # Run coverage acceleration
    await accelerator.accelerate_coverage(target_files)

if __name__ == "__main__":
    asyncio.run(main())
