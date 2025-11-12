#!/usr/bin/env python3
"""
Gemini-Powered Coverage Optimizer for Supreme System V5

Uses Google Gemini 2.5 Flash API (FREE tier) to achieve 80%+ coverage
without any cost. Generates intelligent, comprehensive tests.
"""

import asyncio
import os
import logging
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import re
from datetime import datetime

# Gemini API
import google.generativeai as genai

# OpenAI API (optional backup)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoverageGap:
    """Represents a coverage gap to fill."""
    file_path: str
    line_start: int
    line_end: int
    code_context: str
    function_name: str
    complexity: float
    priority: float


@dataclass
class GeneratedTest:
    """AI-generated test case."""
    test_code: str
    target_file: str
    target_lines: List[int]
    confidence: float
    provider: str
    cost_usd: float = 0.0


class GeminiCoverageOptimizer:
    """
    AI-powered coverage optimizer using Google Gemini (FREE).
    
    Features:
    - Uses Gemini 2.5 Flash (free tier, no cost)
    - Generates comprehensive test suites
    - Targets specific coverage gaps
    - Validates generated tests
    - Fallback to OpenAI if needed
    """

    def __init__(
        self,
        gemini_api_key: str,
        openai_api_key: Optional[str] = None,
        provider: str = "gemini",
        max_cost_usd: float = 1.0
    ):
        """Initialize optimizer with API keys."""
        
        # Gemini setup (FREE)
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # OpenAI setup (optional backup)
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        self.provider = provider
        self.max_cost_usd = max_cost_usd
        self.total_cost = 0.0
        self.tests_generated = 0
        
        logger.info(f"🤖 Gemini Coverage Optimizer initialized")
        logger.info(f"📊 Provider: {provider.upper()}")
        logger.info(f"💰 Max cost: ${max_cost_usd} (Gemini is FREE!)")

    async def optimize_coverage(
        self,
        source_dir: str = "src",
        target_coverage: float = 0.80,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Main optimization loop.
        
        Args:
            source_dir: Source code directory
            target_coverage: Target coverage percentage (0.0-1.0)
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results with coverage metrics
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING AI COVERAGE OPTIMIZATION")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Phase 1: Analyze current coverage
        logger.info("\n📊 Phase 1: Analyzing current coverage...")
        current_coverage = await self._analyze_coverage(source_dir)
        logger.info(f"Current coverage: {current_coverage*100:.1f}%")
        logger.info(f"Target coverage: {target_coverage*100:.1f}%")
        logger.info(f"Gap to fill: {(target_coverage-current_coverage)*100:.1f}%")
        
        if current_coverage >= target_coverage:
            logger.info("✅ Already at target coverage!")
            return self._build_result(current_coverage, current_coverage, 0)
        
        # Phase 2: Iterative optimization
        iteration = 0
        all_generated_tests = []
        
        while iteration < max_iterations and current_coverage < target_coverage:
            iteration += 1
            logger.info(f"\n🔄 Iteration {iteration}/{max_iterations}")
            logger.info("-" * 60)
            
            # Identify gaps
            gaps = await self._identify_coverage_gaps(source_dir)
            logger.info(f"📍 Found {len(gaps)} coverage gaps")
            
            if not gaps:
                logger.info("✅ No more gaps to fill!")
                break
            
            # Prioritize gaps
            prioritized_gaps = self._prioritize_gaps(gaps)
            logger.info(f"🎯 Prioritized {len(prioritized_gaps)} high-impact targets")
            
            # Generate tests with AI
            batch_tests = await self._generate_tests_batch(
                prioritized_gaps[:50],  # Process top 50 per iteration
                iteration
            )
            logger.info(f"🤖 Generated {len(batch_tests)} tests")
            
            all_generated_tests.extend(batch_tests)
            
            # Validate tests
            valid_tests = self._validate_tests(batch_tests)
            logger.info(f"✅ Validated {len(valid_tests)}/{len(batch_tests)} tests")
            
            # Write test files
            await self._write_test_files(valid_tests, iteration)
            
            # Re-measure coverage
            new_coverage = await self._measure_coverage_after_tests()
            improvement = (new_coverage - current_coverage) * 100
            logger.info(f"📈 Coverage: {current_coverage*100:.1f}% → {new_coverage*100:.1f}% (+{improvement:.1f}%)")
            
            current_coverage = new_coverage
            
            # Cost tracking
            logger.info(f"💰 Cost so far: ${self.total_cost:.2f}")
            
            if self.total_cost >= self.max_cost_usd:
                logger.warning(f"⚠️ Cost limit reached: ${self.total_cost:.2f}")
                break
        
        # Final report
        elapsed = (datetime.now() - start_time).total_seconds()
        final_result = self._build_result(
            initial_coverage=await self._analyze_coverage(source_dir),
            final_coverage=current_coverage,
            tests_generated=len(all_generated_tests),
            elapsed_seconds=elapsed
        )
        
        self._print_final_report(final_result)
        return final_result

    async def _generate_tests_batch(
        self,
        gaps: List[CoverageGap],
        iteration: int
    ) -> List[GeneratedTest]:
        """
        Generate tests for a batch of coverage gaps.
        
        Args:
            gaps: List of coverage gaps to fill
            iteration: Current iteration number
            
        Returns:
            List of generated tests
        """
        generated_tests = []
        
        # Process in smaller batches to respect rate limits
        batch_size = 5
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i+batch_size]
            
            logger.info(f"  🤖 Generating tests for batch {i//batch_size + 1}...")
            
            # Generate with primary provider
            if self.provider == "gemini":
                batch_tests = await self._generate_with_gemini(batch)
            elif self.provider == "openai" and self.openai_client:
                batch_tests = await self._generate_with_openai(batch)
            else:
                logger.warning("No provider available!")
                batch_tests = []
            
            generated_tests.extend(batch_tests)
            
            # Rate limiting
            await asyncio.sleep(1)  # Respect rate limits
        
        return generated_tests

    async def _generate_with_gemini(
        self,
        gaps: List[CoverageGap]
    ) -> List[GeneratedTest]:
        """
        Generate tests using Gemini API (FREE).
        
        Args:
            gaps: Coverage gaps to generate tests for
            
        Returns:
            List of generated tests
        """
        generated_tests = []
        
        for gap in gaps:
            try:
                # Build prompt
                prompt = self._build_test_generation_prompt(gap)
                
                # Call Gemini API (FREE!)
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,  # Low for consistency
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048,
                    )
                )
                
                # Extract test code
                test_code = self._extract_test_code(response.text)
                
                if test_code:
                    generated_tests.append(GeneratedTest(
                        test_code=test_code,
                        target_file=gap.file_path,
                        target_lines=list(range(gap.line_start, gap.line_end + 1)),
                        confidence=0.85,  # Gemini quality
                        provider="gemini",
                        cost_usd=0.0  # FREE!
                    ))
                    
                    logger.debug(f"  ✅ Generated test for {gap.function_name}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to generate test for {gap.function_name}: {e}")
                continue
        
        logger.info(f"  💰 Gemini API cost: $0.00 (FREE tier!)")
        return generated_tests

    async def _generate_with_openai(
        self,
        gaps: List[CoverageGap]
    ) -> List[GeneratedTest]:
        """
        Generate tests using OpenAI API (backup, ~$0.50).
        
        Args:
            gaps: Coverage gaps to generate tests for
            
        Returns:
            List of generated tests
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available")
            return []
        
        generated_tests = []
        
        for gap in gaps:
            try:
                # Build prompt
                prompt = self._build_test_generation_prompt(gap)
                
                # Call OpenAI API
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Cheapest: $0.15/$0.60 per 1M tokens
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert test engineer. Generate comprehensive pytest tests."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                
                # Calculate cost
                usage = response.usage
                cost = (
                    usage.prompt_tokens / 1_000_000 * 0.15 +  # Input
                    usage.completion_tokens / 1_000_000 * 0.60  # Output
                )
                self.total_cost += cost
                
                # Extract test code
                test_code = self._extract_test_code(response.choices[0].message.content)
                
                if test_code:
                    generated_tests.append(GeneratedTest(
                        test_code=test_code,
                        target_file=gap.file_path,
                        target_lines=list(range(gap.line_start, gap.line_end + 1)),
                        confidence=0.90,  # GPT-4 quality
                        provider="openai",
                        cost_usd=cost
                    ))
                    
                    logger.debug(f"  ✅ Generated test for {gap.function_name} (${cost:.4f})")
                
                # Safety check
                if self.total_cost >= self.max_cost_usd:
                    logger.warning(f"⚠️ Cost limit reached: ${self.total_cost:.2f}")
                    break
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to generate test: {e}")
                continue
        
        logger.info(f"  💰 OpenAI API cost: ${self.total_cost:.2f}")
        return generated_tests

    def _build_test_generation_prompt(self, gap: CoverageGap) -> str:
        """
        Build prompt for AI test generation.
        
        Args:
            gap: Coverage gap information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Generate a comprehensive pytest test function for the following Python code.

**FILE**: {gap.file_path}
**FUNCTION**: {gap.function_name}
**LINES TO COVER**: {gap.line_start}-{gap.line_end}
**COMPLEXITY**: {gap.complexity}

**CODE CONTEXT**:
```python
{gap.code_context}
```

**REQUIREMENTS**:
1. Generate a pytest test function with proper naming (test_<function>_<scenario>)
2. Include comprehensive docstring explaining what is tested
3. Test BOTH happy path AND edge cases:
   - Normal valid inputs
   - Boundary values (zero, negative, maximum)
   - Invalid inputs (None, empty, wrong type)
   - Error conditions (exceptions)
4. Use hypothesis for property-based testing if applicable:
   ```python
   from hypothesis import given, strategies as st
   @given(value=st.floats(min_value=0, max_value=1000))
   ```
5. Include proper imports (MUST include 'strategies as st' for hypothesis)
6. Add comprehensive assertions:
   - Assert return types
   - Assert value ranges
   - Assert side effects
7. For async functions, use @pytest.mark.asyncio
8. Include error handling tests (pytest.raises)

**IMPORTANT - IMPORT REQUIREMENTS**:
- If using hypothesis: `from hypothesis import given, strategies as st`
- If using pytest: `import pytest`
- If testing async: `import pytest, asyncio`
- Import the actual function being tested

**CONTEXT**: This is a financial trading system where correctness is CRITICAL.
Money is at risk. Tests must be comprehensive and catch all edge cases.

**OUTPUT FORMAT**: Return ONLY the test function code, no explanations.
Start with imports, then the test function.
"""
        return prompt

    def _extract_test_code(self, ai_response: str) -> Optional[str]:
        """
        Extract test code from AI response.
        
        Args:
            ai_response: Raw AI response text
            
        Returns:
            Extracted Python test code or None
        """
        # Remove markdown code blocks
        code = ai_response.strip()
        
        # Extract from ```python blocks
        if "```python" in code:
            matches = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
            if matches:
                code = matches[0]
        elif "```" in code:
            matches = re.findall(r'```\n(.*?)```', code, re.DOTALL)
            if matches:
                code = matches[0]
        
        # Validate it's Python code
        try:
            ast.parse(code)
            return code.strip()
        except SyntaxError:
            logger.warning("Generated code has syntax errors")
            return None

    async def _analyze_coverage(self, source_dir: str) -> float:
        """
        Analyze current test coverage.
        
        Args:
            source_dir: Source directory to analyze
            
        Returns:
            Current coverage as decimal (0.0-1.0)
        """
        try:
            # Check if coverage.xml exists
            coverage_file = Path("coverage.xml")
            if not coverage_file.exists():
                logger.warning("coverage.xml not found, running tests...")
                import subprocess
                subprocess.run(
                    ["pytest", "--cov=src", "--cov-report=xml"],
                    capture_output=True,
                    timeout=300
                )
            
            # Parse coverage.xml
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            coverage = float(root.attrib['line-rate'])
            
            return coverage
            
        except Exception as e:
            logger.error(f"Failed to analyze coverage: {e}")
            return 0.0

    async def _identify_coverage_gaps(self, source_dir: str) -> List[CoverageGap]:
        """
        Identify specific coverage gaps from coverage report.
        
        Args:
            source_dir: Source directory
            
        Returns:
            List of coverage gaps
        """
        gaps = []
        
        try:
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            
            # Iterate through packages and classes
            for package in root.findall(".//package"):
                for cls in package.findall(".//class"):
                    filename = cls.attrib['filename']
                    filepath = Path(source_dir) / filename
                    
                    if not filepath.exists():
                        continue
                    
                    # Find uncovered lines
                    uncovered_lines = []
                    for line in cls.findall(".//line"):
                        if line.attrib.get('hits') == '0':
                            uncovered_lines.append(int(line.attrib['number']))
                    
                    if not uncovered_lines:
                        continue
                    
                    # Group consecutive lines into gaps
                    groups = self._group_consecutive_lines(uncovered_lines)
                    
                    # Create gap objects
                    for line_start, line_end in groups:
                        gap = await self._create_coverage_gap(
                            filepath, line_start, line_end
                        )
                        if gap:
                            gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to identify gaps: {e}")
            return []

    def _group_consecutive_lines(self, lines: List[int]) -> List[Tuple[int, int]]:
        """
        Group consecutive line numbers into ranges.
        
        Args:
            lines: List of line numbers
            
        Returns:
            List of (start, end) tuples
        """
        if not lines:
            return []
        
        lines = sorted(lines)
        groups = []
        start = lines[0]
        end = lines[0]
        
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                groups.append((start, end))
                start = line
                end = line
        
        groups.append((start, end))
        return groups

    async def _create_coverage_gap(
        self,
        filepath: Path,
        line_start: int,
        line_end: int
    ) -> Optional[CoverageGap]:
        """
        Create a CoverageGap object with code context.
        
        Args:
            filepath: Path to source file
            line_start: Start line number
            line_end: End line number
            
        Returns:
            CoverageGap object or None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract context (10 lines before/after)
            context_start = max(0, line_start - 10)
            context_end = min(len(lines), line_end + 10)
            context = ''.join(lines[context_start:context_end])
            
            # Find function name
            function_name = self._find_function_name(lines, line_start)
            
            # Calculate complexity
            complexity = self._calculate_complexity(context)
            
            # Calculate priority
            priority = self._calculate_priority(complexity, function_name, context)
            
            return CoverageGap(
                file_path=str(filepath),
                line_start=line_start,
                line_end=line_end,
                code_context=context,
                function_name=function_name,
                complexity=complexity,
                priority=priority
            )
            
        except Exception as e:
            logger.warning(f"Failed to create gap for {filepath}:{line_start}: {e}")
            return None

    def _find_function_name(self, lines: List[str], line_num: int) -> str:
        """
        Find the function name containing the line.
        
        Args:
            lines: Source file lines
            line_num: Target line number
            
        Returns:
            Function name or 'unknown'
        """
        # Search backwards for function definition
        for i in range(line_num - 1, -1, -1):
            if i >= len(lines):
                continue
            line = lines[i]
            match = re.match(r'\s*def\s+(\w+)\s*\(', line)
            if match:
                return match.group(1)
        
        return "unknown_function"

    def _calculate_complexity(self, code: str) -> float:
        """
        Calculate cyclomatic complexity.
        
        Args:
            code: Source code
            
        Returns:
            Complexity score
        """
        try:
            tree = ast.parse(code)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return float(complexity)
            
        except:
            return 1.0

    def _calculate_priority(self, complexity: float, function_name: str, code: str) -> float:
        """
        Calculate priority score for gap.
        
        Args:
            complexity: Code complexity
            function_name: Function name
            code: Source code
            
        Returns:
            Priority score (0.0-1.0)
        """
        priority = 0.5  # Base priority
        
        # Complexity bonus
        priority += min(complexity / 20, 0.3)
        
        # Critical function bonus
        critical_keywords = ['emergency', 'liquidation', 'risk', 'close', 'order', 'execute']
        if any(kw in function_name.lower() for kw in critical_keywords):
            priority += 0.2
        
        # Error handling bonus
        if 'try:' in code or 'except' in code or 'raise' in code:
            priority += 0.15
        
        return min(priority, 1.0)

    def _prioritize_gaps(self, gaps: List[CoverageGap]) -> List[CoverageGap]:
        """
        Sort gaps by priority.
        
        Args:
            gaps: List of coverage gaps
            
        Returns:
            Sorted list (highest priority first)
        """
        return sorted(gaps, key=lambda g: g.priority, reverse=True)

    def _validate_tests(self, tests: List[GeneratedTest]) -> List[GeneratedTest]:
        """
        Validate generated tests for quality.
        
        Args:
            tests: List of generated tests
            
        Returns:
            List of valid tests
        """
        valid_tests = []
        
        for test in tests:
            # Basic validation
            if not test.test_code:
                continue
            
            # Check for test function
            if 'def test_' not in test.test_code:
                logger.warning("Test doesn't contain test function")
                continue
            
            # Check for assertions
            if 'assert' not in test.test_code:
                logger.warning("Test doesn't contain assertions")
                continue
            
            # Syntax check
            try:
                ast.parse(test.test_code)
            except SyntaxError as e:
                logger.warning(f"Syntax error in generated test: {e}")
                continue
            
            valid_tests.append(test)
        
        return valid_tests

    async def _write_test_files(
        self,
        tests: List[GeneratedTest],
        iteration: int
    ) -> List[Path]:
        """
        Write generated tests to files.
        
        Args:
            tests: List of tests to write
            iteration: Current iteration number
            
        Returns:
            List of created file paths
        """
        test_dir = Path("tests/unit")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by target module
        tests_by_module = {}
        for test in tests:
            module = Path(test.target_file).stem
            if module not in tests_by_module:
                tests_by_module[module] = []
            tests_by_module[module].append(test)
        
        created_files = []
        
        for module, module_tests in tests_by_module.items():
            filename = f"test_{module}_ai_gen_iter{iteration}.py"
            filepath = test_dir / filename
            
            # Build file content
            content = f'''"""AI-generated tests for {module}

Generated by Gemini Coverage Optimizer
Iteration: {iteration}
Tests: {len(module_tests)}
Provider: {module_tests[0].provider if module_tests else 'unknown'}
Generated: {datetime.now().isoformat()}
"""

import pytest
import asyncio
from typing import Any, Dict, List, Optional
from hypothesis import given, strategies as st, assume
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


'''
            
            # Add all tests
            for test in module_tests:
                content += "\n\n" + test.test_code.strip()
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            created_files.append(filepath)
            logger.info(f"  📝 Created {filename} with {len(module_tests)} tests")
        
        return created_files

    async def _measure_coverage_after_tests(self) -> float:
        """
        Run tests and measure new coverage.
        
        Returns:
            New coverage as decimal (0.0-1.0)
        """
        try:
            import subprocess
            
            result = subprocess.run(
                ["pytest", "--cov=src", "--cov-report=xml", "-q"],
                capture_output=True,
                timeout=300
            )
            
            # Parse new coverage
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            coverage = float(root.attrib['line-rate'])
            
            return coverage
            
        except Exception as e:
            logger.error(f"Failed to measure coverage: {e}")
            return 0.0

    def _build_result(self, initial_coverage: float, final_coverage: float, 
                     tests_generated: int, elapsed_seconds: float = 0) -> Dict[str, Any]:
        """Build result dictionary."""
        return {
            'initial_coverage': initial_coverage,
            'final_coverage': final_coverage,
            'improvement': final_coverage - initial_coverage,
            'target_achieved': final_coverage >= 0.80,
            'tests_generated': tests_generated,
            'total_cost_usd': self.total_cost,
            'elapsed_seconds': elapsed_seconds,
            'provider': self.provider
        }

    def _print_final_report(self, result: Dict[str, Any]):
        """Print final optimization report."""
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL COVERAGE OPTIMIZATION REPORT")
        logger.info("="*60)
        logger.info(f"\n📊 Coverage:")
        logger.info(f"  Initial:     {result['initial_coverage']*100:.1f}%")
        logger.info(f"  Final:       {result['final_coverage']*100:.1f}%")
        logger.info(f"  Improvement: +{result['improvement']*100:.1f}%")
        logger.info(f"  Target:      80.0%")
        logger.info(f"  Status:      {'✅ ACHIEVED' if result['target_achieved'] else '❌ NOT ACHIEVED'}")
        logger.info(f"\n🤖 Generation:")
        logger.info(f"  Tests:       {result['tests_generated']}")
        logger.info(f"  Provider:    {result['provider'].upper()}")
        logger.info(f"  Time:        {result['elapsed_seconds']:.0f}s ({result['elapsed_seconds']/60:.1f}min)")
        logger.info(f"\n💰 Cost:")
        logger.info(f"  Total:       ${result['total_cost_usd']:.2f}")
        logger.info(f"  Provider:    {'FREE (Gemini)' if result['provider'] == 'gemini' else 'OpenAI'}")
        logger.info("\n" + "="*60 + "\n")


async def main():
    """Main entry point for coverage optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Coverage Optimizer")
    parser.add_argument("--source-dir", default="src", help="Source directory")
    parser.add_argument("--target-coverage", type=float, default=80, help="Target coverage %")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="AI provider")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--max-cost", type=float, default=1.0, help="Max cost USD")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API keys from environment
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key:
        logger.error("❌ GOOGLE_API_KEY not found in environment")
        logger.info("Set it: export GOOGLE_API_KEY=your_key_here")
        return 1
    
    # Initialize optimizer
    optimizer = GeminiCoverageOptimizer(
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        provider=args.provider,
        max_cost_usd=args.max_cost
    )
    
    # Run optimization
    result = await optimizer.optimize_coverage(
        source_dir=args.source_dir,
        target_coverage=args.target_coverage / 100.0,
        max_iterations=args.max_iterations
    )
    
    # Return exit code
    return 0 if result['target_achieved'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
