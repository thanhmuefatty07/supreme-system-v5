"""

Enterprise Gemini Analyzer

Multi-key parallel processing with rate limiting and error handling

"""

import google.generativeai as genai
import os
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class AnalysisTask:
    """Represents a single analysis task"""
    name: str
    prompt: str
    context_files: List[str]
    output_file: str

class KeyManager:
    """Manages multiple API keys with rate limiting"""

    def __init__(self, keys: List[str], rpm_limit: int = 14):
        self.keys = keys
        self.rpm_limit = rpm_limit
        self.call_history = {i: [] for i in range(len(keys))}
        self.current_idx = 0
        logger.info(f"Initialized KeyManager with {len(keys)} keys")

    def get_available_key(self) -> Tuple[int, str]:
        """Get key index and key string with available capacity"""
        now = time.time()

        for i in range(len(self.keys)):
            idx = (self.current_idx + i) % len(self.keys)

            # Clean old timestamps
            self.call_history[idx] = [
                t for t in self.call_history[idx]
                if now - t < 60
            ]

            # Check capacity
            if len(self.call_history[idx]) < self.rpm_limit:
                self.call_history[idx].append(now)
                self.current_idx = (idx + 1) % len(self.keys)
                return idx, self.keys[idx]

        # All keys at limit
        wait_time = 61 - (now - min([min(h) for h in self.call_history.values() if h]))
        logger.warning(".1f")
        time.sleep(wait_time)
        return self.get_available_key()

class GeminiAnalyzer:
    """Enterprise-grade analyzer using multiple Gemini API keys"""

    def __init__(self, api_keys: List[str]):
        self.key_manager = KeyManager(api_keys)
        self.uploaded_files = []
        self.models = {}

        # Initialize models for each key
        for i, key in enumerate(api_keys):
            genai.configure(api_key=key)
            self.models[i] = genai.GenerativeModel('gemini-2.0-flash-exp')

        logger.info("GeminiAnalyzer initialized")

    def upload_files_parallel(self, max_workers: int = 6) -> List:
        """Upload repo files in parallel using all keys"""
        logger.info("Starting parallel file upload")

        # Collect files to upload
        files_map = {
            'context': [
                'verification_results.json',
                'coverage.json',
                'test_results.txt'
            ],
            'source': list(Path('src').rglob('*.py'))[:50],
            'tests': list(Path('tests').rglob('test_*.py'))[:30]
        }

        all_files = []
        for category, files in files_map.items():
            all_files.extend([str(f) for f in files if Path(f).exists()])

        logger.info(f"Found {len(all_files)} files to upload")

        # Split files across keys
        batch_size = len(all_files) // len(self.key_manager.keys) + 1
        batches = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]

        uploaded = []

        def upload_batch(files: List[str], key_idx: int) -> List:
            """Upload batch with specific key"""
            genai.configure(api_key=self.key_manager.keys[key_idx])
            batch_uploaded = []

            for file_path in files:
                try:
                    uploaded_file = genai.upload_file(file_path)
                    batch_uploaded.append(uploaded_file)
                    logger.info(f"[Key {key_idx}] Uploaded: {file_path}")
                except Exception as e:
                    logger.error(f"[Key {key_idx}] Failed {file_path}: {e}")

            return batch_uploaded

        # Upload in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(upload_batch, batch, i): i
                for i, batch in enumerate(batches[:len(self.key_manager.keys)])
            }

            for future in as_completed(futures):
                uploaded.extend(future.result())

        self.uploaded_files = uploaded
        logger.info(f"Upload complete: {len(uploaded)} files")
        return uploaded

    def run_analysis_task(self, task: AnalysisTask) -> str:
        """Run single analysis task"""
        key_idx, key = self.key_manager.get_available_key()
        genai.configure(api_key=key)
        model = self.models[key_idx]

        logger.info(f"[{task.name}] Running on Key {key_idx}")

        try:
            # Select relevant uploaded files
            context = [f for f in self.uploaded_files if any(
                ctx in str(f.name).lower()
                for ctx in task.context_files
            )][:20]

            response = model.generate_content(
                [task.prompt] + context,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4000,
                    top_p=0.95
                )
            )

            logger.info(f"[{task.name}] Complete")
            return response.text

        except Exception as e:
            logger.error(f"[{task.name}] Failed: {e}")
            return f"ERROR: {str(e)}"

    def run_parallel_analysis(self, tasks: List[AnalysisTask]) -> Dict[str, str]:
        """Run multiple analysis tasks in parallel"""
        logger.info(f"Starting parallel analysis of {len(tasks)} tasks")

        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.run_analysis_task, task): task.name
                for task in tasks
            }

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    logger.error(f"[{task_name}] Exception: {e}")
                    results[task_name] = f"EXCEPTION: {str(e)}"

        logger.info("Parallel analysis complete")
        return results

    def save_results(self, results: Dict[str, str], output_dir: Path):
        """Save analysis results with proper structure"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual reports
        for task_name, content in results.items():
            file_path = output_dir / f"{task_name}.md"
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved: {file_path}")

        # Create combined report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined = f"""# Supreme System V5 - Enterprise Analysis Report

**Generated:** {timestamp}
**Keys Used:** {len(self.key_manager.keys)}
**Files Analyzed:** {len(self.uploaded_files)}

---

"""

        for task_name, content in results.items():
            combined += f"\n## {task_name.replace('_', ' ').title()}\n\n{content}\n\n---\n\n"

        main_report = output_dir / "FULL_ANALYSIS_REPORT.md"
        main_report.write_text(combined, encoding='utf-8')

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'keys_used': len(self.key_manager.keys),
            'files_analyzed': len(self.uploaded_files),
            'tasks_completed': list(results.keys())
        }

        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8'
        )

        logger.info(f"Main report: {main_report}")
        return main_report

def create_analysis_tasks() -> List[AnalysisTask]:
    """Define all analysis tasks"""
    return [
        AnalysisTask(
            name="coverage_deep_dive",
            prompt="""Analyze coverage.json in detail:

1. List top 20 files by impact (lines × uncovered %)

2. Calculate coverage projection if top 10/15/20 tested

3. Identify 3 quick wins (high impact, <3h effort each)

4. Suggest testing order for maximum gain

Output as detailed markdown table with effort estimates.""",
            context_files=['coverage', 'verification'],
            output_file="coverage_analysis.md"
        ),

        AnalysisTask(
            name="failed_tests_categorization",
            prompt="""Categorize the 121 failed tests:

1. Import/collection errors (count + example failures)

2. Logic/assertion bugs (list top 10 with file:line)

3. Environment/config issues (Windows-specific?)

4. Async/race conditions

For each category, provide:

- Count

- Root cause analysis

- Batch fix strategy (if applicable)

- Estimated fix time

Output as structured JSON + markdown explanation.""",
            context_files=['test_results', 'verification'],
            output_file="failed_tests.md"
        ),

        AnalysisTask(
            name="architecture_review",
            prompt="""Review project architecture:

1. Module dependency analysis (identify circular deps)

2. Code quality assessment (smells, anti-patterns)

3. Testing infrastructure evaluation

4. Best practices compliance check

Provide:

- Architecture diagram (mermaid format)

- Issues list with severity (critical/high/medium/low)

- Refactoring recommendations

- Technical debt assessment

Output as comprehensive markdown report.""",
            context_files=['src', 'tests'],
            output_file="architecture.md"
        ),

        AnalysisTask(
            name="test_generation_strategy",
            prompt="""Create comprehensive test generation strategy:

1. Test templates for common patterns (strategy, validator, client)

2. Fixture recommendations (sample data, mocks)

3. Mocking strategies (API calls, async operations)

4. Parametrize examples for edge cases

5. Coverage optimization techniques

Provide:

- 3 complete test file templates (ready to use)

- Testing best practices guide

- Common pitfalls to avoid

Output as actionable markdown guide.""",
            context_files=['tests', 'coverage'],
            output_file="test_strategy.md"
        )
    ]

def main():
    """Main execution"""
    print("=" * 80)
    print("  SUPREME SYSTEM V5 - ENTERPRISE GEMINI ANALYSIS")
    print("  Using 6 API Keys for Parallel Processing")
    print("=" * 80)
    print()

    # Load API keys
    api_keys = [
        os.getenv(f"GEMINI_API_KEY_{i}")
        for i in range(1, 7)
    ]

    api_keys = [k for k in api_keys if k]

    if len(api_keys) < 6:
        logger.error(f"Expected 6 keys, found {len(api_keys)}")
        print("❌ Error: 6 API keys required")
        print("   Set: GEMINI_API_KEY_1 through GEMINI_API_KEY_6 in .env")
        return 1

    print(f"✅ Loaded {len(api_keys)} API keys\n")

    # Initialize analyzer
    start_time = time.time()
    analyzer = GeminiAnalyzer(api_keys)

    # Upload files
    print("📦 PHASE 1: File Upload (Parallel)")
    print("-" * 80)
    uploaded = analyzer.upload_files_parallel(max_workers=6)
    upload_time = time.time() - start_time
    print(".1f")
    print()

    # Run analysis
    print("🧠 PHASE 2: Analysis (Parallel)")
    print("-" * 80)
    tasks = create_analysis_tasks()
    analysis_start = time.time()
    results = analyzer.run_parallel_analysis(tasks)
    analysis_time = time.time() - analysis_start
    print(".1f")
    print()

    # Save results
    print("💾 PHASE 3: Saving Results")
    print("-" * 80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"analysis_reports/{timestamp}")
    report_path = analyzer.save_results(results, output_dir)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"  ✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(".1f")
    print(f"   Files analyzed: {len(uploaded)}")
    print(f"   Tasks completed: {len(results)}")
    print(f"   Report: {report_path}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"   1. Review: {report_path}")
    print(f"   2. Check individual reports in: {output_dir}/")
    print(f"   3. Commit to branch: week2/analysis-and-planning")

    return 0

if __name__ == "__main__":
    exit(main())



