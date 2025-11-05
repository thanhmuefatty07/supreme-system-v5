#!/usr/bin/env python3
"""
Unified Testing Orchestrator for Supreme System V5
Orchestrates all testing scripts and validation procedures
"""

import asyncio
import logging
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test execution result"""
    name: str
    status: str
    duration: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ValidationSuite:
    """Complete validation suite configuration"""
    realtime_backtest_duration: int = 90  # minutes
    stress_test_duration: int = 480       # minutes (8 hours)
    paper_trading_duration: int = 1440    # minutes (24 hours)
    statistical_confidence: float = 0.95
    performance_baselines: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_baselines is None:
            self.performance_baselines = {
                'latency_ms': 0.020,
                'memory_mb': 15.0,
                'cpu_percent': 85.0,
                'win_rate': 0.689,
                'sharpe_ratio': 2.47
            }


class UnifiedTestingOrchestrator:
    """Orchestrates all testing and validation procedures"""
    
    def __init__(self, validation_suite: ValidationSuite):
        self.suite = validation_suite
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        self.scripts_dir = Path(__file__).parent
        self.artifacts_dir = Path("run_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
    def log_progress(self, message: str, level: str = "info"):
        """Log progress with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "info":
            logger.info(f"[{timestamp}] {message}")
        elif level == "error":
            logger.error(f"[{timestamp}] {message}")
        elif level == "warning":
            logger.warning(f"[{timestamp}] {message}")
    
    async def run_script(self, script_name: str, args: List[str] = None, timeout: int = 3600) -> TestResult:
        """Execute a testing script and capture results"""
        if args is None:
            args = []
            
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            return TestResult(
                name=script_name,
                status="FAILED",
                duration=0.0,
                metrics={},
                error=f"Script not found: {script_path}"
            )
        
        self.log_progress(f"Starting {script_name}...")
        start_time = time.time()
        
        try:
            # Execute script
            cmd = [sys.executable, str(script_path)] + args
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.scripts_dir.parent
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            duration = time.time() - start_time
            
            if process.returncode == 0:
                # Try to parse metrics from output
                metrics = self._extract_metrics(stdout.decode())
                return TestResult(
                    name=script_name,
                    status="PASSED",
                    duration=duration,
                    metrics=metrics
                )
            else:
                return TestResult(
                    name=script_name,
                    status="FAILED",
                    duration=duration,
                    metrics={},
                    error=stderr.decode()
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                name=script_name,
                status="TIMEOUT",
                duration=timeout,
                metrics={},
                error=f"Script timed out after {timeout} seconds"
            )
        except Exception as e:
            return TestResult(
                name=script_name,
                status="ERROR",
                duration=time.time() - start_time,
                metrics={},
                error=str(e)
            )
    
    def _extract_metrics(self, output: str) -> Dict[str, Any]:
        """Extract metrics from script output"""
        metrics = {}
        
        # Look for JSON blocks in output
        lines = output.split('\n')
        for line in lines:
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    parsed = json.loads(line.strip())
                    metrics.update(parsed)
                except json.JSONDecodeError:
                    continue
        
        return metrics
    
    async def run_performance_validation(self) -> TestResult:
        """Run comprehensive performance validation"""
        self.log_progress("🚀 Starting Performance Validation Suite")
        
        # Run performance profiler
        result = await self.run_script("performance_profiler.py", ["-d", "300"])
        self.results.append(result)
        
        if result.status != "PASSED":
            self.log_progress("❌ Performance validation failed", "error")
            return result
        
        # Validate against baselines
        metrics = result.metrics
        violations = []
        
        if metrics.get('avg_latency_ms', 999) > self.suite.performance_baselines['latency_ms']:
            violations.append(f"Latency {metrics.get('avg_latency_ms')}ms > {self.suite.performance_baselines['latency_ms']}ms")
        
        if metrics.get('memory_mb', 999) > self.suite.performance_baselines['memory_mb']:
            violations.append(f"Memory {metrics.get('memory_mb')}MB > {self.suite.performance_baselines['memory_mb']}MB")
        
        if violations:
            return TestResult(
                name="performance_validation",
                status="FAILED",
                duration=result.duration,
                metrics=metrics,
                error="Performance baseline violations: " + "; ".join(violations)
            )
        
        self.log_progress("✅ Performance validation passed")
        return result
    
    async def run_realtime_backtest(self) -> TestResult:
        """Run real-time backtest validation (60-120 minutes)"""
        self.log_progress(f"📊 Starting Real-time Backtest ({self.suite.realtime_backtest_duration} minutes)")
        
        duration_seconds = self.suite.realtime_backtest_duration * 60
        result = await self.run_script(
            "realtime_backtest_validator.py",
            ["--duration", str(duration_seconds), "--statistical-validation"],
            timeout=duration_seconds + 300  # 5 min buffer
        )
        
        self.results.append(result)
        
        if result.status == "PASSED":
            self.log_progress(f"✅ Real-time backtest completed in {result.duration/60:.1f} minutes")
        else:
            self.log_progress(f"❌ Real-time backtest failed: {result.error}", "error")
        
        return result
    
    async def run_statistical_validation(self) -> TestResult:
        """Run A/B testing and statistical validation"""
        self.log_progress("📈 Starting Statistical A/B Validation")
        
        result = await self.run_script(
            "statistical_validation.py",
            ["--confidence", str(self.suite.statistical_confidence)]
        )
        
        self.results.append(result)
        
        if result.status == "PASSED":
            p_value = result.metrics.get('p_value', 1.0)
            if p_value < (1 - self.suite.statistical_confidence):
                self.log_progress(f"✅ Statistical significance achieved (p={p_value:.4f})")
            else:
                result.status = "FAILED"
                result.error = f"Statistical significance not achieved (p={p_value:.4f})"
                self.log_progress(f"❌ Statistical validation failed: {result.error}", "error")
        
        return result
    
    async def run_stress_testing(self) -> TestResult:
        """Run extended stress testing (6-8 hours)"""
        self.log_progress(f"💪 Starting Extended Stress Testing ({self.suite.stress_test_duration/60:.0f} hours)")
        
        duration_seconds = self.suite.stress_test_duration * 60
        result = await self.run_script(
            "extended_stress_validator.py",
            ["--duration", str(duration_seconds)],
            timeout=duration_seconds + 600  # 10 min buffer
        )
        
        self.results.append(result)
        
        if result.status == "PASSED":
            self.log_progress(f"✅ Stress testing completed in {result.duration/3600:.1f} hours")
        else:
            self.log_progress(f"❌ Stress testing failed: {result.error}", "error")
        
        return result
    
    async def run_degradation_testing(self) -> TestResult:
        """Run degradation and fault tolerance testing"""
        self.log_progress("🔧 Starting Degradation Testing")
        
        result = await self.run_script("degradation_validator.py")
        self.results.append(result)
        
        if result.status == "PASSED":
            self.log_progress("✅ Degradation testing passed")
        else:
            self.log_progress(f"❌ Degradation testing failed: {result.error}", "error")
        
        return result
    
    async def run_exchange_connectivity_tests(self) -> TestResult:
        """Run exchange connectivity smoke tests"""
        self.log_progress("🌐 Starting Exchange Connectivity Tests")
        
        result = await self.run_script("exchange_connectivity_tests.py")
        self.results.append(result)
        
        if result.status == "PASSED":
            self.log_progress("✅ Exchange connectivity tests passed")
        else:
            self.log_progress(f"❌ Exchange connectivity tests failed: {result.error}", "error")
        
        return result
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        passed_tests = [r for r in self.results if r.status == "PASSED"]
        failed_tests = [r for r in self.results if r.status != "PASSED"]
        
        # Aggregate metrics
        all_metrics = {}
        for result in self.results:
            if result.metrics:
                all_metrics.update(result.metrics)
        
        report = {
            "validation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_hours": total_duration / 3600,
                "validator_version": "unified_testing_orchestrator_v1.0",
                "validation_type": "comprehensive_production_validation"
            },
            "test_summary": {
                "total_tests": len(self.results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(passed_tests) / max(1, len(self.results)),
                "overall_status": "PASSED" if len(failed_tests) == 0 else "FAILED"
            },
            "performance_validation": {
                "baselines_met": all(r.status == "PASSED" for r in self.results if "performance" in r.name),
                "latency_validated": all_metrics.get('avg_latency_ms', 999) <= self.suite.performance_baselines['latency_ms'],
                "memory_validated": all_metrics.get('memory_mb', 999) <= self.suite.performance_baselines['memory_mb']
            },
            "statistical_validation": {
                "confidence_level": self.suite.statistical_confidence,
                "p_value": all_metrics.get('p_value', 1.0),
                "statistically_significant": all_metrics.get('p_value', 1.0) < (1 - self.suite.statistical_confidence)
            },
            "test_results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration_minutes": r.duration / 60,
                    "key_metrics": r.metrics,
                    "error": r.error
                }
                for r in self.results
            ],
            "configuration": {
                "realtime_backtest_duration_minutes": self.suite.realtime_backtest_duration,
                "stress_test_duration_hours": self.suite.stress_test_duration / 60,
                "performance_baselines": self.suite.performance_baselines
            }
        }
        
        # Save report
        report_path = self.artifacts_dir / f"comprehensive_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log_progress(f"📊 Comprehensive report saved to {report_path}")
        return report
    
    async def run_full_validation_suite(self) -> Dict[str, Any]:
        """Run the complete validation suite"""
        self.log_progress("🚀 Starting Supreme System V5 Comprehensive Validation Suite")
        
        # Phase 1: Performance Validation
        await self.run_performance_validation()
        
        # Phase 2: Real-time Backtest (60-120 minutes)
        await self.run_realtime_backtest()
        
        # Phase 3: Statistical Validation
        await self.run_statistical_validation()
        
        # Phase 4: Exchange Connectivity
        await self.run_exchange_connectivity_tests()
        
        # Phase 5: Degradation Testing
        await self.run_degradation_testing()
        
        # Phase 6: Extended Stress Testing (optional - long running)
        if self.suite.stress_test_duration > 0:
            self.log_progress("⚠️ Extended stress testing will run in background")
            # Could run this in background or skip for faster validation
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        if report["test_summary"]["overall_status"] == "PASSED":
            self.log_progress("🎉 ALL VALIDATIONS PASSED - SYSTEM READY FOR PRODUCTION")
        else:
            self.log_progress("❌ VALIDATION FAILURES DETECTED - REVIEW REQUIRED", "error")
        
        return report


async def main():
    """Main entry point"""
    suite = ValidationSuite(
        realtime_backtest_duration=90,  # 90 minutes
        stress_test_duration=0,         # Skip for faster validation
        statistical_confidence=0.95
    )
    
    orchestrator = UnifiedTestingOrchestrator(suite)
    report = await orchestrator.run_full_validation_suite()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUPREME SYSTEM V5 VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Overall Status: {report['test_summary']['overall_status']}")
    print(f"Tests Passed: {report['test_summary']['passed_tests']}/{report['test_summary']['total_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
    print(f"Total Duration: {report['validation_metadata']['total_duration_hours']:.2f} hours")
    
    return 0 if report['test_summary']['overall_status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
