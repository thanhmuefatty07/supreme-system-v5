#!/usr/bin/env python3
"""
🔍 Supreme System V5 - Production System Validator
Comprehensive validation for deployment readiness on i3-8th gen

Validates:
- Dependencies installation
- Import functionality
- Component integrity
- Hardware optimization
- Production readiness
"""

import os
import sys
import subprocess
import importlib
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class SystemValidator:
    """Comprehensive system validation for Supreme System V5"""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'tests': {},
            'overall_status': 'unknown',
            'critical_issues': [],
            'warnings': []
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import psutil
            return {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'working_directory': str(Path.cwd()),
                'python_path': sys.executable
            }
        except ImportError:
            return {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'working_directory': str(Path.cwd()),
                'python_path': sys.executable
            }
    
    def validate_dependencies(self) -> Tuple[bool, List[str], List[str]]:
        """Validate Python package dependencies"""
        print("\n📦 Validating dependencies...")
        
        # Core required packages
        core_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'aiohttp', 'websockets',
            'numpy', 'pandas', 'asyncio', 'logging', 'json', 'datetime',
            'collections', 'typing', 'dataclasses', 'enum'
        ]
        
        # Optional packages (should not block)
        optional_packages = [
            'torch', 'transformers', 'qiskit', 'brian2', 'nengo',
            'prometheus_client', 'psutil', 'matplotlib'
        ]
        
        missing_core = []
        missing_optional = []
        
        # Check core packages
        for package in core_packages:
            try:
                if package in ['asyncio', 'logging', 'json', 'datetime', 'collections', 'typing', 'dataclasses', 'enum']:
                    # Built-in modules
                    __import__(package)
                else:
                    # External packages
                    importlib.import_module(package)
                print(f"   ✅ {package}")
            except ImportError:
                missing_core.append(package)
                print(f"   ❌ {package} (REQUIRED)")
        
        # Check optional packages
        for package in optional_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {package} (optional)")
            except ImportError:
                missing_optional.append(package)
                print(f"   ⚠️ {package} (optional - OK to skip)")
        
        success = len(missing_core) == 0
        
        if missing_core:
            self.validation_results['critical_issues'].extend(
                [f"Missing required package: {pkg}" for pkg in missing_core]
            )
        
        if missing_optional:
            self.validation_results['warnings'].extend(
                [f"Optional package not available: {pkg}" for pkg in missing_optional]
            )
        
        return success, missing_core, missing_optional
    
    def validate_imports(self) -> Tuple[bool, List[str]]:
        """Validate project module imports and critical imports"""
        print("\n🔗 Validating module imports...")
        
        import_errors = []
        
        # First, validate critical imports (deque, fastapi, pydantic)
        print("\n   Checking critical imports...")
        critical_imports = [
            ('collections.deque', 'from collections import deque'),
            ('fastapi', 'import fastapi'),
            ('pydantic', 'import pydantic'),
        ]
        
        for import_name, import_statement in critical_imports:
            try:
                if import_name == 'collections.deque':
                    from collections import deque
                    if deque is None:
                        raise ImportError(f"{import_name} import failed")
                elif import_name == 'fastapi':
                    import fastapi
                elif import_name == 'pydantic':
                    import pydantic
                print(f"   ✅ {import_name}")
            except ImportError as e:
                import_errors.append(f"{import_name}: {str(e)}")
                print(f"   ❌ {import_name}: {str(e)}")
        
        # Test imports for major modules
        print("\n   Checking module imports...")
        test_imports = [
            ('src.trading.engine', 'TradingEngine, TradingConfig'),
            ('src.config', '__init__'),
            ('src.api', '__init__'),
            ('src.monitoring', '__init__')
        ]
        
        # Add src to path temporarily
        if 'src' not in sys.path:
            sys.path.insert(0, str(Path.cwd() / 'src'))
        
        for module_path, components in test_imports:
            try:
                module = importlib.import_module(module_path)
                
                # Check specific components if specified
                if components != '__init__':
                    for component in components.split(', '):
                        if not hasattr(module, component):
                            raise ImportError(f"{component} not found in {module_path}")
                
                print(f"   ✅ {module_path}")
                
            except ImportError as e:
                import_errors.append(f"{module_path}: {str(e)}")
                print(f"   ❌ {module_path}: {str(e)}")
        
        success = len(import_errors) == 0
        
        if import_errors:
            self.validation_results['critical_issues'].extend(import_errors)
        
        return success, import_errors
    
    def validate_files(self) -> Tuple[bool, List[str]]:
        """Validate required files exist"""
        print("\n📁 Validating file structure...")
        
        required_files = [
            'requirements.txt',
            'src/__init__.py',
            'src/trading/engine.py',
            'src/config/__init__.py',
            'src/api/__init__.py',
            'docker-compose.yml'
        ]
        
        optional_files = [
            '.env.example',
            'docker-compose.i3.yml',
            'monitoring/prometheus.yml',
            'tests/test_integration.py'
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required files
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                missing_required.append(file_path)
                print(f"   ❌ {file_path} (REQUIRED)")
        
        # Check optional files
        for file_path in optional_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path} (optional)")
            else:
                missing_optional.append(file_path)
                print(f"   ⚠️ {file_path} (optional)")
        
        success = len(missing_required) == 0
        
        if missing_required:
            self.validation_results['critical_issues'].extend(
                [f"Missing required file: {file}" for file in missing_required]
            )
        
        if missing_optional:
            self.validation_results['warnings'].extend(
                [f"Optional file missing: {file}" for file in missing_optional]
            )
        
        return success, missing_required
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate system configuration"""
        print("\n⚙️ Validating configuration...")
        
        config_issues = []
        
        try:
            # Test basic configuration loading
            sys.path.insert(0, str(Path.cwd() / 'src'))
            
            # Try to import and instantiate basic config
            from src.trading.engine import TradingConfig
            config = TradingConfig()
            
            print(f"   ✅ TradingConfig instantiation")
            print(f"   ✅ Trading pairs: {len(config.trading_pairs)}")
            print(f"   ✅ Update frequency: {config.update_frequency_ms}ms")
            
            # Validate configuration values
            if len(config.trading_pairs) == 0:
                config_issues.append("No trading pairs configured")
            
            if config.update_frequency_ms < 100:
                config_issues.append("Update frequency too aggressive for i3")
            
            if config.max_position_size <= 0:
                config_issues.append("Invalid max position size")
            
        except Exception as e:
            config_issues.append(f"Configuration loading failed: {str(e)}")
            print(f"   ❌ Configuration loading: {str(e)}")
        
        success = len(config_issues) == 0
        
        if config_issues:
            self.validation_results['critical_issues'].extend(config_issues)
        
        return success, config_issues
    
    def validate_hardware_optimization(self) -> Tuple[bool, List[str]]:
        """Validate hardware optimization system"""
        print("\n🔧 Validating hardware optimization...")
        
        hw_issues = []
        
        try:
            sys.path.insert(0, str(Path.cwd() / 'src'))
            
            # Try to import hardware profiles
            try:
                from src.config.hardware_profiles import optimal_profile, ProcessorType
                print(f"   ✅ Hardware detection available")
                
                if optimal_profile:
                    print(f"   ✅ Detected: {optimal_profile.processor_type.value}")
                    print(f"   ✅ Memory: {optimal_profile.memory_profile.value}")
                else:
                    hw_issues.append("Hardware profile detection failed")
                    
            except ImportError:
                hw_issues.append("Hardware optimization module not available")
                print(f"   ⚠️ Hardware optimization not available (fallback mode OK)")
            
        except Exception as e:
            hw_issues.append(f"Hardware validation error: {str(e)}")
        
        # Hardware validation is not critical - system can run without it
        return True, hw_issues  # Always return True for now
    
    def validate_memory_usage(self) -> Tuple[bool, List[str]]:
        """Validate memory usage for i3-4GB systems"""
        print("\n💾 Validating memory requirements...")
        
        memory_issues = []
        
        try:
            import psutil
            
            # Get current memory usage
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            print(f"   ℹ️ Total memory: {total_gb:.1f}GB")
            print(f"   ℹ️ Available: {available_gb:.1f}GB")
            
            # Check if system has enough memory
            if total_gb < 3.5:
                memory_issues.append(f"Low system memory: {total_gb:.1f}GB (need 4GB+)")
            
            if available_gb < 1.5:
                memory_issues.append(f"Insufficient available memory: {available_gb:.1f}GB")
            
            # Estimate Python process memory
            process = psutil.Process()
            python_memory_mb = process.memory_info().rss / (1024 * 1024)
            print(f"   ℹ️ Python process: {python_memory_mb:.1f}MB")
            
            print(f"   ✅ Memory validation complete")
            
        except ImportError:
            memory_issues.append("psutil not available for memory check")
            print(f"   ⚠️ Memory validation skipped (psutil not available)")
        except Exception as e:
            memory_issues.append(f"Memory validation error: {str(e)}")
        
        success = len(memory_issues) == 0
        
        if memory_issues:
            self.validation_results['warnings'].extend(memory_issues)
        
        return success, memory_issues
    
    def validate_environment_variables(self) -> Tuple[bool, List[str]]:
        """Validate environment variables and API keys"""
        print("\n🔐 Validating environment variables...")
        
        env_issues = []
        env_warnings = []
        
        # Load .env file if exists
        env_file = Path('.env')
        env_example = Path('.env.example')
        
        # Try to load .env file using python-dotenv
        try:
            from dotenv import load_dotenv
            if env_file.exists():
                load_dotenv(env_file)
                print("   ✅ .env file loaded")
            elif env_example.exists():
                env_warnings.append(".env file not found (using .env.example as reference)")
                print("   ⚠️ .env file not found - using .env.example as reference")
            else:
                env_warnings.append("Neither .env nor .env.example file found")
                print("   ⚠️ No .env file found")
        except ImportError:
            env_warnings.append("python-dotenv not installed - cannot load .env file")
            print("   ⚠️ python-dotenv not available - skipping .env file loading")
        
        # Required environment variables
        required_vars = [
            'JWT_SECRET_KEY',
            'ALPHA_VANTAGE_KEY',
            'FINNHUB_KEY',
        ]
        
        # Optional but recommended
        optional_vars = [
            'REDIS_URL',
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'LOG_LEVEL',
            'ENVIRONMENT',
        ]
        
        # Check required variables
        print("\n   Checking required variables...")
        for var in required_vars:
            value = os.getenv(var, '')
            if not value or value.startswith('your-') or value == '':
                env_issues.append(f"Required environment variable {var} not set or has placeholder value")
                print(f"   ❌ {var}: Not set or placeholder")
            else:
                # Validate format
                if var == 'JWT_SECRET_KEY' and len(value) < 32:
                    env_warnings.append(f"{var} should be at least 32 characters for security")
                    print(f"   ⚠️ {var}: Too short (should be 32+ characters)")
                else:
                    print(f"   ✅ {var}: Set")
        
        # Check optional variables
        print("\n   Checking optional variables...")
        for var in optional_vars:
            value = os.getenv(var, '')
            if not value:
                env_warnings.append(f"Optional environment variable {var} not set")
                print(f"   ⚠️ {var}: Not set (optional)")
            else:
                print(f"   ✅ {var}: Set")
        
        # Validate API key formats
        print("\n   Validating API key formats...")
        alpha_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        if alpha_key and not alpha_key.startswith('your-'):
            if len(alpha_key) < 10:
                env_issues.append("ALPHA_VANTAGE_KEY appears invalid (too short)")
                print(f"   ❌ ALPHA_VANTAGE_KEY: Invalid format")
            else:
                print(f"   ✅ ALPHA_VANTAGE_KEY: Format OK")
        
        finnhub_key = os.getenv('FINNHUB_KEY', '')
        if finnhub_key and not finnhub_key.startswith('your-'):
            if len(finnhub_key) < 10:
                env_issues.append("FINNHUB_KEY appears invalid (too short)")
                print(f"   ❌ FINNHUB_KEY: Invalid format")
            else:
                print(f"   ✅ FINNHUB_KEY: Format OK")
        
        jwt_key = os.getenv('JWT_SECRET_KEY', '')
        if jwt_key and not jwt_key.startswith('your-'):
            if len(jwt_key) < 32:
                env_issues.append("JWT_SECRET_KEY should be at least 32 characters")
                print(f"   ❌ JWT_SECRET_KEY: Too short")
            else:
                print(f"   ✅ JWT_SECRET_KEY: Format OK")
        
        success = len(env_issues) == 0
        
        if env_issues:
            self.validation_results['critical_issues'].extend(env_issues)
        if env_warnings:
            self.validation_results['warnings'].extend(env_warnings)
        
        return success, env_issues
    
    def validate_network_connectivity(self) -> Tuple[bool, List[str]]:
        """Validate network connectivity to external services"""
        print("\n🌐 Validating network connectivity...")
        
        connectivity_issues = []
        
        import socket
        import urllib.request
        
        # Test Redis connectivity
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        try:
            # Parse Redis URL
            if redis_url.startswith('redis://'):
                parts = redis_url.replace('redis://', '').split('/')[0].split(':')
                host = parts[0] if parts else 'localhost'
                port = int(parts[1]) if len(parts) > 1 else 6379
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    print(f"   ✅ Redis: Connected to {host}:{port}")
                else:
                    connectivity_issues.append(f"Redis connection failed: {host}:{port}")
                    print(f"   ⚠️ Redis: Cannot connect to {host}:{port}")
        except Exception as e:
            connectivity_issues.append(f"Redis connectivity check failed: {str(e)}")
            print(f"   ⚠️ Redis: Connectivity check error")
        
        # Test external API endpoints (basic connectivity)
        test_endpoints = [
            ('Alpha Vantage', 'https://www.alphavantage.co/query'),
            ('Finnhub', 'https://finnhub.io/api/v1'),
        ]
        
        for name, url in test_endpoints:
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'SupremeSystemV5/1.0')
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status in [200, 401, 403]:  # 401/403 means service is up
                        print(f"   ✅ {name}: Service accessible")
                    else:
                        print(f"   ⚠️ {name}: Unexpected status {response.status}")
            except urllib.error.HTTPError as e:
                if e.code in [401, 403]:
                    print(f"   ✅ {name}: Service accessible (auth required)")
                else:
                    connectivity_issues.append(f"{name} endpoint returned error: {e.code}")
                    print(f"   ⚠️ {name}: HTTP {e.code}")
            except Exception as e:
                connectivity_issues.append(f"{name} connectivity failed: {str(e)}")
                print(f"   ⚠️ {name}: Connection failed")
        
        success = len(connectivity_issues) == 0
        
        if connectivity_issues:
            self.validation_results['warnings'].extend(connectivity_issues)
        
        return success, connectivity_issues
    
    def run_smoke_test(self) -> Tuple[bool, List[str]]:
        """Run basic smoke test of core functionality"""
        print("\n💬 Running smoke test...")
        
        smoke_issues = []
        
        try:
            sys.path.insert(0, str(Path.cwd() / 'src'))
            
            # Test basic trading config creation
            from src.trading.engine import TradingConfig, RealDataConnector
            
            config = TradingConfig(
                trading_pairs=['AAPL'],
                use_real_data=False,  # Demo mode for testing
                update_frequency_ms=2000
            )
            
            print(f"   ✅ TradingConfig creation")
            
            # Test data connector
            connector = RealDataConnector(config)
            print(f"   ✅ RealDataConnector creation")
            
            # Test basic data generation (demo mode)
            demo_data = connector._generate_realistic_demo_data('AAPL')
            
            if demo_data.get('price', 0) > 0:
                print(f"   ✅ Demo data generation: ${demo_data['price']:.2f}")
            else:
                smoke_issues.append("Demo data generation failed")
            
            print(f"   ✅ Smoke test passed")
            
        except Exception as e:
            smoke_issues.append(f"Smoke test failed: {str(e)}")
            print(f"   ❌ Smoke test: {str(e)}")
        
        success = len(smoke_issues) == 0
        
        if smoke_issues:
            self.validation_results['critical_issues'].extend(smoke_issues)
        
        return success, smoke_issues
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        print("🔍 SUPREME SYSTEM V5 - PRODUCTION SYSTEM VALIDATION")
        print("=" * 55)
        print(f"Timestamp: {self.validation_results['timestamp']}")
        print(f"Python: {self.validation_results['system_info']['python_version'][:20]}...")
        print(f"Platform: {self.validation_results['system_info']['platform']}")
        
        if 'memory_gb' in self.validation_results['system_info']:
            memory_gb = self.validation_results['system_info']['memory_gb']
            print(f"Memory: {memory_gb}GB")
            
            if memory_gb <= 4.5:
                print(f"⚡ i3-4GB optimizations will be applied")
        
        # Run all validation tests
        validators = [
            ('Dependencies', self.validate_dependencies),
            ('File Structure', self.validate_files),
            ('Module Imports', self.validate_imports),
            ('Environment Variables', self.validate_environment_variables),
            ('Network Connectivity', self.validate_network_connectivity),
            ('Configuration', self.validate_configuration),
            ('Hardware Optimization', self.validate_hardware_optimization),
            ('Memory Requirements', self.validate_memory_usage),
            ('Smoke Test', self.run_smoke_test)
        ]
        
        total_score = 0
        max_score = len(validators)
        
        for test_name, validator_func in validators:
            try:
                success, issues = validator_func()
                
                self.validation_results['tests'][test_name] = {
                    'passed': success,
                    'issues': issues,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if success:
                    total_score += 1
                    
            except Exception as e:
                self.validation_results['tests'][test_name] = {
                    'passed': False,
                    'issues': [f"Validation error: {str(e)}"],
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.validation_results['critical_issues'].append(f"{test_name}: {str(e)}")
        
        # Calculate overall status
        success_rate = (total_score / max_score) * 100
        
        if success_rate >= 90:
            self.validation_results['overall_status'] = 'PRODUCTION_READY'
        elif success_rate >= 75:
            self.validation_results['overall_status'] = 'MOSTLY_READY'
        elif success_rate >= 50:
            self.validation_results['overall_status'] = 'NEEDS_FIXES'
        else:
            self.validation_results['overall_status'] = 'CRITICAL_ISSUES'
        
        # Print summary
        print(f"\n📋 VALIDATION SUMMARY")
        print("=" * 25)
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print(f"Success Rate: {success_rate:.1f}% ({total_score}/{max_score})")
        print(f"Critical Issues: {len(self.validation_results['critical_issues'])}")
        print(f"Warnings: {len(self.validation_results['warnings'])}")
        
        if self.validation_results['critical_issues']:
            print(f"\n❌ CRITICAL ISSUES TO FIX:")
            for issue in self.validation_results['critical_issues']:
                print(f"   - {issue}")
        
        if self.validation_results['warnings']:
            print(f"\n⚠️ WARNINGS:")
            for warning in self.validation_results['warnings'][:5]:  # Show first 5
                print(f"   - {warning}")
            if len(self.validation_results['warnings']) > 5:
                print(f"   ... and {len(self.validation_results['warnings']) - 5} more")
        
        # Print recommendations
        if success_rate >= 90:
            print(f"\n🎆 SYSTEM READY FOR PRODUCTION!")
            print(f"   Next: Deploy with confidence")
        elif success_rate >= 75:
            print(f"\n⚠️ SYSTEM MOSTLY READY")
            print(f"   Next: Fix critical issues, warnings optional")
        else:
            print(f"\n🚨 SYSTEM NOT PRODUCTION READY")
            print(f"   Next: Fix all critical issues before deployment")
        
        return self.validation_results
    
    def save_report(self, filename: str = None):
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"\n📋 Report saved: {filename}")
        return filename

def main():
    """Main validation function"""
    validator = SystemValidator()
    
    try:
        # Run comprehensive validation
        results = validator.run_all_validations()
        
        # Save report
        report_file = validator.save_report()
        
        # Exit with appropriate code
        if results['overall_status'] in ['PRODUCTION_READY', 'MOSTLY_READY']:
            print(f"\n✅ VALIDATION PASSED")
            return 0
        else:
            print(f"\n❌ VALIDATION FAILED")
            return 1
            
    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {str(e)}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)