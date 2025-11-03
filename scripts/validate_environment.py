#!/usr/bin/env python3
"""
Environment Validation Script for Supreme System V5
Validates Python version, dependencies, imports, and configuration
"""

import sys
import os
import json
import importlib
import time
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional

class EnvironmentValidator:
    """Comprehensive environment validation for Supreme System V5"""
    
    REQUIRED_PYTHON_VERSION = (3, 10)
    REQUIRED_PACKAGES = [
        'numpy',
        'pandas', 
        'loguru',
        'prometheus_client',
        'asyncio',
        'websockets',
        'aiohttp',
        'pydantic',
        'pytest'
    ]
    
    REQUIRED_ENV_KEYS = [
        'OPTIMIZED_MODE',
        'EVENT_DRIVEN_PROCESSING',
        'SINGLE_SYMBOL',
        'PROCESS_INTERVAL_SECONDS',
        'NEWS_INTERVAL_MIN',
        'WHALE_INTERVAL_MIN',
        'MAX_CPU_PERCENT',
        'MAX_RAM_GB'
    ]
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'validation_results': {},
            'errors': [],
            'warnings': [],
            'passed': False
        }
    
    def validate_python_version(self) -> bool:
        """Validate Python version >= 3.10"""
        current_version = sys.version_info[:2]
        required = self.REQUIRED_PYTHON_VERSION
        
        passed = current_version >= required
        
        self.results['validation_results']['python_version'] = {
            'current': f"{current_version[0]}.{current_version[1]}",
            'required': f"{required[0]}.{required[1]}",
            'passed': passed
        }
        
        if not passed:
            self.results['errors'].append(
                f"Python {required[0]}.{required[1]}+ required, found {current_version[0]}.{current_version[1]}"
            )
        
        return passed
    
    def validate_dependencies(self) -> bool:
        """Validate all required packages are installed and importable"""
        dependency_results = {}
        all_passed = True
        
        for package in self.REQUIRED_PACKAGES:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                dependency_results[package] = {
                    'installed': True,
                    'version': version,
                    'passed': True
                }
            except ImportError as e:
                dependency_results[package] = {
                    'installed': False,
                    'error': str(e),
                    'passed': False
                }
                all_passed = False
                self.results['errors'].append(f"Missing package: {package}")
        
        self.results['validation_results']['dependencies'] = dependency_results
        return all_passed
    
    def validate_project_imports(self) -> bool:
        """Validate core Supreme System V5 modules can be imported"""
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / 'python'))
        
        core_modules = [
            'supreme_system_v5.optimized.analyzer',
            'supreme_system_v5.optimized.smart_events',
            'supreme_system_v5.optimized.circular_buffer',
            'supreme_system_v5.strategies',
            'supreme_system_v5.risk',
            'supreme_system_v5.monitoring'
        ]
        
        import_results = {}
        all_passed = True
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                import_results[module_name] = {
                    'importable': True,
                    'passed': True
                }
            except ImportError as e:
                import_results[module_name] = {
                    'importable': False,
                    'error': str(e),
                    'passed': False
                }
                all_passed = False
                self.results['errors'].append(f"Cannot import: {module_name} - {e}")
        
        self.results['validation_results']['project_imports'] = import_results
        return all_passed
    
    def validate_environment_config(self) -> bool:
        """Validate .env configuration completeness"""
        project_root = Path(__file__).parent.parent
        env_files = ['.env', '.env.optimized']
        
        config_results = {}
        any_valid_config = False
        
        for env_file in env_files:
            env_path = project_root / env_file
            
            if not env_path.exists():
                config_results[env_file] = {
                    'exists': False,
                    'keys_present': [],
                    'missing_keys': self.REQUIRED_ENV_KEYS,
                    'passed': False
                }
                continue
            
            # Parse .env file
            env_vars = {}
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            except Exception as e:
                config_results[env_file] = {
                    'exists': True,
                    'error': str(e),
                    'passed': False
                }
                continue
            
            # Check required keys
            present_keys = [key for key in self.REQUIRED_ENV_KEYS if key in env_vars]
            missing_keys = [key for key in self.REQUIRED_ENV_KEYS if key not in env_vars]
            
            passed = len(missing_keys) == 0
            if passed:
                any_valid_config = True
            
            config_results[env_file] = {
                'exists': True,
                'keys_present': present_keys,
                'missing_keys': missing_keys,
                'total_keys': len(env_vars),
                'completeness': len(present_keys) / len(self.REQUIRED_ENV_KEYS),
                'passed': passed
            }
        
        self.results['validation_results']['environment_config'] = config_results
        
        if not any_valid_config:
            self.results['errors'].append(
                "No complete .env configuration found. Required keys: " + ", ".join(self.REQUIRED_ENV_KEYS)
            )
        
        return any_valid_config
    
    def validate_system_resources(self) -> bool:
        """Validate system has adequate resources"""
        try:
            import psutil
            
            # Get system info
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check minimum requirements
            min_ram_gb = 2.0  # Minimum for i3 constraints
            min_disk_gb = 1.0
            
            ram_gb = memory.total / (1024**3)
            disk_gb = disk.free / (1024**3)
            
            resource_results = {
                'cpu_cores': cpu_count,
                'ram_total_gb': round(ram_gb, 2),
                'ram_available_gb': round(memory.available / (1024**3), 2),
                'disk_free_gb': round(disk_gb, 2),
                'requirements_met': {
                    'ram': ram_gb >= min_ram_gb,
                    'disk': disk_gb >= min_disk_gb
                },
                'passed': ram_gb >= min_ram_gb and disk_gb >= min_disk_gb
            }
            
            if ram_gb < min_ram_gb:
                self.results['warnings'].append(f"Low RAM: {ram_gb:.1f}GB < {min_ram_gb}GB recommended")
            
            if disk_gb < min_disk_gb:
                self.results['warnings'].append(f"Low disk space: {disk_gb:.1f}GB < {min_disk_gb}GB recommended")
            
        except ImportError:
            resource_results = {
                'psutil_available': False,
                'passed': True  # Assume OK if can't check
            }
            self.results['warnings'].append("psutil not available - cannot check system resources")
        
        self.results['validation_results']['system_resources'] = resource_results
        return resource_results.get('passed', True)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("🔍 Supreme System V5 - Environment Validation")
        print("=" * 50)
        
        checks = [
            ('Python Version', self.validate_python_version),
            ('Dependencies', self.validate_dependencies),
            ('Project Imports', self.validate_project_imports),
            ('Environment Config', self.validate_environment_config),
            ('System Resources', self.validate_system_resources)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                passed = check_func()
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {status}")
                
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.results['errors'].append(f"{check_name}: {e}")
                all_passed = False
        
        self.results['passed'] = all_passed
        
        # Print summary
        print("\n" + "=" * 50)
        if all_passed:
            print("✅ ALL CHECKS PASSED - Environment ready for Supreme System V5")
        else:
            print("❌ VALIDATION FAILED - Issues found:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        if self.results['warnings']:
            print("\n⚠️ Warnings:")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
        
        return self.results
    
    def save_report(self, output_path: Optional[str] = None) -> str:
        """Save validation report to JSON file"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"validation_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Validation report saved to: {output_path}")
        return output_path


def main():
    """Main validation entry point"""
    validator = EnvironmentValidator()
    results = validator.run_comprehensive_validation()
    
    # Save report
    report_path = validator.save_report()
    
    # Exit with appropriate code
    exit_code = 0 if results['passed'] else 1
    print(f"\n🏁 Validation completed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
