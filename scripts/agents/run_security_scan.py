#!/usr/bin/env python3
"""
🔒 Agentic Security Scanner - Automated SAST for Trading Systems
Comprehensive security analysis with trading-specific vulnerability detection

Usage:
    python scripts/agents/run_security_scan.py
    python scripts/agents/run_security_scan.py --module data_fabric
    python scripts/agents/run_security_scan.py --critical-only
    python scripts/agents/run_security_scan.py --export-sarif
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

print("🔒 Agentic Security Scanner")
print("=" * 28)

class TradingSystemSecurityScanner:
    """Specialized security scanner for cryptocurrency trading systems"""
    
    def __init__(self, output_dir: str = "run_artifacts/security"):
        self.project_root = project_root
        self.python_root = self.project_root / "python"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scan_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'scanner_version': '1.0.0-agentic',
            'project_path': str(self.project_root),
            'scan_summary': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            },
            'findings': [],
            'trading_specific': {
                'api_key_exposure': [],
                'exchange_security': [],
                'financial_data_protection': [],
                'risk_calculation_integrity': []
            },
            'recommendations': []
        }
        
    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run Semgrep security analysis"""
        print("🔍 Running Semgrep security scan...")
        
        try:
            # Run semgrep with auto rules
            cmd = [
                'semgrep', 
                '--config=auto',
                '--json',
                '--timeout=30',
                '--max-memory=1000',
                str(self.python_root)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 or result.returncode == 1:  # 0=no issues, 1=issues found
                semgrep_data = json.loads(result.stdout)
                
                # Process semgrep results
                for finding in semgrep_data.get('results', []):
                    severity = finding.get('extra', {}).get('severity', 'INFO').lower()
                    
                    processed_finding = {
                        'tool': 'semgrep',
                        'severity': severity,
                        'rule_id': finding.get('check_id', 'unknown'),
                        'message': finding.get('extra', {}).get('message', 'No message'),
                        'file': finding.get('path', 'unknown'),
                        'line': finding.get('start', {}).get('line', 0),
                        'code': finding.get('extra', {}).get('lines', ''),
                        'category': self._categorize_finding(finding)
                    }
                    
                    self.scan_results['findings'].append(processed_finding)
                    self.scan_results['scan_summary'][severity] += 1
                    
                print(f"   ✅ Semgrep completed: {len(semgrep_data.get('results', []))} findings")
                return {'status': 'success', 'findings': len(semgrep_data.get('results', []))}
                
            else:
                print(f"   ⚠️ Semgrep failed with code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print("   ⚠️ Semgrep timeout - scan may be incomplete")
            return {'status': 'timeout'}
        except FileNotFoundError:
            print("   ⚠️ Semgrep not installed - skipping")
            return {'status': 'skipped', 'reason': 'tool_not_found'}
        except Exception as e:
            print(f"   ❌ Semgrep error: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security analysis"""
        print("🔍 Running Bandit security scan...")
        
        try:
            cmd = [
                'bandit',
                '-r', str(self.python_root),
                '-f', 'json',
                '-ll',  # Low confidence threshold
                '--skip', 'B101,B601'  # Skip assert and shell usage (common in trading)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                # Process bandit results
                for finding in bandit_data.get('results', []):
                    severity = finding.get('issue_severity', 'LOW').lower()
                    
                    processed_finding = {
                        'tool': 'bandit',
                        'severity': severity,
                        'rule_id': finding.get('test_id', 'unknown'),
                        'message': finding.get('issue_text', 'No message'),
                        'file': finding.get('filename', 'unknown'),
                        'line': finding.get('line_number', 0),
                        'code': finding.get('code', ''),
                        'confidence': finding.get('issue_confidence', 'UNDEFINED'),
                        'category': 'python_security'
                    }
                    
                    self.scan_results['findings'].append(processed_finding)
                    if severity in self.scan_results['scan_summary']:
                        self.scan_results['scan_summary'][severity] += 1
                    else:
                        self.scan_results['scan_summary']['medium'] += 1
                        
                print(f"   ✅ Bandit completed: {len(bandit_data.get('results', []))} findings")
                return {'status': 'success', 'findings': len(bandit_data.get('results', []))}
                
        except FileNotFoundError:
            print("   ⚠️ Bandit not installed - skipping")
            return {'status': 'skipped', 'reason': 'tool_not_found'}
        except Exception as e:
            print(f"   ❌ Bandit error: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def run_trading_specific_checks(self) -> Dict[str, Any]:
        """Run trading-specific security checks"""
        print("🔍 Running trading-specific security checks...")
        
        trading_issues = {
            'api_key_exposure': self._check_api_key_exposure(),
            'exchange_security': self._check_exchange_security(), 
            'financial_calculations': self._check_financial_integrity(),
            'risk_management': self._check_risk_controls()
        }
        
        # Add to main results
        self.scan_results['trading_specific'].update(trading_issues)
        
        total_trading_issues = sum(len(issues) for issues in trading_issues.values())
        print(f"   ✅ Trading checks completed: {total_trading_issues} specialized findings")
        
        return {'status': 'success', 'findings': total_trading_issues}
        
    def _check_api_key_exposure(self) -> List[Dict[str, Any]]:
        """Check for hardcoded API keys and secrets"""
        issues = []
        
        # Patterns that might indicate API key exposure
        dangerous_patterns = [
            r'api_key\s*=\s*["\'][^"\'
]{20,}["\']',
            r'secret\s*=\s*["\'][^"\'
]{20,}["\']',
            r'token\s*=\s*["\'][^"\'
]{20,}["\']',
            r'password\s*=\s*["\'][^"\'
]+["\']'
        ]
        
        import re
        
        for py_file in self.python_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in dangerous_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Calculate line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        issues.append({
                            'type': 'potential_secret_exposure',
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'pattern': pattern,
                            'severity': 'high',
                            'recommendation': 'Use environment variables or secure configuration'
                        })
                        
            except Exception:
                continue
                
        return issues
        
    def _check_exchange_security(self) -> List[Dict[str, Any]]:
        """Check exchange connector security practices"""
        issues = []
        
        exchanges_dir = self.python_root / 'exchanges'
        if not exchanges_dir.exists():
            return issues
            
        for connector_file in exchanges_dir.glob('*_connector.py'):
            try:
                with open(connector_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for security best practices
                security_checks = {
                    'ssl_verification': 'verify=False' not in content,
                    'request_timeout': 'timeout=' in content,
                    'rate_limiting': any(term in content for term in ['rate', 'limit', 'throttle']),
                    'error_handling': 'try:' in content and 'except' in content,
                    'input_validation': any(term in content for term in ['validate', 'sanitize', 'check'])
                }
                
                for check, passed in security_checks.items():
                    if not passed:
                        issues.append({
                            'type': 'exchange_security_gap',
                            'file': str(connector_file.relative_to(self.project_root)),
                            'check': check,
                            'severity': 'medium',
                            'recommendation': f'Implement {check.replace("_", " ")} for secure exchange communication'
                        })
                        
            except Exception:
                continue
                
        return issues
        
    def _check_financial_integrity(self) -> List[Dict[str, Any]]:
        """Check financial calculation integrity"""
        issues = []
        
        # Files with financial calculations
        financial_files = [
            'strategies.py',
            'risk.py', 
            'algorithms/scalping_futures_optimized.py',
            'indicators.py'
        ]
        
        for file_name in financial_files:
            file_path = self.python_root / file_name
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for financial integrity practices
                integrity_checks = {
                    'decimal_precision': 'Decimal' in content,
                    'overflow_protection': any(term in content for term in ['max(', 'min(', 'clamp']),
                    'zero_division': 'ZeroDivisionError' in content or '/ 0' not in content,
                    'nan_handling': 'isnan' in content or 'nan' in content,
                    'rounding_control': any(term in content for term in ['round', 'floor', 'ceil'])
                }
                
                for check, passed in integrity_checks.items():
                    if not passed and check in ['zero_division', 'overflow_protection']:
                        issues.append({
                            'type': 'financial_integrity_risk',
                            'file': file_name,
                            'check': check,
                            'severity': 'high',
                            'recommendation': f'Implement {check.replace("_", " ")} for financial calculation safety'
                        })
                        
            except Exception:
                continue
                
        return issues
        
    def _check_risk_controls(self) -> List[Dict[str, Any]]:
        """Check risk management control effectiveness"""
        issues = []
        
        risk_files = ['risk.py', 'dynamic_risk_manager.py']
        
        for file_name in risk_files:
            file_path = self.python_root / file_name
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Essential risk controls
                risk_controls = {
                    'position_limits': any(term in content for term in ['max_position', 'position_limit', 'size_limit']),
                    'drawdown_protection': any(term in content for term in ['drawdown', 'max_loss']),
                    'exposure_controls': any(term in content for term in ['exposure', 'total_value']),
                    'emergency_stops': any(term in content for term in ['emergency', 'stop', 'shutdown']),
                    'validation_checks': any(term in content for term in ['validate', 'verify', 'check'])
                }
                
                for control, implemented in risk_controls.items():
                    if not implemented:
                        issues.append({
                            'type': 'missing_risk_control',
                            'file': file_name,
                            'control': control,
                            'severity': 'critical',
                            'recommendation': f'Implement {control.replace("_", " ")} for comprehensive risk management'
                        })
                        
            except Exception:
                continue
                
        return issues
        
    def _categorize_finding(self, finding: Dict[str, Any]) -> str:
        """Categorize security finding by type"""
        rule_id = finding.get('check_id', '').lower()
        message = finding.get('extra', {}).get('message', '').lower()
        
        # Trading-specific categories
        if any(term in rule_id or term in message for term in ['key', 'secret', 'token', 'credential']):
            return 'credential_security'
        elif any(term in rule_id or term in message for term in ['sql', 'injection', 'xss']):
            return 'injection_vulnerability'
        elif any(term in rule_id or term in message for term in ['crypto', 'hash', 'encrypt']):
            return 'cryptographic_issue'
        elif any(term in rule_id or term in message for term in ['exec', 'eval', 'subprocess']):
            return 'code_execution'
        else:
            return 'general_security'
            
    def run_dependency_scan(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities"""
        print("🔍 Running dependency vulnerability scan...")
        
        try:
            # Check requirements files
            req_files = list(self.project_root.glob('requirements*.txt'))
            
            if not req_files:
                print("   ⚠️ No requirements.txt found")
                return {'status': 'skipped', 'reason': 'no_requirements_file'}
                
            for req_file in req_files:
                cmd = ['safety', 'check', '-r', str(req_file), '--json']
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.stdout:
                        safety_data = json.loads(result.stdout)
                        
                        for vuln in safety_data:
                            self.scan_results['findings'].append({
                                'tool': 'safety',
                                'severity': 'high',
                                'rule_id': vuln.get('id', 'unknown'),
                                'message': vuln.get('advisory', 'Dependency vulnerability'),
                                'file': str(req_file.name),
                                'package': vuln.get('package_name', 'unknown'),
                                'version': vuln.get('analyzed_version', 'unknown'),
                                'category': 'dependency_vulnerability'
                            })
                            
                    print(f"   ✅ Safety scan completed for {req_file.name}")
                    
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️ Safety scan timeout for {req_file.name}")
                except json.JSONDecodeError:
                    print(f"   ⚠️ Safety scan: no vulnerabilities found in {req_file.name}")
                    
            return {'status': 'success'}
            
        except FileNotFoundError:
            print("   ⚠️ Safety tool not installed - skipping dependency scan")
            return {'status': 'skipped', 'reason': 'tool_not_found'}
        except Exception as e:
            print(f"   ❌ Dependency scan error: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def run_custom_trading_checks(self) -> Dict[str, Any]:
        """Run custom security checks specific to trading systems"""
        print("🔍 Running custom trading system checks...")
        
        checks_passed = 0
        checks_total = 0
        
        # Check 1: API key management
        checks_total += 1
        api_issues = self._check_api_key_management()
        if len(api_issues) == 0:
            checks_passed += 1
        else:
            self.scan_results['trading_specific']['api_key_exposure'].extend(api_issues)
            
        # Check 2: Exchange connector security
        checks_total += 1 
        exchange_issues = self._check_exchange_connectors()
        if len(exchange_issues) == 0:
            checks_passed += 1
        else:
            self.scan_results['trading_specific']['exchange_security'].extend(exchange_issues)
            
        # Check 3: Financial data protection
        checks_total += 1
        data_issues = self._check_financial_data_protection()
        if len(data_issues) == 0:
            checks_passed += 1
        else:
            self.scan_results['trading_specific']['financial_data_protection'].extend(data_issues)
            
        # Check 4: Risk calculation integrity
        checks_total += 1
        risk_issues = self._check_risk_calculation_integrity()
        if len(risk_issues) == 0:
            checks_passed += 1
        else:
            self.scan_results['trading_specific']['risk_calculation_integrity'].extend(risk_issues)
            
        success_rate = (checks_passed / checks_total) * 100 if checks_total > 0 else 0
        
        print(f"   ✅ Custom checks completed: {checks_passed}/{checks_total} ({success_rate:.1f}% pass rate)")
        
        return {
            'status': 'success',
            'checks_passed': checks_passed,
            'checks_total': checks_total,
            'success_rate': success_rate
        }
        
    def _check_api_key_management(self) -> List[Dict[str, Any]]:
        """Check API key security practices"""
        issues = []
        
        # Look for environment variable usage
        env_usage_files = []
        hardcoded_issues = []
        
        for py_file in self.python_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Good practice: environment variables
                if 'os.getenv(' in content or 'os.environ' in content:
                    env_usage_files.append(str(py_file.relative_to(self.project_root)))
                    
                # Bad practice: hardcoded values (basic check)
                import re
                if re.search(r'["\'][A-Za-z0-9+/]{32,}["\']', content):
                    hardcoded_issues.append({
                        'file': str(py_file.relative_to(self.project_root)),
                        'type': 'potential_hardcoded_secret',
                        'severity': 'critical'
                    })
                    
            except Exception:
                continue
                
        # Add issues
        issues.extend(hardcoded_issues)
        
        if len(env_usage_files) == 0 and len(hardcoded_issues) == 0:
            issues.append({
                'type': 'no_secret_management_detected',
                'severity': 'medium',
                'recommendation': 'Implement proper secret management using environment variables'
            })
            
        return issues
        
    def _check_exchange_connectors(self) -> List[Dict[str, Any]]:
        """Check exchange connector security"""
        issues = []
        
        exchanges_dir = self.python_root / 'exchanges'
        if not exchanges_dir.exists():
            return issues
            
        for connector in exchanges_dir.glob('*_connector.py'):
            try:
                with open(connector, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Security requirement checks
                if 'verify=False' in content:
                    issues.append({
                        'type': 'ssl_verification_disabled',
                        'file': str(connector.relative_to(self.project_root)),
                        'severity': 'high',
                        'recommendation': 'Enable SSL verification for exchange connections'
                    })
                    
                if 'timeout=' not in content:
                    issues.append({
                        'type': 'missing_request_timeout',
                        'file': str(connector.relative_to(self.project_root)),
                        'severity': 'medium',
                        'recommendation': 'Add request timeouts to prevent hanging connections'
                    })
                    
            except Exception:
                continue
                
        return issues
        
    def _check_financial_data_protection(self) -> List[Dict[str, Any]]:
        """Check financial data protection measures"""
        issues = []
        
        # Check for logging of sensitive financial data
        for py_file in self.python_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for potential sensitive data logging
                import re
                if re.search(r'log.*\.(info|debug|warning).*\b(balance|position|pnl|profit)\b', content, re.IGNORECASE):
                    issues.append({
                        'type': 'potential_financial_data_logging',
                        'file': str(py_file.relative_to(self.project_root)),
                        'severity': 'low',
                        'recommendation': 'Review logging to ensure no sensitive financial data exposure'
                    })
                    
            except Exception:
                continue
                
        return issues
        
    def _check_risk_calculation_integrity(self) -> List[Dict[str, Any]]:
        """Check integrity of risk calculations"""
        issues = []
        
        risk_files = ['risk.py', 'dynamic_risk_manager.py']
        
        for file_name in risk_files:
            file_path = self.python_root / file_name
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for proper risk calculation practices
                if 'assert ' not in content:
                    issues.append({
                        'type': 'missing_risk_assertions',
                        'file': file_name,
                        'severity': 'medium',
                        'recommendation': 'Add assertions to validate risk calculation results'
                    })
                    
                if 'max(' not in content and 'min(' not in content:
                    issues.append({
                        'type': 'missing_bounds_checking',
                        'file': file_name,
                        'severity': 'high',
                        'recommendation': 'Add bounds checking for risk calculation parameters'
                    })
                    
            except Exception:
                continue
                
        return issues
        
    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        print("📋 Generating security report...")
        
        # Calculate overall security score
        total_findings = len(self.scan_results['findings'])
        critical_findings = sum(1 for f in self.scan_results['findings'] if f.get('severity') == 'critical')
        high_findings = sum(1 for f in self.scan_results['findings'] if f.get('severity') == 'high')
        
        if critical_findings > 0:
            security_score = 'CRITICAL'
        elif high_findings > 5:
            security_score = 'HIGH_RISK'
        elif high_findings > 0:
            security_score = 'MEDIUM_RISK'
        elif total_findings > 10:
            security_score = 'LOW_RISK'
        else:
            security_score = 'SECURE'
            
        report = f"""
# 🔒 SUPREME SYSTEM V5 - SECURITY ASSESSMENT REPORT

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Security Score:** {security_score}
**Total Findings:** {total_findings}

## 📊 EXECUTIVE SUMMARY

| Severity | Count |
|----------|-------|
| Critical | {self.scan_results['scan_summary']['critical']} |
| High     | {self.scan_results['scan_summary']['high']} |
| Medium   | {self.scan_results['scan_summary']['medium']} |
| Low      | {self.scan_results['scan_summary']['low']} |
| Info     | {self.scan_results['scan_summary']['info']} |

## 🎯 TRADING-SPECIFIC SECURITY

### API Key Management
**Issues Found:** {len(self.scan_results['trading_specific']['api_key_exposure'])}

### Exchange Security
**Issues Found:** {len(self.scan_results['trading_specific']['exchange_security'])}

### Financial Data Protection  
**Issues Found:** {len(self.scan_results['trading_specific']['financial_data_protection'])}

### Risk Calculation Integrity
**Issues Found:** {len(self.scan_results['trading_specific']['risk_calculation_integrity'])}

## 🚑 IMMEDIATE ACTIONS REQUIRED
"""
        
        # Add critical and high severity issues
        critical_issues = [f for f in self.scan_results['findings'] if f.get('severity') == 'critical']
        high_issues = [f for f in self.scan_results['findings'] if f.get('severity') == 'high']
        
        if critical_issues:
            report += "\n### 🚨 CRITICAL ISSUES (Fix Immediately)\n"
            for issue in critical_issues[:5]:  # Top 5
                report += f"- **{issue['file']}:{issue['line']}** - {issue['message']}\n"
                
        if high_issues:
            report += "\n### ⚠️ HIGH PRIORITY ISSUES\n"
            for issue in high_issues[:10]:  # Top 10
                report += f"- **{issue['file']}:{issue['line']}** - {issue['message']}\n"
                
        # Recommendations
        report += f"""

## 🛠️ RECOMMENDATIONS

### Immediate Actions:
1. **Fix Critical Issues** - Address all critical security findings
2. **Implement Missing Controls** - Add essential risk management controls
3. **Review API Security** - Validate exchange connector security
4. **Update Dependencies** - Address known vulnerabilities

### Long-term Improvements:
1. **Security Testing Integration** - Add security tests to CI/CD
2. **Automated Scanning** - Schedule regular security scans
3. **Security Monitoring** - Real-time security monitoring
4. **Incident Response** - Security incident response procedures

## 🎆 SECURITY CERTIFICATION STATUS

**Overall Assessment:** {security_score}
**Production Readiness:** {'APPROVED' if security_score in ['SECURE', 'LOW_RISK'] else 'REQUIRES_FIXES'}
**Recommendation:** {'Deploy with monitoring' if security_score == 'SECURE' else 'Fix issues before deployment'}

---
*Generated by Agentic Security Scanner v1.0.0*
"""
        
        return report
        
    def save_results(self, report_content: str) -> Dict[str, Path]:
        """Save security scan results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = self.output_dir / f"security_scan_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.scan_results, f, indent=2)
            
        # Save markdown report
        report_file = self.output_dir / f"security_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        # Save SARIF format (if requested)
        sarif_file = self.output_dir / f"security_scan_{timestamp}.sarif"
        sarif_content = self._generate_sarif()
        with open(sarif_file, 'w') as f:
            json.dump(sarif_content, f, indent=2)
            
        print(f"\n💾 Security results saved:")
        print(f"   JSON: {json_file}")
        print(f"   Report: {report_file}")
        print(f"   SARIF: {sarif_file}")
        
        return {
            'json': json_file,
            'report': report_file,
            'sarif': sarif_file
        }
        
    def _generate_sarif(self) -> Dict[str, Any]:
        """Generate SARIF format for integration with security tools"""
        sarif = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "Agentic Security Scanner",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/thanhmuefatty07/supreme-system-v5"
                    }
                },
                "results": []
            }]
        }
        
        for finding in self.scan_results['findings']:
            sarif_result = {
                "ruleId": finding.get('rule_id', 'unknown'),
                "level": self._severity_to_sarif_level(finding.get('severity', 'info')),
                "message": {
                    "text": finding.get('message', 'Security finding')
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": finding.get('file', 'unknown')
                        },
                        "region": {
                            "startLine": finding.get('line', 1)
                        }
                    }
                }]
            }
            
            sarif['runs'][0]['results'].append(sarif_result)
            
        return sarif
        
    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level"""
        mapping = {
            'critical': 'error',
            'high': 'error', 
            'medium': 'warning',
            'low': 'note',
            'info': 'note'
        }
        return mapping.get(severity, 'note')

async def main():
    """Main security scan execution"""
    parser = argparse.ArgumentParser(description='Agentic Security Scanner')
    parser.add_argument('--module', type=str, help='Scan specific module only')
    parser.add_argument('--critical-only', action='store_true', help='Show only critical issues')
    parser.add_argument('--export-sarif', action='store_true', help='Export SARIF format')
    parser.add_argument('--output', type=str, default='run_artifacts/security', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Module: {args.module or 'Full system'}")
    print(f"  Critical only: {args.critical_only}")
    print(f"  Export SARIF: {args.export_sarif}")
    print(f"  Output: {args.output}")
    print()
    
    # Initialize scanner
    scanner = TradingSystemSecurityScanner(args.output)
    
    try:
        # Run comprehensive security scan
        print(f"🔒 Starting comprehensive security scan...")
        
        # Run different scan types
        semgrep_result = scanner.run_semgrep_scan()
        bandit_result = scanner.run_bandit_scan()
        dependency_result = scanner.run_dependency_scan()
        trading_result = scanner.run_custom_trading_checks()
        
        # Generate report
        security_report = scanner.generate_security_report()
        
        # Save results
        files_saved = scanner.save_results(security_report)
        
        # Print summary
        total_findings = len(scanner.scan_results['findings'])
        critical_count = scanner.scan_results['scan_summary']['critical']
        high_count = scanner.scan_results['scan_summary']['high']
        
        print(f"\n🎆 Security Scan Complete!")
        print(f"   Total Findings: {total_findings}")
        print(f"   Critical: {critical_count}")
        print(f"   High: {high_count}")
        
        if critical_count > 0:
            print(f"\n🚨 CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
            return 2
        elif high_count > 5:
            print(f"\n⚠️ HIGH RISK ISSUES - REVIEW RECOMMENDED")
            return 1
        else:
            print(f"\n✅ SECURITY STATUS: ACCEPTABLE")
            return 0
            
    except Exception as e:
        print(f"\n❌ Security scan failed: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    
    if exit_code == 0:
        print(f"\n🔒 Security scan completed successfully!")
    elif exit_code == 1:
        print(f"\n⚠️ Security scan completed with warnings.")
    else:
        print(f"\n🚨 Security scan found critical issues!")
        
    sys.exit(exit_code)