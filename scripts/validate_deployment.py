#!/usr/bin/env python3
"""
Phase 3: Production Deployment Validation Script
Validates all prerequisites and configurations before deployment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{BLUE}{text:^70}{NC}")
    print(f"{BLUE}{'='*70}{NC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✅ {text}{NC}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}❌ {text}{NC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {text}{NC}")


def check_file_exists(file_path: str, description: str) -> Tuple[bool, str]:
    """Check if file exists."""
    if os.path.exists(file_path):
        return True, f"{description} found"
    return False, f"{description} not found"


def check_docker() -> Tuple[bool, str]:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Docker available: {version}"
        return False, "Docker not responding"
    except FileNotFoundError:
        return False, "Docker not installed"
    except Exception as e:
        return False, f"Docker check failed: {e}"


def check_docker_compose() -> Tuple[bool, str]:
    """Check if Docker Compose is available."""
    try:
        # Try docker compose (v2)
        result = subprocess.run(
            ['docker', 'compose', 'version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Docker Compose available: {version}"
        
        # Try docker-compose (v1)
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Docker Compose (v1) available: {version}"
        
        return False, "Docker Compose not available"
    except Exception as e:
        return False, f"Docker Compose check failed: {e}"


def check_kubectl() -> Tuple[bool, str]:
    """Check if kubectl is available."""
    try:
        result = subprocess.run(
            ['kubectl', 'version', '--client'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "kubectl available"
        return False, "kubectl not responding"
    except FileNotFoundError:
        return False, "kubectl not installed (optional for Docker deployment)"
    except Exception as e:
        return False, f"kubectl check failed: {e}"


def check_dockerfile() -> Tuple[bool, str]:
    """Validate Dockerfile."""
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        return False, "Dockerfile not found"
    
    # Read Dockerfile and check for security best practices
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check for non-root user
    if 'USER' in content and 'trader' in content:
        checks.append("Non-root user configured")
    else:
        checks.append("⚠️  Non-root user not found")
    
    # Check for security labels
    if 'LABEL' in content and 'security' in content.lower():
        checks.append("Security labels present")
    
    # Check for health check
    if 'HEALTHCHECK' in content:
        checks.append("Health check configured")
    
    return True, f"Dockerfile validated: {', '.join(checks)}"


def check_kubernetes_manifests() -> Tuple[bool, str]:
    """Check Kubernetes deployment manifests."""
    manifest_path = Path("prod/deployment.yaml")
    if not manifest_path.exists():
        return False, "Kubernetes manifests not found (optional for Docker deployment)"
    
    # Basic validation
    with open(manifest_path, 'r') as f:
        content = f.read()
    
    checks = []
    if 'securityContext' in content:
        checks.append("Security context configured")
    if 'livenessProbe' in content:
        checks.append("Liveness probe configured")
    if 'readinessProbe' in content:
        checks.append("Readiness probe configured")
    if 'resources' in content:
        checks.append("Resource limits configured")
    
    return True, f"K8s manifests validated: {', '.join(checks)}"


def check_environment_variables() -> Tuple[bool, str]:
    """Check required environment variables."""
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]
    
    optional_vars = [
        'GEMINI_API_KEY',
        'POSTGRES_PASSWORD',
        'REDIS_URL'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        return False, f"Missing required env vars: {', '.join(missing_required)}"
    
    warnings = []
    if missing_optional:
        warnings.append(f"Optional vars not set: {', '.join(missing_optional)}")
    
    message = "Environment variables validated"
    if warnings:
        message += f" ({'; '.join(warnings)})"
    
    return True, message


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        
        if free_gb < 5:
            return False, f"Insufficient disk space: {free_gb:.1f}GB (need 5GB+)"
        return True, f"Disk space OK: {free_gb:.1f}GB available"
    except Exception as e:
        return False, f"Disk space check failed: {e}"


def validate_deployment() -> Dict[str, any]:
    """Run all deployment validations."""
    print_header("PHASE 3: PRODUCTION DEPLOYMENT VALIDATION")
    
    results = {
        'checks': [],
        'passed': 0,
        'failed': 0,
        'warnings': 0
    }
    
    # File checks
    file_checks = [
        ('Dockerfile', 'Dockerfile'),
        ('docker-compose.yml', 'Docker Compose config'),
        ('prod/deployment.yaml', 'Kubernetes manifests'),
        ('scripts/deploy_production.sh', 'Deployment script'),
        ('requirements.txt', 'Python requirements'),
    ]
    
    print(f"{BLUE}📁 File Checks:{NC}")
    for file_path, description in file_checks:
        passed, message = check_file_exists(file_path, description)
        results['checks'].append({
            'category': 'files',
            'name': description,
            'passed': passed,
            'message': message
        })
        if passed:
            print_success(message)
            results['passed'] += 1
        else:
            if 'optional' in message.lower() or 'Kubernetes' in message:
                print_warning(message)
                results['warnings'] += 1
            else:
                print_error(message)
                results['failed'] += 1
    
    # Tool checks
    print(f"\n{BLUE}🔧 Tool Checks:{NC}")
    tool_checks = [
        ('docker', check_docker),
        ('docker_compose', check_docker_compose),
        ('kubectl', check_kubectl),
    ]
    
    for name, check_func in tool_checks:
        passed, message = check_func()
        results['checks'].append({
            'category': 'tools',
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            print_success(message)
            results['passed'] += 1
        else:
            if 'optional' in message.lower() or 'kubectl' in message:
                print_warning(message)
                results['warnings'] += 1
            else:
                print_error(message)
                results['failed'] += 1
    
    # Configuration checks
    print(f"\n{BLUE}⚙️  Configuration Checks:{NC}")
    config_checks = [
        ('dockerfile', check_dockerfile),
        ('kubernetes', check_kubernetes_manifests),
        ('environment', check_environment_variables),
        ('disk_space', check_disk_space),
    ]
    
    for name, check_func in config_checks:
        passed, message = check_func()
        results['checks'].append({
            'category': 'configuration',
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            print_success(message)
            results['passed'] += 1
        else:
            if 'optional' in message.lower() or 'Kubernetes' in message:
                print_warning(message)
                results['warnings'] += 1
            else:
                print_error(message)
                results['failed'] += 1
    
    # Summary
    print_header("VALIDATION SUMMARY")
    print(f"{GREEN}✅ Passed: {results['passed']}{NC}")
    print(f"{RED}❌ Failed: {results['failed']}{NC}")
    print(f"{YELLOW}⚠️  Warnings: {results['warnings']}{NC}")
    
    total = results['passed'] + results['failed'] + results['warnings']
    pass_rate = (results['passed'] / total * 100) if total > 0 else 0
    
    print(f"\n{BLUE}Pass Rate: {pass_rate:.1f}%{NC}")
    
    if results['failed'] == 0:
        print_success("\n🎉 All critical checks passed! Ready for deployment.")
        results['ready'] = True
    else:
        print_error(f"\n⚠️  {results['failed']} critical check(s) failed. Please fix before deployment.")
        results['ready'] = False
    
    return results


if __name__ == "__main__":
    try:
        results = validate_deployment()
        
        # Save results to file
        output_file = Path("deployment_validation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{BLUE}Results saved to: {output_file}{NC}")
        
        sys.exit(0 if results['ready'] else 1)
    except KeyboardInterrupt:
        print_error("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
