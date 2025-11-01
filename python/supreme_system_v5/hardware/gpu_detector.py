"""
Supreme System V5 - GPU Detection & Acceleration Manager
Automatically detects and configures GPU acceleration for i3-4GB systems
"""

import subprocess
import platform
import os
import sys
from typing import List, Dict, Optional, Tuple
import importlib.util

class GPUAccelerationManager:
    """
    Comprehensive GPU detection and acceleration setup for trading systems
    Supports NVIDIA CUDA, AMD ROCm, Intel OpenCL, and CPU fallback
    """

    def __init__(self):
        self.detected_gpus = []
        self.acceleration_available = False
        self.gpu_type = None
        self.cuda_version = None
        self.rocm_version = None
        self.memory_mb = 0

        # Acceleration capabilities
        self.cuda_available = False
        self.rocm_available = False
        self.opencl_available = False
        self.cpu_acceleration = True  # Always available

    def detect_hardware(self) -> Dict:
        """Comprehensive hardware detection"""
        hardware_info = {
            'cpu_info': self._get_cpu_info(),
            'memory_info': self._get_memory_info(),
            'gpu_info': self._detect_gpus(),
            'acceleration_options': [],
            'recommendations': []
        }

        # Determine best acceleration strategy
        if hardware_info['gpu_info']:
            hardware_info['acceleration_options'] = self._get_gpu_acceleration()
            hardware_info['recommendations'] = self._generate_gpu_recommendations()
        else:
            hardware_info['acceleration_options'] = self._get_cpu_acceleration()
            hardware_info['recommendations'] = self._generate_cpu_recommendations()

        return hardware_info

    def _detect_gpus(self) -> List[Dict]:
        """Detect all available GPUs"""
        gpus = []

        # NVIDIA GPU Detection
        try:
            nvidia_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )

            if nvidia_result.returncode == 0:
                lines = nvidia_result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_info = {
                                'type': 'nvidia',
                                'index': i,
                                'name': parts[0].strip(),
                                'memory_total_mb': int(parts[1]),
                                'memory_free_mb': int(parts[2]),
                                'driver_version': parts[3].strip(),
                                'acceleration': 'cuda'
                            }
                            gpus.append(gpu_info)

                if gpus:
                    self.cuda_available = True
                    self.gpu_type = 'nvidia'

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # AMD GPU Detection
        try:
            rocm_result = subprocess.run(
                ['rocm-smi', '--showproductname', '--showmeminfo', 'vram'],
                capture_output=True, text=True, timeout=10
            )

            if rocm_result.returncode == 0 and 'AMD' in rocm_result.stdout:
                # Parse ROCm output
                lines = rocm_result.stdout.strip().split('\n')
                gpu_name = "Unknown AMD GPU"

                for line in lines:
                    if 'GPU' in line and 'AMD' in line:
                        gpu_name = line.split(':')[-1].strip()
                        break

                gpu_info = {
                    'type': 'amd',
                    'index': 0,
                    'name': gpu_name,
                    'memory_total_mb': 4096,  # Estimate for typical AMD GPUs
                    'memory_free_mb': 3072,   # Estimate
                    'driver_version': 'ROCm detected',
                    'acceleration': 'rocm'
                }
                gpus.append(gpu_info)
                self.rocm_available = True
                self.gpu_type = 'amd'

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Intel GPU Detection
        try:
            # Check for Intel integrated graphics
            lspci_result = subprocess.run(['lspci'], capture_output=True, text=True)
            intel_found = False

            for line in lspci_result.stdout.split('\n'):
                if 'VGA' in line and 'Intel' in line:
                    gpu_name = line.split(': ')[-1] if ': ' in line else 'Intel Integrated Graphics'
                    gpu_info = {
                        'type': 'intel',
                        'index': 0,
                        'name': gpu_name,
                        'memory_total_mb': 512,   # Shared system memory
                        'memory_free_mb': 256,    # Estimate
                        'driver_version': 'Intel Graphics',
                        'acceleration': 'opencl'
                    }
                    gpus.append(gpu_info)
                    self.opencl_available = True
                    self.gpu_type = 'intel'
                    intel_found = True
                    break

            if not intel_found:
                # Check for Intel Arc discrete GPUs
                if 'Intel' in lspci_result.stdout and ('Arc' in lspci_result.stdout or 'DG2' in lspci_result.stdout):
                    gpu_info = {
                        'type': 'intel',
                        'index': 0,
                        'name': 'Intel Arc GPU',
                        'memory_total_mb': 4096,  # Typical for Arc GPUs
                        'memory_free_mb': 3072,
                        'driver_version': 'Intel Graphics',
                        'acceleration': 'opencl'
                    }
                    gpus.append(gpu_info)
                    self.opencl_available = True
                    self.gpu_type = 'intel'

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        self.detected_gpus = gpus
        return gpus

    def _get_gpu_acceleration(self) -> List[str]:
        """Get GPU acceleration setup commands"""
        commands = []

        for gpu in self.detected_gpus:
            if gpu['type'] == 'nvidia':
                commands.extend([
                    'export CUDA_VISIBLE_DEVICES=0',
                    'export NUMBA_ENABLE_CUDASIM=0',
                    'pip install cupy-cuda11x',
                    'pip install numba[cuda]',
                    'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                    'pip install tensorflow[and-cuda]'
                ])
                self.acceleration_available = True

            elif gpu['type'] == 'amd':
                commands.extend([
                    'export HSA_OVERRIDE_GFX_VERSION=10.3.0',
                    'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2',
                    'pip install tensorflow-rocm'
                ])
                self.acceleration_available = True

            elif gpu['type'] == 'intel':
                commands.extend([
                    'pip install intel-extension-for-pytorch',
                    'pip install onednn-cpu',
                    'export ONEAPI_DEVICE_SELECTOR=opencl:gpu'
                ])
                self.acceleration_available = True

        return commands

    def _get_cpu_acceleration(self) -> List[str]:
        """Get CPU-only acceleration setup commands"""
        commands = [
            'pip install numba',
            'pip install mkl',
            'pip install tbb'
        ]

        # CPU-specific optimizations
        cpu_info = self._get_cpu_info()
        flags = cpu_info.get('flags', [])

        # AVX-512 (Intel Xeon/High-end i9)
        if 'avx512' in ' '.join(flags).lower():
            commands.append('export NUMBA_CPU_FEATURES="+avx512,+fma"')
        # AVX2 (Most modern Intel/AMD)
        elif 'avx2' in ' '.join(flags).lower():
            commands.append('export NUMBA_CPU_FEATURES="+avx2,+fma"')
        # AVX (Older systems)
        elif 'avx' in ' '.join(flags).lower():
            commands.append('export NUMBA_CPU_FEATURES="+avx,+sse4.1"')
        else:
            commands.append('export NUMBA_CPU_FEATURES="+sse4.1,+sse3"')

        return commands

    def _generate_gpu_recommendations(self) -> List[str]:
        """Generate GPU-specific recommendations"""
        recommendations = []

        if self.cuda_available:
            recommendations.extend([
                "NVIDIA GPU detected - CUDA acceleration available",
                "Use cupy for array operations, numba for JIT compilation",
                "Consider torch.cuda for deep learning workloads"
            ])

        elif self.rocm_available:
            recommendations.extend([
                "AMD GPU detected - ROCm acceleration available",
                "Use PyTorch ROCm builds for ML workloads",
                "Consider tensorflow-rocm for neural networks"
            ])

        elif self.opencl_available:
            recommendations.extend([
                "Intel GPU detected - OpenCL acceleration available",
                "Use Intel Extension for PyTorch for optimal performance",
                "Consider oneDNN for deep learning primitives"
            ])

        # Memory considerations
        total_gpu_memory = sum(gpu.get('memory_total_mb', 0) for gpu in self.detected_gpus)
        if total_gpu_memory < 4096:
            recommendations.append("Limited GPU memory - focus on lightweight models")
        elif total_gpu_memory < 8192:
            recommendations.append("Moderate GPU memory - suitable for medium ML models")
        else:
            recommendations.append("Good GPU memory - can handle complex ML workloads")

        return recommendations

    def _generate_cpu_recommendations(self) -> List[str]:
        """Generate CPU-only recommendations"""
        cpu_info = self._get_cpu_info()
        recommendations = []

        core_count = cpu_info.get('cores', 4)
        if core_count >= 8:
            recommendations.append(f"Good CPU ({core_count} cores) - use parallel processing")
        elif core_count >= 4:
            recommendations.append(f"Decent CPU ({core_count} cores) - optimize for parallelism")
        else:
            recommendations.append(f"Limited CPU ({core_count} cores) - focus on efficiency")

        # Architecture-specific recommendations
        model_name = cpu_info.get('model_name', '').lower()
        if 'intel' in model_name:
            if 'i3' in model_name:
                recommendations.append("Intel i3 detected - optimize for single-thread performance")
            elif 'i5' in model_name or 'i7' in model_name:
                recommendations.append("Intel i5/i7 detected - good for parallel workloads")
            elif 'xeon' in model_name:
                recommendations.append("Intel Xeon detected - excellent for compute-intensive tasks")

        elif 'amd' in model_name:
            recommendations.append("AMD CPU detected - Ryzen series good for parallel processing")

        recommendations.extend([
            "Use numba JIT compilation for performance-critical functions",
            "Consider multiprocessing for CPU-bound tasks",
            "Optimize memory access patterns for better cache utilization"
        ])

        return recommendations

    def _get_cpu_info(self) -> Dict:
        """Get detailed CPU information"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()

            lines = cpu_info.split('\n')
            info = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'model name':
                        info['model_name'] = value
                    elif key == 'cpu cores':
                        info['cores'] = int(value)
                    elif key == 'siblings':
                        info['threads'] = int(value)
                    elif key == 'flags':
                        info['flags'] = value.split()

            # Get CPU frequency info
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('cpu MHz'):
                            mhz = float(line.split(':')[1].strip())
                            info['frequency_mhz'] = mhz
                            break
            except:
                pass

            return info

        except Exception:
            # Fallback for Windows or other systems
            try:
                import platform
                return {
                    'model_name': platform.processor(),
                    'cores': os.cpu_count() or 4,
                    'threads': os.cpu_count() or 4,
                    'flags': []
                }
            except:
                return {
                    'model_name': 'Unknown CPU',
                    'cores': 4,
                    'threads': 4,
                    'flags': []
                }

    def _get_memory_info(self) -> Dict:
        """Get memory information"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()

            total_match = None
            available_match = None

            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total_match = line
                elif line.startswith('MemAvailable:'):
                    available_match = line

            total_kb = int(total_match.split()[1]) if total_match else 4194304  # 4GB default
            available_kb = int(available_match.split()[1]) if available_match else 2097152  # 2GB default

            return {
                'total_mb': total_kb // 1024,
                'available_mb': available_kb // 1024,
                'used_mb': (total_kb - available_kb) // 1024
            }

        except Exception:
            # Fallback
            return {
                'total_mb': 4096,
                'available_mb': 2048,
                'used_mb': 2048
            }

    def setup_acceleration(self) -> bool:
        """Setup optimal acceleration for detected hardware"""
        try:
            hardware = self.detect_hardware()

            print(f"🔍 Hardware Detection Results:")
            print(f"   CPU: {hardware['cpu_info'].get('model_name', 'Unknown')}")
            print(f"   Cores: {hardware['cpu_info'].get('cores', 'Unknown')}")
            print(f"   Memory: {hardware['memory_info']['total_mb']}MB total")
            print(f"   GPUs: {len(hardware['gpu_info'])}")

            for gpu in hardware['gpu_info']:
                print(f"     • {gpu['name']} ({gpu['type'].upper()}) - {gpu.get('memory_total_mb', 'Unknown')}MB")

            print(f"\n💡 Recommendations:")
            for rec in hardware['recommendations']:
                print(f"   • {rec}")

            # Execute acceleration setup
            success_count = 0
            for cmd in hardware['acceleration_options'][:5]:  # Limit to first 5 commands
                if cmd.startswith('pip install'):
                    print(f"🔧 Installing {cmd}")
                    result = subprocess.run(cmd.split(), capture_output=True, text=True)
                    if result.returncode == 0:
                        success_count += 1
                    else:
                        print(f"⚠️ Installation warning: {result.stderr[:100]}...")
                elif cmd.startswith('export'):
                    # Set environment variable
                    var_name, var_value = cmd.replace('export ', '').split('=', 1)
                    os.environ[var_name] = var_value
                    print(f"✅ Set {var_name}={var_value}")
                    success_count += 1

            if success_count > 0:
                print(f"✅ Acceleration setup completed ({success_count} components)")
                self.acceleration_available = True
                return True
            else:
                print("⚠️ No acceleration components installed")
                return False

        except Exception as e:
            print(f"❌ Acceleration setup failed: {e}")
            return False

    def get_acceleration_info(self) -> Dict:
        """Get current acceleration capabilities"""
        return {
            'gpu_available': len(self.detected_gpus) > 0,
            'gpu_type': self.gpu_type,
            'cuda_available': self.cuda_available,
            'rocm_available': self.rocm_available,
            'opencl_available': self.opencl_available,
            'cpu_acceleration': self.cpu_acceleration,
            'acceleration_enabled': self.acceleration_available,
            'detected_gpus': self.detected_gpus
        }

    def benchmark_acceleration(self) -> Dict:
        """Benchmark acceleration performance"""
        results = {
            'cpu_only': {},
            'gpu_accelerated': {},
            'speedup': {}
        }

        try:
            # CPU-only benchmark
            print("🔬 Benchmarking CPU-only performance...")
            cpu_time = self._benchmark_array_operations(use_gpu=False)
            results['cpu_only']['array_ops_ms'] = cpu_time

            if self.acceleration_available:
                print("🔬 Benchmarking GPU acceleration performance...")
                gpu_time = self._benchmark_array_operations(use_gpu=True)
                results['gpu_accelerated']['array_ops_ms'] = gpu_time

                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                results['speedup']['array_ops'] = speedup

                print(".2f")
            else:
                print("⚠️ GPU acceleration not available for benchmarking")

        except Exception as e:
            print(f"⚠️ Benchmarking failed: {e}")

        return results

    def _benchmark_array_operations(self, use_gpu: bool = False, size: int = 1000000) -> float:
        """Benchmark array operations"""
        import time
        import numpy as np

        try:
            # Create test data
            a = np.random.random(size)
            b = np.random.random(size)

            if use_gpu and self.cuda_available:
                # GPU benchmark with cupy
                import cupy as cp
                a_gpu = cp.asarray(a)
                b_gpu = cp.asarray(b)

                start_time = time.time()
                c_gpu = a_gpu + b_gpu * 2  # Simple operations
                d_gpu = cp.sin(c_gpu) * cp.cos(c_gpu)
                result = cp.asnumpy(d_gpu)
                end_time = time.time()

                return (end_time - start_time) * 1000  # ms

            else:
                # CPU benchmark
                start_time = time.time()
                c = a + b * 2
                d = np.sin(c) * np.cos(c)
                end_time = time.time()

                return (end_time - start_time) * 1000  # ms

        except ImportError:
            # Fallback to pure numpy
            start_time = time.time()
            c = a + b * 2
            d = np.sin(c) * np.cos(c)
            end_time = time.time()

            return (end_time - start_time) * 1000  # ms

    def optimize_for_memory(self, target_memory_mb: int = 2048):
        """Optimize GPU memory usage for limited RAM systems"""
        if not self.detected_gpus:
            return

        try:
            # Set GPU memory limits
            if self.cuda_available:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                print("✅ PyTorch CUDA memory optimization applied")

            # Limit GPU memory usage
            total_gpu_memory = sum(gpu.get('memory_total_mb', 0) for gpu in self.detected_gpus)
            if total_gpu_memory > 0:
                memory_limit = min(target_memory_mb // 2, total_gpu_memory // 2)
                os.environ['GPU_MEMORY_LIMIT_MB'] = str(memory_limit)
                print(f"✅ GPU memory limit set to {memory_limit}MB")

        except Exception as e:
            print(f"⚠️ GPU memory optimization failed: {e}")

# Global GPU manager instance
gpu_manager = GPUAccelerationManager()
