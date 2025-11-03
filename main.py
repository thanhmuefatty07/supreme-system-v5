#!/usr/bin/env python3
"""
Supreme System V5 - Ultra-Constrained Entry Point
Optimized for 1GB RAM, 2 vCPU with ETH-USDT scalping
Agent Mode: Maximum resource efficiency

Usage:
    python main.py                          # Paper trading
    EXECUTION_MODE=live python main.py     # Live trading (CAUTION)
    ULTRA_CONSTRAINED=1 python main.py     # Force ultra-constrained mode
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Optional
import time

# Add project path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None


class UltraConstrainedLauncher:
    """Ultra-constrained launcher for Supreme System V5"""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.setup_signal_handlers()
        self.validate_environment()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def validate_environment(self) -> bool:
        """Validate environment for ultra-constrained deployment"""
        logger.info("🔍 Validating ultra-constrained environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            logger.error(f"Python 3.10+ required, found {sys.version_info}")
            return False
            
        # Check RAM availability
        if psutil:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            logger.info(f"RAM: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
            
            if available_gb < 0.8:
                logger.error(f"Insufficient RAM: {available_gb:.1f}GB < 0.8GB required")
                return False
            elif available_gb < 1.0:
                logger.warning(f"Low RAM: {available_gb:.1f}GB, monitoring required")
                
        # Check configuration
        env_files = [".env", ".env.ultra_constrained"]
        config_found = any(Path(f).exists() for f in env_files)
        
        if not config_found:
            logger.error("No configuration found. Run 'make setup-ultra' first")
            return False
            
        logger.success("✅ Environment validation passed")
        return True
        
    def load_ultra_constrained_config(self):
        """Load ultra-constrained configuration"""
        # Force ultra-constrained mode
        os.environ["ULTRA_CONSTRAINED"] = "1"
        
        # Load .env file (prefer .env, fallback to .env.ultra_constrained)
        env_file = ".env" if Path(".env").exists() else ".env.ultra_constrained"
        
        if Path(env_file).exists():
            logger.info(f"📋 Loading configuration from {env_file}")
            
            # Simple .env parser
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                        
        # Apply ultra-constrained defaults
        defaults = {
            "SYMBOLS": "ETH-USDT",
            "SCALPING_INTERVAL_MIN": "30",
            "SCALPING_INTERVAL_MAX": "60", 
            "NEWS_POLL_INTERVAL_MINUTES": "12",
            "MAX_RAM_MB": "450",
            "MAX_CPU_PERCENT": "85",
            "LOG_LEVEL": "WARNING",
            "BUFFER_SIZE_LIMIT": "200",
            "DATA_SOURCES": "binance,coingecko",
            "EXECUTION_MODE": "paper"
        }
        
        for key, value in defaults.items():
            if key not in os.environ:
                os.environ[key] = value
                
        logger.info(f"🎯 Ultra-constrained mode active:")
        logger.info(f"   Symbol: {os.environ['SYMBOLS']}")
        logger.info(f"   Scalping: {os.environ['SCALPING_INTERVAL_MIN']}-{os.environ['SCALPING_INTERVAL_MAX']}s")
        logger.info(f"   News: {os.environ['NEWS_POLL_INTERVAL_MINUTES']}min")
        logger.info(f"   RAM Budget: {os.environ['MAX_RAM_MB']}MB")
        logger.info(f"   Mode: {os.environ['EXECUTION_MODE']}")
        
    async def launch_system(self):
        """Launch Supreme System V5 with ultra-constrained profile"""
        try:
            # Load configuration
            self.load_ultra_constrained_config()
            
            # Import core components (after config loaded)
            from supreme_system_v5.master_orchestrator import MasterOrchestrator
            from supreme_system_v5.resource_monitor import SystemResourceMonitor
            
            # Initialize resource monitor
            monitor = SystemResourceMonitor({
                'max_memory_mb': int(os.environ.get('MAX_RAM_MB', 450)),
                'max_cpu_percent': int(os.environ.get('MAX_CPU_PERCENT', 85)),
                'check_interval': 30,
                'emergency_shutdown_enabled': True
            })
            
            # Initialize orchestrator with ultra-constrained settings
            orchestrator_config = {
                'symbols': [os.environ['SYMBOLS']],  # Single symbol
                'execution_mode': os.environ.get('EXECUTION_MODE', 'paper'),
                'resource_limits': {
                    'max_memory_mb': int(os.environ.get('MAX_RAM_MB', 450)),
                    'max_cpu_percent': int(os.environ.get('MAX_CPU_PERCENT', 85))
                },
                'data_sources': os.environ.get('DATA_SOURCES', 'binance,coingecko').split(','),
                'scalping_config': {
                    'interval_min': int(os.environ.get('SCALPING_INTERVAL_MIN', 30)),
                    'interval_max': int(os.environ.get('SCALPING_INTERVAL_MAX', 60)),
                    'jitter_percent': 0.10
                },
                'news_config': {
                    'poll_interval_minutes': int(os.environ.get('NEWS_POLL_INTERVAL_MINUTES', 12)),
                    'enabled': os.environ.get('NEWS_ENABLED', 'true').lower() == 'true'
                },
                'buffer_limits': {
                    'price_history': int(os.environ.get('BUFFER_SIZE_LIMIT', 200)),
                    'indicator_cache': 100,
                    'event_history': 50
                }
            }
            
            logger.info("🚀 Initializing Supreme System V5 - Ultra-Constrained Mode")
            orchestrator = MasterOrchestrator(orchestrator_config)
            
            # Start monitoring
            monitor_task = asyncio.create_task(monitor.start_monitoring())
            
            # Start main system
            logger.info("▶️  Starting trading system...")
            system_task = asyncio.create_task(orchestrator.run())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("🛑 Shutdown initiated...")
            
            # Cancel tasks
            system_task.cancel()
            monitor_task.cancel()
            
            # Wait for cleanup
            try:
                await asyncio.wait_for(system_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
                
            try:
                await asyncio.wait_for(monitor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
                
            logger.success("✅ Supreme System V5 shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            raise
            
    async def shutdown(self):
        """Initiate graceful shutdown"""
        self.shutdown_event.set()
        
        
def setup_logging():
    """Setup logging for ultra-constrained deployment"""
    # Remove default logger
    logger.remove()
    
    # Get log level from environment
    log_level = os.environ.get("LOG_LEVEL", "WARNING")
    
    # Console logging (minimal for ultra-constrained)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        enqueue=True
    )
    
    # File logging (if enabled)
    if os.environ.get("LOG_TO_FILE", "true").lower() == "true":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "supreme_system.log",
            level=log_level,
            rotation=f"{os.environ.get('LOG_ROTATION_MB', 10)}MB",
            retention="3 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
        

def main():
    """Main entry point"""
    # Setup logging first
    setup_logging()
    
    logger.info("🎯 Supreme System V5 - Ultra-Constrained Launcher")
    logger.info("=" * 50)
    
    # Detect environment
    if psutil:
        memory = psutil.virtual_memory()
        ram_gb = memory.total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        logger.info(f"Hardware: {ram_gb:.1f}GB RAM, {cpu_count} cores")
        
        if ram_gb <= 1.5:
            logger.info("🔋 Ultra-constrained mode: Optimized for minimal resources")
        else:
            logger.warning(f"⚠️  Hardware has {ram_gb:.1f}GB RAM - consider higher performance profile")
    else:
        logger.warning("psutil not available - hardware detection disabled")
        
    # Check execution mode
    execution_mode = os.environ.get("EXECUTION_MODE", "paper")
    if execution_mode == "live":
        logger.warning("🚨 LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("Press Ctrl+C within 10 seconds to cancel...")
        time.sleep(10)
        logger.warning("🔥 Proceeding with live trading")
    else:
        logger.info("📊 Paper trading mode - no real money at risk")
        
    # Launch system
    try:
        launcher = UltraConstrainedLauncher()
        asyncio.run(launcher.launch_system())
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise
        
    logger.info("🏁 Supreme System V5 terminated")
    

if __name__ == "__main__":
    main()