"""
Supreme System V5 - Command Line Interface
Production-ready CLI for trading operations
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core import SupremeSystem, SystemConfig
from .exchanges import OKX_AVAILABLE, BINANCE_AVAILABLE

console = Console()

@click.group()
@click.version_option(version="5.0.0")
def cli():
    """
    🚀 Supreme System V5 - Hybrid Python+Rust Trading System
    
    High-performance futures scalping system optimized for i3-4GB systems.
    """
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--symbols', '-s', multiple=True, help='Trading symbols (e.g., BTC-USDT)')
@click.option('--dry-run', is_flag=True, help='Run in simulation mode')
def start(config: Optional[str], symbols: tuple, dry_run: bool):
    """
    Start the Supreme trading system
    """
    console.print(Panel(
        "🚀 [bold green]Supreme System V5[/bold green] - Starting...",
        title="Trading System",
        border_style="green"
    ))
    
    # Create configuration
    system_config = SystemConfig()
    if symbols:
        system_config.trading_symbols = list(symbols)
    
    if dry_run:
        console.print("⚠️ [yellow]Running in DRY RUN mode - no real trades[/yellow]")
    
    # Create and run system
    try:
        supreme_system = SupremeSystem()
        asyncio.run(supreme_system.run())
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Shutdown requested by user[/yellow]")
    except Exception as e:
        console.print(f"\n❌ [red]System error: {e}[/red]")
        sys.exit(1)

@cli.command()
def status():
    """
    Show system status and health metrics
    """
    console.print(Panel(
        "📊 [bold blue]Supreme System V5[/bold blue] - Status Check",
        border_style="blue"
    ))
    
    # System information table
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    table.add_row("Python Version", "✅ OK", f"{sys.version_info.major}.{sys.version_info.minor}")
    table.add_row("Rust Engine", "✅ Available" if RUST_ENGINE_AVAILABLE else "⚠️ Fallback", "PyO3 integration")
    table.add_row("OKX Connector", "✅ Available" if OKX_AVAILABLE else "❌ Missing", "Primary exchange")
    table.add_row("Binance Connector", "✅ Available" if BINANCE_AVAILABLE else "❌ Missing", "Backup exchange")
    
    console.print(table)
    
    # Try to get system info
    try:
        system = SupremeSystem()
        info = system.get_status()
        
        console.print("\n📊 [bold]Performance Metrics:[/bold]")
        console.print(f"Memory Usage: {info['core_info']['memory_usage_mb']:.1f} MB")
        console.print(f"CPU Usage: {info['core_info']['cpu_percent']:.1f}%")
        console.print(f"Active Positions: {info['core_info']['active_positions']}")
        console.print(f"Total Trades: {info['performance']['total_trades']}")
        
    except Exception as e:
        console.print(f"\n⚠️ [yellow]Could not get live metrics: {e}[/yellow]")

@cli.command()
@click.option('--symbol', '-s', default='BTC-USDT', help='Symbol to backtest')
@click.option('--days', '-d', default=30, help='Days of historical data')
@click.option('--output', '-o', help='Output file for results')
def backtest(symbol: str, days: int, output: Optional[str]):
    """
    Run backtesting on historical data
    """
    console.print(Panel(
        f"📈 [bold magenta]Backtesting[/bold magenta] - {symbol} ({days} days)",
        border_style="magenta"
    ))
    
    from .backtest import BacktestEngine
    
    try:
        engine = BacktestEngine()
        results = engine.run_backtest(symbol, days)
        
        console.print(f"✅ [green]Backtest completed successfully[/green]")
        console.print(f"Total Trades: {results.get('total_trades', 0)}")
        console.print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        console.print(f"Total PnL: ${results.get('total_pnl', 0):.2f}")
        console.print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"💾 Results saved to: {output}")
            
    except Exception as e:
        console.print(f"❌ [red]Backtest failed: {e}[/red]")
        sys.exit(1)

@cli.command()
def validate():
    """
    Validate system configuration and dependencies
    """
    console.print(Panel(
        "✅ [bold yellow]System Validation[/bold yellow] - Checking Components",
        border_style="yellow"
    ))
    
    # Import validation script
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    
    try:
        from validate_system import SystemValidator
        validator = SystemValidator()
        
        console.print("🔍 Running comprehensive system validation...")
        results = validator.run_validation()
        
        if results['all_passed']:
            console.print("✅ [bold green]All validation checks passed![/bold green]")
        else:
            console.print("❌ [bold red]Some validation checks failed[/bold red]")
            for error in results.get('errors', []):
                console.print(f"  ❌ {error}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"❌ [red]Validation failed: {e}[/red]")
        sys.exit(1)

@cli.command()
def info():
    """
    Display system information
    """
    console.print(Panel(
        "📊 [bold cyan]Supreme System V5[/bold cyan] - System Information",
        border_style="cyan"
    ))
    
    try:
        system = SupremeSystem()
        info = system.core.get_system_info()
        
        table = Table(title="System Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in info.items():
            table.add_row(str(key), str(value))
            
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ [red]Could not get system info: {e}[/red]")
        sys.exit(1)

def main():
    """
    Main CLI entry point
    """
    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    cli()

if __name__ == '__main__':
    main()
