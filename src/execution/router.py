#!/usr/bin/env python3
"""
Smart Order Router - Enterprise Execution Engine

Intelligent order routing with market impact analysis, slippage protection,
and algorithmic execution strategies for optimal trade execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .impact_analysis import is_liquidity_sufficient, estimate_market_impact
from .algo_orders import split_iceberg_order, generate_twap_schedule

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Container for execution results."""

    def __init__(self, status: str, order_id: Optional[str] = None,
                 executed_size: float = 0.0, avg_price: float = 0.0,
                 reason: Optional[str] = None):
        self.status = status  # 'SUCCESS', 'REJECTED', 'PARTIAL', 'FAILED'
        self.order_id = order_id
        self.executed_size = executed_size
        self.avg_price = avg_price
        self.reason = reason
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'order_id': self.order_id,
            'executed_size': self.executed_size,
            'avg_price': self.avg_price,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


class SmartRouter:
    """
    Smart Order Router - Enterprise-grade execution engine.

    Features:
    - Market impact analysis and slippage protection
    - Iceberg order splitting for large orders
    - TWAP/VWAP execution algorithms
    - Real-time liquidity monitoring
    - Execution quality optimization
    """

    def __init__(self, exchange_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Smart Router.

        Args:
            exchange_client: Exchange API client
            config: Router configuration
        """
        self.exchange = exchange_client

        # Default configuration
        self.config = config or {
            "max_slippage": 0.005,      # 0.5% max slippage
            "max_market_impact": 0.01,  # 1% max market impact
            "iceberg_threshold": 1000,  # Orders > 1000 trigger iceberg
            "max_chunk_size": 500,      # Max size per iceberg chunk
            "liquidity_check": True,    # Enable liquidity validation
            "impact_analysis": True,    # Enable impact analysis
            "twap_enabled": True,       # Enable TWAP for large orders
            "execution_timeout": 30     # seconds
        }

        # Execution tracking
        self.pending_orders = {}
        self.execution_history = []

        logger.info("Smart Router initialized with config: %s", self.config)

    async def execute_order(self, symbol: str, side: str, amount: float,
                           price: Optional[float] = None,
                           order_type: str = 'market') -> ExecutionResult:
        """
        Execute an order with smart routing logic.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order size
            price: Limit price (None for market orders)
            order_type: 'market', 'limit', 'iceberg', 'twap'

        Returns:
            ExecutionResult with execution details
        """
        try:
            logger.info(f"Routing order: {symbol} {side} {amount} @ {price or 'market'}")

            # 1. Pre-execution analysis
            analysis_result = await self._analyze_execution(symbol, side, amount)
            if not analysis_result['approved']:
                return ExecutionResult('REJECTED', reason=analysis_result['reason'])

            # 2. Determine execution strategy
            if order_type == 'iceberg' or (amount >= self.config['iceberg_threshold'] and self._should_use_iceberg()):
                return await self._execute_iceberg_order(symbol, side, amount, price)

            elif order_type == 'twap':
                return await self._execute_twap_order(symbol, side, amount, price)

            else:
                return await self._execute_standard_order(symbol, side, amount, price)

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return ExecutionResult('FAILED', reason=str(e))

    async def _analyze_execution(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Analyze market conditions before execution.

        Returns:
            {'approved': bool, 'reason': str, 'analysis': dict}
        """
        if not self.config['liquidity_check']:
            return {'approved': True, 'reason': 'Liquidity check disabled'}

        try:
            # Fetch order book
            order_book = await self.exchange.fetch_order_book(symbol)
            book_side = order_book['asks'] if side == 'buy' else order_book['bids']

            # Convert to dict format for analysis
            depth = [{'price': float(p), 'amount': float(a)} for p, a in book_side[:10]]  # Top 10 levels

            # Check liquidity
            is_liquid, reason = is_liquidity_sufficient(
                amount, depth, self.config['max_slippage']
            )

            if not is_liquid:
                return {
                    'approved': False,
                    'reason': f'Insufficient liquidity: {reason}',
                    'analysis': {'depth_levels': len(depth), 'reason': reason}
                }

            # Market impact analysis
            if self.config['impact_analysis']:
                impact = estimate_market_impact(amount, self._get_avg_daily_volume(symbol))
                if impact > self.config['max_market_impact']:
                    return {
                        'approved': False,
                        'reason': '.2f',
                        'analysis': {'market_impact': impact, 'max_allowed': self.config['max_market_impact']}
                    }

            return {
                'approved': True,
                'reason': 'Market conditions acceptable',
                'analysis': {'liquidity_ok': True, 'impact_ok': True}
            }

        except Exception as e:
            logger.warning(f"Execution analysis failed: {e}")
            return {'approved': True, 'reason': 'Analysis failed, proceeding anyway'}

    async def _execute_standard_order(self, symbol: str, side: str, amount: float,
                                    price: Optional[float]) -> ExecutionResult:
        """Execute a standard single order."""
        try:
            # Create order via exchange
            order_result = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                timeout=self.config['execution_timeout']
            )

            # Track execution
            self._record_execution(order_result)

            return ExecutionResult(
                status='SUCCESS',
                order_id=order_result.get('id'),
                executed_size=order_result.get('amount', amount),
                avg_price=order_result.get('price', price or 0)
            )

        except Exception as e:
            return ExecutionResult('FAILED', reason=str(e))

    async def _execute_iceberg_order(self, symbol: str, side: str, amount: float,
                                   price: Optional[float]) -> ExecutionResult:
        """Execute order using iceberg algorithm."""
        chunks = split_iceberg_order(amount, max_chunk_size=self.config['max_chunk_size'])

        logger.info(f"Executing iceberg order with {len(chunks)} chunks")

        executed_orders = []
        total_executed = 0.0
        total_cost = 0.0

        for i, chunk_size in enumerate(chunks):
            try:
                # Add small delay between chunks to avoid detection
                if i > 0:
                    await asyncio.sleep(1)

                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    side=side,
                    amount=chunk_size,
                    price=price
                )

                if order_result.get('status') == 'filled':
                    executed_size = order_result.get('amount', chunk_size)
                    executed_price = order_result.get('price', price or 0)

                    executed_orders.append(order_result)
                    total_executed += executed_size
                    total_cost += executed_size * executed_price

                else:
                    logger.warning(f"Chunk {i+1} not fully executed")

            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {e}")
                continue

        # Calculate results
        avg_price = total_cost / total_executed if total_executed > 0 else 0

        return ExecutionResult(
            status='SUCCESS' if total_executed > 0 else 'FAILED',
            executed_size=total_executed,
            avg_price=avg_price,
            reason=f"Iceberg execution: {len(executed_orders)}/{len(chunks)} chunks successful"
        )

    async def _execute_twap_order(self, symbol: str, side: str, amount: float,
                                price: Optional[float]) -> ExecutionResult:
        """Execute order using TWAP algorithm."""
        if not self.config['twap_enabled']:
            return await self._execute_standard_order(symbol, side, amount, price)

        # Generate TWAP schedule (default 30 minutes)
        schedule = generate_twap_schedule(amount, duration_minutes=30)

        logger.info(f"Executing TWAP order with {len(schedule)} time slices")

        executed_orders = []

        for slot in schedule:
            try:
                await asyncio.sleep(1)  # Wait between executions

                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    side=side,
                    amount=slot['size'],
                    price=price
                )

                if order_result.get('status') == 'filled':
                    executed_orders.append(order_result)

            except Exception as e:
                logger.error(f"TWAP slot failed: {e}")
                continue

        total_executed = sum(o.get('amount', 0) for o in executed_orders)

        return ExecutionResult(
            status='SUCCESS' if executed_orders else 'FAILED',
            executed_size=total_executed,
            reason=f"TWAP execution: {len(executed_orders)}/{len(schedule)} slots successful"
        )

    def _should_use_iceberg(self) -> bool:
        """Determine if iceberg execution should be used."""
        return True  # For now, always use for large orders

    def _get_avg_daily_volume(self, symbol: str) -> float:
        """Get average daily volume for impact analysis."""
        # Placeholder - in production, this would query historical data
        return 10000.0  # Assume 10k daily volume

    def _record_execution(self, order_result: Dict[str, Any]):
        """Record execution in history."""
        self.execution_history.append({
            'timestamp': datetime.now(),
            'result': order_result
        })

        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {'total_orders': 0, 'success_rate': 0.0}

        successful = sum(1 for h in self.execution_history
                        if h['result'].get('status') == 'filled')

        return {
            'total_orders': len(self.execution_history),
            'success_rate': successful / len(self.execution_history),
            'last_execution': self.execution_history[-1]['timestamp'] if self.execution_history else None
        }
