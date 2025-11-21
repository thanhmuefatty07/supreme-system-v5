import pytest
from unittest.mock import AsyncMock, MagicMock
import os
import json
from src.execution.router import SmartRouter, ExecutionResult

@pytest.fixture
def router():
    mock_ex = MagicMock()
    # Setup default mocks
    mock_ex.fetch_order_book = AsyncMock(return_value={
        'asks': [[100.0, 10.0]],
        'bids': [[99.0, 10.0]]
    })
    mock_ex.create_order = AsyncMock(return_value={'id': 'test_order_123'})

    router = SmartRouter(mock_ex, log_file="test_trades.jsonl")
    yield router
    # Cleanup
    if os.path.exists("test_trades.jsonl"):
        os.remove("test_trades.jsonl")

@pytest.mark.asyncio
async def test_successful_execution_flow(router):
    """Test complete successful order execution."""
    result = await router.execute_order('BTC/USDT', 'buy', 0.1)

    assert result['status'] == 'FILLED'
    assert result['symbol'] == 'BTC/USDT'
    assert result['side'] == 'buy'
    assert result['quantity'] == 0.1
    assert result['price'] == 100.0
    assert result['order_id'] == 'test_order_123'

    # Verify disk persistence
    assert os.path.exists("test_trades.jsonl")
    with open("test_trades.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['status'] == 'FILLED'
        assert 'timestamp' in data

@pytest.mark.asyncio
async def test_insufficient_liquidity_rejection(router):
    """Test rejection due to insufficient liquidity."""
    # Mock order book with low volume
    router.exchange.fetch_order_book = AsyncMock(return_value={
        'asks': [[100.0, 0.05]],  # Only 0.05 available, need 0.1
        'bids': [[99.0, 10.0]]
    })

    result = await router.execute_order('BTC/USDT', 'buy', 0.1)

    assert result['status'] == 'REJECTED'
    assert 'liquidity' in result['error_message']

@pytest.mark.asyncio
async def test_execution_error_handling(router):
    """Test error handling during execution."""
    router.exchange.create_order.side_effect = Exception("Exchange API Error")

    result = await router.execute_order('ETH/USDT', 'sell', 1.0)

    assert result['status'] == 'REJECTED'
    assert 'Exchange API Error' in result['error_message']

    # Verify error is logged to file
    with open("test_trades.jsonl", "r") as f:
        data = json.loads(f.read())
        assert data['error_message'] == 'Exchange API Error'

@pytest.mark.asyncio
async def test_sell_order_execution(router):
    """Test sell order flow."""
    result = await router.execute_order('ETH/USDT', 'sell', 0.5)

    assert result['status'] == 'FILLED'
    assert result['side'] == 'sell'
    assert result['price'] == 99.0  # Best bid price

@pytest.mark.asyncio
async def test_execution_result_dataclass():
    """Test ExecutionResult dataclass functionality."""
    result = ExecutionResult(
        status='FILLED',
        symbol='BTC/USDT',
        side='buy',
        quantity=0.1,
        price=100.0,
        order_id='test_123',
        timestamp=1234567890.0
    )

    # Test to_dict
    data = result.to_dict()
    assert data['status'] == 'FILLED'
    assert data['symbol'] == 'BTC/USDT'
    assert data['quantity'] == 0.1

@pytest.mark.asyncio
async def test_get_execution_stats(router):
    """Test execution stats retrieval from disk."""
    # Execute successful orders
    await router.execute_order('BTC/USDT', 'buy', 0.1)
    await router.execute_order('ETH/USDT', 'sell', 0.5)

    # Mock insufficient liquidity for the third order
    router.exchange.fetch_order_book = AsyncMock(return_value={
        'asks': [[100.0, 0.05]],  # Only 0.05 available, need 0.2
        'bids': [[99.0, 10.0]]
    })
    await router.execute_order('BTC/USDT', 'buy', 0.2)  # This will be rejected

    stats = router.get_execution_stats()
    assert stats['total_orders'] == 3
    assert stats['success_rate'] == 2/3  # 2 successful, 1 rejected
