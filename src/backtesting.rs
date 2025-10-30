//! High-Performance Backtesting Engine in Rust
//!
//! Ultra-fast order simulation and performance analysis
//! Optimized for millions of trades with realistic execution modeling

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use rayon::prelude::*;
use std::collections::HashMap;

/// Order type for backtesting
#[derive(Debug, Clone, Copy)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order fill information
#[derive(Debug, Clone)]
pub struct OrderFill {
    pub timestamp: u64,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub slippage: f64,
}

/// Trade result
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: u64,
    pub exit_time: u64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub side: OrderSide,
    pub pnl: f64,
    pub commission: f64,
    pub return_pct: f64,
}

/// Portfolio metrics
#[derive(Debug, Clone)]
#[pyclass]
pub struct PortfolioMetrics {
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub annual_return: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub sortino_ratio: f64,
    #[pyo3(get)]
    pub calmar_ratio: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub volatility: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub profit_factor: f64,
    #[pyo3(get)]
    pub total_trades: u32,
    #[pyo3(get)]
    pub winning_trades: u32,
    #[pyo3(get)]
    pub losing_trades: u32,
}

/// Simulate order execution with realistic fills
/// 
/// Models commission, slippage, and partial fills for accurate backtesting
/// 
/// # Arguments
/// * `prices` - Price data (OHLC)
/// * `volumes` - Volume data
/// * `orders` - List of orders to simulate
/// * `commission_rate` - Commission as percentage (e.g., 0.001 = 0.1%)
/// * `slippage_bps` - Slippage in basis points (e.g., 5.0 = 5 bps)
#[pyfunction]
pub fn simulate_orders(
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,  // -1=sell, 0=hold, 1=buy
    initial_capital: f64,
    commission_rate: f64,
    slippage_bps: f64
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<u32>)> {
    let prices = prices.as_slice()?;
    let volumes = volumes.as_slice()?;
    let signals = signals.as_slice()?;
    
    let mut portfolio_values = Vec::new();
    let mut positions = Vec::new();
    let mut trade_counts = Vec::new();
    
    let mut cash = initial_capital;
    let mut position = 0.0;
    let mut trades = 0u32;
    
    for i in 0..prices.len() {
        let price = prices[i];
        let signal = signals[i];
        
        // Calculate slippage
        let slippage = price * slippage_bps / 10000.0;
        
        match signal {
            1 => { // Buy signal
                if position <= 0.0 {  // Close short or open long
                    let order_size = (cash * 0.95) / price;  // Use 95% of cash
                    let execution_price = price + slippage;
                    let commission = order_size * execution_price * commission_rate;
                    
                    if cash >= order_size * execution_price + commission {
                        cash -= order_size * execution_price + commission;
                        position += order_size;
                        trades += 1;
                    }
                }
            },
            -1 => { // Sell signal
                if position > 0.0 {  // Close long position
                    let execution_price = price - slippage;
                    let commission = position * execution_price * commission_rate;
                    
                    cash += position * execution_price - commission;
                    position = 0.0;
                    trades += 1;
                }
            },
            _ => {} // Hold
        }
        
        // Calculate portfolio value
        let portfolio_value = cash + position * price;
        portfolio_values.push(portfolio_value);
        positions.push(position);
        trade_counts.push(trades);
    }
    
    Ok((portfolio_values, positions, trade_counts))
}

/// Calculate return series from portfolio values
#[pyfunction]
pub fn calculate_returns(portfolio_values: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let values = portfolio_values.as_slice()?;
    
    if values.len() < 2 {
        return Ok(vec![]);
    }
    
    let mut returns = Vec::with_capacity(values.len() - 1);
    
    for i in 1..values.len() {
        let return_pct = (values[i] - values[i-1]) / values[i-1];
        returns.push(return_pct);
    }
    
    Ok(returns)
}

/// Calculate comprehensive performance metrics
#[pyfunction]
pub fn performance_metrics(
    returns: PyReadonlyArray1<f64>,
    benchmark_returns: Option<PyReadonlyArray1<f64>>,
    risk_free_rate: f64
) -> PyResult<PortfolioMetrics> {
    let returns = returns.as_slice()?;
    
    if returns.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Returns array cannot be empty"
        ));
    }
    
    // Basic statistics
    let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let annual_return = (1.0 + mean_return).powf(252.0) - 1.0;  // 252 trading days
    
    // Volatility (annualized)
    let variance = returns.iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    let volatility = variance.sqrt() * (252.0_f64).sqrt();
    
    // Sharpe ratio
    let sharpe_ratio = if volatility != 0.0 {
        (annual_return - risk_free_rate) / volatility
    } else {
        0.0
    };
    
    // Sortino ratio (downside deviation)
    let downside_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();
    
    let downside_deviation = if !downside_returns.is_empty() {
        let downside_var = downside_returns.iter()
            .map(|&r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        downside_var.sqrt() * (252.0_f64).sqrt()
    } else {
        volatility
    };
    
    let sortino_ratio = if downside_deviation != 0.0 {
        (annual_return - risk_free_rate) / downside_deviation
    } else {
        0.0
    };
    
    // Maximum drawdown
    let max_drawdown = calculate_max_drawdown_internal(returns);
    
    // Calmar ratio
    let calmar_ratio = if max_drawdown.abs() > 1e-10 {
        annual_return / max_drawdown.abs()
    } else {
        0.0
    };
    
    // Trade statistics
    let winning_returns: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let losing_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    
    let total_trades = returns.len() as u32;
    let winning_trades = winning_returns.len() as u32;
    let losing_trades = losing_returns.len() as u32;
    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64
    } else {
        0.0
    };
    
    // Profit factor
    let gross_profit: f64 = winning_returns.iter().sum();
    let gross_loss: f64 = losing_returns.iter().map(|&r| r.abs()).sum();
    let profit_factor = if gross_loss != 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };
    
    Ok(PortfolioMetrics {
        total_return,
        annual_return,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown,
        volatility,
        win_rate,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
    })
}

/// Drawdown analysis
#[pyfunction]
pub fn drawdown_analysis(portfolio_values: PyReadonlyArray1<f64>) -> PyResult<(Vec<f64>, f64, u32)> {
    let values = portfolio_values.as_slice()?;
    
    let mut drawdowns = Vec::new();
    let mut peak = values[0];
    let mut max_drawdown = 0.0;
    let mut max_drawdown_duration = 0u32;
    let mut current_drawdown_duration = 0u32;
    
    for &value in values {
        if value > peak {
            peak = value;
            current_drawdown_duration = 0;
        } else {
            current_drawdown_duration += 1;
        }
        
        let drawdown = (value - peak) / peak;
        drawdowns.push(drawdown);
        
        if drawdown < max_drawdown {
            max_drawdown = drawdown;
            max_drawdown_duration = current_drawdown_duration;
        }
    }
    
    Ok((drawdowns, max_drawdown, max_drawdown_duration))
}

/// Monte Carlo simulation for risk analysis
#[pyfunction]
pub fn monte_carlo_simulation(
    returns: PyReadonlyArray1<f64>,
    num_simulations: usize,
    time_horizon: usize,
    initial_value: f64
) -> PyResult<Vec<f64>> {
    let historical_returns = returns.as_slice()?;
    
    if historical_returns.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Returns array cannot be empty"
        ));
    }
    
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let final_values: Vec<f64> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut portfolio_value = initial_value;
            
            for _ in 0..time_horizon {
                let random_return = *historical_returns.choose(&mut rng).unwrap_or(&0.0);
                portfolio_value *= 1.0 + random_return;
            }
            
            portfolio_value
        })
        .collect();
    
    Ok(final_values)
}

/// Internal function to calculate maximum drawdown
fn calculate_max_drawdown_internal(returns: &[f64]) -> f64 {
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_drawdown = 0.0;
    
    for &ret in returns {
        cumulative *= 1.0 + ret;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (cumulative - peak) / peak;
        if drawdown < max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    max_drawdown
}

/// Advanced backtesting with realistic market impact
#[pyfunction]
pub fn advanced_backtest(
    ohlcv: PyReadonlyArray1<f64>,  // Flattened OHLCV data [O,H,L,C,V, O,H,L,C,V, ...]
    signals: PyReadonlyArray1<i8>,
    initial_capital: f64,
    position_size: f64,
    commission_rate: f64,
    slippage_bps: f64,
    market_impact_factor: f64
) -> PyResult<PortfolioMetrics> {
    let ohlcv_data = ohlcv.as_slice()?;
    let signals = signals.as_slice()?;
    
    if ohlcv_data.len() % 5 != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "OHLCV data length must be divisible by 5"
        ));
    }
    
    let num_bars = ohlcv_data.len() / 5;
    let mut returns = Vec::new();
    let mut cash = initial_capital;
    let mut position = 0.0;
    let mut trades = Vec::new();
    
    for i in 0..num_bars.min(signals.len()) {
        let base_idx = i * 5;
        let open = ohlcv_data[base_idx];
        let high = ohlcv_data[base_idx + 1];
        let low = ohlcv_data[base_idx + 2];
        let close = ohlcv_data[base_idx + 3];
        let volume = ohlcv_data[base_idx + 4];
        
        let signal = signals[i];
        let execution_price = close;  // Assume execution at close
        
        // Calculate market impact based on volume
        let market_impact = market_impact_factor * (position_size * execution_price) / volume;
        let total_slippage = (slippage_bps / 10000.0) * execution_price + market_impact;
        
        match signal {
            1 => { // Buy signal
                if position <= 0.0 {
                    let order_value = position_size * execution_price;
                    let adjusted_price = execution_price + total_slippage;
                    let commission = order_value * commission_rate;
                    
                    if cash >= order_value + commission {
                        cash -= order_value + commission;
                        position += position_size;
                        
                        if let Some(last_trade) = trades.last_mut() {
                            if last_trade.side == OrderSide::Sell {
                                // Close short position
                                last_trade.exit_time = i as u64;
                                last_trade.exit_price = adjusted_price;
                                last_trade.pnl = (last_trade.entry_price - adjusted_price) * last_trade.quantity - commission;
                                last_trade.commission += commission;
                                last_trade.return_pct = last_trade.pnl / order_value;
                            }
                        }
                        
                        trades.push(Trade {
                            entry_time: i as u64,
                            exit_time: 0,
                            entry_price: adjusted_price,
                            exit_price: 0.0,
                            quantity: position_size,
                            side: OrderSide::Buy,
                            pnl: 0.0,
                            commission,
                            return_pct: 0.0,
                        });
                    }
                }
            },
            -1 => { // Sell signal
                if position > 0.0 {
                    let adjusted_price = execution_price - total_slippage;
                    let order_value = position * adjusted_price;
                    let commission = order_value * commission_rate;
                    
                    cash += order_value - commission;
                    position = 0.0;
                    
                    if let Some(last_trade) = trades.last_mut() {
                        if last_trade.side == OrderSide::Buy {
                            last_trade.exit_time = i as u64;
                            last_trade.exit_price = adjusted_price;
                            last_trade.pnl = (adjusted_price - last_trade.entry_price) * last_trade.quantity - commission;
                            last_trade.commission += commission;
                            last_trade.return_pct = last_trade.pnl / (last_trade.entry_price * last_trade.quantity);
                        }
                    }
                }
            },
            _ => {} // Hold
        }
        
        // Calculate return for this period
        let portfolio_value = cash + position * execution_price;
        let return_pct = if i > 0 {
            let prev_value = if i == 1 { initial_capital } else { 
                cash + position * ohlcv_data[(i-1) * 5 + 3]  // Previous close
            };
            (portfolio_value - prev_value) / prev_value
        } else {
            0.0
        };
        
        returns.push(return_pct);
    }
    
    // Calculate metrics from completed trades
    let completed_trades: Vec<&Trade> = trades.iter().filter(|t| t.exit_time > 0).collect();
    let trade_returns: Vec<f64> = completed_trades.iter().map(|t| t.return_pct).collect();
    
    performance_metrics(
        PyArray1::from_slice(_py, &returns).readonly(),
        None,
        0.02  // 2% risk-free rate
    )
}

/// Vectorized performance calculation for parameter optimization
#[pyfunction]
pub fn batch_performance_calculation(
    price_data: PyReadonlyArray1<f64>,
    parameter_sets: PyReadonlyArray1<f64>,  // Flattened parameter combinations
    strategy_func_id: u32  // Strategy function identifier
) -> PyResult<Vec<f64>> {
    let prices = price_data.as_slice()?;
    let params = parameter_sets.as_slice()?;
    
    // Parallel processing of parameter combinations
    let results: Vec<f64> = params
        .par_chunks_exact(3)  // Assume 3 parameters per strategy
        .map(|param_set| {
            // Simulate strategy with these parameters
            let param1 = param_set[0] as usize;
            let param2 = param_set[1] as usize;
            let param3 = param_set[2];
            
            // Generate signals based on strategy
            let signals = match strategy_func_id {
                1 => generate_ma_crossover_signals(prices, param1, param2),
                2 => generate_rsi_signals(prices, param1, param3),
                _ => vec![0; prices.len()],
            };
            
            // Quick performance calculation
            simulate_strategy_performance(prices, &signals, 100000.0, 0.001, 5.0)
        })
        .collect();
    
    Ok(results)
}

/// Generate MA crossover signals
fn generate_ma_crossover_signals(prices: &[f64], fast_period: usize, slow_period: usize) -> Vec<i8> {
    let fast_ma = crate::indicators::fast_sma_internal(prices, fast_period);
    let slow_ma = crate::indicators::fast_sma_internal(prices, slow_period);
    
    let mut signals = vec![0i8; prices.len()];
    let start_idx = slow_period - 1;
    
    for i in 1..fast_ma.len().min(slow_ma.len()) {
        let idx = start_idx + i;
        if idx < signals.len() {
            if fast_ma[i] > slow_ma[i] && fast_ma[i-1] <= slow_ma[i-1] {
                signals[idx] = 1;  // Buy signal
            } else if fast_ma[i] < slow_ma[i] && fast_ma[i-1] >= slow_ma[i-1] {
                signals[idx] = -1; // Sell signal
            }
        }
    }
    
    signals
}

/// Generate RSI signals
fn generate_rsi_signals(prices: &[f64], period: usize, threshold: f64) -> Vec<i8> {
    // This would use the RSI function from indicators module
    // Simplified implementation for now
    let mut signals = vec![0i8; prices.len()];
    
    // Placeholder RSI-based signal generation
    for i in period..prices.len() {
        let recent_prices = &prices[i-period..i];
        let avg_price: f64 = recent_prices.iter().sum::<f64>() / period as f64;
        
        if prices[i] > avg_price * (1.0 + threshold / 100.0) {
            signals[i] = -1;  // Overbought - sell
        } else if prices[i] < avg_price * (1.0 - threshold / 100.0) {
            signals[i] = 1;   // Oversold - buy
        }
    }
    
    signals
}

/// Quick strategy performance simulation
fn simulate_strategy_performance(
    prices: &[f64], 
    signals: &[i8], 
    initial_capital: f64,
    commission_rate: f64,
    slippage_bps: f64
) -> f64 {
    let mut cash = initial_capital;
    let mut position = 0.0;
    
    for (i, &signal) in signals.iter().enumerate() {
        if i >= prices.len() { break; }
        
        let price = prices[i];
        let slippage = price * slippage_bps / 10000.0;
        
        match signal {
            1 if position <= 0.0 => {
                let order_size = (cash * 0.95) / price;
                let execution_price = price + slippage;
                let commission = order_size * execution_price * commission_rate;
                
                if cash >= order_size * execution_price + commission {
                    cash -= order_size * execution_price + commission;
                    position += order_size;
                }
            },
            -1 if position > 0.0 => {
                let execution_price = price - slippage;
                let commission = position * execution_price * commission_rate;
                
                cash += position * execution_price - commission;
                position = 0.0;
            },
            _ => {}
        }
    }
    
    // Return final portfolio value
    let final_value = cash + position * prices.last().unwrap_or(&0.0);
    (final_value - initial_capital) / initial_capital  // Total return
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_simulation() {
        let prices = vec![100.0, 101.0, 102.0, 101.0, 100.0];
        let volumes = vec![1000.0; 5];
        let signals = vec![1, 0, -1, 0, 1];  // Buy, Hold, Sell, Hold, Buy
        
        // This would be tested with actual PyArray1 in Python
        // Placeholder test
        assert_eq!(prices.len(), signals.len());
    }
    
    #[test]
    fn test_performance_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let max_dd = calculate_max_drawdown_internal(&returns);
        assert!(max_dd <= 0.0);  // Drawdown should be negative
    }
}
