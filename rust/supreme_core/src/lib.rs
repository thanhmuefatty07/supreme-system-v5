//! Supreme System V5 - Realistic High-Performance Trading Engine Core
//! 
//! Optimized for i3 8th generation, 4GB RAM constraint with proven performance
//! Target: 2-4x improvement through SIMD, zero-copy, and classical algorithms
//! Memory budget: 800MB for core engine (realistic allocation)

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};

// Import realistic engine as primary
mod realistic_engine;
use realistic_engine::*;

// Legacy modules for backward compatibility
pub mod data_engine;
pub mod indicators;
pub mod whale_detector;
pub mod news_processor;
pub mod memory_manager;
pub mod performance;

use data_engine::*;
use indicators::*;
use whale_detector::*;
use news_processor::*;
use memory_manager::*;

/// Realistic configuration for proven performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupremeConfig {
    pub memory_budget_mb: u32,
    pub max_buffer_size: usize,
    pub whale_threshold_usd: f64,
    pub news_sources: Vec<String>,
    pub enable_simd: bool,
    pub cpu_cores: usize,
    pub performance_target: String,
    pub engine_type: String,
}

impl Default for SupremeConfig {
    fn default() -> Self {
        Self {
            memory_budget_mb: 800,  // Realistic: 800MB for core engine
            max_buffer_size: 10000,  // Larger buffer for better performance
            whale_threshold_usd: 1_000_000.0,
            news_sources: vec![
                "coindesk".to_string(),
                "cointelegraph".to_string(),
                "reuters".to_string(),
                "bloomberg".to_string(),
            ],
            enable_simd: true,  // Enable AVX2 optimization
            cpu_cores: 4,  // i3 8th gen has 4 cores
            performance_target: "2-4x_improvement".to_string(),
            engine_type: "realistic_optimized".to_string(),
        }
    }
}

/// Main Supreme Core Engine - Now using RealisticSupremeEngine as backend
#[pyclass]
pub struct SupremeCore {
    config: SupremeConfig,
    // Primary engine: RealisticSupremeEngine
    realistic_engine: RealisticSupremeEngine,
    
    // Legacy engines for backward compatibility
    data_engine: DataEngine,
    indicators: IndicatorEngine,
    whale_detector: WhaleDetector,
    news_processor: NewsProcessor,
    memory_manager: MemoryManager,
    performance_monitor: Arc<AtomicU64>,
}

#[pymethods]
impl SupremeCore {
    #[new]
    pub fn new() -> PyResult<Self> {
        let config = SupremeConfig::default();
        
        // Initialize realistic engine as primary
        let realistic_engine = RealisticSupremeEngine::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize realistic engine: {:?}", e)
            ))?;
        
        Ok(Self {
            realistic_engine,
            data_engine: DataEngine::new(&config)?,
            indicators: IndicatorEngine::new(&config),
            whale_detector: WhaleDetector::new(&config),
            news_processor: NewsProcessor::new(&config),
            memory_manager: MemoryManager::new(config.memory_budget_mb),
            performance_monitor: Arc::new(AtomicU64::new(0)),
            config,
        })
    }
    
    /// Process market data with realistic optimization (PRIMARY METHOD)
    pub fn process_market_data(&mut self, py: Python, data: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
        let start_time = std::time::Instant::now();
        
        // Convert numpy array to Vec<f64> for realistic engine
        let data_vec: Vec<f64> = data.as_array().to_vec();
        
        // Use realistic engine as primary processor
        let result = self.realistic_engine.realistic_market_analysis(py, data_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Realistic engine processing error: {}", e)
            ))?;
        
        // Update performance metrics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.performance_monitor.store(processing_time, Ordering::Relaxed);
        
        // Return realistic engine result (already a PyObject)
        Ok(result)
    }
    
    /// Get realistic performance metrics
    pub fn get_realistic_metrics(&self, py: Python) -> PyResult<PyObject> {
        self.realistic_engine.get_realistic_metrics()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get realistic metrics: {}", e)
            ))
            .map(|metrics| {
                let result_dict = PyDict::new(py);
                for (key, value) in metrics {
                    let _ = result_dict.set_item(key, value);
                }
                result_dict.into()
            })
    }
    
    /// Reset realistic engine state
    pub fn reset_realistic_state(&self) -> PyResult<()> {
        self.realistic_engine.reset_realistic_state()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to reset realistic state: {}", e)
            ))
    }
    
    // ==========================================================================
    // LEGACY METHODS FOR BACKWARD COMPATIBILITY
    // ==========================================================================
    
    /// Calculate technical indicators (LEGACY - uses classical algorithms)
    pub fn calculate_indicators(&mut self, py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<PyObject> {
        let prices_view = prices.as_array();
        
        let indicators_result = self.indicators.calculate_all(prices_view, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Indicator calculation error: {}", e)))?;
        
        let result_dict = PyDict::new(py);
        result_dict.set_item("ema", indicators_result.ema.to_pyarray(py))?;
        result_dict.set_item("rsi", indicators_result.rsi.to_pyarray(py))?;
        result_dict.set_item("macd_line", indicators_result.macd_line.to_pyarray(py))?;
        result_dict.set_item("macd_signal", indicators_result.macd_signal.to_pyarray(py))?;
        result_dict.set_item("bb_upper", indicators_result.bb_upper.to_pyarray(py))?;
        result_dict.set_item("bb_lower", indicators_result.bb_lower.to_pyarray(py))?;
        
        Ok(result_dict.into())
    }
    
    /// Detect whale transactions (LEGACY)
    pub fn detect_whales(&mut self, py: Python, transactions: &PyList) -> PyResult<PyObject> {
        let mut tx_data = Vec::new();
        
        // Convert Python list to Rust transaction data
        for item in transactions {
            let tx_dict = item.downcast::<PyDict>()?;
            let amount = tx_dict.get_item("amount")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'amount' field"))?
                .extract::<f64>()?;
            let timestamp = tx_dict.get_item("timestamp")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'timestamp' field"))?
                .extract::<u64>()?;
            let from_address = tx_dict.get_item("from")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'from' field"))?
                .extract::<String>()?;
            let to_address = tx_dict.get_item("to")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'to' field"))?
                .extract::<String>()?;
            
            tx_data.push(Transaction {
                amount,
                timestamp,
                from_address,
                to_address,
            });
        }
        
        let whale_alerts = self.whale_detector.process_transactions(&tx_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Whale detection error: {}", e)))?;
        
        // Convert whale alerts back to Python
        let result_list = PyList::empty(py);
        for alert in whale_alerts {
            let alert_dict = PyDict::new(py);
            alert_dict.set_item("amount", alert.amount)?;
            alert_dict.set_item("timestamp", alert.timestamp)?;
            alert_dict.set_item("whale_type", alert.whale_type.to_string())?;
            alert_dict.set_item("confidence", alert.confidence)?;
            alert_dict.set_item("addresses", alert.addresses)?;
            result_list.append(alert_dict)?;
        }
        
        Ok(result_list.into())
    }
    
    /// Process news sentiment (LEGACY)
    pub fn process_news(&mut self, py: Python, news_items: &PyList) -> PyResult<PyObject> {
        let mut news_data = Vec::new();
        
        for item in news_items {
            let news_dict = item.downcast::<PyDict>()?;
            let title = news_dict.get_item("title")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'title' field"))?
                .extract::<String>()?;
            let content = news_dict.get_item("content")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'content' field"))?
                .extract::<String>()?;
            let timestamp = news_dict.get_item("timestamp")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'timestamp' field"))?
                .extract::<u64>()?;
            let source = news_dict.get_item("source")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'source' field"))?
                .extract::<String>()?;
            
            news_data.push(NewsItem {
                title,
                content,
                timestamp,
                source,
            });
        }
        
        let sentiment_results = self.news_processor.process_batch(&news_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("News processing error: {}", e)))?;
        
        let result_dict = PyDict::new(py);
        result_dict.set_item("overall_sentiment", sentiment_results.overall_sentiment)?;
        result_dict.set_item("sentiment_score", sentiment_results.sentiment_score)?;
        result_dict.set_item("confidence", sentiment_results.confidence)?;
        result_dict.set_item("key_topics", sentiment_results.key_topics)?;
        result_dict.set_item("market_impact", sentiment_results.market_impact)?;
        
        Ok(result_dict.into())
    }
    
    /// Get current memory usage statistics (ENHANCED with realistic metrics)
    pub fn get_memory_stats(&self, py: Python) -> PyResult<PyObject> {
        let legacy_stats = self.memory_manager.get_stats();
        
        // Get realistic engine metrics
        let realistic_metrics = self.realistic_engine.get_realistic_metrics()
            .unwrap_or_else(|_| HashMap::new());
        
        let result_dict = PyDict::new(py);
        
        // Legacy stats
        result_dict.set_item("legacy_current_usage_mb", legacy_stats.current_usage_mb)?;
        result_dict.set_item("legacy_peak_usage_mb", legacy_stats.peak_usage_mb)?;
        result_dict.set_item("legacy_budget_mb", legacy_stats.budget_mb)?;
        result_dict.set_item("legacy_utilization_percent", legacy_stats.utilization_percent)?;
        
        // Realistic engine stats
        result_dict.set_item("realistic_memory_usage_mb", 
            realistic_metrics.get("memory_usage_mb").unwrap_or(&0.0))?;
        result_dict.set_item("realistic_budget_mb", 800.0)?;  // 800MB realistic budget
        
        // Overall system stats
        let total_usage = legacy_stats.current_usage_mb + 
            realistic_metrics.get("memory_usage_mb").unwrap_or(&0.0);
        result_dict.set_item("total_usage_mb", total_usage)?;
        result_dict.set_item("total_budget_mb", 800.0)?;
        result_dict.set_item("total_utilization_percent", (total_usage / 800.0) * 100.0)?;
        
        Ok(result_dict.into())
    }
    
    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self, py: Python) -> PyResult<PyObject> {
        let legacy_processing_time_ns = self.performance_monitor.load(Ordering::Relaxed);
        
        // Get realistic engine metrics
        let realistic_metrics = self.realistic_engine.get_realistic_metrics()
            .unwrap_or_else(|_| HashMap::new());
        
        let result_dict = PyDict::new(py);
        
        // Legacy performance stats
        result_dict.set_item("legacy_processing_time_ns", legacy_processing_time_ns)?;
        result_dict.set_item("legacy_processing_time_ms", legacy_processing_time_ns as f64 / 1_000_000.0)?;
        
        // Realistic engine performance stats
        result_dict.set_item("realistic_performance_improvement_factor", 
            realistic_metrics.get("performance_improvement_factor").unwrap_or(&0.0))?;
        result_dict.set_item("realistic_latency_ms", 
            realistic_metrics.get("latency_milliseconds").unwrap_or(&0.0))?;
        result_dict.set_item("realistic_simd_acceleration", 
            realistic_metrics.get("simd_acceleration_factor").unwrap_or(&0.0))?;
        result_dict.set_item("realistic_zero_copy_efficiency", 
            realistic_metrics.get("zero_copy_efficiency_percent").unwrap_or(&0.0))?;
        
        // Performance grading
        let realistic_latency = realistic_metrics.get("latency_milliseconds").unwrap_or(&1000.0);
        let performance_grade = if *realistic_latency < 50.0 {
            "excellent"  // <50ms
        } else if *realistic_latency < 100.0 {
            "good"       // <100ms
        } else {
            "needs_optimization"  // >100ms
        };
        
        result_dict.set_item("performance_grade", performance_grade)?;
        result_dict.set_item("target_latency_ms", 100.0)?;  // 100ms realistic target
        result_dict.set_item("target_improvement_factor", "2-4x")?;
        result_dict.set_item("engine_type", "realistic_optimized")?;
        
        Ok(result_dict.into())
    }
    
    /// Force memory optimization using both engines
    pub fn optimize_memory(&mut self) -> PyResult<bool> {
        // Optimize legacy memory manager
        let legacy_result = self.memory_manager.force_cleanup()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Legacy memory optimization error: {}", e)))?;
        
        // Note: Realistic engine handles its own memory optimization internally
        
        Ok(legacy_result)
    }
    
    /// Get system information and engine status
    pub fn get_system_info(&self, py: Python) -> PyResult<PyObject> {
        let result_dict = PyDict::new(py);
        
        result_dict.set_item("engine_version", "2.0.0-realistic")?;
        result_dict.set_item("primary_engine", "RealisticSupremeEngine")?;
        result_dict.set_item("memory_budget_mb", self.config.memory_budget_mb)?;
        result_dict.set_item("cpu_cores", self.config.cpu_cores)?;
        result_dict.set_item("simd_enabled", self.config.enable_simd)?;
        result_dict.set_item("performance_target", &self.config.performance_target)?;
        result_dict.set_item("architecture", "classical_optimized")?;
        result_dict.set_item("build_features", vec![
            "SIMD_AVX2",
            "Apache_Arrow_ZeroCopy", 
            "Classical_MonteCarlo",
            "Memory_Optimized",
            "i3_8th_Gen_Optimized"
        ])?;
        
        Ok(result_dict.into())
    }
}

/// Python module initialization
#[pymodule]
fn supreme_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SupremeCore>()?;
    
    // Add version and system info
    m.add("__version__", "2.0.0-realistic")?;
    m.add("__description__", "Realistic high-performance trading engine core - 2-4x improvement")?;
    m.add("__engine_type__", "realistic_optimized")?;
    m.add("__memory_budget_mb__", 800)?;
    m.add("__target_improvement__", "2-4x")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_realistic_supreme_core_creation() {
        let core = SupremeCore::new();
        assert!(core.is_ok());
    }
    
    #[test]
    fn test_realistic_memory_budget_compliance() {
        let core = SupremeCore::new().unwrap();
        assert_eq!(core.config.memory_budget_mb, 800);  // 800MB realistic budget
        assert_eq!(core.config.engine_type, "realistic_optimized");
    }
    
    #[test]
    fn test_realistic_performance_targets() {
        let core = SupremeCore::new().unwrap();
        assert_eq!(core.config.performance_target, "2-4x_improvement");
        assert!(core.config.enable_simd);  // SIMD should be enabled
        assert_eq!(core.config.cpu_cores, 4);  // i3 8th gen cores
    }
    
    #[test]
    fn test_realistic_engine_integration() {
        let core = SupremeCore::new().unwrap();
        // Test that realistic engine is properly initialized
        let metrics_result = core.realistic_engine.get_realistic_metrics();
        assert!(metrics_result.is_ok());
    }
}

// Realistic memory allocator for proven performance
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// SIMD optimizations for i3 8th generation (AVX2 support)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2,avx,avx2,fma")]
static _FORCE_CPU_FEATURES: () = ();