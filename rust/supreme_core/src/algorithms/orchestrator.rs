//! # Professional Algorithm Orchestrator
//!
//! Manages execution of multiple algorithms with dependency resolution,
//! resource management, and error handling.

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use log::{info, warn};
use serde::{Serialize, Deserialize};
use crate::memory::realistic_manager::{RealisticMemoryManager, MemoryPool};
use crate::algorithms::simd_optimizer::ProfessionalSIMDOptimizer;

/// Algorithm execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Algorithm execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Algorithm execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    pub algorithm_name: String,
    pub execution_time_ms: u64,
    pub memory_used_mb: f64,
    pub result_data: Vec<f64>,
    pub quality_score: f64,
    pub simd_used: bool,
    pub status: String,
}

/// Algorithm definition for orchestrator
#[derive(Debug, Clone)]
pub struct AlgorithmDefinition {
    pub name: String,
    pub priority: Priority,
    pub memory_requirement_mb: usize,
    pub dependencies: Vec<String>,
    pub simd_optimized: bool,
    pub execution_timeout_ms: u64,
}

/// Professional algorithm orchestrator
#[derive(Debug)]
pub struct ProfessionalAlgorithmOrchestrator {
    algorithms: HashMap<String, AlgorithmDefinition>,
    execution_queue: VecDeque<String>,
    results: Arc<RwLock<HashMap<String, AlgorithmResult>>>,
    memory_manager: Arc<RealisticMemoryManager>,
    simd_optimizer: Arc<ProfessionalSIMDOptimizer>,
    performance_monitor: Arc<Mutex<OrchestratorMetrics>>,
}

/// Orchestrator performance metrics
#[derive(Debug, Default)]
pub struct OrchestratorMetrics {
    pub algorithms_executed: u32,
    pub total_execution_time_ms: u64,
    pub memory_efficiency: f64,
    pub simd_utilization: f64,
    pub error_rate: f64,
}

impl ProfessionalAlgorithmOrchestrator {
    /// Create new algorithm orchestrator
    pub fn new(
        memory_manager: Arc<RealisticMemoryManager>,
        simd_optimizer: Arc<ProfessionalSIMDOptimizer>
    ) -> Result<Self> {
        let mut orchestrator = Self {
            algorithms: HashMap::new(),
            execution_queue: VecDeque::new(),
            results: Arc::new(RwLock::new(HashMap::new())),
            memory_manager,
            simd_optimizer,
            performance_monitor: Arc::new(Mutex::new(OrchestratorMetrics::default())),
        };
        
        // Register default algorithms
        orchestrator.register_default_algorithms()?;
        
        Ok(orchestrator)
    }
    
    /// Register default financial algorithms
    fn register_default_algorithms(&mut self) -> Result<()> {
        let algorithms = vec![
            AlgorithmDefinition {
                name: "ema".to_string(),
                priority: Priority::High,
                memory_requirement_mb: 50,
                dependencies: vec![],
                simd_optimized: true,
                execution_timeout_ms: 1000,
            },
            AlgorithmDefinition {
                name: "rsi".to_string(),
                priority: Priority::High,
                memory_requirement_mb: 30,
                dependencies: vec![],
                simd_optimized: true,
                execution_timeout_ms: 800,
            },
            AlgorithmDefinition {
                name: "macd".to_string(),
                priority: Priority::Medium,
                memory_requirement_mb: 40,
                dependencies: vec!["ema".to_string()],
                simd_optimized: true,
                execution_timeout_ms: 1200,
            },
            AlgorithmDefinition {
                name: "bollinger_bands".to_string(),
                priority: Priority::Medium,
                memory_requirement_mb: 35,
                dependencies: vec![],
                simd_optimized: true,
                execution_timeout_ms: 1000,
            },
            AlgorithmDefinition {
                name: "volume_analysis".to_string(),
                priority: Priority::Low,
                memory_requirement_mb: 25,
                dependencies: vec![],
                simd_optimized: false,
                execution_timeout_ms: 500,
            },
        ];
        
        for algorithm in algorithms {
            self.algorithms.insert(algorithm.name.clone(), algorithm);
        }
        
        info!("Registered {} default algorithms", self.algorithms.len());
        Ok(())
    }
    
    /// Execute algorithms with dependency resolution
    pub async fn execute_algorithms(&self, market_data: &[f64]) -> Result<HashMap<String, AlgorithmResult>> {
        let start_time = std::time::Instant::now();
        
        info!("Executing algorithms for {} data points", market_data.len());
        
        // Build execution plan with dependency resolution
        let execution_plan = self.build_execution_plan()?;
        
        let mut results = HashMap::new();
        let mut executed_algorithms = 0u32;
        
        for algorithm_name in execution_plan {
            let algorithm = self.algorithms.get(&algorithm_name)
                .ok_or_else(|| anyhow!("Algorithm not found: {}", algorithm_name))?;
            
            // Check memory availability
            if !self.memory_manager.can_safely_allocate(
                algorithm.memory_requirement_mb * 1024 * 1024,
                MemoryPool::Algorithms
            ) {
                warn!("Skipping {} due to memory constraints", algorithm_name);
                continue;
            }
            
            // Execute algorithm
            match self.execute_single_algorithm(algorithm, market_data).await {
                Ok(result) => {
                    info!("Algorithm {} completed in {}ms", 
                          algorithm_name, result.execution_time_ms);
                    results.insert(algorithm_name.clone(), result);
                    executed_algorithms += 1;
                },
                Err(e) => {
                    error!("Algorithm {} failed: {}", algorithm_name, e);
                    // Continue with other algorithms
                }
            }
        }
        
        let total_time = start_time.elapsed();
        
        // Update orchestrator metrics
        let mut monitor = self.performance_monitor.lock().unwrap();
        monitor.algorithms_executed += executed_algorithms;
        monitor.total_execution_time_ms += total_time.as_millis() as u64;
        
        info!("Executed {}/{} algorithms in {}ms", 
              executed_algorithms, self.algorithms.len(), total_time.as_millis());
        
        Ok(results)
    }
    
    /// Build execution plan with dependency resolution
    fn build_execution_plan(&self) -> Result<Vec<String>> {
        let mut plan = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();
        
        // Topological sort for dependency resolution
        for algorithm_name in self.algorithms.keys() {
            if !visited.contains(algorithm_name) {
                self.visit_algorithm(algorithm_name, &mut plan, &mut visited, &mut visiting)?;
            }
        }
        
        // Sort by priority within dependency constraints
        plan.sort_by_key(|name| {
            self.algorithms.get(name)
                .map(|alg| std::cmp::Reverse(alg.priority))
                .unwrap_or(std::cmp::Reverse(Priority::Low))
        });
        
        Ok(plan)
    }
    
    /// Visit algorithm for dependency resolution (DFS)
    fn visit_algorithm(
        &self,
        algorithm_name: &str,
        plan: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>
    ) -> Result<()> {
        if visiting.contains(algorithm_name) {
            return Err(anyhow!("Circular dependency detected: {}", algorithm_name));
        }
        
        if visited.contains(algorithm_name) {
            return Ok(());
        }
        
        visiting.insert(algorithm_name.to_string());
        
        let algorithm = self.algorithms.get(algorithm_name)
            .ok_or_else(|| anyhow!("Algorithm not found: {}", algorithm_name))?;
        
        // Visit dependencies first
        for dependency in &algorithm.dependencies {
            self.visit_algorithm(dependency, plan, visited, visiting)?;
        }
        
        visiting.remove(algorithm_name);
        visited.insert(algorithm_name.to_string());
        plan.push(algorithm_name.to_string());
        
        Ok(())
    }
    
    /// Execute single algorithm with comprehensive monitoring
    async fn execute_single_algorithm(
        &self,
        algorithm: &AlgorithmDefinition,
        data: &[f64]
    ) -> Result<AlgorithmResult> {
        let start_time = std::time::Instant::now();
        
        // Allocate memory for algorithm
        let memory_block = self.memory_manager.allocate_with_pool(
            algorithm.memory_requirement_mb * 1024 * 1024,
            MemoryPool::Algorithms,
            Some(64) // SIMD alignment
        )?;
        
        let result_data = match algorithm.name.as_str() {
            "ema" => {
                if algorithm.simd_optimized {
                    self.simd_optimizer.calculate_ema_optimized(data, 20)?
                } else {
                    self.calculate_ema_fallback(data, 20)?
                }
            },
            "rsi" => {
                if algorithm.simd_optimized {
                    self.simd_optimizer.calculate_rsi_optimized(data, 14)?
                } else {
                    self.calculate_rsi_fallback(data, 14)?
                }
            },
            "macd" => {
                self.calculate_macd(data)?
            },
            "bollinger_bands" => {
                self.calculate_bollinger_bands(data, 20, 2.0)?
            },
            "volume_analysis" => {
                self.calculate_volume_analysis(data)?
            },
            _ => {
                return Err(anyhow!("Unknown algorithm: {}", algorithm.name));
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Calculate quality score based on execution metrics
        let quality_score = self.calculate_quality_score(
            execution_time.as_millis() as u64,
            algorithm.execution_timeout_ms,
            result_data.len()
        );
        
        Ok(AlgorithmResult {
            algorithm_name: algorithm.name.clone(),
            execution_time_ms: execution_time.as_millis() as u64,
            memory_used_mb: memory_block.size() as f64 / 1024.0 / 1024.0,
            result_data,
            quality_score,
            simd_used: algorithm.simd_optimized,
            status: "completed".to_string(),
        })
    }
    
    /// Calculate quality score for algorithm execution
    fn calculate_quality_score(
        &self,
        execution_time_ms: u64,
        timeout_ms: u64,
        result_count: usize
    ) -> f64 {
        let time_score = if execution_time_ms <= timeout_ms {
            1.0 - (execution_time_ms as f64 / timeout_ms as f64) * 0.5
        } else {
            0.5 // Penalty for timeout
        };
        
        let result_score = if result_count > 0 { 1.0 } else { 0.0 };
        
        (time_score + result_score) / 2.0
    }
    
    // Fallback algorithm implementations
    fn calculate_ema_fallback(&self, data: &[f64], window: usize) -> Result<Vec<f64>> {
        let alpha = 2.0 / (window as f64 + 1.0);
        let mut result = Vec::with_capacity(data.len());
        
        if data.is_empty() {
            return Ok(result);
        }
        
        result.push(data[0]);
        
        for i in 1..data.len() {
            let last_ema = result[i - 1];
            result.push(alpha * data[i] + (1.0 - alpha) * last_ema);
        }
        
        Ok(result)
    }
    
    fn calculate_rsi_fallback(&self, data: &[f64], window: usize) -> Result<Vec<f64>> {
        if data.len() < window + 1 {
            return Ok(vec![50.0; data.len()]);
        }
        
        let mut result = vec![50.0; window];
        
        for i in window..data.len() {
            let window_data = &data[i-window..i];
            let mut gains = 0.0;
            let mut losses = 0.0;
            
            for j in 1..window_data.len() {
                let change = window_data[j] - window_data[j-1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }
            
            let avg_gain = gains / window as f64;
            let avg_loss = losses / window as f64;
            
            let rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            
            result.push(rsi);
        }
        
        Ok(result)
    }
    
    fn calculate_macd(&self, data: &[f64]) -> Result<Vec<f64>> {
        // MACD = EMA(12) - EMA(26)
        let ema12 = self.calculate_ema_fallback(data, 12)?;
        let ema26 = self.calculate_ema_fallback(data, 26)?;
        
        let macd: Vec<f64> = ema12.iter()
            .zip(ema26.iter())
            .map(|(fast, slow)| fast - slow)
            .collect();
        
        Ok(macd)
    }
    
    fn calculate_bollinger_bands(&self, data: &[f64], window: usize, std_dev: f64) -> Result<Vec<f64>> {
        let sma = self.calculate_sma(data, window)?;
        let mut bands = Vec::new();
        
        for i in window-1..data.len() {
            let window_data = &data[i-window+1..i+1];
            let mean = window_data.iter().sum::<f64>() / window as f64;
            let variance = window_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            let std = variance.sqrt();
            
            let upper = sma[i] + std_dev * std;
            let lower = sma[i] - std_dev * std;
            
            bands.push(upper);
            bands.push(sma[i]);
            bands.push(lower);
        }
        
        Ok(bands)
    }
    
    fn calculate_sma(&self, data: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut result = Vec::new();
        
        for i in window-1..data.len() {
            let sum: f64 = data[i-window+1..i+1].iter().sum();
            result.push(sum / window as f64);
        }
        
        Ok(result)
    }
    
    fn calculate_volume_analysis(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Simple volume analysis - average, momentum, etc.
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let average = data.iter().sum::<f64>() / data.len() as f64;
        let momentum = data.last().unwrap() - data.first().unwrap();
        
        Ok(vec![average, momentum])
    }
    
    /// Get orchestrator performance report
    pub fn get_performance_report(&self) -> String {
        let monitor = self.performance_monitor.lock().unwrap();
        
        let mut report = String::new();
        report.push_str("\n📡 ALGORITHM ORCHESTRATOR REPORT\n");
        report.push_str("==================================\n");
        
        report.push_str(&format!("Algorithms Executed: {}\n", monitor.algorithms_executed));
        report.push_str(&format!("Total Execution Time: {}ms\n", monitor.total_execution_time_ms));
        report.push_str(&format!("Memory Efficiency: {:.1}%\n", monitor.memory_efficiency * 100.0));
        report.push_str(&format!("SIMD Utilization: {:.1}%\n", monitor.simd_utilization * 100.0));
        report.push_str(&format!("Error Rate: {:.2}%\n", monitor.error_rate * 100.0));
        
        report
    }
}