//! # Memory Fragmentation Test - 72 Hour Analysis
//!
//! Comprehensive memory fragmentation analysis for Supreme System V5
//! Tests memory allocator efficiency and fragmentation patterns over extended periods
//!
//! ## Test Objectives:
//! - Detect memory fragmentation patterns
//! - Measure allocator efficiency over 72 hours
//! - Identify memory leaks and inefficient allocation patterns
//! - Validate memory pool effectiveness
//! - Monitor memory utilization trends

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use log::{info, warn, error};
use anyhow::{Result, Context, anyhow};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::cell::RefCell;
use std::thread;

// Custom memory allocator for fragmentation tracking
#[derive(Debug)]
pub struct FragmentationTracker {
    allocations: HashMap<usize, (Layout, *mut u8)>,
    allocation_sizes: Vec<usize>,
    fragmentation_events: Vec<FragmentationEvent>,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
    deallocation_count: AtomicUsize,
    fragmentation_ratio: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationEvent {
    pub timestamp: f64,
    pub event_type: String,
    pub address: usize,
    pub size: usize,
    pub fragmentation_ratio: f64,
    pub memory_pressure: f64,
}

impl FragmentationTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            allocation_sizes: Vec::new(),
            fragmentation_events: Vec::new(),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            fragmentation_ratio: AtomicU64::new(0),
        }
    }

    pub fn record_allocation(&mut self, ptr: *mut u8, layout: Layout) {
        let addr = ptr as usize;
        let size = layout.size();

        self.allocations.insert(addr, (layout, ptr));
        self.allocation_sizes.push(size);

        let total = self.total_allocated.fetch_add(size, Ordering::SeqCst) + size;
        let peak = self.peak_allocated.load(Ordering::SeqCst);
        if total > peak {
            self.peak_allocated.store(total, Ordering::SeqCst);
        }

        self.allocation_count.fetch_add(1, Ordering::SeqCst);

        // Calculate fragmentation
        self.update_fragmentation_metrics();

        // Record significant allocations
        if size > 1024 * 1024 { // > 1MB
            self.fragmentation_events.push(FragmentationEvent {
                timestamp: self.get_timestamp(),
                event_type: "large_allocation".to_string(),
                address: addr,
                size,
                fragmentation_ratio: self.get_fragmentation_ratio(),
                memory_pressure: self.get_memory_pressure(),
            });
        }
    }

    pub fn record_deallocation(&mut self, ptr: *mut u8) {
        let addr = ptr as usize;

        if let Some((layout, _)) = self.allocations.remove(&addr) {
            let size = layout.size();
            self.total_allocated.fetch_sub(size, Ordering::SeqCst);
            self.deallocation_count.fetch_add(1, Ordering::SeqCst);

            // Remove from size tracking
            if let Some(pos) = self.allocation_sizes.iter().position(|&s| s == size) {
                self.allocation_sizes.swap_remove(pos);
            }

            self.update_fragmentation_metrics();
        }
    }

    fn update_fragmentation_metrics(&mut self) {
        let fragmentation = self.calculate_fragmentation_ratio();
        self.fragmentation_ratio.store(
            (fragmentation * 1_000_000.0) as u64,
            Ordering::SeqCst
        );
    }

    fn calculate_fragmentation_ratio(&self) -> f64 {
        if self.allocation_sizes.is_empty() {
            return 0.0;
        }

        let total_size: usize = self.allocation_sizes.iter().sum();
        let num_allocations = self.allocation_sizes.len();

        if num_allocations == 0 || total_size == 0 {
            return 0.0;
        }

        // Calculate coefficient of variation (CV) as fragmentation measure
        let mean = total_size as f64 / num_allocations as f64;
        let variance = self.allocation_sizes.iter()
            .map(|&size| {
                let diff = size as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / num_allocations as f64;

        let std_dev = variance.sqrt();

        // CV = std_dev / mean (higher values indicate more fragmentation)
        if mean > 0.0 {
            std_dev / mean
        } else {
            0.0
        }
    }

    fn get_fragmentation_ratio(&self) -> f64 {
        self.fragmentation_ratio.load(Ordering::SeqCst) as f64 / 1_000_000.0
    }

    fn get_memory_pressure(&self) -> f64 {
        let total = self.total_allocated.load(Ordering::SeqCst);
        let peak = self.peak_allocated.load(Ordering::SeqCst);

        if peak > 0 {
            total as f64 / peak as f64
        } else {
            0.0
        }
    }

    fn get_timestamp(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    pub fn get_stats(&self) -> MemoryFragmentationStats {
        MemoryFragmentationStats {
            total_allocated: self.total_allocated.load(Ordering::SeqCst),
            peak_allocated: self.peak_allocated.load(Ordering::SeqCst),
            current_allocations: self.allocations.len(),
            allocation_count: self.allocation_count.load(Ordering::SeqCst),
            deallocation_count: self.deallocation_count.load(Ordering::SeqCst),
            fragmentation_ratio: self.get_fragmentation_ratio(),
            memory_pressure: self.get_memory_pressure(),
            fragmentation_events: self.fragmentation_events.len(),
        }
    }

    pub fn get_events(&self) -> &[FragmentationEvent] {
        &self.fragmentation_events
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragmentationStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_allocations: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub fragmentation_ratio: f64,
    pub memory_pressure: f64,
    pub fragmentation_events: usize,
}

// Global fragmentation tracker
thread_local! {
    static FRAGMENTATION_TRACKER: RefCell<FragmentationTracker> = RefCell::new(FragmentationTracker::new());
}

/// Memory fragmentation analyzer
pub struct MemoryFragmentationAnalyzer {
    tracker: Arc<RwLock<FragmentationTracker>>,
    analysis_history: Arc<RwLock<Vec<FragmentationAnalysis>>>,
    test_duration: Duration,
    sampling_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    pub timestamp: f64,
    pub stats: MemoryFragmentationStats,
    pub recommendations: Vec<String>,
    pub risk_level: String,
}

impl MemoryFragmentationAnalyzer {
    pub fn new(test_duration_hours: u64, sampling_interval_seconds: u64) -> Self {
        Self {
            tracker: Arc::new(RwLock::new(FragmentationTracker::new())),
            analysis_history: Arc::new(RwLock::new(Vec::new())),
            test_duration: Duration::from_secs(test_duration_hours * 3600),
            sampling_interval: Duration::from_secs(sampling_interval_seconds),
        }
    }

    pub async fn run_fragmentation_analysis(&self) -> Result<FragmentationTestResult> {
        info!("Starting 72-hour memory fragmentation analysis");

        let start_time = Instant::now();
        let mut sample_count = 0;

        while start_time.elapsed() < self.test_duration {
            let analysis_start = Instant::now();

            // Generate memory allocation patterns
            self.generate_allocation_patterns().await?;

            // Analyze current fragmentation state
            let analysis = self.analyze_fragmentation().await?;
            {
                let mut history = self.analysis_history.write().await;
                history.push(analysis);
            }

            sample_count += 1;

            // Progress reporting
            let elapsed = start_time.elapsed();
            let progress = (elapsed.as_secs_f64() / self.test_duration.as_secs_f64()) * 100.0;

            info!(
                "Fragmentation analysis progress: {:.1}% ({:.0}h/{:.0}h) - Fragmentation: {:.4}, Memory Pressure: {:.2}",
                progress,
                elapsed.as_secs_f64() / 3600.0,
                self.test_duration.as_secs_f64() / 3600.0,
                analysis.stats.fragmentation_ratio,
                analysis.stats.memory_pressure
            );

            // Wait for next sampling interval, accounting for analysis time
            let analysis_time = analysis_start.elapsed();
            if analysis_time < self.sampling_interval {
                tokio::time::sleep(self.sampling_interval - analysis_time).await;
            }
        }

        // Generate final report
        self.generate_final_report().await
    }

    async fn generate_allocation_patterns(&self) -> Result<()> {
        // Simulate realistic memory allocation patterns
        let patterns = vec![
            AllocationPattern::SmallFrequent,
            AllocationPattern::LargeInfrequent,
            AllocationPattern::MixedLifecycle,
            AllocationPattern::TemporarySpikes,
        ];

        for pattern in patterns {
            self.execute_allocation_pattern(pattern).await?;
        }

        Ok(())
    }

    async fn execute_allocation_pattern(&self, pattern: AllocationPattern) -> Result<()> {
        match pattern {
            AllocationPattern::SmallFrequent => {
                // Many small, frequent allocations/deallocations
                for _ in 0..1000 {
                    let size = fastrand::usize(64..1024); // 64B to 1KB
                    let _allocation = self.allocate_test_memory(size).await;
                    // Small allocations are often short-lived
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
            AllocationPattern::LargeInfrequent => {
                // Few large, long-lived allocations
                for _ in 0..10 {
                    let size = fastrand::usize(1024 * 1024..10 * 1024 * 1024); // 1MB to 10MB
                    let allocation = self.allocate_test_memory(size).await;
                    // Large allocations tend to live longer
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    drop(allocation);
                }
            }
            AllocationPattern::MixedLifecycle => {
                // Mix of different sized allocations with varying lifecycles
                let mut allocations = Vec::new();

                for i in 0..100 {
                    let size = match i % 4 {
                        0 => fastrand::usize(128..512),       // Tiny
                        1 => fastrand::usize(1024..65536),    // Small
                        2 => fastrand::usize(65536..1048576), // Medium
                        3 => fastrand::usize(1048576..4194304), // Large
                        _ => 1024,
                    };

                    allocations.push(self.allocate_test_memory(size).await);

                    // Randomly deallocate some allocations
                    if fastrand::bool() && !allocations.is_empty() {
                        let index = fastrand::usize(0..allocations.len());
                        allocations.swap_remove(index);
                    }

                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            AllocationPattern::TemporarySpikes => {
                // Periodic allocation spikes followed by cleanup
                for _ in 0..5 {
                    let mut spike_allocations = Vec::new();

                    // Allocation spike
                    for _ in 0..50 {
                        let size = fastrand::usize(512..524288); // 512B to 512KB
                        spike_allocations.push(self.allocate_test_memory(size).await);
                    }

                    tokio::time::sleep(Duration::from_secs(30)).await;

                    // Cleanup spike
                    spike_allocations.clear();

                    tokio::time::sleep(Duration::from_secs(30)).await;
                }
            }
        }

        Ok(())
    }

    async fn allocate_test_memory(&self, size: usize) -> TestAllocation {
        // Allocate memory and track it
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>()).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            return TestAllocation::Failed;
        }

        // Initialize memory (simulate real usage)
        unsafe {
            std::ptr::write_bytes(ptr, 0xAA, size);
        }

        // Track allocation
        {
            let tracker = self.tracker.read().await;
            FRAGMENTATION_TRACKER.with(|t| {
                let mut tracker_mut = t.borrow_mut();
                tracker_mut.record_allocation(ptr, layout);
            });
        }

        TestAllocation::Success { ptr, layout, _data: vec![0u8; size] }
    }

    async fn analyze_fragmentation(&self) -> Result<FragmentationAnalysis> {
        let stats = {
            let tracker = self.tracker.read().await;
            FRAGMENTATION_TRACKER.with(|t| {
                t.borrow().get_stats()
            })
        };

        let recommendations = self.generate_recommendations(&stats);
        let risk_level = self.assess_risk_level(&stats);

        Ok(FragmentationAnalysis {
            timestamp: {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64()
            },
            stats,
            recommendations,
            risk_level,
        })
    }

    fn generate_recommendations(&self, stats: &MemoryFragmentationStats) -> Vec<String> {
        let mut recommendations = Vec::new();

        if stats.fragmentation_ratio > 0.8 {
            recommendations.push("High fragmentation detected. Consider implementing memory pools for frequently allocated sizes.".to_string());
        }

        if stats.memory_pressure > 0.9 {
            recommendations.push("High memory pressure detected. Consider increasing memory limits or optimizing allocation patterns.".to_string());
        }

        if stats.current_allocations > 10000 {
            recommendations.push("Large number of active allocations. Review object lifecycle management.".to_string());
        }

        if stats.fragmentation_events > 100 {
            recommendations.push("Frequent large allocations causing fragmentation. Consider pre-allocating large objects.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Memory fragmentation is within acceptable limits.".to_string());
        }

        recommendations
    }

    fn assess_risk_level(&self, stats: &MemoryFragmentationStats) -> String {
        let score = (stats.fragmentation_ratio * 0.4) +
                   (stats.memory_pressure * 0.3) +
                   ((stats.current_allocations as f64 / 10000.0).min(1.0) * 0.3);

        match score {
            s if s < 0.3 => "LOW",
            s if s < 0.6 => "MEDIUM",
            s if s < 0.8 => "HIGH",
            _ => "CRITICAL",
        }.to_string()
    }

    async fn generate_final_report(&self) -> Result<FragmentationTestResult> {
        let history = self.analysis_history.read().await;
        let final_stats = if !history.is_empty() {
            history.last().unwrap().clone()
        } else {
            return Err(anyhow!("No analysis data collected"));
        };

        let trend_analysis = self.analyze_trends(&history);

        Ok(FragmentationTestResult {
            test_duration_hours: self.test_duration.as_secs_f64() / 3600.0,
            total_samples: history.len(),
            final_analysis: final_stats,
            trend_analysis,
            recommendations: self.generate_final_recommendations(&history),
        })
    }

    fn analyze_trends(&self, history: &[FragmentationAnalysis]) -> FragmentationTrends {
        if history.is_empty() {
            return FragmentationTrends::default();
        }

        let fragmentation_ratios: Vec<f64> = history.iter().map(|a| a.stats.fragmentation_ratio).collect();
        let memory_pressures: Vec<f64> = history.iter().map(|a| a.stats.memory_pressure).collect();

        FragmentationTrends {
            fragmentation_trend: self.calculate_trend(&fragmentation_ratios),
            memory_pressure_trend: self.calculate_trend(&memory_pressures),
            peak_fragmentation: fragmentation_ratios.iter().fold(0.0, |a, &b| a.max(b)),
            average_fragmentation: fragmentation_ratios.iter().sum::<f64>() / fragmentation_ratios.len() as f64,
            fragmentation_volatility: self.calculate_volatility(&fragmentation_ratios),
        }
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x_squared_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        slope
    }

    fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn generate_final_recommendations(&self, history: &[FragmentationAnalysis]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(latest) = history.last() {
            if latest.stats.fragmentation_ratio > 0.7 {
                recommendations.push("Implement memory pool allocator to reduce fragmentation".to_string());
            }

            if latest.stats.memory_pressure > 0.85 {
                recommendations.push("Increase available memory or optimize memory usage patterns".to_string());
            }

            let critical_periods = history.iter().filter(|a| a.risk_level == "CRITICAL").count();
            if critical_periods > history.len() / 10 {
                recommendations.push("Critical fragmentation periods detected. Immediate memory optimization required".to_string());
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Memory fragmentation analysis completed successfully. No major issues detected.".to_string());
        }

        recommendations
    }
}

#[derive(Debug)]
enum AllocationPattern {
    SmallFrequent,
    LargeInfrequent,
    MixedLifecycle,
    TemporarySpikes,
}

#[derive(Debug)]
enum TestAllocation {
    Success { ptr: *mut u8, layout: Layout, _data: Vec<u8> },
    Failed,
}

impl Drop for TestAllocation {
    fn drop(&mut self) {
        if let TestAllocation::Success { ptr, layout, .. } = self {
            unsafe {
                std::alloc::dealloc(*ptr, *layout);
            }

            // Track deallocation
            FRAGMENTATION_TRACKER.with(|t| {
                let mut tracker = t.borrow_mut();
                tracker.record_deallocation(*ptr);
            });
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationTestResult {
    pub test_duration_hours: f64,
    pub total_samples: usize,
    pub final_analysis: FragmentationAnalysis,
    pub trend_analysis: FragmentationTrends,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FragmentationTrends {
    pub fragmentation_trend: f64,
    pub memory_pressure_trend: f64,
    pub peak_fragmentation: f64,
    pub average_fragmentation: f64,
    pub fragmentation_volatility: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_fragmentation_tracking() {
        let analyzer = MemoryFragmentationAnalyzer::new(1, 1); // 1 hour, 1 second intervals

        // This test would run for a very short time in CI
        // In production, this would run for 72 hours
        let start_time = Instant::now();
        let max_test_duration = Duration::from_secs(10); // Short test for CI

        while start_time.elapsed() < max_test_duration {
            // Generate some allocation patterns
            analyzer.generate_allocation_patterns().await.unwrap();

            // Analyze fragmentation
            let analysis = analyzer.analyze_fragmentation().await.unwrap();

            // Basic validation
            assert!(analysis.stats.fragmentation_ratio >= 0.0);
            assert!(analysis.stats.memory_pressure >= 0.0);
            assert!(!analysis.risk_level.is_empty());

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Generate report
        let result = analyzer.generate_final_report().await.unwrap();

        assert!(result.test_duration_hours > 0.0);
        assert!(result.total_samples > 0);
        assert!(!result.recommendations.is_empty());

        println!("Memory fragmentation test completed successfully");
        println!("Final fragmentation ratio: {:.4}", result.final_analysis.stats.fragmentation_ratio);
        println!("Risk level: {}", result.final_analysis.risk_level);
        println!("Recommendations: {:?}", result.recommendations);
    }

    #[tokio::test]
    async fn test_fragmentation_pattern_analysis() {
        let analyzer = MemoryFragmentationAnalyzer::new(1, 1);

        // Test small frequent allocations
        analyzer.execute_allocation_pattern(AllocationPattern::SmallFrequent).await.unwrap();

        let analysis = analyzer.analyze_fragmentation().await.unwrap();
        assert!(analysis.stats.allocation_count > 0);

        // Test mixed lifecycle pattern
        analyzer.execute_allocation_pattern(AllocationPattern::MixedLifecycle).await.unwrap();

        let analysis2 = analyzer.analyze_fragmentation().await.unwrap();
        assert!(analysis2.stats.allocation_count >= analysis.stats.allocation_count);

        println!("Fragmentation pattern analysis test passed");
    }
}

#[tokio::test]
async fn memory_fragmentation_72h() {
    println!("🧪 Starting 72-Hour Memory Fragmentation Analysis");
    println!("This test analyzes memory fragmentation patterns over extended periods");
    println!("In CI environment, this will run for a limited time. In production, run for 72 hours.");

    // For CI, use short duration. In production, use 72 hours.
    let test_duration_hours = if std::env::var("CI").is_ok() { 1 } else { 72 };
    let sampling_interval_seconds = 300; // 5 minutes in production

    let analyzer = MemoryFragmentationAnalyzer::new(test_duration_hours, sampling_interval_seconds);

    match analyzer.run_fragmentation_analysis().await {
        Ok(result) => {
            println!("✅ Memory fragmentation analysis completed");
            println!("📊 Test Duration: {:.1} hours", result.test_duration_hours);
            println!("📈 Total Samples: {}", result.total_samples);
            println!("🔍 Final Fragmentation Ratio: {:.4}", result.final_analysis.stats.fragmentation_ratio);
            println!("💾 Final Memory Pressure: {:.2}", result.final_analysis.stats.memory_pressure);
            println!("⚠️  Risk Level: {}", result.final_analysis.risk_level);
            println!("📈 Peak Fragmentation: {:.4}", result.trend_analysis.peak_fragmentation);
            println!("📊 Average Fragmentation: {:.4}", result.trend_analysis.average_fragmentation);
            println!("📉 Fragmentation Trend: {:.6}", result.trend_analysis.fragmentation_trend);
            println!("🔄 Fragmentation Volatility: {:.4}", result.trend_analysis.fragmentation_volatility);

            println!("💡 Recommendations:");
            for rec in &result.recommendations {
                println!("   • {}", rec);
            }

            // Success criteria
            let fragmentation_acceptable = result.final_analysis.stats.fragmentation_ratio < 0.8;
            let memory_pressure_acceptable = result.final_analysis.stats.memory_pressure < 0.9;
            let risk_acceptable = result.final_analysis.risk_level != "CRITICAL";

            if fragmentation_acceptable && memory_pressure_acceptable && risk_acceptable {
                println!("✅ Memory fragmentation analysis PASSED");
            } else {
                println!("❌ Memory fragmentation analysis FAILED");
                if !fragmentation_acceptable {
                    println!("   - Fragmentation ratio too high: {:.4}", result.final_analysis.stats.fragmentation_ratio);
                }
                if !memory_pressure_acceptable {
                    println!("   - Memory pressure too high: {:.2}", result.final_analysis.stats.memory_pressure);
                }
                if !risk_acceptable {
                    println!("   - Risk level critical: {}", result.final_analysis.risk_level);
                }
            }

            // Assert success criteria
            assert!(fragmentation_acceptable, "Fragmentation ratio too high");
            assert!(memory_pressure_acceptable, "Memory pressure too high");
            assert!(risk_acceptable, "Risk level is critical");
        }
        Err(e) => {
            panic!("Memory fragmentation analysis failed: {}", e);
        }
    }
}
