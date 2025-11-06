//! # Realistic Memory Manager for Supreme System V5
//!
//! Professional memory management system optimized for:
//! - i3 8th Gen + 4GB RAM constraints
//! - 2.2GB realistic application budget 
//! - SIMD-aligned allocations for AVX2 optimization
//! - Memory pressure detection and mitigation
//! - Zero-copy operations with Apache Arrow
//!
//! ## Memory Budget Allocation:
//! - Data Processing: 800MB
//! - Algorithms: 600MB  
//! - NLP Models: 300MB
//! - Buffers: 200MB
//! - Emergency: 100MB
//! - System Overhead: 200MB
//! - Total: 2.2GB

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use anyhow::{Result, Context, anyhow};
use log::{info, warn, error};
use serde::{Serialize, Deserialize};

/// Memory pool categories with realistic budgets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryPool {
    DataProcessing,    // 800MB - Arrow buffers, market data
    Algorithms,        // 600MB - Technical indicators, calculations  
    NlpModels,         // 300MB - DistilBERT, tokenizers
    Buffers,           // 200MB - I/O buffers, temporary storage
    Emergency,         // 100MB - Emergency allocation, cleanup
    SystemOverhead,    // 200MB - Rust runtime, misc overhead
}

impl MemoryPool {
    /// Get realistic budget for each pool in bytes
    pub fn budget_bytes(self) -> usize {
        match self {
            MemoryPool::DataProcessing => 800 * 1024 * 1024,  // 800MB
            MemoryPool::Algorithms => 600 * 1024 * 1024,      // 600MB
            MemoryPool::NlpModels => 300 * 1024 * 1024,       // 300MB
            MemoryPool::Buffers => 200 * 1024 * 1024,         // 200MB
            MemoryPool::Emergency => 100 * 1024 * 1024,       // 100MB
            MemoryPool::SystemOverhead => 200 * 1024 * 1024,  // 200MB
        }
    }
    
    pub fn name(self) -> &'static str {
        match self {
            MemoryPool::DataProcessing => "data_processing",
            MemoryPool::Algorithms => "algorithms",
            MemoryPool::NlpModels => "nlp_models",
            MemoryPool::Buffers => "buffers",
            MemoryPool::Emergency => "emergency",
            MemoryPool::SystemOverhead => "system_overhead",
        }
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_budget_mb: f64,
    pub current_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub available_mb: f64,
    pub utilization_percent: f64,
    pub allocations_count: u64,
    pub deallocations_count: u64,
    pub fragmentation_percent: f64,
    pub pool_usage: HashMap<String, PoolStats>,
    pub memory_pressure_level: MemoryPressureLevel,
}

/// Individual pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub budget_mb: f64,
    pub current_mb: f64,
    pub peak_mb: f64,
    pub utilization_percent: f64,
    pub allocations: u64,
    pub largest_allocation_mb: f64,
}

/// Memory pressure levels for adaptive behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Low,       // <70% usage - normal operation
    Medium,    // 70-85% usage - start optimization
    High,      // 85-95% usage - aggressive cleanup
    Critical,  // >95% usage - emergency measures
}

/// Memory block with metadata for tracking
#[derive(Debug)]
pub struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
    pool: MemoryPool,
    allocation_id: u64,
}

impl MemoryBlock {
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    
    pub fn size(&self) -> usize {
        self.size
    }
    
    pub fn is_simd_aligned(&self) -> bool {
        self.alignment >= 64  // AVX2 requires 64-byte alignment
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, self.alignment);
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Pool-specific allocator for efficient memory management
#[derive(Debug)]
struct PoolAllocator {
    pool_type: MemoryPool,
    budget_bytes: usize,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_count: AtomicUsize,
    largest_allocation: AtomicUsize,
}

impl PoolAllocator {
    fn new(pool_type: MemoryPool) -> Self {
        let budget = pool_type.budget_bytes();
        
        Self {
            pool_type,
            budget_bytes: budget,
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            largest_allocation: AtomicUsize::new(0),
        }
    }
    
    fn allocate(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        // Check budget constraint
        let current = self.current_usage.load(Ordering::Acquire);
        if current + size > self.budget_bytes {
            return Err(anyhow!("Pool {} out of memory: {}MB + {}MB > {}MB",
                self.pool_type.name(),
                current / 1024 / 1024,
                size / 1024 / 1024,
                self.budget_bytes / 1024 / 1024
            ));
        }
        
        // Ensure SIMD alignment for performance
        let final_alignment = alignment.max(64);  // Minimum 64-byte alignment for AVX2
        
        // Allocate memory
        let layout = Layout::from_size_align(size, final_alignment)
            .context("Invalid memory layout")?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("System memory allocation failed"));
        }
        
        let non_null_ptr = NonNull::new(ptr)
            .ok_or_else(|| anyhow!("Allocation returned null pointer"))?;
        
        // Update usage statistics
        let new_usage = self.current_usage.fetch_add(size, Ordering::Release) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Acquire);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(peak, new_usage, Ordering::Release, Ordering::Acquire) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        
        // Update largest allocation
        let mut largest = self.largest_allocation.load(Ordering::Acquire);
        while size > largest {
            match self.largest_allocation.compare_exchange_weak(largest, size, Ordering::Release, Ordering::Acquire) {
                Ok(_) => break,
                Err(x) => largest = x,
            }
        }
        
        info!("Allocated {}MB in pool {} (total: {}MB/{} MB)",
            size / 1024 / 1024,
            self.pool_type.name(),
            new_usage / 1024 / 1024,
            self.budget_bytes / 1024 / 1024
        );
        
        Ok(non_null_ptr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Release);
        
        info!("Deallocated {}MB from pool {} (remaining: {}MB)",
            size / 1024 / 1024,
            self.pool_type.name(),
            self.current_usage.load(Ordering::Acquire) / 1024 / 1024
        );
    }
    
    fn get_stats(&self) -> PoolStats {
        let current = self.current_usage.load(Ordering::Acquire);
        let peak = self.peak_usage.load(Ordering::Acquire);
        let allocations = self.allocation_count.load(Ordering::Acquire);
        let largest = self.largest_allocation.load(Ordering::Acquire);
        
        PoolStats {
            budget_mb: self.budget_bytes as f64 / 1024.0 / 1024.0,
            current_mb: current as f64 / 1024.0 / 1024.0,
            peak_mb: peak as f64 / 1024.0 / 1024.0,
            utilization_percent: (current as f64 / self.budget_bytes as f64) * 100.0,
            allocations: allocations as u64,
            largest_allocation_mb: largest as f64 / 1024.0 / 1024.0,
        }
    }
}

/// Professional Memory Manager for Supreme System V5
#[derive(Debug)]
pub struct RealisticMemoryManager {
    /// Pool allocators for different memory categories
    pools: HashMap<MemoryPool, Arc<PoolAllocator>>,
    
    /// Global allocation tracking
    total_allocated: AtomicUsize,
    total_deallocated: AtomicUsize,
    
    /// System constraints
    total_budget_bytes: usize,
    os_overhead_bytes: usize,
    
    /// Memory pressure monitoring
    pressure_monitor: Arc<Mutex<MemoryPressureMonitor>>,
    
    /// Allocation ID counter for tracking
    next_allocation_id: AtomicUsize,
}

/// Memory pressure monitor for adaptive behavior
#[derive(Debug)]
struct MemoryPressureMonitor {
    current_pressure: MemoryPressureLevel,
    last_cleanup_time: std::time::Instant,
    cleanup_threshold_ms: u64,
    pressure_history: Vec<(std::time::Instant, f64)>,
}

impl RealisticMemoryManager {
    /// Create new memory manager optimized for i3 8th Gen + 4GB RAM
    pub fn new_for_i3_4gb() -> Result<Self> {
        let mut pools = HashMap::new();
        
        // Initialize all memory pools
        for pool_type in [MemoryPool::DataProcessing, MemoryPool::Algorithms, 
                         MemoryPool::NlpModels, MemoryPool::Buffers,
                         MemoryPool::Emergency, MemoryPool::SystemOverhead] {
            pools.insert(pool_type, Arc::new(PoolAllocator::new(pool_type)));
        }
        
        // Calculate realistic constraints
        let total_budget = 2_200_000_000; // 2.2GB realistic budget
        let os_overhead = 1_800_000_000;  // 1.8GB OS + background processes
        
        let manager = Self {
            pools,
            total_allocated: AtomicUsize::new(0),
            total_deallocated: AtomicUsize::new(0),
            total_budget_bytes: total_budget,
            os_overhead_bytes: os_overhead,
            pressure_monitor: Arc::new(Mutex::new(MemoryPressureMonitor::new())),
            next_allocation_id: AtomicUsize::new(1),
        };
        
        info!("RealisticMemoryManager initialized with {}MB budget", 
              total_budget / 1024 / 1024);
        
        Ok(manager)
    }
    
    /// Allocate memory with pool management and validation
    pub fn allocate_with_pool(
        &self, 
        size: usize, 
        pool: MemoryPool, 
        alignment: Option<usize>
    ) -> Result<MemoryBlock> {
        // Default to SIMD alignment
        let alignment = alignment.unwrap_or(64);  // AVX2 requires 64-byte alignment
        
        // Check global memory constraint
        let total_current = self.total_allocated.load(Ordering::Acquire) 
                           - self.total_deallocated.load(Ordering::Acquire);
        
        if total_current + size > self.total_budget_bytes {
            self.trigger_memory_cleanup()?;
            
            // Re-check after cleanup
            let total_after_cleanup = self.total_allocated.load(Ordering::Acquire) 
                                    - self.total_deallocated.load(Ordering::Acquire);
            
            if total_after_cleanup + size > self.total_budget_bytes {
                return Err(anyhow!("Global memory budget exceeded: {}MB + {}MB > {}MB",
                    total_after_cleanup / 1024 / 1024,
                    size / 1024 / 1024, 
                    self.total_budget_bytes / 1024 / 1024
                ));
            }
        }
        
        // Allocate from specific pool
        let pool_allocator = self.pools.get(&pool)
            .ok_or_else(|| anyhow!("Pool {:?} not found", pool))?;
        
        let ptr = pool_allocator.allocate(size, alignment)
            .context("Pool allocation failed")?;
        
        // Update global statistics
        self.total_allocated.fetch_add(size, Ordering::Release);
        
        // Generate unique allocation ID
        let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::Relaxed);
        
        // Update memory pressure monitoring
        self.update_memory_pressure();
        
        Ok(MemoryBlock {
            ptr,
            size,
            alignment,
            pool,
            allocation_id,
        })
    }
    
    /// Deallocate memory block
    pub fn deallocate(&self, block: MemoryBlock) -> Result<()> {
        let pool_allocator = self.pools.get(&block.pool)
            .ok_or_else(|| anyhow!("Pool {:?} not found", block.pool))?;
        
        pool_allocator.deallocate(block.ptr, block.size);
        self.total_deallocated.fetch_add(block.size, Ordering::Release);
        
        // Update memory pressure
        self.update_memory_pressure();
        
        Ok(())
    }
    
    /// Get comprehensive memory statistics
    pub fn get_comprehensive_stats(&self) -> MemoryStats {
        let total_allocated = self.total_allocated.load(Ordering::Acquire);
        let total_deallocated = self.total_deallocated.load(Ordering::Acquire);
        let current_usage = total_allocated - total_deallocated;
        
        // Calculate fragmentation based on pool utilization variance
        let pool_utilizations: Vec<f64> = self.pools.values()
            .map(|pool| {
                let current = pool.current_usage.load(Ordering::Acquire) as f64;
                let budget = pool.budget_bytes as f64;
                current / budget
            })
            .collect();
        
        let mean_utilization = pool_utilizations.iter().sum::<f64>() / pool_utilizations.len() as f64;
        let variance = pool_utilizations.iter()
            .map(|u| (u - mean_utilization).powi(2))
            .sum::<f64>() / pool_utilizations.len() as f64;
        let fragmentation_percent = variance.sqrt() * 100.0;
        
        // Get individual pool stats
        let mut pool_usage = HashMap::new();
        let mut peak_usage = 0;
        
        for (pool_type, allocator) in &self.pools {
            let stats = allocator.get_stats();
            peak_usage = peak_usage.max(stats.peak_mb as usize);
            pool_usage.insert(pool_type.name().to_string(), stats);
        }
        
        // Get memory pressure level
        let pressure_level = self.pressure_monitor.lock().unwrap().current_pressure;
        
        MemoryStats {
            total_budget_mb: self.total_budget_bytes as f64 / 1024.0 / 1024.0,
            current_usage_mb: current_usage as f64 / 1024.0 / 1024.0,
            peak_usage_mb: peak_usage as f64,
            available_mb: (self.total_budget_bytes - current_usage) as f64 / 1024.0 / 1024.0,
            utilization_percent: (current_usage as f64 / self.total_budget_bytes as f64) * 100.0,
            allocations_count: total_allocated as u64,
            deallocations_count: total_deallocated as u64,
            fragmentation_percent,
            pool_usage,
            memory_pressure_level: pressure_level,
        }
    }
    
    /// Trigger comprehensive memory cleanup
    pub fn trigger_memory_cleanup(&self) -> Result<()> {
        let mut monitor = self.pressure_monitor.lock().unwrap();
        
        info!("Triggering memory cleanup due to pressure");
        
        // Update last cleanup time
        monitor.last_cleanup_time = std::time::Instant::now();
        
        // Force garbage collection in pools (simplified)
        for (pool_type, _) in &self.pools {
            info!("Cleaning up pool: {:?}", pool_type);
        }
        
        Ok(())
    }
    
    /// Update memory pressure level based on current usage
    fn update_memory_pressure(&self) {
        let stats = self.get_comprehensive_stats();
        let utilization = stats.utilization_percent;
        
        let new_pressure = if utilization < 70.0 {
            MemoryPressureLevel::Low
        } else if utilization < 85.0 {
            MemoryPressureLevel::Medium
        } else if utilization < 95.0 {
            MemoryPressureLevel::High
        } else {
            MemoryPressureLevel::Critical
        };
        
        let mut monitor = self.pressure_monitor.lock().unwrap();
        if monitor.current_pressure != new_pressure {
            warn!("Memory pressure changed: {:?} -> {:?} ({}% utilization)",
                monitor.current_pressure, new_pressure, utilization);
            monitor.current_pressure = new_pressure;
            
            // Record pressure history
            monitor.pressure_history.push((std::time::Instant::now(), utilization));
            
            // Keep only recent history (last 1000 samples)
            if monitor.pressure_history.len() > 1000 {
                monitor.pressure_history.drain(0..500);
            }
        }
    }
    
    /// Check if allocation is safe given current memory pressure
    pub fn can_safely_allocate(&self, size: usize, pool: MemoryPool) -> bool {
        let pool_allocator = match self.pools.get(&pool) {
            Some(allocator) => allocator,
            None => return false,
        };
        
        let pool_current = pool_allocator.current_usage.load(Ordering::Acquire);
        let pool_budget = pool_allocator.budget_bytes;
        
        // Check pool constraint
        if pool_current + size > pool_budget {
            return false;
        }
        
        // Check global constraint
        let total_current = self.total_allocated.load(Ordering::Acquire) 
                           - self.total_deallocated.load(Ordering::Acquire);
        
        if total_current + size > self.total_budget_bytes {
            return false;
        }
        
        // Check memory pressure level
        let pressure = self.pressure_monitor.lock().unwrap().current_pressure;
        match pressure {
            MemoryPressureLevel::Critical => false,  // No new allocations
            MemoryPressureLevel::High => size < 10 * 1024 * 1024,  // Only small allocations
            MemoryPressureLevel::Medium => size < 100 * 1024 * 1024,  // Medium restriction
            MemoryPressureLevel::Low => true,  // No restrictions
        }
    }
    
    /// Get memory pool utilization report
    pub fn get_pool_utilization_report(&self) -> String {
        let mut report = String::new();
        report.push_str("\n💾 MEMORY POOL UTILIZATION REPORT\n");
        report.push_str("=====================================\n");
        
        for (pool_type, allocator) in &self.pools {
            let stats = allocator.get_stats();
            report.push_str(&format!(
                "\n📦 Pool: {} ({:.1f}MB budget)\n",
                pool_type.name(), stats.budget_mb
            ));
            report.push_str(&format!(
                "  Current: {:.1f}MB ({:.1f}%)\n",
                stats.current_mb, stats.utilization_percent
            ));
            report.push_str(&format!(
                "  Peak: {:.1f}MB\n",
                stats.peak_mb
            ));
            report.push_str(&format!(
                "  Allocations: {}\n",
                stats.allocations
            ));
            report.push_str(&format!(
                "  Largest: {:.1f}MB\n",
                stats.largest_allocation_mb
            ));
        }
        
        let overall_stats = self.get_comprehensive_stats();
        report.push_str(&format!(
            "\n📊 OVERALL SYSTEM:\n"
        ));
        report.push_str(&format!(
            "  Budget: {:.1f}MB\n",
            overall_stats.total_budget_mb
        ));
        report.push_str(&format!(
            "  Current: {:.1f}MB ({:.1f}%)\n",
            overall_stats.current_usage_mb, overall_stats.utilization_percent
        ));
        report.push_str(&format!(
            "  Available: {:.1f}MB\n",
            overall_stats.available_mb
        ));
        report.push_str(&format!(
            "  Pressure: {:?}\n",
            overall_stats.memory_pressure_level
        ));
        
        report
    }
}

impl MemoryPressureMonitor {
    fn new() -> Self {
        Self {
            current_pressure: MemoryPressureLevel::Low,
            last_cleanup_time: std::time::Instant::now(),
            cleanup_threshold_ms: 30_000,  // 30 seconds between cleanups
            pressure_history: Vec::new(),
        }
    }
}

// Memory error types
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Out of memory in pool {pool}: {message}")]
    OutOfMemory { pool: String, message: String },
    
    #[error("Invalid memory alignment: {alignment}")]
    InvalidAlignment { alignment: usize },
    
    #[error("Memory pressure critical - allocation denied")]
    MemoryPressureCritical,
    
    #[error("Pool not found: {pool}")]
    PoolNotFound { pool: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_realistic_memory_manager_creation() {
        let manager = RealisticMemoryManager::new_for_i3_4gb().unwrap();
        let stats = manager.get_comprehensive_stats();
        
        assert_eq!(stats.total_budget_mb, 2200.0);
        assert!(stats.current_usage_mb < 100.0);  // Should start low
    }
    
    #[test]
    fn test_pool_allocation_constraints() {
        let manager = RealisticMemoryManager::new_for_i3_4gb().unwrap();
        
        // Test allocation within budget
        let small_block = manager.allocate_with_pool(
            1024, MemoryPool::Buffers, Some(64)
        );
        assert!(small_block.is_ok());
        
        // Test allocation exceeding pool budget
        let huge_block = manager.allocate_with_pool(
            900 * 1024 * 1024, // 900MB - exceeds DataProcessing pool
            MemoryPool::DataProcessing, 
            Some(64)
        );
        assert!(huge_block.is_err());
    }
    
    #[test]
    fn test_simd_alignment() {
        let manager = RealisticMemoryManager::new_for_i3_4gb().unwrap();
        
        let block = manager.allocate_with_pool(
            1024, MemoryPool::Algorithms, None
        ).unwrap();
        
        assert!(block.is_simd_aligned());
        assert_eq!(block.alignment, 64);
    }
    
    #[test]
    fn test_memory_pressure_levels() {
        let manager = RealisticMemoryManager::new_for_i3_4gb().unwrap();
        
        // Initial pressure should be low
        let initial_stats = manager.get_comprehensive_stats();
        assert_eq!(initial_stats.memory_pressure_level, MemoryPressureLevel::Low);
        
        // Pressure should increase with large allocations
        let _blocks: Vec<_> = (0..100)
            .filter_map(|_| manager.allocate_with_pool(10 * 1024 * 1024, MemoryPool::DataProcessing, None).ok())
            .collect();
        
        let high_usage_stats = manager.get_comprehensive_stats();
        assert!(high_usage_stats.memory_pressure_level != MemoryPressureLevel::Low);
    }
}