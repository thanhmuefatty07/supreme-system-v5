//! Realistic Memory Manager for Supreme System V5
//! Phase A Implementation: Budgets, Pooling, Pressure, Stats

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pool {
    Data,
    Algorithms,
    Nlp,
    Buffers,
    Emergency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure { Low, Medium, High, Critical }

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_budget: usize,
    pub current_usage: usize,
    pub available: usize,
    pub pressure: Pressure,
    pub pools: HashMap<Pool, (usize /*budget*/, usize /*usage*/)> ,
}

#[derive(Debug)]
pub struct RealisticMemoryManager {
    total_budget: usize,
    current_usage: AtomicUsize,
    budgets: HashMap<Pool, usize>,
    usage: HashMap<Pool, AtomicUsize>,
    stats_lock: Mutex<()>,
}

impl RealisticMemoryManager {
    pub fn new_for_4gb_system() -> Self {
        let total_ram_bytes = 4 * 1024 * 1024 * 1024usize; // 4GB
        let os_overhead = 1_800_000_000usize; // ~1.8GB realistic
        let available = total_ram_bytes.saturating_sub(os_overhead);

        let mut budgets = HashMap::new();
        budgets.insert(Pool::Data, 800_000_000);
        budgets.insert(Pool::Algorithms, 600_000_000);
        budgets.insert(Pool::Nlp, 300_000_000);
        budgets.insert(Pool::Buffers, 200_000_000);
        budgets.insert(Pool::Emergency, 100_000_000);

        let mut usage = HashMap::new();
        for (pool, _) in budgets.iter() {
            usage.insert(*pool, AtomicUsize::new(0));
        }

        Self {
            total_budget: available, // ~2.2GB
            current_usage: AtomicUsize::new(0),
            budgets,
            usage,
            stats_lock: Mutex::new(()),
        }
    }

    fn pressure_for(utilization: f64) -> Pressure {
        if utilization < 0.70 { Pressure::Low }
        else if utilization < 0.85 { Pressure::Medium }
        else if utilization < 0.95 { Pressure::High }
        else { Pressure::Critical }
    }

    pub fn try_allocate(&self, pool: Pool, size: usize) -> Result<(), String> {
        let pool_budget = *self.budgets.get(&pool).ok_or("Invalid pool")?;
        let pool_usage = self.usage.get(&pool).ok_or("Invalid pool")?;

        // Check global
        let current = self.current_usage.load(Ordering::Acquire);
        if current.saturating_add(size) > self.total_budget { return Err("Global budget exceeded".into()); }

        // Check pool
        let pool_cur = pool_usage.load(Ordering::Acquire);
        if pool_cur.saturating_add(size) > pool_budget { return Err("Pool limit exceeded".into()); }

        // Apply
        self.current_usage.fetch_add(size, Ordering::Release);
        pool_usage.fetch_add(size, Ordering::Release);
        Ok(())
    }

    pub fn free(&self, pool: Pool, size: usize) {
        if let Some(pool_usage) = self.usage.get(&pool) {
            pool_usage.fetch_sub(size, Ordering::Release);
        }
        self.current_usage.fetch_sub(size, Ordering::Release);
    }

    pub fn stats(&self) -> MemoryStats {
        let _g = self.stats_lock.lock().ok();
        let cur = self.current_usage.load(Ordering::Acquire);
        let available = self.total_budget.saturating_sub(cur);
        let utilization = if self.total_budget == 0 { 1.0 } else { cur as f64 / self.total_budget as f64 };
        let pressure = Self::pressure_for(utilization);

        let mut pools = HashMap::new();
        for (pool, budget) in self.budgets.iter() {
            let used = self.usage.get(pool).map(|a| a.load(Ordering::Acquire)).unwrap_or(0);
            pools.insert(*pool, (*budget, used));
        }
        MemoryStats { total_budget: self.total_budget, current_usage: cur, available, pressure, pools }
    }
}
