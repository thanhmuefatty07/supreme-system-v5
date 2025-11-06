//! Supreme Core - Safe Foundation C2a
//! Minimal PyO3 memory manager for Python integration

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass]
pub struct SafeMemoryManager {
    total_budget: usize,
    current_usage: AtomicUsize,
}

#[pymethods]
impl SafeMemoryManager {
    #[new]
    pub fn new() -> Self {
        let total_ram = 4 * 1024 * 1024 * 1024usize;
        let os_overhead = 1_800_000_000usize;
        let available = total_ram.saturating_sub(os_overhead);
        
        Self {
            total_budget: available,
            current_usage: AtomicUsize::new(0),
        }
    }
    
    pub fn allocate(&self, size: usize) -> PyResult<()> {
        let current = self.current_usage.load(Ordering::Acquire);
        if current.saturating_add(size) > self.total_budget {
            return Err(PyErr::new::<pyo3::exceptions::PyMemoryError, _>(
                "Memory budget exceeded"
            ));
        }
        
        self.current_usage.fetch_add(size, Ordering::Release);
        Ok(())
    }
    
    pub fn free(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Release);
    }
    
    pub fn get_stats(&self) -> PyResult<HashMap<String, usize>> {
        let current = self.current_usage.load(Ordering::Acquire);
        let mut stats = HashMap::new();
        stats.insert("total_budget".to_string(), self.total_budget);
        stats.insert("current_usage".to_string(), current);
        stats.insert("available".to_string(), self.total_budget.saturating_sub(current));
        Ok(stats)
    }
    
    pub fn reset(&self) {
        self.current_usage.store(0, Ordering::Release);
    }
}

#[pymodule]
fn supreme_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SafeMemoryManager>()?;
    Ok(())
}
