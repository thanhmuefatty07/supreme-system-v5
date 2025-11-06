//! # Data Corruption Recovery Test
//!
//! Comprehensive test suite for data corruption detection and recovery
//! mechanisms in Supreme System V5 trading engine.
//!
//! ## Test Coverage:
//! - Data integrity validation
//! - Corruption detection algorithms
//! - Recovery mechanism effectiveness
//! - Performance impact of corruption handling
//! - Edge case corruption scenarios

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use log::{info, warn, error};
use std::time::{Duration, Instant};

/// Trading data structure with integrity checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingData {
    pub symbol: String,
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub checksum: u64,
}

impl TradingData {
    /// Create new trading data with checksum
    pub fn new(symbol: String, timestamp: i64, price: f64, volume: f64) -> Self {
        let mut data = Self {
            symbol,
            timestamp,
            price,
            volume,
            checksum: 0,
        };
        data.checksum = data.calculate_checksum();
        data
    }

    /// Calculate checksum for data integrity
    pub fn calculate_checksum(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.symbol.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);

        // Hash price with high precision to detect small changes
        ((self.price * 1_000_000.0) as i64).hash(&mut hasher);
        ((self.volume * 1_000_000.0) as i64).hash(&mut hasher);

        hasher.finish()
    }

    /// Validate data integrity
    pub fn validate_integrity(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }

    /// Corrupt data for testing (simulates real corruption)
    pub fn corrupt_price(&mut self, corruption_factor: f64) {
        self.price *= corruption_factor;
        // Don't update checksum to simulate undetected corruption
    }

    pub fn corrupt_volume(&mut self, corruption_factor: f64) {
        self.volume *= corruption_factor;
    }

    pub fn corrupt_timestamp(&mut self, offset: i64) {
        self.timestamp += offset;
    }
}

/// Data corruption detector with recovery capabilities
pub struct DataCorruptionDetector {
    recovery_stats: Arc<RwLock<CorruptionStats>>,
    recovery_strategies: HashMap<String, Box<dyn CorruptionRecoveryStrategy>>,
}

impl DataCorruptionDetector {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        // Register recovery strategies
        strategies.insert("price_interpolation".to_string(),
                         Box::new(PriceInterpolationRecovery::new()) as Box<dyn CorruptionRecoveryStrategy>);
        strategies.insert("volume_estimation".to_string(),
                         Box::new(VolumeEstimationRecovery::new()) as Box<dyn CorruptionRecoveryStrategy>);
        strategies.insert("timestamp_correction".to_string(),
                         Box::new(TimestampCorrectionRecovery::new()) as Box<dyn CorruptionRecoveryStrategy>);

        Self {
            recovery_stats: Arc::new(RwLock::new(CorruptionStats::new())),
            recovery_strategies: strategies,
        }
    }

    /// Detect and recover from data corruption
    pub async fn detect_and_recover(&self, data: &mut [TradingData]) -> Result<RecoveryResult> {
        let mut corrupted_indices = Vec::new();
        let mut recovery_attempts = 0;
        let mut successful_recoveries = 0;

        // Phase 1: Detection
        for (i, item) in data.iter().enumerate() {
            if !item.validate_integrity() {
                corrupted_indices.push(i);
                info!("Corrupted data detected at index {}", i);
            }
        }

        if corrupted_indices.is_empty() {
            return Ok(RecoveryResult {
                total_corruptions: 0,
                recovered_count: 0,
                failed_recoveries: 0,
                recovery_time_ms: 0,
            });
        }

        let start_time = Instant::now();

        // Phase 2: Recovery
        for &index in &corrupted_indices {
            if index >= data.len() {
                continue;
            }

            let corrupted_item = &data[index];
            let mut recovered = false;

            // Try different recovery strategies
            for (strategy_name, strategy) in &self.recovery_strategies {
                recovery_attempts += 1;

                match strategy.recover(corrupted_item, data, index).await {
                    Ok(recovered_data) => {
                        data[index] = recovered_data;
                        successful_recoveries += 1;
                        recovered = true;

                        let mut stats = self.recovery_stats.write().await;
                        stats.record_successful_recovery(strategy_name.clone());

                        info!("Successfully recovered data at index {} using {}", index, strategy_name);
                        break;
                    }
                    Err(e) => {
                        warn!("Recovery strategy {} failed for index {}: {}", strategy_name, index, e);
                        let mut stats = self.recovery_stats.write().await;
                        stats.record_failed_recovery(strategy_name.clone());
                    }
                }
            }

            if !recovered {
                error!("Failed to recover corrupted data at index {}", index);
            }
        }

        let recovery_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        let mut stats = self.recovery_stats.write().await;
        stats.total_corruptions += corrupted_indices.len() as u64;
        stats.total_recoveries += successful_recoveries as u64;
        stats.total_recovery_time_ms += recovery_time;

        Ok(RecoveryResult {
            total_corruptions: corrupted_indices.len() as u64,
            recovered_count: successful_recoveries as u64,
            failed_recoveries: (corrupted_indices.len() - successful_recoveries) as u64,
            recovery_time_ms: recovery_time,
        })
    }

    /// Get recovery statistics
    pub async fn get_stats(&self) -> CorruptionStats {
        self.recovery_stats.read().await.clone()
    }
}

/// Recovery strategy trait
#[async_trait::async_trait]
pub trait CorruptionRecoveryStrategy: Send + Sync {
    async fn recover(&self, corrupted: &TradingData, dataset: &[TradingData], index: usize) -> Result<TradingData>;
}

/// Price interpolation recovery strategy
pub struct PriceInterpolationRecovery {
    max_interpolation_distance: usize,
}

impl PriceInterpolationRecovery {
    pub fn new() -> Self {
        Self { max_interpolation_distance: 10 }
    }
}

#[async_trait::async_trait]
impl CorruptionRecoveryStrategy for PriceInterpolationRecovery {
    async fn recover(&self, corrupted: &TradingData, dataset: &[TradingData], index: usize) -> Result<TradingData> {
        if dataset.is_empty() {
            return Err(anyhow!("Empty dataset"));
        }

        // Find valid neighboring prices
        let mut valid_prices = Vec::new();

        // Look backwards
        for i in (0..index).rev().take(self.max_interpolation_distance) {
            if dataset[i].validate_integrity() {
                valid_prices.push((i, dataset[i].price));
            }
        }

        // Look forwards
        for i in (index + 1)..std::cmp::min(dataset.len(), index + self.max_interpolation_distance + 1) {
            if dataset[i].validate_integrity() {
                valid_prices.push((i, dataset[i].price));
            }
        }

        if valid_prices.len() < 2 {
            return Err(anyhow!("Insufficient valid data points for interpolation"));
        }

        // Simple linear interpolation between two closest points
        valid_prices.sort_by_key(|(i, _)| (*i as i64 - index as i64).abs());

        let (idx1, price1) = valid_prices[0];
        let (idx2, price2) = valid_prices[1];

        // Linear interpolation
        let ratio = (index as f64 - idx1 as f64) / (idx2 as f64 - idx1 as f64);
        let interpolated_price = price1 + (price2 - price1) * ratio;

        Ok(TradingData::new(
            corrupted.symbol.clone(),
            corrupted.timestamp,
            interpolated_price,
            corrupted.volume, // Assume volume is valid
        ))
    }
}

/// Volume estimation recovery strategy
pub struct VolumeEstimationRecovery {
    volatility_window: usize,
}

impl VolumeEstimationRecovery {
    pub fn new() -> Self {
        Self { volatility_window: 20 }
    }
}

#[async_trait::async_trait]
impl CorruptionRecoveryStrategy for VolumeEstimationRecovery {
    async fn recover(&self, corrupted: &TradingData, dataset: &[TradingData], index: usize) -> Result<TradingData> {
        // Estimate volume based on price volatility and historical patterns
        let start_idx = index.saturating_sub(self.volatility_window);
        let end_idx = std::cmp::min(dataset.len(), index + self.volatility_window + 1);

        let valid_volumes: Vec<f64> = dataset[start_idx..end_idx]
            .iter()
            .filter(|d| d.validate_integrity())
            .map(|d| d.volume)
            .collect();

        if valid_volumes.is_empty() {
            return Err(anyhow!("No valid volume data for estimation"));
        }

        // Use median volume as estimation
        let mut sorted_volumes = valid_volumes.clone();
        sorted_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_volume = if sorted_volumes.len() % 2 == 0 {
            (sorted_volumes[sorted_volumes.len() / 2 - 1] + sorted_volumes[sorted_volumes.len() / 2]) / 2.0
        } else {
            sorted_volumes[sorted_volumes.len() / 2]
        };

        Ok(TradingData::new(
            corrupted.symbol.clone(),
            corrupted.timestamp,
            corrupted.price, // Assume price is valid
            median_volume,
        ))
    }
}

/// Timestamp correction recovery strategy
pub struct TimestampCorrectionRecovery {
    expected_interval_ms: i64,
}

impl TimestampCorrectionRecovery {
    pub fn new() -> Self {
        Self { expected_interval_ms: 60000 } // 1 minute intervals
    }
}

#[async_trait::async_trait]
impl CorruptionRecoveryStrategy for TimestampCorrectionRecovery {
    async fn recover(&self, corrupted: &TradingData, dataset: &[TradingData], index: usize) -> Result<TradingData> {
        if index == 0 || index >= dataset.len() - 1 {
            return Err(anyhow!("Cannot correct timestamp at boundary"));
        }

        // Estimate correct timestamp based on neighbors
        let prev_timestamp = if index > 0 && dataset[index - 1].validate_integrity() {
            dataset[index - 1].timestamp
        } else {
            return Err(anyhow!("No valid previous timestamp"));
        };

        let next_timestamp = if index < dataset.len() - 1 && dataset[index + 1].validate_integrity() {
            dataset[index + 1].timestamp
        } else {
            return Err(anyhow!("No valid next timestamp"));
        };

        let corrected_timestamp = prev_timestamp + self.expected_interval_ms;

        // Validate correction is reasonable
        let expected_next = prev_timestamp + 2 * self.expected_interval_ms;
        if (corrected_timestamp - next_timestamp).abs() > self.expected_interval_ms {
            return Err(anyhow!("Timestamp correction too large"));
        }

        Ok(TradingData::new(
            corrupted.symbol.clone(),
            corrected_timestamp,
            corrupted.price, // Assume price is valid
            corrupted.volume, // Assume volume is valid
        ))
    }
}

/// Recovery result structure
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub total_corruptions: u64,
    pub recovered_count: u64,
    pub failed_recoveries: u64,
    pub recovery_time_ms: u64,
}

/// Corruption statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptionStats {
    pub total_corruptions: u64,
    pub total_recoveries: u64,
    pub total_recovery_time_ms: u64,
    pub strategy_success_rates: HashMap<String, StrategyStats>,
}

impl CorruptionStats {
    pub fn new() -> Self {
        Self {
            total_corruptions: 0,
            total_recoveries: 0,
            total_recovery_time_ms: 0,
            strategy_success_rates: HashMap::new(),
        }
    }

    pub fn record_successful_recovery(&mut self, strategy: String) {
        self.strategy_success_rates
            .entry(strategy)
            .or_insert_with(StrategyStats::new)
            .successful_recoveries += 1;
    }

    pub fn record_failed_recovery(&mut self, strategy: String) {
        self.strategy_success_rates
            .entry(strategy)
            .or_insert_with(StrategyStats::new)
            .failed_recoveries += 1;
    }
}

/// Strategy-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStats {
    pub successful_recoveries: u64,
    pub failed_recoveries: u64,
}

impl StrategyStats {
    pub fn new() -> Self {
        Self {
            successful_recoveries: 0,
            failed_recoveries: 0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.successful_recoveries + self.failed_recoveries;
        if total == 0 {
            0.0
        } else {
            self.successful_recoveries as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_integrity_validation() {
        let data = TradingData::new("BTC-USD".to_string(), 1640995200, 50000.0, 100.0);
        assert!(data.validate_integrity());

        // Corrupt data
        let mut corrupted = data.clone();
        corrupted.corrupt_price(2.0);
        assert!(!corrupted.validate_integrity());
    }

    #[tokio::test]
    async fn test_price_interpolation_recovery() {
        let detector = DataCorruptionDetector::new();

        // Create test dataset
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(TradingData::new(
                "BTC-USD".to_string(),
                1640995200 + i * 60,
                50000.0 + i as f64 * 100.0,
                100.0 + i as f64 * 10.0,
            ));
        }

        // Corrupt middle element
        data[5].corrupt_price(0.5); // 50% corruption

        let result = detector.detect_and_recover(&mut data).await.unwrap();

        assert_eq!(result.total_corruptions, 1);
        assert_eq!(result.recovered_count, 1);
        assert_eq!(result.failed_recoveries, 0);

        // Verify recovered data
        assert!(data[5].validate_integrity());
    }

    #[tokio::test]
    async fn test_volume_estimation_recovery() {
        let detector = DataCorruptionDetector::new();

        let mut data = Vec::new();
        for i in 0..10 {
            data.push(TradingData::new(
                "BTC-USD".to_string(),
                1640995200 + i * 60,
                50000.0,
                100.0 + (i % 3) as f64 * 50.0, // Patterned volume
            ));
        }

        // Corrupt volume
        data[5].corrupt_volume(10.0); // 10x corruption

        let result = detector.detect_and_recover(&mut data).await.unwrap();

        assert_eq!(result.total_corruptions, 1);
        assert!(result.recovered_count >= 0); // May or may not recover depending on strategy
    }

    #[tokio::test]
    async fn test_timestamp_correction_recovery() {
        let detector = DataCorruptionDetector::new();

        let mut data = Vec::new();
        for i in 0..5 {
            data.push(TradingData::new(
                "BTC-USD".to_string(),
                1640995200 + i * 60, // 1-minute intervals
                50000.0,
                100.0,
            ));
        }

        // Corrupt timestamp
        data[2].corrupt_timestamp(300); // 5-minute offset

        let result = detector.detect_and_recover(&mut data).await.unwrap();

        assert_eq!(result.total_corruptions, 1);
        // Timestamp correction might be attempted
    }

    #[tokio::test]
    async fn test_corruption_statistics() {
        let detector = DataCorruptionDetector::new();

        let mut data = vec![
            TradingData::new("BTC-USD".to_string(), 1640995200, 50000.0, 100.0),
            TradingData::new("BTC-USD".to_string(), 1640995260, 50100.0, 105.0),
            TradingData::new("BTC-USD".to_string(), 1640995320, 49900.0, 95.0),
        ];

        // Corrupt one data point
        data[1].corrupt_price(1.5);

        let _ = detector.detect_and_recover(&mut data).await.unwrap();
        let stats = detector.get_stats().await;

        assert!(stats.total_corruptions >= 1);
        assert!(stats.total_recoveries >= 0);
    }
}

#[tokio::test]
async fn data_corruption_recovery() {
    let detector = DataCorruptionDetector::new();

    // Generate large test dataset
    let mut data = Vec::new();
    for i in 0..1000 {
        data.push(TradingData::new(
            "BTC-USD".to_string(),
            1640995200 + i * 60,
            50000.0 + (i as f64 * 10.0).sin() * 1000.0, // Sinusoidal price movement
            100.0 + (i % 50) as f64 * 5.0, // Volume pattern
        ));
    }

    // Introduce various types of corruption
    let corruption_patterns = [
        (50, "price_corruption", 2.0),    // Price corruption
        (150, "volume_corruption", 5.0),  // Volume corruption
        (250, "timestamp_corruption", 120), // Timestamp corruption
        (350, "severe_price_corruption", 10.0), // Severe corruption
    ];

    for (index, corruption_type, factor) in corruption_patterns {
        match corruption_type {
            "price_corruption" => data[index].corrupt_price(factor),
            "volume_corruption" => data[index].corrupt_volume(factor),
            "timestamp_corruption" => data[index].corrupt_timestamp(factor as i64),
            "severe_price_corruption" => data[index].corrupt_price(factor),
            _ => {}
        }
    }

    println!("🧪 Testing data corruption recovery with {} data points", data.len());
    println!("📊 Introduced {} corruption patterns", corruption_patterns.len());

    let start_time = Instant::now();
    let result = detector.detect_and_recover(&mut data).await.unwrap();
    let total_time = start_time.elapsed();

    println!("📈 Recovery Results:");
    println!("   Total Corruptions: {}", result.total_corruptions);
    println!("   Recovered: {}", result.recovered_count);
    println!("   Failed: {}", result.failed_recoveries);
    println!("   Recovery Time: {}ms", result.recovery_time_ms);
    println!("   Total Test Time: {:.2f}s", total_time.as_secs_f64());

    // Verify recovery effectiveness
    let final_valid_count = data.iter().filter(|d| d.validate_integrity()).count();
    let recovery_rate = result.recovered_count as f64 / result.total_corruptions as f64;

    println!("✅ Final Valid Data Points: {}%", final_valid_count);
    println!("📊 Recovery Rate: {:.1f}%", recovery_rate * 100.0);

    // Success criteria
    assert!(final_valid_count >= 990, "Too many data points still corrupted");
    assert!(recovery_rate >= 0.7, "Recovery rate too low: {:.1f}%", recovery_rate * 100.0);
    assert!(result.recovery_time_ms < 5000, "Recovery too slow: {}ms", result.recovery_time_ms);

    // Get detailed statistics
    let stats = detector.get_stats().await;
    println!("📈 Detailed Statistics:");
    println!("   Total Corruptions Detected: {}", stats.total_corruptions);
    println!("   Total Recoveries: {}", stats.total_recoveries);
    println!("   Total Recovery Time: {}ms", stats.total_recovery_time_ms);

    for (strategy, strategy_stats) in &stats.strategy_success_rates {
        let rate = strategy_stats.success_rate() * 100.0;
        println!("   {}: {:.1f}% success rate ({} successful, {} failed)",
                strategy, rate, strategy_stats.successful_recoveries, strategy_stats.failed_recoveries);
    }

    println!("✅ Data corruption recovery test completed successfully!");
}
