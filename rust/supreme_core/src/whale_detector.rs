//! Ultra-Fast Whale Transaction Detection and Wallet Analysis
//! 
//! Optimized for real-time detection of large cryptocurrency transactions
//! Target: <5ms processing time per transaction batch

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use std::sync::Arc;
use rayon::prelude::*;

use crate::SupremeConfig;

/// Transaction data structure (zero-copy optimized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub amount: f64,
    pub timestamp: u64,
    pub from_address: String,
    pub to_address: String,
}

/// Whale alert classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WhaleType {
    LargeAccumulator,     // Consistent buying pattern
    LargeDistributor,     // Consistent selling pattern
    ExchangeInflow,       // Moving to exchange (potential sell)
    ExchangeOutflow,      // Moving from exchange (potential hold)
    WalletToWallet,       // Large wallet-to-wallet transfer
    InstitutionalFlow,    // Institutional-size movements
    SuspiciousActivity,   // Unusual patterns detected
    Unknown,              // Cannot classify
}

impl std::fmt::Display for WhaleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhaleType::LargeAccumulator => write!(f, "LargeAccumulator"),
            WhaleType::LargeDistributor => write!(f, "LargeDistributor"),
            WhaleType::ExchangeInflow => write!(f, "ExchangeInflow"),
            WhaleType::ExchangeOutflow => write!(f, "ExchangeOutflow"),
            WhaleType::WalletToWallet => write!(f, "WalletToWallet"),
            WhaleType::InstitutionalFlow => write!(f, "InstitutionalFlow"),
            WhaleType::SuspiciousActivity => write!(f, "SuspiciousActivity"),
            WhaleType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Whale alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlert {
    pub amount: f64,
    pub timestamp: u64,
    pub whale_type: WhaleType,
    pub confidence: f64,
    pub addresses: Vec<String>,
    pub metadata: WhaleMetadata,
}

/// Additional metadata for whale analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhaleMetadata {
    pub transaction_count: u32,
    pub volume_24h: f64,
    pub first_seen: u64,
    pub last_seen: u64,
    pub exchange_tagged: bool,
    pub institutional_tagged: bool,
    pub risk_score: f64,
}

/// Wallet behavior pattern
#[derive(Debug, Clone)]
struct WalletBehavior {
    total_inflow: f64,
    total_outflow: f64,
    transaction_count: u32,
    first_activity: u64,
    last_activity: u64,
    counterparties: HashSet<String>,
    exchange_interactions: u32,
    average_transaction_size: f64,
}

impl WalletBehavior {
    fn new() -> Self {
        Self {
            total_inflow: 0.0,
            total_outflow: 0.0,
            transaction_count: 0,
            first_activity: u64::MAX,
            last_activity: 0,
            counterparties: HashSet::new(),
            exchange_interactions: 0,
            average_transaction_size: 0.0,
        }
    }
    
    fn update_inflow(&mut self, amount: f64, timestamp: u64, counterparty: String) {
        self.total_inflow += amount;
        self.transaction_count += 1;
        self.first_activity = self.first_activity.min(timestamp);
        self.last_activity = self.last_activity.max(timestamp);
        self.counterparties.insert(counterparty);
        self.average_transaction_size = (self.total_inflow + self.total_outflow) / self.transaction_count as f64;
    }
    
    fn update_outflow(&mut self, amount: f64, timestamp: u64, counterparty: String) {
        self.total_outflow += amount;
        self.transaction_count += 1;
        self.first_activity = self.first_activity.min(timestamp);
        self.last_activity = self.last_activity.max(timestamp);
        self.counterparties.insert(counterparty);
        self.average_transaction_size = (self.total_inflow + self.total_outflow) / self.transaction_count as f64;
    }
    
    fn net_flow(&self) -> f64 {
        self.total_inflow - self.total_outflow
    }
    
    fn activity_score(&self) -> f64 {
        let volume_score = (self.total_inflow + self.total_outflow) / 1_000_000.0;  // Normalize by 1M
        let frequency_score = self.transaction_count as f64 / 100.0;  // Normalize by 100 transactions
        let diversity_score = self.counterparties.len() as f64 / 50.0;  // Normalize by 50 counterparties
        
        (volume_score + frequency_score + diversity_score) / 3.0
    }
}

/// Known exchange addresses (simplified - in production would be comprehensive)
static KNOWN_EXCHANGES: &[&str] = &[
    "binance",
    "coinbase", 
    "kraken",
    "huobi",
    "okex",
    "bybit",
    "ftx",
    "bitfinex",
    "kucoin",
    "gate",
];

/// High-performance whale detection engine
pub struct WhaleDetector {
    config: SupremeConfig,
    wallet_behaviors: HashMap<String, WalletBehavior>,
    known_exchanges: HashSet<String>,
    suspicious_addresses: HashSet<String>,
    whale_threshold: f64,
}

impl WhaleDetector {
    pub fn new(config: &SupremeConfig) -> Self {
        let known_exchanges: HashSet<String> = KNOWN_EXCHANGES
            .iter()
            .map(|&s| s.to_string())
            .collect();
        
        Self {
            config: config.clone(),
            wallet_behaviors: HashMap::new(),
            known_exchanges,
            suspicious_addresses: HashSet::new(),
            whale_threshold: config.whale_threshold_usd,
        }
    }
    
    /// Process batch of transactions and detect whale activity
    pub fn process_transactions(&mut self, transactions: &[Transaction]) -> Result<Vec<WhaleAlert>> {
        if transactions.is_empty() {
            return Ok(Vec::new());
        }
        
        // Update wallet behaviors in parallel
        self.update_wallet_behaviors(transactions)?;
        
        // Filter transactions above whale threshold
        let whale_transactions: Vec<&Transaction> = transactions
            .par_iter()
            .filter(|tx| tx.amount >= self.whale_threshold)
            .collect();
        
        if whale_transactions.is_empty() {
            return Ok(Vec::new());
        }
        
        // Parallel whale alert generation
        let alerts: Vec<WhaleAlert> = whale_transactions
            .par_iter()
            .filter_map(|tx| self.analyze_whale_transaction(tx).ok())
            .collect();
        
        Ok(alerts)
    }
    
    /// Update wallet behavior patterns
    fn update_wallet_behaviors(&mut self, transactions: &[Transaction]) -> Result<()> {
        for tx in transactions {
            // Update sender behavior
            self.wallet_behaviors
                .entry(tx.from_address.clone())
                .or_insert_with(WalletBehavior::new)
                .update_outflow(tx.amount, tx.timestamp, tx.to_address.clone());
            
            // Update receiver behavior
            self.wallet_behaviors
                .entry(tx.to_address.clone())
                .or_insert_with(WalletBehavior::new)
                .update_inflow(tx.amount, tx.timestamp, tx.from_address.clone());
            
            // Track exchange interactions
            if self.is_exchange_address(&tx.to_address) {
                if let Some(behavior) = self.wallet_behaviors.get_mut(&tx.from_address) {
                    behavior.exchange_interactions += 1;
                }
            }
            
            if self.is_exchange_address(&tx.from_address) {
                if let Some(behavior) = self.wallet_behaviors.get_mut(&tx.to_address) {
                    behavior.exchange_interactions += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Analyze individual whale transaction
    fn analyze_whale_transaction(&self, tx: &Transaction) -> Result<WhaleAlert> {
        let from_behavior = self.wallet_behaviors.get(&tx.from_address);
        let to_behavior = self.wallet_behaviors.get(&tx.to_address);
        
        // Classify whale type based on transaction patterns
        let (whale_type, confidence) = self.classify_whale_activity(tx, from_behavior, to_behavior);
        
        // Generate metadata
        let metadata = self.generate_whale_metadata(tx, from_behavior, to_behavior);
        
        Ok(WhaleAlert {
            amount: tx.amount,
            timestamp: tx.timestamp,
            whale_type,
            confidence,
            addresses: vec![tx.from_address.clone(), tx.to_address.clone()],
            metadata,
        })
    }
    
    /// Classify whale activity type with confidence scoring
    fn classify_whale_activity(
        &self,
        tx: &Transaction,
        from_behavior: Option<&WalletBehavior>,
        to_behavior: Option<&WalletBehavior>,
    ) -> (WhaleType, f64) {
        let mut confidence = 0.5;  // Base confidence
        
        // Exchange flow detection
        let from_is_exchange = self.is_exchange_address(&tx.from_address);
        let to_is_exchange = self.is_exchange_address(&tx.to_address);
        
        if to_is_exchange {
            confidence += 0.3;
            return (WhaleType::ExchangeInflow, confidence.min(1.0));
        }
        
        if from_is_exchange {
            confidence += 0.3;
            return (WhaleType::ExchangeOutflow, confidence.min(1.0));
        }
        
        // Analyze behavior patterns
        if let Some(from_behavior) = from_behavior {
            if from_behavior.total_outflow > from_behavior.total_inflow * 2.0 {
                // Heavy distributor pattern
                confidence += 0.2;
                return (WhaleType::LargeDistributor, confidence.min(1.0));
            }
            
            if from_behavior.exchange_interactions > 10 {
                confidence += 0.2;
                return (WhaleType::InstitutionalFlow, confidence.min(1.0));
            }
        }
        
        if let Some(to_behavior) = to_behavior {
            if to_behavior.total_inflow > to_behavior.total_outflow * 2.0 {
                // Heavy accumulator pattern
                confidence += 0.2;
                return (WhaleType::LargeAccumulator, confidence.min(1.0));
            }
        }
        
        // Institutional size detection
        if tx.amount > 10_000_000.0 {  // $10M+ is likely institutional
            confidence += 0.3;
            return (WhaleType::InstitutionalFlow, confidence.min(1.0));
        }
        
        // Suspicious activity detection
        if let Some(from_behavior) = from_behavior {
            if from_behavior.counterparties.len() > 100 && from_behavior.average_transaction_size > 5_000_000.0 {
                confidence += 0.4;
                return (WhaleType::SuspiciousActivity, confidence.min(1.0));
            }
        }
        
        // Default to wallet-to-wallet
        (WhaleType::WalletToWallet, confidence)
    }
    
    /// Generate comprehensive metadata for whale alert
    fn generate_whale_metadata(
        &self,
        tx: &Transaction,
        from_behavior: Option<&WalletBehavior>,
        to_behavior: Option<&WalletBehavior>,
    ) -> WhaleMetadata {
        let mut metadata = WhaleMetadata::default();
        
        // Aggregate behavior data
        if let Some(behavior) = from_behavior {
            metadata.transaction_count += behavior.transaction_count;
            metadata.volume_24h += behavior.total_inflow + behavior.total_outflow;
            metadata.first_seen = metadata.first_seen.min(behavior.first_activity);
            metadata.last_seen = metadata.last_seen.max(behavior.last_activity);
            metadata.exchange_tagged = behavior.exchange_interactions > 0;
        }
        
        if let Some(behavior) = to_behavior {
            metadata.transaction_count += behavior.transaction_count;
            metadata.volume_24h += behavior.total_inflow + behavior.total_outflow;
            metadata.first_seen = metadata.first_seen.min(behavior.first_activity);
            metadata.last_seen = metadata.last_seen.max(behavior.last_activity);
            metadata.exchange_tagged = metadata.exchange_tagged || behavior.exchange_interactions > 0;
        }
        
        // Calculate risk score
        metadata.risk_score = self.calculate_risk_score(tx, from_behavior, to_behavior);
        
        // Institutional tagging
        metadata.institutional_tagged = tx.amount > 5_000_000.0 ||  // $5M+
            metadata.volume_24h > 50_000_000.0;  // $50M+ daily volume
        
        metadata
    }
    
    /// Calculate risk score for whale transaction
    fn calculate_risk_score(
        &self,
        tx: &Transaction,
        from_behavior: Option<&WalletBehavior>,
        to_behavior: Option<&WalletBehavior>,
    ) -> f64 {
        let mut risk_score = 0.0;
        
        // Size-based risk
        if tx.amount > 100_000_000.0 {  // $100M+
            risk_score += 0.5;
        } else if tx.amount > 10_000_000.0 {  // $10M+
            risk_score += 0.3;
        } else if tx.amount > 1_000_000.0 {  // $1M+
            risk_score += 0.1;
        }
        
        // Behavior-based risk
        if let Some(behavior) = from_behavior {
            if behavior.counterparties.len() > 1000 {
                risk_score += 0.2;  // Too many counterparties (mixing service?)
            }
            
            if behavior.average_transaction_size > 50_000_000.0 {
                risk_score += 0.2;  // Unusually large average transaction
            }
        }
        
        // Suspicious address risk
        if self.suspicious_addresses.contains(&tx.from_address) || 
           self.suspicious_addresses.contains(&tx.to_address) {
            risk_score += 0.3;
        }
        
        risk_score.min(1.0)
    }
    
    /// Check if address belongs to known exchange
    fn is_exchange_address(&self, address: &str) -> bool {
        let address_lower = address.to_lowercase();
        
        for exchange in &self.known_exchanges {
            if address_lower.contains(exchange) {
                return true;
            }
        }
        
        false
    }
    
    /// Add address to suspicious list
    pub fn mark_suspicious(&mut self, address: String) {
        self.suspicious_addresses.insert(address);
    }
    
    /// Get wallet behavior summary
    pub fn get_wallet_summary(&self, address: &str) -> Option<WalletBehaviorSummary> {
        self.wallet_behaviors.get(address).map(|behavior| {
            WalletBehaviorSummary {
                address: address.to_string(),
                total_inflow: behavior.total_inflow,
                total_outflow: behavior.total_outflow,
                net_flow: behavior.net_flow(),
                transaction_count: behavior.transaction_count,
                activity_score: behavior.activity_score(),
                exchange_interactions: behavior.exchange_interactions,
                counterparty_count: behavior.counterparties.len() as u32,
                average_transaction_size: behavior.average_transaction_size,
                first_activity: behavior.first_activity,
                last_activity: behavior.last_activity,
            }
        })
    }
    
    /// Get performance statistics
    pub fn get_detection_stats(&self) -> DetectionStats {
        let total_wallets = self.wallet_behaviors.len();
        let active_wallets = self.wallet_behaviors
            .values()
            .filter(|b| b.transaction_count > 0)
            .count();
        let total_volume: f64 = self.wallet_behaviors
            .values()
            .map(|b| b.total_inflow + b.total_outflow)
            .sum();
        
        DetectionStats {
            total_wallets_tracked: total_wallets as u32,
            active_wallets: active_wallets as u32,
            total_volume_tracked: total_volume,
            suspicious_addresses: self.suspicious_addresses.len() as u32,
            whale_threshold: self.whale_threshold,
        }
    }
}

/// Wallet behavior summary for external API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletBehaviorSummary {
    pub address: String,
    pub total_inflow: f64,
    pub total_outflow: f64,
    pub net_flow: f64,
    pub transaction_count: u32,
    pub activity_score: f64,
    pub exchange_interactions: u32,
    pub counterparty_count: u32,
    pub average_transaction_size: f64,
    pub first_activity: u64,
    pub last_activity: u64,
}

/// Detection engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStats {
    pub total_wallets_tracked: u32,
    pub active_wallets: u32,
    pub total_volume_tracked: f64,
    pub suspicious_addresses: u32,
    pub whale_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_transaction(amount: f64, from: &str, to: &str) -> Transaction {
        Transaction {
            amount,
            timestamp: 1699200000,  // Fixed timestamp for testing
            from_address: from.to_string(),
            to_address: to.to_string(),
        }
    }
    
    #[test]
    fn test_whale_detection_basic() {
        let config = SupremeConfig::default();
        let mut detector = WhaleDetector::new(&config);
        
        let transactions = vec![
            create_test_transaction(5_000_000.0, "whale1", "whale2"),  // $5M
            create_test_transaction(100_000.0, "small1", "small2"),     // $100K (below threshold)
        ];
        
        let alerts = detector.process_transactions(&transactions).unwrap();
        assert_eq!(alerts.len(), 1);  // Only the $5M transaction should trigger alert
        assert_eq!(alerts[0].amount, 5_000_000.0);
    }
    
    #[test]
    fn test_exchange_flow_detection() {
        let config = SupremeConfig::default();
        let mut detector = WhaleDetector::new(&config);
        
        let transactions = vec![
            create_test_transaction(2_000_000.0, "user1", "binance_hot_wallet"),
        ];
        
        let alerts = detector.process_transactions(&transactions).unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].whale_type, WhaleType::ExchangeInflow);
        assert!(alerts[0].confidence > 0.7);
    }
    
    #[test]
    fn test_wallet_behavior_tracking() {
        let config = SupremeConfig::default();
        let mut detector = WhaleDetector::new(&config);
        
        let transactions = vec![
            create_test_transaction(1_000_000.0, "whale1", "whale2"),
            create_test_transaction(2_000_000.0, "whale1", "whale3"),
        ];
        
        detector.process_transactions(&transactions).unwrap();
        
        let summary = detector.get_wallet_summary("whale1").unwrap();
        assert_eq!(summary.total_outflow, 3_000_000.0);
        assert_eq!(summary.transaction_count, 2);
    }
    
    #[test]
    fn test_performance_target() {
        let config = SupremeConfig::default();
        let mut detector = WhaleDetector::new(&config);
        
        // Generate large batch of transactions
        let transactions: Vec<Transaction> = (0..1000)
            .map(|i| create_test_transaction(
                (1_000_000.0 + i as f64 * 1000.0), 
                &format!("addr_{}", i), 
                &format!("addr_{}", i + 1000)
            ))
            .collect();
        
        let start = std::time::Instant::now();
        let alerts = detector.process_transactions(&transactions).unwrap();
        let duration = start.elapsed();
        
        assert!(!alerts.is_empty());
        assert!(duration.as_millis() < 100);  // Target: <100ms for 1000 transactions
    }
}