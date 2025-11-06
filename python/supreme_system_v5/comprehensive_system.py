#!/usr/bin/env python3
"""
Supreme System V5 - Comprehensive Multi-Algorithm Trading Framework

Ultra-constrained comprehensive system for i3 4GB RAM constraint
Integrates: Multi-algorithms, News processing, Whale tracking, Sentiment analysis
Target memory usage: ≤80MB total system
"""

import asyncio
import json
import time
import logging
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
from threading import Lock

# Core system imports
import numpy as np
import pandas as pd
from scipy import signal

# Memory optimization
import psutil
import resource

# News and social APIs
import requests
import feedparser
from textblob import TextBlob
import tweepy
import praw  # Reddit API

# Rust core integration
try:
    import supreme_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust core not available - falling back to Python implementation")

# Ultra-minimal logging for memory efficiency
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveConfig:
    """Configuration for comprehensive trading system"""
    memory_budget_mb: float = 80.0
    
    # Algorithm configuration
    enabled_algorithms: List[str] = None
    max_concurrent_algorithms: int = 3
    
    # Data sources
    news_sources: List[str] = None
    social_sources: List[str] = None
    exchange_apis: List[str] = None
    
    # Processing intervals
    news_poll_interval: int = 300  # 5 minutes
    social_poll_interval: int = 180  # 3 minutes  
    whale_check_interval: int = 60   # 1 minute
    
    # Thresholds
    whale_threshold_usd: float = 1_000_000.0
    sentiment_threshold: float = 0.7
    news_impact_threshold: float = 0.6
    
    def __post_init__(self):
        if self.enabled_algorithms is None:
            self.enabled_algorithms = [
                'eth_usdt_scalping',
                'momentum_trading', 
                'whale_following',
                'news_trading',
                'arbitrage_detection'
            ]
        
        if self.news_sources is None:
            self.news_sources = [
                'https://feeds.feedburner.com/CoinDeskMain',
                'https://cointelegraph.com/rss',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://bitcoinmagazine.com/.rss/full/'
            ]
        
        if self.social_sources is None:
            self.social_sources = [
                'twitter_crypto_feed',
                'reddit_cryptocurrency', 
                'reddit_ethtrader',
                'telegram_whale_alerts'
            ]


class MemoryMonitor:
    """Ultra-efficient memory monitoring for 4GB constraint"""
    
    def __init__(self, budget_mb: float):
        self.budget_mb = budget_mb
        self.process = psutil.Process()
        self.peak_mb = 0.0
        self.alerts_sent = 0
        
    def get_current_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, memory_mb)
            return memory_mb
        except:
            return 0.0
    
    def check_budget_compliance(self) -> bool:
        """Check if within memory budget"""
        current = self.get_current_mb()
        return current <= self.budget_mb
    
    def force_cleanup_if_needed(self):
        """Force garbage collection if approaching budget"""
        current = self.get_current_mb()
        if current > self.budget_mb * 0.9:  # 90% of budget
            gc.collect()
            self.alerts_sent += 1
            logger.error(f"Memory cleanup forced: {current:.1f}MB > {self.budget_mb*0.9:.1f}MB")
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        return {
            'current_mb': self.get_current_mb(),
            'peak_mb': self.peak_mb,
            'budget_mb': self.budget_mb,
            'utilization': self.get_current_mb() / self.budget_mb,
            'cleanup_events': self.alerts_sent
        }


class NewsProcessor:
    """Real-time news processing and sentiment analysis"""
    
    def __init__(self, config: ComprehensiveConfig):
        self.config = config
        self.news_cache = {}  # Simple cache to avoid reprocessing
        self.sentiment_analyzer = TextBlob  # Lightweight sentiment analysis
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Supreme-System-V5/1.0'})
        
    async def fetch_news_feeds(self) -> List[Dict[str, Any]]:
        """Fetch news from multiple sources asynchronously"""
        news_items = []
        
        for feed_url in self.config.news_sources:
            try:
                # Fetch RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Limit to 5 recent items per source
                    news_item = {
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', ''),
                        'timestamp': int(time.time()),
                        'source': feed_url.split('/')[2] if '//' in feed_url else feed_url,
                        'url': entry.get('link', '')
                    }
                    
                    # Avoid duplicates
                    news_hash = hash(news_item['title'] + news_item['source'])
                    if news_hash not in self.news_cache:
                        self.news_cache[news_hash] = news_item
                        news_items.append(news_item)
                
            except Exception as e:
                logger.error(f"Error fetching news from {feed_url}: {e}")
                continue
        
        # Cleanup old cache entries (memory management)
        if len(self.news_cache) > 100:
            # Keep only recent 50 items
            recent_items = list(self.news_cache.items())[-50:]
            self.news_cache = dict(recent_items)
        
        return news_items
    
    def analyze_sentiment_batch(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment for batch of news items"""
        if not news_items:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'key_topics': [], 'market_impact': 0.0}
        
        sentiments = []
        key_topics = set()
        
        for item in news_items:
            # Combine title and content for analysis
            text = f"{item['title']} {item['content'][:200]}"  # Limit content length
            
            try:
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity  # -1 to 1
                sentiments.append(sentiment)
                
                # Extract key topics (simplified)
                words = blob.words
                crypto_terms = ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth', 'defi', 'nft']
                for word in words:
                    if word.lower() in crypto_terms:
                        key_topics.add(word.lower())
                        
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                sentiments.append(0.0)
        
        # Calculate aggregate metrics
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        confidence = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 0.5
        
        # Market impact estimation (simplified)
        market_impact = abs(overall_sentiment) * confidence * min(len(news_items) / 10.0, 1.0)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': overall_sentiment,  # Alias for compatibility
            'confidence': confidence,
            'key_topics': list(key_topics)[:5],  # Top 5 topics
            'market_impact': market_impact,
            'news_count': len(news_items)
        }


class SocialMediaMonitor:
    """Social media sentiment and trend monitoring"""
    
    def __init__(self, config: ComprehensiveConfig):
        self.config = config
        self.reddit_cache = {}
        self.twitter_cache = {}
        
    async def monitor_reddit_sentiment(self) -> Dict[str, Any]:
        """Monitor Reddit cryptocurrency discussions"""
        try:
            # Simplified Reddit monitoring (would need API keys in production)
            # For now, return mock data structure
            sentiment_data = {
                'subreddit_sentiment': {
                    'cryptocurrency': {'sentiment': 0.2, 'activity': 850},
                    'ethtrader': {'sentiment': 0.35, 'activity': 420},
                    'bitcoin': {'sentiment': -0.1, 'activity': 1200}
                },
                'trending_topics': ['ethereum', 'defi', 'nft', 'bitcoin', 'altcoin'],
                'overall_sentiment': 0.15,
                'activity_level': 'high'
            }
            return sentiment_data
        except Exception as e:
            logger.error(f"Reddit monitoring error: {e}")
            return {'error': str(e)}
    
    async def monitor_twitter_trends(self) -> Dict[str, Any]:
        """Monitor Twitter crypto trends"""
        try:
            # Simplified Twitter monitoring (would need API access)
            trend_data = {
                'trending_hashtags': ['#ethereum', '#defi', '#crypto', '#btc'],
                'mention_volume': {
                    'ethereum': 2500,
                    'bitcoin': 3200,
                    'defi': 800
                },
                'sentiment_by_coin': {
                    'ethereum': 0.25,
                    'bitcoin': -0.05,
                    'defi': 0.40
                },
                'overall_sentiment': 0.20
            }
            return trend_data
        except Exception as e:
            logger.error(f"Twitter monitoring error: {e}")
            return {'error': str(e)}


class MoneyFlowAnalyzer:
    """Money flow and market microstructure analysis"""
    
    def __init__(self, config: ComprehensiveConfig):
        self.config = config
        self.flow_data = {}
        
    def analyze_money_flow(self, market_data: List[Dict]) -> Dict[str, Any]:
        """Analyze money flow patterns"""
        if not market_data:
            return {'error': 'no_market_data'}
        
        try:
            # Extract price and volume data
            prices = [float(d.get('close', 0)) for d in market_data]
            volumes = [float(d.get('volume', 0)) for d in market_data]
            
            if len(prices) < 10:
                return {'error': 'insufficient_data'}
            
            # Calculate money flow indicators
            price_changes = np.diff(prices)
            volume_array = np.array(volumes[1:])  # Align with price changes
            
            # Positive and negative money flow
            positive_flow = np.sum(volume_array[price_changes > 0])
            negative_flow = np.sum(volume_array[price_changes < 0])
            
            # Money flow ratio and index
            total_flow = positive_flow + negative_flow
            if total_flow > 0:
                mfr = positive_flow / negative_flow if negative_flow > 0 else float('inf')
                mfi = 100 - (100 / (1 + mfr))
            else:
                mfr = 1.0
                mfi = 50.0
            
            # Volume-weighted average price (VWAP)
            price_volume = np.array(prices[1:]) * volume_array
            vwap = np.sum(price_volume) / np.sum(volume_array) if np.sum(volume_array) > 0 else prices[-1]
            
            # Money flow heatmap data (simplified)
            price_levels = np.linspace(min(prices), max(prices), 20)
            flow_distribution = np.histogram(prices, bins=price_levels, weights=volumes)[0]
            
            return {
                'money_flow_ratio': mfr,
                'money_flow_index': mfi,
                'positive_flow': positive_flow,
                'negative_flow': negative_flow,
                'vwap': vwap,
                'flow_heatmap': {
                    'price_levels': price_levels.tolist(),
                    'flow_distribution': flow_distribution.tolist()
                },
                'net_flow': positive_flow - negative_flow,
                'flow_strength': 'strong' if abs(mfi - 50) > 20 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Money flow analysis error: {e}")
            return {'error': str(e)}


class MultiAlgorithmFramework:
    """Multi-algorithm trading framework with resource management"""
    
    def __init__(self, config: ComprehensiveConfig, memory_monitor: MemoryMonitor):
        self.config = config
        self.memory_monitor = memory_monitor
        self.algorithms = {}
        self.active_algorithms = []
        self.algorithm_lock = Lock()
        self.performance_stats = {}
        
        # Initialize algorithms
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize available trading algorithms"""
        
        # ETH-USDT Scalping (from existing system)
        if 'eth_usdt_scalping' in self.config.enabled_algorithms:
            self.algorithms['eth_usdt_scalping'] = {
                'name': 'ETH-USDT Scalping',
                'memory_usage_mb': 8.0,
                'active': False,
                'performance': {'win_rate': 0.0, 'pnl': 0.0}
            }
        
        # Momentum Trading
        if 'momentum_trading' in self.config.enabled_algorithms:
            self.algorithms['momentum_trading'] = {
                'name': 'Momentum Trading',
                'memory_usage_mb': 6.0,
                'active': False,
                'performance': {'win_rate': 0.0, 'pnl': 0.0}
            }
        
        # Whale Following Strategy
        if 'whale_following' in self.config.enabled_algorithms:
            self.algorithms['whale_following'] = {
                'name': 'Whale Following',
                'memory_usage_mb': 5.0,
                'active': False,
                'performance': {'win_rate': 0.0, 'pnl': 0.0}
            }
        
        # News-based Trading
        if 'news_trading' in self.config.enabled_algorithms:
            self.algorithms['news_trading'] = {
                'name': 'News Trading',
                'memory_usage_mb': 4.0,
                'active': False,
                'performance': {'win_rate': 0.0, 'pnl': 0.0}
            }
        
        # Arbitrage Detection
        if 'arbitrage_detection' in self.config.enabled_algorithms:
            self.algorithms['arbitrage_detection'] = {
                'name': 'Arbitrage Detection',
                'memory_usage_mb': 7.0,
                'active': False,
                'performance': {'win_rate': 0.0, 'pnl': 0.0}
            }
    
    def activate_algorithm(self, algorithm_id: str) -> bool:
        """Activate algorithm with memory budget checking"""
        with self.algorithm_lock:
            if algorithm_id not in self.algorithms:
                return False
            
            if len(self.active_algorithms) >= self.config.max_concurrent_algorithms:
                logger.error(f"Cannot activate {algorithm_id}: max concurrent limit reached")
                return False
            
            # Check memory budget
            required_memory = self.algorithms[algorithm_id]['memory_usage_mb']
            current_memory = self.memory_monitor.get_current_mb()
            
            if current_memory + required_memory > self.config.memory_budget_mb:
                logger.error(f"Cannot activate {algorithm_id}: would exceed memory budget")
                return False
            
            # Activate algorithm
            self.algorithms[algorithm_id]['active'] = True
            self.active_algorithms.append(algorithm_id)
            
            logger.error(f"Activated algorithm: {algorithm_id}")
            return True
    
    def deactivate_algorithm(self, algorithm_id: str) -> bool:
        """Deactivate algorithm and free resources"""
        with self.algorithm_lock:
            if algorithm_id not in self.algorithms or not self.algorithms[algorithm_id]['active']:
                return False
            
            self.algorithms[algorithm_id]['active'] = False
            if algorithm_id in self.active_algorithms:
                self.active_algorithms.remove(algorithm_id)
            
            # Force memory cleanup
            self.memory_monitor.force_cleanup_if_needed()
            
            logger.error(f"Deactivated algorithm: {algorithm_id}")
            return True
    
    def get_active_algorithms(self) -> List[str]:
        """Get list of currently active algorithms"""
        return self.active_algorithms.copy()
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Get comprehensive algorithm statistics"""
        stats = {
            'total_algorithms': len(self.algorithms),
            'active_algorithms': len(self.active_algorithms),
            'memory_usage_mb': sum(
                self.algorithms[alg_id]['memory_usage_mb'] 
                for alg_id in self.active_algorithms
            ),
            'algorithms': self.algorithms.copy()
        }
        return stats


class ComprehensiveSupremeSystem:
    """Main comprehensive trading system integrating all components"""
    
    def __init__(self, config: ComprehensiveConfig = None):
        self.config = config or ComprehensiveConfig()
        self.start_time = datetime.now()
        
        # Initialize core components
        self.memory_monitor = MemoryMonitor(self.config.memory_budget_mb)
        self.news_processor = NewsProcessor(self.config)
        self.social_monitor = SocialMediaMonitor(self.config)
        self.money_flow_analyzer = MoneyFlowAnalyzer(self.config)
        self.algorithm_framework = MultiAlgorithmFramework(self.config, self.memory_monitor)
        
        # Initialize Rust core if available
        if RUST_AVAILABLE:
            try:
                self.rust_core = supreme_core.SupremeCore()
                logger.error("Rust core initialized successfully")
            except Exception as e:
                logger.error(f"Rust core initialization failed: {e}")
                self.rust_core = None
        else:
            self.rust_core = None
        
        # System state
        self.running = False
        self.last_news_update = 0
        self.last_social_update = 0
        self.last_whale_check = 0
        
        logger.error("Comprehensive Supreme System V5 initialized")
        logger.error(f"Memory budget: {self.config.memory_budget_mb}MB")
        logger.error(f"Enabled algorithms: {len(self.config.enabled_algorithms)}")
    
    async def start_system(self):
        """Start the comprehensive trading system"""
        self.running = True
        logger.error("Starting Comprehensive Supreme System V5")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._news_processing_loop()),
            asyncio.create_task(self._social_monitoring_loop()),
            asyncio.create_task(self._whale_monitoring_loop()),
            asyncio.create_task(self._memory_monitoring_loop()),
            asyncio.create_task(self._trading_execution_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.running = False
    
    async def _news_processing_loop(self):
        """Background news processing loop"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_news_update > self.config.news_poll_interval:
                    # Fetch and process news
                    news_items = await self.news_processor.fetch_news_feeds()
                    
                    if news_items:
                        sentiment_result = self.news_processor.analyze_sentiment_batch(news_items)
                        logger.error(f"News update: {len(news_items)} items, sentiment: {sentiment_result['overall_sentiment']:.2f}")
                    
                    self.last_news_update = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"News processing error: {e}")
                await asyncio.sleep(60)
    
    async def _social_monitoring_loop(self):
        """Background social media monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_social_update > self.config.social_poll_interval:
                    # Monitor social sentiment
                    reddit_data = await self.social_monitor.monitor_reddit_sentiment()
                    twitter_data = await self.social_monitor.monitor_twitter_trends()
                    
                    logger.error(f"Social update: Reddit sentiment {reddit_data.get('overall_sentiment', 0):.2f}, Twitter sentiment {twitter_data.get('overall_sentiment', 0):.2f}")
                    
                    self.last_social_update = current_time
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                logger.error(f"Social monitoring error: {e}")
                await asyncio.sleep(90)
    
    async def _whale_monitoring_loop(self):
        """Background whale transaction monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_whale_check > self.config.whale_check_interval:
                    # Generate mock whale data for demonstration
                    mock_transactions = [
                        {
                            'amount': np.random.uniform(1_000_000, 10_000_000),
                            'timestamp': int(current_time),
                            'from': f'whale_address_{np.random.randint(1, 100)}',
                            'to': f'exchange_address_{np.random.randint(1, 10)}'
                        }
                        for _ in range(np.random.randint(0, 3))
                    ]
                    
                    if mock_transactions and self.rust_core:
                        try:
                            whale_alerts = self.rust_core.detect_whales(mock_transactions)
                            if whale_alerts:
                                logger.error(f"Whale alerts: {len(whale_alerts)} detected")
                        except Exception as e:
                            logger.error(f"Rust whale detection error: {e}")
                    
                    self.last_whale_check = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Whale monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _memory_monitoring_loop(self):
        """Background memory monitoring and optimization loop"""
        while self.running:
            try:
                # Check memory usage
                memory_stats = self.memory_monitor.get_stats()
                
                if memory_stats['current_mb'] > self.config.memory_budget_mb * 0.9:
                    logger.error(f"Memory usage high: {memory_stats['current_mb']:.1f}MB / {self.config.memory_budget_mb}MB")
                    
                    # Force cleanup
                    self.memory_monitor.force_cleanup_if_needed()
                    
                    # Deactivate least important algorithm if needed
                    if memory_stats['current_mb'] > self.config.memory_budget_mb:
                        active_algs = self.algorithm_framework.get_active_algorithms()
                        if active_algs:
                            # Deactivate last activated algorithm
                            self.algorithm_framework.deactivate_algorithm(active_algs[-1])
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _trading_execution_loop(self):
        """Main trading execution loop coordinating all algorithms"""
        while self.running:
            try:
                # Simulate trading execution
                active_algorithms = self.algorithm_framework.get_active_algorithms()
                
                if active_algorithms:
                    # Generate mock market data
                    mock_market_data = [
                        {
                            'timestamp': time.time() - i * 60,
                            'open': 2000 + np.random.uniform(-50, 50),
                            'high': 2000 + np.random.uniform(-30, 70),
                            'low': 2000 + np.random.uniform(-70, 30),
                            'close': 2000 + np.random.uniform(-50, 50),
                            'volume': np.random.uniform(1000, 5000)
                        }
                        for i in range(100)
                    ]
                    
                    # Money flow analysis
                    flow_analysis = self.money_flow_analyzer.analyze_money_flow(mock_market_data)
                    
                    # Execute algorithms with Rust core if available
                    if self.rust_core and len(mock_market_data) > 0:
                        try:
                            prices = np.array([d['close'] for d in mock_market_data[-50:]])
                            indicators = self.rust_core.calculate_indicators(prices, 14)
                            memory_stats = self.rust_core.get_memory_stats()
                            
                            logger.error(f"Trading cycle: {len(active_algorithms)} algorithms active, memory: {memory_stats['current_usage_mb']:.1f}MB")
                            
                        except Exception as e:
                            logger.error(f"Rust trading execution error: {e}")
                
                await asyncio.sleep(5)  # Execute every 5 seconds
                
            except Exception as e:
                logger.error(f"Trading execution error: {e}")
                await asyncio.sleep(15)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        memory_stats = self.memory_monitor.get_stats()
        algorithm_stats = self.algorithm_framework.get_algorithm_stats()
        
        status = {
            'system_info': {
                'running': self.running,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'rust_core_available': self.rust_core is not None,
                'memory_budget_mb': self.config.memory_budget_mb
            },
            'memory_performance': memory_stats,
            'algorithm_status': algorithm_stats,
            'data_sources': {
                'news_sources': len(self.config.news_sources),
                'social_sources': len(self.config.social_sources),
                'last_news_update': self.last_news_update,
                'last_social_update': self.last_social_update,
                'last_whale_check': self.last_whale_check
            },
            'performance_grade': self._calculate_performance_grade(memory_stats, algorithm_stats)
        }
        
        return status
    
    def _calculate_performance_grade(self, memory_stats: Dict, algorithm_stats: Dict) -> str:
        """Calculate overall system performance grade"""
        memory_efficiency = 1.0 - memory_stats['utilization']
        algorithm_utilization = algorithm_stats['active_algorithms'] / max(1, algorithm_stats['total_algorithms'])
        
        overall_score = (memory_efficiency + algorithm_utilization) / 2.0
        
        if overall_score > 0.8:
            return 'excellent'
        elif overall_score > 0.6:
            return 'good'
        elif overall_score > 0.4:
            return 'fair'
        else:
            return 'needs_optimization'
    
    async def stop_system(self):
        """Stop the comprehensive trading system"""
        logger.error("Stopping Comprehensive Supreme System V5")
        self.running = False
        
        # Deactivate all algorithms
        active_algorithms = self.algorithm_framework.get_active_algorithms()
        for alg_id in active_algorithms:
            self.algorithm_framework.deactivate_algorithm(alg_id)
        
        # Final memory cleanup
        self.memory_monitor.force_cleanup_if_needed()
        
        logger.error("System stopped successfully")


# Factory function for easy instantiation
def create_comprehensive_system(memory_budget_mb: float = 80.0) -> ComprehensiveSupremeSystem:
    """Create comprehensive system with specified memory budget"""
    config = ComprehensiveConfig(memory_budget_mb=memory_budget_mb)
    return ComprehensiveSupremeSystem(config)


if __name__ == "__main__":
    async def main():
        # Create and start comprehensive system
        config = ComprehensiveConfig(memory_budget_mb=80.0)
        system = ComprehensiveSupremeSystem(config)
        
        # Activate some algorithms
        system.algorithm_framework.activate_algorithm('eth_usdt_scalping')
        system.algorithm_framework.activate_algorithm('whale_following')
        system.algorithm_framework.activate_algorithm('news_trading')
        
        print("\n🎆 Supreme System V5 - Comprehensive Multi-Algorithm Trading System")
        print(f"Memory Budget: {config.memory_budget_mb}MB")
        print(f"Algorithms Available: {len(config.enabled_algorithms)}")
        print(f"Data Sources: {len(config.news_sources)} news, {len(config.social_sources)} social")
        print("=" * 70)
        
        try:
            # Run for demonstration
            await asyncio.wait_for(system.start_system(), timeout=30.0)
        except asyncio.TimeoutError:
            print("\nDemonstration completed - stopping system...")
        except KeyboardInterrupt:
            print("\nUser interrupted - stopping system...")
        finally:
            await system.stop_system()
            
            # Final status report
            status = system.get_system_status()
            print("\n📊 FINAL SYSTEM STATUS:")
            print(f"Memory Usage: {status['memory_performance']['current_mb']:.1f}MB / {status['memory_performance']['budget_mb']}MB")
            print(f"Active Algorithms: {status['algorithm_status']['active_algorithms']} / {status['algorithm_status']['total_algorithms']}")
            print(f"Performance Grade: {status['performance_grade'].upper()}")
            print(f"Uptime: {status['system_info']['uptime_seconds']:.1f} seconds")
    
    # Run demonstration
    asyncio.run(main())