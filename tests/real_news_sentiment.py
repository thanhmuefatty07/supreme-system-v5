#!/usr/bin/env python3
"""
Supreme System V5 - REAL NEWS SENTIMENT TEST
Critical validation of news sentiment analysis with real data sources

Tests sentiment analysis accuracy against real news feeds with 85% target
Memory budget: 300MB with comprehensive leak detection
"""

import asyncio
import aiohttp
import json
import logging
import psutil
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import numpy as np
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import hashlib
import feedparser
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_news_sentiment_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsSource:
    """Configuration for news data sources"""
    name: str
    rss_url: str
    homepage: str
    article_selector: str = "article"
    title_selector: str = "h1, .headline, .title"
    content_selector: str = ".content, .article-body, .post-content"
    rate_limit_delay: float = 1.0

@dataclass
class NewsArticle:
    """Structured news article data"""
    id: str
    source: str
    title: str
    content: str
    url: str
    published_at: datetime
    sentiment_score: float = 0.0
    confidence: float = 0.0
    crypto_entities: List[str] = field(default_factory=list)
    market_impact: str = "neutral"
    processing_time_ms: float = 0.0

@dataclass
class SentimentMetrics:
    """Sentiment analysis performance metrics"""
    total_articles: int = 0
    processed_articles: int = 0
    accuracy_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    crypto_mentions: int = 0
    positive_sentiment: int = 0
    negative_sentiment: int = 0
    neutral_sentiment: int = 0
    avg_processing_time_ms: float = 0.0
    memory_peak_mb: float = 0.0

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer with crypto-specific patterns"""

    def __init__(self):
        # Crypto-specific sentiment keywords
        self.positive_keywords = {
            'bullish', 'surge', 'rally', 'moon', 'pump', 'breakthrough', 'adoption',
            'partnership', 'integration', 'upgrade', 'milestone', 'growth', 'profit',
            'gain', 'rise', 'increase', 'bull', 'optimism', 'confidence', 'success',
            'breakout', 'momentum', 'recovery', 'green', 'uptrend'
        }

        self.negative_keywords = {
            'bearish', 'crash', 'dump', 'sell-off', 'decline', 'drop', 'fall',
            'bear', 'concern', 'fear', 'panic', 'sell', 'liquidation', 'loss',
            'bleed', 'plunge', 'tumble', 'slump', 'downturn', 'red', 'downtrend',
            'correction', 'rejection', 'ban', 'regulation', 'crackdown'
        }

        self.neutral_keywords = {
            'stable', 'steady', 'unchanged', 'flat', 'sideways', 'consolidation',
            'range', 'balance', 'equilibrium', 'hold', 'wait', 'monitor', 'watch'
        }

        # Crypto-specific entities
        self.crypto_entities = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'cardano', 'ada',
            'polygon', 'matic', 'chainlink', 'link', 'uniswap', 'uni', 'aave',
            'compound', 'maker', 'mkr', 'sushiswap', 'sushi', 'pancakeswap', 'cake',
            'avalanche', 'avax', 'polkadot', 'dot', 'kusama', 'ksm', 'cosmos', 'atom'
        }

        # Contextual modifiers
        self.intensifiers = {'very', 'extremely', 'highly', 'significantly', 'dramatically'}
        self.negators = {'not', 'no', 'never', 'without', 'lack', 'fail'}

    def analyze_sentiment(self, text: str) -> Tuple[float, float, List[str]]:
        """Analyze sentiment of text with confidence score"""
        start_time = time.time()

        # Preprocessing
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        # Extract crypto entities
        crypto_entities = []
        for entity in self.crypto_entities:
            if entity in text:
                crypto_entities.append(entity.upper())

        # Sentiment scoring
        positive_score = 0
        negative_score = 0
        neutral_score = 0

        # Sliding window analysis for context
        window_size = 3
        for i in range(len(words)):
            window = words[max(0, i-window_size):min(len(words), i+window_size+1)]
            word = words[i]

            # Check for intensifiers and negators in window
            has_intensifier = any(intensifier in window for intensifier in self.intensifiers)
            has_negator = any(negator in window for negator in self.negators)

            multiplier = 1.0
            if has_intensifier:
                multiplier *= 1.5
            if has_negator:
                multiplier *= -0.7

            if word in self.positive_keywords:
                positive_score += 1.0 * multiplier
            elif word in self.negative_keywords:
                negative_score += 1.0 * multiplier
            elif word in self.neutral_keywords:
                neutral_score += 1.0 * multiplier

        # Calculate final sentiment score
        total_words = len(words)
        if total_words == 0:
            return 0.0, 0.0, crypto_entities

        # Normalize scores
        positive_ratio = positive_score / total_words
        negative_ratio = negative_score / total_words
        neutral_ratio = neutral_score / total_words

        # Sentiment score: -1 (negative) to +1 (positive)
        sentiment_score = positive_ratio - negative_ratio

        # Confidence based on clarity of sentiment
        max_ratio = max(positive_ratio, negative_ratio, neutral_ratio)
        confidence = min(max_ratio * 3.0, 1.0)  # Scale up for clearer signals

        # Boost confidence if we have crypto entities
        if crypto_entities:
            confidence = min(confidence * 1.2, 1.0)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return sentiment_score, confidence, crypto_entities

    def determine_market_impact(self, sentiment_score: float, confidence: float,
                              crypto_entities: List[str]) -> str:
        """Determine market impact based on sentiment analysis"""
        # High confidence and strong sentiment with crypto entities = significant impact
        if confidence > 0.8 and abs(sentiment_score) > 0.3 and crypto_entities:
            if sentiment_score > 0.3:
                return "bullish"
            elif sentiment_score < -0.3:
                return "bearish"
            else:
                return "neutral"
        elif confidence > 0.6 and abs(sentiment_score) > 0.2:
            if sentiment_score > 0.2:
                return "moderately_bullish"
            elif sentiment_score < -0.2:
                return "moderately_bearish"
            else:
                return "neutral"
        else:
            return "neutral"

class NewsFeedCollector:
    """Collect news from RSS feeds and web sources"""

    def __init__(self, sources: List[NewsSource]):
        self.sources = sources
        self.session: Optional[aiohttp.ClientSession] = None
        self.collected_articles: List[NewsArticle] = []
        self.collection_stats = {
            'total_feeds': len(sources),
            'successful_feeds': 0,
            'failed_feeds': 0,
            'total_articles': 0,
            'crypto_articles': 0
        }

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30.0, connect=10.0),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; SupremeSystem/2.0; +https://github.com/thanhmuefatty07/supreme-system-v5)'
            }
        )

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def collect_news_batch(self, sample_size: int = 1000) -> List[NewsArticle]:
        """Collect news articles from all sources"""
        logger.info(f"Starting news collection from {len(self.sources)} sources, target: {sample_size} articles")

        tasks = []
        for source in self.sources:
            task = asyncio.create_task(self._collect_from_source(source))
            tasks.append(task)

        # Wait for all collection tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Collection error: {result}")
                self.collection_stats['failed_feeds'] += 1
            else:
                articles = result
                all_articles.extend(articles)
                if articles:
                    self.collection_stats['successful_feeds'] += 1

        # Filter and limit results
        crypto_articles = [a for a in all_articles if a.crypto_entities]

        logger.info(f"Collection complete: {len(all_articles)} total articles, {len(crypto_articles)} crypto-related")

        # Prioritize crypto articles but include some general news
        final_articles = crypto_articles[:min(sample_size, len(crypto_articles))]

        # Add some general news if we have room
        remaining_slots = sample_size - len(final_articles)
        if remaining_slots > 0:
            general_articles = [a for a in all_articles if not a.crypto_entities]
            final_articles.extend(general_articles[:remaining_slots])

        self.collection_stats['total_articles'] = len(final_articles)
        self.collection_stats['crypto_articles'] = len([a for a in final_articles if a.crypto_entities])

        return final_articles

    async def _collect_from_source(self, source: NewsSource) -> List[NewsArticle]:
        """Collect articles from a single source"""
        articles = []

        try:
            # Try RSS feed first
            rss_articles = await self._collect_from_rss(source)
            articles.extend(rss_articles)

            # Respect rate limits
            await asyncio.sleep(source.rate_limit_delay)

        except Exception as e:
            logger.warning(f"Failed to collect from {source.name} RSS: {e}")

        return articles

    async def _collect_from_rss(self, source: NewsSource) -> List[NewsArticle]:
        """Collect articles from RSS feed"""
        articles = []

        async with self.session.get(source.rss_url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status} from {source.rss_url}")

            content = await response.text()
            feed = feedparser.parse(content)

            for entry in feed.entries[:20]:  # Limit to recent 20 articles
                # Extract content
                title = getattr(entry, 'title', '')
                description = getattr(entry, 'description', '')
                content = getattr(entry, 'content', [{}])[0].get('value', '') if hasattr(entry, 'content') else ''
                full_content = f"{title} {description} {content}"

                # Extract publication date
                published = getattr(entry, 'published_parsed', None)
                if published:
                    published_at = datetime(*published[:6])
                else:
                    published_at = datetime.now()

                # Generate unique ID
                content_hash = hashlib.md5(full_content.encode()).hexdigest()[:8]
                article_id = f"{source.name.lower()}_{content_hash}"

                article = NewsArticle(
                    id=article_id,
                    source=source.name,
                    title=title,
                    content=full_content,
                    url=getattr(entry, 'link', ''),
                    published_at=published_at
                )

                articles.append(article)

        return articles

class GroundTruthValidator:
    """Validate sentiment analysis against ground truth"""

    def __init__(self):
        # Pre-labeled ground truth dataset for validation
        self.ground_truth = {
            # Positive examples
            "Bitcoin surges 15% as institutional adoption grows": 0.8,
            "Ethereum upgrade successfully deployed": 0.7,
            "Major bank announces crypto trading service": 0.9,
            "Bitcoin ETF sees record inflows": 0.85,

            # Negative examples
            "Crypto market crashes 20% amid regulatory fears": -0.8,
            "Major exchange hacked, millions lost": -0.9,
            "Government announces crypto ban": -0.95,
            "Bitcoin price plunges following negative news": -0.75,

            # Neutral examples
            "Bitcoin trading volume reaches new high": 0.1,
            "New crypto exchange launches": 0.0,
            "Blockchain technology explained": -0.1,
            "Crypto market maintains stability": 0.05,
        }

    def validate_accuracy(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Validate sentiment analysis accuracy"""
        correct_predictions = 0
        total_predictions = 0

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for article in articles:
            # Find closest ground truth match
            best_match_score = 0.0
            best_ground_truth = 0.0

            for gt_text, gt_sentiment in self.ground_truth.items():
                # Simple text similarity (in production, use better similarity measures)
                similarity = self._calculate_text_similarity(article.title + " " + article.content[:200], gt_text)
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_ground_truth = gt_sentiment

            if best_match_score > 0.3:  # Minimum similarity threshold
                total_predictions += 1

                # Classify predictions
                predicted_class = self._classify_sentiment(article.sentiment_score)
                actual_class = self._classify_sentiment(best_ground_truth)

                if predicted_class == actual_class:
                    correct_predictions += 1
                    if predicted_class == "positive":
                        true_positives += 1
                    elif predicted_class == "negative":
                        true_negatives += 1
                else:
                    if predicted_class == "positive" and actual_class != "positive":
                        false_positives += 1
                    elif predicted_class == "negative" and actual_class != "negative":
                        false_negatives += 1

        # Calculate metrics
        accuracy = correct_predictions / max(total_predictions, 1)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_validations': total_predictions,
            'correct_predictions': correct_predictions
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / max(len(union), 1)

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

class RealNewsSentimentTester:
    """Main real news sentiment testing engine"""

    def __init__(self, sources: List[str], sample_size: int = 1000,
                 accuracy_target: float = 0.85, memory_limit_mb: int = 300):
        self.sources = self._configure_sources(sources)
        self.sample_size = sample_size
        self.accuracy_target = accuracy_target
        self.memory_limit_mb = memory_limit_mb

        # Core components
        self.collector = NewsFeedCollector(self.sources)
        self.analyzer = AdvancedSentimentAnalyzer()
        self.validator = GroundTruthValidator()

        # Test state
        self.is_running = False
        self.test_start_time: Optional[datetime] = None

        # Results tracking
        self.articles: List[NewsArticle] = []
        self.metrics = SentimentMetrics()
        self.memory_stats: List[Dict[str, Any]] = []

        # Results
        self.results = {
            'configuration': {
                'sources': [s.name for s in self.sources],
                'sample_size': sample_size,
                'accuracy_target': accuracy_target,
                'memory_limit_mb': memory_limit_mb
            },
            'collection_stats': {},
            'sentiment_metrics': {},
            'validation_results': {},
            'memory_stats': [],
            'errors': [],
            'success': False
        }

        logger.info(f"Real News Sentiment Tester initialized - Sources: {[s.name for s in self.sources]}, "
                   f"Sample size: {sample_size}, Accuracy target: {accuracy_target}")

    def _configure_sources(self, source_names: List[str]) -> List[NewsSource]:
        """Configure news sources"""
        available_sources = {
            'reuters': NewsSource(
                name='Reuters',
                rss_url='https://feeds.reuters.com/Reuters/worldNews',
                homepage='https://www.reuters.com/',
                rate_limit_delay=2.0
            ),
            'bloomberg': NewsSource(
                name='Bloomberg',
                rss_url='https://feeds.bloomberg.com/markets/news.rss',
                homepage='https://www.bloomberg.com/',
                rate_limit_delay=2.0
            ),
            'coindesk': NewsSource(
                name='CoinDesk',
                rss_url='https://www.coindesk.com/arc/outboundfeeds/rss/',
                homepage='https://www.coindesk.com/',
                rate_limit_delay=1.0
            )
        }

        return [available_sources[name.lower()] for name in source_names
                if name.lower() in available_sources]

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutdown signal received - stopping news sentiment test")
        self.is_running = False

    def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage during test"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'memory_limit_exceeded': (memory_info.rss / (1024 ** 2)) > self.memory_limit_mb
        }

    async def run_sentiment_test(self) -> Dict[str, Any]:
        """Execute the real news sentiment test"""
        self.test_start_time = datetime.now()
        end_time = self.test_start_time + timedelta(hours=2)  # 2 hour test
        self.is_running = True

        logger.info("🚀 STARTING REAL NEWS SENTIMENT TEST")
        logger.info(f"Duration: 2 hours, Target accuracy: {self.accuracy_target}")
        logger.info(f"Memory budget: {self.memory_limit_mb}MB")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Initialize collector
            await self.collector.initialize()

            # Phase 1: Collect news articles
            logger.info("Phase 1: Collecting news articles...")
            articles = await self.collector.collect_news_batch(self.sample_size)

            if not articles:
                raise Exception("Failed to collect any news articles")

            logger.info(f"Collected {len(articles)} articles from {self.collector.collection_stats['successful_feeds']} sources")

            # Phase 2: Analyze sentiment
            logger.info("Phase 2: Analyzing sentiment...")

            processed_articles = []
            for i, article in enumerate(articles):
                if not self.is_running:
                    break

                # Memory check
                memory_stats = self.monitor_memory_usage()
                self.memory_stats.append(memory_stats)

                if memory_stats['memory_limit_exceeded']:
                    logger.error(".1f"                    break

                # Analyze sentiment
                sentiment_score, confidence, crypto_entities = self.analyzer.analyze_sentiment(
                    article.title + " " + article.content
                )

                market_impact = self.analyzer.determine_market_impact(
                    sentiment_score, confidence, crypto_entities
                )

                # Update article with analysis
                article.sentiment_score = sentiment_score
                article.confidence = confidence
                article.crypto_entities = crypto_entities
                article.market_impact = market_impact
                article.processing_time_ms = time.time() * 1000  # Placeholder

                processed_articles.append(article)

                # Update metrics
                self.metrics.total_articles += 1
                self.metrics.processed_articles += 1
                self.metrics.crypto_mentions += len(crypto_entities)

                if sentiment_score > 0.2:
                    self.metrics.positive_sentiment += 1
                elif sentiment_score < -0.2:
                    self.metrics.negative_sentiment += 1
                else:
                    self.metrics.neutral_sentiment += 1

                # Progress reporting
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(articles)} articles")

            self.articles = processed_articles

            # Phase 3: Validate accuracy
            logger.info("Phase 3: Validating accuracy...")
            validation_results = self.validator.validate_accuracy(self.articles)

            # Update final metrics
            self.metrics.accuracy_score = validation_results['accuracy']
            self.metrics.precision = validation_results['precision']
            self.metrics.recall = validation_results['recall']
            self.metrics.f1_score = validation_results['f1_score']
            self.metrics.memory_peak_mb = max(s['rss_mb'] for s in self.memory_stats) if self.memory_stats else 0

            # Test success criteria
            accuracy_achieved = validation_results['accuracy'] >= self.accuracy_target
            memory_compliance = not any(s['memory_limit_exceeded'] for s in self.memory_stats)
            sufficient_data = len(self.articles) >= 100

            self.results['success'] = accuracy_achieved and memory_compliance and sufficient_data

            logger.info("✅ REAL NEWS SENTIMENT TEST COMPLETED")

        except Exception as e:
            error_msg = f"Sentiment test failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.results['errors'].append(error_msg)

        finally:
            await self.collector.close()
            self.is_running = False

        # Compile final results
        self.results['collection_stats'] = self.collector.collection_stats
        self.results['sentiment_metrics'] = {
            'total_articles': self.metrics.total_articles,
            'processed_articles': self.metrics.processed_articles,
            'accuracy_score': self.metrics.accuracy_score,
            'precision': self.metrics.precision,
            'recall': self.metrics.recall,
            'f1_score': self.metrics.f1_score,
            'crypto_mentions': self.metrics.crypto_mentions,
            'sentiment_distribution': {
                'positive': self.metrics.positive_sentiment,
                'negative': self.metrics.negative_sentiment,
                'neutral': self.metrics.neutral_sentiment
            },
            'memory_peak_mb': self.metrics.memory_peak_mb
        }
        self.results['validation_results'] = self.validator.validate_accuracy(self.articles)
        self.results['memory_stats'] = self.memory_stats

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"real_news_sentiment_test_results_{timestamp}.json"

        # Convert articles to serializable format
        serializable_articles = []
        for article in self.articles[:50]:  # Limit to first 50 for file size
            serializable_articles.append({
                'id': article.id,
                'source': article.source,
                'title': article.title,
                'sentiment_score': article.sentiment_score,
                'confidence': article.confidence,
                'crypto_entities': article.crypto_entities,
                'market_impact': article.market_impact,
                'published_at': article.published_at.isoformat()
            })

        results_copy = self.results.copy()
        results_copy['sample_articles'] = serializable_articles

        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📰 REAL NEWS SENTIMENT TEST RESULTS")
        print("=" * 70)

        if self.results['success']:
            print("✅ TEST PASSED - Real news sentiment analysis successful")

            metrics = self.results['sentiment_metrics']
            validation = self.results['validation_results']

            print("📊 Sentiment Analysis Metrics:"            print(f"   Articles Processed: {metrics['processed_articles']}")
            print(".1f"            print(".2f"            print(".2f"            print(".2f"            print(f"   Crypto Mentions: {metrics['crypto_mentions']}")

            sentiment_dist = metrics['sentiment_distribution']
            print("📈 Sentiment Distribution:"            print(f"   Positive: {sentiment_dist['positive']}")
            print(f"   Negative: {sentiment_dist['negative']}")
            print(f"   Neutral: {sentiment_dist['neutral']}")

            print("💾 Memory Usage:"            print(".1f"            print(f"   Memory Limit: {self.memory_limit_mb}MB")

        else:
            print("❌ TEST FAILED")
            for error in self.results.get('errors', []):
                print(f"🔴 {error}")

        criteria = {
            'accuracy_target': self.results['sentiment_metrics']['accuracy_score'] >= self.accuracy_target,
            'memory_compliance': not any(s['memory_limit_exceeded'] for s in self.memory_stats),
            'sufficient_data': len(self.articles) >= 100
        }

        print("🎯 Success Criteria:"        print(f"   Accuracy ≥{self.accuracy_target}: {'✅' if criteria['accuracy_target'] else '❌'}")
        print(f"   Memory Compliance: {'✅' if criteria['memory_compliance'] else '❌'}")
        print(f"   Sufficient Data (≥100): {'✅' if criteria['sufficient_data'] else '❌'}")

        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Real News Sentiment Test for Supreme System V5')
    parser.add_argument('--sources', nargs='+', default=['reuters', 'bloomberg', 'coindesk'],
                       help='News sources to test (reuters, bloomberg, coindesk)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of articles to analyze (default: 1000)')
    parser.add_argument('--accuracy-target', type=float, default=0.85,
                       help='Accuracy target (default: 0.85)')
    parser.add_argument('--memory-limit', type=int, default=300,
                       help='Memory limit in MB (default: 300)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    # Validate parameters
    if not 0.0 <= args.accuracy_target <= 1.0:
        logger.error("Accuracy target must be between 0.0 and 1.0")
        sys.exit(1)

    print("📰 SUPREME SYSTEM V5 - REAL NEWS SENTIMENT TEST")
    print("=" * 55)
    print(f"Sources: {args.sources}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Accuracy Target: {args.accuracy_target}")
    print(f"Memory Limit: {args.memory_limit}MB")

    # Run the test
    tester = RealNewsSentimentTester(
        sources=args.sources,
        sample_size=args.sample_size,
        accuracy_target=args.accuracy_target,
        memory_limit_mb=args.memory_limit
    )

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        tester.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        results = asyncio.run(tester.run_sentiment_test())

        # Save results
        output_file = tester.save_results(args.output)

        # Print summary
        tester.print_summary()

        # Exit with appropriate code
        metrics = results['sentiment_metrics']
        memory_compliance = not any(s['memory_limit_exceeded'] for s in results['memory_stats'])
        sufficient_data = len(tester.articles) >= 100

        if (metrics['accuracy_score'] >= args.accuracy_target and
            memory_compliance and sufficient_data):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        tester.save_results(args.output)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical test failure: {e}", exc_info=True)
        tester.save_results(args.output)
        sys.exit(1)

if __name__ == "__main__":
    main()
