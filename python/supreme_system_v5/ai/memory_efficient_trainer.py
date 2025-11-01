"""
Supreme System V5 - Memory-Efficient AI System
Advanced ML models optimized for i3-4GB + Oracle Cloud ARM64
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import gc
import psutil
import os
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class AIModelType(Enum):
    SENTIMENT_ANALYSIS = "sentiment"
    PRICE_PREDICTION = "prediction"
    RISK_ASSESSMENT = "risk"
    NEWS_CLASSIFICATION = "news"
    PATTERN_RECOGNITION = "pattern"

class MemoryOptimization(Enum):
    AGGRESSIVE = "aggressive"  # 512MB limit
    BALANCED = "balanced"     # 1GB limit
    PERFORMANCE = "performance"  # 2GB limit (Oracle only)

@dataclass
class AIModelConfig:
    model_type: AIModelType
    memory_limit_mb: int
    batch_size: int
    max_sequence_length: int
    learning_rate: float
    optimization_level: MemoryOptimization

@dataclass
class AIModelMetrics:
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_score: float
    confidence_score: float
    predictions_count: int

class MemoryEfficientAISystem:
    """
    Memory-efficient AI system for i3-4GB systems with Oracle Cloud integration
    Features: Model quantization, memory pooling, CPU/GPU optimization
    """

    def __init__(self, max_memory_gb: float = 3.0, cloud_integration: bool = True):
        self.max_memory_gb = max_memory_gb
        self.cloud_integration = cloud_integration
        self.device = self._detect_optimal_device()

        # Model storage
        self.models = {}
        self.tokenizers = {}
        self.optimizers = {}

        # Memory management
        self.memory_pool = {}
        self.scaler = GradScaler() if torch.cuda.is_available() else None

        # Performance metrics
        self.metrics_history = []
        self.current_metrics = AIModelMetrics(0, 0, 0, 0, 0)

        # Cloud integration
        self.oracle_endpoint = os.getenv('ORACLE_AI_ENDPOINT')
        self.oracle_api_key = os.getenv('ORACLE_API_KEY')

        # Model configurations for different memory limits
        self.model_configs = {
            MemoryOptimization.AGGRESSIVE: {
                AIModelType.SENTIMENT_ANALYSIS: AIModelConfig(
                    AIModelType.SENTIMENT_ANALYSIS, 256, 1, 128, 0.001, MemoryOptimization.AGGRESSIVE
                ),
                AIModelType.PRICE_PREDICTION: AIModelConfig(
                    AIModelType.PRICE_PREDICTION, 128, 8, 60, 0.001, MemoryOptimization.AGGRESSIVE
                ),
                AIModelType.RISK_ASSESSMENT: AIModelConfig(
                    AIModelType.RISK_ASSESSMENT, 64, 1, 20, 0.001, MemoryOptimization.AGGRESSIVE
                )
            },
            MemoryOptimization.BALANCED: {
                AIModelType.SENTIMENT_ANALYSIS: AIModelConfig(
                    AIModelType.SENTIMENT_ANALYSIS, 512, 4, 256, 0.001, MemoryOptimization.BALANCED
                ),
                AIModelType.PRICE_PREDICTION: AIModelConfig(
                    AIModelType.PRICE_PREDICTION, 256, 16, 120, 0.001, MemoryOptimization.BALANCED
                ),
                AIModelType.PATTERN_RECOGNITION: AIModelConfig(
                    AIModelType.PATTERN_RECOGNITION, 128, 8, 100, 0.001, MemoryOptimization.BALANCED
                )
            },
            MemoryOptimization.PERFORMANCE: {
                AIModelType.SENTIMENT_ANALYSIS: AIModelConfig(
                    AIModelType.SENTIMENT_ANALYSIS, 1024, 8, 512, 0.001, MemoryOptimization.PERFORMANCE
                ),
                AIModelType.PRICE_PREDICTION: AIModelConfig(
                    AIModelType.PRICE_PREDICTION, 512, 32, 240, 0.001, MemoryOptimization.PERFORMANCE
                ),
                AIModelType.NEWS_CLASSIFICATION: AIModelConfig(
                    AIModelType.NEWS_CLASSIFICATION, 256, 4, 512, 0.001, MemoryOptimization.PERFORMANCE
                )
            }
        }

        # Determine optimal memory optimization level
        self.memory_optimization = self._determine_memory_optimization()

    def _detect_optimal_device(self) -> torch.device:
        """Detect optimal compute device"""
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 2.0:  # At least 2GB GPU memory
                return torch.device('cuda:0')
            else:
                print("⚠️ GPU memory too low, using CPU")
                return torch.device('cpu')
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon support
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _determine_memory_optimization(self) -> MemoryOptimization:
        """Determine optimal memory optimization level"""
        system_memory = psutil.virtual_memory().total / (1024**3)

        if system_memory <= 4.0:  # i3-4GB systems
            return MemoryOptimization.AGGRESSIVE
        elif system_memory <= 8.0:  # Standard systems
            return MemoryOptimization.BALANCED
        else:  # Oracle Cloud or high-end systems
            return MemoryOptimization.PERFORMANCE

    async def initialize_models(self) -> bool:
        """Initialize AI models with memory optimization"""
        try:
            print(f"🤖 Initializing AI models (Memory: {self.memory_optimization.value})")

            # Initialize based on optimization level
            configs = self.model_configs[self.memory_optimization]

            # Always initialize sentiment analysis (critical for trading)
            if AIModelType.SENTIMENT_ANALYSIS in configs:
                await self._load_sentiment_model(configs[AIModelType.SENTIMENT_ANALYSIS])

            # Initialize price prediction (critical for trading)
            if AIModelType.PRICE_PREDICTION in configs:
                self._create_price_predictor(configs[AIModelType.PRICE_PREDICTION])

            # Initialize risk assessment
            if AIModelType.RISK_ASSESSMENT in configs:
                self._create_risk_assessor(configs[AIModelType.RISK_ASSESSMENT])

            # Initialize pattern recognition if memory allows
            if self.memory_optimization in [MemoryOptimization.BALANCED, MemoryOptimization.PERFORMANCE]:
                if AIModelType.PATTERN_RECOGNITION in configs:
                    self._create_pattern_recognizer(configs[AIModelType.PATTERN_RECOGNITION])

            # Initialize news classification for Oracle Cloud
            if self.memory_optimization == MemoryOptimization.PERFORMANCE and self.cloud_integration:
                if AIModelType.NEWS_CLASSIFICATION in configs:
                    await self._load_news_classifier(configs[AIModelType.NEWS_CLASSIFICATION])

            memory_usage = self._get_memory_usage()
            print(".1f"            return True

        except Exception as e:
            print(f"❌ AI initialization failed: {e}")
            return False

    async def _load_sentiment_model(self, config: AIModelConfig):
        """Load lightweight sentiment analysis model"""
        try:
            # Use DistilBERT for memory efficiency
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            print(f"📥 Loading sentiment model: {model_name}")

            # Load tokenizer
            self.tokenizers['sentiment'] = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=config.max_sequence_length
            )

            # Load model with memory optimization
            self.models['sentiment'] = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )

            # Move to device and optimize
            self.models['sentiment'].to(self.device)
            self.models['sentiment'].eval()

            # Apply memory optimizations
            if hasattr(self.models['sentiment'], 'gradient_checkpointing_enable'):
                self.models['sentiment'].gradient_checkpointing_enable()

            print("✅ Sentiment model loaded and optimized")

        except Exception as e:
            print(f"⚠️ Sentiment model loading failed: {e}")
            # Fallback to simple sentiment analysis
            self.models['sentiment'] = None

    def _create_price_predictor(self, config: AIModelConfig):
        """Create memory-efficient LSTM price predictor"""
        try:
            class CompactLSTM(nn.Module):
                def __init__(self, input_size=5, hidden_size=32, num_layers=2,
                             output_size=1, dropout=0.2):
                    super(CompactLSTM, self).__init__()

                    self.hidden_size = hidden_size
                    self.num_layers = num_layers

                    # Compact LSTM layers
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        dropout=dropout, batch_first=True,
                        bidirectional=False
                    )

                    # Compact fully connected layers
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size // 2, output_size),
                        nn.Tanh()  # Output between -1 and 1
                    )

                def forward(self, x):
                    # LSTM forward pass
                    lstm_out, _ = self.lstm(x)

                    # Take the last output
                    last_output = lstm_out[:, -1, :]

                    # Fully connected layers
                    prediction = self.fc(last_output)

                    return prediction

            self.models['price_predictor'] = CompactLSTM().to(self.device)

            # Initialize weights
            for name, param in self.models['price_predictor'].named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            # Create optimizer
            self.optimizers['price_predictor'] = torch.optim.Adam(
                self.models['price_predictor'].parameters(),
                lr=config.learning_rate
            )

            print("✅ Price predictor created")

        except Exception as e:
            print(f"⚠️ Price predictor creation failed: {e}")
            self.models['price_predictor'] = None

    def _create_risk_assessor(self, config: AIModelConfig):
        """Create lightweight risk assessment model"""
        try:
            class RiskAssessor(nn.Module):
                def __init__(self, input_features=10):
                    super(RiskAssessor, self).__init__()

                    self.network = nn.Sequential(
                        nn.Linear(input_features, 16),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(8, 3),  # Low, Medium, High risk
                        nn.Softmax(dim=1)
                    )

                def forward(self, x):
                    return self.network(x)

            self.models['risk_assessor'] = RiskAssessor().to(self.device)
            print("✅ Risk assessor created")

        except Exception as e:
            print(f"⚠️ Risk assessor creation failed: {e}")
            self.models['risk_assessor'] = None

    def _create_pattern_recognizer(self, config: AIModelConfig):
        """Create pattern recognition model for technical analysis"""
        try:
            class PatternRecognizer(nn.Module):
                def __init__(self, sequence_length=100, num_features=5):
                    super(PatternRecognizer, self).__init__()

                    self.conv1 = nn.Conv1d(num_features, 16, kernel_size=5, stride=1, padding=2)
                    self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
                    self.pool = nn.AdaptiveAvgPool1d(1)

                    self.fc = nn.Sequential(
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(16, 4),  # Bullish, Bearish, Sideways, Volatile
                        nn.Softmax(dim=1)
                    )

                def forward(self, x):
                    # x shape: (batch, sequence, features)
                    x = x.transpose(1, 2)  # (batch, features, sequence)

                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x).squeeze(-1)

                    return self.fc(x)

            self.models['pattern_recognizer'] = PatternRecognizer().to(self.device)
            print("✅ Pattern recognizer created")

        except Exception as e:
            print(f"⚠️ Pattern recognizer creation failed: {e}")
            self.models['pattern_recognizer'] = None

    async def _load_news_classifier(self, config: AIModelConfig):
        """Load news classification model (Oracle Cloud only)"""
        if not self.cloud_integration:
            return

        try:
            # Use lightweight news classification model
            model_name = "facebook/bart-large-mnli"

            print(f"📥 Loading news classifier: {model_name}")

            self.models['news_classifier'] = pipeline(
                "zero-shot-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.device.index if self.device.type == 'cuda' else -1
            )

            print("✅ News classifier loaded")

        except Exception as e:
            print(f"⚠️ News classifier loading failed: {e}")
            self.models['news_classifier'] = None

    async def analyze_sentiment(self, texts: List[str]) -> Dict:
        """Analyze sentiment with memory management"""
        start_time = time.time()

        if not self.models.get('sentiment'):
            return self._fallback_sentiment_analysis(texts)

        try:
            # Limit processing for memory efficiency
            texts = texts[:5]  # Max 5 texts at once
            sentiments = []

            for text in texts:
                text = text[:256]  # Truncate for memory

                # Tokenize
                inputs = self.tokenizers['sentiment'](
                    text, return_tensors='pt',
                    max_length=128, truncation=True, padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference with memory optimization
                with torch.no_grad():
                    if self.scaler and self.device.type == 'cuda':
                        with autocast():
                            outputs = self.models['sentiment'](**inputs)
                    else:
                        outputs = self.models['sentiment'](**inputs)

                # Get prediction
                predictions = torch.softmax(outputs.logits, dim=1)
                sentiment_score = predictions[0][1].item()  # Positive class
                confidence = abs(sentiment_score - 0.5) * 2

                sentiments.append({
                    'text': text[:50],
                    'sentiment_score': sentiment_score,
                    'confidence': confidence
                })

                # Memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Aggregate results
            avg_sentiment = np.mean([s['sentiment_score'] for s in sentiments])
            avg_confidence = np.mean([s['confidence'] for s in sentiments])

            # Update metrics
            self._update_metrics(time.time() - start_time, len(texts))

            return {
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'individual_sentiments': sentiments,
                'processing_time_ms': (time.time() - start_time) * 1000
            }

        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return self._fallback_sentiment_analysis(texts)

    def _fallback_sentiment_analysis(self, texts: List[str]) -> Dict:
        """Simple rule-based sentiment analysis fallback"""
        positive_words = ['bull', 'bullish', 'up', 'rise', 'gain', 'profit', 'buy', 'long']
        negative_words = ['bear', 'bearish', 'down', 'fall', 'loss', 'sell', 'short', 'crash']

        sentiments = []
        for text in texts:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                sentiment_score = 0.7
            elif negative_count > positive_count:
                sentiment_score = 0.3
            else:
                sentiment_score = 0.5

            sentiments.append({
                'text': text[:50],
                'sentiment_score': sentiment_score,
                'confidence': 0.5
            })

        avg_sentiment = np.mean([s['sentiment_score'] for s in sentiments])

        return {
            'sentiment_score': avg_sentiment,
            'confidence': 0.5,
            'individual_sentiments': sentiments,
            'method': 'fallback'
        }

    def predict_price_movement(self, price_history: List[float],
                             volume_history: List[float],
                             technical_indicators: Dict = None) -> Dict:
        """Predict price movement using LSTM"""
        start_time = time.time()

        if not self.models.get('price_predictor'):
            return self._fallback_price_prediction(price_history)

        try:
            # Prepare sequence data
            sequence_length = 60  # Last 60 data points
            if len(price_history) < sequence_length:
                return self._fallback_price_prediction(price_history)

            # Take recent data
            recent_prices = np.array(price_history[-sequence_length:])
            recent_volumes = np.array(volume_history[-sequence_length:])

            # Normalize
            price_mean, price_std = recent_prices.mean(), recent_prices.std()
            volume_mean, volume_std = recent_volumes.mean(), recent_volumes.std()

            if price_std == 0 or volume_std == 0:
                return self._fallback_price_prediction(price_history)

            normalized_prices = (recent_prices - price_mean) / price_std
            normalized_volumes = (recent_volumes - volume_mean) / volume_std

            # Create features
            price_changes = np.diff(normalized_prices, prepend=normalized_prices[0])
            volume_changes = np.diff(normalized_volumes, prepend=normalized_volumes[0])

            # Add technical indicators if available
            features = [normalized_prices, normalized_volumes, price_changes, volume_changes]
            if technical_indicators:
                # Add RSI, MACD, etc.
                if 'rsi' in technical_indicators:
                    rsi_values = technical_indicators['rsi'][-sequence_length:]
                    features.append(np.array(rsi_values) / 100)
                if 'macd' in technical_indicators:
                    macd_values = technical_indicators['macd'][-sequence_length:]
                    features.append(np.array(macd_values))

            # Stack features
            feature_array = np.column_stack(features)

            # Convert to tensor
            input_tensor = torch.FloatTensor(feature_array).unsqueeze(0).to(self.device)

            # Prediction
            self.models['price_predictor'].eval()
            with torch.no_grad():
                if self.scaler and self.device.type == 'cuda':
                    with autocast():
                        prediction = self.models['price_predictor'](input_tensor)
                else:
                    prediction = self.models['price_predictor'](input_tensor)

            prediction_value = prediction.cpu().numpy()[0][0]
            confidence = min(abs(prediction_value) * 2, 1.0)

            # Update metrics
            self._update_metrics(time.time() - start_time, 1)

            return {
                'prediction': float(prediction_value),
                'confidence': float(confidence),
                'direction': 'bullish' if prediction_value > 0.1 else 'bearish' if prediction_value < -0.1 else 'neutral',
                'model_version': 'compact_lstm_v1.0',
                'processing_time_ms': (time.time() - start_time) * 1000
            }

        except Exception as e:
            print(f"⚠️ Price prediction error: {e}")
            return self._fallback_price_prediction(price_history)

    def _fallback_price_prediction(self, price_history: List[float]) -> Dict:
        """Simple trend-based price prediction fallback"""
        if len(price_history) < 10:
            return {'prediction': 0.0, 'confidence': 0.0, 'direction': 'neutral'}

        # Simple trend analysis
        recent_prices = price_history[-10:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Volatility as confidence
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        confidence = max(0, 1 - volatility * 10)  # Lower volatility = higher confidence

        return {
            'prediction': trend,
            'confidence': confidence,
            'direction': 'bullish' if trend > 0.01 else 'bearish' if trend < -0.01 else 'neutral',
            'method': 'fallback'
        }

    def assess_risk(self, portfolio_metrics: Dict) -> Dict:
        """Assess portfolio risk using AI"""
        start_time = time.time()

        if not self.models.get('risk_assessor'):
            return self._fallback_risk_assessment(portfolio_metrics)

        try:
            # Extract risk features
            features = [
                portfolio_metrics.get('current_drawdown', 0) / 100,
                portfolio_metrics.get('daily_pnl_percent', 0) / 100,
                portfolio_metrics.get('position_count', 0) / 10,
                portfolio_metrics.get('leverage', 1) / 10,
                portfolio_metrics.get('win_rate', 50) / 100,
                portfolio_metrics.get('avg_trade_duration', 300) / 3600,  # Hours
                portfolio_metrics.get('volatility', 0.02) * 100,
                portfolio_metrics.get('sharpe_ratio', 1) / 5,
                portfolio_metrics.get('max_drawdown', 0) / 100,
                min(len(portfolio_metrics.get('open_positions', [])), 10) / 10
            ]

            # Convert to tensor
            feature_tensor = torch.FloatTensor([features]).to(self.device)

            # Risk assessment
            self.models['risk_assessor'].eval()
            with torch.no_grad():
                risk_probabilities = self.models['risk_assessor'](feature_tensor)
                risk_probs = risk_probabilities.cpu().numpy()[0]

            # Interpret results
            risk_levels = ['low', 'medium', 'high']
            max_prob_idx = np.argmax(risk_probs)
            risk_level = risk_levels[max_prob_idx]
            risk_score = risk_probs[max_prob_idx]

            # Update metrics
            self._update_metrics(time.time() - start_time, 1)

            return {
                'risk_level': risk_level,
                'risk_score': float(risk_score),
                'risk_distribution': {
                    'low': float(risk_probs[0]),
                    'medium': float(risk_probs[1]),
                    'high': float(risk_probs[2])
                },
                'processing_time_ms': (time.time() - start_time) * 1000
            }

        except Exception as e:
            print(f"⚠️ Risk assessment error: {e}")
            return self._fallback_risk_assessment(portfolio_metrics)

    def _fallback_risk_assessment(self, portfolio_metrics: Dict) -> Dict:
        """Rule-based risk assessment fallback"""
        drawdown = portfolio_metrics.get('current_drawdown', 0)
        volatility = portfolio_metrics.get('volatility', 0.02)

        # Simple risk scoring
        risk_score = min(drawdown / 100 + volatility * 50, 1.0)

        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'method': 'fallback'
        }

    async def analyze_news_impact(self, news_articles: List[Dict]) -> Dict:
        """Analyze news impact using advanced NLP (Oracle Cloud)"""
        if not self.models.get('news_classifier') or not self.cloud_integration:
            return self._fallback_news_analysis(news_articles)

        try:
            results = []
            for article in news_articles[:3]:  # Limit for memory
                text = article.get('title', '') + ' ' + article.get('summary', '')

                # Classify sentiment and impact
                result = self.models['news_classifier'](
                    text,
                    candidate_labels=['positive', 'negative', 'neutral'],
                    hypothesis_template="This news is {} for the market."
                )

                results.append({
                    'title': article.get('title', ''),
                    'sentiment': result['labels'][0],
                    'confidence': result['scores'][0],
                    'impact_score': self._calculate_news_impact(result)
                })

            # Aggregate results
            avg_impact = np.mean([r['impact_score'] for r in results])

            return {
                'overall_impact': 'positive' if avg_impact > 0.1 else 'negative' if avg_impact < -0.1 else 'neutral',
                'impact_score': avg_impact,
                'individual_analysis': results,
                'method': 'advanced_nlp'
            }

        except Exception as e:
            print(f"⚠️ News analysis error: {e}")
            return self._fallback_news_analysis(news_articles)

    def _fallback_news_analysis(self, news_articles: List[Dict]) -> Dict:
        """Simple keyword-based news analysis"""
        positive_keywords = ['surge', 'rally', 'bullish', 'upgrade', 'partnership', 'adoption']
        negative_keywords = ['crash', 'dump', 'bearish', 'downgrade', 'hack', 'ban', 'sell-off']

        total_impact = 0
        analyzed_articles = []

        for article in news_articles[:5]:
            text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()

            positive_score = sum(1 for word in positive_keywords if word in text)
            negative_score = sum(1 for word in negative_keywords if word in text)

            impact_score = (positive_score - negative_score) * 0.1
            total_impact += impact_score

            analyzed_articles.append({
                'title': article.get('title', ''),
                'impact_score': impact_score,
                'sentiment': 'positive' if impact_score > 0 else 'negative' if impact_score < 0 else 'neutral'
            })

        avg_impact = total_impact / max(len(analyzed_articles), 1)

        return {
            'overall_impact': 'positive' if avg_impact > 0.05 else 'negative' if avg_impact < -0.05 else 'neutral',
            'impact_score': avg_impact,
            'individual_analysis': analyzed_articles,
            'method': 'keyword_analysis'
        }

    def _calculate_news_impact(self, classification_result: Dict) -> float:
        """Calculate numerical impact from classification"""
        label_scores = dict(zip(classification_result['labels'], classification_result['scores']))

        impact = 0
        if 'positive' in label_scores:
            impact += label_scores['positive'] * 0.5
        if 'negative' in label_scores:
            impact -= label_scores['negative'] * 0.5

        return impact

    async def train_on_new_data(self, training_data: Dict, model_type: AIModelType) -> bool:
        """Train model on new data (Oracle Cloud only for memory reasons)"""
        if not self.cloud_integration or self.memory_optimization == MemoryOptimization.AGGRESSIVE:
            print("⚠️ Training not available on local i3-4GB system")
            return False

        try:
            if model_type == AIModelType.PRICE_PREDICTION:
                return await self._train_price_predictor(training_data)
            elif model_type == AIModelType.RISK_ASSESSMENT:
                return await self._train_risk_assessor(training_data)
            else:
                print(f"⚠️ Training not supported for {model_type.value}")
                return False

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    async def _train_price_predictor(self, training_data: Dict) -> bool:
        """Train price predictor on new market data"""
        try:
            # Prepare training data
            prices = training_data.get('prices', [])
            volumes = training_data.get('volumes', [])
            labels = training_data.get('labels', [])  # Price movements

            if len(prices) < 100 or len(prices) != len(labels):
                return False

            # Create sequences
            sequence_length = 60
            X, y = [], []

            for i in range(len(prices) - sequence_length):
                price_seq = prices[i:i+sequence_length]
                volume_seq = volumes[i:i+sequence_length]

                # Normalize
                price_mean, price_std = np.mean(price_seq), np.std(price_seq)
                if price_std == 0:
                    continue

                normalized_prices = (np.array(price_seq) - price_mean) / price_std
                normalized_volumes = np.array(volume_seq) / max(np.mean(volume_seq), 1)

                # Features
                features = np.column_stack([
                    normalized_prices,
                    normalized_volumes,
                    np.diff(normalized_prices, prepend=normalized_prices[0]),
                    np.diff(normalized_volumes, prepend=normalized_volumes[0])
                ])

                X.append(features)
                y.append(labels[i+sequence_length])

            if not X:
                return False

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Training loop
            self.models['price_predictor'].train()
            criterion = nn.MSELoss()

            for epoch in range(5):  # Limited epochs for memory
                self.optimizers['price_predictor'].zero_grad()

                outputs = self.models['price_predictor'](X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)

                if self.scaler and self.device.type == 'cuda':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizers['price_predictor'])
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizers['price_predictor'].step()

                if (epoch + 1) % 2 == 0:
                    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

            print("✅ Price predictor training completed")
            return True

        except Exception as e:
            print(f"❌ Price predictor training failed: {e}")
            return False

    async def _train_risk_assessor(self, training_data: Dict) -> bool:
        """Train risk assessor on historical risk data"""
        try:
            # Prepare training data
            features = training_data.get('features', [])  # Risk features
            labels = training_data.get('labels', [])      # Risk levels (0,1,2)

            if len(features) < 50 or len(features) != len(labels):
                return False

            # Convert to tensors
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.LongTensor(labels).to(self.device)

            # Training loop
            self.models['risk_assessor'].train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.models['risk_assessor'].parameters(), lr=0.001)

            for epoch in range(10):
                optimizer.zero_grad()

                outputs = self.models['risk_assessor'](X_tensor)
                loss = criterion(outputs, y_tensor)

                if self.scaler and self.device.type == 'cuda':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 3 == 0:
                    print(f"Risk assessor epoch {epoch+1}, Loss: {loss.item():.4f}")

            print("✅ Risk assessor training completed")
            return True

        except Exception as e:
            print(f"❌ Risk assessor training failed: {e}")
            return False

    def _update_metrics(self, inference_time: float, predictions_count: int):
        """Update performance metrics"""
        memory_usage = self._get_memory_usage()

        self.current_metrics = AIModelMetrics(
            inference_time_ms=inference_time * 1000,
            memory_usage_mb=memory_usage,
            accuracy_score=0.0,  # Would be calculated from validation data
            confidence_score=0.0,  # Would be calculated from predictions
            predictions_count=self.current_metrics.predictions_count + predictions_count
        )

        # Keep history
        self.metrics_history.append(self.current_metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def get_performance_report(self) -> Dict:
        """Generate AI performance report"""
        if not self.metrics_history:
            return {'error': 'No performance data available'}

        recent_metrics = self.metrics_history[-100:]

        return {
            'total_predictions': sum(m.predictions_count for m in recent_metrics),
            'avg_inference_time_ms': np.mean([m.inference_time_ms for m in recent_metrics]),
            'avg_memory_usage_mb': np.mean([m.memory_usage_mb for m in recent_metrics]),
            'peak_memory_usage_mb': max(m.memory_usage_mb for m in recent_metrics),
            'models_loaded': list(self.models.keys()),
            'device': str(self.device),
            'memory_optimization': self.memory_optimization.value,
            'cloud_integration': self.cloud_integration
        }

    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # Python garbage collection
            gc.collect()

            # Clear model caches if they exist
            for model in self.models.values():
                if hasattr(model, 'cache') and model.cache:
                    model.cache.clear()

            print("🧹 AI memory cleanup completed")

        except Exception as e:
            print(f"⚠️ Memory cleanup error: {e}")

# Global AI system instance
ai_system = MemoryEfficientAISystem()
