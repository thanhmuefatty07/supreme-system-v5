#!/usr/bin/env python3
"""
Checkpoint C — Realistic NLP Sentiment (CPU-only, guarded)
- DistilBERT CPU-only
- Batch nhỏ theo ngân sách
- Guard < 300MB (ước lượng)
"""
from __future__ import annotations
from typing import List

class RealisticSentimentAnalyzer:
    def __init__(self, memory_budget_mb: int = 300):
        self.memory_budget_mb = memory_budget_mb
        self.model = None
        self.tokenizer = None

    async def load(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except Exception as e:
            raise RuntimeError(f"Transformers not available: {e}")
        name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        # NOTE: In production, consider int8 quantization or torch.set_num_threads limits

    def analyze_batch(self, texts: List[str]) -> List[float]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        import torch
        import numpy as np
        scores: List[float] = []
        bs = max(1, min(16, self.memory_budget_mb // 10))
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                out = self.model(**encoded)
                prob = out.logits.softmax(dim=-1)[:, 1]
                scores.extend(prob.cpu().numpy().astype(float).tolist())
        return scores
