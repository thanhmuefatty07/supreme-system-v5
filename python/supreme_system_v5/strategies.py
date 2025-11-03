@@ -147,12 +147,12 @@ class ScalpingStrategy:
         Returns:
             Trading signal dict or None if no action needed
         """
         start_time = time.time()
 
-        # ULTRA OPTIMIZATION: Gate with SmartEventProcessor by price Δ%, volume spike, or max gap time
-        # TEMPORARILY DISABLED: if not self.event_processor.should_process(price, volume, timestamp):
-        # TEMPORARILY DISABLED:     # Event filtered - no processing needed (70-90% reduction)
-        # TEMPORARILY DISABLED:     return None
+        # ULTRA OPTIMIZATION: Gate with SmartEventProcessor by cadence (30–60s ±10% jitter), price/volume significance, or max gap time
+        try:
+            if not self.event_processor.should_process(price, volume, timestamp):
+                return None
+        except Exception:
+            pass  # fail-open to avoid blocking strategy on gating errors
 
         # ULTRA OPTIMIZATION: Maintain bounded price history
         self.price_history.append(price)
 
         # OPTIMIZED: Analyzer handles remaining processing
         processed = self.analyzer.add_price_data(price, volume, timestamp)
