# Supreme System V5 - SQL Init for Postgres
-- Core trading tables
CREATE TABLE IF NOT EXISTS trades (
  id SERIAL PRIMARY KEY,
  trade_id VARCHAR(64) UNIQUE,
  symbol VARCHAR(32) NOT NULL,
  side VARCHAR(8) NOT NULL,
  size NUMERIC(18,8) NOT NULL,
  entry_price NUMERIC(18,8) NOT NULL,
  stop_loss NUMERIC(18,8),
  take_profit NUMERIC(18,8),
  pnl NUMERIC(18,8) DEFAULT 0,
  reason TEXT,
  ts TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(32) NOT NULL,
  price NUMERIC(18,8) NOT NULL,
  volume_24h NUMERIC(24,8),
  change_24h NUMERIC(8,4),
  high_24h NUMERIC(18,8),
  low_24h NUMERIC(18,8),
  bid NUMERIC(18,8),
  ask NUMERIC(18,8),
  source VARCHAR(32),
  quality_score NUMERIC(6,4),
  ts TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data Fabric Quality Tracking Tables

-- Quality history per symbol/source combination
CREATE TABLE IF NOT EXISTS quality_history (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(32) NOT NULL,
  source VARCHAR(32) NOT NULL,
  quality_score NUMERIC(6,4) NOT NULL,
  response_time_ms INTEGER,
  data_freshness_seconds INTEGER,
  error_count INTEGER DEFAULT 0,
  total_requests INTEGER DEFAULT 1,
  success_rate NUMERIC(6,4),
  spread_bps INTEGER, -- Bid-ask spread in basis points
  volume_score NUMERIC(6,4), -- Volume quality indicator
  ts TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Source health tracking
CREATE TABLE IF NOT EXISTS source_health (
  id SERIAL PRIMARY KEY,
  source VARCHAR(32) NOT NULL UNIQUE,
  status VARCHAR(16) DEFAULT 'healthy', -- healthy, degraded, down
  uptime_percent NUMERIC(6,4) DEFAULT 1.0,
  avg_response_time_ms INTEGER,
  avg_quality_score NUMERIC(6,4),
  total_requests BIGINT DEFAULT 0,
  successful_requests BIGINT DEFAULT 0,
  consecutive_failures INTEGER DEFAULT 0,
  last_failure_ts TIMESTAMP WITH TIME ZONE,
  last_success_ts TIMESTAMP WITH TIME ZONE,
  circuit_breaker_active BOOLEAN DEFAULT FALSE,
  circuit_breaker_until TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Symbol quality aggregation
CREATE TABLE IF NOT EXISTS symbol_quality (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(32) NOT NULL UNIQUE,
  overall_quality_score NUMERIC(6,4),
  best_source VARCHAR(32),
  source_count INTEGER DEFAULT 0,
  avg_spread_bps INTEGER,
  total_volume_score NUMERIC(6,4),
  last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data Fabric performance metrics
CREATE TABLE IF NOT EXISTS data_fabric_metrics (
  id SERIAL PRIMARY KEY,
  metric_name VARCHAR(64) NOT NULL,
  metric_value NUMERIC(18,8),
  metric_unit VARCHAR(16),
  tags JSONB, -- Additional metadata
  ts TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_quality_history_symbol_ts ON quality_history(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_quality_history_source_ts ON quality_history(source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_source_health_status ON source_health(status);
CREATE INDEX IF NOT EXISTS idx_data_fabric_metrics_name_ts ON data_fabric_metrics(metric_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_ts ON market_data(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, ts DESC);

-- Update trigger for source_health
CREATE OR REPLACE FUNCTION update_source_health_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_source_health_updated_at_trigger
    BEFORE UPDATE ON source_health
    FOR EACH ROW EXECUTE FUNCTION update_source_health_updated_at();
