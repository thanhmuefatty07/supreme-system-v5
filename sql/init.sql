# Supreme System V5 - SQL Init for Postgres
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
