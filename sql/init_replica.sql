-- Dashboard read-only database setup
-- Optimized for dashboard queries with minimal impact on trading

-- Create materialized views for dashboard performance
CREATE MATERIALIZED VIEW portfolio_summary AS
SELECT
    CURRENT_TIMESTAMP as last_updated,
    SUM(CASE WHEN position_type = 'LONG' THEN current_value ELSE -current_value END) as total_balance,
    SUM(CASE WHEN position_type = 'LONG' THEN unrealized_pnl ELSE -unrealized_pnl END) as total_pnl,
    SUM(CASE WHEN position_type = 'LONG' THEN current_value ELSE current_value END) as gross_exposure,
    COUNT(DISTINCT symbol) as active_positions,
    AVG(CASE WHEN position_type = 'LONG' THEN unrealized_pnl_percent ELSE unrealized_pnl_percent END) as avg_pnl_percent
FROM positions
WHERE status = 'OPEN';

-- Refresh every 5 minutes via cron job
CREATE UNIQUE INDEX idx_portfolio_summary_refresh ON portfolio_summary (last_updated);

-- Portfolio balance history (for charts)
CREATE MATERIALIZED VIEW portfolio_history AS
SELECT
    DATE_TRUNC('minute', created_at) as timestamp,
    SUM(portfolio_value) as balance,
    SUM(daily_pnl) as daily_pnl,
    SUM(total_pnl) as total_pnl,
    MAX(max_drawdown) as max_drawdown
FROM portfolio_snapshots
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('minute', created_at)
ORDER BY timestamp DESC;

-- Trading activity feed (last 1000 trades)
CREATE MATERIALIZED VIEW recent_trades AS
SELECT
    t.id,
    t.symbol,
    t.side,
    t.quantity,
    t.price,
    t.executed_at,
    t.pnl,
    t.fee,
    s.strategy_name,
    CASE
        WHEN t.pnl > 0 THEN 'profit'
        WHEN t.pnl < 0 THEN 'loss'
        ELSE 'neutral'
    END as result
FROM trades t
LEFT JOIN strategies s ON t.strategy_id = s.id
WHERE t.executed_at >= NOW() - INTERVAL '7 days'
ORDER BY t.executed_at DESC
LIMIT 1000;

-- Performance metrics view
CREATE MATERIALIZED VIEW performance_metrics AS
SELECT
    DATE_TRUNC('day', executed_at) as date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    AVG(pnl) as avg_pnl,
    SUM(pnl) as daily_pnl,
    STDDEV(pnl) as pnl_volatility,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade
FROM trades
WHERE executed_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', executed_at)
ORDER BY date DESC;

-- Indexes for dashboard queries
CREATE INDEX idx_portfolio_snapshots_created_at ON portfolio_snapshots (created_at);
CREATE INDEX idx_trades_executed_at ON trades (executed_at);
CREATE INDEX idx_trades_symbol ON trades (symbol);
CREATE INDEX idx_positions_status ON positions (status);

-- Function to refresh materialized views (called by cron)
CREATE OR REPLACE FUNCTION refresh_dashboard_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_history;
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_trades;
    REFRESH MATERIALIZED VIEW CONCURRENTLY performance_metrics;
END;
$$ LANGUAGE plpgsql;

-- Dashboard user permissions (read-only)
CREATE USER dashboard_reader WITH PASSWORD 'dashboard_readonly_pass';
GRANT CONNECT ON DATABASE trading_readonly TO dashboard_reader;
GRANT USAGE ON SCHEMA public TO dashboard_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO dashboard_reader;
