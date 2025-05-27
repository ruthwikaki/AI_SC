-- deployment/docker/init.sql

-- SSA Database Initialization Script
-- This script runs when the PostgreSQL container is first created

-- Create database if not exists (runs as superuser)
SELECT 'CREATE DATABASE ssa_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ssa_db')\gexec

-- Connect to the database
\c ssa_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE order_status AS ENUM (
        'draft', 'submitted', 'approved', 'in_progress',
        'partially_shipped', 'shipped', 'delivered',
        'cancelled', 'returned', 'closed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE shipment_status AS ENUM (
        'pending', 'ready', 'in_transit', 'delivered',
        'returned', 'lost'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create function for updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create function for audit logging
CREATE OR REPLACE FUNCTION create_audit_log()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (
        user_id,
        action,
        resource_type,
        resource_id,
        details,
        ip_address,
        user_agent,
        created_at
    ) VALUES (
        current_setting('app.current_user_id', true)::uuid,
        TG_OP,
        TG_TABLE_NAME,
        NEW.id,
        to_jsonb(NEW),
        current_setting('app.current_ip', true),
        current_setting('app.current_user_agent', true),
        CURRENT_TIMESTAMP
    );
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create base tables for initial setup
-- Note: Full schema is created by SQLAlchemy/Alembic

-- Create settings table for app configuration
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert default settings
INSERT INTO system_settings (key, value, description) VALUES
    ('app.initialized', 'false', 'Whether the application has been initialized'),
    ('app.version', '"1.0.0"', 'Current application version'),
    ('app.maintenance_mode', 'false', 'Whether the app is in maintenance mode'),
    ('features.multi_tenancy', 'false', 'Enable multi-tenancy features'),
    ('features.audit_log', 'true', 'Enable audit logging'),
    ('features.data_export', 'true', 'Enable data export features'),
    ('analytics.retention_days', '90', 'Days to retain analytics data'),
    ('security.password_min_length', '8', 'Minimum password length'),
    ('security.session_timeout', '86400', 'Session timeout in seconds'),
    ('security.max_login_attempts', '5', 'Maximum login attempts before lockout')
ON CONFLICT (key) DO NOTHING;

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(key);

-- Create materialized view for dashboard metrics (example)
-- This will be populated after tables are created
CREATE MATERIALIZED VIEW IF NOT EXISTS dashboard_metrics AS
SELECT 
    NOW() as last_updated,
    0 as total_suppliers,
    0 as total_products,
    0 as total_orders,
    0 as pending_shipments
WITH NO DATA;

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_metrics;
END;
$$ LANGUAGE plpgsql;

-- Performance optimization settings
ALTER DATABASE ssa_db SET random_page_cost = 1.1;
ALTER DATABASE ssa_db SET effective_cache_size = '4GB';
ALTER DATABASE ssa_db SET shared_buffers = '1GB';
ALTER DATABASE ssa_db SET work_mem = '16MB';
ALTER DATABASE ssa_db SET maintenance_work_mem = '256MB';

-- Create read-only user for analytics (optional)
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'ssa_readonly') THEN
      CREATE ROLE ssa_readonly LOGIN PASSWORD 'readonly_password';
   END IF;
END
$do$;

-- Grant permissions to readonly user (after tables are created)
-- This will be done by the application after migrations

-- Create application user if not exists
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'ssa_user') THEN
      CREATE ROLE ssa_user LOGIN PASSWORD 'ssa_password';
   END IF;
END
$do$;

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON DATABASE ssa_db TO ssa_user;
GRANT ALL ON SCHEMA public TO ssa_user;

-- Maintenance settings
COMMENT ON DATABASE ssa_db IS 'Supply Chain Analytics Platform Database';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'SSA Database initialization completed successfully';
END $$;