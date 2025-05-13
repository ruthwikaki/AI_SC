#!/usr/bin/env python3
"""
Database mirroring script for Supply Chain LLM.
This script creates a mirror of a client database for development and testing.
"""

import argparse
import os
import sys
import yaml
import logging
from time import time

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import the necessary modules
from app.db.mirroring.schema_replicator import SchemaReplicator
from app.db.mirroring.data_syncer import DataSyncer
from app.db.connectors.postgres import PostgresConnector
from app.db.connectors.mysql import MySQLConnector
from app.db.connectors.sqlserver import SQLServerConnector
from app.db.connectors.oracle import OracleConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_mirror')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_connector(db_type, connection_params):
    """Get the appropriate database connector."""
    if db_type == 'postgres':
        return PostgresConnector(**connection_params)
    elif db_type == 'mysql':
        return MySQLConnector(**connection_params)
    elif db_type == 'sqlserver':
        return SQLServerConnector(**connection_params)
    elif db_type == 'oracle':
        return OracleConnector(**connection_params)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def main():
    parser = argparse.ArgumentParser(description='Mirror a database for development and testing.')
    parser.add_argument('--config', required=True, help='Path to the connection configuration file')
    parser.add_argument('--source', required=True, help='Source connection name')
    parser.add_argument('--target', required=True, help='Target connection name')
    parser.add_argument('--tables', nargs='+', help='Specific tables to mirror (if not specified, all tables are mirrored)')
    parser.add_argument('--schema-only', action='store_true', help='Mirror schema only, not data')
    parser.add_argument('--data-limit', type=int, default=1000, help='Limit the number of rows per table (default: 1000)')
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)
    
    # Get source and target connection configs
    source_config = next((c for c in config['connections'] if c['name'] == args.source), None)
    target_config = next((c for c in config['connections'] if c['name'] == args.target), None)
    
    if not source_config:
        logger.error(f"Source connection '{args.source}' not found in config")
        sys.exit(1)
        
    if not target_config:
        logger.error(f"Target connection '{args.target}' not found in config")
        sys.exit(1)
    
    # Create connectors
    source_connector = get_connector(source_config['type'], source_config['params'])
    target_connector = get_connector(target_config['type'], target_config['params'])
    
    # Mirror schema
    logger.info(f"Mirroring schema from {args.source} to {args.target}")
    schema_replicator = SchemaReplicator(source_connector, target_connector)
    
    if args.tables:
        schema_replicator.replicate_schema(tables=args.tables)
    else:
        schema_replicator.replicate_schema()
    
    # Mirror data if requested
    if not args.schema_only:
        logger.info(f"Mirroring data from {args.source} to {args.target}")
        data_syncer = DataSyncer(source_connector, target_connector)
        
        if args.tables:
            data_syncer.sync_data(tables=args.tables, limit=args.data_limit)
        else:
            data_syncer.sync_data(limit=args.data_limit)
    
    logger.info("Database mirroring completed successfully")

if __name__ == "__main__":
    start_time = time()
    main()
    logger.info(f"Total execution time: {time() - start_time:.2f} seconds")