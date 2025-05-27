# scripts/manage_db.py

"""
Database management utility script
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, create_engine
from alembic import command
from alembic.config import Config

from app.core.config import settings
from app.db.database import engine, Base
from app.db.init_db import init_db
from app.models import *  # Import all models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self):
        self.engine = engine
        self.alembic_cfg = Config("app/db/migrations/alembic.ini")
    
    def create_database(self):
        """Create database if it doesn't exist"""
        db_name = settings.DATABASE_URL.split('/')[-1].split('?')[0]
        
        # Connect to PostgreSQL without specifying database
        temp_url = settings.DATABASE_URL.rsplit('/', 1)[0] + '/postgres'
        temp_engine = create_engine(temp_url)
        
        with temp_engine.connect() as conn:
            # Check if database exists
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name}
            ).fetchone()
            
            if not exists:
                # Create database
                conn.execute(text("COMMIT"))  # Exit transaction
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database already exists: {db_name}")
        
        temp_engine.dispose()
    
    def drop_database(self, force: bool = False):
        """Drop database"""
        if not force:
            confirm = input("Are you sure you want to drop the database? This cannot be undone! (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Database drop cancelled")
                return
        
        db_name = settings.DATABASE_URL.split('/')[-1].split('?')[0]
        
        # Connect to PostgreSQL without specifying database
        temp_url = settings.DATABASE_URL.rsplit('/', 1)[0] + '/postgres'
        temp_engine = create_engine(temp_url)
        
        with temp_engine.connect() as conn:
            # Terminate existing connections
            conn.execute(text("COMMIT"))
            conn.execute(text(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """))
            
            # Drop database
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
            logger.info(f"Dropped database: {db_name}")
        
        temp_engine.dispose()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Created all tables")
    
    def drop_tables(self, force: bool = False):
        """Drop all tables"""
        if not force:
            confirm = input("Are you sure you want to drop all tables? (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Table drop cancelled")
                return
        
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Dropped all tables")
    
    def reset_database(self, force: bool = False):
        """Reset database (drop and recreate)"""
        if not force:
            confirm = input("Are you sure you want to reset the database? All data will be lost! (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Database reset cancelled")
                return
        
        logger.info("Resetting database...")
        self.drop_tables(force=True)
        self.create_tables()
        logger.info("Database reset complete")
    
    def init_data(self, sample_data: bool = False):
        """Initialize database with default data"""
        from app.db.init_db import init_db
        init_db(sample_data=sample_data)
        logger.info("Database initialized with default data")
    
    def run_migrations(self):
        """Run database migrations"""
        command.upgrade(self.alembic_cfg, "head")
        logger.info("Migrations completed")
    
    def create_migration(self, message: str):
        """Create a new migration"""
        command.revision(self.alembic_cfg, autogenerate=True, message=message)
        logger.info(f"Created migration: {message}")
    
    def rollback_migration(self, revision: str = "-1"):
        """Rollback migration"""
        command.downgrade(self.alembic_cfg, revision)
        logger.info(f"Rolled back to revision: {revision}")
    
    def get_migration_history(self):
        """Show migration history"""
        command.history(self.alembic_cfg)
    
    def get_current_revision(self):
        """Show current revision"""
        command.current(self.alembic_cfg)
    
    def verify_schema(self):
        """Verify database schema integrity"""
        with self.engine.connect() as conn:
            # Check for required extensions
            extensions = conn.execute(text("""
                SELECT extname FROM pg_extension
                WHERE extname IN ('uuid-ossp', 'pg_trgm')
            """)).fetchall()
            
            installed_extensions = [ext[0] for ext in extensions]
            required_extensions = ['uuid-ossp', 'pg_trgm']
            
            missing = set(required_extensions) - set(installed_extensions)
            if missing:
                logger.warning(f"Missing extensions: {missing}")
            else:
                logger.info("All required extensions installed")
            
            # Check table count
            table_count = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)).scalar()
            
            logger.info(f"Total tables: {table_count}")
            
            # Check for required tables
            core_tables = [
                'users', 'roles', 'permissions', 'suppliers', 'products',
                'inventory', 'orders', 'shipments'
            ]
            
            existing_tables = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                AND table_name = ANY(:tables)
            """), {"tables": core_tables}).fetchall()
            
            existing = [t[0] for t in existing_tables]
            missing_tables = set(core_tables) - set(existing)
            
            if missing_tables:
                logger.warning(f"Missing core tables: {missing_tables}")
            else:
                logger.info("All core tables exist")
            
            return {
                "extensions": installed_extensions,
                "missing_extensions": list(missing),
                "table_count": table_count,
                "missing_tables": list(missing_tables)
            }
    
    def backup_schema(self, output_file: str = None):
        """Backup database schema"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"backup/schema_{timestamp}.sql"
        
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        
        db_url_parts = settings.DATABASE_URL.replace('postgresql://', '').split('@')
        user_pass = db_url_parts[0].split(':')
        host_db = db_url_parts[1].split('/')
        
        cmd = (
            f"pg_dump -h {host_db[0].split(':')[0]} "
            f"-U {user_pass[0]} "
            f"-d {host_db[1]} "
            f"--schema-only "
            f"--no-owner "
            f"--no-privileges "
            f"-f {output_file}"
        )
        
        os.system(cmd)
        logger.info(f"Schema backed up to: {output_file}")
    
    def analyze_tables(self):
        """Analyze all tables for query optimization"""
        with self.engine.connect() as conn:
            tables = conn.execute(text("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
            """)).fetchall()
            
            for table in tables:
                conn.execute(text(f"ANALYZE {table[0]}"))
                logger.info(f"Analyzed table: {table[0]}")
            
            logger.info("Table analysis complete")
    
    def check_indexes(self):
        """Check and report on database indexes"""
        with self.engine.connect() as conn:
            # Get index usage stats
            index_stats = conn.execute(text("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                ORDER BY idx_scan
                LIMIT 20
            """)).fetchall()
            
            logger.info("\nLeast used indexes:")
            for stat in index_stats:
                logger.info(f"  {stat[2]} on {stat[1]}: {stat[3]} scans, size: {stat[6]}")
            
            # Get missing indexes suggestion (simplified)
            missing = conn.execute(text("""
                SELECT
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats
                WHERE schemaname = 'public'
                AND n_distinct > 100
                AND correlation < 0.1
                ORDER BY n_distinct DESC
                LIMIT 10
            """)).fetchall()
            
            if missing:
                logger.info("\nColumns that might benefit from indexes:")
                for col in missing:
                    logger.info(f"  {col[1]}.{col[2]} (distinct values: {col[3]})")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SSA Database Management Tool")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create database
    subparsers.add_parser('create', help='Create database')
    
    # Drop database
    drop_parser = subparsers.add_parser('drop', help='Drop database')
    drop_parser.add_argument('--force', action='store_true', help='Force drop without confirmation')
    
    # Create tables
    subparsers.add_parser('create-tables', help='Create all tables')
    
    # Drop tables
    drop_tables_parser = subparsers.add_parser('drop-tables', help='Drop all tables')
    drop_tables_parser.add_argument('--force', action='store_true', help='Force drop without confirmation')
    
    # Reset database
    reset_parser = subparsers.add_parser('reset', help='Reset database (drop and recreate)')
    reset_parser.add_argument('--force', action='store_true', help='Force reset without confirmation')
    
    # Initialize data
    init_parser = subparsers.add_parser('init', help='Initialize with default data')
    init_parser.add_argument('--sample', action='store_true', help='Include sample data')
    
    # Migrations
    subparsers.add_parser('migrate', help='Run migrations')
    
    migration_parser = subparsers.add_parser('create-migration', help='Create new migration')
    migration_parser.add_argument('message', help='Migration message')
    
    rollback_parser = subparsers.add_parser('rollback', help='Rollback migration')
    rollback_parser.add_argument('--revision', default='-1', help='Target revision')
    
    subparsers.add_parser('migration-history', help='Show migration history')
    subparsers.add_parser('current-revision', help='Show current revision')
    
    # Schema operations
    subparsers.add_parser('verify', help='Verify schema integrity')
    
    backup_parser = subparsers.add_parser('backup-schema', help='Backup schema')
    backup_parser.add_argument('--output', help='Output file path')
    
    # Maintenance
    subparsers.add_parser('analyze', help='Analyze tables')
    subparsers.add_parser('check-indexes', help='Check index usage')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DatabaseManager()
    
    try:
        if args.command == 'create':
            manager.create_database()
        
        elif args.command == 'drop':
            manager.drop_database(force=args.force)
        
        elif args.command == 'create-tables':
            manager.create_tables()
        
        elif args.command == 'drop-tables':
            manager.drop_tables(force=args.force)
        
        elif args.command == 'reset':
            manager.reset_database(force=args.force)
        
        elif args.command == 'init':
            manager.init_data(sample_data=args.sample)
        
        elif args.command == 'migrate':
            manager.run_migrations()
        
        elif args.command == 'create-migration':
            manager.create_migration(args.message)
        
        elif args.command == 'rollback':
            manager.rollback_migration(args.revision)
        
        elif args.command == 'migration-history':
            manager.get_migration_history()
        
        elif args.command == 'current-revision':
            manager.get_current_revision()
        
        elif args.command == 'verify':
            result = manager.verify_schema()
            print(f"\nSchema verification result: {result}")
        
        elif args.command == 'backup-schema':
            manager.backup_schema(args.output)
        
        elif args.command == 'analyze':
            manager.analyze_tables()
        
        elif args.command == 'check-indexes':
            manager.check_indexes()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()