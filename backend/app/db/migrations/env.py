"""
Alembic environment configuration
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import your models' Base and all models
from app.db.database import Base
from app.models import *  # Import all models to ensure they're registered
from config import settings

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
target_metadata = Base.metadata

# Get database URL from environment
def get_database_url():
    """Get database URL from settings"""
    return (
        f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

# Set the database URL
config.set_main_option("sqlalchemy.url", get_database_url())

def include_object(object, name, type_, reflected, compare_to):
    """
    Control which database objects to include in migrations.
    Exclude certain system tables or views if needed.
    """
    # Skip PostGIS tables if they exist
    if type_ == "table" and name in ["spatial_ref_sys", "geometry_columns"]:
        return False
    
    # Skip any pg_* system tables
    if type_ == "table" and name.startswith("pg_"):
        return False
    
    return True

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            # Additional options for better migration generation
            render_as_batch=True,  # Support for batch operations
            include_schemas=True,   # Include schema information
        )

        with context.begin_transaction():
            context.run_migrations()

# Run migrations
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()