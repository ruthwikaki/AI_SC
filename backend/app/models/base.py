"""
Base model for all database models
"""
from sqlalchemy.ext.declarative import declarative_base

# Create a single Base instance to be used by all models
Base = declarative_base()

# Metadata for the database
metadata = Base.metadata