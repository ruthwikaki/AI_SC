"""Shared database base for all models"""
from sqlalchemy.ext.declarative import declarative_base

# Single shared Base for all models
Base = declarative_base()
