"""
Query models for natural language processing and SQL generation
"""

from datetime import datetime
from uuid import uuid4
from typing import Optional

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, 
    DateTime, ForeignKey, Index, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base


class NaturalLanguageQuery(Base):
    """Model for storing natural language queries and their SQL translations"""
    __tablename__ = "natural_language_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    natural_language_query = Column(Text, nullable=False)
    generated_sql = Column(Text)
    query_type = Column(String(50))  # 'select', 'aggregate', 'join', etc.
    tables_involved = Column(JSONB, default=[])
    execution_time_ms = Column(Float)
    result_count = Column(Integer)
    success = Column(Boolean, default=False)
    error_message = Column(Text)
    feedback_rating = Column(Integer)  # 1-5 rating
    feedback_comment = Column(Text)
    parameters = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='queries')
    executions = relationship('QueryExecution', back_populates='query', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_nlq_user_created', 'user_id', 'created_at'),
        Index('idx_nlq_success', 'success'),
    )
    
    def __repr__(self):
        return f"<NaturalLanguageQuery(id={self.id}, query='{self.natural_language_query[:50]}...')>"


class SQLQuery(Base):
    """Model for storing validated SQL queries"""
    __tablename__ = "sql_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_hash = Column(String(64), unique=True, nullable=False)  # SHA256 of the SQL
    sql_text = Column(Text, nullable=False)
    query_type = Column(String(50))
    tables_used = Column(JSONB, default=[])
    columns_used = Column(JSONB, default=[])
    is_safe = Column(Boolean, default=True)
    complexity_score = Column(Integer)
    estimated_cost = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SQLQuery(id={self.id}, type={self.query_type})>"


class QueryExecution(Base):
    """Model for tracking query execution history"""
    __tablename__ = "query_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey('natural_language_queries.id'), nullable=False)
    sql_query_id = Column(UUID(as_uuid=True), ForeignKey('sql_queries.id'))
    execution_time_ms = Column(Float)
    rows_returned = Column(Integer)
    error_message = Column(Text)
    success = Column(Boolean)
    executed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    query = relationship('NaturalLanguageQuery', back_populates='executions')
    
    def __repr__(self):
        return f"<QueryExecution(id={self.id}, success={self.success})>"


class SavedQuery(Base):
    """Model for user-saved queries"""
    __tablename__ = "saved_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    natural_language_query = Column(Text, nullable=False)
    sql_query = Column(Text)
    parameters = Column(JSONB, default={})
    is_public = Column(Boolean, default=False)
    tags = Column(JSONB, default=[])
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='saved_queries')
    
    def __repr__(self):
        return f"<SavedQuery(id={self.id}, name={self.name})>"


class QueryTemplate(Base):
    """Model for query templates"""
    __tablename__ = "query_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    template_text = Column(Text, nullable=False)
    sql_template = Column(Text)
    parameters_schema = Column(JSONB, default={})
    category = Column(String(50))
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<QueryTemplate(id={self.id}, name={self.name})>"


class QuerySuggestion(Base):
    """Model for query suggestions"""
    __tablename__ = "query_suggestions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    suggestion_text = Column(Text, nullable=False)
    category = Column(String(50))
    relevance_score = Column(Float, default=1.0)
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QuerySuggestion(id={self.id}, text='{self.suggestion_text[:50]}...')>"


class QueryResultCache(Base):
    """Model for caching query results"""
    __tablename__ = "query_result_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_hash = Column(String(64), unique=True, nullable=False)
    result_data = Column(JSONB, nullable=False)
    row_count = Column(Integer)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QueryResultCache(id={self.id}, expires_at={self.expires_at})>"


class Query(Base):
    """Model for storing user queries (generic wrapper)"""
    __tablename__ = "queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # 'natural_language', 'sql', 'structured'
    context = Column(JSON)  # Additional context for the query
    
    # Status
    status = Column(String(50), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="query_records")
    results = relationship("QueryResult", back_populates="query", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Query(id={self.id}, user_id={self.user_id}, status={self.status})>"


class QueryResult(Base):
    """Model for storing query results"""
    __tablename__ = "query_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False)
    
    # Result details
    result_type = Column(String(50))  # 'data', 'visualization', 'report', 'error'
    result_data = Column(JSON)  # The actual result data
    result_summary = Column(Text)  # Human-readable summary
    
    # Metadata
    row_count = Column(Integer)
    execution_time = Column(Integer)  # in milliseconds
    data_source = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    query = relationship("Query", back_populates="results")
    
    def __repr__(self):
        return f"<QueryResult(id={self.id}, query_id={self.query_id}, type={self.result_type})>"


class QueryHistory(Base):
    """Track query execution history for audit and optimization"""
    __tablename__ = "query_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))
    
    # Execution details
    execution_time = Column(Integer)  # milliseconds
    rows_returned = Column(Integer)
    status = Column(String(50))  # 'success', 'error', 'timeout'
    error_message = Column(Text)
    
    # Performance metrics
    cpu_time = Column(Float)
    memory_used = Column(Integer)  # bytes
    
    # Context
    source = Column(String(100))  # 'web_app', 'api', 'scheduled', etc.
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    
    # Timestamp
    executed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    query = relationship("Query")
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, user_id={self.user_id}, status={self.status})>"
