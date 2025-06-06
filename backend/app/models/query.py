"""
Query-related database models
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from decimal import Decimal

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Text, Float, DECIMAL, ARRAY, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base

class NaturalLanguageQuery(Base):
    """Natural language query history and results"""
    __tablename__ = 'natural_language_queries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    intent_classification = Column(String(100))
    generated_sql = Column(Text)
    sql_parameters = Column(JSONB, default=list)
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    error_message = Column(Text)
    status = Column(String(50), nullable=False, default='pending')
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    confidence_score = Column(DECIMAL(3, 2))
    query_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship('User', back_populates='queries')
    charts = relationship('Chart', back_populates='query')
    
    # Indexes
    __table_args__ = (
        Index('idx_nl_queries_search', 'query_text', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<NaturalLanguageQuery(id={self.id}, query_text={self.query_text[:50]}...)>"
    
    @property
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.status == 'completed' and self.error_message is None
    
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Get execution time in seconds"""
        if self.execution_time_ms:
            return self.execution_time_ms / 1000.0
        return None


class SavedQuery(Base):
    """User-saved queries for reuse"""
    __tablename__ = 'saved_queries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    query_text = Column(Text, nullable=False)
    generated_sql = Column(Text)
    parameters = Column(JSONB, default={})
    tags = Column(ARRAY(Text), default=[])
    is_public = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    execution_count = Column(Integer, default=0)
    last_executed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='saved_queries')
    
    def __repr__(self):
        return f"<SavedQuery(id={self.id}, name={self.name})>"
    
    def increment_execution_count(self):
        """Increment execution count and update last executed time"""
        self.execution_count += 1
        self.last_executed_at = datetime.utcnow()


class QueryResultCache(Base):
    """Cache for query results to improve performance"""
    __tablename__ = 'query_results_cache'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_hash = Column(String(64), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    query_text = Column(Text)
    sql_query = Column(Text)
    result_data = Column(JSONB, nullable=False)
    result_count = Column(Integer)
    execution_time_ms = Column(Integer)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User')
    
    def __repr__(self):
        return f"<QueryResultCache(id={self.id}, query_hash={self.query_hash})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()


class QueryTemplate(Base):
    """Pre-built query templates for common use cases"""
    __tablename__ = 'query_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(Text)
    template_text = Column(Text, nullable=False)
    parameters_schema = Column(JSONB, default={})
    example_usage = Column(Text)
    tags = Column(ARRAY(Text), default=[])
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<QueryTemplate(id={self.id}, name={self.name}, category={self.category})>"
    
    def increment_usage_count(self):
        """Increment usage count"""
        self.usage_count += 1
    
    def render(self, parameters: dict) -> str:
        """Render template with parameters"""
        # Simple template rendering - in production, use Jinja2 or similar
        result = self.template_text
        for key, value in parameters.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class QuerySuggestion(Base):
    """Query suggestions for users"""
    __tablename__ = 'query_suggestions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    suggestion_text = Column(Text, nullable=False)
    category = Column(String(100))
    context_keywords = Column(ARRAY(Text), default=[])
    display_order = Column(Integer, default=0)
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QuerySuggestion(id={self.id}, suggestion_text={self.suggestion_text[:50]}...)>"
    
    def increment_usage_count(self):
        """Increment usage count"""
        self.usage_count += 1
    
    def matches_keywords(self, keywords: List[str]) -> bool:
        """Check if suggestion matches given keywords"""
        if not self.context_keywords:
            return True
        
        # Case-insensitive matching
        context_lower = [k.lower() for k in self.context_keywords]
        keywords_lower = [k.lower() for k in keywords]
        
        # Check if any keyword matches
        return any(keyword in context_lower for keyword in keywords_lower)



