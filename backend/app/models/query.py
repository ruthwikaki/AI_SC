"""
Query models for natural language processing and SQL generation
Located at: /backend/app/models/query.py
"""
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.models.base import BaseModel

class NaturalLanguageQuery(BaseModel):
    """Model for storing natural language queries and their SQL translations"""
    __tablename__ = "natural_language_query"
    __table_args__ = (
        Index('idx_nlq_user_created', 'user_id', 'created_at'),
        Index('idx_nlq_success', 'success'),
    )
    
    # Query fields
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    natural_language_query = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=True)
    
    # Execution details
    execution_time = Column(Float, nullable=True)  # in seconds
    result_count = Column(Integer, nullable=True)
    success = Column(Boolean, default=False, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Query metadata
    database_name = Column(String, nullable=True)
    tables_accessed = Column(JSON, nullable=True)  # List of tables used in query
    columns_accessed = Column(JSON, nullable=True)  # List of columns used
    
    # LLM metadata
    llm_model = Column(String, nullable=True)
    llm_confidence = Column(Float, nullable=True)
    llm_response_time = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    
    # Additional metadata
    query_metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)  # User or system tags
    
    # Caching
    cache_key = Column(String, nullable=True, index=True)
    cached = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="queries")
    feedbacks = relationship("QueryFeedback", back_populates="query", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<NaturalLanguageQuery(id={self.id}, query='{self.natural_language_query[:50]}...')>"

class QueryFeedback(BaseModel):
    """Model for storing user feedback on query results"""
    __tablename__ = "query_feedback"
    
    query_id = Column(String, ForeignKey('natural_language_query.id'), nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    
    # Feedback
    rating = Column(Integer, nullable=True)  # 1-5 rating
    helpful = Column(Boolean, nullable=True)
    accurate = Column(Boolean, nullable=True)
    
    # Detailed feedback
    feedback_text = Column(Text, nullable=True)
    suggested_improvement = Column(Text, nullable=True)
    
    # Categorized issues
    issues = Column(JSON, nullable=True)  # List of issue types
    
    # Relationships
    query = relationship("NaturalLanguageQuery", back_populates="feedbacks")
    user = relationship("User")

class SavedQuery(BaseModel):
    """Model for saved/bookmarked queries"""
    __tablename__ = "saved_query"
    __table_args__ = (
        Index('idx_saved_query_user', 'user_id'),
        Index('idx_saved_query_shared', 'is_shared'),
    )
    
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Query content
    natural_language_query = Column(Text, nullable=False)
    sql_query = Column(Text, nullable=True)
    
    # Sharing settings
    is_shared = Column(Boolean, default=False, nullable=False)
    shared_with_roles = Column(JSON, nullable=True)  # List of role IDs
    shared_with_users = Column(JSON, nullable=True)  # List of user IDs
    
    # Organization
    category = Column(String, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Usage tracking
    use_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(String, nullable=True)
    
    # Parameters for parameterized queries
    parameters = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User")
    schedules = relationship("QuerySchedule", back_populates="saved_query", cascade="all, delete-orphan")

class QuerySchedule(BaseModel):
    """Model for scheduled query execution"""
    __tablename__ = "query_schedule"
    
    saved_query_id = Column(String, ForeignKey('saved_query.id'), nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    
    # Schedule configuration
    cron_expression = Column(String, nullable=True)  # For cron-based scheduling
    frequency = Column(String, nullable=True)  # daily, weekly, monthly
    next_run_at = Column(String, nullable=True)
    last_run_at = Column(String, nullable=True)
    
    # Execution settings
    is_active = Column(Boolean, default=True, nullable=False)
    timeout_seconds = Column(Integer, default=300, nullable=False)
    
    # Notification settings
    notify_on_success = Column(Boolean, default=False, nullable=False)
    notify_on_failure = Column(Boolean, default=True, nullable=False)
    notification_emails = Column(JSON, nullable=True)
    
    # Output settings
    export_format = Column(String, nullable=True)  # csv, excel, pdf
    export_destination = Column(String, nullable=True)  # email, s3, etc.
    
    # Relationships
    saved_query = relationship("SavedQuery", back_populates="schedules")
    user = relationship("User")
    executions = relationship("ScheduledQueryExecution", back_populates="schedule", cascade="all, delete-orphan")

class ScheduledQueryExecution(BaseModel):
    """Model for tracking scheduled query executions"""
    __tablename__ = "scheduled_query_execution"
    
    schedule_id = Column(String, ForeignKey('query_schedule.id'), nullable=False)
    
    # Execution details
    started_at = Column(String, nullable=False)
    completed_at = Column(String, nullable=True)
    status = Column(String, nullable=False)  # pending, running, success, failed
    
    # Results
    result_count = Column(Integer, nullable=True)
    execution_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Output details
    output_location = Column(String, nullable=True)
    output_size = Column(Integer, nullable=True)
    
    # Relationships
    schedule = relationship("QuerySchedule", back_populates="executions")

class QueryTemplate(BaseModel):
    """Model for query templates"""
    __tablename__ = "query_template"
    
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    
    # Template content
    template_text = Column(Text, nullable=False)
    parameters = Column(JSON, nullable=True)  # Parameter definitions
    
    # Example usage
    example_query = Column(Text, nullable=True)
    example_parameters = Column(JSON, nullable=True)
    
    # Visibility
    is_system = Column(Boolean, default=False, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    
    # Usage tracking
    use_count = Column(Integer, default=0, nullable=False)
    rating = Column(Float, nullable=True)
    
    # Tags for discovery
    tags = Column(JSON, nullable=True)

class QueryResultCache(BaseModel):
    """Cache for query results"""
    __tablename__ = "query_result_cache"
    
    query_hash = Column(String, unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    result_data = Column(JSON, nullable=False)
    created_at = Column(String, nullable=False)
    expires_at = Column(String, nullable=False)
    access_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<QueryResultCache(query_hash={self.query_hash})>"


class QuerySuggestion(BaseModel):
    """Query suggestions for users"""
    __tablename__ = "query_suggestion"
    
    suggestion_text = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<QuerySuggestion(text='{self.suggestion_text[:50]}...')>"
