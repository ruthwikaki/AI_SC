"""
Query-related schemas for natural language processing and SQL generation
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


# =====================================================
# Enums
# =====================================================

class QueryStatus(str, Enum):
    """Query execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryIntent(str, Enum):
    """Query intent classification"""
    SELECT = "select"
    AGGREGATE = "aggregate"
    TREND = "trend"
    COMPARISON = "comparison"
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    UNKNOWN = "unknown"


# =====================================================
# Natural Language Query Schemas
# =====================================================

class NaturalLanguageQueryRequest(BaseModel):
    """Request schema for natural language query"""
    query_text: str = Field(..., min_length=3, max_length=1000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    connection_id: Optional[UUID] = None
    include_explanation: bool = True
    include_visualization_suggestions: bool = True
    cache_result: bool = True
    cache_ttl_seconds: int = 3600
    
    @field_validator('query_text')
    def validate_query_text(cls, v):
        """Validate query text"""
        v = v.strip()
        if not v:
            raise ValueError('Query text cannot be empty')
        return v


class QueryExecutionResult(BaseModel):
    """Query execution result"""
    columns: List[Dict[str, str]]  # [{"name": "col1", "type": "string"}]
    rows: List[List[Any]]
    row_count: int
    execution_time_ms: int
    truncated: bool = False
    
    model_config = ConfigDict(from_attributes=True)
        use_enum_values = True


# =====================================================
# Saved Query Schemas
# =====================================================

class SavedQueryBase(BaseModel):
    """Base schema for saved queries"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    query_text: str
    tags: List[str] = Field(default_factory=list)
    is_public: bool = False
    is_favorite: bool = False


class SavedQueryCreate(SavedQueryBase):
    """Schema for creating a saved query"""
    generated_sql: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SavedQueryUpdate(BaseModel):
    """Schema for updating a saved query"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_favorite: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None


class SavedQueryResponse(SavedQueryBase):
    """Response schema for saved query"""
    id: UUID
    user_id: UUID
    generated_sql: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_count: int
    last_executed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# =====================================================
# Query Template Schemas
# =====================================================

class QueryTemplateResponse(BaseModel):
    """Query template response schema"""
    id: UUID
    name: str
    category: str
    description: Optional[str] = None
    template_text: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    example_usage: Optional[str] = None
    tags: List[str] = []
    usage_count: int
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class QueryTemplateExecute(BaseModel):
    """Execute query template request"""
    template_id: UUID
    parameters: Dict[str, Any] = Field(default_factory=dict)
    connection_id: Optional[UUID] = None


# =====================================================
# Query Suggestion Schemas
# =====================================================

class QuerySuggestion(BaseModel):
    """Query suggestion schema"""
    id: UUID
    suggestion_text: str
    category: Optional[str] = None
    context_keywords: List[str] = []
    usage_count: int
    
    model_config = ConfigDict(from_attributes=True)


class QuerySuggestionRequest(BaseModel):
    """Request for query suggestions"""
    partial_query: Optional[str] = None
    category: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    limit: int = Field(default=10, ge=1, le=50)


# =====================================================
# Query History Schemas
# =====================================================

class QueryHistoryItem(BaseModel):
    """Query history item"""
    id: UUID
    query_text: str
    status: QueryStatus
    result_count: Optional[int] = None
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
        use_enum_values = True


class QueryHistoryFilter(BaseModel):
    """Query history filter"""
    status: Optional[QueryStatus] = None
    search: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    skip: int = 0
    limit: int = 50


# =====================================================
# Query Cache Schemas
# =====================================================

class QueryCacheConfig(BaseModel):
    """Query cache configuration"""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    eviction_policy: str = "lru"  # lru, lfu, ttl


class CacheStatistics(BaseModel):
    """Cache statistics"""
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float
    eviction_count: int
    oldest_entry_age_seconds: Optional[int] = None


# =====================================================
# Query Analysis Schemas
# =====================================================

class QueryAnalysis(BaseModel):
    """Query analysis result"""
    query_text: str
    intent: QueryIntent
    entities: List[Dict[str, str]]  # [{"type": "product", "value": "Product A"}]
    time_range: Optional[Dict[str, Any]] = None
    aggregations: List[str] = []
    filters: List[Dict[str, Any]] = []
    confidence_scores: Dict[str, float] = {}


class SQLExplanation(BaseModel):
    """SQL query explanation"""
    original_query: str
    generated_sql: str
    explanation_steps: List[Dict[str, str]]
    tables_used: List[str]
    columns_selected: List[str]
    filters_applied: List[str]
    aggregations_used: List[str]
    joins_performed: List[str]


# =====================================================
# Batch Query Schemas
# =====================================================

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[NaturalLanguageQueryRequest]
    parallel_execution: bool = False
    stop_on_error: bool = False


class BatchQueryResponse(BaseModel):
    """Batch query response"""
    results: List[Union[NaturalLanguageQueryResponse, Dict[str, str]]]
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_execution_time_ms: int


# =====================================================
# Query Validation Schemas
# =====================================================

class QueryValidationRequest(BaseModel):
    """Query validation request"""
    query_text: str
    connection_id: Optional[UUID] = None
    check_syntax: bool = True
    check_permissions: bool = True
    estimate_cost: bool = False


class QueryValidationResponse(BaseModel):
    """Query validation response"""
    is_valid: bool
    syntax_errors: List[str] = []
    permission_errors: List[str] = []
    warnings: List[str] = []
    estimated_rows: Optional[int] = None
    estimated_execution_time_ms: Optional[int] = None
    estimated_cost: Optional[float] = None