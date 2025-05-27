"""
Query repository for natural language query operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import hashlib
import json
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc
from sqlalchemy.exc import IntegrityError

from app.models import (
    NaturalLanguageQuery, SavedQuery, QueryResultCache,
    QueryTemplate, QuerySuggestion
)

logger = logging.getLogger(__name__)


class QueryRepository:
    """Repository for query-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Natural Language Query Operations
    # =====================================================
    
    def create_query(
        self,
        user_id: UUID,
        query_text: str,
        intent_classification: Optional[str] = None,
        model_used: Optional[str] = None
    ) -> NaturalLanguageQuery:
        """Create a new natural language query record"""
        query = NaturalLanguageQuery(
            user_id=user_id,
            query_text=query_text,
            intent_classification=intent_classification,
            model_used=model_used,
            status='pending'
        )
        self.db.add(query)
        self.db.commit()
        self.db.refresh(query)
        return query
    
    def update_query_result(
        self,
        query_id: UUID,
        generated_sql: Optional[str] = None,
        sql_parameters: Optional[List[Any]] = None,
        execution_time_ms: Optional[int] = None,
        result_count: Optional[int] = None,
        error_message: Optional[str] = None,
        status: Optional[str] = None,
        tokens_used: Optional[int] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[NaturalLanguageQuery]:
        """Update query with execution results"""
        query = self.db.query(NaturalLanguageQuery).filter(
            NaturalLanguageQuery.id == query_id
        ).first()
        
        if not query:
            return None
        
        if generated_sql is not None:
            query.generated_sql = generated_sql
        if sql_parameters is not None:
            query.sql_parameters = sql_parameters
        if execution_time_ms is not None:
            query.execution_time_ms = execution_time_ms
        if result_count is not None:
            query.result_count = result_count
        if error_message is not None:
            query.error_message = error_message
        if status is not None:
            query.status = status
        if tokens_used is not None:
            query.tokens_used = tokens_used
        if confidence_score is not None:
            query.confidence_score = confidence_score
        if metadata is not None:
            query.metadata = metadata
        
        self.db.commit()
        self.db.refresh(query)
        return query
    
    def get_query_by_id(self, query_id: UUID) -> Optional[NaturalLanguageQuery]:
        """Get query by ID"""
        return self.db.query(NaturalLanguageQuery).filter(
            NaturalLanguageQuery.id == query_id
        ).first()
    
    def get_user_queries(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[NaturalLanguageQuery], int]:
        """Get user's queries with filters"""
        query = self.db.query(NaturalLanguageQuery).filter(
            NaturalLanguageQuery.user_id == user_id
        )
        
        # Apply filters
        if status:
            query = query.filter(NaturalLanguageQuery.status == status)
        
        if search:
            query = query.filter(
                NaturalLanguageQuery.query_text.ilike(f"%{search}%")
            )
        
        if start_date:
            query = query.filter(NaturalLanguageQuery.created_at >= start_date)
        
        if end_date:
            query = query.filter(NaturalLanguageQuery.created_at <= end_date)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        queries = query.order_by(
            desc(NaturalLanguageQuery.created_at)
        ).offset(skip).limit(limit).all()
        
        return queries, total
    
    def get_similar_queries(
        self,
        query_text: str,
        user_id: Optional[UUID] = None,
        limit: int = 5
    ) -> List[NaturalLanguageQuery]:
        """Find similar queries using trigram similarity"""
        # Use PostgreSQL's trigram similarity
        similarity_threshold = 0.3
        
        query = self.db.query(
            NaturalLanguageQuery,
            func.similarity(NaturalLanguageQuery.query_text, query_text).label('similarity')
        ).filter(
            func.similarity(NaturalLanguageQuery.query_text, query_text) > similarity_threshold,
            NaturalLanguageQuery.status == 'completed',
            NaturalLanguageQuery.error_message.is_(None)
        )
        
        if user_id:
            query = query.filter(NaturalLanguageQuery.user_id == user_id)
        
        return [q[0] for q in query.order_by(desc('similarity')).limit(limit).all()]
    
    # =====================================================
    # Saved Query Operations
    # =====================================================
    
    def save_query(
        self,
        user_id: UUID,
        name: str,
        query_text: str,
        description: Optional[str] = None,
        generated_sql: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False
    ) -> SavedQuery:
        """Save a query for reuse"""
        saved_query = SavedQuery(
            user_id=user_id,
            name=name,
            description=description,
            query_text=query_text,
            generated_sql=generated_sql,
            parameters=parameters or {},
            tags=tags or [],
            is_public=is_public
        )
        self.db.add(saved_query)
        self.db.commit()
        self.db.refresh(saved_query)
        return saved_query
    
    def get_saved_query(self, query_id: UUID, user_id: Optional[UUID] = None) -> Optional[SavedQuery]:
        """Get saved query by ID"""
        query = self.db.query(SavedQuery).filter(SavedQuery.id == query_id)
        
        # Check access permissions
        if user_id:
            query = query.filter(
                or_(
                    SavedQuery.user_id == user_id,
                    SavedQuery.is_public == True
                )
            )
        
        return query.first()
    
    def get_user_saved_queries(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_favorite: Optional[bool] = None
    ) -> Tuple[List[SavedQuery], int]:
        """Get user's saved queries"""
        query = self.db.query(SavedQuery).filter(
            or_(
                SavedQuery.user_id == user_id,
                SavedQuery.is_public == True
            )
        )
        
        # Apply filters
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    SavedQuery.name.ilike(search_filter),
                    SavedQuery.description.ilike(search_filter),
                    SavedQuery.query_text.ilike(search_filter)
                )
            )
        
        if tags:
            query = query.filter(
                SavedQuery.tags.overlap(tags)
            )
        
        if is_favorite is not None:
            query = query.filter(SavedQuery.is_favorite == is_favorite)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        queries = query.order_by(
            desc(SavedQuery.is_favorite),
            desc(SavedQuery.created_at)
        ).offset(skip).limit(limit).all()
        
        return queries, total
    
    def update_saved_query(
        self,
        query_id: UUID,
        user_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[SavedQuery]:
        """Update saved query"""
        saved_query = self.get_saved_query(query_id, user_id)
        if not saved_query or saved_query.user_id != user_id:
            return None
        
        for key, value in update_data.items():
            if hasattr(saved_query, key):
                setattr(saved_query, key, value)
        
        saved_query.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(saved_query)
        return saved_query
    
    def delete_saved_query(self, query_id: UUID, user_id: UUID) -> bool:
        """Delete saved query"""
        saved_query = self.get_saved_query(query_id, user_id)
        if not saved_query or saved_query.user_id != user_id:
            return False
        
        self.db.delete(saved_query)
        self.db.commit()
        return True
    
    def execute_saved_query(self, query_id: UUID, user_id: UUID) -> Optional[SavedQuery]:
        """Record execution of saved query"""
        saved_query = self.get_saved_query(query_id, user_id)
        if not saved_query:
            return None
        
        saved_query.increment_execution_count()
        self.db.commit()
        return saved_query
    
    # =====================================================
    # Query Cache Operations
    # =====================================================
    
    def _generate_cache_key(self, query_text: str, user_id: Optional[UUID] = None) -> str:
        """Generate cache key for query"""
        key_parts = [query_text]
        if user_id:
            key_parts.append(str(user_id))
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_cached_result(
        self,
        query_text: str,
        user_id: Optional[UUID] = None
    ) -> Optional[QueryResultCache]:
        """Get cached query result"""
        cache_key = self._generate_cache_key(query_text, user_id)
        
        cache_entry = self.db.query(QueryResultCache).filter(
            QueryResultCache.query_hash == cache_key,
            QueryResultCache.expires_at > datetime.utcnow()
        ).first()
        
        return cache_entry
    
    def cache_query_result(
        self,
        query_text: str,
        sql_query: str,
        result_data: Dict[str, Any],
        result_count: int,
        execution_time_ms: int,
        user_id: Optional[UUID] = None,
        ttl_seconds: int = 3600
    ) -> QueryResultCache:
        """Cache query result"""
        cache_key = self._generate_cache_key(query_text, user_id)
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        # Remove existing cache entry
        self.db.query(QueryResultCache).filter(
            QueryResultCache.query_hash == cache_key
        ).delete()
        
        # Create new cache entry
        cache_entry = QueryResultCache(
            query_hash=cache_key,
            user_id=user_id,
            query_text=query_text,
            sql_query=sql_query,
            result_data=result_data,
            result_count=result_count,
            execution_time_ms=execution_time_ms,
            expires_at=expires_at
        )
        self.db.add(cache_entry)
        self.db.commit()
        self.db.refresh(cache_entry)
        return cache_entry
    
    def invalidate_cache(self, patterns: Optional[List[str]] = None):
        """Invalidate cache entries"""
        query = self.db.query(QueryResultCache)
        
        if patterns:
            # Invalidate specific patterns
            conditions = []
            for pattern in patterns:
                conditions.append(QueryResultCache.query_text.ilike(f"%{pattern}%"))
            query = query.filter(or_(*conditions))
        
        count = query.delete()
        self.db.commit()
        return count
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries"""
        count = self.db.query(QueryResultCache).filter(
            QueryResultCache.expires_at < datetime.utcnow()
        ).delete()
        self.db.commit()
        return count
    
    # =====================================================
    # Query Template Operations
    # =====================================================
    
    def get_query_templates(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_active: bool = True
    ) -> List[QueryTemplate]:
        """Get query templates"""
        query = self.db.query(QueryTemplate)
        
        if is_active is not None:
            query = query.filter(QueryTemplate.is_active == is_active)
        
        if category:
            query = query.filter(QueryTemplate.category == category)
        
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    QueryTemplate.name.ilike(search_filter),
                    QueryTemplate.description.ilike(search_filter),
                    QueryTemplate.template_text.ilike(search_filter)
                )
            )
        
        if tags:
            query = query.filter(QueryTemplate.tags.overlap(tags))
        
        return query.order_by(
            QueryTemplate.category,
            QueryTemplate.name
        ).all()
    
    def get_template_by_id(self, template_id: UUID) -> Optional[QueryTemplate]:
        """Get template by ID"""
        return self.db.query(QueryTemplate).filter(
            QueryTemplate.id == template_id
        ).first()
    
    def use_template(self, template_id: UUID) -> Optional[QueryTemplate]:
        """Record template usage"""
        template = self.get_template_by_id(template_id)
        if template:
            template.increment_usage_count()
            self.db.commit()
        return template
    
    # =====================================================
    # Query Suggestion Operations
    # =====================================================
    
    def get_query_suggestions(
        self,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[QuerySuggestion]:
        """Get query suggestions"""
        query = self.db.query(QuerySuggestion).filter(
            QuerySuggestion.is_active == True
        )
        
        if category:
            query = query.filter(QuerySuggestion.category == category)
        
        if keywords:
            # Filter by matching keywords
            conditions = []
            for keyword in keywords:
                conditions.append(
                    QuerySuggestion.context_keywords.any(keyword.lower())
                )
            query = query.filter(or_(*conditions))
        
        return query.order_by(
            QuerySuggestion.display_order,
            desc(QuerySuggestion.usage_count)
        ).limit(limit).all()
    
    def record_suggestion_usage(self, suggestion_id: UUID) -> bool:
        """Record that a suggestion was used"""
        suggestion = self.db.query(QuerySuggestion).filter(
            QuerySuggestion.id == suggestion_id
        ).first()
        
        if suggestion:
            suggestion.increment_usage_count()
            self.db.commit()
            return True
        return False
    
    def get_popular_queries(
        self,
        user_id: Optional[UUID] = None,
        days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get popular queries in the last N days"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = self.db.query(
            NaturalLanguageQuery.query_text,
            func.count(NaturalLanguageQuery.id).label('count')
        ).filter(
            NaturalLanguageQuery.created_at >= since_date,
            NaturalLanguageQuery.status == 'completed'
        )
        
        if user_id:
            query = query.filter(NaturalLanguageQuery.user_id == user_id)
        
        results = query.group_by(
            NaturalLanguageQuery.query_text
        ).order_by(
            desc('count')
        ).limit(limit).all()
        
        return [
            {"query_text": r[0], "count": r[1]}
            for r in results
        ]
    
    # =====================================================
    # Analytics and Statistics
    # =====================================================
    
    def get_query_statistics(
        self,
        user_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get query statistics"""
        query = self.db.query(NaturalLanguageQuery)
        
        if user_id:
            query = query.filter(NaturalLanguageQuery.user_id == user_id)
        
        if start_date:
            query = query.filter(NaturalLanguageQuery.created_at >= start_date)
        
        if end_date:
            query = query.filter(NaturalLanguageQuery.created_at <= end_date)
        
        total_queries = query.count()
        successful_queries = query.filter(
            NaturalLanguageQuery.status == 'completed'
        ).count()
        
        avg_execution_time = self.db.query(
            func.avg(NaturalLanguageQuery.execution_time_ms)
        ).filter(
            NaturalLanguageQuery.execution_time_ms.isnot(None)
        ).scalar() or 0
        
        avg_tokens_used = self.db.query(
            func.avg(NaturalLanguageQuery.tokens_used)
        ).filter(
            NaturalLanguageQuery.tokens_used.isnot(None)
        ).scalar() or 0
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "avg_execution_time_ms": float(avg_execution_time),
            "avg_tokens_used": float(avg_tokens_used)
        }