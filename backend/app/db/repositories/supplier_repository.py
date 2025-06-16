# Models should be imported inside methods to avoid circular imports
# Example:
# def get_something(self):
#     from app.models import User  # Import here, not at module level
#     return User.query.all()
"""
Supplier repository for supplier management operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc, asc
from sqlalchemy.exc import IntegrityError

# MOVED TO METHOD LEVEL: from app.models import (
    Supplier, SupplierTier, SupplierRelationship,
    SupplierPerformanceMetric, RiskAssessment, ComplianceCheck,
    Order
)

logger = logging.getLogger(__name__)


class SupplierRepository:
    """Repository for supplier-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Supplier CRUD Operations
    # =====================================================
    
    def create_supplier(self, supplier_data: Dict[str, Any]) -> Supplier:
        """Create a new supplier"""
        try:
            supplier = Supplier(**supplier_data)
            self.db.add(supplier)
            self.db.commit()
            self.db.refresh(supplier)
            
            # Create initial tier classification
            if 'tier' in supplier_data:
                tier = SupplierTier(
                    supplier_id=supplier.id,
                    tier_level=supplier_data['tier'],
                    classification='initial',
                    verified=False
                )
                self.db.add(tier)
                self.db.commit()
            
            return supplier
        except IntegrityError as e:
            self.db.rollback()
            if 'code' in str(e.orig):
                raise ValueError(f"Supplier code already exists: {supplier_data.get('code')}")
            raise
    
    def get_supplier_by_id(self, supplier_id: UUID) -> Optional[Supplier]:
        """Get supplier by ID"""
        return self.db.query(Supplier).options(
            joinedload(Supplier.tiers),
            joinedload(Supplier.performance_metrics)
        ).filter(Supplier.id == supplier_id).first()
    
    def get_supplier_by_code(self, code: str) -> Optional[Supplier]:
        """Get supplier by code"""
        return self.db.query(Supplier).filter(
            Supplier.code == code
        ).first()
    
    def get_suppliers(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        category: Optional[str] = None,
        tier: Optional[int] = None,
        status: Optional[str] = None,
        country: Optional[str] = None,
        sort_by: str = 'name',
        sort_order: str = 'asc'
    ) -> Tuple[List[Supplier], int]:
        """Get suppliers with filters and pagination"""
        query = self.db.query(Supplier)
        
        # Apply filters
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    Supplier.name.ilike(search_filter),
                    Supplier.code.ilike(search_filter),
                    Supplier.contact_name.ilike(search_filter),
                    Supplier.contact_email.ilike(search_filter)
                )
            )
        
        if category:
            query = query.filter(Supplier.category == category)
        
        if tier is not None:
            query = query.filter(Supplier.tier == tier)
        
        if status:
            query = query.filter(Supplier.status == status)
        
        if country:
            query = query.filter(Supplier.country == country)
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        sort_column = getattr(Supplier, sort_by, Supplier.name)
        if sort_order == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        suppliers = query.offset(skip).limit(limit).all()
        
        return suppliers, total
    
    def update_supplier(
        self,
        supplier_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[Supplier]:
        """Update supplier information"""
        supplier = self.get_supplier_by_id(supplier_id)
        if not supplier:
            return None
        
        try:
            for key, value in update_data.items():
                if hasattr(supplier, key):
                    setattr(supplier, key, value)
            
            supplier.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(supplier)
            return supplier
        except IntegrityError:
            self.db.rollback()
            raise
    
    def delete_supplier(self, supplier_id: UUID) -> bool:
        """Soft delete supplier"""
        supplier = self.get_supplier_by_id(supplier_id)
        if not supplier:
            return False
        
        supplier.status = 'inactive'
        supplier.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    # =====================================================
    # Supplier Tier Operations
    # =====================================================
    
    def update_supplier_tier(
        self,
        supplier_id: UUID,
        tier_level: int,
        classification: str,
        verified: bool = False,
        notes: Optional[str] = None
    ) -> Optional[SupplierTier]:
        """Update or create supplier tier classification"""
        # Check if tier exists
        tier = self.db.query(SupplierTier).filter(
            SupplierTier.supplier_id == supplier_id,
            SupplierTier.tier_level == tier_level
        ).first()
        
        if tier:
            # Update existing
            tier.classification = classification
            tier.verified = verified
            if verified:
                tier.verified_date = datetime.utcnow()
            tier.notes = notes
        else:
            # Create new
            tier = SupplierTier(
                supplier_id=supplier_id,
                tier_level=tier_level,
                classification=classification,
                verified=verified,
                verified_date=datetime.utcnow() if verified else None,
                notes=notes
            )
            self.db.add(tier)
        
        # Update supplier's main tier
        supplier = self.get_supplier_by_id(supplier_id)
        if supplier:
            supplier.tier = tier_level
        
        self.db.commit()
        return tier
    
    def get_suppliers_by_tier(self, tier_level: int) -> List[Supplier]:
        """Get all suppliers in a specific tier"""
        return self.db.query(Supplier).join(
            SupplierTier
        ).filter(
            SupplierTier.tier_level == tier_level,
            Supplier.status == 'active'
        ).distinct().all()
    
    # =====================================================
    # Supplier Relationship Operations
    # =====================================================
    
    def create_supplier_relationship(
        self,
        parent_supplier_id: UUID,
        child_supplier_id: UUID,
        relationship_type: str,
        strength: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SupplierRelationship:
        """Create relationship between suppliers"""
        if parent_supplier_id == child_supplier_id:
            raise ValueError("Parent and child supplier cannot be the same")
        
        try:
            relationship = SupplierRelationship(
                parent_supplier_id=parent_supplier_id,
                child_supplier_id=child_supplier_id,
                relationship_type=relationship_type,
                strength=strength,
                metadata=metadata or {},
                is_active=True
            )
            self.db.add(relationship)
            self.db.commit()
            self.db.refresh(relationship)
            return relationship
        except IntegrityError:
            self.db.rollback()
            raise ValueError("Relationship already exists")
    
    def get_supplier_relationships(
        self,
        supplier_id: UUID,
        direction: str = 'both'  # 'parent', 'child', 'both'
    ) -> List[SupplierRelationship]:
        """Get relationships for a supplier"""
        query = self.db.query(SupplierRelationship).filter(
            SupplierRelationship.is_active == True
        )
        
        if direction == 'parent':
            query = query.filter(SupplierRelationship.parent_supplier_id == supplier_id)
        elif direction == 'child':
            query = query.filter(SupplierRelationship.child_supplier_id == supplier_id)
        else:  # both
            query = query.filter(
                or_(
                    SupplierRelationship.parent_supplier_id == supplier_id,
                    SupplierRelationship.child_supplier_id == supplier_id
                )
            )
        
        return query.all()
    
    def get_supplier_network(
        self,
        supplier_id: UUID,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get supplier's network up to specified depth"""
        network = {
            "nodes": [],
            "edges": []
        }
        
        visited = set()
        
        def explore_supplier(sid: UUID, current_depth: int):
            if sid in visited or current_depth > depth:
                return
            
            visited.add(sid)
            supplier = self.get_supplier_by_id(sid)
            if not supplier:
                return
            
            # Add node
            network["nodes"].append({
                "id": str(supplier.id),
                "name": supplier.name,
                "tier": supplier.tier,
                "category": supplier.category
            })
            
            # Get relationships
            relationships = self.get_supplier_relationships(sid)
            for rel in relationships:
                # Add edge
                network["edges"].append({
                    "source": str(rel.parent_supplier_id),
                    "target": str(rel.child_supplier_id),
                    "type": rel.relationship_type,
                    "strength": float(rel.strength) if rel.strength else 1.0
                })
                
                # Explore connected suppliers
                if rel.parent_supplier_id == sid:
                    explore_supplier(rel.child_supplier_id, current_depth + 1)
                else:
                    explore_supplier(rel.parent_supplier_id, current_depth + 1)
        
        explore_supplier(supplier_id, 0)
        return network
    
    # =====================================================
    # Supplier Performance Operations
    # =====================================================
    
    def record_performance_metric(
        self,
        supplier_id: UUID,
        metric_date: date,
        metrics: Dict[str, Any]
    ) -> SupplierPerformanceMetric:
        """Record supplier performance metrics"""
        # Check if metric exists for this date
        existing = self.db.query(SupplierPerformanceMetric).filter(
            SupplierPerformanceMetric.supplier_id == supplier_id,
            SupplierPerformanceMetric.metric_date == metric_date
        ).first()
        
        if existing:
            # Update existing
            for key, value in metrics.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            metric = existing
        else:
            # Create new
            metric = SupplierPerformanceMetric(
                supplier_id=supplier_id,
                metric_date=metric_date,
                **metrics
            )
            self.db.add(metric)
        
        # Calculate overall score
        metric.overall_score = self._calculate_overall_score(metric)
        
        self.db.commit()
        self.db.refresh(metric)
        return metric
    
    def _calculate_overall_score(self, metric: SupplierPerformanceMetric) -> float:
        """Calculate overall performance score"""
        weights = {
            'on_time_delivery_rate': 0.3,
            'quality_score': 0.25,
            'price_competitiveness': 0.15,
            'order_accuracy_rate': 0.2,
            'response_time': 0.1
        }
        
        score = 0
        total_weight = 0
        
        if metric.on_time_delivery_rate is not None:
            score += metric.on_time_delivery_rate * weights['on_time_delivery_rate']
            total_weight += weights['on_time_delivery_rate']
        
        if metric.quality_score is not None:
            score += metric.quality_score * weights['quality_score']
            total_weight += weights['quality_score']
        
        if metric.price_competitiveness is not None:
            score += metric.price_competitiveness * weights['price_competitiveness']
            total_weight += weights['price_competitiveness']
        
        if metric.order_accuracy_rate is not None:
            score += metric.order_accuracy_rate * weights['order_accuracy_rate']
            total_weight += weights['order_accuracy_rate']
        
        # Response time (inverse - lower is better)
        if metric.response_time_hours is not None:
            response_score = max(0, 100 - (metric.response_time_hours / 24 * 100))
            score += response_score * weights['response_time']
            total_weight += weights['response_time']
        
        return (score / total_weight) if total_weight > 0 else 0
    
    def get_supplier_performance(
        self,
        supplier_id: UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[SupplierPerformanceMetric]:
        """Get supplier performance metrics"""
        query = self.db.query(SupplierPerformanceMetric).filter(
            SupplierPerformanceMetric.supplier_id == supplier_id
        )
        
        if start_date:
            query = query.filter(SupplierPerformanceMetric.metric_date >= start_date)
        
        if end_date:
            query = query.filter(SupplierPerformanceMetric.metric_date <= end_date)
        
        return query.order_by(desc(SupplierPerformanceMetric.metric_date)).all()
    
    def get_top_performers(
        self,
        category: Optional[str] = None,
        limit: int = 10,
        metric_period_days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get top performing suppliers"""
        since_date = date.today() - timedelta(days=metric_period_days)
        
        query = self.db.query(
            Supplier,
            func.avg(SupplierPerformanceMetric.overall_score).label('avg_score')
        ).join(
            SupplierPerformanceMetric
        ).filter(
            SupplierPerformanceMetric.metric_date >= since_date,
            Supplier.status == 'active'
        )
        
        if category:
            query = query.filter(Supplier.category == category)
        
        results = query.group_by(
            Supplier.id
        ).order_by(
            desc('avg_score')
        ).limit(limit).all()
        
        return [
            {
                "supplier": supplier,
                "average_score": float(avg_score or 0)
            }
            for supplier, avg_score in results
        ]
    
    # =====================================================
    # Risk Assessment Operations
    # =====================================================
    
    def create_risk_assessment(
        self,
        supplier_id: UUID,
        assessment_date: date,
        risk_category: str,
        risk_level: str,
        risk_score: float,
        risk_factors: List[str],
        mitigation_actions: List[str],
        assessment_by: UUID
    ) -> RiskAssessment:
        """Create supplier risk assessment"""
        assessment = RiskAssessment(
            supplier_id=supplier_id,
            assessment_date=assessment_date,
            risk_category=risk_category,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            mitigation_actions=mitigation_actions,
            assessment_by=assessment_by
        )
        
        # Calculate impact and probability scores (simplified)
        assessment.impact_score = risk_score * 0.6
        assessment.probability_score = risk_score * 0.4
        
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)
        return assessment
    
    def get_supplier_risks(
        self,
        supplier_id: UUID,
        risk_category: Optional[str] = None,
        include_historical: bool = False
    ) -> List[RiskAssessment]:
        """Get supplier risk assessments"""
        query = self.db.query(RiskAssessment).filter(
            RiskAssessment.supplier_id == supplier_id
        )
        
        if risk_category:
            query = query.filter(RiskAssessment.risk_category == risk_category)
        
        if not include_historical:
            # Get only the latest assessment per category
            subquery = self.db.query(
                RiskAssessment.risk_category,
                func.max(RiskAssessment.assessment_date).label('max_date')
            ).filter(
                RiskAssessment.supplier_id == supplier_id
            ).group_by(RiskAssessment.risk_category).subquery()
            
            query = query.join(
                subquery,
                and_(
                    RiskAssessment.risk_category == subquery.c.risk_category,
                    RiskAssessment.assessment_date == subquery.c.max_date
                )
            )
        
        return query.order_by(desc(RiskAssessment.assessment_date)).all()
    
    def get_high_risk_suppliers(
        self,
        risk_threshold: float = 70.0,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get suppliers with high risk scores"""
        # Get latest risk assessment for each supplier
        latest_risks = self.db.query(
            RiskAssessment.supplier_id,
            func.max(RiskAssessment.assessment_date).label('latest_date')
        ).group_by(RiskAssessment.supplier_id).subquery()
        
        query = self.db.query(
            Supplier,
            RiskAssessment
        ).join(
            RiskAssessment
        ).join(
            latest_risks,
            and_(
                RiskAssessment.supplier_id == latest_risks.c.supplier_id,
                RiskAssessment.assessment_date == latest_risks.c.latest_date
            )
        ).filter(
            RiskAssessment.risk_score >= risk_threshold,
            Supplier.status == 'active'
        )
        
        if category:
            query = query.filter(Supplier.category == category)
        
        results = query.all()
        
        return [
            {
                "supplier": supplier,
                "risk_assessment": risk
            }
            for supplier, risk in results
        ]
    
    # =====================================================
    # Compliance Operations
    # =====================================================
    
    def create_compliance_check(
        self,
        supplier_id: UUID,
        check_date: date,
        compliance_type: str,
        status: str,
        score: Optional[float],
        findings: List[str],
        required_actions: List[str],
        due_date: Optional[date],
        checked_by: UUID
    ) -> ComplianceCheck:
        """Create compliance check record"""
        check = ComplianceCheck(
            supplier_id=supplier_id,
            check_date=check_date,
            compliance_type=compliance_type,
            status=status,
            score=score,
            findings=findings,
            required_actions=required_actions,
            due_date=due_date,
            checked_by=checked_by
        )
        self.db.add(check)
        self.db.commit()
        self.db.refresh(check)
        return check
    
    def get_supplier_compliance(
        self,
        supplier_id: UUID,
        compliance_type: Optional[str] = None,
        include_completed: bool = False
    ) -> List[ComplianceCheck]:
        """Get supplier compliance checks"""
        query = self.db.query(ComplianceCheck).filter(
            ComplianceCheck.supplier_id == supplier_id
        )
        
        if compliance_type:
            query = query.filter(ComplianceCheck.compliance_type == compliance_type)
        
        if not include_completed:
            query = query.filter(ComplianceCheck.completed_date.is_(None))
        
        return query.order_by(desc(ComplianceCheck.check_date)).all()
    
    def update_compliance_status(
        self,
        check_id: UUID,
        status: str,
        completed: bool = False
    ) -> Optional[ComplianceCheck]:
        """Update compliance check status"""
        check = self.db.query(ComplianceCheck).filter(
            ComplianceCheck.id == check_id
        ).first()
        
        if not check:
            return None
        
        check.status = status
        if completed:
            check.completed_date = date.today()
        
        self.db.commit()
        self.db.refresh(check)
        return check
    
    # =====================================================
    # Analytics and Statistics
    # =====================================================
    
    def get_supplier_statistics(self) -> Dict[str, Any]:
        """Get overall supplier statistics"""
        total_suppliers = self.db.query(Supplier).filter(
            Supplier.status == 'active'
        ).count()
        
        tier_distribution = self.db.query(
            Supplier.tier,
            func.count(Supplier.id)
        ).filter(
            Supplier.status == 'active'
        ).group_by(Supplier.tier).all()
        
        category_distribution = self.db.query(
            Supplier.category,
            func.count(Supplier.id)
        ).filter(
            Supplier.status == 'active'
        ).group_by(Supplier.category).all()
        
        country_distribution = self.db.query(
            Supplier.country,
            func.count(Supplier.id)
        ).filter(
            Supplier.status == 'active',
            Supplier.country.isnot(None)
        ).group_by(Supplier.country).all()
        
        # Average ratings
        avg_rating = self.db.query(
            func.avg(Supplier.rating)
        ).filter(
            Supplier.status == 'active',
            Supplier.rating.isnot(None)
        ).scalar() or 0
        
        # Recent performance
        recent_performance = self.db.query(
            func.avg(SupplierPerformanceMetric.overall_score)
        ).join(
            Supplier
        ).filter(
            Supplier.status == 'active',
            SupplierPerformanceMetric.metric_date >= date.today() - timedelta(days=30)
        ).scalar() or 0
        
        return {
            "total_suppliers": total_suppliers,
            "tier_distribution": dict(tier_distribution),
            "category_distribution": dict(category_distribution),
            "country_distribution": dict(country_distribution),
            "average_rating": float(avg_rating),
            "recent_average_performance": float(recent_performance)
        }
    
    def get_supplier_spend_analysis(
        self,
        start_date: date,
        end_date: date,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze spending by supplier"""
        query = self.db.query(
            Supplier,
            func.count(Order.id).label('order_count'),
            func.sum(Order.total_amount).label('total_spend')
        ).join(
            Order, Supplier.id == Order.supplier_id
        ).filter(
            Order.order_date.between(start_date, end_date),
            Order.order_type == 'purchase'
        )
        
        if category:
            query = query.filter(Supplier.category == category)
        
        results = query.group_by(Supplier.id).order_by(
            desc('total_spend')
        ).all()
        
        return [
            {
                "supplier": supplier,
                "order_count": order_count,
                "total_spend": float(total_spend or 0)
            }
            for supplier, order_count, total_spend in results
        ]
