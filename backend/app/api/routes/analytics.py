from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, date, timedelta
from enum import Enum
import uuid

from app.analytics.inventory_optimization.safety_stock_calculator import calculate_safety_stock
from app.analytics.inventory_optimization.abc_analysis import perform_abc_analysis
from app.analytics.inventory_optimization.forecast_engine import generate_forecast
from app.analytics.logistics_analytics.route_optimizer import optimize_routes
from app.analytics.logistics_analytics.carrier_performance import analyze_carrier_performance
from app.analytics.logistics_analytics.delivery_analytics import analyze_delivery_performance
from app.analytics.supplier_performance.scorecard import generate_supplier_scorecard
from app.analytics.supplier_performance.risk_analysis import analyze_supplier_risk
from app.analytics.supplier_performance.compliance_checker import check_supplier_compliance
from app.db.interfaces.user_interface import User
from app.security.rbac_manager import check_permission
from app.utils.logger import get_logger
from app.db.schema.schema_discovery import discover_client_schema
from app.llm.prompt.schema_provider import get_database_schema
from app.llm.controller.active_model_manager import get_active_model
from app.llm.prompt.template_manager import get_template

from app.api.routes.auth import get_current_active_user

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}}
)

# Enums
class TimeFrame(str, Enum):
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"
    YEAR_TO_DATE = "year_to_date"

class ForecastMethod(str, Enum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    LSTM = "lstm"

class ABCMethod(str, Enum):
    VALUE = "value"
    VOLUME = "volume"
    CRITICALITY = "criticality"
    COMBINED = "combined"

# Models
class AnalyticsRequest(BaseModel):
    """Base analytics request"""
    client_id: Optional[str] = None
    connection_id: Optional[str] = None
    time_frame: TimeFrame = TimeFrame.LAST_MONTH
    custom_start_date: Optional[date] = None
    custom_end_date: Optional[date] = None

class InventoryAnalysisRequest(AnalyticsRequest):
    """Request for inventory analysis"""
    product_ids: Optional[List[str]] = None
    product_categories: Optional[List[str]] = None
    warehouse_ids: Optional[List[str]] = None
    include_inactive: bool = False

class SafetyStockRequest(InventoryAnalysisRequest):
    """Request for safety stock calculation"""
    service_level: float = 0.95  # Default 95% service level
    lead_time_days: Optional[int] = None
    use_historical_data: bool = True

class ABCAnalysisRequest(InventoryAnalysisRequest):
    """Request for ABC inventory analysis"""
    method: ABCMethod = ABCMethod.VALUE
    a_threshold: float = 0.8  # A items account for 80% of value
    b_threshold: float = 0.95  # A+B items account for 95% of value

class ForecastRequest(InventoryAnalysisRequest):
    """Request for demand forecasting"""
    forecast_periods: int = 12  # Default to 12 periods
    period_type: str = "month"  # month, week, day
    method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING
    include_confidence_intervals: bool = True
    confidence_level: float = 0.95

class LogisticsAnalysisRequest(AnalyticsRequest):
    """Request for logistics analysis"""
    warehouse_ids: Optional[List[str]] = None
    carrier_ids: Optional[List[str]] = None
    destination_regions: Optional[List[str]] = None
    include_returns: bool = False

class SupplierAnalysisRequest(AnalyticsRequest):
    """Request for supplier analysis"""
    supplier_ids: Optional[List[str]] = None
    supplier_categories: Optional[List[str]] = None
    include_tier2_suppliers: bool = False

class RouteOptimizationRequest(BaseModel):
    """Request for route optimization"""
    client_id: Optional[str] = None
    connection_id: Optional[str] = None
    origin_warehouse_id: str
    delivery_locations: List[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]] = None
    optimization_objective: str = "distance"  # distance, time, cost

class SupplierComplianceRequest(BaseModel):
    """Request for supplier compliance check"""
    client_id: Optional[str] = None
    connection_id: Optional[str] = None
    supplier_ids: List[str]
    compliance_types: List[str] = ["quality", "delivery", "documentation"]
    as_of_date: Optional[date] = None

# Helper functions
def get_date_range(time_frame: TimeFrame, custom_start_date: Optional[date] = None, custom_end_date: Optional[date] = None) -> tuple:
    """Calculate the date range based on the time frame"""
    today = date.today()
    end_date = today
    
    if time_frame == TimeFrame.CUSTOM:
        if not custom_start_date or not custom_end_date:
            raise ValueError("Custom time frame requires both start and end dates")
        return custom_start_date, custom_end_date
    
    if time_frame == TimeFrame.LAST_WEEK:
        start_date = today - timedelta(days=7)
    elif time_frame == TimeFrame.LAST_MONTH:
        start_date = today.replace(day=1) - timedelta(days=1)
        start_date = start_date.replace(day=1)
    elif time_frame == TimeFrame.LAST_QUARTER:
        quarter_month = ((today.month - 1) // 3) * 3 + 1
        start_date = today.replace(month=quarter_month, day=1) - timedelta(days=90)
        quarter_month = ((start_date.month - 1) // 3) * 3 + 1
        start_date = start_date.replace(month=quarter_month, day=1)
    elif time_frame == TimeFrame.LAST_YEAR:
        start_date = today.replace(year=today.year-1, month=today.month, day=today.day)
    elif time_frame == TimeFrame.YEAR_TO_DATE:
        start_date = today.replace(month=1, day=1)
    else:
        raise ValueError(f"Unsupported time frame: {time_frame}")
    
    return start_date, end_date

# Routes
@router.post("/inventory/safety-stock", response_model=Dict[str, Any])
async def calculate_safety_stock_levels(
    request: SafetyStockRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Calculate optimal safety stock levels"""
    # Check user has permission
    check_permission(current_user.role, "analytics:inventory:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Calculate safety stock
        results = await calculate_safety_stock(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            product_ids=request.product_ids,
            product_categories=request.product_categories,
            warehouse_ids=request.warehouse_ids,
            service_level=request.service_level,
            lead_time_days=request.lead_time_days,
            use_historical_data=request.use_historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"ss-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Calculated safety stock levels for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error calculating safety stock: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating safety stock: {str(e)}"
        )

@router.post("/inventory/abc-analysis", response_model=Dict[str, Any])
async def abc_inventory_analysis(
    request: ABCAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Perform ABC analysis on inventory"""
    # Check user has permission
    check_permission(current_user.role, "analytics:inventory:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Perform ABC analysis
        results = await perform_abc_analysis(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            product_ids=request.product_ids,
            product_categories=request.product_categories,
            warehouse_ids=request.warehouse_ids,
            method=request.method,
            a_threshold=request.a_threshold,
            b_threshold=request.b_threshold,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"abc-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Performed ABC analysis for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error performing ABC analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing ABC analysis: {str(e)}"
        )

@router.post("/inventory/forecast", response_model=Dict[str, Any])
async def forecast_demand(
    request: ForecastRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Generate demand forecast"""
    # Check user has permission
    check_permission(current_user.role, "analytics:inventory:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range for historical data
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Generate forecast
        results = await generate_forecast(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            product_ids=request.product_ids,
            product_categories=request.product_categories,
            warehouse_ids=request.warehouse_ids,
            forecast_periods=request.forecast_periods,
            period_type=request.period_type,
            method=request.method,
            include_confidence_intervals=request.include_confidence_intervals,
            confidence_level=request.confidence_level,
            historical_start_date=start_date,
            historical_end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "historical_date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "forecast_horizon": {
                "periods": request.forecast_periods,
                "period_type": request.period_type
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"forecast-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Generated demand forecast for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecast: {str(e)}"
        )

@router.post("/logistics/carrier-performance", response_model=Dict[str, Any])
async def carrier_performance_analysis(
    request: LogisticsAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze carrier performance"""
    # Check user has permission
    check_permission(current_user.role, "analytics:logistics:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Analyze carrier performance
        results = await analyze_carrier_performance(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            carrier_ids=request.carrier_ids,
            warehouse_ids=request.warehouse_ids,
            destination_regions=request.destination_regions,
            include_returns=request.include_returns,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"carrier-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Analyzed carrier performance for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error analyzing carrier performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing carrier performance: {str(e)}"
        )

@router.post("/logistics/delivery-performance", response_model=Dict[str, Any])
async def delivery_performance_analysis(
    request: LogisticsAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze delivery performance"""
    # Check user has permission
    check_permission(current_user.role, "analytics:logistics:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Analyze delivery performance
        results = await analyze_delivery_performance(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            warehouse_ids=request.warehouse_ids,
            carrier_ids=request.carrier_ids,
            destination_regions=request.destination_regions,
            include_returns=request.include_returns,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"delivery-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Analyzed delivery performance for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error analyzing delivery performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing delivery performance: {str(e)}"
        )

@router.post("/logistics/route-optimization", response_model=Dict[str, Any])
async def route_optimization(
    request: RouteOptimizationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Optimize delivery routes"""
    # Check user has permission
    check_permission(current_user.role, "analytics:logistics:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Optimize routes
        results = await optimize_routes(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            origin_warehouse_id=request.origin_warehouse_id,
            delivery_locations=request.delivery_locations,
            constraints=request.constraints,
            optimization_objective=request.optimization_objective
        )
        
        response = {
            "request_parameters": request.dict(exclude={"client_id", "connection_id"}),
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"route-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Optimized routes for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error optimizing routes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error optimizing routes: {str(e)}"
        )

@router.post("/supplier/scorecard", response_model=Dict[str, Any])
async def supplier_scorecard(
    request: SupplierAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Generate supplier scorecard"""
    # Check user has permission
    check_permission(current_user.role, "analytics:supplier:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Generate supplier scorecard
        results = await generate_supplier_scorecard(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            supplier_ids=request.supplier_ids,
            supplier_categories=request.supplier_categories,
            include_tier2_suppliers=request.include_tier2_suppliers,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"scorecard-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Generated supplier scorecard for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error generating supplier scorecard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating supplier scorecard: {str(e)}"
        )

@router.post("/supplier/risk-analysis", response_model=Dict[str, Any])
async def supplier_risk_analysis(
    request: SupplierAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze supplier risk"""
    # Check user has permission
    check_permission(current_user.role, "analytics:supplier:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get date range
        start_date, end_date = get_date_range(
            time_frame=request.time_frame,
            custom_start_date=request.custom_start_date,
            custom_end_date=request.custom_end_date
        )
        
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Analyze supplier risk
        results = await analyze_supplier_risk(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            supplier_ids=request.supplier_ids,
            supplier_categories=request.supplier_categories,
            include_tier2_suppliers=request.include_tier2_suppliers,
            start_date=start_date,
            end_date=end_date
        )
        
        response = {
            "request_parameters": request.dict(),
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"risk-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Analyzed supplier risk for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error analyzing supplier risk: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing supplier risk: {str(e)}"
        )

@router.post("/supplier/compliance", response_model=Dict[str, Any])
async def supplier_compliance_check(
    request: SupplierComplianceRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Check supplier compliance"""
    # Check user has permission
    check_permission(current_user.role, "analytics:supplier:view")
    
    # Get client ID
    client_id = request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get database schema
        schema = await discover_client_schema(client_id, request.connection_id)
        
        # Check supplier compliance
        results = await check_supplier_compliance(
            client_id=client_id,
            connection_id=request.connection_id,
            schema=schema,
            supplier_ids=request.supplier_ids,
            compliance_types=request.compliance_types,
            as_of_date=request.as_of_date or date.today()
        )
        
        response = {
            "request_parameters": request.dict(),
            "as_of_date": request.as_of_date or date.today(),
            "results": results,
            "analysis_date": datetime.now(),
            "analysis_id": f"compliance-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Checked supplier compliance for client: {client_id}")
        return response
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error checking supplier compliance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking supplier compliance: {str(e)}"
        )

@router.post("/custom-analysis", response_model=Dict[str, Any])
async def custom_analysis(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Perform custom analysis using LLM"""
    # Check user has permission
    check_permission(current_user.role, "analytics:custom:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get database schema
        schema = await discover_client_schema(client_id, connection_id)
        
        # Prepare schema context for LLM
        schema_context = get_database_schema(schema)
        
        # Get active LLM model
        llm_model = get_active_model()
        
        # Get prompt template
        template = get_template("custom_analysis")
        
        # Build context for analysis
        context = {
            "query": query,
            "parameters": parameters or {},
            "schema": schema_context,
            "domain": "supply_chain"
        }
        
        # Execute LLM to design and perform the analysis
        llm_response = await llm_model.generate(
            prompt_template=template,
            context=context
        )
        
        # Extract SQL from LLM response
        sql = llm_response.get("sql", "")
        
        # Execute the SQL against the database
        db_connector = PostgresConnector(client_id=client_id, connection_id=connection_id)
        results = await db_connector.execute_query(sql)
        
        # Let the LLM interpret the results
        interpretation_template = get_template("analysis_interpretation")
        interpretation_context = {
            "query": query,
            "parameters": parameters or {},
            "sql": sql,
            "results": results,
            "schema": schema_context
        }
        
        interpretation = await llm_model.generate(
            prompt_template=interpretation_template,
            context=interpretation_context
        )
        
        response = {
            "query": query,
            "parameters": parameters or {},
            "sql": sql,
            "results": results,
            "interpretation": interpretation.get("interpretation", ""),
            "visualization_suggestions": interpretation.get("visualization_suggestions", []),
            "analysis_date": datetime.now(),
            "analysis_id": f"custom-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Performed custom analysis for client: {client_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error performing custom analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing custom analysis: {str(e)}"
        )