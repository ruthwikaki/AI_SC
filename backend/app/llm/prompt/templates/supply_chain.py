# app/llm/prompt/templates/supply_chain.py

# Supply Chain Prompt Templates

# Template for safety stock calculation
SAFETY_STOCK_CALCULATION = {
    "name": "safety_stock_calculation",
    "description": "Calculates safety stock levels based on demand and lead time",
    "type": "standard",
    "content": """Calculate appropriate safety stock levels for the following product(s) based on the provided data and service level requirements.

PRODUCT DATA:
{product_data}

SERVICE LEVEL: {service_level}

LEAD TIME DATA:
{lead_time_data}

HISTORICAL DEMAND:
{demand_data}

Apply the following formula for safety stock calculation:
Safety Stock = Z-score × Standard Deviation of Demand during Lead Time

Where:
- Z-score is derived from the service level (e.g., 1.645 for 95% service level)
- Standard Deviation of Demand during Lead Time incorporates both demand and lead time variability

For each product, provide:
1. Calculated safety stock level
2. Explanation of calculation
3. Recommendations for inventory policy

Format your response as JSON with these keys:
- "calculations": Array of product calculations
- "methodology": Brief explanation of calculation method
- "recommendations": General inventory management recommendations"""
}

# Template for ABC inventory analysis
ABC_INVENTORY_ANALYSIS = {
    "name": "abc_inventory_analysis",
    "description": "Performs ABC analysis on inventory items",
    "type": "standard",
    "content": """Perform an ABC inventory analysis on the provided product data to categorize items based on their importance.

PRODUCT DATA:
{product_data}

ANALYSIS PARAMETERS:
- Criteria: {criteria} (e.g., annual usage value, pick frequency, etc.)
- Class A threshold: {class_a_threshold} (e.g., top 80% of value)
- Class B threshold: {class_b_threshold} (e.g., next 15% of value)
- Class C threshold: {class_c_threshold} (e.g., final 5% of value)

For the analysis:
1. Calculate the value (or specified criteria) for each product
2. Sort products in descending order of value
3. Calculate cumulative percentage of total value
4. Assign ABC classes based on the thresholds
5. Provide specific inventory management recommendations for each class

Format your response as JSON with these keys:
- "analysis_summary": Summary statistics of the analysis
- "class_a_items": List of Class A items with details
- "class_b_items": List of Class B items with details
- "class_c_items": List of Class C items with details
- "recommendations": Specific recommendations for each class"""
}

# Template for demand forecasting
DEMAND_FORECAST = {
    "name": "demand_forecast",
    "description": "Generates demand forecasts based on historical data",
    "type": "standard",
    "content": """Generate a demand forecast for the specified products based on historical data and relevant factors.

HISTORICAL DEMAND DATA:
{historical_data}

FORECAST PARAMETERS:
- Forecast horizon: {forecast_horizon} periods
- Method: {forecast_method}
- Confidence level: {confidence_level}

ADDITIONAL FACTORS:
{additional_factors}

For each product/time period in the forecast horizon:
1. Calculate the point forecast
2. Provide confidence intervals based on the specified confidence level
3. Identify potential factors that could impact the forecast
4. Flag any anomalies or concerns

Format your response as JSON with these keys:
- "forecast": Array of forecast periods with predicted values and confidence intervals
- "methodology": Description of forecasting methodology used
- "factors": Relevant factors considered in the forecast
- "anomalies": Any identified anomalies or concerns
- "recommendations": Recommendations for inventory planning based on the forecast"""
}

# Template for supplier performance analysis
SUPPLIER_PERFORMANCE = {
    "name": "supplier_performance",
    "description": "Analyzes supplier performance metrics",
    "type": "standard",
    "content": """Analyze the performance of the specified supplier(s) based on the provided metrics and data.

SUPPLIER DATA:
{supplier_data}

PERFORMANCE METRICS:
{performance_metrics}

TIME PERIOD: {time_period}

For each supplier, analyze:
1. On-time delivery performance
2. Quality metrics and defect rates
3. Price competitiveness and cost trends
4. Responsiveness and communication
5. Overall performance score and recommendations

Compare the supplier's performance to:
- Historical performance trends
- Industry benchmarks where available
- Performance targets or SLAs
- Other suppliers in similar categories

Format your response as JSON with these keys:
- "supplier_scores": Performance scores across all metrics
- "strengths": Key strengths identified
- "areas_for_improvement": Areas needing improvement
- "trends": Notable performance trends
- "action_items": Recommended actions to address issues or capitalize on strengths
- "risk_assessment": Assessment of supply risk based on performance"""
}

# Template for route optimization
ROUTE_OPTIMIZATION = {
    "name": "route_optimization",
    "description": "Optimizes delivery routes for efficiency",
    "type": "standard",
    "content": """Optimize delivery routes for the provided locations and constraints to minimize cost and maximize efficiency.

ORIGIN POINT:
{origin}

DELIVERY LOCATIONS:
{delivery_locations}

CONSTRAINTS:
{constraints}

OPTIMIZATION OBJECTIVE: {optimization_objective}

Using the provided data:
1. Determine the optimal sequence of deliveries
2. Calculate estimated transit times between stops
3. Consider all constraints (time windows, vehicle capacity, etc.)
4. Provide the complete route plan with expected arrival times
5. Calculate total distance, time, and cost

Format your response as JSON with these keys:
- "route": Ordered array of stops with details
- "metrics": Summary of route metrics (distance, time, cost)
- "constraints_satisfied": How all constraints were satisfied
- "alternative_routes": Possible alternative routes with trade-offs
- "recommendations": Additional recommendations for route improvement"""
}

# Template for multi-tier supply chain risk analysis
SUPPLY_CHAIN_RISK = {
    "name": "supply_chain_risk",
    "description": "Analyzes risks across the multi-tier supply chain",
    "type": "standard",
    "content": """Analyze risks across the multi-tier supply chain network based on the provided data and identify potential mitigation strategies.

SUPPLY CHAIN NETWORK:
{network_data}

RISK FACTORS:
{risk_factors}

HISTORICAL DISRUPTIONS:
{historical_disruptions}

For the analysis:
1. Identify critical nodes and bottlenecks in the supply chain
2. Evaluate risk exposure at each tier (Tier 1, Tier 2, Tier 3+)
3. Assess the potential impact of identified risks
4. Calculate risk scores for key suppliers and components
5. Recommend mitigation strategies for high-risk areas

Consider these risk categories:
- Supplier financial health
- Geographic/regional risks
- Single-source dependencies
- Capacity constraints
- Quality issues
- Compliance and regulatory risks
- Transportation and logistics risks

Format your response as JSON with these keys:
- "critical_nodes": Identified critical nodes and bottlenecks
- "tier_analysis": Risk assessment by tier
- "high_risk_areas": Specific high-risk areas requiring attention
- "risk_scores": Calculated risk scores for key elements
- "mitigation_strategies": Recommended risk mitigation strategies
- "monitoring_recommendations": Ongoing risk monitoring recommendations"""
}