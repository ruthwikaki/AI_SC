# Add this import at the top of the file if not already present
import json

# Add these methods inside the UserInterface class

async def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
    """
    Get user's dashboard preferences.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with dashboard preferences
    """
    try:
        connector = await get_connector_for_client(self.admin_db_client_id)
        
        query = """
        SELECT preference_value 
        FROM user_preferences
        WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
        """
        
        result = await connector.execute_query(query, {"user_id": user_id})
        
        if result["data"] and result["data"][0].get("preference_value"):
            # Parse JSON preference value
            pref_value = result["data"][0]["preference_value"]
            if isinstance(pref_value, str):
                return json.loads(pref_value)
            return pref_value
        
        # Return default preferences
        return {
            "selected_metrics": [
                "inventory_value",
                "order_fill_rate",
                "on_time_delivery",
                "supplier_performance"
            ],
            "refresh_interval": 300,  # 5 minutes
            "time_frame": "last_month",
            "chart_preferences": {
                "show_trends": True,
                "show_abc_analysis": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard preferences: {str(e)}")
        # Return defaults on error
        return {
            "selected_metrics": [
                "inventory_value",
                "order_fill_rate",
                "on_time_delivery",
                "supplier_performance"
            ],
            "refresh_interval": 300,
            "time_frame": "last_month"
        }
    finally:
        if locals().get("connector"):
            await connector.close()

async def save_user_dashboard_preferences(
    self, 
    user_id: str, 
    preferences: Dict[str, Any]
) -> bool:
    """
    Save user's dashboard preferences.
    
    Args:
        user_id: User ID
        preferences: Dashboard preferences dictionary
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        connector = await get_connector_for_client(self.admin_db_client_id)
        
        # Check if preferences exist
        check_query = """
        SELECT id FROM user_preferences
        WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
        """
        
        check_result = await connector.execute_query(
            check_query, 
            {"user_id": user_id}
        )
        
        if check_result["data"]:
            # Update existing
            query = """
            UPDATE user_preferences
            SET preference_value = :value, updated_at = :updated_at
            WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
            """
            
            params = {
                "user_id": user_id,
                "value": json.dumps(preferences),
                "updated_at": datetime.now().isoformat()
            }
        else:
            # Insert new
            query = """
            INSERT INTO user_preferences (
                id, user_id, preference_key, preference_value, created_at
            )
            VALUES (
                :id, :user_id, :key, :value, :created_at
            )
            """
            
            params = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "key": "dashboard_metrics",
                "value": json.dumps(preferences),
                "created_at": datetime.now().isoformat()
            }
        
        await connector.execute_query(query, params)
        logger.info(f"Saved dashboard preferences for user: {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dashboard preferences: {str(e)}")
        return False
    finally:
        if locals().get("connector"):
            await connector.close()

async def get_user_accessible_metrics(self, user_id: str) -> List[str]:
    """
    Get list of metrics accessible to the user based on their role.
    
    Args:
        user_id: User ID
        
    Returns:
        List of accessible metric keys
    """
    try:
        # Get user to check role
        user = await self.get_user(user_id)
        if not user:
            return []
        
        role = user.role
        
        # Define metrics by role
        role_metrics = {
            "admin": [
                "inventory_value",
                "order_fill_rate",
                "on_time_delivery",
                "supplier_performance",
                "total_revenue",
                "cost_savings",
                "cash_cycle",
                "network_efficiency"
            ],
            "manager": [
                "inventory_value",
                "order_fill_rate",
                "on_time_delivery",
                "supplier_performance",
                "total_revenue"
            ],
            "user": [
                "inventory_value",
                "order_fill_rate",
                "on_time_delivery"
            ],
            "viewer": [
                "inventory_value",
                "order_fill_rate"
            ]
        }
        
        return role_metrics.get(role, role_metrics["user"])
        
    except Exception as e:
        logger.error(f"Error getting accessible metrics: {str(e)}")
        return ["inventory_value", "order_fill_rate"]  # Minimum defaults