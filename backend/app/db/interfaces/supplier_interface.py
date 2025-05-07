from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import json


from app.db.schema.schema_discovery import get_connector_for_client
from app.db.schema.schema_mapper import get_domain_mappings
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


class SupplierInterface:
    """Interface for supplier-related database operations"""
   
    def __init__(self, client_id: str, connection_id: Optional[str] = None):
        """
        Initialize the supplier interface.
       
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
        """
        self.client_id = client_id
        self.connection_id = connection_id
   
    async def get_suppliers(
        self,
        supplier_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tiers: Optional[List[int]] = None,
        include_inactive: bool = False,
        search_term: Optional[str] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get suppliers with filtering and pagination.
       
        Args:
            supplier_ids: Optional list of supplier IDs to filter by
            categories: Optional list of supplier categories to filter by
            tiers: Optional list of supplier tiers to filter by
            include_inactive: Whether to include inactive suppliers
            search_term: Optional search term for supplier name
            page: Page number (1-based)
            page_size: Number of items per page
           
        Returns:
            Tuple of (suppliers_list, total_count)
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings to find the relevant tables and columns
            mappings = await get_domain_mappings(self.client_id)
           
            # Find supplier table
            supplier_table = self._find_table_by_concept(mappings, "supplier")
           
            if not supplier_table:
                logger.warning(f"No supplier table mapping found for client {self.client_id}")
                return [], 0
           
            # Get supplier columns
            supplier_columns = self._get_columns_for_table(mappings, supplier_table, "supplier")
           
            # Build query
            select_clause = ["*"]
            where_clause = []
            params = {}
           
            # Add supplier ID filter if requested
            if supplier_ids:
                id_col = self._get_primary_key(mappings, supplier_table)
                placeholders = []
                for i, supp_id in enumerate(supplier_ids):
                    param_name = f"supplier_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = supp_id
               
                where_clause.append(f"{id_col} IN ({', '.join(placeholders)})")
           
            # Add category filter if requested
            if categories and 'category' in supplier_columns:
                category_col = supplier_columns['category']
                placeholders = []
                for i, category in enumerate(categories):
                    param_name = f"category_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = category
               
                where_clause.append(f"{category_col} IN ({', '.join(placeholders)})")
           
            # Add tier filter if requested
            if tiers and 'tier' in supplier_columns:
                tier_col = supplier_columns['tier']
                placeholders = []
                for i, tier in enumerate(tiers):
                    param_name = f"tier_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = tier
               
                where_clause.append(f"{tier_col} IN ({', '.join(placeholders)})")
           
            # Add active filter if requested
            if not include_inactive and 'is_active' in supplier_columns:
                where_clause.append(f"{supplier_columns['is_active']} = :is_active")
                params["is_active"] = True
           
            # Add search filter if requested
            if search_term and 'name' in supplier_columns:
                name_col = supplier_columns['name']
                where_clause.append(f"{name_col} LIKE :search_term")
                params["search_term"] = f"%{search_term}%"
           
            # Build the final query
            query = f"SELECT {', '.join(select_clause)} FROM {supplier_table}"
           
            if where_clause:
                query += " WHERE " + " AND ".join(where_clause)
           
            # Add count query for pagination
            count_query = f"SELECT COUNT(*) as total FROM {supplier_table}"
            if where_clause:
                count_query += " WHERE " + " AND ".join(where_clause)
           
            # Add ordering
            if 'name' in supplier_columns:
                query += f" ORDER BY {supplier_columns['name']}"
           
            # Add pagination
            offset = (page - 1) * page_size
            query += f" LIMIT {page_size} OFFSET {offset}"
           
            # Execute queries
            count_result = await connector.execute_query(count_query, params)
            result = await connector.execute_query(query, params)
           
            # Get total count
            total_count = 0
            if count_result["data"]:
                # Different databases might return the count with different column names
                if "total" in count_result["data"][0]:
                    total_count = int(count_result["data"][0]["total"])
                elif "count" in count_result["data"][0]:
                    total_count = int(count_result["data"][0]["count"])
           
            return result["data"], total_count
           
        except Exception as e:
            logger.error(f"Error retrieving suppliers: {str(e)}")
            return [], 0
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_supplier(self, supplier_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific supplier by ID.
       
        Args:
            supplier_id: Supplier ID
           
        Returns:
            Supplier data or None if not found
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find supplier table
            supplier_table = self._find_table_by_concept(mappings, "supplier")
           
            if not supplier_table:
                logger.warning(f"No supplier table mapping found for client {self.client_id}")
                return None
           
            # Get ID column
            id_col = self._get_primary_key(mappings, supplier_table)
           
            # Query supplier
            query = f"SELECT * FROM {supplier_table} WHERE {id_col} = :supplier_id"
            result = await connector.execute_query(query, {"supplier_id": supplier_id})
           
            # Return first item if found
            if result["data"]:
                return result["data"][0]
           
            return None
           
        except Exception as e:
            logger.error(f"Error retrieving supplier: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_supplier_orders(
        self,
        supplier_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get orders for a specific supplier.
       
        Args:
            supplier_id: Supplier ID
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            status: Optional order status for filtering
            page: Page number (1-based)
            page_size: Number of items per page
           
        Returns:
            Tuple of (orders_list, total_count)
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order and supplier tables
            order_table = self._find_table_by_concept(mappings, "purchase_order")
           
            if not order_table:
                logger.warning(f"No order table mapping found for client {self.client_id}")
                return [], 0
           
            # Get order columns
            order_columns = self._get_columns_for_table(mappings, order_table, "purchase_order")
           
            # Build query
            select_clause = ["*"]
            where_clause = []
            params = {}
           
            # Add supplier filter
            if 'supplier' in order_columns:
                supplier_col = order_columns['supplier']
                where_clause.append(f"{supplier_col} = :supplier_id")
                params["supplier_id"] = supplier_id
           
            # Add date filters if requested
            if start_date and 'order_date' in order_columns:
                order_date_col = order_columns['order_date']
                where_clause.append(f"{order_date_col} >= :start_date")
                params["start_date"] = start_date.isoformat()
           
            if end_date and 'order_date' in order_columns:
                order_date_col = order_columns['order_date']
                where_clause.append(f"{order_date_col} <= :end_date")
                params["end_date"] = end_date.isoformat()
           
            # Add status filter if requested
            if status and 'status' in order_columns:
                status_col = order_columns['status']
                where_clause.append(f"{status_col} = :status")
                params["status"] = status
           
            # Build the final query
            query = f"SELECT {', '.join(select_clause)} FROM {order_table}"
           
            if where_clause:
                query += " WHERE " + " AND ".join(where_clause)
           
            # Add count query for pagination
            count_query = f"SELECT COUNT(*) as total FROM {order_table}"
            if where_clause:
                count_query += " WHERE " + " AND ".join(where_clause)
           
            # Add ordering
            if 'order_date' in order_columns:
                query += f" ORDER BY {order_columns['order_date']} DESC"
           
            # Add pagination
            offset = (page - 1) * page_size
            query += f" LIMIT {page_size} OFFSET {offset}"
           
            # Execute queries
            count_result = await connector.execute_query(count_query, params)
            result = await connector.execute_query(query, params)
           
            # Get total count
            total_count = 0
            if count_result["data"]:
                # Different databases might return the count with different column names
                if "total" in count_result["data"][0]:
                    total_count = int(count_result["data"][0]["total"])
                elif "count" in count_result["data"][0]:
                    total_count = int(count_result["data"][0]["count"])
           
            return result["data"], total_count
           
        except Exception as e:
            logger.error(f"Error retrieving supplier orders: {str(e)}")
            return [], 0
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_supplier_performance(
        self,
        supplier_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate supplier performance metrics.
       
        Args:
            supplier_id: Supplier ID
            start_date: Optional start date for metrics
            end_date: Optional end date for metrics
           
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, "purchase_order")
           
            if not order_table:
                logger.warning(f"No order table mapping found for client {self.client_id}")
                return {"error": "No order data available"}
           
            # Get order columns
            order_columns = self._get_columns_for_table(mappings, order_table, "purchase_order")
           
            # Set date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=90)  # Default to 90 days
           
            # Build query params
            params = {
                "supplier_id": supplier_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
           
            # Get required column names
            supplier_col = order_columns.get('supplier', 'supplier_id')
            order_date_col = order_columns.get('order_date', 'order_date')
            expected_del_col = order_columns.get('expected_delivery', 'expected_delivery')
            actual_del_col = order_columns.get('actual_delivery', 'actual_delivery')
            status_col = order_columns.get('status', 'status')
           
            # Query 1: On-time delivery performance
            otd_query = f"""
            SELECT
                COUNT(*) as total_orders,
                SUM(CASE WHEN {actual_del_col} <= {expected_del_col} THEN 1 ELSE 0 END) as on_time_orders
            FROM {order_table}
            WHERE {supplier_col} = :supplier_id
            AND {order_date_col} BETWEEN :start_date AND :end_date
            AND {actual_del_col} IS NOT NULL
            """
           
            # Query 2: Average delivery delay
            delay_query = f"""
            SELECT
                AVG(CASE WHEN {actual_del_col} IS NOT NULL
                    THEN (JULIANDAY({actual_del_col}) - JULIANDAY({expected_del_col}))
                    ELSE 0 END) as avg_delay_days
            FROM {order_table}
            WHERE {supplier_col} = :supplier_id
            AND {order_date_col} BETWEEN :start_date AND :end_date
            AND {actual_del_col} IS NOT NULL
            """
           
            # Query 3: Order fulfillment rate
            fulfill_query = f"""
            SELECT
                COUNT(*) as total_completed,
                SUM(CASE WHEN {status_col} = 'cancelled' THEN 1 ELSE 0 END) as cancelled_orders
            FROM {order_table}
            WHERE {supplier_col} = :supplier_id
            AND {order_date_col} BETWEEN :start_date AND :end_date
            AND ({status_col} = 'completed' OR {status_col} = 'cancelled')
            """
           
            # Execute queries
            try:
                otd_result = await connector.execute_query(otd_query, params)
            except Exception:
                # Fall back to a simpler query if the database doesn't support complex functions
                otd_query = f"""
                SELECT COUNT(*) as total_orders
                FROM {order_table}
                WHERE {supplier_col} = :supplier_id
                AND {order_date_col} BETWEEN :start_date AND :end_date
                """
                otd_result = await connector.execute_query(otd_query, params)
                otd_result["data"][0]["on_time_orders"] = 0
           
            try:
                delay_result = await connector.execute_query(delay_query, params)
            except Exception:
                # Fall back
                delay_result = {"data": [{"avg_delay_days": 0}]}
           
            try:
                fulfill_result = await connector.execute_query(fulfill_query, params)
            except Exception:
                # Fall back
                fulfill_result = {"data": [{"total_completed": 0, "cancelled_orders": 0}]}
           
            # Calculate metrics
            total_orders = otd_result["data"][0].get("total_orders", 0)
            on_time_orders = otd_result["data"][0].get("on_time_orders", 0)
            on_time_delivery_rate = (on_time_orders / total_orders) if total_orders > 0 else 0
           
            avg_delay_days = delay_result["data"][0].get("avg_delay_days", 0)
            if avg_delay_days is None:
                avg_delay_days = 0
           
            total_completed = fulfill_result["data"][0].get("total_completed", 0)
            cancelled_orders = fulfill_result["data"][0].get("cancelled_orders", 0)
            fulfillment_rate = ((total_completed - cancelled_orders) / total_completed) if total_completed > 0 else 0
           
            # Create performance metrics
            performance = {
                "supplier_id": supplier_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "metrics": {
                    "on_time_delivery_rate": on_time_delivery_rate,
                    "average_delay_days": avg_delay_days,
                    "order_fulfillment_rate": fulfillment_rate,
                    "total_orders": total_orders
                }
            }
           
            return performance
           
        except Exception as e:
            logger.error(f"Error calculating supplier performance: {str(e)}")
            return {"error": f"Error calculating performance: {str(e)}"}
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def update_supplier(
        self,
        supplier_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update a supplier.
       
        Args:
            supplier_id: Supplier ID
            update_data: Data to update
           
        Returns:
            Updated supplier data or None if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find supplier table
            supplier_table = self._find_table_by_concept(mappings, "supplier")
           
            if not supplier_table:
                logger.warning(f"No supplier table mapping found for client {self.client_id}")
                return None
           
            # Get supplier columns and primary key
            supplier_columns = self._get_columns_for_table(mappings, supplier_table, "supplier")
            id_col = self._get_primary_key(mappings, supplier_table)
           
            # Build update query
            set_clauses = []
            params = {"supplier_id": supplier_id}
           
            # Map domain attributes to database columns
            for attr, value in update_data.items():
                if attr in supplier_columns:
                    db_column = supplier_columns[attr]
                    set_clauses.append(f"{db_column} = :{attr}")
                    params[attr] = value
           
            # Add updated_at if the column exists
            try:
                columns_result = await connector.get_table_schema(supplier_table)
                column_names = [col["name"].lower() for col in columns_result.get("columns", [])]
                if "updated_at" in column_names:
                    set_clauses.append("updated_at = :updated_at")
                    params["updated_at"] = datetime.now().isoformat()
            except Exception:
                # If we can't get schema info, just skip this
                pass
           
            # If nothing to update, return current supplier
            if not set_clauses:
                return await self.get_supplier(supplier_id)
           
            # Build the update query
            query = f"""
            UPDATE {supplier_table}
            SET {", ".join(set_clauses)}
            WHERE {id_col} = :supplier_id
            """
           
            # Execute update
            await connector.execute_query(query, params)
           
            # Return updated supplier
            return await self.get_supplier(supplier_id)
           
        except Exception as e:
            logger.error(f"Error updating supplier: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    def _find_table_by_concept(self, mappings: List[Dict[str, Any]], concept: str) -> Optional[str]:
        """Find a table name by domain concept"""
        for mapping in mappings:
            if mapping.get("domain_concept") == concept:
                return mapping.get("custom_table")
        return None
   
    def _get_columns_for_table(self, mappings: List[Dict[str, Any]], table_name: str, concept: str) -> Dict[str, str]:
        """Get column mappings for a table and concept"""
        columns = {}
        for mapping in mappings:
            if mapping.get("custom_table") == table_name and mapping.get("domain_concept") == concept:
                # Map domain attribute to column name
                columns[mapping.get("domain_attribute")] = mapping.get("custom_column")
        return columns
   
    def _get_primary_key(self, mappings: List[Dict[str, Any]], table_name: str) -> str:
        """Get primary key column for a table"""
        # Try to find ID mapping
        for mapping in mappings:
            if mapping.get("custom_table") == table_name and mapping.get("domain_attribute") == "id":
                return mapping.get("custom_column")
       
        # Default to 'id' if no mapping found
        return "id"