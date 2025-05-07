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


class OrderInterface:
    """Interface for order-related database operations"""
   
    def __init__(self, client_id: str, connection_id: Optional[str] = None):
        """
        Initialize the order interface.
       
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
        """
        self.client_id = client_id
        self.connection_id = connection_id
   
    async def get_orders(
        self,
        order_ids: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None,
        customer_ids: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        order_type: str = "purchase_order",  # or "sales_order"
        page: int = 1,
        page_size: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get orders with filtering and pagination.
       
        Args:
            order_ids: Optional list of order IDs to filter by
            supplier_ids: Optional list of supplier IDs to filter by
            customer_ids: Optional list of customer IDs to filter by
            statuses: Optional list of order statuses to filter by
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            order_type: Type of order (purchase_order or sales_order)
            page: Page number (1-based)
            page_size: Number of items per page
           
        Returns:
            Tuple of (orders_list, total_count)
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings to find the relevant tables and columns
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, order_type)
           
            if not order_table:
                logger.warning(f"No {order_type} table mapping found for client {self.client_id}")
                return [], 0
           
            # Get order columns
            order_columns = self._get_columns_for_table(mappings, order_table, order_type)
           
            # Build query
            select_clause = ["*"]
            where_clause = []
            params = {}
           
            # Add order ID filter if requested
            if order_ids:
                id_col = self._get_primary_key(mappings, order_table)
                placeholders = []
                for i, order_id in enumerate(order_ids):
                    param_name = f"order_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = order_id
               
                where_clause.append(f"{id_col} IN ({', '.join(placeholders)})")
           
            # Add supplier filter if requested for purchase orders
            if supplier_ids and order_type == "purchase_order" and 'supplier' in order_columns:
                supplier_col = order_columns['supplier']
                placeholders = []
                for i, supplier_id in enumerate(supplier_ids):
                    param_name = f"supplier_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = supplier_id
               
                where_clause.append(f"{supplier_col} IN ({', '.join(placeholders)})")
           
            # Add customer filter if requested for sales orders
            if customer_ids and order_type == "sales_order" and 'customer' in order_columns:
                customer_col = order_columns['customer']
                placeholders = []
                for i, customer_id in enumerate(customer_ids):
                    param_name = f"customer_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = customer_id
               
                where_clause.append(f"{customer_col} IN ({', '.join(placeholders)})")
           
            # Add status filter if requested
            if statuses and 'status' in order_columns:
                status_col = order_columns['status']
                placeholders = []
                for i, status in enumerate(statuses):
                    param_name = f"status_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = status
               
                where_clause.append(f"{status_col} IN ({', '.join(placeholders)})")
           
            # Add date filters if requested
            if start_date and 'order_date' in order_columns:
                order_date_col = order_columns['order_date']
                where_clause.append(f"{order_date_col} >= :start_date")
                params["start_date"] = start_date.isoformat()
           
            if end_date and 'order_date' in order_columns:
                order_date_col = order_columns['order_date']
                where_clause.append(f"{order_date_col} <= :end_date")
                params["end_date"] = end_date.isoformat()
           
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
            logger.error(f"Error retrieving orders: {str(e)}")
            return [], 0
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_order(
        self,
        order_id: str,
        order_type: str = "purchase_order",  # or "sales_order"
        include_items: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific order by ID.
       
        Args:
            order_id: Order ID
            order_type: Type of order (purchase_order or sales_order)
            include_items: Whether to include order line items
           
        Returns:
            Order data or None if not found
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, order_type)
           
            if not order_table:
                logger.warning(f"No {order_type} table mapping found for client {self.client_id}")
                return None
           
            # Get ID column
            id_col = self._get_primary_key(mappings, order_table)
           
            # Query order
            query = f"SELECT * FROM {order_table} WHERE {id_col} = :order_id"
            result = await connector.execute_query(query, {"order_id": order_id})
           
            # If order not found
            if not result["data"]:
                return None
           
            # Get order data
            order = result["data"][0]
           
            # Add line items if requested
            if include_items:
                items = await self._get_order_items(order_id)
                order["items"] = items
           
            return order
           
        except Exception as e:
            logger.error(f"Error retrieving order: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
async def create_order(
        self,
        order_data: Dict[str, Any],
        order_type: str = "purchase_order"  # or "sales_order"
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new order.
       
        Args:
            order_data: Order data
            order_type: Type of order (purchase_order or sales_order)
           
        Returns:
            Created order data or None if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, order_type)
           
            if not order_table:
                logger.warning(f"No {order_type} table mapping found for client {self.client_id}")
                return None
           
            # Get order columns
            order_columns = self._get_columns_for_table(mappings, order_table, order_type)
           
            # Prepare order data
            db_columns = []
            placeholders = []
            params = {}
           
            # Generate ID if not provided
            if "id" not in order_data:
                order_data["id"] = str(uuid.uuid4())
           
            # Map domain attributes to database columns
            for attr, value in order_data.items():
                if attr == "items":
                    continue  # Handle items separately
               
                if attr in order_columns:
                    db_column = order_columns[attr]
                    db_columns.append(db_column)
                    placeholders.append(f":{attr}")
                    params[attr] = value
           
            # Add created_at if not provided
            if "created_at" not in params:
                db_columns.append("created_at")
                placeholders.append(":created_at")
                params["created_at"] = datetime.now().isoformat()
           
            # Build the insert query
            query = f"""
            INSERT INTO {order_table} ({', '.join(db_columns)})
            VALUES ({', '.join(placeholders)})
            """
           
            # Execute insert
            await connector.execute_query(query, params)
           
            # Create order items if provided
            if "items" in order_data and order_data["items"]:
                await self._create_order_items(order_data["id"], order_data["items"])
           
            # Return created order
            return await self.get_order(order_data["id"], order_type)
           
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def update_order_status(
        self,
        order_id: str,
        status: str,
        order_type: str = "purchase_order"  # or "sales_order"
    ) -> Optional[Dict[str, Any]]:
        """
        Update an order's status.
       
        Args:
            order_id: Order ID
            status: New status
            order_type: Type of order (purchase_order or sales_order)
           
        Returns:
            Updated order data or None if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, order_type)
           
            if not order_table:
                logger.warning(f"No {order_type} table mapping found for client {self.client_id}")
                return None
           
            # Get order columns and primary key
            order_columns = self._get_columns_for_table(mappings, order_table, order_type)
            id_col = self._get_primary_key(mappings, order_table)
           
            # Find status column
            status_col = order_columns.get('status')
            if not status_col:
                logger.warning(f"No status column mapping found for {order_type}")
                return None
           
            # Build update query
            query = f"""
            UPDATE {order_table}
            SET {status_col} = :status
            """
           
            # Add updated_at if the column exists
            try:
                columns_result = await connector.get_table_schema(order_table)
                column_names = [col["name"].lower() for col in columns_result.get("columns", [])]
                if "updated_at" in column_names:
                    query += ", updated_at = :updated_at"
            except Exception:
                # If we can't get schema info, just skip this
                pass
           
            # Add where clause
            query += f" WHERE {id_col} = :order_id"
           
            # Execute update
            params = {
                "order_id": order_id,
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
           
            await connector.execute_query(query, params)
           
            # Log status change
            await self._log_order_status_change(order_id, status, order_type)
           
            # Return updated order
            return await self.get_order(order_id, order_type)
           
        except Exception as e:
            logger.error(f"Error updating order status: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_order_history(
        self,
        days: int = 30,
        order_type: str = "purchase_order"  # or "sales_order"
    ) -> Dict[str, Any]:
        """
        Get order history statistics.
       
        Args:
            days: Number of days to look back
            order_type: Type of order (purchase_order or sales_order)
           
        Returns:
            Dictionary with order history statistics
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order table
            order_table = self._find_table_by_concept(mappings, order_type)
           
            if not order_table:
                logger.warning(f"No {order_type} table mapping found for client {self.client_id}")
                return {"error": "No order data available"}
           
            # Get order columns
            order_columns = self._get_columns_for_table(mappings, order_table, order_type)
           
            # Get required column names
            order_date_col = order_columns.get('order_date', 'order_date')
            status_col = order_columns.get('status', 'status')
            total_col = order_columns.get('total_amount', 'total_amount')
           
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
           
            # Query 1: Total orders by status
            status_query = f"""
            SELECT
                {status_col} as status,
                COUNT(*) as count
            FROM {order_table}
            WHERE {order_date_col} BETWEEN :start_date AND :end_date
            GROUP BY {status_col}
            """
           
            # Query 2: Orders by day
            daily_query = f"""
            SELECT
                date({order_date_col}) as date,
                COUNT(*) as count,
                SUM({total_col}) as total_value
            FROM {order_table}
            WHERE {order_date_col} BETWEEN :start_date AND :end_date
            GROUP BY date({order_date_col})
            ORDER BY date({order_date_col})
            """
           
            # Query 3: Total statistics
            totals_query = f"""
            SELECT
                COUNT(*) as total_orders,
                SUM({total_col}) as total_value,
                AVG({total_col}) as average_value
            FROM {order_table}
            WHERE {order_date_col} BETWEEN :start_date AND :end_date
            """
           
            # Common params
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
           
            # Execute queries
            try:
                status_result = await connector.execute_query(status_query, params)
                daily_result = await connector.execute_query(daily_query, params)
                totals_result = await connector.execute_query(totals_query, params)
            except Exception as ex:
                logger.error(f"Error executing order history queries: {str(ex)}")
                # Return empty results on error
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "by_status": [],
                    "by_date": [],
                    "totals": {
                        "total_orders": 0,
                        "total_value": 0,
                        "average_value": 0
                    }
                }
           
            # Process results
            by_status = {}
            for item in status_result["data"]:
                by_status[item.get("status", "unknown")] = item.get("count", 0)
           
            # Create the response
            history = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "by_status": by_status,
                "by_date": daily_result["data"],
                "totals": totals_result["data"][0] if totals_result["data"] else {
                    "total_orders": 0,
                    "total_value": 0,
                    "average_value": 0
                }
            }
           
            return history
           
        except Exception as e:
            logger.error(f"Error retrieving order history: {str(e)}")
            return {"error": f"Error retrieving order history: {str(e)}"}
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def _get_order_items(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get line items for an order.
       
        Args:
            order_id: Order ID
           
        Returns:
            List of order line items
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order_line table
            order_line_table = self._find_table_by_concept(mappings, "order_line")
           
            if not order_line_table:
                logger.warning(f"No order_line table mapping found for client {self.client_id}")
                return []
           
            # Get order line columns
            order_line_columns = self._get_columns_for_table(mappings, order_line_table, "order_line")
           
            # Find the order column
            order_col = order_line_columns.get('order', 'order_id')
           
            # Query order items
            query = f"SELECT * FROM {order_line_table} WHERE {order_col} = :order_id"
            result = await connector.execute_query(query, {"order_id": order_id})
           
            return result["data"]
           
        except Exception as e:
            logger.error(f"Error retrieving order items: {str(e)}")
            return []
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def _create_order_items(self, order_id: str, items: List[Dict[str, Any]]) -> bool:
        """
        Create line items for an order.
       
        Args:
            order_id: Order ID
            items: List of item data
           
        Returns:
            True if created, False if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
           
            # Find order_line table
            order_line_table = self._find_table_by_concept(mappings, "order_line")
           
            if not order_line_table:
                logger.warning(f"No order_line table mapping found for client {self.client_id}")
                return False
           
            # Get order line columns
            order_line_columns = self._get_columns_for_table(mappings, order_line_table, "order_line")
           
            # Process each item
            for item in items:
                # Prepare item data
                db_columns = []
                placeholders = []
                params = {}
               
                # Add order ID
                order_col = order_line_columns.get('order', 'order_id')
                db_columns.append(order_col)
                placeholders.append(":order_id")
                params["order_id"] = order_id
               
                # Generate ID if not provided
                if "id" not in item:
                    item["id"] = str(uuid.uuid4())
               
                # Map domain attributes to database columns
                for attr, value in item.items():
                    if attr in order_line_columns:
                        db_column = order_line_columns[attr]
                        db_columns.append(db_column)
                        placeholders.append(f":{attr}")
                        params[attr] = value
               
                # Build the insert query
                query = f"""
                INSERT INTO {order_line_table} ({', '.join(db_columns)})
                VALUES ({', '.join(placeholders)})
                """
               
                # Execute insert
                await connector.execute_query(query, params)
           
            return True
           
        except Exception as e:
            logger.error(f"Error creating order items: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def _log_order_status_change(
        self,
        order_id: str,
        status: str,
        order_type: str
    ) -> bool:
        """
        Log an order status change for tracking purposes.
       
        Args:
            order_id: Order ID
            status: New status
            order_type: Type of order
           
        Returns:
            True if logged, False if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
           
            # Check if order_status_history table exists
            try:
                tables = await connector.get_tables()
                if "order_status_history" not in tables:
                    logger.info("Order status history table not found, no logging performed")
                    return True
            except Exception:
                # If get_tables fails, assume table doesn't exist
                return True
           
            # Insert history record
            insert_query = """
            INSERT INTO order_status_history (
                order_id, status, order_type, timestamp, user_id
            )
            VALUES (
                :order_id, :status, :order_type, :timestamp, :user_id
            )
            """
           
            params = {
                "order_id": order_id,
                "status": status,
                "order_type": order_type,
                "timestamp": datetime.now().isoformat(),
                "user_id": None  # In a real implementation, would come from auth context
            }
           
            await connector.execute_query(insert_query, params)
           
            return True
           
        except Exception as e:
            logger.error(f"Error logging order status change: {str(e)}")
            return False
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