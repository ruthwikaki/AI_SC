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

class InventoryInterface:
    """Interface for inventory-related database operations"""
    
    def __init__(self, client_id: str, connection_id: Optional[str] = None):
        """
        Initialize the inventory interface.
        
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
        """
        self.client_id = client_id
        self.connection_id = connection_id
    
    async def get_inventory_levels(
        self,
        product_ids: Optional[List[str]] = None,
        warehouse_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        below_safety_stock: bool = False,
        include_inactive: bool = False,
        page: int = 1,
        page_size: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get current inventory levels.
        
        Args:
            product_ids: Optional list of product IDs to filter by
            warehouse_ids: Optional list of warehouse IDs to filter by
            categories: Optional list of product categories to filter by
            below_safety_stock: Whether to only return items below safety stock
            include_inactive: Whether to include inactive products
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (inventory_list, total_count)
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Get domain mappings to find the relevant tables and columns
            mappings = await get_domain_mappings(self.client_id)
            
            # Find inventory, product, and warehouse tables
            inventory_table = self._find_table_by_concept(mappings, "inventory")
            product_table = self._find_table_by_concept(mappings, "product")
            warehouse_table = self._find_table_by_concept(mappings, "warehouse")
            
            if not inventory_table:
                logger.warning(f"No inventory table mapping found for client {self.client_id}")
                return [], 0
            
            # Build query based on available mappings
            inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
            
            # Build select clause
            select_clause = ["inv.*"]
            join_clause = []
            where_clause = []
            params = {}
            
            # Add product join if available
            if product_table:
                product_columns = self._get_columns_for_table(mappings, product_table, "product")
                select_clause.append(f"prod.{product_columns.get('name', 'name')} as product_name")
                
                # Find the linking column (product_id in inventory table)
                product_id_col = inventory_columns.get("product", "product_id")
                
                join_clause.append(f"""
                LEFT JOIN {product_table} prod 
                ON inv.{product_id_col} = prod.{self._get_primary_key(mappings, product_table)}
                """)
                
                # Add category filter if requested
                if categories and 'category' in product_columns:
                    placeholders = []
                    for i, category in enumerate(categories):
                        param_name = f"category_{i}"
                        placeholders.append(f":{param_name}")
                        params[param_name] = category
                    
                    category_col = product_columns['category']
                    where_clause.append(f"prod.{category_col} IN ({', '.join(placeholders)})")
                
                # Add inactive filter if requested
                if not include_inactive and 'is_active' in product_columns:
                    where_clause.append(f"prod.{product_columns['is_active']} = :is_active")
                    params["is_active"] = True
            
            # Add warehouse join if available
            if warehouse_table:
                warehouse_columns = self._get_columns_for_table(mappings, warehouse_table, "warehouse")
                select_clause.append(f"wh.{warehouse_columns.get('name', 'name')} as warehouse_name")
                
                # Find the linking column (warehouse_id in inventory table)
                warehouse_id_col = inventory_columns.get("location", "warehouse_id")
                
                join_clause.append(f"""
                LEFT JOIN {warehouse_table} wh 
                ON inv.{warehouse_id_col} = wh.{self._get_primary_key(mappings, warehouse_table)}
                """)
                
                # Add warehouse filter if requested
                if warehouse_ids:
                    placeholders = []
                    for i, wh_id in enumerate(warehouse_ids):
                        param_name = f"warehouse_{i}"
                        placeholders.append(f":{param_name}")
                        params[param_name] = wh_id
                    
                    where_clause.append(f"inv.{warehouse_id_col} IN ({', '.join(placeholders)})")
            
            # Add product ID filter if requested
            if product_ids and 'product' in inventory_columns:
                product_id_col = inventory_columns['product']
                placeholders = []
                for i, prod_id in enumerate(product_ids):
                    param_name = f"product_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = prod_id
                
                where_clause.append(f"inv.{product_id_col} IN ({', '.join(placeholders)})")
            
            # Add safety stock filter if requested
            if below_safety_stock and 'safety_stock' in inventory_columns:
                quantity_col = inventory_columns.get('quantity', 'quantity')
                safety_stock_col = inventory_columns['safety_stock']
                where_clause.append(f"inv.{quantity_col} < inv.{safety_stock_col}")
            
            # Build the final query
            query = f"SELECT {', '.join(select_clause)} FROM {inventory_table} inv"
            
            if join_clause:
                query += " " + " ".join(join_clause)
            
            if where_clause:
                query += " WHERE " + " AND ".join(where_clause)
            
            # Add count query for pagination
            count_query = f"SELECT COUNT(*) as total FROM {inventory_table} inv"
            if join_clause:
                count_query += " " + " ".join(join_clause)
            if where_clause:
                count_query += " WHERE " + " AND ".join(where_clause)
            
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
            logger.error(f"Error retrieving inventory levels: {str(e)}")
            return [], 0
        finally:
            if locals().get("connector"):
                await connector.close()
    
    async def get_inventory_item(
        self,
        product_id: str,
        warehouse_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific inventory item.
        
        Args:
            product_id: Product ID
            warehouse_id: Optional warehouse ID (if not provided, returns all warehouses)
            
        Returns:
            Inventory item data or None if not found
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
            
            # Find inventory table
            inventory_table = self._find_table_by_concept(mappings, "inventory")
            
            if not inventory_table:
                logger.warning(f"No inventory table mapping found for client {self.client_id}")
                return None
            
            # Get columns
            inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
            
            # Build query
            product_id_col = inventory_columns.get("product", "product_id")
            
            query = f"SELECT * FROM {inventory_table} WHERE {product_id_col} = :product_id"
            params = {"product_id": product_id}
            
            if warehouse_id:
                warehouse_id_col = inventory_columns.get("location", "warehouse_id")
                query += f" AND {warehouse_id_col} = :warehouse_id"
                params["warehouse_id"] = warehouse_id
            
            # Execute query
            result = await connector.execute_query(query, params)
            
            # Return first item if found
            if result["data"]:
                return result["data"][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving inventory item: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
    
    async def update_inventory_quantity(
        self,
        product_id: str,
        warehouse_id: str,
        quantity: int,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update inventory quantity.
        
        Args:
            product_id: Product ID
            warehouse_id: Warehouse ID
            quantity: New quantity (absolute value, not adjustment)
            reason: Optional reason for the update
            
        Returns:
            True if updated, False if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
            
            # Find inventory table
            inventory_table = self._find_table_by_concept(mappings, "inventory")
            
            if not inventory_table:
                logger.warning(f"No inventory table mapping found for client {self.client_id}")
                return False
            
            # Get columns
            inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
            
            # Get column names
            product_id_col = inventory_columns.get("product", "product_id")
            warehouse_id_col = inventory_columns.get("location", "warehouse_id")
            quantity_col = inventory_columns.get("quantity", "quantity")
            updated_at_col = "updated_at"  # Common column name for tracking updates
            
            # Check if item exists
            check_query = f"""
            SELECT {quantity_col} FROM {inventory_table}
            WHERE {product_id_col} = :product_id AND {warehouse_id_col} = :warehouse_id
            """
            
            check_params = {
                "product_id": product_id,
                "warehouse_id": warehouse_id
            }
            
            check_result = await connector.execute_query(check_query, check_params)
            
            if check_result["data"]:
                # Update existing record
                update_query = f"""
                UPDATE {inventory_table}
                SET {quantity_col} = :quantity
                """
                
                # Add updated_at column if it exists in the table
                columns_result = await connector.get_table_schema(inventory_table)
                column_names = [col["name"].lower() for col in columns_result.get("columns", [])]
                
                if updated_at_col.lower() in column_names:
                    update_query += f", {updated_at_col} = :updated_at"
                
                # Complete the query
                update_query += f" WHERE {product_id_col} = :product_id AND {warehouse_id_col} = :warehouse_id"
                
                update_params = {
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "quantity": quantity,
                    "updated_at": datetime.now().isoformat()
                }
                
                await connector.execute_query(update_query, update_params)
                
            else:
                # Insert new record
                insert_query = f"""
                INSERT INTO {inventory_table} (
                    {product_id_col}, {warehouse_id_col}, {quantity_col}
                )
                VALUES (
                    :product_id, :warehouse_id, :quantity
                )
                """
                
                insert_params = {
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "quantity": quantity
                }
                
                await connector.execute_query(insert_query, insert_params)
            
            # Log inventory change if reason provided
            if reason:
                await self._log_inventory_change(
                    product_id=product_id,
                    warehouse_id=warehouse_id,
                    quantity=quantity,
                    reason=reason
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating inventory quantity: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
    
    async def adjust_inventory_quantity(
        self,
        product_id: str,
        warehouse_id: str,
        adjustment: int,
        reason: Optional[str] = None
    ) -> bool:
        """
        Adjust inventory quantity by a relative amount.
        
        Args:
            product_id: Product ID
            warehouse_id: Warehouse ID
            adjustment: Amount to adjust by (positive or negative)
            reason: Optional reason for the adjustment
            
        Returns:
            True if adjusted, False if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
            
            # Find inventory table
            inventory_table = self._find_table_by_concept(mappings, "inventory")
            
            if not inventory_table:
                logger.warning(f"No inventory table mapping found for client {self.client_id}")
                return False
            
            # Get columns
            inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
            
            # Get column names
            product_id_col = inventory_columns.get("product", "product_id")
            warehouse_id_col = inventory_columns.get("location", "warehouse_id")
            quantity_col = inventory_columns.get("quantity", "quantity")
            updated_at_col = "updated_at"  # Common column name for tracking updates
            
            # Check if item exists
            check_query = f"""
            SELECT {quantity_col} FROM {inventory_table}
            WHERE {product_id_col} = :product_id AND {warehouse_id_col} = :warehouse_id
            """
            
            check_params = {
                "product_id": product_id,
                "warehouse_id": warehouse_id
            }
            
            check_result = await connector.execute_query(check_query, check_params)
            
            if check_result["data"]:
                # Get current quantity
                current_quantity = check_result["data"][0][quantity_col]
                
                # Calculate new quantity
                new_quantity = current_quantity + adjustment
                
                # Ensure non-negative
                if new_quantity < 0:
                    new_quantity = 0
                
                # Update existing record
                update_query = f"""
                UPDATE {inventory_table}
                SET {quantity_col} = :quantity
                """
                
                # Add updated_at column if it exists in the table
                columns_result = await connector.get_table_schema(inventory_table)
                column_names = [col["name"].lower() for col in columns_result.get("columns", [])]
                
                if updated_at_col.lower() in column_names:
                    update_query += f", {updated_at_col} = :updated_at"
                
                # Complete the query
                update_query += f" WHERE {product_id_col} = :product_id AND {warehouse_id_col} = :warehouse_id"
                
                update_params = {
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "quantity": new_quantity,
                    "updated_at": datetime.now().isoformat()
                }
                
                await connector.execute_query(update_query, update_params)
                
            else:
                # Only insert if adjustment is positive
                if adjustment <= 0:
                    return True
                
                # Insert new record
                insert_query = f"""
                INSERT INTO {inventory_table} (
                    {product_id_col}, {warehouse_id_col}, {quantity_col}
                )
                VALUES (
                    :product_id, :warehouse_id, :quantity
                )
                """
                
                insert_params = {
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "quantity": adjustment
                }
                
                await connector.execute_query(insert_query, insert_params)
            
            # Log inventory change if reason provided
            if reason:
                await self._log_inventory_change(
                    product_id=product_id,
                    warehouse_id=warehouse_id,
                    quantity_change=adjustment,
                    reason=reason
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting inventory quantity: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
    
    async def get_low_stock_items(
        self, 
        threshold: Optional[float] = None,
        warehouse_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get items with inventory below threshold or safety stock.
        
        Args:
            threshold: Optional manual threshold (if not provided, uses safety stock)
            warehouse_ids: Optional list of warehouse IDs to filter by
            
        Returns:
            List of low stock items
        """
        try:
            # Get inventory levels with below safety stock flag
            items, _ = await self.get_inventory_levels(
                warehouse_ids=warehouse_ids,
                below_safety_stock=(threshold is None),
                page=1,
                page_size=1000  # Larger limit for low stock items
            )
            
            # If using manual threshold, filter the results
            if threshold is not None and items:
                # Get domain mappings
                mappings = await get_domain_mappings(self.client_id)
                
                # Find inventory table
                inventory_table = self._find_table_by_concept(mappings, "inventory")
                
                if inventory_table:
                    # Get columns
                    inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
                    quantity_col = inventory_columns.get("quantity", "quantity")
                    
                    # Filter items below threshold
                    items = [item for item in items if item.get(quantity_col, 0) < threshold]
            
            return items
            
        except Exception as e:
            logger.error(f"Error retrieving low stock items: {str(e)}")
            return []
    
    async def get_inventory_value(
        self,
        warehouse_ids: Optional[List[str]] = None,
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate total inventory value.
        
        Args:
            warehouse_ids: Optional list of warehouse IDs to filter by
            as_of_date: Optional date to calculate value as of
            
        Returns:
            Dictionary with total value and breakdown
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Get domain mappings
            mappings = await get_domain_mappings(self.client_id)
            
            # Find inventory, product tables
            inventory_table = self._find_table_by_concept(mappings, "inventory")
            product_table = self._find_table_by_concept(mappings, "product")
            
            if not inventory_table or not product_table:
                logger.warning(f"Missing table mappings for client {self.client_id}")
                return {"total_value": 0, "breakdown": []}
            
            # Get columns
            inventory_columns = self._get_columns_for_table(mappings, inventory_table, "inventory")
            product_columns = self._get_columns_for_table(mappings, product_table, "product")
            
            # Get column names
            product_id_col = inventory_columns.get("product", "product_id")
            quantity_col = inventory_columns.get("quantity", "quantity")
            warehouse_id_col = inventory_columns.get("location", "warehouse_id")
            
            product_cost_col = product_columns.get("cost", "cost")
            product_name_col = product_columns.get("name", "name")
            
            # Build query
            query = f"""
            SELECT 
                p.{product_name_col} as product_name,
                i.{product_id_col} as product_id,
                i.{warehouse_id_col} as warehouse_id,
                i.{quantity_col} as quantity,
                p.{product_cost_col} as unit_cost,
                (i.{quantity_col} * p.{product_cost_col}) as total_value
            FROM {inventory_table} i
            JOIN {product_table} p ON i.{product_id_col} = p.id
            """
            
            params = {}
            where_clauses = []
            
            # Add warehouse filter if requested
            if warehouse_ids:
                placeholders = []
                for i, wh_id in enumerate(warehouse_ids):
                    param_name = f"warehouse_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = wh_id
                
                where_clauses.append(f"i.{warehouse_id_col} IN ({', '.join(placeholders)})")
            
            # Add as_of_date filter if provided
            if as_of_date:
                # This would require historical inventory tracking
                # For now, we'll just show current inventory
                pass
            
            # Add where clause if needed
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Execute query
            result = await connector.execute_query(query, params)
            
            # Calculate total value
            total_value = sum(item.get("total_value", 0) for item in result["data"])
            
            return {
                "total_value": total_value,
                "breakdown": result["data"],
                "as_of_date": as_of_date.isoformat() if as_of_date else datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating inventory value: {str(e)}")
            return {"total_value": 0, "breakdown": []}
        finally:
            if locals().get("connector"):
                await connector.close()
    
    async def _log_inventory_change(
        self,
        product_id: str,
        warehouse_id: str,
        quantity_change: Optional[int] = None,
        quantity: Optional[int] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Log an inventory change for tracking purposes.
        
        Args:
            product_id: Product ID
            warehouse_id: Warehouse ID
            quantity_change: Optional amount changed (if adjustment)
            quantity: Optional new quantity (if set directly)
            reason: Optional reason for the change
            
        Returns:
            True if logged, False if error
        """
        try:
            # Get connector
            connector = await get_connector_for_client(self.client_id, self.connection_id)
            
            # Check if inventory_changes table exists
            try:
                tables = await connector.get_tables()
                if "inventory_changes" not in tables:
                    logger.info("Inventory changes table not found, no logging performed")
                    return True
            except Exception:
                # If get_tables fails, assume table doesn't exist
                return True
            
            # Insert change record
            insert_query = """
            INSERT INTO inventory_changes (
                product_id, warehouse_id, quantity_change, new_quantity, reason, timestamp, user_id
            )
            VALUES (
                :product_id, :warehouse_id, :quantity_change, :new_quantity, :reason, :timestamp, :user_id
            )
            """
            
            params = {
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "quantity_change": quantity_change,
                "new_quantity": quantity,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "user_id": None  # In a real implementation, would come from auth context
            }
            
            await connector.execute_query(insert_query, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging inventory change: {str(e)}")
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