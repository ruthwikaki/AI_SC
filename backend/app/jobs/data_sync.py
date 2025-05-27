# jobs/data_sync.py

"""
Data synchronization jobs for external data sources
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import pandas as pd
import httpx

from app.db.database import get_db_session
from app.core.config import settings
from app.models import (
    DatabaseConnection, SyncJob, SyncHistory,
    Supplier, Product, Material, Inventory,
    Order, OrderItem, Shipment
)
from app.db.repositories.supplier_repository import SupplierRepository
from app.db.repositories.inventory_repository import InventoryRepository
from app.db.repositories.order_repository import OrderRepository
from .scheduler import scheduled_job

logger = logging.getLogger(__name__)


class DataSyncService:
    """Service for synchronizing data from external sources"""
    
    def __init__(self, db: Session):
        self.db = db
        self.supplier_repo = SupplierRepository(db)
        self.inventory_repo = InventoryRepository(db)
        self.order_repo = OrderRepository(db)
    
    async def sync_from_database(
        self,
        connection_id: str,
        sync_type: str,
        full_sync: bool = False
    ) -> Dict[str, Any]:
        """
        Sync data from external database
        
        Args:
            connection_id: Database connection ID
            sync_type: Type of data to sync
            full_sync: Whether to perform full sync or incremental
            
        Returns:
            Sync results
        """
        # Get database connection
        connection = self.db.query(DatabaseConnection).filter(
            DatabaseConnection.id == connection_id
        ).first()
        
        if not connection:
            raise ValueError(f"Database connection {connection_id} not found")
        
        # Create sync job
        sync_job = SyncJob(
            connection_id=connection_id,
            sync_type=sync_type,
            status="running",
            started_at=datetime.utcnow(),
            parameters={"full_sync": full_sync}
        )
        self.db.add(sync_job)
        self.db.commit()
        
        try:
            # Connect to external database
            import asyncpg
            
            conn = await asyncpg.connect(
                host=connection.connection_config.get("host"),
                port=connection.connection_config.get("port", 5432),
                user=connection.connection_config.get("user"),
                password=connection.connection_config.get("password"),
                database=connection.connection_config.get("database")
            )
            
            # Get last sync timestamp
            last_sync = None
            if not full_sync:
                last_history = self.db.query(SyncHistory).filter(
                    and_(
                        SyncHistory.connection_id == connection_id,
                        SyncHistory.sync_type == sync_type,
                        SyncHistory.status == "success"
                    )
                ).order_by(SyncHistory.completed_at.desc()).first()
                
                if last_history:
                    last_sync = last_history.completed_at
            
            # Sync based on type
            if sync_type == "suppliers":
                result = await self._sync_suppliers(conn, last_sync)
            elif sync_type == "products":
                result = await self._sync_products(conn, last_sync)
            elif sync_type == "inventory":
                result = await self._sync_inventory(conn, last_sync)
            elif sync_type == "orders":
                result = await self._sync_orders(conn, last_sync)
            else:
                raise ValueError(f"Unknown sync type: {sync_type}")
            
            # Update sync job
            sync_job.status = "completed"
            sync_job.completed_at = datetime.utcnow()
            sync_job.records_synced = result.get("records_synced", 0)
            sync_job.errors = result.get("errors", [])
            
            # Create sync history
            sync_history = SyncHistory(
                job_id=sync_job.id,
                connection_id=connection_id,
                sync_type=sync_type,
                status="success",
                records_synced=result.get("records_synced", 0),
                started_at=sync_job.started_at,
                completed_at=datetime.utcnow(),
                details=result
            )
            self.db.add(sync_history)
            
            await conn.close()
            self.db.commit()
            
            return result
            
        except Exception as e:
            # Update sync job with error
            sync_job.status = "failed"
            sync_job.completed_at = datetime.utcnow()
            sync_job.errors = [str(e)]
            
            # Create failed sync history
            sync_history = SyncHistory(
                job_id=sync_job.id,
                connection_id=connection_id,
                sync_type=sync_type,
                status="failed",
                records_synced=0,
                started_at=sync_job.started_at,
                completed_at=datetime.utcnow(),
                error_message=str(e)
            )
            self.db.add(sync_history)
            self.db.commit()
            
            logger.error(f"Sync failed: {str(e)}")
            raise
    
    async def _sync_suppliers(
        self,
        conn,
        last_sync: Optional[datetime]
    ) -> Dict[str, Any]:
        """Sync supplier data"""
        query = """
            SELECT 
                supplier_code, name, category, country, city,
                contact_email, contact_phone, status,
                updated_at
            FROM suppliers
        """
        
        if last_sync:
            query += " WHERE updated_at > $1"
            rows = await conn.fetch(query, last_sync)
        else:
            rows = await conn.fetch(query)
        
        records_synced = 0
        errors = []
        
        for row in rows:
            try:
                # Check if supplier exists
                supplier = self.db.query(Supplier).filter(
                    Supplier.code == row["supplier_code"]
                ).first()
                
                if supplier:
                    # Update existing
                    supplier.name = row["name"]
                    supplier.category = row["category"]
                    supplier.country = row["country"]
                    supplier.city = row["city"]
                    supplier.contact_email = row["contact_email"]
                    supplier.contact_phone = row["contact_phone"]
                    supplier.status = row["status"]
                else:
                    # Create new
                    supplier = Supplier(
                        code=row["supplier_code"],
                        name=row["name"],
                        category=row["category"],
                        country=row["country"],
                        city=row["city"],
                        contact_email=row["contact_email"],
                        contact_phone=row["contact_phone"],
                        status=row["status"]
                    )
                    self.db.add(supplier)
                
                records_synced += 1
                
            except Exception as e:
                errors.append({
                    "supplier_code": row["supplier_code"],
                    "error": str(e)
                })
        
        self.db.commit()
        
        return {
            "records_synced": records_synced,
            "errors": errors,
            "sync_type": "suppliers"
        }
    
    async def _sync_products(
        self,
        conn,
        last_sync: Optional[datetime]
    ) -> Dict[str, Any]:
        """Sync product data"""
        query = """
            SELECT 
                sku, name, description, category, unit_of_measure,
                unit_cost, selling_price, status, updated_at
            FROM products
        """
        
        if last_sync:
            query += " WHERE updated_at > $1"
            rows = await conn.fetch(query, last_sync)
        else:
            rows = await conn.fetch(query)
        
        records_synced = 0
        errors = []
        
        for row in rows:
            try:
                product = self.db.query(Product).filter(
                    Product.sku == row["sku"]
                ).first()
                
                if product:
                    product.name = row["name"]
                    product.description = row["description"]
                    product.category = row["category"]
                    product.unit_of_measure = row["unit_of_measure"]
                    product.unit_cost = Decimal(str(row["unit_cost"]))
                    product.selling_price = Decimal(str(row["selling_price"]))
                    product.status = row["status"]
                else:
                    product = Product(
                        sku=row["sku"],
                        name=row["name"],
                        description=row["description"],
                        category=row["category"],
                        unit_of_measure=row["unit_of_measure"],
                        unit_cost=Decimal(str(row["unit_cost"])),
                        selling_price=Decimal(str(row["selling_price"])),
                        status=row["status"]
                    )
                    self.db.add(product)
                
                records_synced += 1
                
            except Exception as e:
                errors.append({
                    "sku": row["sku"],
                    "error": str(e)
                })
        
        self.db.commit()
        
        return {
            "records_synced": records_synced,
            "errors": errors,
            "sync_type": "products"
        }
    
    async def _sync_inventory(
        self,
        conn,
        last_sync: Optional[datetime]
    ) -> Dict[str, Any]:
        """Sync inventory data"""
        query = """
            SELECT 
                location_code, product_sku, quantity_on_hand,
                quantity_allocated, reorder_point, safety_stock,
                last_counted_date, updated_at
            FROM inventory_levels
        """
        
        if last_sync:
            query += " WHERE updated_at > $1"
            rows = await conn.fetch(query, last_sync)
        else:
            rows = await conn.fetch(query)
        
        records_synced = 0
        errors = []
        
        for row in rows:
            try:
                # Find product
                product = self.db.query(Product).filter(
                    Product.sku == row["product_sku"]
                ).first()
                
                if not product:
                    errors.append({
                        "location": row["location_code"],
                        "sku": row["product_sku"],
                        "error": "Product not found"
                    })
                    continue
                
                # Update or create inventory
                inventory = self.db.query(Inventory).filter(
                    and_(
                        Inventory.location_id == row["location_code"],
                        Inventory.product_id == product.id
                    )
                ).first()
                
                if inventory:
                    inventory.quantity_on_hand = Decimal(str(row["quantity_on_hand"]))
                    inventory.quantity_allocated = Decimal(str(row["quantity_allocated"]))
                    inventory.reorder_point = Decimal(str(row["reorder_point"]))
                    inventory.safety_stock = Decimal(str(row["safety_stock"]))
                    inventory.last_counted_date = row["last_counted_date"]
                else:
                    inventory = Inventory(
                        location_id=row["location_code"],
                        product_id=product.id,
                        quantity_on_hand=Decimal(str(row["quantity_on_hand"])),
                        quantity_allocated=Decimal(str(row["quantity_allocated"])),
                        reorder_point=Decimal(str(row["reorder_point"])),
                        safety_stock=Decimal(str(row["safety_stock"])),
                        last_counted_date=row["last_counted_date"]
                    )
                    self.db.add(inventory)
                
                records_synced += 1
                
            except Exception as e:
                errors.append({
                    "location": row["location_code"],
                    "sku": row["product_sku"],
                    "error": str(e)
                })
        
        self.db.commit()
        
        return {
            "records_synced": records_synced,
            "errors": errors,
            "sync_type": "inventory"
        }
    
    async def _sync_orders(
        self,
        conn,
        last_sync: Optional[datetime]
    ) -> Dict[str, Any]:
        """Sync order data"""
        # Sync orders
        order_query = """
            SELECT 
                order_number, order_type, supplier_code,
                order_date, status, total_amount,
                updated_at
            FROM orders
        """
        
        if last_sync:
            order_query += " WHERE updated_at > $1"
            order_rows = await conn.fetch(order_query, last_sync)
        else:
            order_rows = await conn.fetch(order_query)
        
        records_synced = 0
        errors = []
        
        for row in order_rows:
            try:
                # Find supplier if applicable
                supplier = None
                if row["supplier_code"]:
                    supplier = self.db.query(Supplier).filter(
                        Supplier.code == row["supplier_code"]
                    ).first()
                
                # Update or create order
                order = self.db.query(Order).filter(
                    Order.order_number == row["order_number"]
                ).first()
                
                if order:
                    order.status = row["status"]
                    order.total_amount = Decimal(str(row["total_amount"]))
                else:
                    order = Order(
                        order_number=row["order_number"],
                        type=row["order_type"],
                        supplier_id=supplier.id if supplier else None,
                        order_date=row["order_date"],
                        status=row["status"],
                        total_amount=Decimal(str(row["total_amount"])),
                        created_by_id=None  # System created
                    )
                    self.db.add(order)
                    self.db.flush()
                
                # Sync order items
                item_query = """
                    SELECT 
                        line_number, product_sku, quantity_ordered,
                        unit_price, quantity_shipped
                    FROM order_items
                    WHERE order_number = $1
                """
                
                item_rows = await conn.fetch(item_query, row["order_number"])
                
                for item_row in item_rows:
                    product = self.db.query(Product).filter(
                        Product.sku == item_row["product_sku"]
                    ).first()
                    
                    if product:
                        order_item = self.db.query(OrderItem).filter(
                            and_(
                                OrderItem.order_id == order.id,
                                OrderItem.line_number == item_row["line_number"]
                            )
                        ).first()
                        
                        if not order_item:
                            order_item = OrderItem(
                                order_id=order.id,
                                line_number=item_row["line_number"],
                                product_id=product.id,
                                quantity_ordered=Decimal(str(item_row["quantity_ordered"])),
                                unit_price=Decimal(str(item_row["unit_price"])),
                                quantity_shipped=Decimal(str(item_row["quantity_shipped"]))
                            )
                            self.db.add(order_item)
                
                records_synced += 1
                
            except Exception as e:
                errors.append({
                    "order_number": row["order_number"],
                    "error": str(e)
                })
        
        self.db.commit()
        
        return {
            "records_synced": records_synced,
            "errors": errors,
            "sync_type": "orders"
        }
    
    async def sync_from_api(
        self,
        api_endpoint: str,
        api_key: str,
        sync_type: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Sync data from external API"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = await client.get(
                api_endpoint,
                headers=headers,
                params=params or {}
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}")
            
            data = response.json()
            
            # Process based on sync type
            if sync_type == "suppliers":
                return await self._process_supplier_api_data(data)
            elif sync_type == "inventory":
                return await self._process_inventory_api_data(data)
            else:
                raise ValueError(f"Unknown API sync type: {sync_type}")
    
    async def _process_supplier_api_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process supplier data from API"""
        records_synced = 0
        errors = []
        
        for item in data:
            try:
                supplier = self.db.query(Supplier).filter(
                    Supplier.code == item["code"]
                ).first()
                
                if supplier:
                    # Update fields
                    for key, value in item.items():
                        if hasattr(supplier, key):
                            setattr(supplier, key, value)
                else:
                    supplier = Supplier(**item)
                    self.db.add(supplier)
                
                records_synced += 1
                
            except Exception as e:
                errors.append({
                    "code": item.get("code"),
                    "error": str(e)
                })
        
        self.db.commit()
        
        return {
            "records_synced": records_synced,
            "errors": errors
        }
    
    async def _process_inventory_api_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process inventory data from API"""
        records_synced = 0
        errors = []
        
        for item in data:
            try:
                # Implementation similar to _process_supplier_api_data
                records_synced += 1
            except Exception as e:
                errors.append({"error": str(e)})
        
        return {
            "records_synced": records_synced,
            "errors": errors
        }


# Job functions
@scheduled_job(name="sync_supplier_data", description="Synchronize supplier data from external sources")
async def sync_supplier_data():
    """Sync supplier data from all configured sources"""
    with get_db_session() as db:
        service = DataSyncService(db)
        
        # Get active database connections
        connections = db.query(DatabaseConnection).filter(
            and_(
                DatabaseConnection.is_active == True,
                DatabaseConnection.sync_config.has_key("suppliers")
            )
        ).all()
        
        results = []
        
        for connection in connections:
            try:
                result = await service.sync_from_database(
                    connection_id=str(connection.id),
                    sync_type="suppliers",
                    full_sync=False
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync suppliers from {connection.name}: {str(e)}")
        
        return results


@scheduled_job(name="sync_inventory_data", description="Synchronize inventory data from external sources")
async def sync_inventory_data():
    """Sync inventory data from all configured sources"""
    with get_db_session() as db:
        service = DataSyncService(db)
        
        connections = db.query(DatabaseConnection).filter(
            and_(
                DatabaseConnection.is_active == True,
                DatabaseConnection.sync_config.has_key("inventory")
            )
        ).all()
        
        results = []
        
        for connection in connections:
            try:
                result = await service.sync_from_database(
                    connection_id=str(connection.id),
                    sync_type="inventory",
                    full_sync=False
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync inventory from {connection.name}: {str(e)}")
        
        return results


@scheduled_job(name="sync_order_data", description="Synchronize order data from external sources")
async def sync_order_data():
    """Sync order data from all configured sources"""
    with get_db_session() as db:
        service = DataSyncService(db)
        
        connections = db.query(DatabaseConnection).filter(
            and_(
                DatabaseConnection.is_active == True,
                DatabaseConnection.sync_config.has_key("orders")
            )
        ).all()
        
        results = []
        
        for connection in connections:
            try:
                result = await service.sync_from_database(
                    connection_id=str(connection.id),
                    sync_type="orders",
                    full_sync=False
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync orders from {connection.name}: {str(e)}")
        
        return results


@scheduled_job(name="run_full_sync", description="Run full data synchronization")
async def run_full_sync():
    """Run full synchronization for all data types"""
    results = {
        "suppliers": await sync_supplier_data(),
        "inventory": await sync_inventory_data(),
        "orders": await sync_order_data()
    }
    
    logger.info(f"Full sync completed: {json.dumps(results, indent=2)}")
    return results


async def check_sync_status(connection_id: str) -> Dict[str, Any]:
    """Check the status of sync jobs for a connection"""
    with get_db_session() as db:
        # Get latest sync job
        latest_job = db.query(SyncJob).filter(
            SyncJob.connection_id == connection_id
        ).order_by(SyncJob.started_at.desc()).first()
        
        # Get sync history stats
        history_stats = db.query(
            SyncHistory.sync_type,
            func.count(SyncHistory.id).label("total_syncs"),
            func.sum(SyncHistory.records_synced).label("total_records"),
            func.max(SyncHistory.completed_at).label("last_sync")
        ).filter(
            SyncHistory.connection_id == connection_id
        ).group_by(SyncHistory.sync_type).all()
        
        return {
            "latest_job": {
                "id": str(latest_job.id) if latest_job else None,
                "status": latest_job.status if latest_job else None,
                "started_at": latest_job.started_at if latest_job else None,
                "completed_at": latest_job.completed_at if latest_job else None,
                "records_synced": latest_job.records_synced if latest_job else 0
            },
            "history": [
                {
                    "sync_type": stat.sync_type,
                    "total_syncs": stat.total_syncs,
                    "total_records": stat.total_records or 0,
                    "last_sync": stat.last_sync
                }
                for stat in history_stats
            ]
        }


async def run_data_sync(sync_type: str, source_config: Dict[str, Any]):
    """Run data sync with custom configuration"""
    with get_db_session() as db:
        service = DataSyncService(db)
        
        if source_config.get("type") == "database":
            return await service.sync_from_database(
                connection_id=source_config["connection_id"],
                sync_type=sync_type,
                full_sync=source_config.get("full_sync", False)
            )
        elif source_config.get("type") == "api":
            return await service.sync_from_api(
                api_endpoint=source_config["endpoint"],
                api_key=source_config["api_key"],
                sync_type=sync_type,
                params=source_config.get("params")
            )
        else:
            raise ValueError(f"Unknown source type: {source_config.get('type')}")