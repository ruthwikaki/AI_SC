# scripts/seed_data.py

"""
Seed database with test data for development and testing
"""

import os
import sys
import random
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faker import Faker
from sqlalchemy.orm import Session

from app.db.database import get_db_session
from app.models import (
    User, Role, Permission, Supplier, SupplierTier,
    Product, Material, Inventory, Order, OrderItem,
    Shipment, ShipmentItem, ChartType, Dashboard,
    NaturalLanguageQuery, QueryTemplate
)
from app.security.password_utils import hash_password

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()


class DataSeeder:
    """Seed database with test data"""
    
    def __init__(self, db: Session):
        self.db = db
        self.users = []
        self.suppliers = []
        self.products = []
        self.materials = []
        self.orders = []
    
    def seed_all(self):
        """Seed all data"""
        logger.info("Starting data seeding...")
        
        self.seed_users()
        self.seed_suppliers()
        self.seed_products_materials()
        self.seed_inventory()
        self.seed_orders()
        self.seed_shipments()
        self.seed_queries()
        self.seed_dashboards()
        
        logger.info("Data seeding completed!")
    
    def seed_users(self, count: int = 10):
        """Seed user data"""
        logger.info(f"Seeding {count} users...")
        
        # Get roles
        roles = self.db.query(Role).all()
        
        for i in range(count):
            user = User(
                email=fake.email(),
                password_hash=hash_password("password123"),
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                is_active=True,
                is_verified=True
            )
            
            # Assign random role
            user.roles.append(random.choice(roles))
            
            self.db.add(user)
            self.users.append(user)
        
        self.db.commit()
        logger.info(f"Created {count} users")
    
    def seed_suppliers(self, count: int = 50):
        """Seed supplier data"""
        logger.info(f"Seeding {count} suppliers...")
        
        categories = ['Electronics', 'Raw Materials', 'Components', 'Packaging', 'Logistics']
        countries = ['US', 'CN', 'DE', 'JP', 'IN', 'MX', 'CA', 'UK', 'FR', 'IT']
        
        for i in range(count):
            supplier = Supplier(
                code=f"SUP{str(i+1).zfill(5)}",
                name=fake.company(),
                category=random.choice(categories),
                status="active" if random.random() > 0.1 else "inactive",
                tier=random.randint(1, 3),
                country=random.choice(countries),
                city=fake.city(),
                address=fake.address(),
                contact_name=fake.name(),
                contact_email=fake.email(),
                contact_phone=fake.phone_number(),
                payment_terms=random.choice(['Net 30', 'Net 60', 'Net 90', 'COD']),
                lead_time_days=random.randint(1, 30),
                minimum_order_value=Decimal(str(random.randint(100, 10000))),
                rating=Decimal(str(round(random.uniform(2.5, 5.0), 1))),
                certifications=['ISO 9001'] if random.random() > 0.5 else []
            )
            
            # Add tier classification
            tier = SupplierTier(
                supplier_id=supplier.id,
                tier_level=supplier.tier,
                classification=f"Tier {supplier.tier} Supplier",
                verified=random.random() > 0.3
            )
            
            self.db.add(supplier)
            self.db.add(tier)
            self.suppliers.append(supplier)
        
        # Create some supplier relationships
        for i in range(20):
            parent = random.choice(self.suppliers[:25])  # Top tier suppliers
            child = random.choice(self.suppliers[25:])   # Lower tier suppliers
            
            if parent != child:
                relationship = {
                    "parent_supplier_id": parent.id,
                    "child_supplier_id": child.id,
                    "relationship_type": random.choice(['primary', 'backup', 'subcontractor']),
                    "strength": round(random.random(), 2)
                }
                
                # Execute raw SQL to avoid relationship conflicts
                self.db.execute(
                    "INSERT INTO supplier_relationships (parent_supplier_id, child_supplier_id, relationship_type, strength) "
                    "VALUES (:parent_supplier_id, :child_supplier_id, :relationship_type, :strength) "
                    "ON CONFLICT DO NOTHING",
                    relationship
                )
        
        self.db.commit()
        logger.info(f"Created {count} suppliers with relationships")
    
    def seed_products_materials(self, products: int = 100, materials: int = 200):
        """Seed products and materials"""
        logger.info(f"Seeding {products} products and {materials} materials...")
        
        # Materials
        material_types = ['Metal', 'Plastic', 'Electronic', 'Chemical', 'Fabric', 'Glass']
        units = ['kg', 'units', 'meters', 'liters', 'pieces']
        
        for i in range(materials):
            material = Material(
                code=f"MAT{str(i+1).zfill(5)}",
                name=f"{fake.word().capitalize()} {random.choice(material_types)}",
                description=fake.sentence(),
                type=random.choice(material_types),
                unit_of_measure=random.choice(units),
                unit_cost=Decimal(str(round(random.uniform(0.1, 100), 2))),
                lead_time_days=random.randint(1, 14),
                minimum_order_quantity=Decimal(str(random.randint(10, 1000))),
                hazmat=random.random() < 0.1,
                perishable=random.random() < 0.2
            )
            
            self.db.add(material)
            self.materials.append(material)
        
        # Products
        product_categories = ['Finished Goods', 'Sub-Assembly', 'Component', 'Accessory']
        
        for i in range(products):
            cost = Decimal(str(round(random.uniform(10, 1000), 2)))
            margin = random.uniform(0.2, 0.5)
            
            product = Product(
                sku=f"PROD{str(i+1).zfill(5)}",
                name=fake.catch_phrase(),
                description=fake.text(max_nb_chars=200),
                category=random.choice(product_categories),
                subcategory=fake.word().capitalize(),
                unit_of_measure="units",
                weight_kg=Decimal(str(round(random.uniform(0.1, 50), 3))),
                volume_m3=Decimal(str(round(random.uniform(0.001, 1), 3))),
                unit_cost=cost,
                selling_price=cost * Decimal(str(1 + margin)),
                status="active" if random.random() > 0.1 else "discontinued",
                launch_date=fake.date_between(start_date='-2y', end_date='today')
            )
            
            # Assign random supplier
            product.suppliers.append(random.choice(self.suppliers))
            
            self.db.add(product)
            self.products.append(product)
        
        # Create Bill of Materials for some products
        assembly_products = [p for p in self.products if p.category in ['Finished Goods', 'Sub-Assembly']]
        
        for product in assembly_products[:30]:
            num_materials = random.randint(2, 8)
            selected_materials = random.sample(self.materials, num_materials)
            
            for material in selected_materials:
                bom_entry = {
                    "product_id": product.id,
                    "material_id": material.id,
                    "quantity": Decimal(str(round(random.uniform(0.1, 10), 3))),
                    "unit_of_measure": material.unit_of_measure,
                    "scrap_percentage": Decimal(str(random.randint(0, 5)))
                }
                
                self.db.execute(
                    "INSERT INTO product_materials (product_id, material_id, quantity, unit_of_measure, scrap_percentage) "
                    "VALUES (:product_id, :material_id, :quantity, :unit_of_measure, :scrap_percentage)",
                    bom_entry
                )
        
        self.db.commit()
        logger.info(f"Created {products} products and {materials} materials with BOMs")
    
    def seed_inventory(self):
        """Seed inventory data"""
        logger.info("Seeding inventory data...")
        
        locations = ['WH-MAIN', 'WH-EAST', 'WH-WEST', 'WH-CENTRAL', 'STORE-001', 'STORE-002']
        
        inventory_count = 0
        
        # Create inventory for products
        for product in self.products:
            # Each product in 1-3 locations
            num_locations = random.randint(1, min(3, len(locations)))
            selected_locations = random.sample(locations, num_locations)
            
            for location in selected_locations:
                quantity = Decimal(str(random.randint(0, 1000)))
                allocated = Decimal(str(random.randint(0, min(100, int(quantity)))))
                
                inventory = Inventory(
                    location_id=location,
                    product_id=product.id,
                    quantity_on_hand=quantity,
                    quantity_allocated=allocated,
                    quantity_in_transit=Decimal(str(random.randint(0, 50))),
                    reorder_point=Decimal(str(random.randint(10, 100))),
                    reorder_quantity=Decimal(str(random.randint(50, 500))),
                    safety_stock=Decimal(str(random.randint(5, 50))),
                    lot_number=f"LOT{fake.random_number(digits=6)}",
                    cost_per_unit=product.unit_cost,
                    last_counted_date=fake.date_time_between(start_date='-30d', end_date='now')
                )
                
                self.db.add(inventory)
                inventory_count += 1
        
        # Create inventory for some materials
        for material in random.sample(self.materials, 50):
            location = random.choice(locations)
            
            inventory = Inventory(
                location_id=location,
                material_id=material.id,
                quantity_on_hand=Decimal(str(random.randint(100, 5000))),
                quantity_allocated=Decimal(str(random.randint(0, 500))),
                quantity_in_transit=Decimal(str(random.randint(0, 200))),
                reorder_point=Decimal(str(random.randint(100, 500))),
                reorder_quantity=Decimal(str(random.randint(500, 2000))),
                safety_stock=Decimal(str(random.randint(50, 200))),
                cost_per_unit=material.unit_cost
            )
            
            self.db.add(inventory)
            inventory_count += 1
        
        self.db.commit()
        logger.info(f"Created {inventory_count} inventory records")
    
    def seed_orders(self, count: int = 200):
        """Seed order data"""
        logger.info(f"Seeding {count} orders...")
        
        order_types = ['purchase', 'sales']
        statuses = ['draft', 'submitted', 'approved', 'in_progress', 'shipped', 'delivered', 'closed']
        
        for i in range(count):
            order_date = fake.date_time_between(start_date='-6M', end_date='now')
            order_type = random.choice(order_types)
            
            order = Order(
                order_number=f"ORD{datetime.now().year}{str(i+1).zfill(6)}",
                type=order_type,
                supplier_id=random.choice(self.suppliers).id if order_type == 'purchase' else None,
                customer_id=random.choice(self.users).id if order_type == 'sales' else None,
                order_date=order_date,
                requested_delivery_date=order_date + timedelta(days=random.randint(7, 30)),
                status=random.choice(statuses),
                priority=random.choice(['low', 'normal', 'high', 'urgent']),
                shipping_address=fake.address(),
                billing_address=fake.address(),
                currency='USD',
                payment_terms=random.choice(['Net 30', 'Net 60', 'COD', 'Prepaid']),
                notes=fake.sentence() if random.random() > 0.7 else None,
                created_by_id=random.choice(self.users).id
            )
            
            # Calculate actual delivery date for completed orders
            if order.status in ['delivered', 'closed']:
                # 80% on time, 20% late
                if random.random() < 0.8:
                    order.actual_delivery_date = order.requested_delivery_date - timedelta(days=random.randint(0, 2))
                else:
                    order.actual_delivery_date = order.requested_delivery_date + timedelta(days=random.randint(1, 7))
            
            self.db.add(order)
            self.db.flush()
            
            # Add order items
            num_items = random.randint(1, 10)
            total_amount = Decimal('0')
            
            for line_num in range(1, num_items + 1):
                product = random.choice(self.products)
                quantity = Decimal(str(random.randint(1, 100)))
                unit_price = product.selling_price if order_type == 'sales' else product.unit_cost
                discount = Decimal(str(random.randint(0, 15)))
                
                order_item = OrderItem(
                    order_id=order.id,
                    line_number=line_num,
                    product_id=product.id,
                    quantity_ordered=quantity,
                    unit_price=unit_price,
                    discount_percentage=discount,
                    tax_percentage=Decimal('8.5'),  # Standard tax
                    quantity_shipped=quantity if order.status in ['shipped', 'delivered', 'closed'] else Decimal('0'),
                    quantity_received=quantity if order.status in ['delivered', 'closed'] else Decimal('0')
                )
                
                line_total = quantity * unit_price * (1 - discount/100) * Decimal('1.085')
                total_amount += line_total
                
                self.db.add(order_item)
            
            # Update order totals
            order.total_amount = total_amount
            order.tax_amount = total_amount * Decimal('0.085') / Decimal('1.085')
            order.shipping_amount = Decimal(str(random.randint(10, 100)))
            order.discount_amount = total_amount * Decimal('0.05') if random.random() > 0.7 else Decimal('0')
            order.net_amount = order.total_amount + order.shipping_amount - order.discount_amount
            
            self.orders.append(order)
        
        self.db.commit()
        logger.info(f"Created {count} orders with items")
    
    def seed_shipments(self):
        """Seed shipment data"""
        logger.info("Seeding shipment data...")
        
        carriers = ['FedEx', 'UPS', 'DHL', 'USPS', 'Local Carrier']
        locations = ['WH-MAIN', 'WH-EAST', 'WH-WEST', 'WH-CENTRAL']
        
        shipment_count = 0
        
        # Create shipments for shipped orders
        shipped_orders = [o for o in self.orders if o.status in ['shipped', 'delivered', 'closed']]
        
        for order in shipped_orders:
            # Some orders might have multiple shipments
            num_shipments = 1 if random.random() > 0.2 else 2
            
            for ship_num in range(num_shipments):
                ship_date = order.order_date + timedelta(days=random.randint(1, 5))
                
                shipment = Shipment(
                    shipment_number=f"SHP{datetime.now().year}{str(shipment_count+1).zfill(6)}",
                    order_id=order.id,
                    status='delivered' if order.status in ['delivered', 'closed'] else 'in_transit',
                    carrier=random.choice(carriers),
                    tracking_number=fake.uuid4() if random.random() > 0.1 else None,
                    ship_date=ship_date,
                    estimated_delivery_date=ship_date + timedelta(days=random.randint(1, 5)),
                    actual_delivery_date=order.actual_delivery_date if hasattr(order, 'actual_delivery_date') else None,
                    from_location=random.choice(locations),
                    to_location=order.shipping_address,
                    shipping_cost=Decimal(str(random.randint(20, 200))),
                    weight_kg=Decimal(str(round(random.uniform(1, 100), 2))),
                    volume_m3=Decimal(str(round(random.uniform(0.01, 2), 3)))
                )
                
                self.db.add(shipment)
                self.db.flush()
                
                # Add shipment items
                order_items = self.db.query(OrderItem).filter(
                    OrderItem.order_id == order.id
                ).all()
                
                for item in order_items:
                    if num_shipments == 1 or (ship_num == 0 and random.random() > 0.3):
                        shipment_item = ShipmentItem(
                            shipment_id=shipment.id,
                            order_item_id=item.id,
                            quantity_shipped=item.quantity_ordered,
                            lot_number=f"LOT{fake.random_number(digits=6)}" if random.random() > 0.5 else None
                        )
                        self.db.add(shipment_item)
                
                shipment_count += 1
        
        self.db.commit()
        logger.info(f"Created {shipment_count} shipments")
    
    def seed_queries(self, count: int = 50):
        """Seed natural language queries"""
        logger.info(f"Seeding {count} queries...")
        
        query_examples = [
            "Show me top suppliers by performance",
            "What is our current inventory level for electronics?",
            "Which products are below reorder point?",
            "Display orders from last month",
            "Show me supplier risk assessment",
            "What are the bottlenecks in our supply chain?",
            "Give me ABC analysis for warehouse MAIN",
            "Show delivery performance by carrier",
            "Which suppliers have the best on-time delivery rate?",
            "What is the total inventory value?",
            "Show me products with high turnover",
            "Display safety stock recommendations",
            "Which orders are delayed?",
            "Show supplier tier distribution",
            "What are our top 10 products by revenue?"
        ]
        
        for i in range(count):
            query = NaturalLanguageQuery(
                user_id=random.choice(self.users).id,
                query_text=random.choice(query_examples) + f" ({i})",
                query_type=random.choice(['analytics', 'inventory', 'supplier', 'order']),
                is_successful=random.random() > 0.1,
                response_time_ms=random.randint(100, 2000),
                tokens_used=random.randint(100, 1000)
            )
            
            self.db.add(query)
        
        # Add query templates
        templates = [
            {
                "name": "Supplier Performance",
                "query_template": "Show supplier performance for {time_period}",
                "category": "supplier",
                "parameters": {"time_period": "last month"}
            },
            {
                "name": "Inventory Status",
                "query_template": "What is the inventory status for {location}?",
                "category": "inventory",
                "parameters": {"location": "all"}
            },
            {
                "name": "Order Summary",
                "query_template": "Show {order_type} orders from {date_range}",
                "category": "order",
                "parameters": {"order_type": "all", "date_range": "last 30 days"}
            }
        ]
        
        for template in templates:
            query_template = QueryTemplate(
                name=template["name"],
                description=f"Template for {template['name'].lower()} queries",
                query_template=template["query_template"],
                category=template["category"],
                parameters=template["parameters"],
                is_active=True,
                created_by_id=self.users[0].id
            )
            self.db.add(query_template)
        
        self.db.commit()
        logger.info(f"Created {count} queries and {len(templates)} templates")
    
    def seed_dashboards(self, count: int = 5):
        """Seed dashboard data"""
        logger.info(f"Seeding {count} dashboards...")
        
        # Ensure we have chart types
        chart_types = self.db.query(ChartType).all()
        if not chart_types:
            logger.warning("No chart types found, skipping dashboard seeding")
            return
        
        dashboard_configs = [
            {
                "name": "Supply Chain Overview",
                "description": "Main supply chain metrics dashboard",
                "layout": {
                    "grid": [
                        {"x": 0, "y": 0, "w": 6, "h": 4, "chart": "supplier_performance"},
                        {"x": 6, "y": 0, "w": 6, "h": 4, "chart": "inventory_value"},
                        {"x": 0, "y": 4, "w": 12, "h": 6, "chart": "order_trends"},
                        {"x": 0, "y": 10, "w": 6, "h": 4, "chart": "delivery_performance"},
                        {"x": 6, "y": 10, "w": 6, "h": 4, "chart": "risk_matrix"}
                    ]
                }
            },
            {
                "name": "Inventory Management",
                "description": "Inventory metrics and analysis",
                "layout": {
                    "grid": [
                        {"x": 0, "y": 0, "w": 12, "h": 6, "chart": "inventory_levels"},
                        {"x": 0, "y": 6, "w": 6, "h": 4, "chart": "abc_analysis"},
                        {"x": 6, "y": 6, "w": 6, "h": 4, "chart": "turnover_ratio"}
                    ]
                }
            },
            {
                "name": "Supplier Analytics",
                "description": "Supplier performance and risk dashboard",
                "layout": {
                    "grid": [
                        {"x": 0, "y": 0, "w": 8, "h": 6, "chart": "supplier_scorecard"},
                        {"x": 8, "y": 0, "w": 4, "h": 6, "chart": "supplier_distribution"},
                        {"x": 0, "y": 6, "w": 12, "h": 4, "chart": "supplier_trends"}
                    ]
                }
            }
        ]
        
        for i, config in enumerate(dashboard_configs[:count]):
            dashboard = Dashboard(
                name=config["name"],
                description=config["description"],
                layout_config=config["layout"],
                is_public=i == 0,  # First dashboard is public
                tags=['main'] if i == 0 else ['analytics'],
                created_by_id=random.choice(self.users).id
            )
            
            self.db.add(dashboard)
        
        self.db.commit()
        logger.info(f"Created {min(count, len(dashboard_configs))} dashboards")


def main():
    """Main entry point"""
    logger.info("Starting database seeding...")
    
    with get_db_session() as db:
        seeder = DataSeeder(db)
        
        try:
            # Check if data already exists
            user_count = db.query(User).count()
            if user_count > 1:  # More than just admin user
                response = input("Database already contains data. Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Seeding cancelled")
                    return
            
            seeder.seed_all()
            
            # Print summary
            logger.info("\n=== Seeding Summary ===")
            logger.info(f"Users: {len(seeder.users)}")
            logger.info(f"Suppliers: {len(seeder.suppliers)}")
            logger.info(f"Products: {len(seeder.products)}")
            logger.info(f"Materials: {len(seeder.materials)}")
            logger.info(f"Orders: {len(seeder.orders)}")
            logger.info(f"Total records created: Multiple thousands")
            logger.info("=====================")
            
        except Exception as e:
            logger.error(f"Error during seeding: {str(e)}")
            raise


if __name__ == "__main__":
    main()