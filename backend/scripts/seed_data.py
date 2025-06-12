
import os
import sys
import random
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Simple password hash function
import hashlib
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123456789@localhost:5432/Supplychain_AI")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def seed_database():
    """Seed database with test data"""
    
    logger.info("Starting database seeding...")
    
    # Test connection
    session = Session()
    try:
        result = session.execute(text("SELECT 1"))
        logger.info("âœ“ Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return
    finally:
        session.close()
    
    # Get existing IDs for foreign keys
    session = Session()
    user_ids = []
    supplier_ids = []
    product_ids = []
    try:
        # Get user IDs
        existing_users = session.execute(text("SELECT id FROM users")).fetchall()
        user_ids = [u[0] for u in existing_users]
        logger.info(f"Found {len(user_ids)} users")
        
        # Get supplier IDs
        existing_suppliers = session.execute(text("SELECT id FROM suppliers")).fetchall()
        supplier_ids = [s[0] for s in existing_suppliers]
        logger.info(f"Found {len(supplier_ids)} suppliers")
        
        # Get product IDs
        existing_products = session.execute(text("SELECT id FROM products")).fetchall()
        product_ids = [p[0] for p in existing_products]
        logger.info(f"Found {len(product_ids)} products")
    finally:
        session.close()
    
    # 1. Seed inventory (without quantity_available - it's generated)
    session = Session()
    try:
        logger.info("\nSeeding inventory...")
        inventory_count = session.execute(text("SELECT COUNT(*) FROM inventory")).scalar()
        
        if inventory_count > 0:
            logger.info(f"  Found {inventory_count} existing inventory records")
        else:
            locations = ['WH-MAIN', 'WH-EAST', 'WH-WEST', 'WH-NORTH', 'WH-SOUTH']
            created_inventory = 0
            
            # Create inventory for products
            for i, product_id in enumerate(product_ids[:30]):
                location = locations[i % len(locations)]
                quantity_on_hand = random.randint(50, 500)
                quantity_reserved = random.randint(0, min(50, quantity_on_hand))
                
                inventory_data = {
                    "id": str(uuid.uuid4()),
                    "product_id": product_id,
                    "location_code": location,
                    "quantity_on_hand": quantity_on_hand,
                    "quantity_reserved": quantity_reserved,
                    # Don't include quantity_available - it's generated
                    "reorder_point": random.randint(20, 100),
                    "reorder_quantity": random.randint(50, 200),
                    "last_counted_date": fake.date_between(start_date='-30d', end_date='today'),
                    "last_movement_date": datetime.now() - timedelta(days=random.randint(0, 7)),
                    "created_at": datetime.now()
                }
                
                session.execute(
                    text("INSERT INTO inventory (id, product_id, location_code, "
                         "quantity_on_hand, quantity_reserved, "
                         "reorder_point, reorder_quantity, last_counted_date, "
                         "last_movement_date, created_at) "
                         "VALUES (:id, :product_id, :location_code, :quantity_on_hand, "
                         ":quantity_reserved, :reorder_point, "
                         ":reorder_quantity, :last_counted_date, :last_movement_date, :created_at)"),
                    inventory_data
                )
                created_inventory += 1
            
            logger.info(f"  âœ“ Created {created_inventory} inventory records")
            session.commit()
    except Exception as e:
        logger.error(f"Error seeding inventory: {e}")
        session.rollback()
    finally:
        session.close()
    
    # 2. Seed orders (with proper actual_delivery_date handling)
    session = Session()
    order_ids = []
    try:
        logger.info("\nSeeding orders...")
        order_count = session.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        
        if order_count > 0:
            logger.info(f"  Found {order_count} existing orders")
        else:
            if user_ids and supplier_ids:
                for i in range(50):
                    order_id = str(uuid.uuid4())
                    order_number = f"ORD{datetime.now().year}{i+1:05d}"
                    order_date = fake.date_between(start_date='-180d', end_date='today')
                    status = random.choice(['pending', 'approved', 'shipped', 'delivered'])
                    
                    order_data = {
                        "id": order_id,
                        "order_number": order_number,
                        "order_type": random.choice(['purchase', 'sales']),
                        "supplier_id": random.choice(supplier_ids),
                        "status": status,
                        "order_date": order_date,
                        "requested_delivery_date": order_date + timedelta(days=random.randint(7, 30)),
                        "actual_delivery_date": None,  # Default to None
                        "total_amount": round(random.uniform(100, 10000), 2),
                        "currency": "USD",
                        "payment_status": random.choice(['pending', 'paid', 'partial']),
                        "created_by": random.choice(user_ids),
                        "created_at": datetime.now()
                    }
                    
                    # Only set actual_delivery_date for delivered orders
                    if status == 'delivered':
                        order_data['actual_delivery_date'] = order_data['requested_delivery_date'] + timedelta(days=random.randint(-2, 5))
                    
                    session.execute(
                        text("INSERT INTO orders (id, order_number, order_type, supplier_id, "
                             "status, order_date, requested_delivery_date, actual_delivery_date, "
                             "total_amount, currency, payment_status, created_by, created_at) "
                             "VALUES (:id, :order_number, :order_type, :supplier_id, :status, "
                             ":order_date, :requested_delivery_date, :actual_delivery_date, "
                             ":total_amount, :currency, :payment_status, :created_by, :created_at)"),
                        order_data
                    )
                    order_ids.append(order_id)
                
                logger.info(f"  âœ“ Created {len(order_ids)} orders")
                session.commit()
    except Exception as e:
        logger.error(f"Error seeding orders: {e}")
        session.rollback()
    finally:
        session.close()
    
    # 3. Seed order items
    session = Session()
    try:
        if order_ids and product_ids:
            logger.info("\nSeeding order items...")
            created_items = 0
            
            for order_id in order_ids:
                num_items = random.randint(1, 5)
                for line_num in range(1, num_items + 1):
                    quantity = random.randint(1, 100)
                    unit_price = round(random.uniform(10, 500), 2)
                    discount = round(random.uniform(0, 10), 1)
                    tax = 8.5
                    line_total = quantity * unit_price * (1 - discount/100) * (1 + tax/100)
                    
                    item_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": order_id,
                        "line_number": line_num,
                        "product_id": random.choice(product_ids),
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_percent": discount,
                        "tax_percent": tax,
                        "line_total": round(line_total, 2),
                        "created_at": datetime.now()
                    }
                    
                    session.execute(
                        text("INSERT INTO order_items (id, order_id, line_number, product_id, "
                             "quantity, unit_price, discount_percent, tax_percent, "
                             "line_total, created_at) "
                             "VALUES (:id, :order_id, :line_number, :product_id, "
                             ":quantity, :unit_price, :discount_percent, :tax_percent, "
                             ":line_total, :created_at)"),
                        item_data
                    )
                    created_items += 1
            
            logger.info(f"  âœ“ Created {created_items} order items")
            session.commit()
    except Exception as e:
        logger.error(f"Error seeding order items: {e}")
        session.rollback()
    finally:
        session.close()
    
    # Final summary
    session = Session()
    try:
        logger.info("\n" + "="*60)
        logger.info("âœ… DATABASE SEEDING COMPLETED!")
        logger.info("="*60)
        
        total_users = session.execute(text("SELECT COUNT(*) FROM users")).scalar()
        total_suppliers = session.execute(text("SELECT COUNT(*) FROM suppliers")).scalar()
        total_products = session.execute(text("SELECT COUNT(*) FROM products")).scalar()
        total_inventory = session.execute(text("SELECT COUNT(*) FROM inventory")).scalar()
        total_orders = session.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        total_order_items = session.execute(text("SELECT COUNT(*) FROM order_items")).scalar()
        total_queries = session.execute(text("SELECT COUNT(*) FROM natural_language_queries")).scalar()
        
        logger.info("\nDatabase Summary:")
        logger.info(f"  â€¢ Users: {total_users}")
        logger.info(f"  â€¢ Suppliers: {total_suppliers}")
        logger.info(f"  â€¢ Products: {total_products}")
        logger.info(f"  â€¢ Inventory: {total_inventory} records")
        logger.info(f"  â€¢ Orders: {total_orders}")
        logger.info(f"  â€¢ Order Items: {total_order_items}")
        logger.info(f"  â€¢ Natural Language Queries: {total_queries}")
        
        logger.info("\nðŸŽ‰ Your Supply Chain AI database is ready for testing!")
        logger.info("\nNext steps:")
        logger.info("1. Go back to backend directory: cd ..")
        logger.info("2. Start the server: python main.py")
        logger.info("3. Test the API at: http://localhost:8000/docs")
        logger.info("\nExample test query:")
        logger.info('curl -X POST "http://localhost:8000/api/queries/natural-language" \\')
        logger.info('  -H "Content-Type: application/json" \\')
        logger.info('  -d \'{"query": "Show me products with low inventory"}\'')
        
    except Exception as e:
        logger.error(f"Error getting final counts: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()