"""
Database initialization script
Creates tables, indexes, and loads initial data
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
from uuid import uuid4

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine, SessionLocal, Base, DatabaseManager
from app.models import (
    User, Role, Permission, ChartType, 
    Supplier, Product, Material, Inventory
)
from app.security.password_utils import hash_password
from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

# Initial data configurations
INITIAL_ROLES = [
    {
        "name": "admin",
        "display_name": "Administrator",
        "description": "Full system access",
        "is_system": True
    },
    {
        "name": "analyst",
        "display_name": "Data Analyst",
        "description": "Can create queries and analytics",
        "is_system": True
    },
    {
        "name": "viewer",
        "display_name": "Viewer",
        "description": "Read-only access",
        "is_system": True
    },
    {
        "name": "user",
        "display_name": "Standard User",
        "description": "Basic user access",
        "is_system": True
    }
]

INITIAL_PERMISSIONS = [
    # Query permissions
    {"resource": "queries", "action": "create", "display_name": "Create queries"},
    {"resource": "queries", "action": "read", "display_name": "View queries"},
    {"resource": "queries", "action": "update", "display_name": "Update queries"},
    {"resource": "queries", "action": "delete", "display_name": "Delete queries"},
    {"resource": "queries", "action": "execute", "display_name": "Execute queries"},
    
    # Analytics permissions
    {"resource": "analytics", "action": "execute", "display_name": "Execute analytics"},
    {"resource": "analytics", "action": "read", "display_name": "View analytics"},
    {"resource": "analytics", "action": "schedule", "display_name": "Schedule analytics"},
    
    # Visualization permissions
    {"resource": "visualizations", "action": "create", "display_name": "Create visualizations"},
    {"resource": "visualizations", "action": "read", "display_name": "View visualizations"},
    {"resource": "visualizations", "action": "update", "display_name": "Update visualizations"},
    {"resource": "visualizations", "action": "delete", "display_name": "Delete visualizations"},
    
    # Dashboard permissions
    {"resource": "dashboards", "action": "create", "display_name": "Create dashboards"},
    {"resource": "dashboards", "action": "read", "display_name": "View dashboards"},
    {"resource": "dashboards", "action": "update", "display_name": "Update dashboards"},
    {"resource": "dashboards", "action": "delete", "display_name": "Delete dashboards"},
    
    # Supply chain permissions
    {"resource": "suppliers", "action": "manage", "display_name": "Manage suppliers"},
    {"resource": "inventory", "action": "manage", "display_name": "Manage inventory"},
    {"resource": "orders", "action": "manage", "display_name": "Manage orders"},
    
    # System permissions
    {"resource": "settings", "action": "manage", "display_name": "Manage settings"},
    {"resource": "users", "action": "manage", "display_name": "Manage users"},
    {"resource": "database", "action": "manage", "display_name": "Manage database connections"}
]

ROLE_PERMISSIONS_MAPPING = {
    "admin": ["*"],  # All permissions
    "analyst": [
        "queries:*", "analytics:*", "visualizations:*", "dashboards:*",
        "suppliers:read", "inventory:read", "orders:read"
    ],
    "user": [
        "queries:create", "queries:read", "analytics:read", 
        "visualizations:read", "dashboards:read",
        "suppliers:read", "inventory:read", "orders:read"
    ],
    "viewer": [
        "*:read"  # All read permissions
    ]
}

INITIAL_CHART_TYPES = [
    {
        "name": "bar",
        "display_name": "Bar Chart",
        "component_name": "BarChart",
        "description": "Display data as vertical or horizontal bars",
        "supported_data_types": ["categorical", "numeric"],
        "min_data_points": 1,
        "max_data_points": 100,
        "default_config": {"orientation": "vertical", "showValues": True}
    },
    {
        "name": "line",
        "display_name": "Line Chart",
        "component_name": "LineChart",
        "description": "Show trends over time or continuous data",
        "supported_data_types": ["time-series", "numeric"],
        "min_data_points": 2,
        "max_data_points": 1000,
        "default_config": {"interpolation": "linear", "showPoints": True}
    },
    {
        "name": "pie",
        "display_name": "Pie Chart",
        "component_name": "PieChart",
        "description": "Display proportions of a whole",
        "supported_data_types": ["categorical"],
        "min_data_points": 2,
        "max_data_points": 20,
        "default_config": {"showLabels": True, "showPercentages": True}
    },
    {
        "name": "heatmap",
        "display_name": "Heat Map",
        "component_name": "HeatMap",
        "description": "Visualize data density or relationships in a matrix",
        "supported_data_types": ["matrix", "categorical"],
        "min_data_points": 4,
        "max_data_points": 10000,
        "default_config": {"colorScheme": "YlOrRd", "showValues": False}
    },
    {
        "name": "sankey",
        "display_name": "Sankey Diagram",
        "component_name": "SankeyDiagram",
        "description": "Show flow and relationships between entities",
        "supported_data_types": ["flow", "hierarchical"],
        "min_data_points": 3,
        "max_data_points": 500,
        "default_config": {"nodeWidth": 15, "nodePadding": 10}
    },
    {
        "name": "network",
        "display_name": "Network Graph",
        "component_name": "NetworkGraph",
        "description": "Visualize network relationships and connections",
        "supported_data_types": ["network", "hierarchical"],
        "min_data_points": 2,
        "max_data_points": 1000,
        "default_config": {"layout": "force", "showLabels": True}
    }
]

def create_extensions(db: Session):
    """Create required PostgreSQL extensions"""
    extensions = [
        "uuid-ossp",     # UUID generation
        "pg_trgm",       # Trigram similarity search
        "btree_gist",    # Advanced indexing
    ]
    
    for ext in extensions:
        try:
            db.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
            db.commit()
            logger.info(f"Extension {ext} created/verified")
        except Exception as e:
            logger.warning(f"Could not create extension {ext}: {e}")
            db.rollback()

def create_indexes(db: Session):
    """Create additional indexes for performance"""
    indexes = [
        # User indexes
        "CREATE INDEX IF NOT EXISTS idx_users_email_lower ON users(LOWER(email))",
        "CREATE INDEX IF NOT EXISTS idx_users_department ON users(department)",
        
        # Query indexes
        "CREATE INDEX IF NOT EXISTS idx_nl_queries_user_created ON natural_language_queries(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_nl_queries_status ON natural_language_queries(status)",
        
        # Supply chain indexes
        "CREATE INDEX IF NOT EXISTS idx_suppliers_country ON suppliers(country)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
        "CREATE INDEX IF NOT EXISTS idx_inventory_location ON inventory(location_code)",
        "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date DESC)",
        
        # Full text search indexes
        "CREATE INDEX IF NOT EXISTS idx_suppliers_fts ON suppliers USING gin(to_tsvector('english', name || ' ' || COALESCE(contact_name, '')))",
        "CREATE INDEX IF NOT EXISTS idx_products_fts ON products USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')))",
    ]
    
    for index in indexes:
        try:
            db.execute(text(index))
            db.commit()
            logger.info(f"Index created: {index.split('idx_')[1].split(' ')[0]}")
        except Exception as e:
            logger.warning(f"Could not create index: {e}")
            db.rollback()

def create_initial_roles(db: Session) -> Dict[str, Role]:
    """Create initial roles"""
    roles = {}
    for role_data in INITIAL_ROLES:
        role = db.query(Role).filter_by(name=role_data["name"]).first()
        if not role:
            role = Role(**role_data)
            db.add(role)
            logger.info(f"Created role: {role_data['name']}")
        roles[role_data["name"]] = role
    db.commit()
    return roles

def create_initial_permissions(db: Session) -> Dict[str, Permission]:
    """Create initial permissions"""
    permissions = {}
    for perm_data in INITIAL_PERMISSIONS:
        key = f"{perm_data['resource']}:{perm_data['action']}"
        perm = db.query(Permission).filter_by(
            resource=perm_data["resource"],
            action=perm_data["action"]
        ).first()
        if not perm:
            perm = Permission(**perm_data)
            db.add(perm)
            logger.info(f"Created permission: {key}")
        permissions[key] = perm
    db.commit()
    return permissions

def assign_role_permissions(db: Session, roles: Dict[str, Role], permissions: Dict[str, Permission]):
    """Assign permissions to roles"""
    for role_name, permission_patterns in ROLE_PERMISSIONS_MAPPING.items():
        role = roles.get(role_name)
        if not role:
            continue
            
        for pattern in permission_patterns:
            if pattern == "*":
                # Assign all permissions
                for perm in permissions.values():
                    if perm not in role.permissions:
                        role.permissions.append(perm)
            elif pattern.endswith(":*"):
                # Assign all actions for a resource
                resource = pattern.split(":")[0]
                for key, perm in permissions.items():
                    if perm.resource == resource and perm not in role.permissions:
                        role.permissions.append(perm)
            elif "*:" in pattern:
                # Assign specific action for all resources
                action = pattern.split(":")[1]
                for key, perm in permissions.items():
                    if perm.action == action and perm not in role.permissions:
                        role.permissions.append(perm)
            else:
                # Assign specific permission
                perm = permissions.get(pattern.replace(":", ":"))
                if perm and perm not in role.permissions:
                    role.permissions.append(perm)
        
        logger.info(f"Assigned {len(role.permissions)} permissions to role: {role_name}")
    
    db.commit()

def create_admin_user(db: Session, admin_role: Role) -> User:
    """Create default admin user"""
    admin_email = settings.ADMIN_EMAIL or "admin@example.com"
    admin_password = settings.ADMIN_PASSWORD or "admin123"
    
    admin = db.query(User).filter_by(email=admin_email).first()
    if not admin:
        admin = User(
            email=admin_email,
            username="admin",
            password_hash=hash_password(admin_password),
            first_name="System",
            last_name="Administrator",
            role="admin",
            is_active=True,
            is_verified=True,
            email_verified_at=datetime.utcnow()
        )
        admin.roles.append(admin_role)
        db.add(admin)
        db.commit()
        logger.info(f"Created admin user: {admin_email}")
    else:
        logger.info(f"Admin user already exists: {admin_email}")
    
    return admin

def create_chart_types(db: Session):
    """Create initial chart types"""
    for chart_data in INITIAL_CHART_TYPES:
        chart_type = db.query(ChartType).filter_by(name=chart_data["name"]).first()
        if not chart_type:
            chart_type = ChartType(**chart_data)
            db.add(chart_type)
            logger.info(f"Created chart type: {chart_data['name']}")
    db.commit()

def create_sample_suppliers(db: Session, count: int = 20) -> List[Supplier]:
    """Create sample suppliers"""
    if db.query(Supplier).count() > 0:
        logger.info("Suppliers already exist, skipping sample data")
        return db.query(Supplier).all()
    
    suppliers = []
    supplier_names = [
        "Acme Manufacturing", "Global Parts Co", "Premier Components",
        "TechSupply Inc", "Industrial Solutions", "Quality Materials Ltd",
        "FastShip Logistics", "Reliable Suppliers", "MegaCorp Industries",
        "Pacific Trading", "European Imports", "Asian Electronics",
        "Green Materials Co", "NextGen Supplies", "Smart Components",
        "Direct Factory Outlet", "Wholesale Partners", "B2B Solutions",
        "Enterprise Suppliers", "Commercial Trading"
    ]
    
    countries = ["US", "CN", "DE", "JP", "GB", "FR", "IT", "CA", "MX", "IN"]
    categories = ["Electronics", "Mechanical", "Raw Materials", "Packaging", "Components"]
    
    for i in range(min(count, len(supplier_names))):
        supplier = Supplier(
            code=f"SUP{str(i+1).zfill(4)}",
            name=supplier_names[i],
            category=random.choice(categories),
            status="active",
            tier=random.randint(1, 3),
            country=random.choice(countries),
            city=f"City {i+1}",
            lead_time_days=random.randint(3, 30),
            rating=round(random.uniform(3.0, 5.0), 1),
            contact_email=f"contact@{supplier_names[i].lower().replace(' ', '')}.com"
        )
        suppliers.append(supplier)
        db.add(supplier)
    
    db.commit()
    logger.info(f"Created {len(suppliers)} sample suppliers")
    return suppliers

def create_sample_products(db: Session, count: int = 50) -> List[Product]:
    """Create sample products"""
    if db.query(Product).count() > 0:
        logger.info("Products already exist, skipping sample data")
        return db.query(Product).all()
    
    products = []
    categories = ["Electronics", "Components", "Accessories", "Raw Materials", "Finished Goods"]
    
    for i in range(count):
        product = Product(
            sku=f"PRD{str(i+1).zfill(5)}",
            name=f"Product {i+1}",
            description=f"Description for product {i+1}",
            category=random.choice(categories),
            unit_of_measure="units",
            unit_cost=round(random.uniform(10, 1000), 2),
            selling_price=round(random.uniform(20, 2000), 2),
            status="active",
            weight_kg=round(random.uniform(0.1, 50), 2)
        )
        products.append(product)
        db.add(product)
    
    db.commit()
    logger.info(f"Created {len(products)} sample products")
    return products

def create_sample_inventory(db: Session, products: List[Product]):
    """Create sample inventory records"""
    if db.query(Inventory).count() > 0:
        logger.info("Inventory already exists, skipping sample data")
        return
    
    locations = ["WH01", "WH02", "WH03", "STORE01", "STORE02"]
    
    inventory_records = []
    for product in products[:30]:  # Create inventory for first 30 products
        for location in random.sample(locations, random.randint(1, 3)):
            qty = random.randint(0, 1000)
            inventory = Inventory(
                product_id=product.id,
                location_code=location,
                quantity_on_hand=qty,
                quantity_reserved=random.randint(0, min(100, qty)),
                reorder_point=random.randint(50, 200),
                reorder_quantity=random.randint(100, 500),
                last_counted_date=datetime.utcnow().date()
            )
            inventory_records.append(inventory)
            db.add(inventory)
    
    db.commit()
    logger.info(f"Created {len(inventory_records)} sample inventory records")

def init_db(drop_existing: bool = False):
    """Initialize database with tables and initial data"""
    logger.info("Starting database initialization...")
    
    if drop_existing:
        logger.warning("Dropping existing tables...")
        DatabaseManager.drop_all()
    
    # Create tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create extensions
        create_extensions(db)
        
        # Create indexes
        create_indexes(db)
        
        # Create initial data
        roles = create_initial_roles(db)
        permissions = create_initial_permissions(db)
        assign_role_permissions(db, roles, permissions)
        
        # Create admin user
        admin_role = roles.get("admin")
        if admin_role:
            create_admin_user(db, admin_role)
        
        # Create chart types
        create_chart_types(db)
        
        # Create sample data if in development
        if settings.ENVIRONMENT == "development":
            logger.info("Creating sample data for development...")
            suppliers = create_sample_suppliers(db)
            products = create_sample_products(db)
            create_sample_inventory(db, products)
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize the database")
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Drop existing tables before creating new ones"
    )
    args = parser.parse_args()
    
    init_db(drop_existing=args.drop)
