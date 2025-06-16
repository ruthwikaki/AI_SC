"""Initialize database with proper import order"""
from .database import Base, engine

def init_db():
    """Initialize database tables"""
    # Import all models to ensure they are registered with Base
    # Using lazy imports to prevent circular dependencies
    
    # Import models in correct order
    from ..models import user  # User model first
    from ..models import supply_chain  # Supply chain models
    from ..models import analytics  # Analytics models
    from ..models import visualization  # Visualization models
    from ..models import system  # System models
    from ..models import query  # Query models
    from ..models import extended_models  # Extended models last
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

if __name__ == "__main__":
    init_db()
