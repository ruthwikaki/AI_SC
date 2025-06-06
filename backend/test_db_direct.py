import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:123456789@localhost:5432/Supplychain_AI"

try:
    print(f"Connecting to: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM products"))
        count = result.scalar()
        print(f"✓ Connected! Found {count} products")
        
        # Test the query
        result = conn.execute(text("""
            SELECT p.name, p.sku, i.quantity_on_hand, i.reorder_point 
            FROM products p 
            JOIN inventory i ON p.id = i.product_id 
            WHERE i.quantity_on_hand < i.reorder_point 
            LIMIT 5
        """))
        
        rows = result.fetchall()
        print(f"\nProducts with low inventory ({len(rows)} found):")
        for row in rows:
            print(f"  - {row.name} (SKU: {row.sku}) - On hand: {row.quantity_on_hand}, Reorder at: {row.reorder_point}")
            
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
