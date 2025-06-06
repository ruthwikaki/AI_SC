import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import engine, check_database_connection

print("Testing database connection...")

if check_database_connection():
    print("✓ Database connection successful!")
    
    # List tables
    with engine.connect() as conn:
        result = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 10")
        tables = result.fetchall()
        print(f"\nFound {len(tables)} tables (showing first 10):")
        for table in tables:
            print(f"  - {table[0]}")
else:
    print("✗ Database connection failed!")
