#!/usr/bin/env python3
"""Initialize Alembic for database migrations"""

import os
import subprocess
import sys

def init_alembic():
    """Initialize Alembic in the backend directory"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if alembic.ini already exists
    if os.path.exists(os.path.join(backend_dir, "alembic.ini")):
        print("Alembic already initialized")
        return
    
    print("Initializing Alembic...")
    
    # Run alembic init
    subprocess.run([sys.executable, "-m", "alembic", "init", "alembic"], cwd=backend_dir)
    
    # Update alembic.ini with correct database URL
    alembic_ini = os.path.join(backend_dir, "alembic.ini")
    if os.path.exists(alembic_ini):
        with open(alembic_ini, 'r') as f:
            content = f.read()
        
        # Replace the database URL
        content = content.replace(
            "sqlalchemy.url = driver://user:pass@localhost/dbname",
            "sqlalchemy.url = postgresql://postgres:123456789@localhost:5432/Supplychain_AI"
        )
        
        with open(alembic_ini, 'w') as f:
            f.write(content)
        
        print("âœ… Alembic initialized successfully")
    
    # Update env.py to import your models
    env_py = os.path.join(backend_dir, "alembic", "env.py")
    if os.path.exists(env_py):
        with open(env_py, 'r') as f:
            lines = f.readlines()
        
        # Find the target_metadata line and update it
        for i, line in enumerate(lines):
            if "target_metadata = None" in line:
                lines[i] = """# Import your models
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import Base
from app.models import user, query, visualization, supply_chain, analytics

target_metadata = Base.metadata
"""
                break
        
        with open(env_py, 'w') as f:
            f.writelines(lines)
        
        print("âœ… Alembic env.py updated")

if __name__ == "__main__":
    init_alembic()