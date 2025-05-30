#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start all services for Supply Chain LLM application."""

import os
import sys
import subprocess
import time
import psutil
import signal


def check_port(port):
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False


def kill_process_on_port(port):
    """Kill process using a specific port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
                    time.sleep(1)
                    return True
        except:
            pass
    return False


def start_services():
    print("=" * 60)
    print(" 🚀 Starting Supply Chain LLM - Full Stack")
    print("=" * 60)
    
    # Store the root directory
    root_dir = os.getcwd()
    
    # Check and clear ports if needed
    ports = {
        5432: "PostgreSQL",
        8000: "Backend API",
        8001: "ML Service",
        3001: "Frontend"  # Updated to match your vite.config.js
    }
    
    for port, service in ports.items():
        if check_port(port):
            print(f"⚠️  Port {port} ({service}) is in use. Attempting to clear...")
            kill_process_on_port(port)
    
    # Start PostgreSQL via Docker
    print("\n1. Starting PostgreSQL Database...")
    try:
        # Check if postgres container exists
        check_container = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=supply_chain_db", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        
        if "supply_chain_db" in check_container.stdout:
            print("   Removing existing PostgreSQL container...")
            subprocess.run(["docker", "stop", "supply_chain_db"], capture_output=True)
            subprocess.run(["docker", "rm", "supply_chain_db"], capture_output=True)
        
        # Start new PostgreSQL container
        subprocess.run([
            "docker", "run", "-d",
            "--name", "supply_chain_db",
            "-e", "POSTGRES_USER=postgres",
            "-e", "POSTGRES_PASSWORD=123456789",
            "-e", "POSTGRES_DB=AI_SC",
            "-p", "5432:5432",
            "-v", "supply_chain_data:/var/lib/postgresql/data",
            "postgres:15"
        ])
        print("   ✅ PostgreSQL started on port 5432")
        time.sleep(10)  # Wait longer for PostgreSQL to start
    except Exception as e:
        print(f"   ❌ Failed to start PostgreSQL: {e}")
        print("   Make sure Docker Desktop is running!")
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    env_path = os.path.join(root_dir, "backend", ".env")
    if not os.path.exists(env_path):
        print("\n2. Creating backend .env file...")
        # FIXED: Only include the fields that your backend Settings expects
        env_content = """DATABASE_URL=postgresql://postgres:123456789@localhost:5432/AI_SC
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, "w") as f:
            f.write(env_content)
        print("   ✅ Created backend/.env")
    
    # Initialize database
    print("\n3. Setting up Backend...")
    backend_dir = os.path.join(root_dir, "backend")
    os.chdir(backend_dir)
    
    # Install requirements
    print("   Installing backend dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Check if alembic.ini exists
    if os.path.exists("alembic.ini"):
        print("   Running database migrations...")
        try:
            subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
            print("   ✅ Database migrations completed")
        except:
            print("   ⚠️  Migrations failed, but continuing...")
    else:
        print("   ⚠️  No alembic.ini found, skipping migrations")
    
    # Start backend
    print("\n4. Starting Backend API...")
    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=backend_dir
    )
    print("   ✅ Backend API starting on http://localhost:8000")
    time.sleep(5)  # Wait for backend to start
    
    # Start ML service
    print("\n5. Starting ML Service...")
    ml_dir = os.path.join(root_dir, "ml")
    if os.path.exists(os.path.join(ml_dir, "inference", "server.py")):
        os.chdir(ml_dir)
        ml_process = subprocess.Popen(
            [sys.executable, "inference/server.py", "--port", "8001"],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=ml_dir
        )
        print("   ✅ ML Service starting on http://localhost:8001")
    else:
        print("   ⚠️  ML service not found, skipping...")
        ml_process = None
    
    # Start frontend
    print("\n6. Starting Frontend...")
    frontend_dir = os.path.join(root_dir, "frontend")
    os.chdir(frontend_dir)
    
    # Check if node_modules exists
    if not os.path.exists("node_modules"):
        print("   Installing frontend dependencies...")
        subprocess.run(["npm", "install"], shell=True)
    
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        shell=True,
        cwd=frontend_dir
    )
    print("   ✅ Frontend starting on http://localhost:3001")
    
    print("\n" + "=" * 60)
    print(" ✅ All services started successfully!")
    print("=" * 60)
    print("\n📋 Service URLs:")
    print("   - Frontend:    http://localhost:3001")  # Updated port
    print("   - Backend API: http://localhost:8000")
    print("   - API Docs:    http://localhost:8000/docs")
    print("   - ML Service:  http://localhost:8001")
    print("   - PostgreSQL:  localhost:5432")
    print("\n📌 Database credentials:")
    print("   - User: postgres")
    print("   - Password: 123456789")
    print("   - Database: AI_SC")
    print("\nPress Ctrl+C to stop all services")
    
    # Wait for all processes
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        backend_process.terminate()
        if ml_process:
            ml_process.terminate()
        frontend_process.terminate()
        subprocess.run(["docker", "stop", "supply_chain_db"], capture_output=True)
        print("✅ All services stopped")


if __name__ == "__main__":
    start_services()