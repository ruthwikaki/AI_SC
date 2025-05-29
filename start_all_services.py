import os
import sys
import subprocess
import time
import psutil

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
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
                    time.sleep(1)
                    return True
        except:
            pass
    return False

def start_services():
    print("="*60)
    print(" 🚀 Starting Supply Chain LLM - Full Stack")
    print("="*60)
    
    # Check and clear ports if needed
    ports = {
        5432: "PostgreSQL",
        8000: "Backend API",
        8001: "ML Service",
        3001: "Frontend"
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
            "-e", "POSTGRES_USER=scuser",
            "-e", "POSTGRES_PASSWORD=scpass123",
            "-e", "POSTGRES_DB=supply_chain_llm",
            "-p", "5432:5432",
            "-v", "supply_chain_data:/var/lib/postgresql/data",
            "postgres:15"
        ])
        print("   ✅ PostgreSQL started on port 5432")
        time.sleep(5)  # Wait for PostgreSQL to start
    except Exception as e:
        print(f"   ❌ Failed to start PostgreSQL: {e}")
        print("   Make sure Docker is installed and running!")
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    env_path = os.path.join("backend", ".env")
    if not os.path.exists(env_path):
        print("\n2. Creating backend .env file...")
        env_content = """DATABASE_URL=postgresql://scuser:scpass123@localhost:5432/supply_chain_llm
DB_HOST=localhost
DB_PORT=5432
DB_NAME=supply_chain_llm
DB_USER=scuser
DB_PASSWORD=scpass123
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_DELTA=30
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
ML_SERVICE_URL=http://localhost:8001
REDIS_URL=redis://localhost:6379
"""
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, "w") as f:
            f.write(env_content)
        print("   ✅ Created backend/.env")
    
    # Initialize database
    print("\n3. Initializing database...")
    os.chdir("backend")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run database migrations
    print("   Running database migrations...")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"])
    print("   ✅ Database initialized")
    
    # Start backend
    print("\n4. Starting Backend API...")
    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    print("   ✅ Backend API starting on http://localhost:8000")
    
    # Start ML service
    print("\n5. Starting ML Service...")
    os.chdir("../ml")
    ml_process = subprocess.Popen(
        [sys.executable, "inference/server.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    print("   ✅ ML Service starting on http://localhost:8001")
    
    # Start frontend
    print("\n6. Starting Frontend...")
    os.chdir("../frontend")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        shell=True
    )
    print("   ✅ Frontend starting on http://localhost:3001")
    
    print("\n" + "="*60)
    print(" ✅ All services started successfully!")
    print("="*60)
    print("\n📋 Service URLs:")
    print("   - Frontend:    http://localhost:3001")
    print("   - Backend API: http://localhost:8000")
    print("   - API Docs:    http://localhost:8000/docs")
    print("   - ML Service:  http://localhost:8001")
    print("   - PostgreSQL:  localhost:5432")
    print("\n📌 Default credentials:")
    print("   - Email: admin@example.com")
    print("   - Password: admin123")
    print("\nPress Ctrl+C to stop all services")
    
    # Wait for all processes
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        backend_process.terminate()
        ml_process.terminate()
        frontend_process.terminate()
        subprocess.run(["docker", "stop", "supply_chain_db"], capture_output=True)
        print("✅ All services stopped")

if __name__ == "__main__":
    start_services()
