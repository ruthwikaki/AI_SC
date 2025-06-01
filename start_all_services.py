#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start all services for the Supply Chain LLM application."""

import os
import sys
import subprocess
import time
import signal
import psutil
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION SECTION
# ────────────────────────────────────────────────────────────────────────────────

# Backend configuration
BACKEND_APP_IMPORT = "app.api.server:api_app"
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = "8000"
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")

# Frontend configuration
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
FRONTEND_DEV_COMMAND = ["npm", "run", "dev"]

# Database configuration
DB_CONTAINER_NAME = "supply_chain_db"
DB_USER = "postgres"
DB_PASSWORD = "123456789"
DB_NAME = "AI_SC"
DB_PORT = "5432"

# ML Service configuration (optional)
ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
ML_PORT = "8001"

# ────────────────────────────────────────────────────────────────────────────────
# END CONFIGURATION SECTION
# ────────────────────────────────────────────────────────────────────────────────

processes = []
docker_available = False

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print(" 🚀 Starting Supply Chain LLM - Full Stack")
    print("=" * 60)

def check_port(port: int) -> bool:
    """Return True if the given TCP port is in use."""
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            return True
    return False

def kill_process_on_port(port: int) -> bool:
    """Kill any process listening on the given TCP port."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"⚠️  Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
                    time.sleep(1)
                    return True
        except Exception:
            continue
    return False

def check_docker():
    """Check if Docker is available and running"""
    global docker_available
    
    # First check if docker command exists
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False
    
    # Then check if Docker daemon is running
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_available = True
            return True
        else:
            # Docker is installed but not running
            print("⚠️  Docker is installed but not running.")
            return False
    except:
        return False

def start_docker_desktop():
    """Try to start Docker Desktop on Windows"""
    if sys.platform == "win32":
        print("🐳 Attempting to start Docker Desktop...")
        docker_paths = [
            r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
            r"C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe",
            os.path.expandvars(r"%ProgramFiles%\Docker\Docker\Docker Desktop.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Docker\Docker\Docker Desktop.exe"),
        ]
        
        for docker_path in docker_paths:
            if os.path.exists(docker_path):
                try:
                    subprocess.Popen([docker_path])
                    print("⏳ Waiting for Docker Desktop to start (this may take a minute)...")
                    
                    # Wait up to 60 seconds for Docker to start
                    for i in range(60):
                        time.sleep(1)
                        if check_docker():
                            print("✅ Docker Desktop started successfully!")
                            return True
                        if i % 10 == 0 and i > 0:
                            print(f"   Still waiting... ({i} seconds)")
                    
                except Exception as e:
                    print(f"❌ Failed to start Docker Desktop: {e}")
                break
        
    return False

def start_postgresql():
    """Start PostgreSQL database using Docker or provide alternatives"""
    print("\n📦 Checking PostgreSQL Database setup...")
    
    # Check if Docker is available
    if not check_docker():
        print("\n❌ Docker is not available.")
        
        # Try to start Docker Desktop on Windows
        if sys.platform == "win32":
            print("\n🔍 Docker Desktop not running. You have several options:")
            print("   1. Start Docker Desktop manually and run this script again")
            print("   2. Let this script try to start Docker Desktop")
            print("   3. Use an existing PostgreSQL installation")
            print("   4. Continue without database (limited functionality)")
            
            choice = input("\nYour choice [1-4]: ").strip()
            
            if choice == "2":
                if start_docker_desktop():
                    # Docker started successfully, continue with container setup
                    pass
                else:
                    print("❌ Could not start Docker Desktop automatically.")
                    print("   Please start it manually and run this script again.")
                    sys.exit(1)
            elif choice == "3":
                print("\n📝 Using existing PostgreSQL installation.")
                print("   Make sure PostgreSQL is running on port 5432")
                print("   with database 'AI_SC' and user 'postgres'")
                input("\nPress Enter to continue...")
                return
            elif choice == "4":
                print("\n⚠️  Continuing without database. Some features won't work.")
                return
            else:
                print("\n📋 Please start Docker Desktop manually and run this script again.")
                sys.exit(0)
        else:
            print("\n📋 Please install and start Docker, then run this script again.")
            print("   Alternatively, set up PostgreSQL manually on port 5432.")
            sys.exit(1)
    
    if docker_available:
        print("✅ Docker is available and running.")
        
        # Stop and remove existing container if it exists
        subprocess.run(["docker", "stop", DB_CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", DB_CONTAINER_NAME], capture_output=True)
        
        # Start new PostgreSQL container
        cmd = [
            "docker", "run", "-d",
            "--name", DB_CONTAINER_NAME,
            "-e", f"POSTGRES_USER={DB_USER}",
            "-e", f"POSTGRES_PASSWORD={DB_PASSWORD}",
            "-e", f"POSTGRES_DB={DB_NAME}",
            "-p", f"{DB_PORT}:{DB_PORT}",
            "-v", "supply_chain_data:/var/lib/postgresql/data",
            "postgres:15"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ PostgreSQL started in Docker on port {DB_PORT}")
                print("⏳ Waiting for database to be ready...")
                time.sleep(10)  # Give PostgreSQL time to start
            else:
                print(f"❌ Failed to start PostgreSQL container: {result.stderr}")
                print("\n🔧 Troubleshooting tips:")
                print("   1. Make sure Docker Desktop is running")
                print("   2. Check if port 5432 is already in use")
                print("   3. Try: docker system prune -a (to clean up Docker)")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error starting PostgreSQL: {e}")
            sys.exit(1)

def create_backend_env():
    """Create backend .env file if it doesn't exist"""
    env_path = os.path.join(BACKEND_DIR, ".env")
    if not os.path.exists(env_path):
        print("\n📝 Creating backend .env file...")
        env_content = f"""DATABASE_URL=postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/{DB_NAME}
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Created backend/.env")

def setup_backend_dependencies():
    """Install backend Python dependencies"""
    print("\n📦 Installing backend dependencies...")
    requirements_path = os.path.join(BACKEND_DIR, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("⚠️  No requirements.txt found in backend directory")
        return
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=BACKEND_DIR,
            check=True
        )
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install backend dependencies: {e}")
        sys.exit(1)

def run_database_migrations():
    """Run database migrations if Alembic is configured"""
    alembic_ini = os.path.join(BACKEND_DIR, "alembic.ini")
    if os.path.exists(alembic_ini):
        print("\n🔄 Running database migrations...")
        try:
            subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd=BACKEND_DIR,
                check=True
            )
            print("✅ Database migrations completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Migration failed (non-critical): {e}")
    else:
        print("ℹ️  No Alembic configuration found, skipping migrations")

def init_database_tables():
    """Initialize database tables using SQLAlchemy"""
    print("\n🏗️  Initializing database tables...")
    init_script = os.path.join(BACKEND_DIR, "app", "db", "init_db.py")
    
    if os.path.exists(init_script):
        try:
            subprocess.run(
                [sys.executable, init_script],
                cwd=BACKEND_DIR,
                check=True
            )
            print("✅ Database tables initialized")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Database initialization failed (non-critical): {e}")
    else:
        # Try to create tables using a simple script
        try:
            init_code = """
import sys
sys.path.insert(0, '.')
from app.db.database import engine, Base
from app.models import user, query, visualization, supply_chain, analytics
Base.metadata.create_all(bind=engine)
print("Tables created successfully")
"""
            subprocess.run(
                [sys.executable, "-c", init_code],
                cwd=BACKEND_DIR,
                check=True
            )
            print("✅ Database tables created")
        except:
            print("ℹ️  Could not auto-create tables (will be created on first run)")

def start_backend() -> subprocess.Popen:
    """Launches the FastAPI backend via uvicorn"""
    # Kill any existing process on the backend port
    if check_port(int(BACKEND_PORT)):
        print(f"⚠️  Port {BACKEND_PORT} is in use. Killing existing process...")
        kill_process_on_port(int(BACKEND_PORT))
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        BACKEND_APP_IMPORT,
        "--host", BACKEND_HOST,
        "--port", BACKEND_PORT,
        "--reload"
    ]
    
    print(f"\n▶️  Starting FastAPI backend: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time output
    
    return subprocess.Popen(
        cmd,
        cwd=BACKEND_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    )

def setup_frontend_dependencies():
    """Install frontend npm dependencies"""
    print("\n📦 Checking frontend dependencies...")
    
    if not os.path.exists(os.path.join(FRONTEND_DIR, "package.json")):
        print("⚠️  No package.json found in frontend directory")
        return
    
    # Check if node_modules exists
    if os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print("✅ Frontend dependencies already installed")
        return
    
    print("📦 Installing frontend dependencies (this may take a few minutes)...")
    try:
        # Use shell=True on Windows for npm commands
        if sys.platform == "win32":
            subprocess.run("npm install", cwd=FRONTEND_DIR, shell=True, check=True)
        else:
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
        print("✅ Frontend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install frontend dependencies: {e}")
        sys.exit(1)

def clear_frontend_env():
    """Clear or fix frontend .env to ensure Vite proxy works"""
    frontend_env = os.path.join(FRONTEND_DIR, ".env")
    if os.path.exists(frontend_env):
        print("\n⚠️  Clearing VITE_API_URL in frontend/.env to enable proxy...")
        with open(frontend_env, 'r') as f:
            lines = f.readlines()
        
        with open(frontend_env, 'w') as f:
            for line in lines:
                if line.strip().startswith('VITE_API_URL'):
                    f.write('# VITE_API_URL=  # Commented out to use Vite proxy\n')
                else:
                    f.write(line)
        print("✅ Frontend .env updated")

def start_frontend() -> subprocess.Popen:
    """Launches the Vite dev server from the frontend directory"""
    if not os.path.isdir(FRONTEND_DIR):
        print(f"❌ ERROR: Frontend directory not found: {FRONTEND_DIR}")
        sys.exit(1)
    
    # Kill any existing process on port 3001
    if check_port(3001):
        print("⚠️  Port 3001 is in use. Killing existing process...")
        kill_process_on_port(3001)
    
    print(f"\n▶️  Starting Vite frontend in {FRONTEND_DIR}")
    
    # Use shell=True on Windows for npm commands
    if sys.platform == "win32":
        return subprocess.Popen(
            "npm run dev",
            cwd=FRONTEND_DIR,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy()
        )
    else:
        return subprocess.Popen(
            FRONTEND_DEV_COMMAND,
            cwd=FRONTEND_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy()
        )

def start_ml_service() -> subprocess.Popen:
    """Start ML service if it exists"""
    ml_server = os.path.join(ML_DIR, "inference", "server.py")
    if os.path.exists(ml_server):
        print(f"\n▶️  Starting ML Service on port {ML_PORT}...")
        
        # Kill any existing process on ML port
        if check_port(int(ML_PORT)):
            print(f"⚠️  Port {ML_PORT} is in use. Killing existing process...")
            kill_process_on_port(int(ML_PORT))
        
        return subprocess.Popen(
            [sys.executable, ml_server, "--port", ML_PORT],
            cwd=ML_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy()
        )
    else:
        print("\nℹ️  ML service not found, skipping...")
        return None

def shutdown_processes(signum=None, frame=None):
    """Send SIGINT to all processes and stop Docker container"""
    print("\n\n🛑 Shutting down all services...")
    
    # Stop all subprocess
    for p in processes:
        if p and p.poll() is None:  # Check if process is still running
            try:
                if sys.platform == "win32":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGINT)
            except Exception:
                pass
    
    # Stop PostgreSQL Docker container if Docker is available
    if docker_available:
        print("🐘 Stopping PostgreSQL...")
        subprocess.run(["docker", "stop", DB_CONTAINER_NAME], capture_output=True)
    
    print("✅ All services stopped")
    sys.exit(0)

def print_service_urls():
    """Print service URLs for easy access"""
    print("\n" + "=" * 60)
    print(" ✅ All services started successfully!")
    print("=" * 60)
    print("\n📌 Service URLs:")
    print(f"   • Frontend:    http://localhost:3001")
    print(f"   • Backend API: http://localhost:{BACKEND_PORT}")
    print(f"   • API Docs:    http://localhost:{BACKEND_PORT}/api/docs")
    print(f"   • Health Check:http://localhost:{BACKEND_PORT}/api/health")
    if docker_available:
        print(f"   • PostgreSQL:  localhost:{DB_PORT} (Docker)")
    else:
        print(f"   • PostgreSQL:  Configure manually on localhost:{DB_PORT}")
    if os.path.exists(os.path.join(ML_DIR, "inference", "server.py")):
        print(f"   • ML Service:  http://localhost:{ML_PORT}")
    print("\n🎯 Press Ctrl+C to stop all services")
    print("=" * 60 + "\n")

def main():
    # Print startup banner
    print_banner()
    
    # Catch CTRL+C (SIGINT) so we can clean up all processes
    signal.signal(signal.SIGINT, shutdown_processes)
    
    # Windows-specific signal handling
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, shutdown_processes)
    
    try:
        # 1) Start PostgreSQL
        start_postgresql()
        
        # 2) Setup backend
        create_backend_env()
        setup_backend_dependencies()
        run_database_migrations()
        init_database_tables()
        
        # 3) Start backend
        backend_proc = start_backend()
        processes.append(backend_proc)
        time.sleep(5)  # Give backend time to start
        
        # 4) Setup and start frontend
        clear_frontend_env()
        setup_frontend_dependencies()
        frontend_proc = start_frontend()
        processes.append(frontend_proc)
        
        # 5) Start ML service (optional)
        ml_proc = start_ml_service()
        if ml_proc:
            processes.append(ml_proc)
        
        # 6) Print service URLs
        print_service_urls()
        
        # 7) Wait for processes
        try:
            for p in processes:
                if p:
                    p.wait()
        except KeyboardInterrupt:
            shutdown_processes()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        shutdown_processes()

if __name__ == "__main__":
    main()