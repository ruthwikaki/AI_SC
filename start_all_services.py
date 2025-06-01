#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start all services for the Supply Chain LLM application."""

import os
import sys
import subprocess
import time
import psutil


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
                    print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
                    time.sleep(1)
                    return True
        except Exception:
            continue
    return False


def run_command(cmd: list, cwd: str = None, env: dict = None, silent: bool = False, use_shell: bool = False) -> bool:
    """Run a subprocess command, optionally silencing output or using shell."""
    kwargs = {"cwd": cwd, "env": env or os.environ.copy()}
    if silent:
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
    if use_shell:
        # When shell=True, pass command as a single string
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        return subprocess.call(cmd_str, shell=True, **kwargs) == 0
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"⚠️  Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return result.returncode == 0


def start_services():
    print("=" * 60)
    print(" 🚀 Starting Supply Chain LLM - Full Stack")
    print("=" * 60)

    root_dir = os.getcwd()

    # Define required ports and service names
    ports = {
        5432: "PostgreSQL",
        8000: "Backend API",
        8001: "ML Service",
        3001: "Frontend"
    }

    # Free up occupied ports
    for port, service in ports.items():
        if check_port(port):
            print(f"⚠️  Port {port} ({service}) is in use. Killing existing process...")
            kill_process_on_port(port)

    # 1. Start PostgreSQL via Docker
    print("\n1. Starting PostgreSQL Database...")
    run_command(["docker", "stop", "supply_chain_db"], silent=True, use_shell=sys.platform=='win32')
    run_command(["docker", "rm", "supply_chain_db"], silent=True, use_shell=sys.platform=='win32')
    run_command([
        "docker", "run", "-d",
        "--name", "supply_chain_db",
        "-e", "POSTGRES_USER=postgres",
        "-e", "POSTGRES_PASSWORD=123456789",
        "-e", "POSTGRES_DB=AI_SC",
        "-p", "5432:5432",
        "-v", "supply_chain_data:/var/lib/postgresql/data",
        "postgres:15"
    ], use_shell=sys.platform=='win32')
    print("   ✅ PostgreSQL started on port 5432")
    time.sleep(10)

    # 2. Create .env for backend
    env_path = os.path.join(root_dir, "backend", ".env")
    if not os.path.exists(env_path):
        print("\n2. Creating backend .env file...")
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        env_content = (
            "DATABASE_URL=postgresql://postgres:123456789@localhost:5432/AI_SC\n"
            "JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production\n"
            "JWT_ALGORITHM=HS256\n"
            "ACCESS_TOKEN_EXPIRE_MINUTES=30\n"
        )
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("   ✅ Created backend/.env")

    # 3. Setup and start backend
    print("\n3. Initializing Backend...")
    backend_dir = os.path.join(root_dir, "backend")
    print("   Installing backend dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=backend_dir)
    if os.path.exists(os.path.join(backend_dir, "alembic.ini")):
        print("   Running database migrations...")
        run_command([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=backend_dir)
    else:
        print("   ⚠️  No alembic.ini found; skipping migrations")
    print("\n4. Starting Backend API...")
    backend_proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=backend_dir,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    print("   ✅ Backend API starting on http://localhost:8000")
    time.sleep(5)

    # 4. Start ML service if present
    print("\n5. Starting ML Service...")
    ml_proc = None
    ml_dir = os.path.join(root_dir, "ml")
    if os.path.isdir(ml_dir) and os.path.isfile(os.path.join(ml_dir, "inference", "server.py")):
        ml_proc = subprocess.Popen(
            [sys.executable, "inference/server.py", "--port", "8001"], cwd=ml_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        print("   ✅ ML Service starting on http://localhost:8001")
    else:
        print("   ⚠️  ML service not found; skipping")

    # 5. Start frontend
    print("\n6. Starting Frontend...")
    frontend_dir = os.path.join(root_dir, "frontend")
    if not os.path.isdir(frontend_dir):
        print("   ❌ Frontend directory not found: skipping UI startup")
    else:
        print("   Installing frontend dependencies...")
        # On Windows, npm may require shell=True
        subprocess.run("npm install", cwd=frontend_dir, shell=(sys.platform=='win32'))
        print("   ✅ Frontend dependencies installed")
        frontend_proc = subprocess.Popen(
            "npm run dev", cwd=frontend_dir,
            shell=True
        )
        print("   ✅ Frontend starting on http://localhost:3001")

    print("\n" + "=" * 60)
    print(" ✅ All services started successfully!")
    print("=" * 60)

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        if 'backend_proc' in locals():
            backend_proc.terminate()
        if 'ml_proc' in locals() and ml_proc:
            ml_proc.terminate()
        if 'frontend_proc' in locals():
            frontend_proc.terminate()
        run_command(["docker", "stop", "supply_chain_db"], silent=True, use_shell=sys.platform=='win32')
        print("✅ All services stopped")


if __name__ == '__main__':
    start_services()
