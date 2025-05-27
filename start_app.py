import os
import subprocess
import sys
import time
import signal
import shutil
import venv
import socket

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(PROJECT_ROOT, 'backend')
ML_PATH = os.path.join(PROJECT_ROOT, 'ml')
ML_INFERENCE_PATH = os.path.join(ML_PATH, 'inference')
FRONTEND_PATH = os.path.join(PROJECT_ROOT, 'frontend')
MODELS_PATH = os.path.join(ML_PATH, 'models')

# Virtual environment paths
BACKEND_VENV = os.path.join(PROJECT_ROOT, 'backend_venv')
ML_VENV = os.path.join(PROJECT_ROOT, 'ml_venv')

# Python executables
if os.name == 'nt':  # Windows
    BACKEND_PYTHON = os.path.join(BACKEND_VENV, 'Scripts', 'python.exe')
    ML_PYTHON = os.path.join(ML_VENV, 'Scripts', 'python.exe')
    PIP_BACKEND = os.path.join(BACKEND_VENV, 'Scripts', 'pip.exe')
    PIP_ML = os.path.join(ML_VENV, 'Scripts', 'pip.exe')
else:  # macOS/Linux
    BACKEND_PYTHON = os.path.join(BACKEND_VENV, 'bin', 'python')
    ML_PYTHON = os.path.join(ML_VENV, 'bin', 'python')
    PIP_BACKEND = os.path.join(BACKEND_VENV, 'bin', 'pip')
    PIP_ML = os.path.join(ML_VENV, 'bin', 'pip')

# Database settings
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'AI_SC',
    'user': 'postgres',
    'password': '123456789'
}

# Service ports
PORTS = {
    'frontend': 3001,
    'backend': 8000,
    'ml': 8001
}

# Process tracking
processes = []

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def run_command(cmd, cwd=None, env=None, check=True):
    """Run a command and return its output"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    print(f"Running: {cmd} in {cwd or 'current directory'}")
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        env=full_env,
        shell=True, 
        text=True, 
        capture_output=True
    )
    
    if result.stdout:
        print(f"Output: {result.stdout.strip()}")
    if result.stderr:
        print(f"Error: {result.stderr.strip()}")
    
    if check and result.returncode != 0:
        print(f"Command failed with exit code: {result.returncode}")
        return None
    
    return result.stdout

def check_port(port, service_name):
    """Check if a port is available using socket"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"✅ {service_name} is responding on port {port}")
            return True
        else:
            print(f"❌ {service_name} is not responding on port {port}")
            return False
    except Exception as e:
        print(f"❌ Error checking {service_name} on port {port}: {e}")
        return False

def check_venv(venv_path):
    """Check if virtual environment exists and create if it doesn't"""
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        return False
    return True

def install_dependencies():
    """Install dependencies in virtual environments"""
    print_header("Setting up virtual environments and dependencies")
    
    # Backend dependencies
    backend_exists = check_venv(BACKEND_VENV)
    if not backend_exists:
        print("Installing backend dependencies...")
        run_command(f'"{PIP_BACKEND}" install --upgrade pip')
        backend_req_path = os.path.join(BACKEND_PATH, "requirements.txt")
        if os.path.exists(backend_req_path):
            run_command(f'"{PIP_BACKEND}" install -r "{backend_req_path}"')
        else:
            print(f"⚠️ Backend requirements.txt not found at {backend_req_path}")
        run_command(f'"{PIP_BACKEND}" install psycopg2-binary fastapi uvicorn')
    else:
        print("Backend virtual environment exists.")
    
    # ML dependencies
    ml_exists = check_venv(ML_VENV)
    if not ml_exists:
        print("Installing ML dependencies...")
        run_command(f'"{PIP_ML}" install --upgrade pip')
        ml_req_path = os.path.join(ML_PATH, "requirements.txt")
        if os.path.exists(ml_req_path):
            run_command(f'"{PIP_ML}" install -r "{ml_req_path}"')
        else:
            print(f"⚠️ ML requirements.txt not found at {ml_req_path}")
            run_command(f'"{PIP_ML}" install torch transformers fastapi uvicorn')
    else:
        print("ML virtual environment exists.")
    
    # Check if frontend dependencies are installed
    node_modules_path = os.path.join(FRONTEND_PATH, 'node_modules')
    if not os.path.exists(node_modules_path):
        print("Installing frontend dependencies...")
        # Check if package.json exists
        package_json = os.path.join(FRONTEND_PATH, 'package.json')
        if not os.path.exists(package_json):
            print(f"❌ package.json not found at {package_json}")
            return False
        result = run_command("npm install", cwd=FRONTEND_PATH, check=False)
        if result is None:
            print("❌ Failed to install frontend dependencies")
            return False
    else:
        print("Frontend dependencies exist.")
    
    return True

def check_frontend_build():
    """Check if frontend build files exist"""
    print_header("Checking frontend configuration")
    
    # Check essential files
    essential_files = [
        os.path.join(FRONTEND_PATH, 'package.json'),
        os.path.join(FRONTEND_PATH, 'vite.config.js'),
        os.path.join(FRONTEND_PATH, 'src', 'App.jsx'),
        os.path.join(FRONTEND_PATH, 'src', 'index.jsx'),
        os.path.join(FRONTEND_PATH, 'index.html')
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing essential file: {file_path}")
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {os.path.basename(file_path)}")
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} essential files")
        return False
    
    print("✅ All essential frontend files found")
    return True

def check_models():
    """Check if LLM models are downloaded"""
    print_header("Checking LLM models")
    
    model_paths = [
        os.path.join(MODELS_PATH, 'mistral', 'weights'),
        os.path.join(MODELS_PATH, 'llama3', 'weights'),
        os.path.join(MODELS_PATH, 'tokenizers')
    ]
    
    missing = []
    for path in model_paths:
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(path) or not os.listdir(path):
            missing.append(path)
    
    if missing:
        print("⚠️ Some model files appear to be missing.")
        print("This is okay for initial testing - the application can run without models.")
        print("You can download models later if needed.")
    else:
        print("✅ LLM model files found.")

def check_database():
    """Check PostgreSQL connection"""
    print_header("Checking database connection")
    
    try:
        # Test database connection using the backend environment
        test_conn_script = f'''
import sys
try:
    import psycopg2
    conn = psycopg2.connect(
        host="{DB_CONFIG['host']}",
        port="{DB_CONFIG['port']}",
        database="{DB_CONFIG['database']}",
        user="{DB_CONFIG['user']}",
        password="{DB_CONFIG['password']}",
        connect_timeout=5
    )
    conn.close()
    print("Connection successful")
except ImportError:
    print("psycopg2 not available")
except Exception as e:
    print(f"Connection failed: {{e}}")
'''
        
        result = run_command(f'"{BACKEND_PYTHON}" -c "{test_conn_script}"', check=False)
        if result and 'Connection successful' in result:
            print("✅ PostgreSQL is running and connection successful")
            return True
        elif result and 'psycopg2 not available' in result:
            print("⚠️ psycopg2 not installed - installing now...")
            run_command(f'"{PIP_BACKEND}" install psycopg2-binary')
            # Try again
            result = run_command(f'"{BACKEND_PYTHON}" -c "{test_conn_script}"', check=False)
            if result and 'Connection successful' in result:
                print("✅ PostgreSQL connection successful after installing psycopg2")
                return True
            else:
                print("❌ PostgreSQL connection still failed")
                return False
        else:
            print("⚠️ PostgreSQL connection failed - continuing anyway")
            print("The application may work without database for testing")
            return False
            
    except Exception as e:
        print(f"⚠️ Database connection test failed: {e}")
        return False

def start_service(cmd, cwd=None, env=None, name=None):
    """Start a service and return the process"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    print(f"\n🚀 Starting {name or 'service'}...")
    print(f"Command: {cmd}")
    print(f"Working directory: {cwd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=full_env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait briefly and check if process started
        time.sleep(2)
        
        if process.poll() is not None:
            print(f"❌ {name or 'Service'} failed to start with exit code {process.returncode}")
            # Try to read any error output
            try:
                output = process.stdout.read()
                if output:
                    print(f"Error output: {output}")
            except:
                pass
            return None
        
        processes.append((process, name))
        print(f"✅ {name or 'Service'} started successfully")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start {name or 'service'}: {e}")
        return None

def wait_for_service(port, service_name, timeout=30):
    """Wait for a service to be available on a port"""
    print(f"⏳ Waiting for {service_name} to be ready on port {port}...")
    
    for i in range(timeout):
        if check_port(port, service_name):
            return True
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            print(f"   Still waiting for {service_name}... ({i}/{timeout}s)")
    
    print(f"⚠️ {service_name} did not become ready within {timeout} seconds")
    return False

def start_services():
    """Start all services"""
    print_header("Starting services")
    
    # Prepare environment variables
    env = {
        'DB_HOST': DB_CONFIG['host'],
        'DB_PORT': DB_CONFIG['port'],
        'DB_NAME': DB_CONFIG['database'],
        'DB_USER': DB_CONFIG['user'],
        'DB_PASSWORD': DB_CONFIG['password']
    }
    
    # Start frontend first (it's most likely to work)
    print("🎨 Starting Frontend service...")
    frontend_cmd = "npm run dev"
    frontend_process = start_service(frontend_cmd, cwd=FRONTEND_PATH, name="Frontend")
    if not frontend_process:
        print("❌ Failed to start Frontend service")
        return False
    
    # Wait for frontend
    print("⏳ Waiting for frontend to start...")
    time.sleep(5)  # Give frontend time to compile
    if not wait_for_service(PORTS['frontend'], "Frontend", timeout=20):
        print("⚠️ Frontend may still be starting...")
    
    # Try to start backend (optional for basic testing)
    if os.path.exists(os.path.join(BACKEND_PATH, 'main.py')):
        print("🔧 Starting Backend service...")
        backend_cmd = f'"{BACKEND_PYTHON}" main.py'
        backend_process = start_service(backend_cmd, cwd=BACKEND_PATH, env=env, name="Backend")
        if backend_process:
            if wait_for_service(PORTS['backend'], "Backend", timeout=15):
                print("✅ Backend started successfully")
        else:
            print("⚠️ Backend failed to start - frontend will still work")
    else:
        print("⚠️ Backend main.py not found - skipping backend startup")
    
    # Try to start ML service (optional)
    if os.path.exists(os.path.join(ML_INFERENCE_PATH, 'server.py')):
        print("🤖 Starting ML service...")
        ml_cmd = f'"{ML_PYTHON}" server.py'
        ml_process = start_service(ml_cmd, cwd=ML_INFERENCE_PATH, name="ML Service")
        if ml_process:
            if wait_for_service(PORTS['ml'], "ML Service", timeout=15):
                print("✅ ML Service started successfully")
        else:
            print("⚠️ ML Service failed to start - continuing without it")
    else:
        print("⚠️ ML server.py not found - skipping ML service startup")
    
    print(f"\n🎉 Services started!")
    print(f"Frontend: http://localhost:{PORTS['frontend']}")
    print(f"Backend: http://localhost:{PORTS['backend']} (if running)")
    print(f"ML Service: http://localhost:{PORTS['ml']} (if running)")
    print("\nPress Ctrl+C to stop all services\n")
    return True

def monitor_processes():
    """Monitor process output and status"""
    print("👀 Monitoring processes... (Press Ctrl+C to stop)")
    
    try:
        while True:
            all_running = True
            
            for process, name in processes:
                if process.poll() is not None:
                    print(f"⚠️ {name} process terminated with exit code {process.returncode}")
                    all_running = False
                    break
            
            if not all_running:
                return False
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        return True

def cleanup():
    """Clean up all processes"""
    if not processes:
        return
        
    print("🧹 Shutting down services...")
    for process, name in processes:
        try:
            print(f"Terminating {name}...")
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Force killing {name}...")
            process.kill()
        except:
            pass
    print("✅ All processes terminated")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown requested...")
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    try:
        print_header("🚀 AI Supply Chain Application Startup")
        
        # Setup dependencies
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return 1
        
        # Check frontend configuration
        if not check_frontend_build():
            print("❌ Frontend configuration issues detected")
            print("Please make sure all frontend files exist")
            return 1
        
        # Check database connection (non-critical)
        check_database()
        
        # Check LLM models (non-critical)
        check_models()
        
        # Start all services
        if not start_services():
            print("❌ Failed to start services")
            return 1
        
        # Monitor processes
        monitor_processes()
        
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup()

if __name__ == "__main__":
    sys.exit(main())