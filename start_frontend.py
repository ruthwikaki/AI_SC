import os
import subprocess
import sys
import time
import signal

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, 'frontend')

# Service port
FRONTEND_PORT = 3000

# Process tracking
frontend_process = None

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*60)
    print(f" {message}")
    print("="*60)

def run_command(cmd, cwd=None, check=True):
    """Run a command and return its output"""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        shell=True, 
        text=True, 
        capture_output=True
    )
    
    if result.stdout:
        print(f"Output: {result.stdout.strip()}")
    if result.stderr and check:
        print(f"Error: {result.stderr.strip()}")
    
    if check and result.returncode != 0:
        print(f"Command failed with exit code: {result.returncode}")
        return None
    
    return result.stdout

def check_frontend_setup():
    """Check if frontend is properly set up"""
    print_header("Checking frontend configuration")
    
    # Check essential files
    essential_files = [
        os.path.join(FRONTEND_PATH, 'package.json'),
        os.path.join(FRONTEND_PATH, 'index.html')
    ]
    
    # Check for either Vite or Next.js config
    config_files = [
        os.path.join(FRONTEND_PATH, 'vite.config.js'),
        os.path.join(FRONTEND_PATH, 'next.config.js')
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {os.path.basename(file_path)}")
    
    # Check for at least one config file
    config_found = False
    for config in config_files:
        if os.path.exists(config):
            print(f"✅ Found: {os.path.basename(config)}")
            config_found = True
            break
    
    if not config_found:
        print("❌ No Vite or Next.js config found")
        missing_files.append("config file")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} essential files")
        return False
    
    print("\n✅ Frontend configuration looks good")
    return True

def install_dependencies():
    """Install frontend dependencies"""
    print_header("Installing frontend dependencies")
    
    # Check if node_modules exists
    node_modules_path = os.path.join(FRONTEND_PATH, 'node_modules')
    if os.path.exists(node_modules_path):
        print("✅ Dependencies already installed")
        return True
    
    print("📦 Installing dependencies with npm...")
    result = run_command("npm install", cwd=FRONTEND_PATH, check=False)
    
    if result is None:
        print("\n❌ Failed to install dependencies")
        print("Make sure Node.js and npm are installed")
        return False
    
    print("✅ Dependencies installed successfully")
    return True

def start_frontend():
    """Start the frontend development server"""
    global frontend_process
    
    print_header("Starting Frontend Server")
    
    # Check which start command to use
    package_json_path = os.path.join(FRONTEND_PATH, 'package.json')
    if os.path.exists(package_json_path):
        # Read package.json to find the right command
        import json
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
            scripts = package_data.get('scripts', {})
            
            if 'dev' in scripts:
                start_cmd = "npm run dev"
            elif 'start' in scripts:
                start_cmd = "npm start"
            else:
                print("❌ No 'dev' or 'start' script found in package.json")
                return False
    
    print(f"🚀 Starting frontend with: {start_cmd}")
    print(f"📍 Working directory: {FRONTEND_PATH}")
    
    try:
        frontend_process = subprocess.Popen(
            start_cmd,
            cwd=FRONTEND_PATH,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait and check if process started
        time.sleep(3)
        
        if frontend_process.poll() is not None:
            print(f"❌ Frontend failed to start")
            return False
        
        print(f"\n✅ Frontend started successfully!")
        print(f"🌐 Open your browser to: http://localhost:{FRONTEND_PORT}")
        print(f"\n📋 Frontend Features:")
        print("  - Supply Chain Analytics Dashboard")
        print("  - Natural Language Query Interface")
        print("  - Data Visualizations")
        print("  - Multi-tier Supply Chain Network View")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Monitor the process output
        while True:
            output = frontend_process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if process is still running
            if frontend_process.poll() is not None:
                print(f"\n⚠️ Frontend process stopped with exit code {frontend_process.returncode}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down frontend server...")
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
        print("✅ Frontend server stopped")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown requested...")
    if frontend_process:
        frontend_process.terminate()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)

def main():
    try:
        print_header("🚀 Supply Chain Frontend Startup")
        print("This will start only the frontend application")
        print("Backend API will not be available\n")
        
        # Check frontend setup
        if not check_frontend_setup():
            print("\n❌ Frontend setup issues detected")
            return 1
        
        # Install dependencies
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            return 1
        
        # Start frontend
        if not start_frontend():
            print("\n❌ Failed to start frontend")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())