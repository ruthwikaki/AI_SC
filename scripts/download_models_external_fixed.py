#!/usr/bin/env python3
"""
Model download script for Supply Chain LLM - External Model Storage
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import json
import platform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_download')

# Default model directory based on OS
def get_default_model_dir():
    """Get default model directory based on operating system"""
    system = platform.system()
    
    if system == "Windows":
        # Try different locations in order of preference
        possible_locations = [
            os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'SupplyChainLLM', 'models'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'SupplyChainLLM', 'models'),
            os.path.join('C:\\', 'SupplyChainLLM', 'models'),
        ]
        
        # Return the first location that we can create
        for location in possible_locations:
            try:
                os.makedirs(location, exist_ok=True)
                return location
            except:
                continue
                
        # Fallback to current directory
        return os.path.join(os.getcwd(), 'models')
    
    elif system == "Darwin":  # macOS
        return os.path.expanduser('~/Library/Application Support/SupplyChainLLM/models')
    else:  # Linux and others
        return os.path.expanduser('~/.local/share/SupplyChainLLM/models')

def validate_directory(path):
    """Validate and create directory if possible"""
    try:
        # Expand user path
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        
        # Check if parent directory exists
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            # For Windows, check if drive exists
            if platform.system() == "Windows":
                drive = os.path.splitdrive(path)[0]
                if drive and not os.path.exists(drive + '\\'):
                    raise ValueError(f"Drive {drive} does not exist")
        
        # Try to create the directory
        os.makedirs(path, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(path, '.test_write')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except:
            raise ValueError(f"No write permission for directory: {path}")
        
        return path
        
    except Exception as e:
        raise ValueError(f"Cannot use directory {path}: {str(e)}")

# Model registry
MODEL_REGISTRY = {
    'mistral-7b-gguf': {
        'description': 'Mistral 7B Instruct GGUF (Quantized, 4.37GB)',
        'files': {
            'model': {
                'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
                'filename': 'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
                'size': '4.37GB'
            }
        }
    },
    'phi-3-mini': {
        'description': 'Microsoft Phi-3 Mini GGUF (Lightweight, 2.39GB)',
        'files': {
            'model': {
                'url': 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
                'filename': 'Phi-3-mini-4k-instruct-q4.gguf',
                'size': '2.39GB'
            }
        }
    },
    'tinyllama': {
        'description': 'TinyLlama 1.1B Chat GGUF (Very lightweight, 669MB)',
        'files': {
            'model': {
                'url': 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
                'filename': 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
                'size': '669MB'
            }
        }
    },
    'deepseek-coder': {
        'description': 'DeepSeek Coder 1.3B GGUF (Good for SQL, 888MB)',
        'files': {
            'model': {
                'url': 'https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf',
                'filename': 'deepseek-coder-1.3b-instruct.Q4_K_M.gguf',
                'size': '888MB'
            }
        }
    }
}

def get_available_space(path):
    """Get available space in GB for a given path"""
    try:
        if platform.system() == 'Windows':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path),
                ctypes.pointer(free_bytes),
                ctypes.pointer(total_bytes),
                None
            )
            return free_bytes.value / (1024**3)  # Convert to GB
        else:
            stat = os.statvfs(path)
            return (stat.f_bavail * stat.f_frsize) / (1024**3)  # Convert to GB
    except:
        return None

def download_file(url, target_path):
    """Download a file with progress bar."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(target_path):
            logger.info(f"File already exists: {target_path}")
            return True
        
        logger.info(f"Downloading: {url}")
        logger.info(f"Target: {target_path}")
        
        # Check available space
        available_gb = get_available_space(os.path.dirname(target_path))
        if available_gb:
            logger.info(f"Available space: {available_gb:.2f} GB")
        
        # Download with streaming
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        total_size_gb = total_size / (1024**3)
        
        # Check if we have enough space (with 1GB buffer)
        if available_gb and total_size_gb > (available_gb - 1):
            raise ValueError(f"Not enough space. Need {total_size_gb:.2f} GB, have {available_gb:.2f} GB")
        
        block_size = 1024 * 1024  # 1 MB chunks
        
        # Download to temp file first
        temp_path = target_path + '.downloading'
        
        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=os.path.basename(target_path)) as pbar:
                for data in response.iter_content(block_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))
        
        # Move temp file to final location
        os.rename(temp_path, target_path)
        logger.info(f"Download completed: {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        # Clean up temp file if it exists
        temp_path = target_path + '.downloading'
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def create_model_config(model_name, model_info, model_dir):
    """Create configuration file for the model."""
    model_file = model_info['files']['model']['filename']
    model_path = os.path.join(model_dir, model_name, model_file)
    
    config = {
        "model_type": "gguf",
        "model_name": model_name,
        "model_path": os.path.abspath(model_path),
        "filename": model_file,
        "description": model_info['description'],
        "size": model_info['files']['model']['size'],
        "context_length": 4096,
        "max_tokens": 2048,
        "temperature": 0.1,
        "format": "gguf"
    }
    
    config_path = os.path.join(model_dir, model_name, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created config: {config_path}")
    return config_path

def create_env_file(model_name, model_dir):
    """Create or update .env.models file with model paths."""
    model_file = MODEL_REGISTRY[model_name]['files']['model']['filename']
    model_path = os.path.join(model_dir, model_name, model_file)
    
    env_path = os.path.join(os.path.dirname(__file__), '..', 'backend', '.env.models')
    
    content = f"""# Model Configuration - Auto-generated
# This file contains paths to models stored outside the codebase

# Model Directory
MODEL_BASE_DIR={model_dir}

# Active Model
DEFAULT_MODEL={model_name}
MODEL_PATH={os.path.abspath(model_path)}

# Available Models
"""
    
    # Add all downloaded models
    for name in MODEL_REGISTRY:
        model_file = MODEL_REGISTRY[name]['files']['model']['filename']
        path = os.path.join(model_dir, name, model_file)
        if os.path.exists(path):
            content += f"{name.upper().replace('-', '_')}_PATH={os.path.abspath(path)}\n"
    
    with open(env_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created/Updated: {env_path}")
    return env_path

def suggest_locations():
    """Suggest good locations for model storage"""
    suggestions = []
    
    if platform.system() == "Windows":
        # Check each drive
        for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                space = get_available_space(drive)
                if space and space > 5:  # At least 5GB free
                    suggestions.append({
                        'path': f"{letter}:\\LLM_Models\\SupplyChain",
                        'space': space
                    })
        
        # Add user directories
        user_dirs = [
            os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'LLM_Models'),
            os.path.join(os.environ.get('USERPROFILE', ''), 'LLM_Models'),
        ]
        
        for dir_path in user_dirs:
            space = get_available_space(dir_path)
            if space:
                suggestions.append({
                    'path': dir_path,
                    'space': space
                })
    
    return suggestions

def main():
    parser = argparse.ArgumentParser(description='Download LLM models to external directory.')
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()) + ['list'], 
                       default='tinyllama',
                       help='Model to download (default: tinyllama)')
    parser.add_argument('--model-dir', 
                       default=None,
                       help=f'Model directory (default: auto-detect)')
    parser.add_argument('--suggest-locations', action='store_true',
                       help='Suggest good locations for model storage')
    args = parser.parse_args()
    
    # Suggest locations
    if args.suggest_locations:
        print("\nSuggested model storage locations:")
        print("-" * 70)
        suggestions = suggest_locations()
        if suggestions:
            for s in sorted(suggestions, key=lambda x: x['space'], reverse=True):
                print(f"{s['path']:<50} {s['space']:>10.2f} GB free")
        else:
            print("No suitable locations found with >5GB free space")
        return
    
    # Determine model directory
    if args.model_dir:
        try:
            model_dir = validate_directory(args.model_dir)
        except ValueError as e:
            logger.error(str(e))
            print("\nSuggested alternatives:")
            for s in suggest_locations()[:3]:
                print(f"  --model-dir \"{s['path']}\" ({s['space']:.1f} GB free)")
            return 1
    else:
        model_dir = get_default_model_dir()
        logger.info(f"Using default model directory: {model_dir}")
    
    # List models
    if args.model == 'list':
        print(f"\nModel Directory: {model_dir}")
        space = get_available_space(model_dir)
        if space:
            print(f"Available Space: {space:.2f} GB")
        print("\nAvailable Models:")
        print("-" * 70)
        for name, info in MODEL_REGISTRY.items():
            model_path = os.path.join(model_dir, name, info['files']['model']['filename'])
            status = "✓ Downloaded" if os.path.exists(model_path) else "✗ Not downloaded"
            print(f"\n{name}: {status}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['files']['model']['size']}")
        print(f"\nTotal models: {len(MODEL_REGISTRY)}")
        return
    
    # Create model directory
    model_subdir = os.path.join(model_dir, args.model)
    os.makedirs(model_subdir, exist_ok=True)
    
    # Download selected model
    model_info = MODEL_REGISTRY[args.model]
    print(f"\nSelected: {model_info['description']}")
    print(f"Location: {model_subdir}")
    
    # Download model file
    model_file = model_info['files']['model']['filename']
    target_path = os.path.join(model_subdir, model_file)
    
    if download_file(model_info['files']['model']['url'], target_path):
        # Create config
        config_path = create_model_config(args.model, model_info, model_dir)
        
        # Create/update .env.models
        env_path = create_env_file(args.model, model_dir)
        
        print(f"""
✅ Model downloaded successfully!

Model stored at: {target_path}

To use this model:

1. The .env.models file has been created/updated at:
   {env_path}

2. Your backend will automatically load this model.

3. Restart your backend server.

The model is stored outside your codebase and won't be committed to git!
""")
        return 0
    else:
        print(f"\n❌ Download failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
