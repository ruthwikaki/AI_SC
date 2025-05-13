#!/usr/bin/env python3
"""
Model download script for Supply Chain LLM.
This script downloads LLM models and tokenizers needed for inference.
"""

import argparse
import os
import sys
import yaml
import logging
import hashlib
from pathlib import Path
import requests
from tqdm import tqdm

# Add the ML directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_download')

# Model registry - in production, this would likely be stored in a configuration file
MODEL_REGISTRY = {
    'mistral': {
        'weights': {
            'url': 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model.bin',
            'sha256': '...',  # Add the actual hash here
            'target_path': 'mistral/weights/pytorch_model.bin'
        },
        'tokenizer': {
            'url': 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model',
            'sha256': '...',  # Add the actual hash here
            'target_path': 'tokenizers/mistral.model'
        },
        'config': {
            'url': 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json',
            'sha256': '...',  # Add the actual hash here
            'target_path': 'mistral/config.json'
        }
    },
    'llama3': {
        'weights': {
            'url': 'https://example.com/llama3-7b.bin',  # Replace with actual URL
            'sha256': '...',  # Add the actual hash here
            'target_path': 'llama3/weights/pytorch_model.bin'
        },
        'tokenizer': {
            'url': 'https://example.com/llama3-tokenizer.model',  # Replace with actual URL
            'sha256': '...',  # Add the actual hash here
            'target_path': 'tokenizers/llama3.model'
        },
        'config': {
            'url': 'https://example.com/llama3-config.json',  # Replace with actual URL
            'sha256': '...',  # Add the actual hash here
            'target_path': 'llama3/config.json'
        }
    }
}

def verify_hash(file_path, expected_hash):
    """Verify the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    computed_hash = sha256_hash.hexdigest()
    
    if computed_hash != expected_hash:
        logger.warning(f"Hash mismatch for {file_path}:")
        logger.warning(f"  Expected: {expected_hash}")
        logger.warning(f"  Computed: {computed_hash}")
        return False
    
    return True

def download_file(url, target_path, expected_hash=None):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunks
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(target_path)) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        
        if expected_hash:
            if not verify_hash(target_path, expected_hash):
                logger.error(f"Hash verification failed for {target_path}. File may be corrupted.")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download LLM models for Supply Chain LLM.')
    parser.add_argument('--output-dir', default='./ml/models', help='Output directory for models')
    parser.add_argument('--models', nargs='+', choices=['mistral', 'llama3', 'all'], default=['all'], help='Models to download')
    parser.add_argument('--skip-verification', action='store_true', help='Skip hash verification')
    args = parser.parse_args()

    # Determine which models to download
    models_to_download = list(MODEL_REGISTRY.keys()) if 'all' in args.models else args.models
    
    # Download each model
    for model in models_to_download:
        logger.info(f"Downloading {model} model files...")
        
        for component, info in MODEL_REGISTRY[model].items():
            target_path = os.path.join(args.output_dir, info['target_path'])
            
            if os.path.exists(target_path):
                logger.info(f"{model} {component} already exists at {target_path}")
                
                if not args.skip_verification and 'sha256' in info:
                    if verify_hash(target_path, info['sha256']):
                        logger.info(f"Hash verification successful for {target_path}")
                    else:
                        logger.warning(f"Hash verification failed for {target_path}. Re-downloading...")
                        download_file(info['url'], target_path, None if args.skip_verification else info.get('sha256'))
            else:
                logger.info(f"Downloading {model} {component} to {target_path}")
                download_file(info['url'], target_path, None if args.skip_verification else info.get('sha256'))
    
    logger.info("Model download completed successfully")

if __name__ == "__main__":
    main()