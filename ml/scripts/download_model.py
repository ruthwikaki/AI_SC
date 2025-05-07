#!/usr/bin/env python3
"""
Download script for the Supply Chain LLM system.

This script downloads LLM models and weights from various sources and sets up
the directory structure for the supply chain analytics system.
"""

import os
import sys
import argparse
import logging
import json
import hashlib
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_downloader")

# Model repository information
MODEL_REGISTRY = {
    "mistral-7b": {
        "description": "Mistral 7B base model (fp16)",
        "url": "https://models.example.com/mistral/mistral-7b-v0.1.tar.gz",
        "sha256": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c",
        "size_gb": 13.5,
        "type": "mistral",
        "variants": ["int8", "int4"],
    },
    "mistral-7b-instruct": {
        "description": "Mistral 7B instruction-tuned model (fp16)",
        "url": "https://models.example.com/mistral/mistral-7b-instruct-v0.1.tar.gz",
        "sha256": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d",
        "size_gb": 13.5,
        "type": "mistral",
        "variants": ["int8", "int4"],
    },
    "llama3-8b": {
        "description": "LLaMA3 8B base model (fp16)",
        "url": "https://models.example.com/llama3/llama3-8b.tar.gz",
        "sha256": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
        "size_gb": 15.8,
        "type": "llama3",
        "variants": ["int8", "int4"],
    },
    "llama3-8b-instruct": {
        "description": "LLaMA3 8B instruction-tuned model (fp16)",
        "url": "https://models.example.com/llama3/llama3-8b-instruct.tar.gz",
        "sha256": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
        "size_gb": 15.8,
        "type": "llama3",
        "variants": ["int8", "int4"],
    },
    "supply-chain-finetuned-7b": {
        "description": "Supply chain domain-adapted model (fp16)",
        "url": "https://models.example.com/supply-chain/finetuned-7b-v1.0.tar.gz",
        "sha256": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "size_gb": 13.5,
        "type": "mistral",
        "variants": ["int8", "int4"],
    }
}

class ModelDownloader:
    """Handles downloading and setup of models."""
    
    def __init__(self, models_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize the model downloader.
        
        Args:
            models_dir: Directory to store models
            cache_dir: Directory to cache downloaded files
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "supply-chain-llm"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry information
        self.model_registry = MODEL_REGISTRY
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models in the registry.
        
        Returns:
            List of model information
        """
        return [
            {
                "name": name,
                **info
            }
            for name, info in self.model_registry.items()
        ]
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """
        List models that have been downloaded.
        
        Returns:
            List of downloaded model information
        """
        downloaded = []
        
        for model_type in ["mistral", "llama3"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue
                
            # Check for model weights directory
            weights_dir = type_dir / "weights"
            if weights_dir.exists():
                # Find matching model in registry
                model_info = None
                for name, info in self.model_registry.items():
                    if info["type"] == model_type:
                        # This is a potential match, check for more specific info
                        config_path = type_dir / "config.json"
                        if config_path.exists():
                            try:
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                
                                # If we have a model_id in config, use it for exact matching
                                if "model_id" in config and config["model_id"] == name:
                                    model_info = {"name": name, **info}
                                    break
                            except Exception as e:
                                logger.warning(f"Error reading config: {str(e)}")
                
                # If we didn't find a match, use basic info
                if not model_info:
                    model_info = {
                        "name": f"{model_type} (unknown variant)",
                        "type": model_type,
                        "description": "Unknown model variant",
                    }
                
                # Add path information
                model_info["path"] = str(weights_dir)
                
                # Check for quantized versions
                quantized_variants = []
                for quant_dir in weights_dir.glob("quantized_*"):
                    if quant_dir.is_dir():
                        variant_name = quant_dir.name.replace("quantized_", "")
                        quantized_variants.append(variant_name)
                
                if quantized_variants:
                    model_info["quantized_variants"] = quantized_variants
                
                downloaded.append(model_info)
        
        return downloaded
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download a model from the registry.
        
        Args:
            model_name: Name of the model to download
            force: Force download even if already exists
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists in registry
        if model_name not in self.model_registry:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        model_info = self.model_registry[model_name]
        model_type = model_info["type"]
        model_url = model_info["url"]
        model_hash = model_info["sha256"]
        model_size_gb = model_info["size_gb"]
        
        # Determine target directory
        target_dir = self.models_dir / model_type
        weights_dir = target_dir / "weights"
        
        # Check if model already exists
        if weights_dir.exists() and not force:
            logger.info(f"Model {model_type} already exists at {weights_dir}. Use --force to redownload.")
            return True
        
        # Determine cache path
        url_filename = model_url.split("/")[-1]
        cache_path = self.cache_dir / url_filename
        
        # Download if not in cache or force download
        if not cache_path.exists() or force:
            logger.info(f"Downloading {model_name} ({model_size_gb:.1f} GB)...")
            try:
                self._download_file(model_url, cache_path)
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                return False
        else:
            logger.info(f"Using cached download at {cache_path}")
        
        # Verify hash
        if not self._verify_file_hash(cache_path, model_hash):
            logger.error(f"Hash verification failed for {cache_path}")
            return False
        
        # Extract model
        logger.info(f"Extracting model to {target_dir}...")
        try:
            self._extract_archive(cache_path, target_dir)
        except Exception as e:
            logger.error(f"Error extracting model: {str(e)}")
            return False
        
        # Create or update config file
        config_path = target_dir / "config.json"
        self._create_model_config(config_path, model_name, model_info)
        
        logger.info(f"Successfully downloaded and set up {model_name}")
        return True
    
    def _download_file(self, url: str, output_path: Path) -> None:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download
            output_path: Path to save the file
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def _verify_file_hash(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file hash matches expected value.
        
        Args:
            file_path: Path to file
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if hash matches, False otherwise
        """
        logger.info(f"Verifying hash for {file_path}...")
        
        sha256_hash = hashlib.sha256()
        block_size = 1024 * 1024  # 1 MB
        
        with tqdm(total=file_path.stat().st_size, unit='B', unit_scale=True, desc="Verifying") as pbar:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(block_size), b''):
                    sha256_hash.update(chunk)
                    pbar.update(len(chunk))
        
        file_hash = sha256_hash.hexdigest()
        
        if file_hash != expected_hash:
            logger.error(f"Hash mismatch: {file_hash} != {expected_hash}")
            return False
        
        return True
    
    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        """
        Extract archive file to target directory.
        
        Args:
            archive_path: Path to archive file
            target_dir: Directory to extract to
        """
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine archive type and extract
        if archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Get total size for progress bar
                total_size = sum(m.size for m in tar.getmembers())
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for member in tar.getmembers():
                        tar.extract(member, path=target_dir)
                        pbar.update(member.size)
        
        elif archive_path.name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get total size for progress bar
                total_size = sum(info.file_size for info in zip_ref.infolist())
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for info in zip_ref.infolist():
                        zip_ref.extract(info, path=target_dir)
                        pbar.update(info.file_size)
        
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        # Check if extracted to a subdirectory
        subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "weights").exists():
            # Move contents up one level
            source_dir = subdirs[0]
            for item in source_dir.iterdir():
                shutil.move(str(item), str(target_dir / item.name))
            
            # Remove empty directory
            source_dir.rmdir()
    
    def _create_model_config(
        self, 
        config_path: Path, 
        model_name: str, 
        model_info: Dict[str, Any]
    ) -> None:
        """
        Create or update model configuration file.
        
        Args:
            config_path: Path to configuration file
            model_name: Name of the model
            model_info: Model information
        """
        # Load existing config if it exists
        config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading existing config: {str(e)}")
        
        # Update with model information
        config.update({
            "model_id": model_name,
            "model_type": model_info["type"],
            "description": model_info["description"],
            "model_parameters": {
                "size": f"{model_info['size_gb']:.1f}B"
            },
            "download_timestamp": import_time.strftime("%Y-%m-%d %H:%M:%S"),
            "supported_quantization": model_info.get("variants", [])
        })
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created/updated config at {config_path}")


def main():
    """Main function to run the model downloader."""
    parser = argparse.ArgumentParser(description="Download LLM models for the supply chain system")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory to store models")
    parser.add_argument("--cache-dir", type=str, help="Directory to cache downloaded files")
    parser.add_argument("--model", type=str, help="Model to download")
    parser.add_argument("--force", action="store_true", help="Force download even if already exists")
    parser.add_argument("--list-available", action="store_true", help="List available models")
    parser.add_argument("--list-downloaded", action="store_true", help="List downloaded models")
    parser.add_argument("--list-all", action="store_true", help="List all models with download status")
    
    args = parser.parse_args()
    
    try:
        downloader = ModelDownloader(args.models_dir, args.cache_dir)
        
        if args.list_available:
            # List available models
            models = downloader.list_available_models()
            print("\nAvailable Models:")
            print("=" * 80)
            for model in models:
                print(f"- {model['name']}")
                print(f"  Description: {model['description']}")
                print(f"  Size: {model['size_gb']:.1f} GB")
                print(f"  Supported quantization: {', '.join(model.get('variants', []))}")
                print()
        
        elif args.list_downloaded:
            # List downloaded models
            models = downloader.list_downloaded_models()
            if not models:
                print("\nNo models downloaded yet.")
                return
                
            print("\nDownloaded Models:")
            print("=" * 80)
            for model in models:
                print(f"- {model['name']}")
                if "description" in model:
                    print(f"  Description: {model['description']}")
                print(f"  Path: {model['path']}")
                if "quantized_variants" in model:
                    print(f"  Quantized variants: {', '.join(model['quantized_variants'])}")
                print()
        
        elif args.list_all:
            # List all models with download status
            available = {model["name"]: model for model in downloader.list_available_models()}
            downloaded = {model["name"]: model for model in downloader.list_downloaded_models()}
            
            print("\nAll Models:")
            print("=" * 80)
            for name, model in available.items():
                status = "Downloaded" if name in downloaded else "Not downloaded"
                print(f"- {name}")
                print(f"  Description: {model['description']}")
                print(f"  Size: {model['size_gb']:.1f} GB")
                print(f"  Status: {status}")
                if name in downloaded and "quantized_variants" in downloaded[name]:
                    print(f"  Quantized variants: {', '.join(downloaded[name]['quantized_variants'])}")
                print()
        
        elif args.model:
            # Download the specified model
            success = downloader.download_model(args.model, args.force)
            
            if success:
                print(f"Successfully downloaded {args.model}")
            else:
                print(f"Failed to download {args.model}")
                sys.exit(1)
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import time  # For timestamp
    main()