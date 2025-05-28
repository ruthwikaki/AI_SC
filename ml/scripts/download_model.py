#!/usr/bin/env python3
"""
Download script for the Supply Chain LLM system.

This script downloads LLM models from Hugging Face and sets up
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
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_downloader")

# Model repository information - Updated for Hugging Face
MODEL_REGISTRY = {
    "mistral-7b": {
        "description": "Mistral 7B base model (fp16)",
        "source": "huggingface",
        "hf_repo": "mistralai/Mistral-7B-v0.1",
        "size_gb": 13.5,
        "type": "mistral",
        "requires_token": False,
        "variants": ["int8", "int4"],
    },
    "mistral-7b-instruct": {
        "description": "Mistral 7B instruction-tuned model (fp16)",
        "source": "huggingface",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.1",
        "size_gb": 13.5,
        "type": "mistral",
        "requires_token": False,
        "variants": ["int8", "int4"],
    },
    "llama3-8b": {
        "description": "LLaMA3 8B base model (fp16)",
        "source": "huggingface",
        "hf_repo": "meta-llama/Meta-Llama-3-8B",
        "size_gb": 15.8,
        "type": "llama3",
        "requires_token": True,
        "variants": ["int8", "int4"],
    },
    "llama3-8b-instruct": {
        "description": "LLaMA3 8B instruction-tuned model (fp16)",
        "source": "huggingface",
        "hf_repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "size_gb": 15.8,
        "type": "llama3",
        "requires_token": True,
        "variants": ["int8", "int4"],
    },
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
        
        # Try to import huggingface_hub
        try:
            from huggingface_hub import snapshot_download
            self.hf_available = True
            self.snapshot_download = snapshot_download
        except ImportError:
            self.hf_available = False
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface-hub")
    
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
            if weights_dir.exists() and any(weights_dir.iterdir()):
                # Find matching model in registry
                model_info = None
                for name, info in self.model_registry.items():
                    if info["type"] == model_type:
                        # Check for config.json or model files
                        config_path = weights_dir / "config.json"
                        model_files = list(weights_dir.glob("*.safetensors")) + list(weights_dir.glob("*.bin"))
                        
                        if config_path.exists() or model_files:
                            model_info = {"name": name, **info}
                            break
                
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
    
    def download_model(self, model_name: str, force: bool = False, hf_token: Optional[str] = None) -> bool:
        """
        Download a model from the registry.
        
        Args:
            model_name: Name of the model to download
            force: Force download even if already exists
            hf_token: Hugging Face token (required for some models)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists in registry
        if model_name not in self.model_registry:
            logger.error(f"Model {model_name} not found in registry")
            logger.info("Available models: " + ", ".join(self.model_registry.keys()))
            return False
        
        model_info = self.model_registry[model_name]
        model_type = model_info["type"]
        
        # Determine target directory
        target_dir = self.models_dir / model_type / "weights"
        
        # Check if model already exists
        if target_dir.exists() and any(target_dir.iterdir()) and not force:
            logger.info(f"Model {model_name} already exists at {target_dir}. Use --force to redownload.")
            return True
        
        # Create directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download based on source
        if model_info.get("source") == "huggingface":
            return self._download_from_huggingface(model_name, model_info, target_dir, hf_token)
        else:
            logger.error(f"Unknown source for model {model_name}")
            return False
    
    def _download_from_huggingface(
        self, 
        model_name: str, 
        model_info: Dict[str, Any], 
        target_dir: Path,
        hf_token: Optional[str] = None
    ) -> bool:
        """
        Download model from Hugging Face.
        
        Args:
            model_name: Name of the model
            model_info: Model information from registry
            target_dir: Directory to save model
            hf_token: Hugging Face token
            
        Returns:
            True if successful, False otherwise
        """
        if not self.hf_available:
            logger.error("huggingface_hub is not installed. Install with: pip install huggingface-hub")
            return False
        
        hf_repo = model_info["hf_repo"]
        requires_token = model_info.get("requires_token", False)
        
        # Check if token is required
        if requires_token and not hf_token:
            logger.error(f"Model {model_name} requires a Hugging Face token.")
            logger.info("Please provide a token with --hf-token or set HF_TOKEN environment variable")
            logger.info(f"Visit https://huggingface.co/{hf_repo} to accept the license")
            logger.info("Get your token from https://huggingface.co/settings/tokens")
            return False
        
        logger.info(f"Downloading {model_name} from Hugging Face ({model_info['size_gb']:.1f} GB)...")
        logger.info(f"Repository: {hf_repo}")
        
        try:
            # Download the model
            self.snapshot_download(
                repo_id=hf_repo,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token if requires_token else None
            )
            
            # Create model info file
            info_path = target_dir.parent / "model_info.json"
            model_metadata = {
                "model_id": model_name,
                "model_type": model_info["type"],
                "description": model_info["description"],
                "source": "huggingface",
                "repository": hf_repo,
                "download_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "size_gb": model_info["size_gb"],
                "supported_quantization": model_info.get("variants", [])
            }
            
            with open(info_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info(f"✓ Successfully downloaded {model_name} to {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error downloading model: {str(e)}")
            if "401" in str(e) or "403" in str(e):
                logger.error("Authentication failed. Please check your token.")
                logger.info(f"Make sure you have accepted the license at https://huggingface.co/{hf_repo}")
            return False
    
    def get_model_size(self, model_name: str) -> Optional[float]:
        """
        Get the size of a model in GB.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Size in GB or None if not found
        """
        if model_name in self.model_registry:
            return self.model_registry[model_name].get("size_gb")
        return None


def main():
    """Main function to run the model downloader."""
    parser = argparse.ArgumentParser(description="Download LLM models for the supply chain system")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory to store models")
    parser.add_argument("--cache-dir", type=str, help="Directory to cache downloaded files")
    parser.add_argument("--model", type=str, help="Model to download")
    parser.add_argument("--force", action="store_true", help="Force download even if already exists")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--list-available", action="store_true", help="List available models")
    parser.add_argument("--list-downloaded", action="store_true", help="List downloaded models")
    parser.add_argument("--list-all", action="store_true", help="List all models with download status")
    
    args = parser.parse_args()
    
    # Check for HF_TOKEN environment variable if not provided
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN")
    
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
                print(f"  Source: {model.get('source', 'custom')}")
                if model.get('requires_token'):
                    print(f"  ⚠️  Requires Hugging Face token")
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
                status = "✓ Downloaded" if name in downloaded else "✗ Not downloaded"
                print(f"- {name} [{status}]")
                print(f"  Description: {model['description']}")
                print(f"  Size: {model['size_gb']:.1f} GB")
                if model.get('requires_token'):
                    print(f"  ⚠️  Requires Hugging Face token")
                if name in downloaded and "quantized_variants" in downloaded[name]:
                    print(f"  Quantized variants: {', '.join(downloaded[name]['quantized_variants'])}")
                print()
        
        elif args.model:
            # Download the specified model
            success = downloader.download_model(args.model, args.force, args.hf_token)
            
            if success:
                print(f"\n✓ Successfully downloaded {args.model}")
                print(f"You can now use this model in your Supply Chain LLM system")
            else:
                print(f"\n✗ Failed to download {args.model}")
                sys.exit(1)
        
        else:
            parser.print_help()
            print("\nExamples:")
            print("  # List available models")
            print("  python download_model.py --list-available")
            print()
            print("  # Download Mistral (no token needed)")
            print("  python download_model.py --model mistral-7b")
            print()
            print("  # Download Llama3 (requires token)")
            print("  python download_model.py --model llama3-8b --hf-token YOUR_TOKEN")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()