#!/usr/bin/env python3
"""
Quantize script for the Supply Chain LLM system.

This script is a wrapper around model_quantizer.py, providing a simple
interface for quantizing models to reduce memory usage and improve inference speed.
"""

import os
import sys
import argparse
import logging
import importlib.util
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quantize")

def import_model_quantizer(script_path):
    """
    Dynamically import the ModelQuantizer class from model_quantizer.py.
    
    Args:
        script_path: Path to model_quantizer.py
        
    Returns:
        Imported ModelQuantizer class
    """
    try:
        # Get the absolute path
        abs_path = Path(script_path).resolve()
        
        # Check if file exists
        if not abs_path.exists():
            raise FileNotFoundError(f"Script not found: {abs_path}")
        
        # Create a spec from the file
        spec = importlib.util.spec_from_file_location("model_quantizer_module", abs_path)
        if spec is None:
            raise ImportError(f"Could not create module spec from {abs_path}")
        
        # Create a module from the spec
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise ImportError(f"Could not create module from spec for {abs_path}")
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Return the ModelQuantizer class
        if not hasattr(module, "ModelQuantizer"):
            raise AttributeError(f"Module {abs_path} has no attribute 'ModelQuantizer'")
        
        return module.ModelQuantizer
    
    except Exception as e:
        logger.error(f"Error importing ModelQuantizer: {str(e)}")
        sys.exit(1)

def find_model_quantizer_script():
    """
    Find the model_quantizer.py script in the same directory as this script
    or in the parent's scripts directory.
    
    Returns:
        Path to model_quantizer.py
    """
    # Try current directory
    current_dir = Path(__file__).parent
    script_path = current_dir / "model_quantizer.py"
    
    if script_path.exists():
        return script_path
    
    # Try parent's scripts directory
    parent_scripts_dir = current_dir.parent / "scripts"
    script_path = parent_scripts_dir / "model_quantizer.py"
    
    if script_path.exists():
        return script_path
    
    # Try sibling directory
    sibling_dir = current_dir.parent / "scripts"
    script_path = sibling_dir / "model_quantizer.py"
    
    if script_path.exists():
        return script_path
    
    # If not found, raise an error
    raise FileNotFoundError("Could not find model_quantizer.py script")

def main():
    """Main function to run the model quantizer wrapper."""
    parser = argparse.ArgumentParser(description="Quantize LLM models for reduced memory usage")
    
    # Main arguments
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing models")
    parser.add_argument("--model", type=str, help="Model to quantize (e.g., mistral, llama3)")
    parser.add_argument("--precision", type=str, default="int8", choices=["int8", "int4"], 
                        help="Precision to quantize to")
    parser.add_argument("--method", type=str, default="gptq", choices=["gptq", "awq"],
                        help="Quantization method to use")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    # Advanced arguments
    parser.add_argument("--script-path", type=str, help="Path to model_quantizer.py script")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find the model_quantizer.py script
        script_path = args.script_path
        if script_path is None:
            try:
                script_path = find_model_quantizer_script()
                logger.debug(f"Found model_quantizer.py at: {script_path}")
            except FileNotFoundError as e:
                logger.error(f"Error: {str(e)}")
                logger.error("Please provide path to model_quantizer.py using --script-path")
                sys.exit(1)
        
        # Import the ModelQuantizer class
        ModelQuantizer = import_model_quantizer(script_path)
        
        # Create quantizer instance
        quantizer = ModelQuantizer(args.models_dir)
        
        if args.list:
            # List available models
            models = quantizer.get_models_list()
            print("\nAvailable Models:")
            print("=" * 80)
            for model in models:
                print(f"- {model['name']}")
                print(f"  Path: {model['path']}")
                print(f"  Size: {model['size_mb']} MB")
                
                if "parameters" in model:
                    print(f"  Parameters: {model['parameters']}")
                
                print()
        elif args.model:
            # Quantize the specified model
            success = quantizer.quantize_model(
                model_name=args.model,
                precision=args.precision,
                method=args.method
            )
            
            if success:
                print(f"Successfully quantized {args.model} to {args.precision} using {args.method}")
            else:
                print(f"Failed to quantize {args.model}")
                sys.exit(1)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()