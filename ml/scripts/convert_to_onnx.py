#!/usr/bin/env python3
"""
ONNX Conversion script for the Supply Chain LLM system.

This script converts PyTorch models to ONNX format for optimized inference
in production environments. ONNX enables faster inference on various hardware
and integration with TensorRT for additional speedups.
"""

import os
import sys
import argparse
import logging
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("onnx_converter")

class ONNXConverter:
    """Converts PyTorch models to ONNX format for optimized inference."""
    
    def __init__(self, models_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the ONNX converter.
        
        Args:
            models_dir: Directory containing model files
            output_dir: Directory to save converted models
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir) if output_dir else self.models_dir / "onnx"
        
        # Check if directory exists
        if not self.models_dir.exists():
            raise ValueError(f"Models directory does not exist: {models_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_model(
        self, 
        model_name: str, 
        precision: str = "fp16",
        opset_version: int = 15,
        max_batch_size: int = 1,
        max_sequence_length: int = 2048,
        optimize: bool = True,
        quantized: bool = False,
        quantization_type: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> bool:
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            model_name: Name of the model to convert
            precision: Precision for conversion (fp32, fp16, int8)
            opset_version: ONNX opset version
            max_batch_size: Maximum batch size for dynamic axes
            max_sequence_length: Maximum sequence length for dynamic axes
            optimize: Whether to optimize the ONNX model
            quantized: Whether the source model is already quantized
            quantization_type: Type of quantization (gptq, awq)
            cache_dir: Cache directory for HuggingFace models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine model directory and type
            if model_name.startswith("mistral"):
                model_type = "mistral"
                model_dir = self.models_dir / "mistral" / "weights"
                config_path = self.models_dir / "mistral" / "config.json"
            elif model_name.startswith("llama3"):
                model_type = "llama3"
                model_dir = self.models_dir / "llama3" / "weights"
                config_path = self.models_dir / "llama3" / "config.json"
            else:
                logger.error(f"Unsupported model type: {model_name}")
                return False
            
            # If the model is quantized, adjust the path
            if quantized and quantization_type:
                model_dir = model_dir / f"quantized_{precision}_{quantization_type}"
                if not model_dir.exists():
                    logger.error(f"Quantized model directory not found: {model_dir}")
                    return False
            
            # Determine output directory for this model
            if quantized:
                output_model_dir = self.output_dir / model_type / f"quantized_{precision}_{quantization_type}"
            else:
                output_model_dir = self.output_dir / model_type / precision
            
            output_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if output files already exist
            output_model_path = output_model_dir / "model.onnx"
            if output_model_path.exists():
                logger.info(f"ONNX model already exists at {output_model_path}. Skipping.")
                return True
            
            # Log start
            logger.info(f"Converting {model_name} to ONNX format with {precision} precision...")
            start_time = time.time()
            
            # Import dependencies here to avoid loading them unless needed
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            
            # Load model configuration
            logger.info(f"Loading model configuration from {model_dir}...")
            config = AutoConfig.from_pretrained(str(model_dir))
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                cache_dir=cache_dir
            )
            
            # Determine torch dtype
            if precision == "fp16":
                torch_dtype = torch.float16
            elif precision == "fp32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32  # Default to fp32
            
            # Load model based on quantization status
            if quantized:
                logger.info(f"Loading quantized model from {model_dir}...")
                if quantization_type == "gptq":
                    from auto_gptq import AutoGPTQForCausalLM
                    model = AutoGPTQForCausalLM.from_quantized(
                        str(model_dir),
                        use_safetensors=True,
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                elif quantization_type == "awq":
                    try:
                        from awq import AutoAWQForCausalLM
                        model = AutoAWQForCausalLM.from_quantized(
                            str(model_dir),
                            trust_remote_code=True,
                            cache_dir=cache_dir
                        )
                    except ImportError:
                        logger.error("AWQ package not installed. Install with: pip install awq")
                        return False
                else:
                    logger.error(f"Unsupported quantization type: {quantization_type}")
                    return False
            else:
                logger.info(f"Loading model from {model_dir}...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
            
            # Ensure model is in eval mode
            model.eval()
            
            # Create dummy inputs for tracing
            # We need an example batch with the correct shape and data type
            input_ids = torch.ones((max_batch_size, max_sequence_length), dtype=torch.long).to(model.device)
            attention_mask = torch.ones((max_batch_size, max_sequence_length), dtype=torch.bool).to(model.device)
            
            # Define dynamic axes for variable batch size and sequence length
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
            
            # Export model to ONNX
            logger.info(f"Exporting model to ONNX format with opset version {opset_version}...")
            
            # Prepare dummy inputs as a dictionary
            dummy_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            # Export the model to ONNX format
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_inputs,),  # Model inputs as a tuple
                    str(output_model_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False
                )
            
            # Optimize ONNX model if requested
            if optimize:
                logger.info("Optimizing ONNX model...")
                self._optimize_onnx_model(output_model_path, output_model_path)
            
            # Save tokenizer configuration
            logger.info("Saving tokenizer configuration...")
            tokenizer.save_pretrained(str(output_model_dir))
            
            # Update model configuration to reflect ONNX conversion
            self._update_config_for_onnx(
                config_path,
                output_model_dir / "onnx_config.json",
                precision,
                opset_version,
                max_batch_size,
                max_sequence_length,
                optimize,
                quantized,
                quantization_type
            )
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Conversion completed in {elapsed_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {model_name} to ONNX: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _optimize_onnx_model(self, input_path: Path, output_path: Path) -> bool:
        """
        Optimize an ONNX model using ONNX Runtime.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path to save optimized model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
            
            # Load the model
            model = onnx.load(str(input_path))
            
            # Optimize the model
            opt_model = optimizer.optimize_model(
                str(input_path),
                model_type="gpt2",  # Use GPT-2 as the closest model type
                num_heads=16,  # This should be model-specific
                hidden_size=4096,  # This should be model-specific
                optimization_options=None
            )
            
            # Save the optimized model
            opt_model.save_model_to_file(str(output_path))
            
            return True
            
        except ImportError:
            logger.warning("ONNX Runtime not installed. Skipping optimization.")
            return False
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {str(e)}")
            return False
    
    def _update_config_for_onnx(
        self,
        source_config_path: Path,
        output_config_path: Path,
        precision: str,
        opset_version: int,
        max_batch_size: int,
        max_sequence_length: int,
        optimize: bool,
        quantized: bool,
        quantization_type: Optional[str]
    ) -> None:
        """
        Create an ONNX configuration file.
        
        Args:
            source_config_path: Path to source model configuration
            output_config_path: Path to save ONNX configuration
            precision: Precision used for conversion
            opset_version: ONNX opset version
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
            optimize: Whether the model was optimized
            quantized: Whether the source model was quantized
            quantization_type: Type of quantization
        """
        try:
            # Load original config if it exists
            original_config = {}
            if source_config_path.exists():
                with open(source_config_path, 'r') as f:
                    original_config = json.load(f)
            
            # Create ONNX config
            onnx_config = {
                "onnx_conversion": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "precision": precision,
                    "opset_version": opset_version,
                    "max_batch_size": max_batch_size,
                    "max_sequence_length": max_sequence_length,
                    "optimized": optimize
                },
                "model_info": {
                    "source_quantized": quantized
                }
            }
            
            if quantized and quantization_type:
                onnx_config["model_info"]["quantization_type"] = quantization_type
            
            # Include original model parameters if available
            if "model_parameters" in original_config:
                onnx_config["model_parameters"] = original_config["model_parameters"]
            
            # Include model ID if available
            if "model_id" in original_config:
                onnx_config["model_id"] = original_config["model_id"]
            
            # Save configuration
            with open(output_config_path, 'w') as f:
                json.dump(onnx_config, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating ONNX config: {str(e)}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models that can be converted to ONNX.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Check for Mistral models
        mistral_dir = self.models_dir / "mistral"
        if mistral_dir.exists():
            weights_dir = mistral_dir / "weights"
            if weights_dir.exists():
                model_info = {
                    "name": "mistral",
                    "path": str(weights_dir),
                    "onnx_converted": self._check_onnx_conversion("mistral")
                }
                
                # Check for quantized versions
                quantized_variants = []
                for quant_dir in weights_dir.glob("quantized_*"):
                    if quant_dir.is_dir():
                        variant_name = quant_dir.name.replace("quantized_", "")
                        quant_info = {
                            "name": f"mistral-{variant_name}",
                            "path": str(quant_dir),
                            "onnx_converted": self._check_onnx_conversion("mistral", variant_name)
                        }
                        quantized_variants.append(quant_info)
                
                if quantized_variants:
                    model_info["quantized_variants"] = quantized_variants
                
                models.append(model_info)
        
        # Check for LLaMA3 models
        llama3_dir = self.models_dir / "llama3"
        if llama3_dir.exists():
            weights_dir = llama3_dir / "weights"
            if weights_dir.exists():
                model_info = {
                    "name": "llama3",
                    "path": str(weights_dir),
                    "onnx_converted": self._check_onnx_conversion("llama3")
                }
                
                # Check for quantized versions
                quantized_variants = []
                for quant_dir in weights_dir.glob("quantized_*"):
                    if quant_dir.is_dir():
                        variant_name = quant_dir.name.replace("quantized_", "")
                        quant_info = {
                            "name": f"llama3-{variant_name}",
                            "path": str(quant_dir),
                            "onnx_converted": self._check_onnx_conversion("llama3", variant_name)
                        }
                        quantized_variants.append(quant_info)
                
                if quantized_variants:
                    model_info["quantized_variants"] = quantized_variants
                
                models.append(model_info)
        
        return models
    
    def _check_onnx_conversion(
        self, 
        model_type: str, 
        quantized_variant: Optional[str] = None
    ) -> bool:
        """
        Check if a model has been converted to ONNX.
        
        Args:
            model_type: Type of model (mistral, llama3)
            quantized_variant: Name of quantized variant (e.g. int8_gptq)
            
        Returns:
            True if converted, False otherwise
        """
        if quantized_variant:
            # For quantized models
            onnx_dir = self.output_dir / model_type / f"quantized_{quantized_variant}"
        else:
            # For fp16 models
            onnx_dir = self.output_dir / model_type / "fp16"
        
        onnx_model_path = onnx_dir / "model.onnx"
        return onnx_model_path.exists()


def main():
    """Main function to run the ONNX converter."""
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing models")
    parser.add_argument("--output-dir", type=str, help="Directory to save converted models")
    parser.add_argument("--model", type=str, help="Model to convert (e.g., mistral, llama3)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"],
                        help="Precision to use for conversion")
    parser.add_argument("--opset-version", type=int, default=15, help="ONNX opset version")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--max-sequence-length", type=int, default=2048, 
                        help="Maximum sequence length")
    parser.add_argument("--no-optimize", action="store_true", help="Skip ONNX model optimization")
    parser.add_argument("--quantized", action="store_true", 
                        help="Model is already quantized")
    parser.add_argument("--quantization-type", type=str, choices=["gptq", "awq"],
                        help="Type of quantization")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for HuggingFace models")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    try:
        converter = ONNXConverter(args.models_dir, args.output_dir)
        
        if args.list:
            # List available models
            models = converter.get_available_models()
            print("\nAvailable Models for ONNX Conversion:")
            print("=" * 80)
            for model in models:
                onnx_status = "Converted to ONNX" if model["onnx_converted"] else "Not converted"
                print(f"- {model['name']}")
                print(f"  Path: {model['path']}")
                print(f"  ONNX Status: {onnx_status}")
                
                if "quantized_variants" in model:
                    print("  Quantized variants:")
                    for variant in model["quantized_variants"]:
                        var_onnx_status = "Converted to ONNX" if variant["onnx_converted"] else "Not converted"
                        print(f"    - {variant['name']}")
                        print(f"      Path: {variant['path']}")
                        print(f"      ONNX Status: {var_onnx_status}")
                
                print()
        elif args.model:
            # Convert the specified model
            success = converter.convert_model(
                model_name=args.model,
                precision=args.precision,
                opset_version=args.opset_version,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                optimize=not args.no_optimize,
                quantized=args.quantized,
                quantization_type=args.quantization_type,
                cache_dir=args.cache_dir
            )
            
            if success:
                print(f"Successfully converted {args.model} to ONNX format")
            else:
                print(f"Failed to convert {args.model}")
                sys.exit(1)
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()