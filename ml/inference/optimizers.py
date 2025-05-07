#!/usr/bin/env python3
"""
Inference optimizations for the Supply Chain LLM system.

This module provides optimizations for model inference, including
ONNX conversion, quantization, and TensorRT integration.
"""

import os
import logging
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Import from handlers module to reuse ModelInfo type
from handlers import ModelInfo, ModelType, ModelPrecision

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference_optimizers")

class ModelOptimizer:
    """Handles optimization of models for inference."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the model optimizer.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        
        # Check if directory exists
        if not self.models_dir.exists():
            raise ValueError(f"Models directory does not exist: {models_dir}")
        
        # Ensure output directories exist
        self.onnx_dir = self.models_dir / "onnx"
        self.tensorrt_dir = self.models_dir / "tensorrt"
        
        self.onnx_dir.mkdir(exist_ok=True)
        self.tensorrt_dir.mkdir(exist_ok=True)
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def optimize_model(
        self, 
        model_info: ModelInfo, 
        precision: str = "fp16", 
        device: str = "cuda"
    ) -> bool:
        """
        Optimize a model for inference.
        
        Args:
            model_info: Information about the model
            precision: Target precision (fp32, fp16, int8)
            device: Target device (cuda, cpu)
            
        Returns:
            True if successful, False otherwise
        """
        if model_info.type == ModelType.PYTORCH:
            return await self.convert_to_onnx(model_info, precision, device)
        elif model_info.type == ModelType.ONNX:
            return await self.convert_to_tensorrt(model_info, precision, device)
        else:
            logger.error(f"Cannot optimize model type: {model_info.type}")
            return False
    
    async def convert_to_onnx(
        self, 
        model_info: ModelInfo, 
        precision: str = "fp16", 
        device: str = "cuda"
    ) -> bool:
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            model_info: Information about the model
            precision: Target precision (fp32, fp16)
            device: Target device (cuda, cpu)
            
        Returns:
            True if successful, False otherwise
        """
        # Use lock to prevent concurrent conversion
        async with self.lock:
            try:
                # Check if model is PyTorch
                if model_info.type != ModelType.PYTORCH:
                    logger.error(f"Model {model_info.id} is not a PyTorch model")
                    return False
                
                # Determine target directories
                model_type = model_info.id.split("-")[0]  # Extract model type (mistral, llama3, etc.)
                quantized = "quantization_method" in model_info.metadata if model_info.metadata else False
                quant_method = model_info.metadata.get("quantization_method", "") if model_info.metadata else ""
                
                # Create target directories
                model_dir = self.onnx_dir / model_type
                model_dir.mkdir(exist_ok=True)
                
                if quantized:
                    target_dir = model_dir / f"quantized_{model_info.precision}_{quant_method}"
                else:
                    target_dir = model_dir / precision
                
                target_dir.mkdir(exist_ok=True)
                
                # Check if target already exists
                target_path = target_dir / "model.onnx"
                if target_path.exists():
                    logger.info(f"ONNX model already exists at {target_path}")
                    return True
                
                # Import dependencies
                try:
                    import torch
                    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
                except ImportError as e:
                    logger.error(f"Required packages not installed: {str(e)}")
                    return False
                
                # Log start
                logger.info(f"Converting {model_info.id} to ONNX format with {precision} precision...")
                
                # Load tokenizer
                logger.info(f"Loading tokenizer from {model_info.path}...")
                tokenizer = AutoTokenizer.from_pretrained(model_info.path)
                
                # Save tokenizer to target directory
                tokenizer.save_pretrained(str(target_dir))
                
                # Determine torch dtype
                if precision == "fp16":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                # Load model based on quantization status
                if quantized:
                    logger.info(f"Loading quantized model from {model_info.path}...")
                    
                    if quant_method.lower() == "gptq":
                        try:
                            from auto_gptq import AutoGPTQForCausalLM
                            model = AutoGPTQForCausalLM.from_quantized(
                                model_info.path,
                                use_safetensors=True,
                                trust_remote_code=True
                            )
                        except ImportError:
                            logger.error("auto-gptq package not installed")
                            return False
                    
                    elif quant_method.lower() == "awq":
                        try:
                            from awq import AutoAWQForCausalLM
                            model = AutoAWQForCausalLM.from_quantized(
                                model_info.path,
                                trust_remote_code=True
                            )
                        except ImportError:
                            logger.error("awq package not installed")
                            return False
                    
                    else:
                        logger.error(f"Unsupported quantization method: {quant_method}")
                        return False
                
                else:
                    # Set device map
                    if device.lower() == "cuda" and torch.cuda.is_available():
                        device_map = "auto"
                    else:
                        device_map = "cpu"
                    
                    # Load model
                    logger.info(f"Loading PyTorch model from {model_info.path}...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info.path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True
                    )
                
                # Set model to evaluation mode
                model.eval()
                
                # Generate dummy input for tracing
                max_sequence_length = 512  # Adjust as needed
                input_ids = torch.ones((1, max_sequence_length), dtype=torch.long).to(model.device)
                attention_mask = torch.ones((1, max_sequence_length), dtype=torch.bool).to(model.device)
                
                # Define dynamic axes for variable batch size and sequence length
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
                
                # Prepare inputs dictionary
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                
                # Export to ONNX
                logger.info(f"Exporting model to ONNX format at {target_path}...")
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        (dummy_inputs,),
                        str(target_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["logits"],
                        dynamic_axes=dynamic_axes,
                        opset_version=15,
                        do_constant_folding=True,
                        export_params=True,
                        verbose=False
                    )
                
                # Optimize ONNX model
                if await self._optimize_onnx_model(target_path, target_path):
                    logger.info("ONNX model optimized successfully")
                else:
                    logger.warning("ONNX model optimization skipped")
                
                # Create configuration file
                self._create_onnx_config(
                    target_dir / "onnx_config.json",
                    model_info,
                    precision
                )
                
                logger.info(f"Successfully converted {model_info.id} to ONNX format")
                return True
                
            except Exception as e:
                logger.error(f"Error converting to ONNX: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    async def _optimize_onnx_model(self, input_path: Path, output_path: Path) -> bool:
        """
        Optimize an ONNX model.
        
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
            
            # Load model
            onnx_model = onnx.load(str(input_path))
            
            # Create optimizer
            opt_model = optimizer.optimize_model(
                str(input_path),
                model_type="gpt2",  # Use GPT2 as the base type
                num_heads=16,  # Adjust as needed
                hidden_size=4096,  # Adjust as needed
                optimization_options=None
            )
            
            # Save optimized model
            opt_model.save_model_to_file(str(output_path))
            
            return True
            
        except ImportError as e:
            logger.warning(f"ONNX optimization packages not installed: {str(e)}")
            logger.warning("Install with: pip install onnx onnxruntime-tools")
            return False
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {str(e)}")
            return False
    
    def _create_onnx_config(
        self, 
        config_path: Path, 
        model_info: ModelInfo, 
        precision: str
    ) -> None:
        """
        Create a configuration file for the ONNX model.
        
        Args:
            config_path: Path to save configuration
            model_info: Original model information
            precision: Target precision
        """
        try:
            # Create config dictionary
            config = {
                "model_id": model_info.id.replace("pytorch", "onnx"),
                "original_model": model_info.id,
                "model_type": model_info.id.split("-")[0],
                "precision": precision,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "opset_version": 15,
            }
            
            # Add metadata from original model
            if model_info.metadata:
                config["metadata"] = model_info.metadata
            
            # Save config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating ONNX config: {str(e)}")
    
    async def convert_to_tensorrt(
        self, 
        model_info: ModelInfo, 
        precision: str = "fp16", 
        device: str = "cuda"
    ) -> bool:
        """
        Convert an ONNX model to TensorRT format.
        
        Args:
            model_info: Information about the model
            precision: Target precision (fp32, fp16, int8)
            device: Target device (cuda, cpu)
            
        Returns:
            True if successful, False otherwise
        """
        # TensorRT only works on CUDA
        if device.lower() != "cuda":
            logger.error("TensorRT conversion requires CUDA")
            return False
        
        # Use lock to prevent concurrent conversion
        async with self.lock:
            try:
                # Check if model is ONNX
                if model_info.type != ModelType.ONNX:
                    logger.error(f"Model {model_info.id} is not an ONNX model")
                    return False
                
                # Try to import TensorRT
                try:
                    import tensorrt as trt
                    import pycuda.driver as cuda
                    import pycuda.autoinit
                except ImportError as e:
                    logger.error(f"TensorRT or PyCUDA not installed: {str(e)}")
                    return False
                
                # Determine target directories
                model_type = model_info.id.split("-")[0]  # Extract model type (mistral, llama3, etc.)
                
                # Create target directories
                model_dir = self.tensorrt_dir / model_type
                model_dir.mkdir(exist_ok=True)
                
                target_dir = model_dir / precision
                target_dir.mkdir(exist_ok=True)
                
                # Determine target path
                target_path = target_dir / f"model.engine"
                if target_path.exists():
                    logger.info(f"TensorRT engine already exists at {target_path}")
                    return True
                
                # Get source ONNX model path
                source_path = Path(model_info.path) / "model.onnx"
                if not source_path.exists():
                    logger.error(f"ONNX model not found at {source_path}")
                    return False
                
                # Get tokenizer path
                tokenizer_path = Path(model_info.path)
                
                # Copy tokenizer to target directory
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                tokenizer.save_pretrained(str(target_dir))
                
                # Log start
                logger.info(f"Converting {model_info.id} to TensorRT format with {precision} precision...")
                
                # Initialize TensorRT logger
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                
                # Create builder and network
                builder = trt.Builder(TRT_LOGGER)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                config = builder.create_builder_config()
                parser = trt.OnnxParser(network, TRT_LOGGER)
                
                # Set builder configuration
                if precision == "fp16":
                    config.set_flag(trt.BuilderFlag.FP16)
                elif precision == "int8":
                    config.set_flag(trt.BuilderFlag.INT8)
                
                # Set max workspace size (adjust as needed)
                config.max_workspace_size = 1 << 30  # 1 GB
                
                # Parse ONNX model
                with open(source_path, "rb") as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                        return False
                
                # Build engine
                logger.info("Building TensorRT engine (this may take a while)...")
                plan = builder.build_serialized_network(network, config)
                if not plan:
                    logger.error("Failed to create TensorRT engine")
                    return False
                
                # Save engine
                with open(target_path, "wb") as f:
                    f.write(plan)
                
                # Create configuration file
                self._create_tensorrt_config(
                    target_dir / "config.json",
                    model_info,
                    precision
                )
                
                logger.info(f"Successfully converted {model_info.id} to TensorRT format")
                return True
                
            except Exception as e:
                logger.error(f"Error converting to TensorRT: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    def _create_tensorrt_config(
        self, 
        config_path: Path, 
        model_info: ModelInfo, 
        precision: str
    ) -> None:
        """
        Create a configuration file for the TensorRT model.
        
        Args:
            config_path: Path to save configuration
            model_info: Original model information
            precision: Target precision
        """
        try:
            # Create config dictionary
            config = {
                "model_id": model_info.id.replace("onnx", "tensorrt"),
                "original_model": model_info.id,
                "model_type": model_info.id.split("-")[0],
                "precision": precision,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Add metadata from original model
            if model_info.metadata:
                config["metadata"] = model_info.metadata
            
            # Save config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating TensorRT config: {str(e)}")
    
    async def quantize_onnx_model(
        self, 
        model_info: ModelInfo, 
        precision: str = "int8"
    ) -> bool:
        """
        Quantize an ONNX model.
        
        Args:
            model_info: Information about the model
            precision: Target precision (int8)
            
        Returns:
            True if successful, False otherwise
        """
        # Use lock to prevent concurrent quantization
        async with self.lock:
            try:
                # Check if model is ONNX
                if model_info.type != ModelType.ONNX:
                    logger.error(f"Model {model_info.id} is not an ONNX model")
                    return False
                
                # Try to import onnxruntime
                try:
                    import onnx
                    import onnxruntime as ort
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                except ImportError as e:
                    logger.error(f"ONNX quantization packages not installed: {str(e)}")
                    return False
                
                # Determine target directories
                model_type = model_info.id.split("-")[0]  # Extract model type (mistral, llama3, etc.)
                
                # Create target directories
                model_dir = self.onnx_dir / model_type
                
                # Create target directories
                quant_dir_name = f"quantized_{precision}"
                target_dir = model_dir / quant_dir_name
                target_dir.mkdir(exist_ok=True, parents=True)
                
                # Determine source and target paths
                source_path = Path(model_info.path) / "model.onnx"
                target_path = target_dir / "model.onnx"
                
                if not source_path.exists():
                    logger.error(f"ONNX model not found at {source_path}")
                    return False
                
                if target_path.exists():
                    logger.info(f"Quantized ONNX model already exists at {target_path}")
                    return True
                
                # Get tokenizer path
                tokenizer_path = Path(model_info.path)
                
                # Copy tokenizer to target directory
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                tokenizer.save_pretrained(str(target_dir))
                
                # Log start
                logger.info(f"Quantizing {model_info.id} to {precision}...")
                
                # Determine QuantType
                if precision == "int8":
                    quant_type = QuantType.QInt8
                else:
                    logger.error(f"Unsupported quantization precision: {precision}")
                    return False
                
                # Perform quantization
                logger.info(f"Quantizing ONNX model from {source_path} to {target_path}...")
                quantize_dynamic(
                    model_input=str(source_path),
                    model_output=str(target_path),
                    per_channel=False,
                    reduce_range=False,
                    weight_type=quant_type
                )
                
                # Create configuration file
                self._create_onnx_quantized_config(
                    target_dir / "onnx_config.json",
                    model_info,
                    precision
                )
                
                logger.info(f"Successfully quantized {model_info.id} to {precision}")
                return True
                
            except Exception as e:
                logger.error(f"Error quantizing ONNX model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    def _create_onnx_quantized_config(
        self, 
        config_path: Path, 
        model_info: ModelInfo, 
        precision: str
    ) -> None:
        """
        Create a configuration file for the quantized ONNX model.
        
        Args:
            config_path: Path to save configuration
            model_info: Original model information
            precision: Target precision
        """
        try:
            # Create config dictionary
            config = {
                "model_id": f"{model_info.id}_{precision}",
                "original_model": model_info.id,
                "model_type": model_info.id.split("-")[0],
                "precision": precision,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "quantization": {
                    "method": "dynamic",
                    "precision": precision
                }
            }
            
            # Add metadata from original model
            if model_info.metadata:
                config["metadata"] = model_info.metadata
            
            # Save config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating quantized ONNX config: {str(e)}")
    
    async def benchmark_model(
        self,
        model_info: ModelInfo,
        input_length: int = 128,
        output_length: int = 32,
        batch_size: int = 1,
        num_iterations: int = 10,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Benchmark a model for performance.
        
        Args:
            model_info: Information about the model
            input_length: Length of input sequence
            output_length: Length of output sequence
            batch_size: Batch size
            num_iterations: Number of iterations
            device: Device to run on
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            # Get model type
            model_type = model_info.type
            
            # Run benchmark based on model type
            if model_type == ModelType.PYTORCH:
                return await self._benchmark_pytorch(
                    model_info, input_length, output_length, 
                    batch_size, num_iterations, device
                )
            elif model_type == ModelType.ONNX:
                return await self._benchmark_onnx(
                    model_info, input_length, output_length, 
                    batch_size, num_iterations, device
                )
            elif model_type == ModelType.TENSORRT:
                return await self._benchmark_tensorrt(
                    model_info, input_length, output_length, 
                    batch_size, num_iterations, device
                )
            else:
                logger.error(f"Unsupported model type for benchmarking: {model_type}")
                return {
                    "success": False,
                    "error": f"Unsupported model type: {model_type}"
                }
                
        except Exception as e:
            logger.error(f"Error benchmarking model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _benchmark_pytorch(
        self,
        model_info: ModelInfo,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_iterations: int,
        device: str
    ) -> Dict[str, Any]:
        """
        Benchmark a PyTorch model.
        
        Args:
            model_info: Model information
            input_length: Length of input sequence
            output_length: Length of output sequence
            batch_size: Batch size
            num_iterations: Number of iterations
            device: Device to run on
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import time
            import numpy as np
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info.path)
            
            # Check if model is quantized
            is_quantized = "quantization_method" in model_info.metadata if model_info.metadata else False
            quant_method = model_info.metadata.get("quantization_method") if model_info.metadata else None
            
            # Load model based on quantization status
            if is_quantized and quant_method:
                logger.info(f"Loading quantized PyTorch model ({quant_method})...")
                
                if quant_method.lower() == "gptq":
                    try:
                        from auto_gptq import AutoGPTQForCausalLM
                        model = AutoGPTQForCausalLM.from_quantized(
                            model_info.path,
                            use_safetensors=True,
                            trust_remote_code=True
                        )
                    except ImportError:
                        logger.error("auto-gptq package not installed")
                        return {"success": False, "error": "auto-gptq package not installed"}
                
                elif quant_method.lower() == "awq":
                    try:
                        from awq import AutoAWQForCausalLM
                        model = AutoAWQForCausalLM.from_quantized(
                            model_info.path,
                            trust_remote_code=True
                        )
                    except ImportError:
                        logger.error("awq package not installed")
                        return {"success": False, "error": "awq package not installed"}
                
                else:
                    logger.error(f"Unsupported quantization method: {quant_method}")
                    return {"success": False, "error": f"Unsupported quantization method: {quant_method}"}
            
            else:
                # Determine torch dtype
                if model_info.precision == ModelPrecision.FP16:
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                # Set device map
                if device.lower() == "cuda" and torch.cuda.is_available():
                    device_map = "auto"
                else:
                    device_map = "cpu"
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_info.path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            # Set evaluation mode
            model.eval()
            
            # Generate input tokens
            input_text = "The supply chain analysis shows" * (input_length // 5 + 1)
            input_text = input_text[:input_length * 4]  # Approximate conversion
            
            # Tokenize input
            inputs = tokenizer(
                [input_text] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=input_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if device.lower() == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": output_length,
                "do_sample": False,  # Deterministic for benchmarking
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            }
            
            # Run warmup iteration
            with torch.no_grad():
                _ = model.generate(**inputs, **gen_kwargs)
            
            # Benchmark generation time
            latencies = []
            first_token_latencies = []
            
            for _ in range(num_iterations):
                # Clear CUDA cache
                if device.lower() == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Record first token latency
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        **gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                end_time = time.time()
                latencies.append(end_time - start_time)
                
                # Approximate first token latency
                first_token_latencies.append(latencies[-1] / output_length)
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            avg_first_token_latency = np.mean(first_token_latencies)
            tokens_per_second = output_length * batch_size / avg_latency
            
            return {
                "success": True,
                "model_id": model_info.id,
                "model_type": str(model_info.type),
                "precision": str(model_info.precision),
                "input_length": input_length,
                "output_length": output_length,
                "batch_size": batch_size,
                "iterations": num_iterations,
                "avg_latency_ms": avg_latency * 1000,
                "avg_first_token_latency_ms": avg_first_token_latency * 1000,
                "tokens_per_second": tokens_per_second,
                "device": device
            }
                
        except Exception as e:
            logger.error(f"Error benchmarking PyTorch model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _benchmark_onnx(
        self,
        model_info: ModelInfo,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_iterations: int,
        device: str
    ) -> Dict[str, Any]:
        """
        Benchmark an ONNX model.
        
        Args:
            model_info: Model information
            input_length: Length of input sequence
            output_length: Length of output sequence
            batch_size: Batch size
            num_iterations: Number of iterations
            device: Device to run on
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            import time
            import numpy as np
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info.path)
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select provider based on device
            if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                device = "cpu"  # Override device if CUDA not available
            
            # Load ONNX model
            model_path = Path(model_info.path) / "model.onnx"
            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input and output names
            input_names = [input.name for input in session.get_inputs()]
            output_names = [output.name for output in session.get_outputs()]
            
            # Generate input tokens
            input_text = "The supply chain analysis shows" * (input_length // 5 + 1)
            input_text = input_text[:input_length * 4]  # Approximate conversion
            
            # Tokenize input
            inputs = tokenizer(
                [input_text] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=input_length,
                return_tensors="pt"
            )
            
            # Convert inputs to numpy
            onnx_inputs = {}
            for name in input_names:
                if name == "input_ids":
                    onnx_inputs[name] = inputs["input_ids"].numpy()
                elif name == "attention_mask":
                    onnx_inputs[name] = inputs["attention_mask"].numpy() if "attention_mask" in inputs else np.ones_like(inputs["input_ids"].numpy())
                else:
                    logger.warning(f"Unhandled input: {name}")
            
            # Run warmup iteration
            _ = session.run(output_names, onnx_inputs)
            
            # Benchmark inference time
            latencies = []
            for _ in range(num_iterations):
                start_time = time.time()
                _ = session.run(output_names, onnx_inputs)
                end_time = time.time()
                latencies.append(end_time - start_time)
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            tokens_per_second = input_length * batch_size / avg_latency  # For ONNX we benchmark input processing, not generation
            
            # Note: For ONNX models, we don't have an easy way to benchmark generation or first token latency
            # since we would need to implement the full generation loop
            return {
                "success": True,
                "model_id": model_info.id,
                "model_type": str(model_info.type),
                "precision": str(model_info.precision),
                "input_length": input_length,
                "batch_size": batch_size,
                "iterations": num_iterations,
                "avg_inference_latency_ms": avg_latency * 1000,
                "tokens_per_second": tokens_per_second,
                "device": device,
                "note": "ONNX benchmark measures input processing, not generation"
            }
                
        except Exception as e:
            logger.error(f"Error benchmarking ONNX model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _benchmark_tensorrt(
        self,
        model_info: ModelInfo,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_iterations: int,
        device: str
    ) -> Dict[str, Any]:
        """
        Benchmark a TensorRT model.
        
        Args:
            model_info: Model information
            input_length: Length of input sequence
            output_length: Length of output sequence
            batch_size: Batch size
            num_iterations: Number of iterations
            device: Device to run on
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            from transformers import AutoTokenizer
            import time
            import numpy as np
            
            # TensorRT requires CUDA
            if device.lower() != "cuda":
                return {
                    "success": False,
                    "error": "TensorRT requires CUDA"
                }
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info.path)
            
            # Find engine file
            engine_files = list(Path(model_info.path).glob("*.engine"))
            if not engine_files:
                return {
                    "success": False,
                    "error": f"No TensorRT engine found in {model_info.path}"
                }
            
            engine_path = engine_files[0]
            
            # Initialize TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Generate input tokens
            input_text = "The supply chain analysis shows" * (input_length // 5 + 1)
            input_text = input_text[:input_length * 4]  # Approximate conversion
            
            # Tokenize input
            inputs = tokenizer(
                [input_text] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=input_length,
                return_tensors="pt"
            )
            
            # Prepare TensorRT bindings
            input_binding_idxs = []
            output_binding_idxs = []
            bindings = [None] * engine.num_bindings
            
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    input_binding_idxs.append(i)
                else:
                    output_binding_idxs.append(i)
            
            # Allocate device memory for inputs
            input_dict = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy() if "attention_mask" in inputs else np.ones_like(inputs["input_ids"].numpy())
            }
            
            for i in input_binding_idxs:
                binding_name = engine.get_binding_name(i)
                if "input_ids" in binding_name and "input_ids" in input_dict:
                    input_data = input_dict["input_ids"]
                    input_mem = cuda.mem_alloc(input_data.nbytes)
                    cuda.memcpy_htod(input_mem, input_data)
                    bindings[i] = int(input_mem)
                elif "attention_mask" in binding_name and "attention_mask" in input_dict:
                    input_data = input_dict["attention_mask"]
                    input_mem = cuda.mem_alloc(input_data.nbytes)
                    cuda.memcpy_htod(input_mem, input_data)
                    bindings[i] = int(input_mem)
            
            # Allocate device memory for outputs
            outputs = {}
            for i in output_binding_idxs:
                shape = engine.get_binding_shape(i)
                dtype = np.float32  # Assuming float32 output
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                output_mem = cuda.mem_alloc(size)
                bindings[i] = int(output_mem)
                outputs[i] = {
                    "mem": output_mem,
                    "shape": shape,
                    "dtype": dtype
                }
            
            # Run warmup iteration
            context.execute_v2(bindings)
            
            # Benchmark inference time
            latencies = []
            for _ in range(num_iterations):
                start_time = time.time()
                context.execute_v2(bindings)
                end_time = time.time()
                latencies.append(end_time - start_time)
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            tokens_per_second = input_length * batch_size / avg_latency  # For TensorRT we benchmark input processing, not generation
            
            return {
                "success": True,
                "model_id": model_info.id,
                "model_type": str(model_info.type),
                "precision": str(model_info.precision),
                "input_length": input_length,
                "batch_size": batch_size,
                "iterations": num_iterations,
                "avg_inference_latency_ms": avg_latency * 1000,
                "tokens_per_second": tokens_per_second,
                "device": device,
                "note": "TensorRT benchmark measures input processing, not generation"
            }
                
        except Exception as e:
            logger.error(f"Error benchmarking TensorRT model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }