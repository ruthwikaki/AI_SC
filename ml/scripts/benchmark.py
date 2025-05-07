#!/usr/bin/env python3
"""
Benchmarking script for the Supply Chain LLM system.

This script measures performance metrics for various model formats and configurations
to help determine the optimal setup for production deployment, comparing PyTorch,
quantized, and ONNX model versions.
"""

import os
import sys
import argparse
import logging
import json
import time
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_benchmark")

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    model_type: str
    format: str  # pytorch, onnx, tensorrt
    precision: str  # fp32, fp16, int8, int4
    batch_size: int
    sequence_length: int
    total_tokens: int
    first_token_latency_ms: float
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    timestamp: str
    hardware_info: Dict[str, Any]
    environment_info: Dict[str, Any]
    additional_metrics: Dict[str, Any]

class ModelBenchmark:
    """Handles performance benchmarking of models in various formats."""
    
    def __init__(
        self,
        models_dir: str,
        output_dir: Optional[str] = None,
        benchmark_prompts_file: Optional[str] = None
    ):
        """
        Initialize the model benchmarking tool.
        
        Args:
            models_dir: Directory containing model files
            output_dir: Directory to save benchmark results
            benchmark_prompts_file: File containing benchmark prompts
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmark_results")
        
        # Check if models directory exists
        if not self.models_dir.exists():
            raise ValueError(f"Models directory does not exist: {models_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark prompts
        self.prompts = self._load_benchmark_prompts(benchmark_prompts_file)
        
        # Check for CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA is not available. Benchmarks will run on CPU only.")
    
    def _load_benchmark_prompts(self, prompts_file: Optional[str]) -> List[str]:
        """
        Load benchmark prompts from a file or use defaults.
        
        Args:
            prompts_file: Path to prompts file
            
        Returns:
            List of prompts
        """
        default_prompts = [
            "Analyze the impact of raw material shortages on production schedule for the next quarter.",
            "Provide a forecast for shipping delays based on current port congestion levels.",
            "Calculate optimal inventory levels for high-demand products during peak season.",
            "Compare cost efficiency of rail versus truck transportation for cross-country distribution.",
            "Evaluate supplier performance based on delivery time, quality, and cost metrics.",
            "Generate a risk assessment for the current supply chain network configuration.",
            "Recommend warehouse layout changes to improve picking efficiency and reduce fulfillment time.",
            "Analyze historical data to identify patterns in customer returns related to shipping damage.",
            "Calculate the cost-benefit analysis of implementing RFID technology across all warehouses.",
            "Provide a demand forecast for the next 12 months accounting for seasonality and market trends."
        ]
        
        if not prompts_file:
            return default_prompts
        
        try:
            prompts = []
            with open(prompts_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        prompts.append(line)
            
            if not prompts:
                logger.warning(f"No prompts found in {prompts_file}. Using defaults.")
                return default_prompts
            
            return prompts
        
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_file}: {str(e)}")
            logger.info("Using default prompts")
            return default_prompts
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models for benchmarking.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Check for PyTorch models
        for model_type in ["mistral", "llama3"]:
            model_dir = self.models_dir / model_type
            if not model_dir.exists():
                continue
            
            weights_dir = model_dir / "weights"
            if weights_dir.exists():
                model_info = {
                    "name": model_type,
                    "type": model_type,
                    "format": "pytorch",
                    "precision": "fp16",  # Assuming fp16 for base models
                    "path": str(weights_dir)
                }
                models.append(model_info)
                
                # Check for quantized variants
                for quant_dir in weights_dir.glob("quantized_*"):
                    if quant_dir.is_dir():
                        # Extract precision and method from directory name
                        parts = quant_dir.name.split("_")
                        if len(parts) >= 3:
                            precision = parts[1]
                            method = parts[2]
                            
                            quant_model_info = {
                                "name": f"{model_type}_{precision}_{method}",
                                "type": model_type,
                                "format": "pytorch",
                                "precision": precision,
                                "quantization_method": method,
                                "path": str(quant_dir)
                            }
                            models.append(quant_model_info)
        
        # Check for ONNX models
        onnx_dir = self.models_dir / "onnx"
        if onnx_dir.exists():
            for model_type in ["mistral", "llama3"]:
                model_type_dir = onnx_dir / model_type
                if not model_type_dir.exists():
                    continue
                
                # Check for standard ONNX models (fp16, fp32)
                for precision in ["fp16", "fp32"]:
                    precision_dir = model_type_dir / precision
                    if precision_dir.exists() and (precision_dir / "model.onnx").exists():
                        onnx_model_info = {
                            "name": f"{model_type}_{precision}_onnx",
                            "type": model_type,
                            "format": "onnx",
                            "precision": precision,
                            "path": str(precision_dir)
                        }
                        models.append(onnx_model_info)
                
                # Check for quantized ONNX models
                for quant_dir in model_type_dir.glob("quantized_*"):
                    if quant_dir.is_dir() and (quant_dir / "model.onnx").exists():
                        # Extract precision and method from directory name
                        parts = quant_dir.name.split("_")
                        if len(parts) >= 3:
                            precision = parts[1]
                            method = parts[2]
                            
                            quant_onnx_info = {
                                "name": f"{model_type}_{precision}_{method}_onnx",
                                "type": model_type,
                                "format": "onnx",
                                "precision": precision,
                                "quantization_method": method,
                                "path": str(quant_dir)
                            }
                            models.append(quant_onnx_info)
        
        # Check for TensorRT models
        tensorrt_dir = self.models_dir / "tensorrt"
        if tensorrt_dir.exists():
            for model_type in ["mistral", "llama3"]:
                model_type_dir = tensorrt_dir / model_type
                if not model_type_dir.exists():
                    continue
                
                # Check for standard TensorRT models
                for precision in ["fp16", "fp32", "int8"]:
                    precision_dir = model_type_dir / precision
                    if precision_dir.exists() and list(precision_dir.glob("*.engine")):
                        tensorrt_model_info = {
                            "name": f"{model_type}_{precision}_tensorrt",
                            "type": model_type,
                            "format": "tensorrt",
                            "precision": precision,
                            "path": str(precision_dir)
                        }
                        models.append(tensorrt_model_info)
        
        return models
    
    def benchmark_model(
        self,
        model_info: Dict[str, Any],
        batch_sizes: List[int],
        sequence_lengths: List[int],
        num_runs: int = 3,
        warmup_runs: int = 1,
        use_cuda: bool = True,
        output_tokens: int = 100
    ) -> List[BenchmarkResult]:
        """
        Benchmark a model with various parameters.
        
        Args:
            model_info: Model information dictionary
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_runs: Number of benchmark runs for each configuration
            warmup_runs: Number of warmup runs
            use_cuda: Whether to use CUDA if available
            output_tokens: Number of tokens to generate
            
        Returns:
            List of benchmark results
        """
        # Check CUDA availability if requested
        if use_cuda and not self.cuda_available:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            use_cuda = False
        
        # Get model format
        model_format = model_info["format"]
        model_name = model_info["name"]
        model_type = model_info["type"]
        model_path = model_info["path"]
        model_precision = model_info["precision"]
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return []
        
        # Load model based on format
        if model_format == "pytorch":
            model, tokenizer = self._load_pytorch_model(model_info, use_cuda)
        elif model_format == "onnx":
            model, tokenizer = self._load_onnx_model(model_info, use_cuda)
        elif model_format == "tensorrt":
            model, tokenizer = self._load_tensorrt_model(model_info, use_cuda)
        else:
            logger.error(f"Unsupported model format: {model_format}")
            return []
        
        if model is None or tokenizer is None:
            logger.error(f"Failed to load model: {model_name}")
            return []
        
        results = []
        
        # Run benchmarks for each configuration
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                logger.info(f"Benchmarking {model_name} with batch size {batch_size}, " 
                           f"sequence length {seq_length}, {output_tokens} output tokens")
                
                # Prepare input prompts
                if batch_size <= len(self.prompts):
                    batch_prompts = self.prompts[:batch_size]
                else:
                    # If we need more prompts than available, repeat them
                    batch_prompts = []
                    for i in range(batch_size):
                        batch_prompts.append(self.prompts[i % len(self.prompts)])
                
                # Tokenize inputs
                inputs = tokenizer(
                    batch_prompts, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=seq_length, 
                    return_tensors="pt"
                )
                
                if use_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Perform warmup runs
                for _ in range(warmup_runs):
                    try:
                        with torch.no_grad():
                            _ = self._generate_tokens(
                                model, 
                                tokenizer, 
                                inputs, 
                                model_format, 
                                output_tokens=5  # Short generation for warmup
                            )
                    except Exception as e:
                        logger.error(f"Error during warmup: {str(e)}")
                
                # Run actual benchmarks
                batch_results = []
                for run in range(num_runs):
                    try:
                        # Measure memory before run
                        if use_cuda:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.empty_cache()
                            start_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                        else:
                            start_mem = 0  # Not available for CPU
                        
                        # Start timing
                        start_time = time.time()
                        
                        # Generate tokens
                        output, first_token_time = self._generate_tokens(
                            model, 
                            tokenizer, 
                            inputs, 
                            model_format, 
                            output_tokens=output_tokens
                        )
                        
                        # End timing
                        end_time = time.time()
                        
                        # Measure memory after run
                        if use_cuda:
                            end_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                            gpu_util = self._get_gpu_utilization()
                        else:
                            end_mem = 0  # Not available for CPU
                            gpu_util = 0
                        
                        # Calculate metrics
                        total_time = end_time - start_time
                        total_tokens = output_tokens * batch_size
                        tokens_per_second = total_tokens / total_time
                        memory_usage = end_mem - start_mem
                        
                        # Create result object
                        result = BenchmarkResult(
                            model_name=model_name,
                            model_type=model_type,
                            format=model_format,
                            precision=model_precision,
                            batch_size=batch_size,
                            sequence_length=seq_length,
                            total_tokens=total_tokens,
                            first_token_latency_ms=first_token_time * 1000,
                            tokens_per_second=tokens_per_second,
                            memory_usage_mb=memory_usage,
                            gpu_utilization_percent=gpu_util,
                            timestamp=datetime.now().isoformat(),
                            hardware_info=self._get_hardware_info(),
                            environment_info=self._get_environment_info(),
                            additional_metrics={}
                        )
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error in benchmark run {run}: {str(e)}")
                
                # Calculate average results for this configuration
                if batch_results:
                    avg_result = self._average_results(batch_results)
                    results.append(avg_result)
                    
                    # Log results
                    logger.info(f"Results for {model_name}, batch_size={batch_size}, "
                                f"seq_length={seq_length}:")
                    logger.info(f"  First token latency: {avg_result.first_token_latency_ms:.2f} ms")
                    logger.info(f"  Tokens per second: {avg_result.tokens_per_second:.2f}")
                    logger.info(f"  Memory usage: {avg_result.memory_usage_mb:.2f} MB")
                    logger.info(f"  GPU utilization: {avg_result.gpu_utilization_percent:.2f}%")
        
        # Save results
        self._save_results(results, model_name)
        
        return results
    
    def _load_pytorch_model(
        self, 
        model_info: Dict[str, Any], 
        use_cuda: bool
    ) -> Tuple[Any, Any]:
        """
        Load a PyTorch model.
        
        Args:
            model_info: Model information
            use_cuda: Whether to use CUDA
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_path = model_info["path"]
            model_precision = model_info["precision"]
            
            # Determine if model is quantized
            is_quantized = "quantization_method" in model_info
            quant_method = model_info.get("quantization_method", None)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model based on quantization status
            if is_quantized:
                logger.info(f"Loading quantized PyTorch model ({quant_method})...")
                
                if quant_method == "gptq":
                    try:
                        from auto_gptq import AutoGPTQForCausalLM
                        model = AutoGPTQForCausalLM.from_quantized(
                            model_path,
                            use_safetensors=True,
                            trust_remote_code=True
                        )
                    except ImportError:
                        logger.error("auto-gptq package not installed")
                        return None, None
                
                elif quant_method == "awq":
                    try:
                        from awq import AutoAWQForCausalLM
                        model = AutoAWQForCausalLM.from_quantized(
                            model_path,
                            trust_remote_code=True
                        )
                    except ImportError:
                        logger.error("awq package not installed")
                        return None, None
                
                else:
                    logger.error(f"Unsupported quantization method: {quant_method}")
                    return None, None
            
            else:
                # Load full precision model
                logger.info(f"Loading PyTorch model ({model_precision})...")
                
                # Determine torch dtype
                if model_precision == "fp16":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                # Load model
                device_map = "auto" if use_cuda else "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
            
            # Move model to appropriate device
            if use_cuda and not is_quantized:
                model = model.cuda()
            
            # Set eval mode
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _load_onnx_model(
        self, 
        model_info: Dict[str, Any], 
        use_cuda: bool
    ) -> Tuple[Any, Any]:
        """
        Load an ONNX model.
        
        Args:
            model_info: Model information
            use_cuda: Whether to use CUDA
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            model_path = model_info["path"]
            onnx_path = Path(model_path) / "model.onnx"
            
            if not onnx_path.exists():
                logger.error(f"ONNX model not found at {onnx_path}")
                return None, None
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Setup ONNX Runtime session
            logger.info(f"Loading ONNX model from {onnx_path}...")
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select execution provider
            if use_cuda:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Create session
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=providers
            )
            
            return session, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _load_tensorrt_model(
        self, 
        model_info: Dict[str, Any], 
        use_cuda: bool
    ) -> Tuple[Any, Any]:
        """
        Load a TensorRT model.
        
        Args:
            model_info: Model information
            use_cuda: Whether to use CUDA
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Import here to avoid requiring these dependencies unless needed
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            from transformers import AutoTokenizer
            
            model_path = model_info["path"]
            
            # Find engine file
            engine_files = list(Path(model_path).glob("*.engine"))
            if not engine_files:
                logger.error(f"No TensorRT engine found in {model_path}")
                return None, None
            
            engine_path = engine_files[0]
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load TensorRT engine
            logger.info(f"Loading TensorRT engine from {engine_path}...")
            
            # Initialize TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            return (engine, context), tokenizer
            
        except ImportError as e:
            logger.error(f"TensorRT or PyCUDA not installed: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading TensorRT model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _generate_tokens(
        self,
        model,
        tokenizer,
        inputs,
        model_format: str,
        output_tokens: int = 100
    ) -> Tuple[Any, float]:
        """
        Generate tokens from a model.
        
        Args:
            model: Model object
            tokenizer: Tokenizer object
            inputs: Input tensors
            model_format: Format of the model
            output_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (output, first_token_time)
        """
        if model_format == "pytorch":
            return self._generate_pytorch(model, tokenizer, inputs, output_tokens)
        elif model_format == "onnx":
            return self._generate_onnx(model, tokenizer, inputs, output_tokens)
        elif model_format == "tensorrt":
            return self._generate_tensorrt(model, tokenizer, inputs, output_tokens)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
    
    def _generate_pytorch(
        self,
        model,
        tokenizer,
        inputs,
        output_tokens: int
    ) -> Tuple[Any, float]:
        """
        Generate tokens from a PyTorch model.
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer
            inputs: Input tensors
            output_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (output, first_token_time)
        """
        # Configure generation parameters
        gen_config = {
            "max_new_tokens": output_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        }
        
        # Track first token time
        first_token_time = 0
        
        # Define callback to track first token generation
        class FirstTokenCallback:
            def __init__(self):
                self.first_token_generated = False
                self.start_time = time.time()
                self.first_token_time = 0
            
            def __call__(self, beam_idx, token_idx, token, past_key_values, **kwargs):
                if not self.first_token_generated:
                    self.first_token_time = time.time() - self.start_time
                    self.first_token_generated = True
                return False
        
        callback = FirstTokenCallback()
        
        # Generate tokens
        with torch.no_grad():
            output = model.generate(
                **inputs,
                **gen_config,
                streamer=None,
                callback=callback
            )
        
        return output, callback.first_token_time
    
    def _generate_onnx(
        self,
        session,
        tokenizer,
        inputs,
        output_tokens: int
    ) -> Tuple[Any, float]:
        """
        Generate tokens from an ONNX model.
        
        Args:
            session: ONNX Runtime session
            tokenizer: Tokenizer
            inputs: Input tensors
            output_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (output, first_token_time)
        """
        # Get ONNX input and output names
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        # Prepare inputs
        onnx_inputs = {}
        for name in input_names:
            if name == "input_ids":
                onnx_inputs[name] = inputs["input_ids"].cpu().numpy()
            elif name == "attention_mask":
                onnx_inputs[name] = inputs["attention_mask"].cpu().numpy()
            else:
                # Handle other possible inputs
                logger.warning(f"Unhandled ONNX input: {name}")
        
        # Start timing
        start_time = time.time()
        
        # Run inference for first token
        onnx_outputs = session.run(output_names, onnx_inputs)
        logits = onnx_outputs[0] if "logits" in output_names else onnx_outputs[0]
        
        # Get first token time
        first_token_time = time.time() - start_time
        
        # Simple greedy token generation for remaining tokens
        # This is a simplified version for benchmarking
        for _ in range(output_tokens - 1):
            # Get next token (greedy)
            next_token_id = logits[:, -1, :].argmax(axis=-1)
            
            # Update input_ids with new token
            input_ids = np.concatenate(
                [onnx_inputs["input_ids"], next_token_id.reshape(-1, 1)], 
                axis=1
            )
            onnx_inputs["input_ids"] = input_ids
            
            # Update attention mask if needed
            if "attention_mask" in onnx_inputs:
                batch_size = onnx_inputs["attention_mask"].shape[0]
                onnx_inputs["attention_mask"] = np.concatenate(
                    [
                        onnx_inputs["attention_mask"], 
                        np.ones((batch_size, 1), dtype=np.int64)
                    ], 
                    axis=1
                )
            
            # Get next token logits
            onnx_outputs = session.run(output_names, onnx_inputs)
            logits = onnx_outputs[0] if "logits" in output_names else onnx_outputs[0]
        
        return input_ids, first_token_time
    
    def _generate_tensorrt(
        self,
        model,
        tokenizer,
        inputs,
        output_tokens: int
    ) -> Tuple[Any, float]:
        """
        Generate tokens from a TensorRT model.
        
        Args:
            model: TensorRT model tuple (engine, context)
            tokenizer: Tokenizer
            inputs: Input tensors
            output_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (output, first_token_time)
        """
        # Unpack model
        engine, context = model
        
        # Allocate buffers
        import pycuda.driver as cuda
        
        # Get input and output binding indices
        input_binding_idxs = []
        output_binding_idxs = []
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                input_binding_idxs.append(i)
            else:
                output_binding_idxs.append(i)
        
        # Prepare input and output buffers
        bindings = [None] * engine.num_bindings
        
        # Allocate output buffers
        outputs = []
        output_shapes = []
        for i in output_binding_idxs:
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = trt.volume(shape) * dtype.itemsize
            output = cuda.mem_alloc(size)
            bindings[i] = int(output)
            outputs.append(output)
            output_shapes.append(shape)
        
        # Prepare input data
        input_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        
        # Allocate input buffers
        input_bindings = {}
        for i in input_binding_idxs:
            name = engine.get_binding_name(i)
            if "input_ids" in name:
                input_bindings["input_ids"] = i
                shape = engine.get_binding_shape(i)
                size = trt.volume(shape) * trt.nptype(engine.get_binding_dtype(i)).itemsize
                buf = cuda.mem_alloc(size)
                bindings[i] = int(buf)
                cuda.memcpy_htod(buf, input_ids)
            elif "attention_mask" in name:
                input_bindings["attention_mask"] = i
                shape = engine.get_binding_shape(i)
                size = trt.volume(shape) * trt.nptype(engine.get_binding_dtype(i)).itemsize
                buf = cuda.mem_alloc(size)
                bindings[i] = int(buf)
                cuda.memcpy_htod(buf, attention_mask)
        
        # Start timing
        start_time = time.time()
        
        # Execute inference for first token
        context.execute_v2(bindings)
        
        # Get first token time
        first_token_time = time.time() - start_time
        
        # Simplified generation loop for benchmarking
        # In a real implementation, you would need to handle token generation properly
        for _ in range(output_tokens - 1):
            # Execute inference for next token
            context.execute_v2(bindings)
        
        # For benchmarking purposes, we don't need to actually decode the tokens
        # We're just measuring performance
        
        return None, first_token_time
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """
        Average multiple benchmark results.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Averaged benchmark result
        """
        if not results:
            raise ValueError("Cannot average empty results list")
        
        # Get first result for static fields
        first = results[0]
        
        # Calculate averages for numeric fields
        ftl_avg = sum(r.first_token_latency_ms for r in results) / len(results)
        tps_avg = sum(r.tokens_per_second for r in results) / len(results)
        mem_avg = sum(r.memory_usage_mb for r in results) / len(results)
        gpu_avg = sum(r.gpu_utilization_percent for r in results) / len(results)
        
        # Create averaged result
        return BenchmarkResult(
            model_name=first.model_name,
            model_type=first.model_type,
            format=first.format,
            precision=first.precision,
            batch_size=first.batch_size,
            sequence_length=first.sequence_length,
            total_tokens=first.total_tokens,
            first_token_latency_ms=ftl_avg,
            tokens_per_second=tps_avg,
            memory_usage_mb=mem_avg,
            gpu_utilization_percent=gpu_avg,
            timestamp=datetime.now().isoformat(),
            hardware_info=first.hardware_info,
            environment_info=first.environment_info,
            additional_metrics={}
        )
    
    def _save_results(self, results: List[BenchmarkResult], model_name: str) -> None:
        """
        Save benchmark results to CSV and JSON files.
        
        Args:
            results: List of benchmark results
            model_name: Name of the model
        """
        if not results:
            logger.warning("No results to save")
            return
        
        # Create timestamp string for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model_name.replace('/', '_')}_{timestamp}"
        
        # Save to CSV
        csv_path = self.output_dir / f"{base_filename}.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                # Flatten nested dictionaries for CSV
                fieldnames = [
                    "model_name", "model_type", "format", "precision", "batch_size",
                    "sequence_length", "total_tokens", "first_token_latency_ms",
                    "tokens_per_second", "memory_usage_mb", "gpu_utilization_percent",
                    "timestamp"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Extract only the fields we want for CSV
                    row = {
                        "model_name": result.model_name,
                        "model_type": result.model_type,
                        "format": result.format,
                        "precision": result.precision,
                        "batch_size": result.batch_size,
                        "sequence_length": result.sequence_length,
                        "total_tokens": result.total_tokens,
                        "first_token_latency_ms": result.first_token_latency_ms,
                        "tokens_per_second": result.tokens_per_second,
                        "memory_usage_mb": result.memory_usage_mb,
                        "gpu_utilization_percent": result.gpu_utilization_percent,
                        "timestamp": result.timestamp
                    }
                    writer.writerow(row)
            
            logger.info(f"Saved results to {csv_path}")
        
        except Exception as e:
            logger.error(f"Error saving CSV results: {str(e)}")
        
        # Save to JSON (includes all fields)
        json_path = self.output_dir / f"{base_filename}.json"
        try:
            with open(json_path, 'w') as f:
                # Convert dataclass objects to dictionaries
                json_results = [asdict(result) for result in results]
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved results to {json_path}")
        
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")
        
        # Generate and save plots
        self._generate_plots(results, base_filename)
    
    def _generate_plots(self, results: List[BenchmarkResult], base_filename: str) -> None:
        """
        Generate plots from benchmark results.
        
        Args:
            results: List of benchmark results
            base_filename: Base filename for plots
        """
        try:
            # Group results by batch size and sequence length
            batch_sizes = sorted(set(r.batch_size for r in results))
            seq_lengths = sorted(set(r.sequence_length for r in results))
            
            # Plot tokens per second vs batch size
            plt.figure(figsize=(10, 6))
            for seq_len in seq_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    x = [r.batch_size for r in seq_results]
                    y = [r.tokens_per_second for r in seq_results]
                    plt.plot(x, y, marker='o', label=f"Seq Length {seq_len}")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Tokens per Second")
            plt.title(f"Performance by Batch Size - {results[0].model_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f"{base_filename}_throughput.png")
            
            # Plot first token latency vs batch size
            plt.figure(figsize=(10, 6))
            for seq_len in seq_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    x = [r.batch_size for r in seq_results]
                    y = [r.first_token_latency_ms for r in seq_results]
                    plt.plot(x, y, marker='o', label=f"Seq Length {seq_len}")
            
            plt.xlabel("Batch Size")
            plt.ylabel("First Token Latency (ms)")
            plt.title(f"Latency by Batch Size - {results[0].model_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f"{base_filename}_latency.png")
            
            # Plot memory usage vs batch size
            plt.figure(figsize=(10, 6))
            for seq_len in seq_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    x = [r.batch_size for r in seq_results]
                    y = [r.memory_usage_mb for r in seq_results]
                    plt.plot(x, y, marker='o', label=f"Seq Length {seq_len}")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Memory Usage (MB)")
            plt.title(f"Memory Usage by Batch Size - {results[0].model_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f"{base_filename}_memory.png")
            
            logger.info(f"Generated plots in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
    
    def _get_gpu_utilization(self) -> float:
        """
        Get current GPU utilization percentage.
        
        Returns:
            GPU utilization percentage
        """
        try:
            if self.cuda_available:
                # Try nvidia-smi through subprocess
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode == 0:
                    # Parse the output which should be a percentage
                    try:
                        util = float(result.stdout.strip())
                        return util
                    except ValueError:
                        pass
            
            # Default if we can't get GPU utilization
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information.
        
        Returns:
            Dictionary of hardware information
        """
        info = {}
        
        try:
            import platform
            info["platform"] = platform.platform()
            info["processor"] = platform.processor()
            
            if self.cuda_available:
                info["cuda_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                
                # Try to get GPU memory
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        try:
                            memory = float(result.stdout.strip())
                            info["gpu_memory_mb"] = memory
                        except ValueError:
                            pass
                except Exception:
                    pass
            else:
                info["cuda_available"] = False
                
            # Get CPU info
            try:
                import psutil
                info["cpu_count"] = psutil.cpu_count(logical=False)
                info["cpu_count_logical"] = psutil.cpu_count(logical=True)
                info["system_memory_gb"] = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Error getting hardware info: {str(e)}")
        
        return info
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dictionary of environment information
        """
        info = {}
        
        try:
            import platform
            import sys
            
            info["python_version"] = platform.python_version()
            info["python_implementation"] = platform.python_implementation()
            
            # Get package versions
            try:
                import torch
                info["torch_version"] = torch.__version__
                
                import transformers
                info["transformers_version"] = transformers.__version__
                
                try:
                    import onnxruntime
                    info["onnxruntime_version"] = onnxruntime.__version__
                except ImportError:
                    pass
                
                try:
                    import tensorrt
                    info["tensorrt_version"] = tensorrt.__version__
                except ImportError:
                    pass
                
                try:
                    import auto_gptq
                    info["auto_gptq_version"] = auto_gptq.__version__
                except ImportError:
                    pass
                
                try:
                    import awq
                    info["awq_version"] = awq.__version__
                except ImportError:
                    pass
            
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Error getting environment info: {str(e)}")
        
        return info

    def compare_models(
        self, 
        results_files: List[str], 
        output_file: Optional[str] = None
    ) -> None:
        """
        Compare multiple model benchmark results.
        
        Args:
            results_files: List of result file paths
            output_file: Path to save comparison plots
        """
        # Load result files
        all_results = []
        for file_path in results_files:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Results file not found: {file_path}")
                continue
                
            if path.suffix == ".json":
                # Load JSON results
                try:
                    with open(path, 'r') as f:
                        results = json.load(f)
                    
                    # Convert dictionaries to BenchmarkResult objects
                    results = [BenchmarkResult(**r) for r in results]
                    all_results.extend(results)
                
                except Exception as e:
                    logger.error(f"Error loading JSON results from {file_path}: {str(e)}")
                    
            elif path.suffix == ".csv":
                # Load CSV results
                try:
                    with open(path, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Convert string values to appropriate types
                            for k, v in row.items():
                                if k in ["batch_size", "sequence_length", "total_tokens"]:
                                    row[k] = int(v)
                                elif k in ["first_token_latency_ms", "tokens_per_second", 
                                          "memory_usage_mb", "gpu_utilization_percent"]:
                                    row[k] = float(v)
                            
                            # Add empty dictionaries for nested fields
                            row["hardware_info"] = {}
                            row["environment_info"] = {}
                            row["additional_metrics"] = {}
                            
                            result = BenchmarkResult(**row)
                            all_results.append(result)
                
                except Exception as e:
                    logger.error(f"Error loading CSV results from {file_path}: {str(e)}")
        
        if not all_results:
            logger.error("No results loaded for comparison")
            return
        
        # Generate comparison plots
        self._generate_comparison_plots(all_results, output_file)
    
    def _generate_comparison_plots(
        self, 
        results: List[BenchmarkResult], 
        output_file: Optional[str] = None
    ) -> None:
        """
        Generate comparison plots for multiple models.
        
        Args:
            results: List of all benchmark results
            output_file: Output file path for plots
        """
        if not output_file:
            output_file = str(self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Group results by model name
        model_results = {}
        for result in results:
            key = f"{result.model_name}_{result.format}_{result.precision}"
            if key not in model_results:
                model_results[key] = []
            model_results[key].append(result)
        
        # Plot tokens per second comparison
        plt.figure(figsize=(12, 8))
        
        # For each model, find results with batch size = 1
        batch_size = 1
        for model_key, model_data in model_results.items():
            batch_results = [r for r in model_data if r.batch_size == batch_size]
            if batch_results:
                x = [r.sequence_length for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                y = [r.tokens_per_second for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                plt.plot(x, y, marker='o', label=model_key)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Tokens per Second")
        plt.title(f"Performance Comparison (Batch Size = {batch_size})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_file}_throughput.png")
        
        # Plot first token latency comparison
        plt.figure(figsize=(12, 8))
        
        for model_key, model_data in model_results.items():
            batch_results = [r for r in model_data if r.batch_size == batch_size]
            if batch_results:
                x = [r.sequence_length for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                y = [r.first_token_latency_ms for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                plt.plot(x, y, marker='o', label=model_key)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("First Token Latency (ms)")
        plt.title(f"Latency Comparison (Batch Size = {batch_size})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_file}_latency.png")
        
        # Plot memory usage comparison
        plt.figure(figsize=(12, 8))
        
        for model_key, model_data in model_results.items():
            batch_results = [r for r in model_data if r.batch_size == batch_size]
            if batch_results:
                x = [r.sequence_length for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                y = [r.memory_usage_mb for r in sorted(batch_results, key=lambda r: r.sequence_length)]
                plt.plot(x, y, marker='o', label=model_key)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"Memory Usage Comparison (Batch Size = {batch_size})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_file}_memory.png")
        
        logger.info(f"Generated comparison plots with prefix {output_file}")


def main():
    """Main function to run the model benchmarking tool."""
    parser = argparse.ArgumentParser(description="Benchmark LLM models in various formats")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing models")
    parser.add_argument("--output-dir", type=str, help="Directory to save benchmark results")
    parser.add_argument("--model", type=str, help="Model to benchmark (format: name_format_precision)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4", 
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--sequence-lengths", type=str, default="128,512,1024", 
                        help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--output-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--prompts-file", type=str, help="File containing benchmark prompts")
    parser.add_argument("--compare", type=str, help="Comma-separated list of result files to compare")
    parser.add_argument("--compare-output", type=str, help="Output file prefix for comparison plots")
    
    args = parser.parse_args()
    
    try:
        benchmark = ModelBenchmark(
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            benchmark_prompts_file=args.prompts_file
        )
        
        if args.list:
            # List available models
            models = benchmark.get_available_models()
            print("\nAvailable Models for Benchmarking:")
            print("=" * 80)
            for model in models:
                print(f"- Name: {model['name']}")
                print(f"  Type: {model['type']}")
                print(f"  Format: {model['format']}")
                print(f"  Precision: {model['precision']}")
                print(f"  Path: {model['path']}")
                if "quantization_method" in model:
                    print(f"  Quantization: {model['quantization_method']}")
                print()
        
        elif args.compare:
            # Compare results from multiple models
            results_files = args.compare.split(",")
            benchmark.compare_models(results_files, args.compare_output)
        
        elif args.model:
            # Parse batch sizes and sequence lengths
            batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
            sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(",")]
            
            # Find the model
            models = benchmark.get_available_models()
            model_info = None
            for model in models:
                if model["name"] == args.model:
                    model_info = model
                    break
            
            if not model_info:
                logger.error(f"Model not found: {args.model}")
                print("Available models:")
                for model in models:
                    print(f"- {model['name']}")
                sys.exit(1)
            
            # Run benchmark
            results = benchmark.benchmark_model(
                model_info=model_info,
                batch_sizes=batch_sizes,
                sequence_lengths=sequence_lengths,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                use_cuda=not args.no_cuda,
                output_tokens=args.output_tokens
            )
            
            if not results:
                logger.error("Benchmark failed to produce results")
                sys.exit(1)
            
            # Print summary
            print("\nBenchmark Results Summary:")
            print("=" * 80)
            for result in results:
                print(f"- Config: Batch={result.batch_size}, SeqLen={result.sequence_length}")
                print(f"  First Token Latency: {result.first_token_latency_ms:.2f} ms")
                print(f"  Throughput: {result.tokens_per_second:.2f} tokens/sec")
                print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")
                print(f"  GPU Utilization: {result.gpu_utilization_percent:.2f}%")
                print()
            
            print(f"Detailed results and plots saved to {args.output_dir}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()