#!/usr/bin/env python3
"""
Request handlers for the Supply Chain LLM inference server.

This module provides handlers for different types of inference requests,
managing model loading, input processing, and output generation.
"""

import os
import logging
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference_handlers")

class ModelType(str, Enum):
    """Types of models supported by the inference server."""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    
class ModelPrecision(str, Enum):
    """Precision types for models."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class ModelInfo:
    """Information about a model available for inference."""
    id: str
    name: str
    type: ModelType
    precision: ModelPrecision
    path: str
    optimized: bool = False
    metadata: Dict[str, Any] = None
    loaded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "precision": self.precision,
            "optimized": self.optimized,
            "loaded": self.loaded,
            "metadata": self.metadata or {}
        }

class InferenceHandler:
    """Handles inference requests for various model types."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the inference handler.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        
        # Check if directory exists
        if not self.models_dir.exists():
            raise ValueError(f"Models directory does not exist: {models_dir}")
        
        # Dictionary to store loaded models
        self.loaded_models: Dict[str, Any] = {}
        
        # Dictionary to store model info
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Dictionary to store tokenizers
        self.tokenizers: Dict[str, Any] = {}
        
        # Dictionary to store embedding models
        self.embedding_models: Dict[str, Any] = {}
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models in the models directory.
        
        Returns:
            List of model information
        """
        discovered_models = []
        
        # Check for ONNX models
        onnx_dir = self.models_dir / "onnx"
        if onnx_dir.exists():
            onnx_models = self._discover_onnx_models(onnx_dir)
            discovered_models.extend(onnx_models)
        
        # Check for PyTorch models
        pytorch_models = self._discover_pytorch_models(self.models_dir)
        discovered_models.extend(pytorch_models)
        
        # Check for TensorRT models
        tensorrt_dir = self.models_dir / "tensorrt"
        if tensorrt_dir.exists():
            tensorrt_models = self._discover_tensorrt_models(tensorrt_dir)
            discovered_models.extend(tensorrt_models)
        
        # Store model info
        for model in discovered_models:
            self.model_info[model.id] = model
        
        return discovered_models
    
    def _discover_onnx_models(self, onnx_dir: Path) -> List[ModelInfo]:
        """
        Discover ONNX models.
        
        Args:
            onnx_dir: Directory containing ONNX models
            
        Returns:
            List of ONNX model information
        """
        models = []
        
        # Check for model types
        for model_type_dir in onnx_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name  # mistral, llama3, etc.
            
            # Check for precision versions
            for precision_dir in model_type_dir.iterdir():
                if not precision_dir.is_dir():
                    continue
                
                # Check if this is a standard precision directory (fp16, fp32)
                if precision_dir.name in ["fp16", "fp32"]:
                    precision = precision_dir.name
                    
                    # Check if ONNX model file exists
                    model_path = precision_dir / "model.onnx"
                    if model_path.exists():
                        # Load metadata if available
                        metadata = {}
                        config_path = precision_dir / "onnx_config.json"
                        if config_path.exists():
                            try:
                                with open(config_path, 'r') as f:
                                    metadata = json.load(f)
                            except Exception as e:
                                logger.warning(f"Error loading ONNX config: {str(e)}")
                        
                        # Create model info
                        model_id = f"{model_type}-{precision}-onnx"
                        model_name = f"{model_type.capitalize()} ({precision}, ONNX)"
                        
                        model_info = ModelInfo(
                            id=model_id,
                            name=model_name,
                            type=ModelType.ONNX,
                            precision=ModelPrecision(precision),
                            path=str(precision_dir),
                            optimized=True,  # ONNX models are considered optimized
                            metadata=metadata
                        )
                        
                        models.append(model_info)
                
                # Check if this is a quantized model directory
                elif precision_dir.name.startswith("quantized_"):
                    # Extract precision and method from directory name
                    parts = precision_dir.name.split("_")
                    if len(parts) >= 3:
                        precision = parts[1]
                        method = parts[2]
                        
                        # Check if ONNX model file exists
                        model_path = precision_dir / "model.onnx"
                        if model_path.exists():
                            # Load metadata if available
                            metadata = {}
                            config_path = precision_dir / "onnx_config.json"
                            if config_path.exists():
                                try:
                                    with open(config_path, 'r') as f:
                                        metadata = json.load(f)
                                except Exception as e:
                                    logger.warning(f"Error loading ONNX config: {str(e)}")
                            
                            # Create model info
                            model_id = f"{model_type}-{precision}-{method}-onnx"
                            model_name = f"{model_type.capitalize()} ({precision}, {method}, ONNX)"
                            
                            model_info = ModelInfo(
                                id=model_id,
                                name=model_name,
                                type=ModelType.ONNX,
                                precision=ModelPrecision(precision),
                                path=str(precision_dir),
                                optimized=True,
                                metadata={
                                    "quantization_method": method,
                                    **metadata
                                }
                            )
                            
                            models.append(model_info)
        
        return models
    
    def _discover_pytorch_models(self, models_dir: Path) -> List[ModelInfo]:
        """
        Discover PyTorch models.
        
        Args:
            models_dir: Directory containing models
            
        Returns:
            List of PyTorch model information
        """
        models = []
        
        # Check for model types
        for model_type in ["mistral", "llama3"]:
            model_type_dir = models_dir / model_type
            if not model_type_dir.exists():
                continue
            
            weights_dir = model_type_dir / "weights"
            if not weights_dir.exists():
                continue
            
            # Check if standard model exists
            if weights_dir.exists() and any(weights_dir.iterdir()):
                # Load metadata if available
                metadata = {}
                config_path = model_type_dir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading model config: {str(e)}")
                
                # Create model info
                model_id = f"{model_type}-pytorch"
                model_name = f"{model_type.capitalize()} (PyTorch)"
                
                model_info = ModelInfo(
                    id=model_id,
                    name=model_name,
                    type=ModelType.PYTORCH,
                    precision=ModelPrecision.FP16,  # Assume FP16 by default
                    path=str(weights_dir),
                    metadata=metadata
                )
                
                models.append(model_info)
            
            # Check for quantized versions
            for quant_dir in weights_dir.glob("quantized_*"):
                if not quant_dir.is_dir():
                    continue
                
                # Extract precision and method from directory name
                parts = quant_dir.name.split("_")
                if len(parts) >= 3:
                    precision = parts[1]
                    method = parts[2]
                    
                    # Load metadata if available
                    metadata = {}
                    config_path = model_type_dir / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Error loading model config: {str(e)}")
                    
                    # Create model info
                    model_id = f"{model_type}-{precision}-{method}-pytorch"
                    model_name = f"{model_type.capitalize()} ({precision}, {method}, PyTorch)"
                    
                    model_info = ModelInfo(
                        id=model_id,
                        name=model_name,
                        type=ModelType.PYTORCH,
                        precision=ModelPrecision(precision),
                        path=str(quant_dir),
                        metadata={
                            "quantization_method": method,
                            **metadata
                        }
                    )
                    
                    models.append(model_info)
        
        return models
    
    def _discover_tensorrt_models(self, tensorrt_dir: Path) -> List[ModelInfo]:
        """
        Discover TensorRT models.
        
        Args:
            tensorrt_dir: Directory containing TensorRT models
            
        Returns:
            List of TensorRT model information
        """
        models = []
        
        # Check for model types
        for model_type_dir in tensorrt_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name  # mistral, llama3, etc.
            
            # Check for precision versions
            for precision_dir in model_type_dir.iterdir():
                if not precision_dir.is_dir():
                    continue
                
                precision = precision_dir.name  # fp16, fp32, int8
                
                # Check if engine files exist
                engine_files = list(precision_dir.glob("*.engine"))
                if engine_files:
                    # Load metadata if available
                    metadata = {}
                    config_path = precision_dir / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Error loading TensorRT config: {str(e)}")
                    
                    # Create model info
                    model_id = f"{model_type}-{precision}-tensorrt"
                    model_name = f"{model_type.capitalize()} ({precision}, TensorRT)"
                    
                    model_info = ModelInfo(
                        id=model_id,
                        name=model_name,
                        type=ModelType.TENSORRT,
                        precision=ModelPrecision(precision),
                        path=str(precision_dir),
                        optimized=True,  # TensorRT models are considered optimized
                        metadata=metadata
                    )
                    
                    models.append(model_info)
        
        return models
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available models.
        
        Returns:
            List of model information dictionaries
        """
        return [model.to_dict() for model in self.model_info.values()]
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get model information by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information or None if not found
        """
        return self.model_info.get(model_id)
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            model_id: ID of the model
            
        Returns:
            True if loaded, False otherwise
        """
        return model_id in self.loaded_models
    
    async def load_model(
        self, 
        model_id: str, 
        use_optimized: bool = True, 
        device: str = "cuda"
    ) -> bool:
        """
        Load a model for inference.
        
        Args:
            model_id: ID of the model to load
            use_optimized: Whether to use optimized version if available
            device: Device to load the model on
            
        Returns:
            True if successful, False otherwise
        """
        # Get model info
        model_info = self.model_info.get(model_id)
        if not model_info:
            # Try to discover models
            self.discover_models()
            model_info = self.model_info.get(model_id)
            
            if not model_info:
                logger.error(f"Model {model_id} not found")
                return False
        
        # Check if model already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        # Use lock to prevent concurrent loading
        async with self.lock:
            try:
                model_type = model_info.type
                model_path = model_info.path
                
                # Load model based on type
                if model_type == ModelType.ONNX:
                    model, tokenizer = await self._load_onnx_model(model_info, device)
                elif model_type == ModelType.PYTORCH:
                    model, tokenizer = await self._load_pytorch_model(model_info, device)
                elif model_type == ModelType.TENSORRT:
                    model, tokenizer = await self._load_tensorrt_model(model_info, device)
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    return False
                
                if model is None or tokenizer is None:
                    logger.error(f"Failed to load model {model_id}")
                    return False
                
                # Store model and tokenizer
                self.loaded_models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                
                # Update model info
                model_info.loaded = True
                self.model_info[model_id] = model_info
                
                logger.info(f"Model {model_id} loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    async def _load_onnx_model(
        self, 
        model_info: ModelInfo, 
        device: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load an ONNX model.
        
        Args:
            model_info: Model information
            device: Device to load on
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            model_path = Path(model_info.path) / "model.onnx"
            
            if not model_path.exists():
                logger.error(f"ONNX model not found at {model_path}")
                return None, None
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_info.path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.path,
                use_fast=True
            )
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select provider based on device
            if device.lower() == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Load ONNX model
            logger.info(f"Loading ONNX model from {model_path}...")
            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            return session, tokenizer
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    async def _load_pytorch_model(
        self, 
        model_info: ModelInfo, 
        device: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a PyTorch model.
        
        Args:
            model_info: Model information
            device: Device to load on
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Import here to avoid loading if not needed
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Check if model is quantized
            is_quantized = "quantization_method" in model_info.metadata if model_info.metadata else False
            quant_method = model_info.metadata.get("quantization_method") if model_info.metadata else None
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_info.path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.path,
                use_fast=True
            )
            
            # Handle tokenizer special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on quantization
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
                        return None, None
                
                elif quant_method.lower() == "awq":
                    try:
                        from awq import AutoAWQForCausalLM
                        model = AutoAWQForCausalLM.from_quantized(
                            model_info.path,
                            trust_remote_code=True
                        )
                    except ImportError:
                        logger.error("awq package not installed")
                        return None, None
                
                else:
                    logger.error(f"Unsupported quantization method: {quant_method}")
                    return None, None
            
            else:
                # Determine torch dtype
                if model_info.precision == ModelPrecision.FP16:
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                # Set device_map
                if device.lower() == "cuda" and torch.cuda.is_available():
                    device_map = "auto"
                else:
                    device_map = "cpu"
                
                # Load model
                logger.info(f"Loading PyTorch model ({model_info.precision})...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_info.path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            # Set evaluation mode
            model.eval()
            
            return model, tokenizer
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    async def _load_tensorrt_model(
        self, 
        model_info: ModelInfo, 
        device: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a TensorRT model.
        
        Args:
            model_info: Model information
            device: Device to load on
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Import here to avoid loading if not needed
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            from transformers import AutoTokenizer
            
            # Find engine file
            engine_files = list(Path(model_info.path).glob("*.engine"))
            if not engine_files:
                logger.error(f"No TensorRT engine found in {model_info.path}")
                return None, None
            
            engine_path = engine_files[0]
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_info.path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.path,
                use_fast=True
            )
            
            # Handle tokenizer special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Initialize TensorRT engine
            logger.info(f"Loading TensorRT engine from {engine_path}...")
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Return engine and context as model
            return (engine, context), tokenizer
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading TensorRT model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model is loaded
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded")
            return False
        
        try:
            # Remove model and tokenizer
            del self.loaded_models[model_id]
            del self.tokenizers[model_id]
            
            # Update model info
            if model_id in self.model_info:
                model_info = self.model_info[model_id]
                model_info.loaded = False
                self.model_info[model_id] = model_info
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    def unload_all_models(self) -> bool:
        """
        Unload all loaded models.
        
        Returns:
            True if successful, False otherwise
        """
        success = True
        for model_id in list(self.loaded_models.keys()):
            if not self.unload_model(model_id):
                success = False
        
        return success
    
    async def generate_completion(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text completion from a prompt.
        
        Args:
            model_id: ID of the model to use
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            do_sample: Whether to use sampling or greedy generation
            stop_sequences: Sequences that stop generation
            
        Returns:
            Dictionary with generation results
        """
        # Check if model is loaded
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        # Get model and tokenizer
        model = self.loaded_models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        # Get model info
        model_info = self.model_info[model_id]
        model_type = model_info.type
        
        # Process stop sequences
        stop_sequences = stop_sequences or []
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Move tensors to appropriate device
        if model_type == ModelType.PYTORCH:
            input_ids = input_ids.to(next(model.parameters()).device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(next(model.parameters()).device)
        
        # Count input tokens
        tokens_processed = input_ids.shape[1]
        
        # Generate based on model type
        if model_type == ModelType.PYTORCH:
            result = await self._generate_pytorch(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stop_sequences=stop_sequences
            )
        elif model_type == ModelType.ONNX:
            result = await self._generate_onnx(
                session=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stop_sequences=stop_sequences
            )
        elif model_type == ModelType.TENSORRT:
            result = await self._generate_tensorrt(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stop_sequences=stop_sequences
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Add token counts
        result["tokens_processed"] = tokens_processed
        
        return result
    
    async def generate_chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion from messages.
        
        Args:
            model_id: ID of the model to use
            messages: List of chat messages (role, content)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            do_sample: Whether to use sampling or greedy generation
            stop_sequences: Sequences that stop generation
            
        Returns:
            Dictionary with generation results
        """
        # Check if model is loaded
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        # Get model and tokenizer
        model = self.loaded_models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        # Format messages into prompt using chat template
        try:
            # Apply the chat template if available
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to basic formatting
                prompt = self._format_chat_messages(messages)
        except Exception as e:
            logger.warning(f"Error applying chat template: {str(e)}")
            # Fallback to basic formatting
            prompt = self._format_chat_messages(messages)
        
        # Generate completion using the formatted prompt
        result = await self.generate_completion(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_sequences=stop_sequences
        )
        
        return result
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt.
        
        Args:
            messages: List of chat messages (role, content)
            
        Returns:
            Formatted prompt string
        """
        formatted = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        # Add assistant prefix for the model to continue
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    async def generate_embeddings(
        self,
        model_id: str,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text.
        
        Args:
            model_id: ID of the model to use
            texts: List of texts to embed
            
        Returns:
            Dictionary with embedding results
        """
        # Check if model is loaded as embedding model
        if model_id not in self.embedding_models:
            raise ValueError(f"Embedding model {model_id} not loaded")
        
        # Get model
        model = self.embedding_models[model_id]
        
        # Process texts
        embeddings = []
        tokens_processed = 0
        
        for text in texts:
            # Get embedding
            embedding = model.encode(text)
            embeddings.append(embedding.tolist())
            
            # Count tokens (approximate)
            tokens_processed += len(text.split())
        
        return {
            "embeddings": embeddings,
            "tokens_processed": tokens_processed
        }
    
    async def _generate_pytorch(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        stop_sequences: List[str]
    ) -> Dict[str, Any]:
        """
        Generate text with a PyTorch model.
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            
        Returns:
            Dictionary with generation results
        """
        import torch
        
        # Create stopping criteria for stop sequences
        stopping_criteria = None
        if stop_sequences:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_sequences, input_length):
                    self.tokenizer = tokenizer
                    self.stop_sequences = stop_sequences
                    self.input_length = input_length
                
                def __call__(self, input_ids, scores, **kwargs):
                    # Get the generated text so far
                    generated_ids = input_ids[0, self.input_length:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Check if any stop sequence appears in the generated text
                    for stop_seq in self.stop_sequences:
                        if stop_seq in generated_text:
                            return True
                    
                    return False
            
            stopping_criteria = StoppingCriteriaList([
                StopSequenceCriteria(tokenizer, stop_sequences, input_ids.shape[1])
            ])
        
        # Set generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask if attention_mask is not None else None,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": top_p if do_sample else 1.0,
            "top_k": top_k if do_sample else None,
            "temperature": temperature if do_sample else 1.0,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria
        }
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # Generate tokens
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
        
        # Extract generated tokens (remove input tokens)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        
        # Decode text
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_seq)]
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_ids)
        }
    
    async def _generate_onnx(
        self,
        session,
        tokenizer,
        input_ids,
        attention_mask,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        stop_sequences: List[str]
    ) -> Dict[str, Any]:
        """
        Generate text with an ONNX model.
        
        Args:
            session: ONNX Runtime session
            tokenizer: Tokenizer
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            
        Returns:
            Dictionary with generation results
        """
        import numpy as np
        
        # Get ONNX input and output names
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        # Prepare inputs
        onnx_inputs = {}
        for name in input_names:
            if name == "input_ids":
                onnx_inputs[name] = input_ids.cpu().numpy()
            elif name == "attention_mask":
                onnx_inputs[name] = attention_mask.cpu().numpy() if attention_mask is not None else np.ones_like(input_ids.cpu().numpy())
            else:
                # Handle other possible inputs
                logger.warning(f"Unhandled ONNX input: {name}")
        
        # Initialize generated token ids
        all_token_ids = input_ids.cpu().numpy()
        
        # Track generated text for stop sequences
        generated_text = ""
        
        # Generate tokens one by one
        for _ in range(max_tokens):
            # Run inference
            onnx_outputs = session.run(output_names, onnx_inputs)
            
            # Get logits from output
            logits = onnx_outputs[0] if "logits" in output_names else onnx_outputs[0]
            
            # Get next token logits (last token)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for sequence_idx in range(next_token_logits.shape[0]):
                    previous_tokens = all_token_ids[sequence_idx].tolist()
                    for prev_token in set(previous_tokens):
                        next_token_logits[sequence_idx, prev_token] /= repetition_penalty
            
            # Get next token
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_indices = np.argsort(-next_token_logits, axis=-1)[:, :top_k]
                    indices_to_remove = np.ones_like(next_token_logits, dtype=bool)
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove[batch_idx, top_k_indices[batch_idx]] = False
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits = np.sort(-next_token_logits, axis=-1)
                    sorted_indices = np.argsort(-next_token_logits, axis=-1)
                    cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits), axis=-1, keepdims=True), axis=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to remove by one to the right
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                    sorted_indices_to_remove[:, 0] = False
                    
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                
                # Convert to probabilities and sample
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits), axis=-1, keepdims=True)
                next_token_id = np.array([np.random.choice(probs.shape[1], p=probs[0])])
            else:
                # Greedy decoding
                next_token_id = np.argmax(next_token_logits, axis=-1)
            
            # Add to generated tokens
            next_token_id = next_token_id.reshape(-1, 1)
            all_token_ids = np.concatenate([all_token_ids, next_token_id], axis=1)
            
            # Update inputs for next iteration
            onnx_inputs["input_ids"] = all_token_ids
            if "attention_mask" in onnx_inputs:
                onnx_inputs["attention_mask"] = np.concatenate(
                    [onnx_inputs["attention_mask"], np.ones_like(next_token_id)], 
                    axis=1
                )
            
            # Check for stop sequences
            new_token_text = tokenizer.decode(next_token_id[0])
            generated_text += new_token_text
            
            if any(stop_seq in generated_text for stop_seq in stop_sequences):
                break
            
            # Check for EOS token
            if next_token_id[0, 0] == tokenizer.eos_token_id:
                break
        
        # Get generated text (remove input)
        input_length = input_ids.shape[1]
        generated_ids = all_token_ids[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_seq)]
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_ids)
        }
    
    async def _generate_tensorrt(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        stop_sequences: List[str]
    ) -> Dict[str, Any]:
        """
        Generate text with a TensorRT model.
        
        Args:
            model: TensorRT model tuple (engine, context)
            tokenizer: Tokenizer
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            
        Returns:
            Dictionary with generation results
        """
        import numpy as np
        import pycuda.driver as cuda
        import tensorrt as trt
        
        # Unpack model
        engine, context = model
        
        # Get engine bindings
        binding_names = []
        for i in range(engine.num_bindings):
            binding_names.append(engine.get_binding_name(i))
        
        # Prepare inputs
        input_dict = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy() if attention_mask is not None else np.ones_like(input_ids.cpu().numpy())
        }
        
        # Initialize generated token ids
        all_token_ids = input_ids.cpu().numpy()
        
        # Track generated text for stop sequences
        generated_text = ""
        
        # Generate tokens one by one
        for _ in range(max_tokens):
            # Allocate buffers
            bindings = [None] * engine.num_bindings
            
            # Process each binding
            outputs = {}
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                
                if engine.binding_is_input(i):
                    # Input binding
                    if binding_name in input_dict:
                        input_data = input_dict[binding_name]
                        # Allocate device memory
                        input_mem = cuda.mem_alloc(input_data.nbytes)
                        # Copy data to device
                        cuda.memcpy_htod(input_mem, input_data)
                        bindings[i] = int(input_mem)
                else:
                    # Output binding
                    shape = engine.get_binding_shape(i)
                    dtype = np.float32  # Assuming float32 output
                    size = trt.volume(shape) * np.dtype(dtype).itemsize
                    output_mem = cuda.mem_alloc(size)
                    bindings[i] = int(output_mem)
                    outputs[binding_name] = {"mem": output_mem, "shape": shape, "dtype": dtype}
            
            # Execute inference
            context.execute_v2(bindings)
            
            # Process outputs
            logits = None
            for name, out in outputs.items():
                if "logits" in name:
                    # Assuming output is logits
                    shape = out["shape"]
                    output = np.empty(shape, dtype=out["dtype"])
                    cuda.memcpy_dtoh(output, out["mem"])
                    logits = output
            
            if logits is None:
                logger.error("No logits found in TensorRT outputs")
                break
            
            # Get next token logits (last token)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Get next token
            if do_sample:
                # Sampling implementation (similar to ONNX version)
                # Apply top-k filtering
                if top_k > 0:
                    top_k_indices = np.argsort(-next_token_logits, axis=-1)[:, :top_k]
                    indices_to_remove = np.ones_like(next_token_logits, dtype=bool)
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove[batch_idx, top_k_indices[batch_idx]] = False
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits = np.sort(-next_token_logits, axis=-1)
                    sorted_indices = np.argsort(-next_token_logits, axis=-1)
                    cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits), axis=-1, keepdims=True), axis=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                    sorted_indices_to_remove[:, 0] = False
                    
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                
                # Convert to probabilities and sample
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits), axis=-1, keepdims=True)
                next_token_id = np.array([np.random.choice(probs.shape[1], p=probs[0])])
            else:
                # Greedy decoding
                next_token_id = np.argmax(next_token_logits, axis=-1)
            
            # Add to generated tokens
            next_token_id = np.array([[next_token_id]])
            all_token_ids = np.concatenate([all_token_ids, next_token_id], axis=1)
            
            # Update inputs for next iteration
            input_dict["input_ids"] = all_token_ids
            input_dict["attention_mask"] = np.concatenate(
                [input_dict["attention_mask"], np.ones_like(next_token_id)], 
                axis=1
            )
            
            # Check for stop sequences
            new_token_text = tokenizer.decode(next_token_id[0])
            generated_text += new_token_text
            
            if any(stop_seq in generated_text for stop_seq in stop_sequences):
                break
            
            # Check for EOS token
            if next_token_id[0, 0] == tokenizer.eos_token_id:
                break
        
        # Get generated text (remove input)
        input_length = input_ids.shape[1]
        generated_ids = all_token_ids[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_seq)]
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_ids)
        }

# Helper functions for softmax operations
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()