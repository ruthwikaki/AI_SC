#!/usr/bin/env python3
"""
Inference server for the Supply Chain LLM system.

This server provides a high-performance API for model inference using ONNX and TensorRT,
optimized for production use in supply chain analytics applications.
"""

import os
import sys
import argparse
import logging
import json
import time
import asyncio
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import handlers and optimizers
from handlers import InferenceHandler, ModelInfo
from optimizers import ModelOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference_server")

# Define models for API
class GenerationParams(BaseModel):
    """Parameters for text generation."""
    max_tokens: int = Field(default=100, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling or greedy generation")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences that stop generation")
    stream: bool = Field(default=False, description="Whether to stream the response")

class CompletionRequest(BaseModel):
    """Request for text completion."""
    prompt: str = Field(..., description="Input prompt for completion")
    model_id: str = Field(..., description="ID of the model to use")
    params: Optional[GenerationParams] = Field(default=None, description="Generation parameters")

class MessageContent(BaseModel):
    """Content of a chat message."""
    type: str = Field(default="text", description="Type of content")
    text: str = Field(..., description="Text content")

class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: Union[str, List[MessageContent]] = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model_id: str = Field(..., description="ID of the model to use")
    params: Optional[GenerationParams] = Field(default=None, description="Generation parameters")

class CompletionResponse(BaseModel):
    """Response for text completion."""
    generated_text: str = Field(..., description="Generated text")
    model_id: str = Field(..., description="ID of the model used")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_processed: int = Field(..., description="Number of tokens processed in the prompt")
    generation_time_ms: float = Field(..., description="Time taken for generation in milliseconds")

class CompletionStreamResponse(BaseModel):
    """Response chunk for streamed text completion."""
    token: str = Field(..., description="Generated token")
    is_finished: bool = Field(default=False, description="Whether this is the final token")

class ChatResponse(BaseModel):
    """Response for chat completion."""
    message: ChatMessage = Field(..., description="Generated assistant message")
    model_id: str = Field(..., description="ID of the model used")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_processed: int = Field(..., description="Number of tokens processed in the messages")
    generation_time_ms: float = Field(..., description="Time taken for generation in milliseconds")

class EmbeddingRequest(BaseModel):
    """Request for text embedding."""
    text: Union[str, List[str]] = Field(..., description="Text to embed")
    model_id: str = Field(..., description="ID of the embedding model to use")

class EmbeddingResponse(BaseModel):
    """Response for text embedding."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model_id: str = Field(..., description="ID of the model used")
    tokens_processed: int = Field(..., description="Number of tokens processed")
    embedding_time_ms: float = Field(..., description="Time taken for embedding in milliseconds")

class HealthResponse(BaseModel):
    """Response for health check."""
    status: str = Field(..., description="Server status")
    models: List[Dict[str, Any]] = Field(..., description="Available models")
    version: str = Field(..., description="Server version")

# Application state
class AppState:
    """Application state container."""
    def __init__(self):
        self.inference_handler = None
        self.model_optimizer = None
        self.models_loaded = False
        self.models_loading = False
        self.models_dir = None
        
app_state = AppState()

# Application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing inference server...")
    # State is initialized but models not loaded yet
    
    # Startup complete
    yield
    
    # Shutdown
    logger.info("Shutting down inference server...")
    if app_state.inference_handler:
        app_state.inference_handler.unload_all_models()

# Create FastAPI app
app = FastAPI(
    title="Supply Chain LLM Inference Server",
    description="High-performance API for LLM inference in supply chain applications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and get available models."""
    # Check if models are loaded
    if not app_state.models_loaded:
        if not app_state.models_loading:
            return HealthResponse(
                status="initializing",
                models=[],
                version="1.0.0"
            )
        else:
            return HealthResponse(
                status="loading_models",
                models=[],
                version="1.0.0"
            )
    
    # Get available models
    models = []
    if app_state.inference_handler:
        models = app_state.inference_handler.get_model_info()
    
    return HealthResponse(
        status="healthy",
        models=models,
        version="1.0.0"
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    """Generate text completion from a prompt."""
    # Check if models are loaded
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still being loaded")
    
    # Check if requested model is available
    model_id = request.model_id
    if not app_state.inference_handler.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Set default parameters if not provided
    if request.params is None:
        request.params = GenerationParams()
    
    try:
        # Start timing
        start_time = time.time()
        
        # Run generation
        result = await app_state.inference_handler.generate_completion(
            model_id=model_id,
            prompt=request.prompt,
            max_tokens=request.params.max_tokens,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            top_k=request.params.top_k,
            repetition_penalty=request.params.repetition_penalty,
            do_sample=request.params.do_sample,
            stop_sequences=request.params.stop_sequences
        )
        
        # Calculate timing
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Return response
        return CompletionResponse(
            generated_text=result["text"],
            model_id=model_id,
            tokens_generated=result["tokens_generated"],
            tokens_processed=result["tokens_processed"],
            generation_time_ms=generation_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def generate_chat_completion(request: ChatRequest):
    """Generate chat completion from messages."""
    # Check if models are loaded
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still being loaded")
    
    # Check if requested model is available
    model_id = request.model_id
    if not app_state.inference_handler.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Set default parameters if not provided
    if request.params is None:
        request.params = GenerationParams()
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process messages to compatible format
        processed_messages = []
        for msg in request.messages:
            # Handle complex content (list of content objects)
            if isinstance(msg.content, list):
                # Extract just the text content for now
                text_contents = [item.text for item in msg.content if item.type == "text"]
                content = " ".join(text_contents)
            else:
                content = msg.content
                
            processed_messages.append({
                "role": msg.role,
                "content": content
            })
        
        # Run generation
        result = await app_state.inference_handler.generate_chat_completion(
            model_id=model_id,
            messages=processed_messages,
            max_tokens=request.params.max_tokens,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            top_k=request.params.top_k,
            repetition_penalty=request.params.repetition_penalty,
            do_sample=request.params.do_sample,
            stop_sequences=request.params.stop_sequences
        )
        
        # Calculate timing
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Construct response message
        response_message = ChatMessage(
            role="assistant",
            content=result["text"]
        )
        
        # Return response
        return ChatResponse(
            message=response_message,
            model_id=model_id,
            tokens_generated=result["tokens_generated"],
            tokens_processed=result["tokens_processed"],
            generation_time_ms=generation_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text."""
    # Check if models are loaded
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still being loaded")
    
    # Check if requested model is available
    model_id = request.model_id
    if not app_state.inference_handler.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process input
        texts = request.text if isinstance(request.text, list) else [request.text]
        
        # Run embedding generation
        result = await app_state.inference_handler.generate_embeddings(
            model_id=model_id,
            texts=texts
        )
        
        # Calculate timing
        embedding_time_ms = (time.time() - start_time) * 1000
        
        # Return response
        return EmbeddingResponse(
            embeddings=result["embeddings"],
            model_id=model_id,
            tokens_processed=result["tokens_processed"],
            embedding_time_ms=embedding_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/load_models")
async def load_models(
    background_tasks: BackgroundTasks,
    models_dir: str,
    model_ids: Optional[List[str]] = None,
    force_reload: bool = False,
    optimize: bool = True
):
    """Load models from directory (admin endpoint)."""
    # Check if already loading
    if app_state.models_loading:
        return {"status": "models_loading", "message": "Models are already being loaded"}
    
    # Start loading in background
    background_tasks.add_task(
        _load_models_task,
        models_dir=models_dir,
        model_ids=model_ids,
        force_reload=force_reload,
        optimize=optimize
    )
    
    return {"status": "loading_started", "message": "Model loading started in background"}

@app.post("/admin/unload_model")
async def unload_model(model_id: str):
    """Unload a specific model (admin endpoint)."""
    # Check if models are loaded
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still being loaded")
    
    try:
        app_state.inference_handler.unload_model(model_id)
        return {"status": "success", "message": f"Model {model_id} unloaded"}
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/optimize_model")
async def optimize_model(
    background_tasks: BackgroundTasks,
    model_id: str,
    precision: str = "fp16",
    device: str = "cuda"
):
    """Optimize a specific model (admin endpoint)."""
    # Check if models are loaded
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still being loaded")
    
    # Check if model optimizer is available
    if not app_state.model_optimizer:
        raise HTTPException(status_code=503, detail="Model optimizer not initialized")
    
    # Start optimization in background
    background_tasks.add_task(
        _optimize_model_task,
        model_id=model_id,
        precision=precision,
        device=device
    )
    
    return {"status": "optimization_started", "message": f"Optimization of model {model_id} started in background"}

# Background tasks
async def _load_models_task(
    models_dir: str,
    model_ids: Optional[List[str]] = None,
    force_reload: bool = False,
    optimize: bool = True
):
    """Task to load models in the background."""
    try:
        # Set loading flag
        app_state.models_loading = True
        app_state.models_dir = models_dir
        
        # Initialize inference handler if not already
        if app_state.inference_handler is None:
            app_state.inference_handler = InferenceHandler(models_dir)
        
        # Initialize model optimizer if not already
        if app_state.model_optimizer is None and optimize:
            app_state.model_optimizer = ModelOptimizer(models_dir)
        
        # Load all available models if no specific IDs provided
        if model_ids is None:
            # Discover available models
            available_models = app_state.inference_handler.discover_models()
            model_ids = [model.id for model in available_models]
        
        # Load each model
        for model_id in model_ids:
            if force_reload and app_state.inference_handler.is_model_loaded(model_id):
                app_state.inference_handler.unload_model(model_id)
                
            if not app_state.inference_handler.is_model_loaded(model_id):
                logger.info(f"Loading model {model_id}...")
                await app_state.inference_handler.load_model(model_id)
                logger.info(f"Model {model_id} loaded successfully")
        
        # Set loaded flag
        app_state.models_loaded = True
        app_state.models_loading = False
        
        logger.info(f"All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Reset flags
        app_state.models_loading = False

async def _optimize_model_task(model_id: str, precision: str, device: str):
    """Task to optimize a model in the background."""
    try:
        logger.info(f"Starting optimization of model {model_id}...")
        
        # Check if model is loaded
        if not app_state.inference_handler.is_model_loaded(model_id):
            logger.error(f"Model {model_id} is not loaded, cannot optimize")
            return
        
        # Get model info
        model_info = app_state.inference_handler.get_model_by_id(model_id)
        if not model_info:
            logger.error(f"Model {model_id} info not found")
            return
        
        # Optimize model
        success = await app_state.model_optimizer.optimize_model(
            model_info=model_info,
            precision=precision,
            device=device
        )
        
        if success:
            logger.info(f"Model {model_id} optimized successfully")
            
            # Reload the optimized model
            app_state.inference_handler.unload_model(model_id)
            await app_state.inference_handler.load_model(model_id, use_optimized=True)
            
            logger.info(f"Optimized model {model_id} loaded successfully")
        else:
            logger.error(f"Failed to optimize model {model_id}")
        
    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Main entry point
def main():
    """Main function to run the inference server."""
    parser = argparse.ArgumentParser(description="Run the inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", type=str, help="Directory containing models")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    if args.models_dir:
        logger.info(f"Auto-loading models from {args.models_dir}")
        asyncio.run(_load_models_task(models_dir=args.models_dir))
    
    # Configure Uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Run server
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        log_config=log_config
    )

if __name__ == "__main__":
    main()