from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime, timedelta
import json


from app.llm.controller.active_model_manager import get_active_model, fallback_to_backup_model, get_available_models, set_active_model
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


# Health check status
_health_status = {
    "last_check": None,
    "is_healthy": False,
    "issues": [],
    "check_count": 0,
    "failure_count": 0,
    "consecutive_failures": 0
}


# Health check task
_health_check_task = None


async def start_health_checker() -> None:
    """
    Start the health check background task.
    
    This should be called during application startup.
    """
    global _health_check_task
    
    if _health_check_task is None or _health_check_task.done():
        _health_check_task = asyncio.create_task(_health_check_loop())
        logger.info("Started LLM health checker")


async def stop_health_checker() -> None:
    """
    Stop the health check background task.
    
    This should be called during application shutdown.
    """
    global _health_check_task
    
    if _health_check_task and not _health_check_task.done():
        _health_check_task.cancel()
        try:
            await _health_check_task
        except asyncio.CancelledError:
            pass
        
        _health_check_task = None
        logger.info("Stopped LLM health checker")


async def _health_check_loop() -> None:
    """
    Background task loop for regular health checks.
    """
    check_interval = settings.llm_health_check_interval or 300  # Default: 5 minutes
    
    while True:
        try:
            await check_model_health()
            
            # Wait for next check
            await asyncio.sleep(check_interval)
            
        except asyncio.CancelledError:
            # Task was cancelled
            break
        except Exception as e:
            logger.error(f"Error in health check loop: {str(e)}")
            # Wait a bit before retrying
            await asyncio.sleep(60)


async def check_model_health() -> Dict[str, Any]:
    """
    Perform a health check on the active model.
    
    Returns:
        Health check result
    """
    global _health_status
    
    model = get_active_model()
    
    if not model:
        _update_health_status(False, ["No active model"])
        return _get_health_status()
    
    try:
        # Run health check on model
        start_time = time.time()
        health_result = await model.health_check()
        latency = time.time() - start_time
        
        is_healthy = health_result.get("is_healthy", False)
        issues = health_result.get("issues", [])
        
        # Check latency threshold
        max_latency = settings.llm_max_health_check_latency or 5.0  # Default: 5 seconds
        if latency > max_latency:
            is_healthy = False
            issues.append(f"Health check latency too high: {latency:.2f}s > {max_latency:.2f}s")
        
        # Update health status
        _update_health_status(is_healthy, issues)
        
        # Check if we need to fallback to backup model
        if not is_healthy:
            max_failures = settings.llm_max_consecutive_failures or 3
            
            if _health_status["consecutive_failures"] >= max_failures:
                logger.warning(f"Model {model.name} has failed health checks {_health_status['consecutive_failures']} times in a row")
                
                # Try to fallback to backup model
                await fallback_to_backup_model()
        
        return _get_health_status()
        
    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        _update_health_status(False, [f"Error checking health: {str(e)}"])
        return _get_health_status()


def _update_health_status(is_healthy: bool, issues: List[str]) -> None:
    """
    Update the health status.
    
    Args:
        is_healthy: Whether the model is healthy
        issues: List of issues found
    """
    global _health_status
    
    _health_status["last_check"] = datetime.now()
    _health_status["is_healthy"] = is_healthy
    _health_status["issues"] = issues
    _health_status["check_count"] += 1
    
    if not is_healthy:
        _health_status["failure_count"] += 1
        _health_status["consecutive_failures"] += 1
    else:
        _health_status["consecutive_failures"] = 0


def _get_health_status() -> Dict[str, Any]:
    """
    Get the current health status.
    
    Returns:
        Health status dictionary
    """
    status = dict(_health_status)
    
    # Format datetime for serialization
    if status["last_check"]:
        status["last_check"] = status["last_check"].isoformat()
    
    # Add active model info
    model = get_active_model()
    status["active_model"] = model.name if model else None
    
    return status


async def run_canary_test() -> Dict[str, Any]:
    """
    Run a canary test on the active model.
    
    This test sends a simple prompt to validate the model is working correctly.
    
    Returns:
        Canary test results
    """
    model = get_active_model()
    
    if not model:
        return {
            "success": False,
            "error": "No active model",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Simple canary prompt that should have predictable output
        canary_prompt = "Complete this sequence: 1, 2, 3, 4, 5, ..."
        
        # Measure response time
        start_time = time.time()
        response = await model.generate(prompt=canary_prompt)
        latency = time.time() - start_time
        
        # Basic validation of response
        success = False
        error = None
        
        if "6" in response.text:
            success = True
        else:
            error = "Unexpected response"
        
        result = {
            "success": success,
            "error": error,
            "model_name": model.name,
            "latency_ms": latency * 1000,
            "tokens_used": response.tokens_used,
            "timestamp": datetime.now().isoformat(),
            "response_excerpt": response.text[:100] if response.text else None
        }
        
        # Log result
        if success:
            logger.info(f"Canary test passed for model {model.name} (latency: {latency:.2f}s)")
        else:
            logger.warning(f"Canary test failed for model {model.name}: {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running canary test: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model.name if model else None,
            "timestamp": datetime.now().isoformat()
        }


async def test_all_models() -> Dict[str, Any]:
    """
    Test all available models.
    
    Returns:
        Test results for all models
    """
    models = get_available_models()
    results = {}
    
    # Remember current active model
    original_model = get_active_model()
    original_model_name = original_model.name if original_model else None
    
    for model_info in models:
        model_name = model_info["name"]
        
        try:
            # Set this model as active
            if await set_active_model(model_name):
                # Run canary test
                result = await run_canary_test()
                results[model_name] = result
            else:
                results[model_name] = {
                    "success": False,
                    "error": "Failed to activate model",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            results[model_name] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Restore original model
    if original_model_name:
        await set_active_model(original_model_name)
    
    return {
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "active_model": original_model_name
    }


async def get_model_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the LLM system.
    
    Returns:
        Status information
    """
    # Get active model
    model = get_active_model()
    
    # Get health status
    health = _get_health_status()
    
    # Get available models
    models = get_available_models()
    
    status = {
        "active_model": model.name if model else None,
        "health_status": health,
        "available_models": models,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add metrics for active model
    if model:
        status["metrics"] = model.get_metrics()
    
    return status