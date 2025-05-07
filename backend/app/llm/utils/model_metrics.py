# app/llm/utils/model_metrics.py

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import asyncio
import statistics
from collections import defaultdict, deque

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ModelMetricsCollector:
    """
    Collector for LLM model performance metrics.
    
    This class tracks various metrics like token usage, latency,
    success rates, and more for LLM model usage.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of requests to keep in history
        """
        self.max_history = max_history
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency_ms = 0
        
        # Detailed histories (for time-series analysis)
        self.latency_history = deque(maxlen=max_history)
        self.token_history = deque(maxlen=max_history)
        self.result_history = deque(maxlen=max_history)
        
        # Metrics by time period
        self.hourly_metrics = defaultdict(lambda: defaultdict(int))
        self.daily_metrics = defaultdict(lambda: defaultdict(int))
        
        # Last reset time
        self.reset_time = datetime.now()
    
    def record_request(
        self,
        success: bool,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        endpoint: str = "generate"
    ) -> None:
        """
        Record metrics for a request.
        
        Args:
            success: Whether the request was successful
            latency_ms: Latency in milliseconds
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: Model name
            endpoint: API endpoint name
        """
        # Update global counters
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_latency_ms += latency_ms
        
        # Record in histories
        timestamp = datetime.now()
        
        self.latency_history.append({
            "timestamp": timestamp,
            "latency_ms": latency_ms,
            "model": model,
            "endpoint": endpoint
        })
        
        self.token_history.append({
            "timestamp": timestamp,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
            "endpoint": endpoint
        })
        
        self.result_history.append({
            "timestamp": timestamp,
            "success": success,
            "model": model,
            "endpoint": endpoint
        })
        
        # Update time-based metrics
        hour_key = timestamp.strftime("%Y-%m-%d %H:00:00")
        day_key = timestamp.strftime("%Y-%m-%d")
        
        # Hourly metrics
        self.hourly_metrics[hour_key]["requests"] += 1
        self.hourly_metrics[hour_key]["success"] += 1 if success else 0
        self.hourly_metrics[hour_key]["tokens"] += prompt_tokens + completion_tokens
        self.hourly_metrics[hour_key]["latency_ms"] += latency_ms
        
        # Daily metrics
        self.daily_metrics[day_key]["requests"] += 1
        self.daily_metrics[day_key]["success"] += 1 if success else 0
        self.daily_metrics[day_key]["tokens"] += prompt_tokens + completion_tokens
        self.daily_metrics[day_key]["latency_ms"] += latency_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate derived metrics
        avg_latency_ms = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        avg_tokens_per_request = self.total_tokens / self.total_requests if self.total_requests > 0 else 0
        
        # Calculate recent metrics (last 50 requests)
        recent_latencies = [item["latency_ms"] for item in list(self.latency_history)[-50:]]
        recent_avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0
        
        recent_success = sum(1 for item in list(self.result_history)[-50:] if item["success"])
        recent_success_rate = recent_success / min(50, len(self.result_history)) if self.result_history else 0
        
        # Assemble metrics
        metrics = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_tokens_per_request": avg_tokens_per_request,
            
            "avg_latency_ms": avg_latency_ms,
            "recent_avg_latency_ms": recent_avg_latency,
            
            "time_period": {
                "from": self.reset_time.isoformat(),
                "to": datetime.now().isoformat()
            }
        }
        
        return metrics
    
    def get_hourly_metrics(self, days: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Get hourly metrics for the specified number of days.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            Dictionary of hourly metrics
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:00:00")
        
        # Filter hourly metrics by cutoff date
        recent_metrics = {
            hour: metrics 
            for hour, metrics in self.hourly_metrics.items() 
            if hour >= cutoff_str
        }
        
        # Calculate additional metrics for each hour
        result = {}
        for hour, metrics in recent_metrics.items():
            requests = metrics["requests"]
            result[hour] = {
                "requests": requests,
                "successful_requests": metrics["success"],
                "failed_requests": requests - metrics["success"],
                "success_rate": metrics["success"] / requests if requests > 0 else 0,
                "total_tokens": metrics["tokens"],
                "avg_tokens_per_request": metrics["tokens"] / requests if requests > 0 else 0,
                "avg_latency_ms": metrics["latency_ms"] / requests if requests > 0 else 0,
            }
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency_ms = 0
        
        self.latency_history.clear()
        self.token_history.clear()
        self.result_history.clear()
        
        self.hourly_metrics.clear()
        self.daily_metrics.clear()
        
        self.reset_time = datetime.now()
        
        logger.info("Model metrics have been reset")

# Global metrics collector instance
metrics_collector = ModelMetricsCollector()

def record_model_metrics(
    success: bool,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    endpoint: str = "generate"
) -> None:
    """
    Record metrics for a model request.
    
    Args:
        success: Whether the request was successful
        latency_ms: Latency in milliseconds
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        model: Model name
        endpoint: API endpoint name
    """
    global metrics_collector
    metrics_collector.record_request(
        success=success,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        endpoint=endpoint
    )

def get_model_metrics() -> Dict[str, Any]:
    """
    Get current model metrics.
    
    Returns:
        Dictionary of metrics
    """
    global metrics_collector
    return metrics_collector.get_metrics()

def get_model_hourly_metrics(days: int = 1) -> Dict[str, Dict[str, Any]]:
    """
    Get hourly model metrics.
    
    Args:
        days: Number of days to retrieve
        
    Returns:
        Dictionary of hourly metrics
    """
    global metrics_collector
    return metrics_collector.get_hourly_metrics(days)

def reset_model_metrics() -> None:
    """Reset all model metrics."""
    global metrics_collector
    metrics_collector.reset()

async def start_metrics_cleanup_task() -> None:
    """Start background task to clean up old metrics data."""
    while True:
        try:
            # Clean up metrics older than 30 days
            cutoff = datetime.now() - timedelta(days=30)
            cutoff_str_hourly = cutoff.strftime("%Y-%m-%d %H:00:00")
            cutoff_str_daily = cutoff.strftime("%Y-%m-%d")
            
            # Clean hourly metrics
            global metrics_collector
            metrics_collector.hourly_metrics = {
                hour: metrics
                for hour, metrics in metrics_collector.hourly_metrics.items()
                if hour >= cutoff_str_hourly
            }
            
            # Clean daily metrics
            metrics_collector.daily_metrics = {
                day: metrics
                for day, metrics in metrics_collector.daily_metrics.items()
                if day >= cutoff_str_daily
            }
            
            # Wait for next cleanup (once per day)
            await asyncio.sleep(86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error in metrics cleanup task: {str(e)}")
            await asyncio.sleep(3600)  # Try again in an hour