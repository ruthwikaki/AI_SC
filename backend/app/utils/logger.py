"""Logger configuration without circular imports"""
import logging
import sys
from pathlib import Path

# Logger configuration without importing config
class LoggerSetup:
    _logger = None
    
    @classmethod
    def get_logger(cls, name: str = __name__):
        """Get or create logger with lazy config loading"""
        if cls._logger is None:
            cls._setup_logger()
        return logging.getLogger(name)
    
    @classmethod
    def _setup_logger(cls):
        """Setup logger configuration"""
        # Lazy load settings
        try:
            from ..config import get_settings
            settings = get_settings()
            log_level = settings.LOG_LEVEL
        except:
            log_level = "INFO"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        cls._logger = logging.getLogger()

# Export function
def get_logger(name: str = __name__):
    return LoggerSetup.get_logger(name)

__all__ = ['get_logger']
