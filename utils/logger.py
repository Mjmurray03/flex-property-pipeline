"""
Logging utility for Flex Property Pipeline
Provides colored console output and file logging with rotation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Try to import colorlog, fall back to standard logging if not available
try:
    import colorlog
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

def setup_logging(
    name: str = None,
    level: str = 'INFO',
    log_file: str = None,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up logging with both console and file output
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/pipeline_YYYYMMDD.log)
        console: Enable console logging
        file_logging: Enable file logging
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create logs directory if needed
    if file_logging:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        if log_file is None:
            log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Console handler with color support
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if HAS_COLOR:
            # Colored console output
            console_format = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
                datefmt='%H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_format)
        else:
            # Standard console output
            console_format = logging.Formatter(
                '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
        
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Detailed format for file logging
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        logger.addHandler(file_handler)
    
    # Log initial message
    if logger.handlers:
        logger.info(f"Logging initialized - Level: {level}")
        if file_logging:
            logger.info(f"Log file: {log_file}")
    
    return logger

class ProgressLogger:
    """Helper class for logging progress of long-running operations"""
    
    def __init__(self, logger: logging.Logger, total: int, task_name: str = "Processing"):
        self.logger = logger
        self.total = total
        self.current = 0
        self.task_name = task_name
        self.start_time = datetime.now()
        self.last_log_percent = 0
    
    def update(self, increment: int = 1, message: str = None):
        """Update progress and log if threshold reached"""
        self.current += increment
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Log every 10% or on completion
        if percent >= self.last_log_percent + 10 or self.current >= self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            
            log_msg = f"{self.task_name}: {self.current}/{self.total} ({percent:.1f}%) - Rate: {rate:.1f}/sec"
            
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
            self.last_log_percent = (percent // 10) * 10
    
    def complete(self, message: str = None):
        """Mark task as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        complete_msg = f"{self.task_name} complete: {self.current} items in {elapsed:.1f} seconds"
        
        if message:
            complete_msg += f" - {message}"
        
        self.logger.info(complete_msg)

class DatabaseLogger:
    """Logger specifically for database operations"""
    
    def __init__(self):
        self.logger = setup_logging('database')
    
    def log_query(self, collection: str, operation: str, filter_dict: dict = None):
        """Log database query"""
        msg = f"{operation} on {collection}"
        if filter_dict:
            msg += f" with filter: {filter_dict}"
        self.logger.debug(msg)
    
    def log_insert(self, collection: str, count: int):
        """Log insert operation"""
        self.logger.info(f"Inserted {count} documents into {collection}")
    
    def log_storage(self, used_mb: float, available_mb: float):
        """Log storage status"""
        percent_used = (used_mb / (used_mb + available_mb)) * 100
        
        if percent_used > 90:
            self.logger.warning(f"Storage critical: {used_mb:.1f}MB used, {available_mb:.1f}MB available ({percent_used:.1f}%)")
        elif percent_used > 75:
            self.logger.warning(f"Storage warning: {used_mb:.1f}MB used, {available_mb:.1f}MB available ({percent_used:.1f}%)")
        else:
            self.logger.info(f"Storage: {used_mb:.1f}MB used, {available_mb:.1f}MB available ({percent_used:.1f}%)")

# Convenience function for one-off logging
def log_error(message: str, exception: Exception = None):
    """Quick error logging"""
    logger = logging.getLogger('error')
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)

def log_success(message: str):
    """Quick success logging"""
    logger = logging.getLogger('success')
    logger.info(f"✅ {message}")

def log_warning(message: str):
    """Quick warning logging"""
    logger = logging.getLogger('warning')
    logger.warning(f"⚠️  {message}")

# Test the logger
if __name__ == "__main__":
    # Test basic logging
    logger = setup_logging(level='DEBUG')
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test progress logger
    progress = ProgressLogger(logger, 100, "Testing progress")
    for i in range(100):
        progress.update()
    progress.complete("All done!")
    
    # Test database logger
    db_logger = DatabaseLogger()
    db_logger.log_query('parcels', 'find', {'zoning': 'IL'})
    db_logger.log_insert('parcels', 1000)
    db_logger.log_storage(400, 100)
    
    print("\n✅ Logger test complete - check logs/ directory for output")
