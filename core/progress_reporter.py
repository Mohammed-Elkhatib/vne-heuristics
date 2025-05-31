"""
Progress reporting system for long-running VNE operations.
"""

import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Progress reporting system for VNE operations."""
    
    def __init__(self):
        """Initialize the progress reporter."""
        self.logger = logging.getLogger(__name__)
        self._enabled = False
        self._interval = 10
        self._start_time: Optional[float] = None
    
    def initialize(self, enabled: bool = True, interval: int = 10) -> None:
        """Initialize progress reporting."""
        self._enabled = enabled
        self._interval = interval
        
        if enabled:
            self.logger.debug(f"Progress reporting enabled with interval {interval}")
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking."""
        if not self._enabled:
            return
        
        self._start_time = time.time()
        self.logger.info(f"Starting {description}: 0/{total} items")
    
    def update(self, current: int, total: int, description: str = "Processing") -> None:
        """Update progress."""
        if not self._enabled:
            return
        
        # Check if we should report progress
        if current % self._interval == 0 or current == total:
            self._report_progress(current, total, description)
    
    def _report_progress(self, current: int, total: int, description: str) -> None:
        """Report current progress."""
        if self._start_time is None:
            return
            
        elapsed = time.time() - self._start_time
        percentage = (current / total) * 100 if total > 0 else 0
        
        if elapsed > 0:
            rate = current / elapsed
            rate_info = f", {rate:.1f} items/s"
        else:
            rate_info = ""
        
        self.logger.info(
            f"{description}: {current}/{total} ({percentage:.1f}%{rate_info})"
        )
    
    def finish(self, total: int, description: str = "Processing") -> None:
        """Finish progress tracking."""
        if not self._enabled or self._start_time is None:
            return
        
        elapsed = time.time() - self._start_time
        rate = total / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Completed {description}: {total} items in {elapsed:.1f}s ({rate:.1f} items/s)"
        )
