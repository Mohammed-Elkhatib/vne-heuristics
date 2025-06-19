"""
Progress reporting system for long-running VNE operations.

Enhanced with ETA calculation and additional context support for better
user feedback during algorithm execution.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressReporter:
    """
    Progress reporting system for VNE operations.

    Features:
    - Progress percentage tracking
    - Processing rate (items/second)
    - ETA (Estimated Time to Arrival) calculation
    - Configurable reporting intervals
    - Optional additional context information
    """

    def __init__(self):
        """Initialize the progress reporter."""
        self.logger = logging.getLogger(__name__)
        self._enabled = False
        self._interval = 10
        self._start_time: Optional[float] = None
        self._last_report_time: Optional[float] = None

    def initialize(self, enabled: bool = True, interval: int = 10) -> None:
        """
        Initialize progress reporting.

        Args:
            enabled: Whether to enable progress reporting
            interval: Report progress every N items (minimum 1)
        """
        self._enabled = enabled
        self._interval = max(1, interval)  # Ensure minimum interval of 1

        if enabled:
            self.logger.debug(f"Progress reporting enabled with interval {interval}")

    def start(self, total: int, description: str = "Processing") -> None:
        """
        Start progress tracking.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        if not self._enabled:
            return

        self._start_time = time.time()
        self._last_report_time = self._start_time
        self.logger.info(f"Starting {description}: 0/{total} items")

    def update(self, current: int, total: int, description: str = "Processing",
               extra_info: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            current: Current item number
            total: Total number of items
            description: Description of the operation
            extra_info: Optional additional information to display
        """
        if not self._enabled:
            return

        # Check if we should report progress
        if current % self._interval == 0 or current == total:
            self._report_progress(current, total, description, extra_info)

    def _report_progress(self, current: int, total: int, description: str,
                        extra_info: Optional[str] = None) -> None:
        """Report current progress with ETA calculation."""
        if self._start_time is None:
            return

        now = time.time()
        elapsed = now - self._start_time
        percentage = (current / total) * 100 if total > 0 else 0

        # Build progress message components
        message_parts = [f"{description}: {current}/{total} ({percentage:.1f}%)"]

        if elapsed > 0:
            rate = current / elapsed
            message_parts.append(f"{rate:.1f} items/s")

            # Calculate ETA for incomplete tasks
            if 0 < current < total:
                remaining_items = total - current
                eta_seconds = remaining_items / rate

                # Format ETA nicely
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"

                message_parts.append(f"ETA: {eta_str}")

        # Add extra info if provided
        if extra_info:
            message_parts.append(extra_info)

        # Join message parts
        message = ", ".join(message_parts)

        self.logger.info(message)
        self._last_report_time = now

    def finish(self, total: int, description: str = "Processing") -> None:
        """
        Finish progress tracking and report summary.

        Args:
            total: Total number of items processed
            description: Description of the operation
        """
        if not self._enabled or self._start_time is None:
            return

        elapsed = time.time() - self._start_time

        if elapsed > 0:
            rate = total / elapsed

            # Format elapsed time nicely
            if elapsed < 60:
                elapsed_str = f"{elapsed:.1f}s"
            elif elapsed < 3600:
                elapsed_str = f"{elapsed/60:.1f}m"
            else:
                elapsed_str = f"{elapsed/3600:.1f}h"

            self.logger.info(
                f"Completed {description}: {total} items in {elapsed_str} "
                f"({rate:.1f} items/s)"
            )
        else:
            self.logger.info(f"Completed {description}: {total} items")

    def report_custom(self, message: str) -> None:
        """
        Report a custom progress message.

        Args:
            message: Custom message to report
        """
        if self._enabled:
            self.logger.info(message)
