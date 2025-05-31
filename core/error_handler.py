"""
Centralized error handling system for VNE CLI.
"""

import logging
import sys

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for VNE CLI."""

    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)

    def handle_configuration_error(self, error) -> None:
        """Handle configuration-related errors."""
        self.logger.error(f"Configuration error: {error}")
        print(f"Configuration Error: {error}", file=sys.stderr)
        print("Check your configuration file syntax", file=sys.stderr)

    def handle_cli_error(self, error) -> None:
        """Handle CLI-specific errors."""
        self.logger.error(f"CLI error: {error}")
        print(f"Error: {error}", file=sys.stderr)

    def handle_user_interruption(self) -> None:
        """Handle user interruption."""
        self.logger.info("Operation cancelled by user")
        print("Operation cancelled by user", file=sys.stderr)

    def handle_unexpected_error(self, error: Exception) -> None:
        """Handle unexpected errors."""
        self.logger.error(f"Unexpected error: {error}", exc_info=True)
        print(f"Unexpected Error: {error}", file=sys.stderr)
