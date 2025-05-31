"""
Base command class for the VNE CLI command pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import VNECommandLineInterface


class BaseCommand(ABC):
    """Abstract base class for CLI commands using the Command pattern."""

    def __init__(self, cli: 'VNECommandLineInterface'):
        """Initialize command with CLI reference."""
        self.cli = cli
        self.config = cli.get_config() if hasattr(cli, 'get_config') else None
        self.progress_reporter = cli.get_progress_reporter() if hasattr(cli, 'get_progress_reporter') else None
        self.error_handler = cli.get_error_handler() if hasattr(cli, 'get_error_handler') else None
        self.algorithm_registry = cli.algorithm_registry if hasattr(cli, 'algorithm_registry') else None

    @abstractmethod
    def execute(self, args) -> int:
        """Execute the command."""
        pass
