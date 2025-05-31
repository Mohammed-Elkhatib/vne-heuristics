#!/usr/bin/env python3
"""
Enhanced Main CLI for the VNE (Virtual Network Embedding) project.

This module provides the primary entry point for the VNE heuristics project,
with a modular design that separates CLI logic from business logic.

Key Enhancements:
- Modular architecture with clean separation of concerns
- Dynamic algorithm discovery system
- Proper use of generator modules instead of inline implementation
- Enhanced error handling with specific exception types
- Better progress reporting and logging
- Extensible command system

Usage Examples:
    # Generate networks using proper generators
    python main.py generate substrate --nodes 20 --topology erdos_renyi
    python main.py generate vnrs --count 50 --substrate data/substrate_20.csv

    # Run algorithms with dynamic discovery
    python main.py run --algorithm yu2008 --substrate data/substrate_20.csv --vnrs data/vnrs_batch1.csv

    # Calculate comprehensive metrics
    python main.py metrics --results data/output/results.json --output data/output/metrics.csv
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import enhanced configuration management
from config_management import (
    ConfigurationManager,
    VNEConfig,
    ConfigurationError,
    load_config_from_args
)

# Import CLI modules
from cli.argument_parser import create_main_parser
from cli.commands import (
    GenerateCommand,
    RunCommand,
    MetricsCommand,
    ConfigCommand
)
from cli.exceptions import (
    VNECLIError,
    CommandError,
    ValidationError as CLIValidationError
)

# Import core modules
from core.algorithm_registry import AlgorithmRegistry
from core.progress_reporter import ProgressReporter
from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class VNECommandLineInterface:
    """
    Enhanced command-line interface for the VNE project.

    This class provides a clean, modular architecture for handling VNE operations
    through a command-line interface. It follows the Command pattern for
    extensibility and maintainability.

    Key Features:
    - Dynamic algorithm discovery and registration
    - Modular command system for easy extension
    - Comprehensive error handling and reporting
    - Progress tracking for long-running operations
    - Proper configuration management integration

    Architecture:
    - CLI layer: Argument parsing and user interaction
    - Command layer: Individual command implementations
    - Core layer: Business logic and algorithms
    - Model layer: Data structures and persistence
    """

    def __init__(self):
        """Initialize the enhanced CLI with modular components."""
        self.config_manager = ConfigurationManager()
        self.config: Optional[VNEConfig] = None
        self.algorithm_registry = AlgorithmRegistry()
        self.progress_reporter = ProgressReporter()
        self.error_handler = ErrorHandler()

        # Command registry
        self.commands = {
            'generate': GenerateCommand(self),
            'run': RunCommand(self),
            'metrics': MetricsCommand(self),
            'config': ConfigCommand(self)
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point for the CLI with enhanced error handling.

        Args:
            args: Command-line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse command-line arguments
            parser = create_main_parser(self.algorithm_registry)
            parsed_args = parser.parse_args(args)

            # Handle case where no command is provided
            if not hasattr(parsed_args, 'command') or not parsed_args.command:
                parser.print_help()
                return 1

            # Load and validate configuration
            self.config = self._load_configuration(parsed_args)

            # Setup directories and logging
            self._setup_environment()

            # Execute the requested command
            return self._execute_command(parsed_args)

        except KeyboardInterrupt:
            self.error_handler.handle_user_interruption()
            return 130
        except ConfigurationError as e:
            self.error_handler.handle_configuration_error(e)
            return 1
        except VNECLIError as e:
            self.error_handler.handle_cli_error(e)
            return 1
        except Exception as e:
            self.error_handler.handle_unexpected_error(e)
            return 1

    def _load_configuration(self, parsed_args) -> VNEConfig:
        """
        Load and validate configuration from various sources.

        Args:
            parsed_args: Parsed command-line arguments

        Returns:
            Validated VNEConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.logger.info("Loading configuration")
            config = load_config_from_args(parsed_args)

            # Validate configuration for the specific command
            if hasattr(parsed_args, 'command'):
                self._validate_command_configuration(parsed_args.command, config)

            return config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _validate_command_configuration(self, command: str, config: VNEConfig) -> None:
        """
        Validate configuration specific to the requested command.

        Args:
            command: Command name
            config: Configuration to validate

        Raises:
            ConfigurationError: If configuration is invalid for the command
        """
        # Command-specific validation
        validation_rules = {
            'generate': self._validate_generate_config,
            'run': self._validate_run_config,
            'metrics': self._validate_metrics_config,
        }

        if command in validation_rules:
            validation_rules[command](config)

    def _validate_generate_config(self, config: VNEConfig) -> None:
        """Validate configuration for generate command."""
        ng = config.network_generation
        if ng.substrate_nodes <= 0:
            raise ConfigurationError("substrate_nodes must be positive for generation")
        if ng.vnr_count <= 0:
            raise ConfigurationError("vnr_count must be positive for generation")

    def _validate_run_config(self, config: VNEConfig) -> None:
        """Validate configuration for run command."""
        alg = config.algorithm
        if alg.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive for algorithm execution")

    def _validate_metrics_config(self, config: VNEConfig) -> None:
        """Validate configuration for metrics command."""
        # Metrics command has minimal configuration requirements
        pass

    def _setup_environment(self) -> None:
        """Setup directories and environment for operation."""
        try:
            # Create necessary directories
            self.config_manager.create_directories()

            # Initialize progress reporting
            self.progress_reporter.initialize(
                enabled=self.config.experiment.progress_reporting,
                interval=self.config.experiment.progress_interval
            )

            self.logger.info("Environment setup completed")

        except Exception as e:
            raise VNECLIError(f"Failed to setup environment: {e}")

    def _execute_command(self, parsed_args) -> int:
        """
        Execute the requested command using the command pattern.

        Args:
            parsed_args: Parsed command-line arguments

        Returns:
            Exit code from command execution

        Raises:
            CommandError: If command execution fails
        """
        command_name = parsed_args.command

        if command_name not in self.commands:
            raise CommandError(f"Unknown command: {command_name}")

        try:
            self.logger.info(f"Executing command: {command_name}")
            command = self.commands[command_name]
            return command.execute(parsed_args)

        except Exception as e:
            raise CommandError(f"Command '{command_name}' failed: {e}")

    def get_available_algorithms(self) -> Dict[str, type]:
        """
        Get available algorithms from the registry.

        Returns:
            Dictionary mapping algorithm names to classes
        """
        return self.algorithm_registry.get_algorithms()

    def get_config(self) -> Optional[VNEConfig]:
        """Get current configuration."""
        return self.config

    def get_progress_reporter(self) -> ProgressReporter:
        """Get progress reporter instance."""
        return self.progress_reporter

    def get_error_handler(self) -> ErrorHandler:
        """Get error handler instance."""
        return self.error_handler


def main() -> int:
    """
    Main entry point with enhanced error handling and logging setup.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Setup basic logging before configuration is loaded
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    try:
        # Create and run CLI
        cli = VNECommandLineInterface()
        exit_code = cli.run()

        if exit_code == 0:
            logger.info("VNE CLI completed successfully")
        else:
            logger.error(f"VNE CLI completed with exit code: {exit_code}")

        return exit_code

    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
