"""
Enhanced Configuration Management for Virtual Network Embedding (VNE) project.

This module provides centralized configuration management with support for
default settings, configuration files, environment variables, and validation.
Enhanced with better error handling, modular design, and robust UTF-8 support.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field, asdict
import yaml

logger = logging.getLogger(__name__)


# Configuration dataclasses (unchanged but documented better)
@dataclass
class NetworkGenerationConfig:
    """Configuration for network generation parameters."""
    # Substrate network parameters
    substrate_nodes: int = 50
    substrate_topology: str = "erdos_renyi"  # "erdos_renyi", "barabasi_albert", "grid"
    substrate_edge_probability: float = 0.15
    substrate_attachment_count: int = 3  # For Barabási-Albert

    # Resource parameters
    cpu_range: tuple = (50, 200)
    memory_range: tuple = (50, 200)
    bandwidth_range: tuple = (50, 200)
    coordinate_range: tuple = (0.0, 100.0)

    # VNR parameters
    vnr_count: int = 100
    vnr_nodes_range: tuple = (2, 10)
    vnr_topology: str = "random"  # "random", "star", "linear", "tree"
    vnr_edge_probability: float = 0.5

    # VNR resource requirements (as ratios of substrate resources)
    vnr_cpu_ratio_range: tuple = (0.1, 0.3)
    vnr_memory_ratio_range: tuple = (0.1, 0.3)
    vnr_bandwidth_ratio_range: tuple = (0.1, 0.3)

    # Time parameters
    arrival_pattern: str = "poisson"  # "poisson", "uniform"
    arrival_rate: float = 10.0  # VNRs per time unit
    lifetime_distribution: str = "exponential"  # "exponential", "uniform", "fixed"
    lifetime_mean: float = 1000.0  # Time units

    # Random seed for reproducibility
    random_seed: Optional[int] = None


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters."""
    # General algorithm settings
    max_embedding_attempts: int = 1
    timeout_seconds: float = 300.0  # 5 minutes
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True

    # Resource calculation weights
    cpu_weight: float = 1.0
    memory_weight: float = 1.0
    bandwidth_weight: float = 1.0

    # Cost calculation parameters
    node_cost_factor: float = 10.0
    link_cost_factor: float = 5.0
    failure_cost: float = 1.0

    # Path finding parameters
    max_path_length: int = 10
    k_shortest_paths: int = 3
    enable_path_splitting: bool = False

    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilePathConfig:
    """Configuration for file paths and naming conventions."""
    # Base directories
    data_dir: str = "data"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    results_dir: str = "data/output/results"
    metrics_dir: str = "data/output/metrics"
    networks_dir: str = "data/input/networks"
    vnrs_dir: str = "data/input/vnrs"

    # File naming patterns
    substrate_filename_pattern: str = "substrate_{nodes}_{topology}.csv"
    vnr_filename_pattern: str = "vnrs_{count}_{substrate_name}.csv"
    results_filename_pattern: str = "results_{algorithm}_{timestamp}.json"
    metrics_filename_pattern: str = "metrics_{algorithm}_{timestamp}.csv"

    # Backup and archiving
    enable_backups: bool = True
    backup_suffix: str = ".bak"
    max_backup_age_days: int = 30


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    # Logging levels
    root_level: str = "INFO"
    algorithm_level: str = "INFO"
    file_level: str = "DEBUG"
    console_level: str = "INFO"

    # Log file settings
    log_file: str = "vne.log"
    max_file_size_mb: int = 10
    backup_count: int = 5

    # Log format
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Enable/disable specific loggers
    enable_algorithm_logging: bool = True
    enable_metrics_logging: bool = True
    enable_io_logging: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Execution parameters
    parallel_execution: bool = False
    max_workers: int = 4
    progress_reporting: bool = True
    progress_interval: int = 10  # Report every N VNRs

    # Result handling
    save_intermediate_results: bool = True
    compress_results: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # Validation parameters
    validate_inputs: bool = True
    validate_results: bool = True
    strict_validation: bool = False


@dataclass
class VNEConfig:
    """Main configuration container for the VNE project."""
    network_generation: NetworkGenerationConfig = field(default_factory=NetworkGenerationConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    file_paths: FilePathConfig = field(default_factory=FilePathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Global settings
    version: str = "1.0.0"
    debug_mode: bool = False
    verbose: bool = False
    quiet: bool = False


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class FileHandlingError(ConfigurationError):
    """Exception raised for file handling errors."""
    pass


class ValidationError(ConfigurationError):
    """Exception raised for validation errors."""
    pass


class EnvironmentError(ConfigurationError):
    """Exception raised for environment variable errors."""
    pass


class ConfigurationManager:
    """
    Enhanced centralized configuration management for the VNE project.

    This class handles loading configuration from files, environment variables,
    and command-line parameters with proper precedence and validation.

    Configuration precedence (highest to lowest):
    1. Command-line parameters
    2. Environment variables
    3. Configuration file
    4. Default values

    Enhancements:
    - Better error handling with specific exceptions
    - Modular method design
    - Robust UTF-8 handling
    - Improved logging configuration
    - Better validation with detailed messages

    Example:
        >>> config_manager = ConfigurationManager()
        >>> config = config_manager.load_config("config.yaml")
        >>> print(f"Substrate nodes: {config.network_generation.substrate_nodes}")
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.config = VNEConfig()
        self.config_file_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        self._encoding = 'utf-8'

    def load_config(self, config_file: Optional[Union[str, Path]] = None,
                   env_prefix: str = "VNE_",
                   **overrides) -> VNEConfig:
        """
        Load configuration from file, environment, and overrides.

        Args:
            config_file: Path to configuration file (JSON/YAML)
            env_prefix: Prefix for environment variables
            **overrides: Direct parameter overrides

        Returns:
            Complete VNEConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.logger.info("Starting configuration loading process")

        # Start with default configuration
        self.config = VNEConfig()

        # Load from configuration file if provided
        if config_file:
            self._load_configuration_file(config_file)

        # Load from environment variables
        self._load_environment_variables(env_prefix)

        # Apply direct overrides
        self._apply_parameter_overrides(overrides)

        # Validate configuration
        self._validate_complete_configuration()

        # Setup logging based on configuration
        self._configure_logging_system()

        self.logger.info("Configuration loaded and validated successfully")
        return self.config

    def _load_configuration_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from JSON or YAML file with robust error handling.

        Args:
            config_file: Path to configuration file

        Raises:
            FileHandlingError: If file operations fail
            ValidationError: If file format is invalid
        """
        config_path = Path(config_file)
        self.config_file_path = config_path

        self.logger.debug(f"Loading configuration from: {config_path}")

        if not config_path.exists():
            raise FileHandlingError(f"Configuration file not found: {config_path}")

        if not config_path.is_file():
            raise FileHandlingError(f"Configuration path is not a file: {config_path}")

        try:
            file_data = self._read_config_file(config_path)
            self._merge_config_data(file_data)
            self.logger.info(f"Successfully loaded configuration from {config_path}")

        except Exception as e:
            raise FileHandlingError(f"Failed to load configuration file {config_path}: {e}")

    def _read_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Read and parse configuration file with proper encoding handling.

        Args:
            config_path: Path to configuration file

        Returns:
            Parsed configuration data

        Raises:
            ValidationError: If file format is invalid
        """
        try:
            with open(config_path, 'r', encoding=self._encoding) as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValidationError(
                        f"Unsupported config file format: {config_path.suffix}. "
                        f"Supported formats: .yaml, .yml, .json"
                    )

        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML format in {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format in {config_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValidationError(f"Encoding error in {config_path}: {e}")

    def _load_environment_variables(self, prefix: str) -> None:
        """
        Load configuration from environment variables with improved error handling.

        Args:
            prefix: Environment variable prefix
        """
        self.logger.debug(f"Loading environment variables with prefix: {prefix}")

        env_mappings = self._get_environment_mappings(prefix)
        loaded_count = 0

        for env_var, (section, key, converter) in env_mappings.items():
            try:
                value = os.getenv(env_var)
                if value is not None:
                    self._apply_environment_variable(env_var, section, key, converter, value)
                    loaded_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to process environment variable {env_var}: {e}")

        if loaded_count > 0:
            self.logger.info(f"Loaded {loaded_count} environment variables")

    def _get_environment_mappings(self, prefix: str) -> Dict[str, Tuple[str, Optional[str], callable]]:
        """Get environment variable mappings."""
        return {
            f"{prefix}SUBSTRATE_NODES": ("network_generation", "substrate_nodes", int),
            f"{prefix}SUBSTRATE_TOPOLOGY": ("network_generation", "substrate_topology", str),
            f"{prefix}VNR_COUNT": ("network_generation", "vnr_count", int),
            f"{prefix}ALGORITHM_TIMEOUT": ("algorithm", "timeout_seconds", float),
            f"{prefix}DATA_DIR": ("file_paths", "data_dir", str),
            f"{prefix}LOG_LEVEL": ("logging", "root_level", str),
            f"{prefix}DEBUG": ("debug_mode", None, self._str_to_bool),
            f"{prefix}VERBOSE": ("verbose", None, self._str_to_bool),
            f"{prefix}QUIET": ("quiet", None, self._str_to_bool),
        }

    def _apply_environment_variable(self, env_var: str, section: str,
                                   key: Optional[str], converter: callable, value: str) -> None:
        """Apply a single environment variable to configuration."""
        try:
            converted_value = converter(value)

            if key is None:
                # Direct config attribute
                if not hasattr(self.config, section):
                    raise EnvironmentError(f"Unknown configuration section: {section}")
                setattr(self.config, section, converted_value)
            else:
                # Nested config attribute
                if not hasattr(self.config, section):
                    raise EnvironmentError(f"Unknown configuration section: {section}")

                section_obj = getattr(self.config, section)
                if not hasattr(section_obj, key):
                    raise EnvironmentError(f"Unknown parameter: {section}.{key}")

                setattr(section_obj, key, converted_value)

            self.logger.debug(f"Set {env_var} = {converted_value}")

        except ValueError as e:
            raise EnvironmentError(f"Invalid value for {env_var}: {value} ({e})")

    def _apply_parameter_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply direct parameter overrides with improved error handling.

        Args:
            overrides: Dictionary of parameter overrides
        """
        if not overrides:
            return

        self.logger.debug(f"Applying {len(overrides)} parameter overrides")

        for key, value in overrides.items():
            try:
                self._apply_single_override(key, value)
            except Exception as e:
                self.logger.warning(f"Failed to apply override {key}={value}: {e}")

    def _apply_single_override(self, key: str, value: Any) -> None:
        """Apply a single parameter override."""
        if '.' in key:
            # Handle nested keys like "network_generation.substrate_nodes"
            section, param = key.split('.', 1)

            if not hasattr(self.config, section):
                raise ValidationError(f"Unknown configuration section: {section}")

            section_obj = getattr(self.config, section)
            if not hasattr(section_obj, param):
                raise ValidationError(f"Unknown parameter: {key}")

            setattr(section_obj, param, value)
            self.logger.debug(f"Override: {key} = {value}")
        else:
            # Direct config attribute
            if not hasattr(self.config, key):
                raise ValidationError(f"Unknown configuration parameter: {key}")

            setattr(self.config, key, value)
            self.logger.debug(f"Override: {key} = {value}")

    def _merge_config_data(self, data: Dict[str, Any]) -> None:
        """
        Merge configuration data from file with current config.

        Args:
            data: Configuration data to merge
        """
        merged_sections = 0

        for section_name, section_data in data.items():
            if not hasattr(self.config, section_name):
                self.logger.warning(f"Unknown configuration section: {section_name}")
                continue

            if isinstance(section_data, dict):
                self._merge_section_data(section_name, section_data)
            else:
                setattr(self.config, section_name, section_data)

            merged_sections += 1

        self.logger.debug(f"Merged {merged_sections} configuration sections")

    def _merge_section_data(self, section_name: str, section_data: Dict[str, Any]) -> None:
        """Merge data for a specific configuration section."""
        section_obj = getattr(self.config, section_name)
        merged_params = 0

        for param_name, param_value in section_data.items():
            if hasattr(section_obj, param_name):
                setattr(section_obj, param_name, param_value)
                merged_params += 1
            else:
                self.logger.warning(f"Unknown parameter: {section_name}.{param_name}")

        self.logger.debug(f"Merged {merged_params} parameters in section {section_name}")

    def _validate_complete_configuration(self) -> None:
        """
        Validate complete configuration with detailed error messages.

        Raises:
            ValidationError: If configuration is invalid
        """
        self.logger.debug("Validating complete configuration")

        validation_errors = []

        # Network generation validation
        validation_errors.extend(self._validate_network_generation())

        # Algorithm validation
        validation_errors.extend(self._validate_algorithm_config())

        # File paths validation
        validation_errors.extend(self._validate_file_paths())

        # Logging validation
        validation_errors.extend(self._validate_logging_config())

        if validation_errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in validation_errors
            )
            raise ValidationError(error_message)

        self.logger.debug("Configuration validation passed")

    def _validate_network_generation(self) -> List[str]:
        """Validate network generation configuration."""
        errors = []
        ng = self.config.network_generation

        if ng.substrate_nodes <= 0:
            errors.append("substrate_nodes must be positive")

        if ng.vnr_count <= 0:
            errors.append("vnr_count must be positive")

        if not (0 <= ng.substrate_edge_probability <= 1):
            errors.append("substrate_edge_probability must be between 0 and 1")

        if ng.substrate_attachment_count <= 0:
            errors.append("substrate_attachment_count must be positive")

        # Validate topology
        valid_topologies = ["erdos_renyi", "barabasi_albert", "grid"]
        if ng.substrate_topology not in valid_topologies:
            errors.append(f"substrate_topology must be one of: {valid_topologies}")

        return errors

    def _validate_algorithm_config(self) -> List[str]:
        """Validate algorithm configuration."""
        errors = []
        alg = self.config.algorithm

        if alg.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")

        if alg.max_embedding_attempts <= 0:
            errors.append("max_embedding_attempts must be positive")

        if alg.max_path_length <= 0:
            errors.append("max_path_length must be positive")

        if alg.k_shortest_paths <= 0:
            errors.append("k_shortest_paths must be positive")

        return errors

    def _validate_file_paths(self) -> List[str]:
        """Validate file path configuration."""
        errors = []

        try:
            Path(self.config.file_paths.data_dir)
        except Exception as e:
            errors.append(f"Invalid data_dir path: {e}")

        return errors

    def _validate_logging_config(self) -> List[str]:
        """Validate logging configuration."""
        errors = []
        log_config = self.config.logging

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level_name, level_value in [
            ("root_level", log_config.root_level),
            ("algorithm_level", log_config.algorithm_level),
            ("file_level", log_config.file_level),
            ("console_level", log_config.console_level)
        ]:
            if level_value not in valid_log_levels:
                errors.append(f"Invalid {level_name}: {level_value}. Must be one of: {valid_log_levels}")

        if log_config.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")

        if log_config.backup_count < 0:
            errors.append("backup_count must be non-negative")

        return errors

    def _configure_logging_system(self) -> None:
        """Configure logging system with enhanced UTF-8 support and error handling."""
        log_config = self.config.logging

        try:
            # Create logs directory
            log_file_path = Path(log_config.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_config.root_level))

            # Clear existing handlers
            root_logger.handlers.clear()

            # Create formatter
            formatter = logging.Formatter(log_config.format, log_config.date_format)

            # Setup console handler
            if not self.config.quiet:
                self._setup_console_handler(formatter, log_config)

            # Setup file handler
            self._setup_file_handler(formatter, log_config)

            self.logger.info("Logging system configured successfully")

        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger.error(f"Failed to configure logging system: {e}")
            logger.info("Using fallback logging configuration")

    def _setup_console_handler(self, formatter: logging.Formatter, log_config: LoggingConfig) -> None:
        """Setup console handler with proper UTF-8 encoding."""
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_config.console_level))
            console_handler.setFormatter(formatter)

            # Enhanced UTF-8 support
            if hasattr(console_handler.stream, 'reconfigure'):
                try:
                    console_handler.stream.reconfigure(encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Could not reconfigure console encoding: {e}")

            # Force UTF-8 for Windows
            if sys.platform == 'win32' and hasattr(console_handler.stream, 'buffer'):
                try:
                    import codecs
                    console_handler.stream = codecs.getwriter('utf-8')(console_handler.stream.buffer)
                except Exception as e:
                    logger.warning(f"Could not set UTF-8 encoding on Windows: {e}")

            logging.getLogger().addHandler(console_handler)

        except Exception as e:
            logger.warning(f"Failed to setup console handler: {e}")

    def _setup_file_handler(self, formatter: logging.Formatter, log_config: LoggingConfig) -> None:
        """Setup file handler with rotation and UTF-8 encoding."""
        try:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_config.log_file,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_config.file_level))
            file_handler.setFormatter(formatter)

            logging.getLogger().addHandler(file_handler)

        except Exception as e:
            logger.warning(f"Failed to setup file handler: {e}")

    def save_config(self, output_file: Union[str, Path], format: str = "yaml") -> None:
        """
        Save current configuration to file with enhanced error handling.

        Args:
            output_file: Path to output file
            format: Output format ("yaml" or "json")

        Raises:
            FileHandlingError: If save operation fails
        """
        output_path = Path(output_file)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = asdict(self.config)
            config_dict = self._convert_tuples_to_lists(config_dict)

            with open(output_path, 'w', encoding=self._encoding) as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2,
                            allow_unicode=True, encoding=None)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValidationError(f"Unsupported format: {format}")

            self.logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise FileHandlingError(f"Failed to save configuration to {output_path}: {e}")

    def _convert_tuples_to_lists(self, obj):
        """Convert tuples to lists recursively for YAML compatibility."""
        if isinstance(obj, dict):
            return {key: self._convert_tuples_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        elif isinstance(obj, list):
            return [self._convert_tuples_to_lists(item) for item in obj]
        else:
            return obj

    def get_algorithm_config(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get algorithm-specific configuration.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Dictionary with algorithm configuration
        """
        base_config = asdict(self.config.algorithm)

        # Get algorithm-specific parameters
        algo_params = base_config.get('algorithm_params', {}).get(algorithm_name, {})

        # Merge with base configuration
        base_config.update(algo_params)

        return base_config

    def create_directories(self) -> None:
        """Create all necessary directories based on configuration."""
        directories = [
            self.config.file_paths.data_dir,
            self.config.file_paths.input_dir,
            self.config.file_paths.output_dir,
            self.config.file_paths.results_dir,
            self.config.file_paths.metrics_dir,
            self.config.file_paths.networks_dir,
            self.config.file_paths.vnrs_dir,
        ]

        created_count = 0
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                created_count += 1
                self.logger.debug(f"Created directory: {directory}")
            except Exception as e:
                self.logger.warning(f"Failed to create directory {directory}: {e}")

        self.logger.info(f"Created {created_count} directories")

    def get_file_path(self, file_type: str, **kwargs) -> Path:
        """
        Generate file path based on configuration patterns.

        Args:
            file_type: Type of file ("substrate", "vnr", "results", "metrics")
            **kwargs: Parameters for filename pattern

        Returns:
            Complete file path

        Raises:
            ValidationError: If file_type is unknown
        """
        paths_config = self.config.file_paths

        pattern_mapping = {
            "substrate": (paths_config.substrate_filename_pattern, paths_config.networks_dir),
            "vnr": (paths_config.vnr_filename_pattern, paths_config.vnrs_dir),
            "results": (paths_config.results_filename_pattern, paths_config.results_dir),
            "metrics": (paths_config.metrics_filename_pattern, paths_config.metrics_dir),
        }

        if file_type not in pattern_mapping:
            raise ValidationError(f"Unknown file type: {file_type}. Valid types: {list(pattern_mapping.keys())}")

        pattern, directory = pattern_mapping[file_type]
        filename = pattern.format(**kwargs)
        return Path(directory) / filename

    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert string to boolean with enhanced parsing."""
        if isinstance(value, bool):
            return value

        true_values = {'true', '1', 'yes', 'on', 'enabled', 'y', 't'}
        false_values = {'false', '0', 'no', 'off', 'disabled', 'n', 'f'}

        normalized_value = value.lower().strip()

        if normalized_value in true_values:
            return True
        elif normalized_value in false_values:
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")

    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"VNEConfig(substrate_nodes={self.config.network_generation.substrate_nodes}, "
               f"vnr_count={self.config.network_generation.vnr_count}, "
               f"algorithm_timeout={self.config.algorithm.timeout_seconds})")


def create_default_config_file(output_file: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a default configuration file with all available options.

    Args:
        output_file: Path to output file
        format: Output format ("yaml" or "json")

    Raises:
        FileHandlingError: If creation fails
    """
    try:
        config_manager = ConfigurationManager()
        config_manager.save_config(output_file, format)
        logger.info(f"Created default configuration file: {output_file}")

    except Exception as e:
        raise FileHandlingError(f"Failed to create default configuration file: {e}")


def load_config_from_args(args) -> VNEConfig:
    """
    Load configuration from command-line arguments with enhanced error handling.

    Args:
        args: Parsed command-line arguments (from argparse)

    Returns:
        VNEConfig instance

    Raises:
        ConfigurationError: If loading fails
    """
    try:
        config_manager = ConfigurationManager()

        # Prepare overrides from command-line args
        overrides = {}

        # Map common command-line arguments to config parameters
        arg_mappings = {
            'substrate_nodes': 'network_generation.substrate_nodes',
            'substrate_topology': 'network_generation.substrate_topology',
            'vnr_count': 'network_generation.vnr_count',
            'data_dir': 'file_paths.data_dir',
            'verbose': 'verbose',
            'quiet': 'quiet',
            'debug': 'debug_mode',
        }

        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                overrides[config_path] = getattr(args, arg_name)

        # Load configuration with overrides
        config_file = getattr(args, 'config', None)
        return config_manager.load_config(config_file, **overrides)

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from arguments: {e}")


# Export clean interface
__all__ = [
    'VNEConfig',
    'NetworkGenerationConfig',
    'AlgorithmConfig',
    'FilePathConfig',
    'LoggingConfig',
    'ExperimentConfig',
    'ConfigurationManager',
    'ConfigurationError',
    'FileHandlingError',
    'ValidationError',
    'EnvironmentError',
    'create_default_config_file',
    'load_config_from_args'
]
