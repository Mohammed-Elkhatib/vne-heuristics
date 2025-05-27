"""
Configuration management for Virtual Network Embedding (VNE) project.

This module provides centralized configuration management with support for
default settings, configuration files, environment variables, and validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import yaml

logger = logging.getLogger(__name__)


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


class ConfigurationManager:
    """
    Centralized configuration management for the VNE project.
    
    This class handles loading configuration from files, environment variables,
    and command-line parameters with proper precedence and validation.
    
    Configuration precedence (highest to lowest):
    1. Command-line parameters
    2. Environment variables
    3. Configuration file
    4. Default values
    
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
        # Start with default configuration
        self.config = VNEConfig()
        
        # Load from configuration file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Load from environment variables
        self._load_from_environment(env_prefix)
        
        # Apply direct overrides
        self._apply_overrides(overrides)
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging based on configuration
        self._setup_logging()
        
        self.logger.info("Configuration loaded successfully")
        return self.config
    
    def _load_from_file(self, config_file: Union[str, Path]) -> None:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_file)
        self.config_file_path = config_path
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge loaded data with current config
            self._merge_config_data(data)
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {e}")
    
    def _load_from_environment(self, prefix: str) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
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
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    if key is None:
                        # Direct config attribute
                        setattr(self.config, section, converted_value)
                    else:
                        # Nested config attribute
                        section_obj = getattr(self.config, section)
                        setattr(section_obj, key, converted_value)
                    
                    self.logger.debug(f"Set {env_var} = {converted_value}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply direct parameter overrides."""
        for key, value in overrides.items():
            try:
                # Handle nested keys like "network_generation.substrate_nodes"
                if '.' in key:
                    section, param = key.split('.', 1)
                    if hasattr(self.config, section):
                        section_obj = getattr(self.config, section)
                        if hasattr(section_obj, param):
                            setattr(section_obj, param, value)
                            self.logger.debug(f"Override: {key} = {value}")
                        else:
                            self.logger.warning(f"Unknown parameter: {key}")
                    else:
                        self.logger.warning(f"Unknown section: {section}")
                else:
                    # Direct config attribute
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        self.logger.debug(f"Override: {key} = {value}")
                    else:
                        self.logger.warning(f"Unknown parameter: {key}")
            except Exception as e:
                self.logger.warning(f"Failed to apply override {key}={value}: {e}")
    
    def _merge_config_data(self, data: Dict[str, Any]) -> None:
        """Merge configuration data from file with current config."""
        for section_name, section_data in data.items():
            if hasattr(self.config, section_name):
                section_obj = getattr(self.config, section_name)
                if isinstance(section_data, dict):
                    for param_name, param_value in section_data.items():
                        if hasattr(section_obj, param_name):
                            setattr(section_obj, param_name, param_value)
                        else:
                            self.logger.warning(f"Unknown parameter: {section_name}.{param_name}")
                else:
                    setattr(self.config, section_name, section_data)
            else:
                self.logger.warning(f"Unknown configuration section: {section_name}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Validate network generation parameters
        if self.config.network_generation.substrate_nodes <= 0:
            errors.append("substrate_nodes must be positive")
        
        if self.config.network_generation.vnr_count <= 0:
            errors.append("vnr_count must be positive")
        
        if not (0 <= self.config.network_generation.substrate_edge_probability <= 1):
            errors.append("substrate_edge_probability must be between 0 and 1")
        
        # Validate algorithm parameters
        if self.config.algorithm.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.config.algorithm.max_embedding_attempts <= 0:
            errors.append("max_embedding_attempts must be positive")
        
        # Validate file paths
        try:
            Path(self.config.file_paths.data_dir)
        except Exception:
            errors.append("Invalid data_dir path")
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.root_level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.config.logging.root_level}")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.config.logging
        
        # Create logs directory if it doesn't exist
        log_file_path = Path(log_config.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.root_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_config.format, log_config.date_format)
        
        # Console handler with UTF-8 encoding
        if not self.config.quiet:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_config.console_level))
            console_handler.setFormatter(formatter)

            # Set encoding to UTF-8 to handle Unicode characters
            import sys
            if hasattr(console_handler.stream, 'reconfigure'):
                try:
                    console_handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass  # Fallback if reconfigure fails

            root_logger.addHandler(console_handler)

        # File handler with rotation and UTF-8 encoding
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.log_file,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count,
                encoding='utf-8'  # Explicitly set UTF-8 encoding for file
            )
            file_handler.setLevel(getattr(logging, log_config.file_level))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # If file handler fails, continue without it
            print(f"Warning: Could not setup file logging: {e}")
    
    def save_config(self, output_file: Union[str, Path], format: str = "yaml") -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Path to output file
            format: Output format ("yaml" or "json")
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
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
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def get_file_path(self, file_type: str, **kwargs) -> Path:
        """
        Generate file path based on configuration patterns.
        
        Args:
            file_type: Type of file ("substrate", "vnr", "results", "metrics")
            **kwargs: Parameters for filename pattern
            
        Returns:
            Complete file path
        """
        paths_config = self.config.file_paths
        
        if file_type == "substrate":
            filename = paths_config.substrate_filename_pattern.format(**kwargs)
            return Path(paths_config.networks_dir) / filename
        
        elif file_type == "vnr":
            filename = paths_config.vnr_filename_pattern.format(**kwargs)
            return Path(paths_config.vnrs_dir) / filename
        
        elif file_type == "results":
            filename = paths_config.results_filename_pattern.format(**kwargs)
            return Path(paths_config.results_dir) / filename
        
        elif file_type == "metrics":
            filename = paths_config.metrics_filename_pattern.format(**kwargs)
            return Path(paths_config.metrics_dir) / filename
        
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"VNEConfig(substrate_nodes={self.config.network_generation.substrate_nodes}, " \
               f"vnr_count={self.config.network_generation.vnr_count})"


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


def create_default_config_file(output_file: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a default configuration file with all available options.
    
    Args:
        output_file: Path to output file
        format: Output format ("yaml" or "json")
    """
    config_manager = ConfigurationManager()
    config_manager.save_config(output_file, format)


def load_config_from_args(args) -> VNEConfig:
    """
    Load configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments (from argparse)
        
    Returns:
        VNEConfig instance
    """
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


# Example usage and testing
def example_usage():
    """Example usage of the configuration management system."""
    print("=== VNE Configuration Management Example ===")
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Load default configuration
    config = config_manager.load_config()
    print(f"Default config: {config}")
    
    # Create example config file
    config_file = Path("example_config.yaml")
    config_manager.save_config(config_file, "yaml")
    print(f"Saved example config to {config_file}")
    
    # Load with overrides
    config = config_manager.load_config(
        config_file,
        **{"network_generation.substrate_nodes": 100}
    )
    print(f"Modified config: {config.network_generation.substrate_nodes} nodes")
    
    # Create directories
    config_manager.create_directories()
    print("Created directory structure")


if __name__ == "__main__":
    example_usage()
    