"""
Configuration classes and shared utilities for VNE network generation.

This module provides configuration classes and common utilities used by both
substrate and VNR generators to ensure consistent parameter management and
avoid code duplication.
"""

import logging
import random
from typing import Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class NetworkGenerationConfig:
    """
    Configuration for network generation parameters.
    
    This class contains parameters for both substrate network generation and
    VNR generation, along with constraint configuration that applies to both.
    """
    
    # === SUBSTRATE NETWORK PARAMETERS ===
    substrate_nodes: int = 100
    substrate_topology: str = "erdos_renyi"  # "erdos_renyi", "barabasi_albert", "grid"
    substrate_edge_probability: float = 0.1
    substrate_attachment_count: int = 3  # For Barabási-Albert topology
    
    # === CONSTRAINT CONFIGURATION ===
    # Primary constraints (always enforced)
    # - CPU constraints: always enabled
    # - Bandwidth constraints: always enabled
    
    # Secondary constraints (optional)
    enable_memory_constraints: bool = False
    enable_delay_constraints: bool = False
    enable_cost_constraints: bool = False
    enable_reliability_constraints: bool = False
    
    # === SUBSTRATE RESOURCE PARAMETERS ===
    # CPU and bandwidth ranges (always used)
    cpu_range: Tuple[int, int] = (50, 100)
    bandwidth_range: Tuple[int, int] = (50, 100)
    coordinate_range: Tuple[float, float] = (0.0, 100.0)
    
    # Optional resource ranges (used only if respective constraints enabled)
    memory_range: Tuple[int, int] = (50, 100)
    delay_range: Tuple[float, float] = (1.0, 10.0)
    cost_range: Tuple[float, float] = (1.0, 5.0)
    reliability_range: Tuple[float, float] = (0.9, 1.0)
    
    # === VNR PARAMETERS ===
    vnr_nodes_range: Tuple[int, int] = (2, 10)
    vnr_topology: str = "random"  # "random", "star", "linear", "tree"
    vnr_edge_probability: float = 0.5
    
    # === VNR RESOURCE REQUIREMENTS ===
    # Primary requirements (always used)
    vnr_cpu_ratio_range: Tuple[float, float] = (0.1, 0.3)
    vnr_bandwidth_ratio_range: Tuple[float, float] = (0.1, 0.3)
    
    # Optional requirements (used only if respective constraints enabled)
    vnr_memory_ratio_range: Tuple[float, float] = (0.1, 0.3)
    vnr_delay_ratio_range: Tuple[float, float] = (0.1, 0.5)  # As ratio of max substrate delay
    vnr_reliability_min_range: Tuple[float, float] = (0.8, 0.95)  # Minimum reliability requirement
    
    # === TEMPORAL PARAMETERS ===
    arrival_pattern: str = "poisson"  # "poisson", "uniform", "custom"
    arrival_rate: float = 10.0  # VNRs per time unit
    holding_time_distribution: str = "exponential"  # "exponential", "uniform", "fixed"
    holding_time_mean: float = 1000.0  # Time units
    
    # === REPRODUCIBILITY ===
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_ranges()
        self._validate_parameters()
    
    def _validate_ranges(self):
        """Validate that all ranges have min <= max."""
        ranges_to_check = [
            ("cpu_range", self.cpu_range),
            ("bandwidth_range", self.bandwidth_range),
            ("memory_range", self.memory_range),
            ("delay_range", self.delay_range),
            ("cost_range", self.cost_range),
            ("reliability_range", self.reliability_range),
            ("coordinate_range", self.coordinate_range),
            ("vnr_nodes_range", self.vnr_nodes_range),
            ("vnr_cpu_ratio_range", self.vnr_cpu_ratio_range),
            ("vnr_bandwidth_ratio_range", self.vnr_bandwidth_ratio_range),
            ("vnr_memory_ratio_range", self.vnr_memory_ratio_range),
            ("vnr_delay_ratio_range", self.vnr_delay_ratio_range),
            ("vnr_reliability_min_range", self.vnr_reliability_min_range),
        ]
        
        for name, (min_val, max_val) in ranges_to_check:
            if min_val > max_val:
                raise ValueError(f"{name} has min > max: ({min_val}, {max_val})")
    
    def _validate_parameters(self):
        """Validate individual parameters."""
        if self.substrate_nodes <= 0:
            raise ValueError("substrate_nodes must be positive")
        
        if not (0 <= self.substrate_edge_probability <= 1):
            raise ValueError("substrate_edge_probability must be between 0 and 1")
        
        if self.substrate_attachment_count <= 0:
            raise ValueError("substrate_attachment_count must be positive")
        
        if not (0 <= self.vnr_edge_probability <= 1):
            raise ValueError("vnr_edge_probability must be between 0 and 1")
        
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        
        if self.holding_time_mean <= 0:
            raise ValueError("holding_time_mean must be positive")
        
        # Validate reliability ranges
        if not (0 <= self.reliability_range[0] <= self.reliability_range[1] <= 1):
            raise ValueError("reliability_range values must be between 0 and 1")
        
        if not (0 <= self.vnr_reliability_min_range[0] <= self.vnr_reliability_min_range[1] <= 1):
            raise ValueError("vnr_reliability_min_range values must be between 0 and 1")
        
        # Validate ratio ranges
        ratio_ranges = [
            ("vnr_cpu_ratio_range", self.vnr_cpu_ratio_range),
            ("vnr_bandwidth_ratio_range", self.vnr_bandwidth_ratio_range),
            ("vnr_memory_ratio_range", self.vnr_memory_ratio_range),
            ("vnr_delay_ratio_range", self.vnr_delay_ratio_range),
        ]
        
        for name, (min_val, max_val) in ratio_ranges:
            if min_val < 0 or max_val > 1:
                raise ValueError(f"{name} values should typically be between 0 and 1")
    
    def get_constraint_summary(self) -> dict:
        """
        Get a summary of enabled constraints.
        
        Returns:
            Dictionary showing which constraints are enabled
        """
        return {
            'cpu_constraints': True,  # Always enabled
            'bandwidth_constraints': True,  # Always enabled
            'memory_constraints': self.enable_memory_constraints,
            'delay_constraints': self.enable_delay_constraints,
            'cost_constraints': self.enable_cost_constraints,
            'reliability_constraints': self.enable_reliability_constraints
        }
    
    def is_yu2008_compatible(self) -> bool:
        """
        Check if configuration is compatible with Yu et al. 2008 algorithm.
        
        Yu 2008 only uses CPU and bandwidth constraints.
        
        Returns:
            True if configuration uses only CPU and bandwidth constraints
        """
        return (not self.enable_memory_constraints and
                not self.enable_delay_constraints and
                not self.enable_cost_constraints and
                not self.enable_reliability_constraints)
    
    def enable_all_constraints(self):
        """Enable all optional constraints."""
        self.enable_memory_constraints = True
        self.enable_delay_constraints = True
        self.enable_cost_constraints = True
        self.enable_reliability_constraints = True
    
    def disable_all_optional_constraints(self):
        """Disable all optional constraints (Yu 2008 style)."""
        self.enable_memory_constraints = False
        self.enable_delay_constraints = False
        self.enable_cost_constraints = False
        self.enable_reliability_constraints = False
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        constraints = []
        if self.enable_memory_constraints:
            constraints.append("memory")
        if self.enable_delay_constraints:
            constraints.append("delay")
        if self.enable_cost_constraints:
            constraints.append("cost")
        if self.enable_reliability_constraints:
            constraints.append("reliability")
        
        constraint_str = "+".join(constraints) if constraints else "none"
        
        return (f"NetworkGenerationConfig(substrate_nodes={self.substrate_nodes}, "
                f"vnr_nodes_range={self.vnr_nodes_range}, "
                f"constraints=cpu+bandwidth+{constraint_str})")


def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducible generation.
    
    This function should be called before any generation operations to ensure
    reproducible results across substrate and VNR generation.
    
    Args:
        seed: Random seed value, or None for random seeding
        
    Example:
        >>> set_random_seed(42)
        >>> # Now all subsequent random generation will be reproducible
    """
    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")
    else:
        logger.info("Using random seeding")


def validate_topology_name(topology: str, valid_topologies: list) -> None:
    """
    Validate that a topology name is supported.
    
    Args:
        topology: Topology name to validate
        valid_topologies: List of valid topology names
        
    Raises:
        ValueError: If topology is not in valid_topologies
    """
    if topology not in valid_topologies:
        raise ValueError(f"Unsupported topology: {topology}. "
                        f"Valid options: {', '.join(valid_topologies)}")


def validate_distribution_name(distribution: str, valid_distributions: list) -> None:
    """
    Validate that a distribution name is supported.
    
    Args:
        distribution: Distribution name to validate
        valid_distributions: List of valid distribution names
        
    Raises:
        ValueError: If distribution is not in valid_distributions
    """
    if distribution not in valid_distributions:
        raise ValueError(f"Unsupported distribution: {distribution}. "
                        f"Valid options: {', '.join(valid_distributions)}")


def calculate_resource_requirement(substrate_range: Tuple[float, float], 
                                 ratio_range: Tuple[float, float],
                                 as_integer: bool = True) -> float:
    """
    Calculate resource requirement based on substrate capacity and ratio.
    
    This utility function is used by both substrate and VNR generators to
    consistently calculate resource requirements as ratios of substrate capacity.
    
    Args:
        substrate_range: Range of substrate resource capacity
        ratio_range: Range of ratios to apply
        as_integer: Whether to return integer result
        
    Returns:
        Calculated resource requirement
        
    Example:
        >>> # Calculate VNR CPU requirement as 10-30% of substrate CPU capacity
        >>> cpu_req = calculate_resource_requirement((50, 100), (0.1, 0.3))
    """
    # Use average substrate capacity as reference
    substrate_avg = (substrate_range[0] + substrate_range[1]) / 2
    
    # Generate random ratio
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    
    # Calculate requirement
    requirement = substrate_avg * ratio
    
    return int(requirement) if as_integer else requirement


# Constants for validation
VALID_SUBSTRATE_TOPOLOGIES = ["erdos_renyi", "barabasi_albert", "grid"]
VALID_VNR_TOPOLOGIES = ["random", "star", "linear", "tree"]
VALID_ARRIVAL_PATTERNS = ["poisson", "uniform", "custom"]
VALID_HOLDING_TIME_DISTRIBUTIONS = ["exponential", "uniform", "fixed"]
VALID_GEOGRAPHIC_AREAS = ["metro", "regional", "national"]


class GenerationError(Exception):
    """Base exception for generation-related errors."""
    pass


class ConfigurationError(GenerationError):
    """Exception raised for configuration-related errors."""
    pass


class TopologyError(GenerationError):
    """Exception raised for topology-related errors."""
    pass


class ResourceError(GenerationError):
    """Exception raised for resource-related errors."""
    pass