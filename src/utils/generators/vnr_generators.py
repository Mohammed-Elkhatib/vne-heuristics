"""
VNR (Virtual Network Request) Generators for Virtual Network Embedding.

This module provides functions to generate random VNRs with configurable
topologies, resource requirements, and temporal patterns for VNE experiments.
"""

import logging
import random
import time
from typing import List, Dict, Any, Tuple, Optional

from src.models.virtual_request import VirtualNetworkRequest
from src.models.vnr_batch import VNRBatch
from src.models.substrate import SubstrateNetwork
from src.utils.generators.generation_config import (
    NetworkGenerationConfig,
    validate_topology_name,
    validate_distribution_name,
    calculate_resource_requirement,
    VALID_VNR_TOPOLOGIES,
    VALID_ARRIVAL_PATTERNS,
    VALID_HOLDING_TIME_DISTRIBUTIONS,
    TopologyError,
    ResourceError,
    ConfigurationError
)


logger = logging.getLogger(__name__)


def generate_vnr(substrate_nodes: List[str],
                vnr_nodes_count: Optional[int] = None,
                topology: str = "random",
                edge_probability: float = 0.5,
                enable_memory_constraints: bool = False,
                enable_delay_constraints: bool = False,
                enable_reliability_constraints: bool = False,
                cpu_ratio_range: Tuple[float, float] = (0.1, 0.3),
                memory_ratio_range: Tuple[float, float] = (0.1, 0.3),
                bandwidth_ratio_range: Tuple[float, float] = (0.1, 0.3),
                delay_ratio_range: Tuple[float, float] = (0.1, 0.5),
                reliability_min_range: Tuple[float, float] = (0.8, 0.95),
                substrate_cpu_range: Tuple[float, float] = (50, 100),
                substrate_bandwidth_range: Tuple[float, float] = (50, 100),
                substrate_delay_max: float = 10.0,
                holding_time: Optional[float] = None,
                arrival_time: float = 0.0,
                priority: int = 1,
                **kwargs) -> VirtualNetworkRequest:
    """
    Generate a single Virtual Network Request.

    Args:
        substrate_nodes: List of substrate node IDs for reference
        vnr_nodes_count: Number of virtual nodes (random if None)
        topology: VNR topology ("random", "star", "linear", "tree")
        edge_probability: Edge probability for random topology
        enable_memory_constraints: Whether VNR should have memory requirements
        enable_delay_constraints: Whether VNR should have delay constraints
        enable_reliability_constraints: Whether VNR should have reliability requirements
        cpu_ratio_range: CPU requirement as ratio of substrate capacity
        memory_ratio_range: Memory requirement as ratio of substrate capacity
        bandwidth_ratio_range: Bandwidth requirement as ratio of substrate capacity
        delay_ratio_range: Delay constraint as ratio of max substrate delay
        reliability_min_range: Minimum reliability requirement range
        substrate_cpu_range: Reference substrate CPU range for ratio calculations
        substrate_bandwidth_range: Reference substrate bandwidth range for ratio calculations
        substrate_delay_max: Maximum substrate delay for ratio calculations
        holding_time: VNR holding time (generated if None)
        arrival_time: VNR arrival time
        priority: VNR priority level
        **kwargs: Additional parameters

    Returns:
        VirtualNetworkRequest instance

    Raises:
        TopologyError: If topology is not supported
        ResourceError: If resource ranges are invalid
        ValueError: If parameters are invalid

    Example:
        >>> # Yu 2008 style VNR (CPU + Bandwidth only)
        >>> vnr = generate_vnr(["0", "1", "2"], vnr_nodes_count=3)
        >>> # Full constraint VNR
        >>> vnr = generate_vnr(["0", "1", "2"], vnr_nodes_count=3,
        ...     enable_memory_constraints=True, enable_delay_constraints=True)
    """
    # Validate inputs
    validate_topology_name(topology, VALID_VNR_TOPOLOGIES)

    if vnr_nodes_count is None:
        vnr_nodes_count = random.randint(2, min(10, len(substrate_nodes)))

    if vnr_nodes_count <= 0:
        raise ValueError("VNR node count must be positive")

    if vnr_nodes_count > len(substrate_nodes):
        raise ValueError(f"VNR node count ({vnr_nodes_count}) exceeds substrate nodes ({len(substrate_nodes)})")

    if not (0 <= edge_probability <= 1):
        raise ValueError("Edge probability must be between 0 and 1")

    if priority < 0:
        raise ValueError("Priority must be non-negative")

    # Validate resource ratio ranges
    _validate_ratio_ranges(cpu_ratio_range, memory_ratio_range, bandwidth_ratio_range,
                          delay_ratio_range, reliability_min_range)

    vnr_id = kwargs.get('vnr_id', f"vnr_{int(time.time() * 1000000) % 1000000}")

    if holding_time is None:
        holding_time = random.expovariate(1.0 / 1000.0)  # Default exponential distribution

    logger.debug(f"Generating VNR {vnr_id}: {vnr_nodes_count} nodes, topology={topology}, "
                f"constraints=memory:{enable_memory_constraints}, delay:{enable_delay_constraints}, "
                f"reliability:{enable_reliability_constraints}")

    # Create VNR instance
    vnr = VirtualNetworkRequest(
        vnr_id=vnr_id,
        arrival_time=arrival_time,
        holding_time=holding_time,
        priority=priority
    )

    # Generate virtual nodes
    _generate_vnr_nodes(
        vnr, vnr_nodes_count, enable_memory_constraints,
        cpu_ratio_range, memory_ratio_range,
        substrate_cpu_range
    )

    # Generate virtual links based on topology
    if topology == "random":
        _generate_random_vnr_links(
            vnr, vnr_nodes_count, edge_probability,
            bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )
    elif topology == "star":
        _generate_star_vnr_links(
            vnr, vnr_nodes_count,
            bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )
    elif topology == "linear":
        _generate_linear_vnr_links(
            vnr, vnr_nodes_count,
            bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )
    elif topology == "tree":
        _generate_tree_vnr_links(
            vnr, vnr_nodes_count,
            bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )
    else:
        # This should not happen due to validation, but just in case
        raise TopologyError(f"Unsupported VNR topology: {topology}")

    logger.debug(f"Generated VNR {vnr_id}")
    return vnr


def generate_vnr_from_config(substrate_nodes: List[str],
                           config: NetworkGenerationConfig,
                           substrate_network: Optional[SubstrateNetwork] = None,
                           **kwargs) -> VirtualNetworkRequest:
    """
    Generate VNR using configuration object.

    Args:
        substrate_nodes: List of substrate node IDs
        config: NetworkGenerationConfig instance
        substrate_network: Optional substrate network for parameter inference
        **kwargs: Override specific parameters

    Returns:
        VirtualNetworkRequest instance

    Example:
        >>> config = NetworkGenerationConfig(enable_memory_constraints=True)
        >>> vnr = generate_vnr_from_config(substrate_nodes, config)
    """
    # Infer substrate parameters if substrate network provided
    if substrate_network:
        stats = substrate_network.get_network_statistics()
        substrate_cpu_range = (0, stats['total_cpu'] / stats['node_count']) if stats['node_count'] > 0 else (50, 100)
        substrate_bandwidth_range = (0, stats['total_bandwidth'] / stats['link_count']) if stats['link_count'] > 0 else (50, 100)

        # Get max delay from substrate network links
        substrate_delay_max = 10.0  # Default fallback
        if hasattr(substrate_network, 'graph'):
            delays = []
            for src, dst in substrate_network.graph.edges:
                link_res = substrate_network.get_link_resources(src, dst)
                if link_res and link_res.delay > 0:
                    delays.append(link_res.delay)
            if delays:
                substrate_delay_max = max(delays)
    else:
        # Use config defaults
        substrate_cpu_range = config.cpu_range
        substrate_bandwidth_range = config.bandwidth_range
        substrate_delay_max = config.delay_range[1]

    # Start with config values
    params = {
        'vnr_nodes_count': random.randint(*config.vnr_nodes_range),
        'topology': config.vnr_topology,
        'edge_probability': config.vnr_edge_probability,
        'enable_memory_constraints': config.enable_memory_constraints,
        'enable_delay_constraints': config.enable_delay_constraints,
        'enable_reliability_constraints': config.enable_reliability_constraints,
        'cpu_ratio_range': config.vnr_cpu_ratio_range,
        'memory_ratio_range': config.vnr_memory_ratio_range,
        'bandwidth_ratio_range': config.vnr_bandwidth_ratio_range,
        'delay_ratio_range': config.vnr_delay_ratio_range,
        'reliability_min_range': config.vnr_reliability_min_range,
        'substrate_cpu_range': substrate_cpu_range,
        'substrate_bandwidth_range': substrate_bandwidth_range,
        'substrate_delay_max': substrate_delay_max,
        'holding_time': generate_holding_time(config.holding_time_distribution, config.holding_time_mean),
    }

    # Override with any provided kwargs
    params.update(kwargs)

    return generate_vnr(substrate_nodes, **params)


def _generate_vnr_nodes(vnr: VirtualNetworkRequest,
                       vnr_nodes_count: int,
                       enable_memory_constraints: bool,
                       cpu_ratio_range: Tuple[float, float],
                       memory_ratio_range: Tuple[float, float],
                       substrate_cpu_range: Tuple[float, float]) -> None:
    """Generate virtual nodes with resource requirements."""
    for i in range(vnr_nodes_count):
        virtual_node_id = i

        # Always generate CPU requirement
        cpu_requirement = calculate_resource_requirement(
            substrate_cpu_range, cpu_ratio_range, as_integer=True
        )

        # Generate memory requirement only if constraints enabled
        memory_requirement = 0.0
        if enable_memory_constraints:
            memory_requirement = calculate_resource_requirement(
                substrate_cpu_range, memory_ratio_range, as_integer=True  # Use CPU range as reference
            )

        vnr.add_virtual_node(
            node_id=virtual_node_id,
            cpu_requirement=cpu_requirement,
            memory_requirement=memory_requirement
        )


def _generate_random_vnr_links(vnr: VirtualNetworkRequest, nodes: int,
                              edge_probability: float,
                              bandwidth_ratio_range: Tuple[float, float],
                              delay_ratio_range: Tuple[float, float],
                              reliability_min_range: Tuple[float, float],
                              substrate_bandwidth_range: Tuple[float, float],
                              substrate_delay_max: float,
                              enable_delay_constraints: bool,
                              enable_reliability_constraints: bool) -> None:
    """Generate random links for VNR."""
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_probability:
                _add_vnr_link(
                    vnr, i, j, bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
                    substrate_bandwidth_range, substrate_delay_max,
                    enable_delay_constraints, enable_reliability_constraints
                )


def _generate_star_vnr_links(vnr: VirtualNetworkRequest, nodes: int,
                            bandwidth_ratio_range: Tuple[float, float],
                            delay_ratio_range: Tuple[float, float],
                            reliability_min_range: Tuple[float, float],
                            substrate_bandwidth_range: Tuple[float, float],
                            substrate_delay_max: float,
                            enable_delay_constraints: bool,
                            enable_reliability_constraints: bool) -> None:
    """Generate star topology links for VNR (node 0 is center)."""
    center_node = 0

    for i in range(1, nodes):
        _add_vnr_link(
            vnr, center_node, i, bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )


def _generate_linear_vnr_links(vnr: VirtualNetworkRequest, nodes: int,
                              bandwidth_ratio_range: Tuple[float, float],
                              delay_ratio_range: Tuple[float, float],
                              reliability_min_range: Tuple[float, float],
                              substrate_bandwidth_range: Tuple[float, float],
                              substrate_delay_max: float,
                              enable_delay_constraints: bool,
                              enable_reliability_constraints: bool) -> None:
    """Generate linear topology links for VNR."""
    for i in range(nodes - 1):
        _add_vnr_link(
            vnr, i, i + 1, bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )


def _generate_tree_vnr_links(vnr: VirtualNetworkRequest, nodes: int,
                            bandwidth_ratio_range: Tuple[float, float],
                            delay_ratio_range: Tuple[float, float],
                            reliability_min_range: Tuple[float, float],
                            substrate_bandwidth_range: Tuple[float, float],
                            substrate_delay_max: float,
                            enable_delay_constraints: bool,
                            enable_reliability_constraints: bool) -> None:
    """Generate tree topology links for VNR."""
    # Simple binary tree structure
    for i in range(1, nodes):
        parent = (i - 1) // 2

        _add_vnr_link(
            vnr, parent, i, bandwidth_ratio_range, delay_ratio_range, reliability_min_range,
            substrate_bandwidth_range, substrate_delay_max,
            enable_delay_constraints, enable_reliability_constraints
        )


def _add_vnr_link(vnr: VirtualNetworkRequest, src: int, dst: int,
                 bandwidth_ratio_range: Tuple[float, float],
                 delay_ratio_range: Tuple[float, float],
                 reliability_min_range: Tuple[float, float],
                 substrate_bandwidth_range: Tuple[float, float],
                 substrate_delay_max: float,
                 enable_delay_constraints: bool,
                 enable_reliability_constraints: bool) -> None:
    """Add a virtual link with appropriate resource requirements."""
    # Always generate bandwidth requirement
    bandwidth_req = calculate_resource_requirement(
        substrate_bandwidth_range, bandwidth_ratio_range, as_integer=True
    )

    # Generate delay constraint only if enabled
    delay_constraint = 0.0
    if enable_delay_constraints:
        delay_ratio = random.uniform(delay_ratio_range[0], delay_ratio_range[1])
        delay_constraint = substrate_delay_max * delay_ratio

    # Generate reliability requirement only if enabled
    reliability_requirement = 0.0
    if enable_reliability_constraints:
        reliability_requirement = random.uniform(*reliability_min_range)

    vnr.add_virtual_link(
        src_node=src,
        dst_node=dst,
        bandwidth_requirement=bandwidth_req,
        delay_constraint=delay_constraint,
        reliability_requirement=reliability_requirement
    )


def generate_vnr_batch(count: int,
                      substrate_nodes: List[str],
                      config: Optional[NetworkGenerationConfig] = None,
                      substrate_network: Optional[SubstrateNetwork] = None,
                      **kwargs) -> VNRBatch:
    """
    Generate a batch of VNRs for experiments.

    Args:
        count: Number of VNRs to generate
        substrate_nodes: List of substrate node IDs
        config: Generation configuration (uses defaults if None)
        substrate_network: Optional substrate network for parameter inference
        **kwargs: Override specific parameters

    Returns:
        VNRBatch instance

    Raises:
        ValueError: If count is not positive
        ConfigurationError: If configuration is invalid

    Example:
        >>> config = NetworkGenerationConfig(enable_memory_constraints=True)
        >>> batch = generate_vnr_batch(100, substrate_nodes, config)
    """
    if count <= 0:
        raise ValueError("VNR count must be positive")

    if config is None:
        config = NetworkGenerationConfig()

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    logger.info(f"Generating batch of {count} VNRs with constraints: "
               f"memory={config.enable_memory_constraints}, delay={config.enable_delay_constraints}, "
               f"reliability={config.enable_reliability_constraints}")

    vnrs = []
    arrival_times = generate_arrival_times(count, config.arrival_pattern, config.arrival_rate)

    for i in range(count):
        arrival_time = arrival_times[i]

        vnr = generate_vnr_from_config(
            substrate_nodes=substrate_nodes,
            config=config,
            substrate_network=substrate_network,
            arrival_time=arrival_time,
            vnr_id=f"vnr_{i:04d}"
        )

        vnrs.append(vnr)

    # Create VNRBatch
    batch = VNRBatch(vnrs, f"generated_batch_{count}_vnrs")
    logger.info(f"Generated VNR batch with {len(vnrs)} VNRs")
    return batch


def generate_vnr_workload(substrate_network: SubstrateNetwork,
                         duration: float,
                         avg_arrival_rate: float,
                         config: Optional[NetworkGenerationConfig] = None) -> VNRBatch:
    """
    Generate a complete VNR workload for a simulation period.

    Args:
        substrate_network: Target substrate network
        duration: Simulation duration
        avg_arrival_rate: Average VNR arrival rate
        config: Generation configuration

    Returns:
        VNRBatch with VNRs that have arrival times within the duration

    Raises:
        ValueError: If duration or arrival rate are not positive

    Example:
        >>> workload = generate_vnr_workload(substrate, 10000.0, 0.1)
        >>> print(f"Generated {len(workload)} VNRs for simulation")
    """
    if duration <= 0:
        raise ValueError("Duration must be positive")

    if avg_arrival_rate <= 0:
        raise ValueError("Arrival rate must be positive")

    if config is None:
        config = NetworkGenerationConfig()

    # Estimate number of VNRs needed
    estimated_count = int(duration * avg_arrival_rate * 1.2)  # 20% buffer

    # Get substrate node list
    substrate_nodes = [str(node_id) for node_id in substrate_network.graph.nodes()]

    # Inherit constraint configuration from substrate network
    constraint_config = substrate_network.get_constraint_configuration()
    config.enable_memory_constraints = constraint_config['memory_constraints']
    config.enable_delay_constraints = constraint_config['delay_constraints']
    config.enable_cost_constraints = constraint_config['cost_constraints']
    config.enable_reliability_constraints = constraint_config['reliability_constraints']

    # Update arrival rate
    config.arrival_rate = avg_arrival_rate

    # Generate VNR batch
    vnr_batch = generate_vnr_batch(
        count=estimated_count,
        substrate_nodes=substrate_nodes,
        config=config,
        substrate_network=substrate_network
    )

    # Filter VNRs that arrive within the duration
    valid_vnrs = [vnr for vnr in vnr_batch.vnrs if vnr.arrival_time <= duration]

    # Create filtered batch
    filtered_batch = VNRBatch(valid_vnrs, f"workload_{duration}_{avg_arrival_rate}")

    logger.info(f"Generated workload: {len(valid_vnrs)} VNRs over {duration} time units "
               f"(filtered from {estimated_count} generated)")

    return filtered_batch


def generate_arrival_times(count: int,
                          pattern: str = "poisson",
                          rate: float = 10.0,
                          start_time: float = 0.0,
                          **kwargs) -> List[float]:
    """
    Generate arrival times for VNRs based on specified pattern.

    Args:
        count: Number of arrival times to generate
        pattern: Arrival pattern ("poisson", "uniform", "custom")
        rate: Arrival rate (events per time unit)
        start_time: Starting time
        **kwargs: Additional parameters for specific patterns

    Returns:
        List of arrival times sorted in ascending order

    Raises:
        ValueError: If parameters are invalid
        ConfigurationError: If custom function is missing for custom pattern

    Example:
        >>> arrival_times = generate_arrival_times(50, "poisson", 5.0)
        >>> print(f"First arrival: {arrival_times[0]:.2f}")
    """
    validate_distribution_name(pattern, VALID_ARRIVAL_PATTERNS)

    if count <= 0:
        raise ValueError("Count must be positive")

    if rate <= 0:
        raise ValueError("Arrival rate must be positive")

    logger.debug(f"Generating {count} arrival times with {pattern} pattern, rate={rate}")

    arrival_times = []

    if pattern == "poisson":
        current_time = start_time
        for _ in range(count):
            # Inter-arrival time follows exponential distribution
            inter_arrival = random.expovariate(rate)
            current_time += inter_arrival
            arrival_times.append(current_time)

    elif pattern == "uniform":
        # Uniform distribution over specified time interval
        end_time = kwargs.get('end_time', start_time + count / rate)
        arrival_times = [random.uniform(start_time, end_time) for _ in range(count)]
        arrival_times.sort()

    elif pattern == "custom":
        # Custom pattern defined by user function
        custom_function = kwargs.get('custom_function')
        if custom_function is None:
            raise ConfigurationError("Custom pattern requires 'custom_function' parameter")

        if not callable(custom_function):
            raise ConfigurationError("custom_function must be callable")

        arrival_times = [custom_function(i, count, rate, start_time) for i in range(count)]
        arrival_times.sort()

    else:
        # This should not happen due to validation, but just in case
        raise ConfigurationError(f"Unsupported arrival pattern: {pattern}")

    logger.debug(f"Generated arrival times from {min(arrival_times):.2f} to {max(arrival_times):.2f}")
    return arrival_times


def generate_holding_time(distribution: str, mean_holding_time: float) -> float:
    """
    Generate VNR holding time based on specified distribution.

    Args:
        distribution: Distribution type ("exponential", "uniform", "fixed")
        mean_holding_time: Mean holding time

    Returns:
        Generated holding time

    Raises:
        ValueError: If mean_holding_time is not positive
        ConfigurationError: If distribution is not supported

    Example:
        >>> holding_time = generate_holding_time("exponential", 1000.0)
    """
    validate_distribution_name(distribution, VALID_HOLDING_TIME_DISTRIBUTIONS)

    if mean_holding_time <= 0:
        raise ValueError("Mean holding time must be positive")

    if distribution == "exponential":
        return random.expovariate(1.0 / mean_holding_time)
    elif distribution == "uniform":
        # Uniform distribution around mean
        half_range = mean_holding_time * 0.5
        return random.uniform(mean_holding_time - half_range, mean_holding_time + half_range)
    elif distribution == "fixed":
        return mean_holding_time
    else:
        # This should not happen due to validation, but just in case
        raise ConfigurationError(f"Unsupported holding time distribution: {distribution}")


def validate_vnr(vnr: VirtualNetworkRequest) -> Dict[str, bool]:
    """
    Validate a generated VNR for consistency and realism.

    Args:
        vnr: VirtualNetworkRequest instance to validate

    Returns:
        Dictionary of validation results

    Example:
        >>> validation = validate_vnr(vnr)
        >>> if all(validation.values()):
        ...     print("VNR validation passed")
    """
    validation_results = {
        'has_nodes': False,
        'has_links': False,
        'connected': False,
        'realistic_requirements': False,
        'consistent_data': False,
        'constraint_compliance': False
    }

    try:
        # Check basic structure
        if hasattr(vnr, 'virtual_nodes') and len(vnr.virtual_nodes) > 0:
            validation_results['has_nodes'] = True

        if hasattr(vnr, 'virtual_links') and len(vnr.virtual_links) > 0:
            validation_results['has_links'] = True

        # Check connectivity (for multi-node VNRs)
        if validation_results['has_nodes'] and validation_results['has_links']:
            import networkx as nx
            validation_results['connected'] = nx.is_connected(vnr.graph)
        elif len(vnr.virtual_nodes) == 1:
            # Single node VNR is trivially connected
            validation_results['connected'] = True

        # Check requirement realism
        validation_results['realistic_requirements'] = _validate_vnr_requirements(vnr)

        # Check data consistency
        validation_results['consistent_data'] = _validate_vnr_consistency(vnr)

        # Check constraint compliance
        validation_results['constraint_compliance'] = _validate_vnr_constraint_compliance(vnr)

    except Exception as e:
        logger.error(f"VNR validation error: {e}")
        validation_results['error'] = str(e)

    logger.debug(f"VNR validation results: {validation_results}")
    return validation_results


def _validate_vnr_requirements(vnr: VirtualNetworkRequest) -> bool:
    """Validate VNR resource requirements are realistic."""
    try:
        # Check node requirements
        for node_id, node_req in vnr.virtual_nodes.items():
            if node_req.cpu_requirement <= 0:
                return False
            if node_req.memory_requirement < 0:
                return False

        # Check link requirements
        for (src, dst), link_req in vnr.virtual_links.items():
            if link_req.bandwidth_requirement <= 0:
                return False
            if link_req.delay_constraint < 0:
                return False
            if not (0 <= link_req.reliability_requirement <= 1):
                return False

        return True
    except Exception:
        return False


def _validate_vnr_consistency(vnr: VirtualNetworkRequest) -> bool:
    """Validate VNR data consistency."""
    try:
        # Check that all virtual links reference existing virtual nodes
        for (src, dst) in vnr.virtual_links.keys():
            if src not in vnr.virtual_nodes or dst not in vnr.virtual_nodes:
                return False

        # Check for self-loops
        for (src, dst) in vnr.virtual_links.keys():
            if src == dst:
                logger.warning(f"Self-loop detected in VNR {vnr.vnr_id}: {src} -> {dst}")

        # Check temporal parameters
        if vnr.holding_time <= 0 and vnr.holding_time != float('inf'):
            return False

        if vnr.arrival_time < 0:
            return False

        return True
    except Exception:
        return False


def _validate_vnr_constraint_compliance(vnr: VirtualNetworkRequest) -> bool:
    """Validate that VNR complies with constraint usage patterns."""
    try:
        constraint_summary = vnr.get_constraint_summary()

        # Check that nodes without memory requirements have memory_requirement = 0
        for node_req in vnr.virtual_nodes.values():
            if node_req.memory_requirement == 0:
                continue  # This is correct
            elif node_req.memory_requirement > 0 and constraint_summary['uses_memory_constraints']:
                continue  # This is correct
            else:
                return False  # Memory requirement but no memory constraints used

        # Check that links without constraints have appropriate default values
        for link_req in vnr.virtual_links.values():
            if link_req.delay_constraint == 0 and not constraint_summary['uses_delay_constraints']:
                continue  # Correct
            elif link_req.delay_constraint > 0 and constraint_summary['uses_delay_constraints']:
                continue  # Correct
            elif link_req.delay_constraint == 0 and constraint_summary['uses_delay_constraints']:
                continue  # Valid - this link doesn't need delay constraint
            else:
                return False  # Inconsistent delay constraint usage

            if link_req.reliability_requirement == 0 and not constraint_summary['uses_reliability_constraints']:
                continue  # Correct
            elif link_req.reliability_requirement > 0 and constraint_summary['uses_reliability_constraints']:
                continue  # Correct
            elif link_req.reliability_requirement == 0 and constraint_summary['uses_reliability_constraints']:
                continue  # Valid - this link doesn't need reliability constraint
            else:
                return False  # Inconsistent reliability constraint usage

        return True
    except Exception:
        return False


def _validate_ratio_ranges(*ratio_ranges) -> None:
    """Validate that ratio ranges are valid."""
    for i, range_tuple in enumerate(ratio_ranges):
        if len(range_tuple) != 2:
            raise ResourceError(f"Ratio range {i} must have exactly 2 values")

        min_val, max_val = range_tuple
        if min_val > max_val:
            raise ResourceError(f"Ratio range {i} has min > max: ({min_val}, {max_val})")

        if min_val < 0:
            raise ResourceError(f"Ratio range {i} has negative minimum: {min_val}")

        # Check for reasonable ratio ranges (warn if > 1.0)
        if max_val > 1.0:
            logger.warning(f"Ratio range {i} has max > 1.0: this means VNR requirements "
                          f"can exceed substrate capacity")


def create_vnr_scenarios_config() -> Dict[str, Dict[str, Any]]:
    """
    Create predefined VNR generation scenarios.

    These scenarios complement the substrate scenarios and can be used
    with any substrate network to create different workload patterns.

    Returns:
        Dictionary of scenario name to VNR-specific parameters

    Example:
        >>> scenarios = create_vnr_scenarios_config()
        >>> light_load = scenarios['light_load']
        >>> config = NetworkGenerationConfig(**light_load)
    """
    scenarios = {
        'light_load': {
            'vnr_nodes_range': (2, 4),
            'vnr_cpu_ratio_range': (0.05, 0.15),
            'vnr_bandwidth_ratio_range': (0.05, 0.15),
            'vnr_memory_ratio_range': (0.05, 0.15),
            'arrival_rate': 5.0,
            'holding_time_mean': 2000.0,
            'vnr_topology': 'linear'
        },

        'medium_load': {
            'vnr_nodes_range': (3, 6),
            'vnr_cpu_ratio_range': (0.1, 0.3),
            'vnr_bandwidth_ratio_range': (0.1, 0.3),
            'vnr_memory_ratio_range': (0.1, 0.3),
            'arrival_rate': 10.0,
            'holding_time_mean': 1000.0,
            'vnr_topology': 'random'
        },

        'heavy_load': {
            'vnr_nodes_range': (4, 8),
            'vnr_cpu_ratio_range': (0.2, 0.5),
            'vnr_bandwidth_ratio_range': (0.2, 0.5),
            'vnr_memory_ratio_range': (0.2, 0.5),
            'arrival_rate': 20.0,
            'holding_time_mean': 500.0,
            'vnr_topology': 'random'
        },

        'bursty_arrivals': {
            'vnr_nodes_range': (2, 6),
            'arrival_pattern': 'uniform',  # Creates bursts when used with short time windows
            'arrival_rate': 30.0,
            'holding_time_mean': 800.0,
            'holding_time_distribution': 'uniform'
        },

        'long_holding': {
            'vnr_nodes_range': (3, 7),
            'arrival_rate': 5.0,
            'holding_time_mean': 5000.0,  # Very long holding times
            'holding_time_distribution': 'fixed'
        },

        'short_holding': {
            'vnr_nodes_range': (2, 5),
            'arrival_rate': 25.0,
            'holding_time_mean': 200.0,  # Very short holding times
            'holding_time_distribution': 'exponential'
        },

        'small_vnrs': {
            'vnr_nodes_range': (2, 3),
            'vnr_cpu_ratio_range': (0.05, 0.2),
            'vnr_bandwidth_ratio_range': (0.05, 0.2),
            'vnr_topology': 'linear',
            'arrival_rate': 15.0
        },

        'large_vnrs': {
            'vnr_nodes_range': (6, 12),
            'vnr_cpu_ratio_range': (0.1, 0.4),
            'vnr_bandwidth_ratio_range': (0.1, 0.4),
            'vnr_topology': 'random',
            'vnr_edge_probability': 0.6,
            'arrival_rate': 5.0
        },

        'star_topology_vnrs': {
            'vnr_nodes_range': (3, 8),
            'vnr_topology': 'star',
            'vnr_cpu_ratio_range': (0.1, 0.3),
            'vnr_bandwidth_ratio_range': (0.1, 0.3),
            'arrival_rate': 12.0
        },

        'tree_topology_vnrs': {
            'vnr_nodes_range': (4, 10),
            'vnr_topology': 'tree',
            'vnr_cpu_ratio_range': (0.08, 0.25),
            'vnr_bandwidth_ratio_range': (0.08, 0.25),
            'arrival_rate': 10.0
        },

        'delay_sensitive': {
            'vnr_nodes_range': (2, 5),
            'vnr_delay_ratio_range': (0.05, 0.2),  # Strict delay requirements
            'vnr_cpu_ratio_range': (0.1, 0.25),
            'vnr_bandwidth_ratio_range': (0.1, 0.25),
            'vnr_topology': 'linear',  # Simpler topology for delay-sensitive
            'arrival_rate': 8.0
        },

        'reliability_critical': {
            'vnr_nodes_range': (3, 6),
            'vnr_reliability_min_range': (0.95, 0.99),  # High reliability requirements
            'vnr_cpu_ratio_range': (0.15, 0.35),
            'vnr_bandwidth_ratio_range': (0.15, 0.35),
            'arrival_rate': 6.0,
            'holding_time_mean': 1500.0
        },

        'mixed_constraints': {
            'vnr_nodes_range': (3, 8),
            'vnr_cpu_ratio_range': (0.1, 0.35),
            'vnr_memory_ratio_range': (0.1, 0.35),
            'vnr_bandwidth_ratio_range': (0.1, 0.35),
            'vnr_delay_ratio_range': (0.1, 0.4),
            'vnr_reliability_min_range': (0.85, 0.95),
            'arrival_rate': 12.0,
            'vnr_topology': 'random'
        }
    }

    logger.info(f"Created {len(scenarios)} predefined VNR scenarios")
    return scenarios
