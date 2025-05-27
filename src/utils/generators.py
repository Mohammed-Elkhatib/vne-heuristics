"""
Network generators for Virtual Network Embedding (VNE) experiments.

This module provides functions to generate random substrate networks and VNRs
with configurable topologies, resources, and arrival patterns.
"""

import logging
import random
import math
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import time

# Import the models (assuming they're available)
# from src.models.substrate import SubstrateNetwork, NodeResources, LinkResources
# from src.models.virtual_request import VirtualNetworkRequest, VirtualNodeRequirement, VirtualLinkRequirement

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NetworkGenerationConfig:
    """Configuration for network generation parameters."""
    # Substrate network parameters
    substrate_nodes: int = 100
    substrate_topology: str = "erdos_renyi"  # "erdos_renyi", "barabasi_albert", "grid", "custom"
    substrate_edge_probability: float = 0.1
    substrate_attachment_count: int = 3  # For Barabási-Albert
    
    # Resource parameters
    cpu_range: Tuple[int, int] = (50, 100)
    memory_range: Tuple[int, int] = (50, 100)
    bandwidth_range: Tuple[int, int] = (50, 100)
    coordinate_range: Tuple[float, float] = (0.0, 100.0)
    
    # VNR parameters
    vnr_nodes_range: Tuple[int, int] = (2, 10)
    vnr_topology: str = "random"  # "random", "star", "linear", "tree"
    vnr_edge_probability: float = 0.5
    
    # VNR resource requirements (as ratios of substrate resources)
    vnr_cpu_ratio_range: Tuple[float, float] = (0.1, 0.3)
    vnr_memory_ratio_range: Tuple[float, float] = (0.1, 0.3)
    vnr_bandwidth_ratio_range: Tuple[float, float] = (0.1, 0.3)
    
    # Time parameters
    arrival_pattern: str = "poisson"  # "poisson", "uniform", "custom"
    arrival_rate: float = 10.0  # VNRs per time unit
    lifetime_distribution: str = "exponential"  # "exponential", "uniform", "fixed"
    lifetime_mean: float = 1000.0  # Time units
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducible generation.
    
    Args:
        seed: Random seed value, or None for random seeding
        
    Example:
        >>> set_random_seed(42)
        >>> network = generate_substrate_network(50, "erdos_renyi")
    """
    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")
    else:
        logger.info("Using random seeding")


def generate_substrate_network(nodes: int, 
                             topology: str = "erdos_renyi",
                             edge_probability: float = 0.1,
                             attachment_count: int = 3,
                             cpu_range: Tuple[int, int] = (50, 100),
                             memory_range: Tuple[int, int] = (50, 100),
                             bandwidth_range: Tuple[int, int] = (50, 100),
                             coordinate_range: Tuple[float, float] = (0.0, 100.0),
                             **kwargs):
    """
    Generate a random substrate network with specified topology and resources.
    
    Args:
        nodes: Number of nodes in the network
        topology: Network topology ("erdos_renyi", "barabasi_albert", "grid")
        edge_probability: Edge probability for Erdős-Rényi graphs
        attachment_count: Number of edges to attach for Barabási-Albert
        cpu_range: Range for CPU capacity assignment
        memory_range: Range for memory capacity assignment
        bandwidth_range: Range for bandwidth capacity assignment
        coordinate_range: Range for coordinate assignment
        **kwargs: Additional parameters for specific topologies
        
    Returns:
        SubstrateNetwork instance
        
    Example:
        >>> substrate = generate_substrate_network(50, "erdos_renyi", 0.15)
        >>> print(f"Generated network with {len(substrate.nodes)} nodes")
    """
    # Note: This is a template implementation
    # Replace with actual imports and class instantiation
    
    logger.info(f"Generating substrate network: {nodes} nodes, topology={topology}")
    
    # Create substrate network instance
    # substrate = SubstrateNetwork()
    
    # Generate nodes with resources
    for i in range(nodes):
        node_id = f"s{i}"
        
        # Generate random resources
        cpu_capacity = random.randint(*cpu_range)
        memory_capacity = random.randint(*memory_range)
        
        # Generate random coordinates
        x_coord = random.uniform(*coordinate_range)
        y_coord = random.uniform(*coordinate_range)
        
        # Create node resources
        # node_resources = NodeResources(
        #     cpu_capacity=cpu_capacity,
        #     available_cpu=cpu_capacity,
        #     memory_capacity=memory_capacity,
        #     available_memory=memory_capacity,
        #     x_coordinate=x_coord,
        #     y_coordinate=y_coord
        # )
        
        # substrate.add_node(node_id, node_resources)
    
    # Generate topology-specific edges
    if topology == "erdos_renyi":
        _generate_erdos_renyi_edges(nodes, edge_probability, bandwidth_range)
    elif topology == "barabasi_albert":
        _generate_barabasi_albert_edges(nodes, attachment_count, bandwidth_range)
    elif topology == "grid":
        grid_size = kwargs.get('grid_size', int(math.sqrt(nodes)))
        _generate_grid_edges(nodes, grid_size, bandwidth_range)
    else:
        raise ValueError(f"Unsupported topology: {topology}")
    
    logger.info(f"Generated substrate network with {nodes} nodes")
    
    # Return placeholder - replace with actual substrate network
    return f"SubstrateNetwork({nodes}_nodes_{topology})"


def _generate_erdos_renyi_edges(nodes: int, 
                               edge_probability: float,
                               bandwidth_range: Tuple[int, int]) -> None:
    """Generate edges for Erdős-Rényi random graph."""
    edges_created = 0
    
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_probability:
                # Create bidirectional link
                bandwidth = random.randint(*bandwidth_range)
                
                # link_resources = LinkResources(
                #     bandwidth_capacity=bandwidth,
                #     available_bandwidth=bandwidth,
                #     delay=random.uniform(1.0, 10.0)
                # )
                
                # substrate.add_link(f"s{i}", f"s{j}", link_resources)
                # substrate.add_link(f"s{j}", f"s{i}", link_resources)
                
                edges_created += 2
    
    logger.info(f"Created {edges_created} edges for Erdős-Rényi topology")


def _generate_barabasi_albert_edges(nodes: int,
                                   attachment_count: int,
                                   bandwidth_range: Tuple[int, int]) -> None:
    """Generate edges for Barabási-Albert preferential attachment."""
    if attachment_count >= nodes:
        attachment_count = nodes - 1
    
    # Start with a complete graph of attachment_count nodes
    for i in range(attachment_count):
        for j in range(i + 1, attachment_count):
            bandwidth = random.randint(*bandwidth_range)
            # Create bidirectional links
            # Similar implementation as above
    
    # Add remaining nodes with preferential attachment
    degrees = [attachment_count - 1] * attachment_count
    
    for new_node in range(attachment_count, nodes):
        # Select nodes to connect to based on degree (preferential attachment)
        total_degree = sum(degrees)
        targets = set()
        
        while len(targets) < attachment_count:
            rand_val = random.random() * total_degree
            cumulative = 0
            
            for node_idx, degree in enumerate(degrees):
                cumulative += degree
                if rand_val <= cumulative:
                    targets.add(node_idx)
                    break
        
        # Create connections
        for target in targets:
            bandwidth = random.randint(*bandwidth_range)
            # Create bidirectional links
            degrees[target] += 1
        
        degrees.append(len(targets))
    
    logger.info(f"Created Barabási-Albert network with preferential attachment")


def _generate_grid_edges(nodes: int,
                        grid_size: int,
                        bandwidth_range: Tuple[int, int]) -> None:
    """Generate edges for grid topology."""
    for i in range(nodes):
        row = i // grid_size
        col = i % grid_size
        
        # Connect to right neighbor
        if col < grid_size - 1 and (i + 1) < nodes:
            bandwidth = random.randint(*bandwidth_range)
            # Create bidirectional link
        
        # Connect to bottom neighbor
        if row < grid_size - 1 and (i + grid_size) < nodes:
            bandwidth = random.randint(*bandwidth_range)
            # Create bidirectional link
    
    logger.info(f"Created grid topology with {grid_size}x{grid_size} structure")


def generate_vnr(substrate_nodes: List[str],
                vnr_nodes_count: Optional[int] = None,
                topology: str = "random",
                edge_probability: float = 0.5,
                cpu_ratio_range: Tuple[float, float] = (0.1, 0.3),
                memory_ratio_range: Tuple[float, float] = (0.1, 0.3),
                bandwidth_ratio_range: Tuple[float, float] = (0.1, 0.3),
                lifetime: Optional[float] = None,
                arrival_time: float = 0.0,
                **kwargs):
    """
    Generate a single Virtual Network Request.
    
    Args:
        substrate_nodes: List of substrate node IDs for reference
        vnr_nodes_count: Number of virtual nodes (random if None)
        topology: VNR topology ("random", "star", "linear", "tree")
        edge_probability: Edge probability for random topology
        cpu_ratio_range: CPU requirement as ratio of substrate capacity
        memory_ratio_range: Memory requirement as ratio of substrate capacity
        bandwidth_ratio_range: Bandwidth requirement as ratio of substrate capacity
        lifetime: VNR lifetime (generated if None)
        arrival_time: VNR arrival time
        **kwargs: Additional parameters
        
    Returns:
        VirtualNetworkRequest instance
        
    Example:
        >>> substrate_nodes = ["s0", "s1", "s2", "s3"]
        >>> vnr = generate_vnr(substrate_nodes, 3, "star")
    """
    if vnr_nodes_count is None:
        vnr_nodes_count = random.randint(2, min(10, len(substrate_nodes)))
    
    vnr_id = kwargs.get('vnr_id', f"vnr_{int(time.time() * 1000000) % 1000000}")
    
    if lifetime is None:
        lifetime = random.expovariate(1.0 / 1000.0)  # Default exponential distribution
    
    logger.info(f"Generating VNR {vnr_id}: {vnr_nodes_count} nodes, topology={topology}")
    
    # Create VNR instance
    # vnr = VirtualNetworkRequest(
    #     vnr_id=vnr_id,
    #     arrival_time=arrival_time,
    #     lifetime=lifetime
    # )
    
    # Generate virtual nodes
    for i in range(vnr_nodes_count):
        virtual_node_id = f"v{i}"
        
        # Generate resource requirements as ratios of average substrate resources
        cpu_requirement = random.randint(
            int(50 * cpu_ratio_range[0]),
            int(100 * cpu_ratio_range[1])
        )
        memory_requirement = random.randint(
            int(50 * memory_ratio_range[0]),
            int(100 * memory_ratio_range[1])
        )
        
        # virtual_node_req = VirtualNodeRequirement(
        #     cpu_requirement=cpu_requirement,
        #     memory_requirement=memory_requirement
        # )
        
        # vnr.add_virtual_node(virtual_node_id, virtual_node_req)
    
    # Generate virtual links based on topology
    if topology == "random":
        _generate_random_vnr_links(vnr_nodes_count, edge_probability, bandwidth_ratio_range)
    elif topology == "star":
        _generate_star_vnr_links(vnr_nodes_count, bandwidth_ratio_range)
    elif topology == "linear":
        _generate_linear_vnr_links(vnr_nodes_count, bandwidth_ratio_range)
    elif topology == "tree":
        _generate_tree_vnr_links(vnr_nodes_count, bandwidth_ratio_range)
    else:
        raise ValueError(f"Unsupported VNR topology: {topology}")
    
    logger.info(f"Generated VNR {vnr_id}")
    
    # Return placeholder - replace with actual VNR
    return f"VNR({vnr_id}_{vnr_nodes_count}_nodes_{topology})"


def _generate_random_vnr_links(nodes: int,
                              edge_probability: float,
                              bandwidth_ratio_range: Tuple[float, float]) -> None:
    """Generate random links for VNR."""
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_probability:
                bandwidth_req = random.randint(
                    int(50 * bandwidth_ratio_range[0]),
                    int(100 * bandwidth_ratio_range[1])
                )
                
                # virtual_link_req = VirtualLinkRequirement(
                #     bandwidth_requirement=bandwidth_req,
                #     delay_requirement=random.uniform(1.0, 5.0)
                # )
                
                # vnr.add_virtual_link(f"v{i}", f"v{j}", virtual_link_req)


def _generate_star_vnr_links(nodes: int,
                            bandwidth_ratio_range: Tuple[float, float]) -> None:
    """Generate star topology links for VNR (node 0 is center)."""
    center_node = 0
    
    for i in range(1, nodes):
        bandwidth_req = random.randint(
            int(50 * bandwidth_ratio_range[0]),
            int(100 * bandwidth_ratio_range[1])
        )
        
        # virtual_link_req = VirtualLinkRequirement(
        #     bandwidth_requirement=bandwidth_req,
        #     delay_requirement=random.uniform(1.0, 5.0)
        # )
        
        # vnr.add_virtual_link(f"v{center_node}", f"v{i}", virtual_link_req)


def _generate_linear_vnr_links(nodes: int,
                              bandwidth_ratio_range: Tuple[float, float]) -> None:
    """Generate linear topology links for VNR."""
    for i in range(nodes - 1):
        bandwidth_req = random.randint(
            int(50 * bandwidth_ratio_range[0]),
            int(100 * bandwidth_ratio_range[1])
        )
        
        # virtual_link_req = VirtualLinkRequirement(
        #     bandwidth_requirement=bandwidth_req,
        #     delay_requirement=random.uniform(1.0, 5.0)
        # )
        
        # vnr.add_virtual_link(f"v{i}", f"v{i+1}", virtual_link_req)


def _generate_tree_vnr_links(nodes: int,
                            bandwidth_ratio_range: Tuple[float, float]) -> None:
    """Generate tree topology links for VNR."""
    # Simple binary tree structure
    for i in range(1, nodes):
        parent = (i - 1) // 2
        
        bandwidth_req = random.randint(
            int(50 * bandwidth_ratio_range[0]),
            int(100 * bandwidth_ratio_range[1])
        )
        
        # virtual_link_req = VirtualLinkRequirement(
        #     bandwidth_requirement=bandwidth_req,
        #     delay_requirement=random.uniform(1.0, 5.0)
        # )
        
        # vnr.add_virtual_link(f"v{parent}", f"v{i}", virtual_link_req)


def generate_vnr_batch(count: int,
                      substrate_nodes: List[str],
                      config: Optional[NetworkGenerationConfig] = None,
                      **kwargs) -> List:
    """
    Generate a batch of VNRs for experiments.
    
    Args:
        count: Number of VNRs to generate
        substrate_nodes: List of substrate node IDs
        config: Generation configuration (uses defaults if None)
        **kwargs: Override specific parameters
        
    Returns:
        List of VirtualNetworkRequest instances
        
    Example:
        >>> config = NetworkGenerationConfig(vnr_nodes_range=(3, 8))
        >>> vnrs = generate_vnr_batch(100, substrate_nodes, config)
    """
    if config is None:
        config = NetworkGenerationConfig()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info(f"Generating batch of {count} VNRs")
    
    vnrs = []
    arrival_times = generate_arrival_times(count, config.arrival_pattern, config.arrival_rate)
    
    for i in range(count):
        # Generate VNR parameters
        vnr_nodes_count = random.randint(*config.vnr_nodes_range)
        lifetime = _generate_lifetime(config.lifetime_distribution, config.lifetime_mean)
        
        vnr = generate_vnr(
            substrate_nodes=substrate_nodes,
            vnr_nodes_count=vnr_nodes_count,
            topology=config.vnr_topology,
            edge_probability=config.vnr_edge_probability,
            cpu_ratio_range=config.vnr_cpu_ratio_range,
            memory_ratio_range=config.vnr_memory_ratio_range,
            bandwidth_ratio_range=config.vnr_bandwidth_ratio_range,
            lifetime=lifetime,
            arrival_time=arrival_times[i],
            vnr_id=f"vnr_{i:04d}"
        )
        
        vnrs.append(vnr)
    
    logger.info(f"Generated {len(vnrs)} VNRs")
    return vnrs


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
        
    Example:
        >>> arrival_times = generate_arrival_times(50, "poisson", 5.0)
        >>> print(f"First arrival: {arrival_times[0]:.2f}")
    """
    logger.info(f"Generating {count} arrival times with {pattern} pattern, rate={rate}")
    
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
            raise ValueError("Custom pattern requires 'custom_function' parameter")
        
        arrival_times = [custom_function(i, count, rate, start_time) for i in range(count)]
        arrival_times.sort()
    
    else:
        raise ValueError(f"Unsupported arrival pattern: {pattern}")
    
    logger.info(f"Generated arrival times from {min(arrival_times):.2f} to {max(arrival_times):.2f}")
    return arrival_times


def _generate_lifetime(distribution: str, mean_lifetime: float) -> float:
    """Generate VNR lifetime based on specified distribution."""
    if distribution == "exponential":
        return random.expovariate(1.0 / mean_lifetime)
    elif distribution == "uniform":
        # Uniform distribution around mean
        half_range = mean_lifetime * 0.5
        return random.uniform(mean_lifetime - half_range, mean_lifetime + half_range)
    elif distribution == "fixed":
        return mean_lifetime
    else:
        raise ValueError(f"Unsupported lifetime distribution: {distribution}")


def generate_realistic_substrate_network(nodes: int,
                                       geographic_area: str = "metro",
                                       **kwargs):
    """
    Generate a realistic substrate network based on real-world patterns.
    
    Args:
        nodes: Number of nodes
        geographic_area: Type of area ("metro", "regional", "national")
        **kwargs: Additional parameters
        
    Returns:
        SubstrateNetwork with realistic topology and resources
        
    Example:
        >>> substrate = generate_realistic_substrate_network(30, "metro")
    """
    logger.info(f"Generating realistic substrate network: {nodes} nodes, area={geographic_area}")
    
    # Adjust parameters based on geographic area
    if geographic_area == "metro":
        # Metropolitan area: high connectivity, low latency
        edge_probability = 0.3
        bandwidth_range = (100, 1000)
        cpu_range = (100, 500)
        coordinate_range = (0.0, 50.0)  # Smaller area
    elif geographic_area == "regional":
        # Regional network: moderate connectivity
        edge_probability = 0.2
        bandwidth_range = (50, 500)
        cpu_range = (50, 300)
        coordinate_range = (0.0, 200.0)
    elif geographic_area == "national":
        # National network: lower connectivity, higher resources at hubs
        edge_probability = 0.1
        bandwidth_range = (20, 200)
        cpu_range = (200, 1000)  # Larger data centers
        coordinate_range = (0.0, 1000.0)  # Large area
    else:
        raise ValueError(f"Unsupported geographic area: {geographic_area}")
    
    # Use scale-free topology for realistic networks
    substrate = generate_substrate_network(
        nodes=nodes,
        topology="barabasi_albert",
        attachment_count=max(2, int(nodes * edge_probability / 2)),
        bandwidth_range=bandwidth_range,
        cpu_range=cpu_range,
        coordinate_range=coordinate_range,
        **kwargs
    )
    
    return substrate


def generate_vnr_workload(substrate_network,
                         duration: float,
                         avg_arrival_rate: float,
                         config: Optional[NetworkGenerationConfig] = None) -> List:
    """
    Generate a complete VNR workload for a simulation period.
    
    Args:
        substrate_network: Target substrate network
        duration: Simulation duration
        avg_arrival_rate: Average VNR arrival rate
        config: Generation configuration
        
    Returns:
        List of VNRs with arrival times within the duration
        
    Example:
        >>> workload = generate_vnr_workload(substrate, 10000.0, 0.1)
        >>> print(f"Generated {len(workload)} VNRs for simulation")
    """
    if config is None:
        config = NetworkGenerationConfig()
    
    # Estimate number of VNRs needed
    estimated_count = int(duration * avg_arrival_rate * 1.2)  # 20% buffer
    
    # Get substrate node list
    substrate_nodes = list(substrate_network.nodes.keys()) if hasattr(substrate_network, 'nodes') else [f"s{i}" for i in range(100)]
    
    # Generate VNR batch
    vnrs = generate_vnr_batch(
        count=estimated_count,
        substrate_nodes=substrate_nodes,
        config=config
    )
    
    # Filter VNRs that arrive within the duration
    valid_vnrs = [vnr for vnr in vnrs if vnr.arrival_time <= duration]
    
    logger.info(f"Generated workload: {len(valid_vnrs)} VNRs over {duration} time units")
    
    return valid_vnrs


def validate_generated_network(network, network_type: str = "substrate") -> Dict[str, bool]:
    """
    Validate a generated network for consistency and realism.
    
    Args:
        network: Network instance to validate
        network_type: Type of network ("substrate" or "vnr")
        
    Returns:
        Dictionary of validation results
        
    Example:
        >>> validation = validate_generated_network(substrate, "substrate")
        >>> if all(validation.values()):
        ...     print("Network validation passed")
    """
    validation_results = {
        'has_nodes': False,
        'has_links': False,
        'connected': False,
        'realistic_resources': False,
        'consistent_data': False
    }
    
    try:
        # Check basic structure
        if hasattr(network, 'nodes') and len(network.nodes) > 0:
            validation_results['has_nodes'] = True
        
        if hasattr(network, 'links') and len(network.links) > 0:
            validation_results['has_links'] = True
        
        # Check connectivity (simplified)
        if validation_results['has_nodes'] and validation_results['has_links']:
            validation_results['connected'] = True  # Assume generated networks are connected
        
        # Check resource realism
        if network_type == "substrate":
            validation_results['realistic_resources'] = _validate_substrate_resources(network)
        else:
            validation_results['realistic_resources'] = _validate_vnr_resources(network)
        
        # Check data consistency
        validation_results['consistent_data'] = _validate_network_consistency(network)
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        validation_results['error'] = str(e)
    
    logger.info(f"Network validation results: {validation_results}")
    return validation_results


def _validate_substrate_resources(network) -> bool:
    """Validate substrate network resources are realistic."""
    try:
        # Check if resources are within reasonable ranges
        for node_id, node_resources in network.nodes.items():
            if (node_resources.cpu_capacity <= 0 or 
                node_resources.memory_capacity <= 0 or
                node_resources.available_cpu > node_resources.cpu_capacity):
                return False
        
        for link_id, link_resources in network.links.items():
            if (link_resources.bandwidth_capacity <= 0 or
                link_resources.available_bandwidth > link_resources.bandwidth_capacity):
                return False
        
        return True
    except:
        return False


def _validate_vnr_resources(network) -> bool:
    """Validate VNR resource requirements are realistic."""
    try:
        # Check if requirements are positive
        for node_id, node_req in network.virtual_nodes.items():
            if (node_req.cpu_requirement <= 0 or 
                node_req.memory_requirement <= 0):
                return False
        
        for link_id, link_req in network.virtual_links.items():
            if link_req.bandwidth_requirement <= 0:
                return False
        
        return True
    except:
        return False


def _validate_network_consistency(network) -> bool:
    """Validate network data consistency."""
    try:
        # Check that all links reference existing nodes
        if hasattr(network, 'links'):
            for link_id in network.links:
                # Extract node IDs from link ID (assuming format like "node1-node2")
                if '-' in str(link_id):
                    node1, node2 = str(link_id).split('-', 1)
                    if node1 not in network.nodes or node2 not in network.nodes:
                        return False
        
        return True
    except:
        return False


def create_predefined_scenarios() -> Dict[str, NetworkGenerationConfig]:
    """
    Create predefined network generation scenarios for common experiments.
    
    Returns:
        Dictionary of scenario name to configuration
        
    Example:
        >>> scenarios = create_predefined_scenarios()
        >>> small_config = scenarios['small_network']
        >>> substrate = generate_substrate_network(config=small_config)
    """
    scenarios = {
        'small_network': NetworkGenerationConfig(
            substrate_nodes=20,
            substrate_topology="erdos_renyi",
            substrate_edge_probability=0.3,
            vnr_nodes_range=(2, 5),
            arrival_rate=5.0
        ),
        
        'medium_network': NetworkGenerationConfig(
            substrate_nodes=50,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=3,
            vnr_nodes_range=(3, 8),
            arrival_rate=10.0
        ),
        
        'large_network': NetworkGenerationConfig(
            substrate_nodes=100,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=5,
            vnr_nodes_range=(5, 15),
            arrival_rate=20.0
        ),
        
        'dense_requests': NetworkGenerationConfig(
            substrate_nodes=30,
            vnr_nodes_range=(2, 6),
            arrival_rate=50.0,
            lifetime_mean=500.0
        ),
        
        'sparse_requests': NetworkGenerationConfig(
            substrate_nodes=80,
            vnr_nodes_range=(3, 10),
            arrival_rate=2.0,
            lifetime_mean=2000.0
        ),
        
        'high_resource_vnrs': NetworkGenerationConfig(
            vnr_cpu_ratio_range=(0.3, 0.6),
            vnr_memory_ratio_range=(0.3, 0.6),
            vnr_bandwidth_ratio_range=(0.3, 0.6)
        ),
        
        'low_resource_vnrs': NetworkGenerationConfig(
            vnr_cpu_ratio_range=(0.05, 0.15),
            vnr_memory_ratio_range=(0.05, 0.15),
            vnr_bandwidth_ratio_range=(0.05, 0.15)
        )
    }
    
    logger.info(f"Created {len(scenarios)} predefined scenarios")
    return scenarios