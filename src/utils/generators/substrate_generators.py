"""
Substrate Network Generators for Virtual Network Embedding.

This module provides functions to generate random substrate networks with
configurable topologies, resources, and constraint configurations for VNE experiments.
"""

import logging
import random
import math
from typing import Dict, Tuple

from src.models.substrate import SubstrateNetwork
from src.utils.generators.generation_config import (
    NetworkGenerationConfig,
    validate_topology_name,
    VALID_SUBSTRATE_TOPOLOGIES,
    VALID_GEOGRAPHIC_AREAS,
    TopologyError,
    ResourceError
)


logger = logging.getLogger(__name__)


def generate_substrate_network(nodes: int, 
                             topology: str = "erdos_renyi",
                             edge_probability: float = 0.1,
                             attachment_count: int = 3,
                             enable_memory_constraints: bool = False,
                             enable_delay_constraints: bool = False,
                             enable_cost_constraints: bool = False,
                             enable_reliability_constraints: bool = False,
                             cpu_range: Tuple[int, int] = (50, 100),
                             memory_range: Tuple[int, int] = (50, 100),
                             bandwidth_range: Tuple[int, int] = (50, 100),
                             delay_range: Tuple[float, float] = (1.0, 10.0),
                             cost_range: Tuple[float, float] = (1.0, 5.0),
                             reliability_range: Tuple[float, float] = (0.9, 1.0),
                             coordinate_range: Tuple[float, float] = (0.0, 100.0),
                             **kwargs) -> SubstrateNetwork:
    """
    Generate a random substrate network with specified topology and resources.
    
    Args:
        nodes: Number of nodes in the network
        topology: Network topology ("erdos_renyi", "barabasi_albert", "grid")
        edge_probability: Edge probability for Erdős-Rényi graphs
        attachment_count: Number of edges to attach for Barabási-Albert
        enable_memory_constraints: Whether to enable memory constraints
        enable_delay_constraints: Whether to enable delay constraints
        enable_cost_constraints: Whether to enable cost constraints
        enable_reliability_constraints: Whether to enable reliability constraints
        cpu_range: Range for CPU capacity assignment
        memory_range: Range for memory capacity assignment (ignored if memory constraints disabled)
        bandwidth_range: Range for bandwidth capacity assignment
        delay_range: Range for delay assignment (ignored if delay constraints disabled)
        cost_range: Range for cost assignment (ignored if cost constraints disabled)
        reliability_range: Range for reliability assignment (ignored if reliability constraints disabled)
        coordinate_range: Range for coordinate assignment
        **kwargs: Additional parameters for specific topologies
        
    Returns:
        SubstrateNetwork instance
        
    Raises:
        TopologyError: If topology is not supported
        ResourceError: If resource ranges are invalid
        ValueError: If parameters are invalid
        
    Example:
        >>> # Yu 2008 style (CPU + Bandwidth only)
        >>> substrate = generate_substrate_network(50, "erdos_renyi")
        >>> # Full constraint network
        >>> substrate = generate_substrate_network(50, "erdos_renyi", 
        ...     enable_memory_constraints=True, enable_delay_constraints=True)
    """
    # Validate inputs
    validate_topology_name(topology, VALID_SUBSTRATE_TOPOLOGIES)
    
    if nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    
    if not (0 <= edge_probability <= 1):
        raise ValueError("Edge probability must be between 0 and 1")
    
    if attachment_count <= 0:
        raise ValueError("Attachment count must be positive")
    
    # Validate resource ranges
    _validate_resource_ranges(cpu_range, memory_range, bandwidth_range, 
                             delay_range, cost_range, reliability_range)
    
    logger.info(f"Generating substrate network: {nodes} nodes, topology={topology}, "
               f"constraints=memory:{enable_memory_constraints}, delay:{enable_delay_constraints}, "
               f"cost:{enable_cost_constraints}, reliability:{enable_reliability_constraints}")
    
    # Create substrate network with constraint configuration
    substrate = SubstrateNetwork(
        enable_memory_constraints=enable_memory_constraints,
        enable_delay_constraints=enable_delay_constraints,
        enable_cost_constraints=enable_cost_constraints,
        enable_reliability_constraints=enable_reliability_constraints
    )
    
    # Generate nodes with resources
    _generate_substrate_nodes(
        substrate, nodes, cpu_range, memory_range, coordinate_range,
        enable_memory_constraints
    )
    
    # Generate topology-specific edges
    edges_created = 0
    if topology == "erdos_renyi":
        edges_created = _generate_erdos_renyi_edges(
            substrate, nodes, edge_probability, bandwidth_range,
            delay_range, cost_range, reliability_range,
            enable_delay_constraints, enable_cost_constraints, enable_reliability_constraints
        )
    elif topology == "barabasi_albert":
        edges_created = _generate_barabasi_albert_edges(
            substrate, nodes, attachment_count, bandwidth_range,
            delay_range, cost_range, reliability_range,
            enable_delay_constraints, enable_cost_constraints, enable_reliability_constraints
        )
    elif topology == "grid":
        grid_size = kwargs.get('grid_size', int(math.sqrt(nodes)))
        edges_created = _generate_grid_edges(
            substrate, nodes, grid_size, bandwidth_range,
            delay_range, cost_range, reliability_range,
            enable_delay_constraints, enable_cost_constraints, enable_reliability_constraints
        )
    else:
        # This should not happen due to validation, but just in case
        raise TopologyError(f"Unsupported topology: {topology}")
    
    logger.info(f"Generated substrate network with {nodes} nodes, {edges_created} edges")
    return substrate


def generate_substrate_from_config(config: NetworkGenerationConfig, **kwargs) -> SubstrateNetwork:
    """
    Generate substrate network using configuration object.
    
    Args:
        config: NetworkGenerationConfig instance
        **kwargs: Override specific parameters
        
    Returns:
        SubstrateNetwork instance
        
    Example:
        >>> config = NetworkGenerationConfig(substrate_nodes=50, enable_memory_constraints=True)
        >>> substrate = generate_substrate_from_config(config)
    """
    # Start with config values
    params = {
        'nodes': config.substrate_nodes,
        'topology': config.substrate_topology,
        'edge_probability': config.substrate_edge_probability,
        'attachment_count': config.substrate_attachment_count,
        'enable_memory_constraints': config.enable_memory_constraints,
        'enable_delay_constraints': config.enable_delay_constraints,
        'enable_cost_constraints': config.enable_cost_constraints,
        'enable_reliability_constraints': config.enable_reliability_constraints,
        'cpu_range': config.cpu_range,
        'memory_range': config.memory_range,
        'bandwidth_range': config.bandwidth_range,
        'delay_range': config.delay_range,
        'cost_range': config.cost_range,
        'reliability_range': config.reliability_range,
        'coordinate_range': config.coordinate_range,
    }
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    return generate_substrate_network(**params)


def _generate_substrate_nodes(substrate: SubstrateNetwork, 
                            nodes: int,
                            cpu_range: Tuple[int, int],
                            memory_range: Tuple[int, int],
                            coordinate_range: Tuple[float, float],
                            enable_memory_constraints: bool) -> None:
    """Generate nodes with resources for substrate network."""
    for i in range(nodes):
        node_id = i
        
        # Always generate CPU
        cpu_capacity = random.randint(*cpu_range)
        
        # Generate memory only if constraints enabled
        memory_capacity = random.randint(*memory_range) if enable_memory_constraints else 0.0
        
        # Generate random coordinates
        x_coord = random.uniform(*coordinate_range)
        y_coord = random.uniform(*coordinate_range)
        
        substrate.add_node(
            node_id=node_id,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
            x_coord=x_coord,
            y_coord=y_coord
        )


def _generate_erdos_renyi_edges(substrate: SubstrateNetwork, nodes: int, 
                               edge_probability: float,
                               bandwidth_range: Tuple[int, int],
                               delay_range: Tuple[float, float],
                               cost_range: Tuple[float, float],
                               reliability_range: Tuple[float, float],
                               enable_delay_constraints: bool,
                               enable_cost_constraints: bool,
                               enable_reliability_constraints: bool) -> int:
    """Generate edges for Erdős-Rényi random graph."""
    edges_created = 0
    
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_probability:
                # Always generate bandwidth
                bandwidth = random.randint(*bandwidth_range)
                
                # Generate optional parameters based on constraints
                delay = random.uniform(*delay_range) if enable_delay_constraints else 0.0
                cost = random.uniform(*cost_range) if enable_cost_constraints else 1.0
                reliability = random.uniform(*reliability_range) if enable_reliability_constraints else 1.0
                
                substrate.add_link(i, j, bandwidth, delay, cost, reliability)
                edges_created += 1
    
    logger.debug(f"Created {edges_created} edges for Erdős-Rényi topology")
    return edges_created


def _generate_barabasi_albert_edges(substrate: SubstrateNetwork, nodes: int,
                                   attachment_count: int,
                                   bandwidth_range: Tuple[int, int],
                                   delay_range: Tuple[float, float],
                                   cost_range: Tuple[float, float],
                                   reliability_range: Tuple[float, float],
                                   enable_delay_constraints: bool,
                                   enable_cost_constraints: bool,
                                   enable_reliability_constraints: bool) -> int:
    """Generate edges for Barabási-Albert preferential attachment."""
    if attachment_count >= nodes:
        attachment_count = nodes - 1
        logger.warning(f"Attachment count reduced to {attachment_count} (nodes - 1)")
    
    edges_created = 0
    
    # Start with a complete graph of attachment_count nodes
    for i in range(attachment_count):
        for j in range(i + 1, attachment_count):
            bandwidth = random.randint(*bandwidth_range)
            delay = random.uniform(*delay_range) if enable_delay_constraints else 0.0
            cost = random.uniform(*cost_range) if enable_cost_constraints else 1.0
            reliability = random.uniform(*reliability_range) if enable_reliability_constraints else 1.0
            
            substrate.add_link(i, j, bandwidth, delay, cost, reliability)
            edges_created += 1
    
    # Add remaining nodes with preferential attachment
    degrees = [attachment_count - 1] * attachment_count
    
    for new_node in range(attachment_count, nodes):
        # Select nodes to connect to based on degree (preferential attachment)
        total_degree = sum(degrees)
        if total_degree == 0:
            # Fallback: connect to random nodes
            targets = set(random.sample(range(new_node), min(attachment_count, new_node)))
        else:
            targets = set()
            attempts = 0
            max_attempts = attachment_count * 10  # Prevent infinite loops
            
            while len(targets) < attachment_count and attempts < max_attempts:
                rand_val = random.random() * total_degree
                cumulative = 0
                
                for node_idx, degree in enumerate(degrees):
                    cumulative += degree
                    if rand_val <= cumulative:
                        targets.add(node_idx)
                        break
                attempts += 1
        
        # Create connections
        for target in targets:
            bandwidth = random.randint(*bandwidth_range)
            delay = random.uniform(*delay_range) if enable_delay_constraints else 0.0
            cost = random.uniform(*cost_range) if enable_cost_constraints else 1.0
            reliability = random.uniform(*reliability_range) if enable_reliability_constraints else 1.0
            
            substrate.add_link(new_node, target, bandwidth, delay, cost, reliability)
            edges_created += 1
            degrees[target] += 1
        
        degrees.append(len(targets))
    
    logger.debug(f"Created {edges_created} edges for Barabási-Albert topology")
    return edges_created


def _generate_grid_edges(substrate: SubstrateNetwork, nodes: int,
                        grid_size: int,
                        bandwidth_range: Tuple[int, int],
                        delay_range: Tuple[float, float],
                        cost_range: Tuple[float, float],
                        reliability_range: Tuple[float, float],
                        enable_delay_constraints: bool,
                        enable_cost_constraints: bool,
                        enable_reliability_constraints: bool) -> int:
    """Generate edges for grid topology."""
    edges_created = 0
    
    for i in range(nodes):
        row = i // grid_size
        col = i % grid_size
        
        # Connect to right neighbor
        if col < grid_size - 1 and (i + 1) < nodes:
            bandwidth = random.randint(*bandwidth_range)
            delay = random.uniform(*delay_range) if enable_delay_constraints else 0.0
            cost = random.uniform(*cost_range) if enable_cost_constraints else 1.0
            reliability = random.uniform(*reliability_range) if enable_reliability_constraints else 1.0
            
            substrate.add_link(i, i + 1, bandwidth, delay, cost, reliability)
            edges_created += 1
        
        # Connect to bottom neighbor
        if row < grid_size - 1 and (i + grid_size) < nodes:
            bandwidth = random.randint(*bandwidth_range)
            delay = random.uniform(*delay_range) if enable_delay_constraints else 0.0
            cost = random.uniform(*cost_range) if enable_cost_constraints else 1.0
            reliability = random.uniform(*reliability_range) if enable_reliability_constraints else 1.0
            
            substrate.add_link(i, i + grid_size, bandwidth, delay, cost, reliability)
            edges_created += 1
    
    logger.debug(f"Created {edges_created} edges for grid topology")
    return edges_created


def generate_realistic_substrate_network(nodes: int,
                                       geographic_area: str = "metro",
                                       enable_memory_constraints: bool = False,
                                       enable_delay_constraints: bool = True,
                                       enable_cost_constraints: bool = False,
                                       enable_reliability_constraints: bool = False,
                                       **kwargs) -> SubstrateNetwork:
    """
    Generate a realistic substrate network based on real-world patterns.
    
    Args:
        nodes: Number of nodes
        geographic_area: Type of area ("metro", "regional", "national")
        enable_memory_constraints: Whether to enable memory constraints
        enable_delay_constraints: Whether to enable delay constraints
        enable_cost_constraints: Whether to enable cost constraints
        enable_reliability_constraints: Whether to enable reliability constraints
        **kwargs: Additional parameters
        
    Returns:
        SubstrateNetwork with realistic topology and resources
        
    Raises:
        ValueError: If geographic_area is not supported
        
    Example:
        >>> # Realistic metro network with delay constraints
        >>> substrate = generate_realistic_substrate_network(30, "metro", 
        ...     enable_delay_constraints=True)
    """
    if geographic_area not in VALID_GEOGRAPHIC_AREAS:
        raise ValueError(f"Unsupported geographic area: {geographic_area}. "
                        f"Valid options: {', '.join(VALID_GEOGRAPHIC_AREAS)}")
    
    logger.info(f"Generating realistic substrate network: {nodes} nodes, area={geographic_area}")
    
    # Adjust parameters based on geographic area
    if geographic_area == "metro":
        # Metropolitan area: high connectivity, low latency
        edge_probability = 0.3
        bandwidth_range = (100, 1000)
        cpu_range = (100, 500)
        memory_range = (200, 800)
        coordinate_range = (0.0, 50.0)  # Smaller area
        delay_range = (0.5, 2.0)  # Low delay
        cost_range = (1.0, 2.0)   # Low cost
        reliability_range = (0.95, 1.0)  # High reliability
    elif geographic_area == "regional":
        # Regional network: moderate connectivity
        edge_probability = 0.2
        bandwidth_range = (50, 500)
        cpu_range = (50, 300)
        memory_range = (100, 600)
        coordinate_range = (0.0, 200.0)
        delay_range = (1.0, 5.0)  # Medium delay
        cost_range = (1.0, 3.0)   # Medium cost
        reliability_range = (0.90, 0.98)  # Good reliability
    elif geographic_area == "national":
        # National network: lower connectivity, higher resources at hubs
        edge_probability = 0.1
        bandwidth_range = (20, 200)
        cpu_range = (200, 1000)  # Larger data centers
        memory_range = (500, 2000)
        coordinate_range = (0.0, 1000.0)  # Large area
        delay_range = (2.0, 15.0)  # Higher delay
        cost_range = (2.0, 10.0)   # Higher cost
        reliability_range = (0.85, 0.95)  # Variable reliability
    
    # Use scale-free topology for realistic networks
    substrate = generate_substrate_network(
        nodes=nodes,
        topology="barabasi_albert",
        attachment_count=max(2, int(nodes * edge_probability / 2)),
        enable_memory_constraints=enable_memory_constraints,
        enable_delay_constraints=enable_delay_constraints,
        enable_cost_constraints=enable_cost_constraints,
        enable_reliability_constraints=enable_reliability_constraints,
        bandwidth_range=bandwidth_range,
        cpu_range=cpu_range,
        memory_range=memory_range,
        delay_range=delay_range,
        cost_range=cost_range,
        reliability_range=reliability_range,
        coordinate_range=coordinate_range,
        **kwargs
    )
    
    return substrate


def validate_substrate_network(substrate: SubstrateNetwork) -> Dict[str, bool]:
    """
    Validate a generated substrate network for consistency and realism.
    
    Args:
        substrate: SubstrateNetwork instance to validate
        
    Returns:
        Dictionary of validation results
        
    Example:
        >>> validation = validate_substrate_network(substrate)
        >>> if all(validation.values()):
        ...     print("Substrate network validation passed")
    """
    validation_results = {
        'has_nodes': False,
        'has_links': False,
        'connected': False,
        'realistic_resources': False,
        'consistent_data': False,
        'constraint_compliance': False
    }
    
    try:
        # Check basic structure
        if hasattr(substrate, 'graph') and len(substrate.graph.nodes) > 0:
            validation_results['has_nodes'] = True
        
        if hasattr(substrate, 'graph') and len(substrate.graph.edges) > 0:
            validation_results['has_links'] = True
        
        # Check connectivity
        if validation_results['has_nodes'] and validation_results['has_links']:
            import networkx as nx
            validation_results['connected'] = nx.is_connected(substrate.graph)
        
        # Check resource realism
        validation_results['realistic_resources'] = _validate_substrate_resources(substrate)
        
        # Check data consistency
        validation_results['consistent_data'] = _validate_substrate_consistency(substrate)
        
        # Check constraint compliance
        validation_results['constraint_compliance'] = _validate_constraint_compliance(substrate)
        
    except Exception as e:
        logger.error(f"Substrate validation error: {e}")
        validation_results['error'] = str(e)
    
    logger.debug(f"Substrate validation results: {validation_results}")
    return validation_results


def _validate_substrate_resources(substrate: SubstrateNetwork) -> bool:
    """Validate substrate network resources are realistic."""
    try:
        constraint_config = substrate.get_constraint_configuration()
        
        # Check node resources
        for node_id in substrate.graph.nodes:
            node_resources = substrate.get_node_resources(node_id)
            if not node_resources:
                return False
            
            # CPU should always be positive
            if node_resources.cpu_capacity <= 0:
                return False
            
            # Check resource usage doesn't exceed capacity
            if node_resources.cpu_used > node_resources.cpu_capacity:
                return False
            
            # Check memory constraints if enabled
            if constraint_config['memory_constraints']:
                if node_resources.memory_capacity <= 0:
                    return False
                if node_resources.memory_used > node_resources.memory_capacity:
                    return False
            else:
                # Memory should be 0 if constraints disabled
                if node_resources.memory_capacity != 0.0:
                    return False
        
        # Check link resources
        for src, dst in substrate.graph.edges:
            link_resources = substrate.get_link_resources(src, dst)
            if not link_resources:
                return False
            
            # Bandwidth should always be positive
            if link_resources.bandwidth_capacity <= 0:
                return False
            
            # Check resource usage doesn't exceed capacity
            if link_resources.bandwidth_used > link_resources.bandwidth_capacity:
                return False
            
            # Check optional constraints
            if constraint_config['delay_constraints']:
                if link_resources.delay < 0:
                    return False
            else:
                if link_resources.delay != 0.0:
                    return False
            
            if constraint_config['reliability_constraints']:
                if not (0 <= link_resources.reliability <= 1):
                    return False
            else:
                if link_resources.reliability != 1.0:
                    return False
        
        return True
    except Exception:
        return False


def _validate_substrate_consistency(substrate: SubstrateNetwork) -> bool:
    """Validate substrate network data consistency."""
    try:
        # Check that all links reference existing nodes
        for src, dst in substrate.graph.edges:
            if src not in substrate.graph.nodes or dst not in substrate.graph.nodes:
                return False
        
        # Check for self-loops (usually not desired in substrate networks)
        for src, dst in substrate.graph.edges:
            if src == dst:
                logger.warning(f"Self-loop detected in substrate: {src} -> {dst}")
        
        return True
    except Exception:
        return False


def _validate_constraint_compliance(substrate: SubstrateNetwork) -> bool:
    """Validate that substrate network complies with its constraint configuration."""
    try:
        constraint_config = substrate.get_constraint_configuration()
        
        # Check that disabled constraints actually have default values
        for node_id in substrate.graph.nodes:
            node_resources = substrate.get_node_resources(node_id)
            
            if not constraint_config['memory_constraints']:
                if node_resources.memory_capacity != 0.0:
                    return False
        
        for src, dst in substrate.graph.edges:
            link_resources = substrate.get_link_resources(src, dst)
            
            if not constraint_config['delay_constraints']:
                if link_resources.delay != 0.0:
                    return False
            
            if not constraint_config['cost_constraints']:
                if link_resources.cost != 1.0:
                    return False
            
            if not constraint_config['reliability_constraints']:
                if link_resources.reliability != 1.0:
                    return False
        
        return True
    except Exception:
        return False


def _validate_resource_ranges(*ranges) -> None:
    """Validate that resource ranges are valid."""
    for i, range_tuple in enumerate(ranges):
        if len(range_tuple) != 2:
            raise ResourceError(f"Range {i} must have exactly 2 values")
        
        min_val, max_val = range_tuple
        if min_val > max_val:
            raise ResourceError(f"Range {i} has min > max: ({min_val}, {max_val})")
        
        if min_val < 0:
            raise ResourceError(f"Range {i} has negative minimum: {min_val}")


def create_predefined_scenarios() -> Dict[str, NetworkGenerationConfig]:
    """
    Create predefined substrate network generation scenarios for common experiments.
    
    Returns:
        Dictionary of scenario name to configuration
        
    Example:
        >>> scenarios = create_predefined_scenarios()
        >>> yu2008_config = scenarios['yu2008_baseline']
        >>> substrate = generate_substrate_from_config(yu2008_config)
    """
    scenarios = {
        'yu2008_baseline': NetworkGenerationConfig(
            substrate_nodes=50,
            substrate_topology="erdos_renyi",
            substrate_edge_probability=0.15,
            enable_memory_constraints=False,  # Yu 2008 style
            enable_delay_constraints=False,
            enable_cost_constraints=False,
            enable_reliability_constraints=False,
            cpu_range=(50, 200),
            bandwidth_range=(50, 200),
            vnr_nodes_range=(2, 8),
            arrival_rate=10.0
        ),
        
        'small_network': NetworkGenerationConfig(
            substrate_nodes=20,
            substrate_topology="erdos_renyi",
            substrate_edge_probability=0.3,
            enable_memory_constraints=False,
            cpu_range=(30, 100),
            bandwidth_range=(30, 100),
            vnr_nodes_range=(2, 5),
            arrival_rate=5.0
        ),
        
        'medium_network': NetworkGenerationConfig(
            substrate_nodes=50,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=3,
            enable_memory_constraints=True,  # Include memory
            cpu_range=(50, 300),
            memory_range=(100, 400),
            bandwidth_range=(50, 300),
            vnr_nodes_range=(3, 8),
            arrival_rate=10.0
        ),
        
        'large_network': NetworkGenerationConfig(
            substrate_nodes=100,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=5,
            enable_memory_constraints=True,
            enable_delay_constraints=True,  # Full constraints
            enable_cost_constraints=True,
            enable_reliability_constraints=True,
            cpu_range=(100, 500),
            memory_range=(200, 800),
            bandwidth_range=(100, 500),
            delay_range=(1.0, 10.0),
            cost_range=(1.0, 5.0),
            reliability_range=(0.9, 1.0),
            vnr_nodes_range=(5, 15),
            arrival_rate=20.0
        ),
        
        'dense_topology': NetworkGenerationConfig(
            substrate_nodes=30,
            substrate_topology="erdos_renyi",
            substrate_edge_probability=0.4,  # High connectivity
            enable_memory_constraints=False,
            cpu_range=(50, 150),
            bandwidth_range=(100, 300),
            vnr_nodes_range=(2, 6),
            arrival_rate=15.0
        ),
        
        'sparse_topology': NetworkGenerationConfig(
            substrate_nodes=80,
            substrate_topology="erdos_renyi",
            substrate_edge_probability=0.05,  # Low connectivity
            enable_memory_constraints=False,
            cpu_range=(100, 400),
            bandwidth_range=(50, 200),
            vnr_nodes_range=(3, 10),
            arrival_rate=5.0
        ),
        
        'metro_realistic': NetworkGenerationConfig(
            substrate_nodes=25,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=4,
            enable_memory_constraints=True,
            enable_delay_constraints=True,
            cpu_range=(100, 500),
            memory_range=(200, 800),
            bandwidth_range=(100, 1000),
            delay_range=(0.5, 2.0),
            coordinate_range=(0.0, 50.0),
            vnr_nodes_range=(2, 6),
            arrival_rate=12.0
        ),
        
        'national_realistic': NetworkGenerationConfig(
            substrate_nodes=60,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=2,
            enable_memory_constraints=True,
            enable_delay_constraints=True,
            enable_cost_constraints=True,
            cpu_range=(200, 1000),
            memory_range=(500, 2000),
            bandwidth_range=(50, 300),
            delay_range=(5.0, 20.0),
            cost_range=(2.0, 10.0),
            coordinate_range=(0.0, 1000.0),
            vnr_nodes_range=(3, 12),
            arrival_rate=8.0
        ),
        
        'grid_topology': NetworkGenerationConfig(
            substrate_nodes=36,  # 6x6 grid
            substrate_topology="grid",
            enable_memory_constraints=False,
            cpu_range=(50, 200),
            bandwidth_range=(50, 200),
            vnr_nodes_range=(2, 8),
            arrival_rate=10.0
        ),
        
        'high_capacity': NetworkGenerationConfig(
            substrate_nodes=40,
            substrate_topology="barabasi_albert",
            substrate_attachment_count=4,
            enable_memory_constraints=True,
            cpu_range=(200, 800),
            memory_range=(400, 1200),
            bandwidth_range=(200, 800),
            vnr_nodes_range=(3, 10),
            arrival_rate=15.0
        )
    }
    
    logger.info(f"Created {len(scenarios)} predefined substrate scenarios")
    return scenarios
