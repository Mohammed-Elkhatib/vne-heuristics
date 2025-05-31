"""
Substrate Network Model for Virtual Network Embedding.

This module provides the SubstrateNetwork class for representing and managing
physical substrate networks with resource allocation capabilities.
"""

import logging
import csv
import threading
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import networkx as nx
from pathlib import Path


logger = logging.getLogger(__name__)


class SubstrateNetworkError(Exception):
    """Base exception for substrate network operations."""
    pass


class ResourceAllocationError(SubstrateNetworkError):
    """Exception raised when resource allocation fails."""
    pass


class FileFormatError(SubstrateNetworkError):
    """Exception raised when file format is invalid."""
    pass


@dataclass
class NodeResources:
    """Represents node resource attributes."""
    cpu_capacity: float
    memory_capacity: float = 0.0  # Always present, but may be ignored network-wide
    cpu_used: float = 0.0
    memory_used: float = 0.0
    x_coord: float = 0.0
    y_coord: float = 0.0
    node_type: str = "default"

    @property
    def cpu_available(self) -> float:
        """Get available CPU resources."""
        return max(0.0, self.cpu_capacity - self.cpu_used)

    @property
    def memory_available(self) -> float:
        """Get available memory resources."""
        return max(0.0, self.memory_capacity - self.memory_used)

    def __str__(self) -> str:
        """String representation of node resources."""
        return (f"NodeResources(CPU: {self.cpu_available:.1f}/{self.cpu_capacity:.1f}, "
               f"Memory: {self.memory_available:.1f}/{self.memory_capacity:.1f})")


@dataclass
class LinkResources:
    """Represents link resource attributes."""
    bandwidth_capacity: float
    bandwidth_used: float = 0.0
    delay: float = 0.0           # Always present, but may be ignored network-wide
    cost: float = 1.0            # Always present, but may be ignored network-wide
    reliability: float = 1.0     # Always present, but may be ignored network-wide

    @property
    def bandwidth_available(self) -> float:
        """Get available bandwidth resources."""
        return max(0.0, self.bandwidth_capacity - self.bandwidth_used)

    @property
    def utilization(self) -> float:
        """Get link utilization ratio."""
        if self.bandwidth_capacity == 0:
            return 0.0
        return self.bandwidth_used / self.bandwidth_capacity

    def __str__(self) -> str:
        """String representation of link resources."""
        return (f"LinkResources(Bandwidth: {self.bandwidth_available:.1f}/{self.bandwidth_capacity:.1f}, "
               f"Delay: {self.delay:.2f}, Cost: {self.cost:.2f}, Reliability: {self.reliability:.3f})")


class SubstrateNetwork:
    """
    Represents a physical substrate network for Virtual Network Embedding.

    This class manages the substrate network topology, node and link resources,
    and provides methods for resource allocation/deallocation and network I/O.

    Constraints can be enabled/disabled at the network level:
    - CPU and Bandwidth constraints are always enforced (primary constraints)
    - Memory, delay, cost, and reliability constraints are optional (secondary constraints)

    Attributes:
        graph: NetworkX graph representing the network topology
        _lock: Threading lock for resource allocation safety
        enable_memory_constraints: Whether to enforce memory constraints
        enable_delay_constraints: Whether to enforce delay constraints
        enable_cost_constraints: Whether to enforce cost constraints
        enable_reliability_constraints: Whether to enforce reliability constraints

    Example:
        >>> # Yu 2008 style (CPU + Bandwidth only)
        >>> substrate = SubstrateNetwork()
        >>> # Full constraint network
        >>> substrate2 = SubstrateNetwork(enable_memory_constraints=True, enable_delay_constraints=True)
        >>> substrate.add_node(1, cpu_capacity=100, memory_capacity=200)
        >>> substrate.add_link(1, 2, bandwidth_capacity=1000, delay=5.0)
        >>> success = substrate.allocate_node_resources(1, cpu=50, memory=100)
    """

    def __init__(self, directed: bool = False,
                 enable_memory_constraints: bool = False,
                 enable_delay_constraints: bool = False,
                 enable_cost_constraints: bool = False,
                 enable_reliability_constraints: bool = False):
        """
        Initialize substrate network.

        Args:
            directed: Whether to use directed graph (default: False)
            enable_memory_constraints: Whether to enforce memory constraints (default: False)
            enable_delay_constraints: Whether to enforce delay constraints (default: False)
            enable_cost_constraints: Whether to enforce cost constraints (default: False)
            enable_reliability_constraints: Whether to enforce reliability constraints (default: False)
        """
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._lock = threading.Lock()

        # Constraint configuration
        self.enable_memory_constraints = enable_memory_constraints
        self.enable_delay_constraints = enable_delay_constraints
        self.enable_cost_constraints = enable_cost_constraints
        self.enable_reliability_constraints = enable_reliability_constraints

        constraint_summary = []
        if enable_memory_constraints:
            constraint_summary.append("memory")
        if enable_delay_constraints:
            constraint_summary.append("delay")
        if enable_cost_constraints:
            constraint_summary.append("cost")
        if enable_reliability_constraints:
            constraint_summary.append("reliability")

        enabled_constraints = ", ".join(constraint_summary) if constraint_summary else "none"

        logger.info(f"Initialized {'directed' if directed else 'undirected'} substrate network "
                   f"with constraints: CPU+bandwidth (always), {enabled_constraints} (optional)")

    def add_node(self, node_id: int, cpu_capacity: float, memory_capacity: float = 0.0,
                 x_coord: float = 0.0, y_coord: float = 0.0,
                 node_type: str = "default") -> None:
        """
        Add a node to the substrate network.

        Args:
            node_id: Unique identifier for the node
            cpu_capacity: CPU capacity of the node
            memory_capacity: Memory capacity of the node (ignored if memory constraints disabled)
            x_coord: X coordinate for visualization
            y_coord: Y coordinate for visualization
            node_type: Type of the node (e.g., "server", "switch")

        Raises:
            ValueError: If capacity values are invalid
        """
        if cpu_capacity < 0:
            raise ValueError("CPU capacity must be non-negative")
        if memory_capacity < 0:
            raise ValueError("Memory capacity must be non-negative")

        resources = NodeResources(
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
            x_coord=x_coord,
            y_coord=y_coord,
            node_type=node_type
        )

        self.graph.add_node(node_id, resources=resources)

        if self.enable_memory_constraints:
            logger.debug(f"Added node {node_id} with CPU={cpu_capacity}, Memory={memory_capacity}")
        else:
            logger.debug(f"Added node {node_id} with CPU={cpu_capacity}")

    def add_link(self, src: int, dst: int, bandwidth_capacity: float,
                 delay: float = 0.0, cost: float = 1.0, reliability: float = 1.0) -> None:
        """
        Add a link to the substrate network.

        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth_capacity: Bandwidth capacity of the link
            delay: Link delay (ignored if delay constraints disabled)
            cost: Link cost (ignored if cost constraints disabled)
            reliability: Link reliability (ignored if reliability constraints disabled)

        Raises:
            ValueError: If capacity or parameter values are invalid
        """
        if bandwidth_capacity < 0:
            raise ValueError("Bandwidth capacity must be non-negative")
        if delay < 0:
            raise ValueError("Delay must be non-negative")
        if not (0 <= reliability <= 1):
            raise ValueError("Reliability must be between 0 and 1")

        resources = LinkResources(
            bandwidth_capacity=bandwidth_capacity,
            delay=delay,
            cost=cost,
            reliability=reliability
        )

        self.graph.add_edge(src, dst, resources=resources)

        used_params = [f"bandwidth={bandwidth_capacity}"]
        ignored_params = []

        if self.enable_delay_constraints:
            used_params.append(f"delay={delay}")
        else:
            ignored_params.append(f"delay={delay}")

        if self.enable_cost_constraints:
            used_params.append(f"cost={cost}")
        else:
            ignored_params.append(f"cost={cost}")

        if self.enable_reliability_constraints:
            used_params.append(f"reliability={reliability}")
        else:
            ignored_params.append(f"reliability={reliability}")

        log_message = f"Added link ({src}, {dst}) with {', '.join(used_params)}"
        if ignored_params:
            log_message += f" (ignored: {', '.join(ignored_params)})"

        logger.debug(log_message)

    def allocate_node_resources(self, node_id: int, cpu: float, memory: float = 0.0) -> bool:
        """
        Allocate resources from a substrate node.

        Args:
            node_id: ID of the node to allocate resources from
            cpu: Amount of CPU to allocate
            memory: Amount of memory to allocate (ignored if memory constraints disabled)

        Returns:
            True if allocation successful, False otherwise

        Raises:
            ResourceAllocationError: If node doesn't exist or invalid parameters
        """
        with self._lock:
            if node_id not in self.graph.nodes:
                raise ResourceAllocationError(f"Node {node_id} does not exist")

            resources = self.graph.nodes[node_id]['resources']

            if cpu < 0 or memory < 0:
                raise ResourceAllocationError("Cannot allocate negative resources")

            # Check CPU constraint (always enforced)
            if resources.cpu_available < cpu:
                logger.warning(f"Insufficient CPU on node {node_id}: "
                             f"requested={cpu}, available={resources.cpu_available}")
                return False

            # Check memory constraint (only if enabled)
            if self.enable_memory_constraints and memory > 0:
                if resources.memory_available < memory:
                    logger.warning(f"Insufficient memory on node {node_id}: "
                                 f"requested={memory}, available={resources.memory_available}")
                    return False

            # Perform allocation
            resources.cpu_used += cpu
            if self.enable_memory_constraints and memory > 0:
                resources.memory_used += memory

            if self.enable_memory_constraints and memory > 0:
                logger.debug(f"Allocated resources on node {node_id}: CPU={cpu}, Memory={memory}")
            else:
                logger.debug(f"Allocated resources on node {node_id}: CPU={cpu}")

            return True

    def deallocate_node_resources(self, node_id: int, cpu: float, memory: float = 0.0) -> bool:
        """
        Deallocate resources from a substrate node.

        Args:
            node_id: ID of the node to deallocate resources from
            cpu: Amount of CPU to deallocate
            memory: Amount of memory to deallocate (ignored if memory constraints disabled)

        Returns:
            True if deallocation successful, False otherwise

        Raises:
            ResourceAllocationError: If node doesn't exist or invalid parameters
        """
        with self._lock:
            if node_id not in self.graph.nodes:
                raise ResourceAllocationError(f"Node {node_id} does not exist")

            resources = self.graph.nodes[node_id]['resources']

            if cpu < 0 or memory < 0:
                raise ResourceAllocationError("Cannot deallocate negative resources")

            # Deallocate CPU (always)
            cpu_to_free = min(cpu, resources.cpu_used)
            resources.cpu_used = max(0, resources.cpu_used - cpu_to_free)

            # Deallocate memory (only if constraints enabled)
            if self.enable_memory_constraints and memory > 0:
                memory_to_free = min(memory, resources.memory_used)
                resources.memory_used = max(0, resources.memory_used - memory_to_free)

            if self.enable_memory_constraints and memory > 0:
                logger.debug(f"Deallocated resources on node {node_id}: CPU={cpu_to_free}, Memory={memory_to_free}")
            else:
                logger.debug(f"Deallocated resources on node {node_id}: CPU={cpu_to_free}")

            return True

    def allocate_link_resources(self, src: int, dst: int, bandwidth: float) -> bool:
        """
        Allocate bandwidth from a substrate link.

        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth: Amount of bandwidth to allocate

        Returns:
            True if allocation successful, False otherwise

        Raises:
            ResourceAllocationError: If link doesn't exist or invalid parameters
        """
        with self._lock:
            if not self.graph.has_edge(src, dst):
                raise ResourceAllocationError(f"Link ({src}, {dst}) does not exist")

            resources = self.graph.edges[src, dst]['resources']

            if bandwidth < 0:
                raise ResourceAllocationError("Cannot allocate negative bandwidth")

            if resources.bandwidth_available < bandwidth:
                logger.warning(f"Insufficient bandwidth on link ({src}, {dst}): "
                             f"requested={bandwidth}, available={resources.bandwidth_available}")
                return False

            resources.bandwidth_used += bandwidth

            logger.debug(f"Allocated bandwidth on link ({src}, {dst}): {bandwidth}")
            return True

    def deallocate_link_resources(self, src: int, dst: int, bandwidth: float) -> bool:
        """
        Deallocate bandwidth from a substrate link.

        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth: Amount of bandwidth to deallocate

        Returns:
            True if deallocation successful, False otherwise

        Raises:
            ResourceAllocationError: If link doesn't exist or invalid parameters
        """
        with self._lock:
            if not self.graph.has_edge(src, dst):
                raise ResourceAllocationError(f"Link ({src}, {dst}) does not exist")

            resources = self.graph.edges[src, dst]['resources']

            if bandwidth < 0:
                raise ResourceAllocationError("Cannot deallocate negative bandwidth")

            # Ensure we don't deallocate more than allocated
            bandwidth_to_free = min(bandwidth, resources.bandwidth_used)
            resources.bandwidth_used = max(0, resources.bandwidth_used - bandwidth_to_free)

            logger.debug(f"Deallocated bandwidth on link ({src}, {dst}): {bandwidth_to_free}")
            return True

    def check_node_resources(self, node_id: int, cpu: float, memory: float = 0.0) -> Dict[str, bool]:
        """
        Check if a node has sufficient available resources.

        Args:
            node_id: ID of the node to check
            cpu: Required CPU resources
            memory: Required memory resources (ignored if memory constraints disabled)

        Returns:
            True if sufficient resources available, False otherwise

        Raises:
            ResourceAllocationError: If node doesn't exist
        """
        if node_id not in self.graph.nodes:
            raise ResourceAllocationError(f"Node {node_id} does not exist")

        resources = self.graph.nodes[node_id]['resources']

        # Check CPU (always enforced)
        cpu_ok = resources.cpu_available >= cpu

        result = {'cpu': cpu_ok}

        # Check memory (only if constraints enabled and memory requested)
        if self.enable_memory_constraints and memory > 0:
            memory_ok = resources.memory_available >= memory
            result['memory'] = memory_ok

        return result

    def check_link_resources(self, src: int, dst: int, bandwidth: float) -> bool:
        """
        Check if a link has sufficient available bandwidth.

        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth: Required bandwidth

        Returns:
            True if sufficient bandwidth available, False otherwise

        Raises:
            ResourceAllocationError: If link doesn't exist
        """
        if not self.graph.has_edge(src, dst):
            raise ResourceAllocationError(f"Link ({src}, {dst}) does not exist")

        resources = self.graph.edges[src, dst]['resources']
        return resources.bandwidth_available >= bandwidth

    def get_node_resources(self, node_id: int) -> Optional[NodeResources]:
        """
        Get resource information for a node.

        Args:
            node_id: ID of the node

        Returns:
            NodeResources object or None if node doesn't exist
        """
        if node_id not in self.graph.nodes:
            return None
        return self.graph.nodes[node_id]['resources']

    def get_link_resources(self, src: int, dst: int) -> Optional[LinkResources]:
        """
        Get resource information for a link.

        Args:
            src: Source node ID
            dst: Destination node ID

        Returns:
            LinkResources object or None if link doesn't exist
        """
        if not self.graph.has_edge(src, dst):
            return None
        return self.graph.edges[src, dst]['resources']

    def get_constraint_configuration(self) -> Dict[str, bool]:
        """
        Get the current constraint configuration.

        Returns:
            Dictionary of constraint types and their enabled status
        """
        return {
            'cpu_constraints': True,  # Always enabled
            'bandwidth_constraints': True,  # Always enabled
            'memory_constraints': self.enable_memory_constraints,
            'delay_constraints': self.enable_delay_constraints,
            'cost_constraints': self.enable_cost_constraints,
            'reliability_constraints': self.enable_reliability_constraints
        }

    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive network statistics.

        Returns:
            Dictionary containing network statistics
        """
        nodes = self.graph.nodes
        edges = self.graph.edges

        # CPU statistics (always calculated)
        total_cpu = sum(nodes[n]['resources'].cpu_capacity for n in nodes)
        used_cpu = sum(nodes[n]['resources'].cpu_used for n in nodes)

        # Memory statistics (only if constraints enabled)
        memory_stats = {}
        if self.enable_memory_constraints:
            total_memory = sum(nodes[n]['resources'].memory_capacity for n in nodes)
            used_memory = sum(nodes[n]['resources'].memory_used for n in nodes)
            memory_stats = {
                'total_memory': total_memory,
                'used_memory': used_memory,
                'available_memory': total_memory - used_memory,
                'memory_utilization': used_memory / total_memory if total_memory > 0 else 0,
            }
        else:
            memory_stats = {
                'total_memory': 0,
                'used_memory': 0,
                'available_memory': 0,
                'memory_utilization': 0,
            }

        # Bandwidth statistics (always calculated)
        total_bandwidth = sum(edges[e]['resources'].bandwidth_capacity for e in edges)
        used_bandwidth = sum(edges[e]['resources'].bandwidth_used for e in edges)

        # FIX: Handle empty graph connectivity check
        is_connected = False
        if len(nodes) > 1:  # Only check connectivity if we have multiple nodes
            if not self.graph.is_directed():
                is_connected = nx.is_connected(self.graph)
            else:
                is_connected = nx.is_weakly_connected(self.graph)
        elif len(nodes) == 1:  # Single node is trivially connected
            is_connected = True
        # Empty graph (len(nodes) == 0) remains False

        stats = {
            'node_count': len(nodes),
            'link_count': len(edges),
            'is_connected': is_connected,  # Fixed connectivity check
            'average_degree': sum(dict(self.graph.degree()).values()) / len(nodes) if nodes else 0,
            'total_cpu': total_cpu,
            'used_cpu': used_cpu,
            'available_cpu': total_cpu - used_cpu,
            'cpu_utilization': used_cpu / total_cpu if total_cpu > 0 else 0,
            'total_bandwidth': total_bandwidth,
            'used_bandwidth': used_bandwidth,
            'available_bandwidth': total_bandwidth - used_bandwidth,
            'bandwidth_utilization': used_bandwidth / total_bandwidth if total_bandwidth > 0 else 0,
            'constraint_configuration': self.get_constraint_configuration()
        }

        # Add memory statistics
        stats.update(memory_stats)

        return stats

    def validate_network(self) -> List[str]:
        """
        Validate network consistency and return list of issues.

        Returns:
            List of validation error messages (empty if no issues)
        """
        issues = []

        # Check for nodes with invalid resources
        for node_id in self.graph.nodes:
            resources = self.graph.nodes[node_id]['resources']

            if resources.cpu_used > resources.cpu_capacity:
                issues.append(f"Node {node_id}: CPU overallocation ({resources.cpu_used}/{resources.cpu_capacity})")

            if self.enable_memory_constraints and resources.memory_used > resources.memory_capacity:
                issues.append(f"Node {node_id}: Memory overallocation ({resources.memory_used}/{resources.memory_capacity})")

        # Check for links with invalid resources
        for src, dst in self.graph.edges:
            resources = self.graph.edges[src, dst]['resources']
            if resources.bandwidth_used > resources.bandwidth_capacity:
                issues.append(f"Link ({src}, {dst}): Bandwidth overallocation ({resources.bandwidth_used}/{resources.bandwidth_capacity})")

        # Check for isolated nodes (if network should be connected)
        if len(self.graph.nodes) > 1:
            if not nx.is_connected(self.graph) and not self.graph.is_directed():
                issues.append("Network is not connected")
            elif self.graph.is_directed() and not nx.is_weakly_connected(self.graph):
                issues.append("Directed network is not weakly connected")

        return issues

    def load_from_csv(self, nodes_file: str, links_file: str) -> None:
        """
        Load substrate network from CSV files.

        Args:
            nodes_file: Path to nodes CSV file
            links_file: Path to links CSV file

        Expected CSV formats:
            nodes.csv: node_id,cpu_capacity,memory_capacity,x_coord,y_coord,node_type
            links.csv: src_node,dst_node,bandwidth_capacity,delay,cost,reliability

        Raises:
            FileNotFoundError: If files don't exist
            FileFormatError: If file format is invalid
        """
        try:
            # Load nodes
            with open(nodes_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.add_node(
                        node_id=int(row['node_id']),
                        cpu_capacity=float(row['cpu_capacity']),
                        memory_capacity=float(row.get('memory_capacity', 0.0)),
                        x_coord=float(row.get('x_coord', 0.0)),
                        y_coord=float(row.get('y_coord', 0.0)),
                        node_type=row.get('node_type', 'default')
                    )

            # Load links
            with open(links_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.add_link(
                        src=int(row['src_node']),
                        dst=int(row['dst_node']),
                        bandwidth_capacity=float(row['bandwidth_capacity']),
                        delay=float(row.get('delay', 0.0)),
                        cost=float(row.get('cost', 1.0)),
                        reliability=float(row.get('reliability', 1.0))
                    )

            logger.info(f"Loaded substrate network from {nodes_file} and {links_file}")

        except (KeyError, ValueError) as e:
            raise FileFormatError(f"Invalid file format: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {e}")

    def save_to_csv(self, nodes_file: str, links_file: str) -> None:
        """
        Save substrate network to CSV files.

        Args:
            nodes_file: Path for nodes CSV file
            links_file: Path for links CSV file
        """
        # Ensure directories exist
        Path(nodes_file).parent.mkdir(parents=True, exist_ok=True)
        Path(links_file).parent.mkdir(parents=True, exist_ok=True)

        # Save nodes
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'cpu_capacity', 'memory_capacity',
                           'cpu_used', 'memory_used', 'x_coord', 'y_coord', 'node_type'])

            for node_id in self.graph.nodes:
                resources = self.graph.nodes[node_id]['resources']
                writer.writerow([
                    node_id,
                    resources.cpu_capacity,
                    resources.memory_capacity,
                    resources.cpu_used,
                    resources.memory_used,
                    resources.x_coord,
                    resources.y_coord,
                    resources.node_type
                ])

        # Save links
        with open(links_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['src_node', 'dst_node', 'bandwidth_capacity',
                           'bandwidth_used', 'delay', 'cost', 'reliability'])

            for src, dst in self.graph.edges:
                resources = self.graph.edges[src, dst]['resources']
                writer.writerow([
                    src,
                    dst,
                    resources.bandwidth_capacity,
                    resources.bandwidth_used,
                    resources.delay,
                    resources.cost,
                    resources.reliability
                ])

        logger.info(f"Saved substrate network to {nodes_file} and {links_file}")

    def reset_allocations(self) -> None:
        """Reset all resource allocations to zero."""
        with self._lock:
            for node_id in self.graph.nodes:
                resources = self.graph.nodes[node_id]['resources']
                resources.cpu_used = 0.0
                resources.memory_used = 0.0

            for src, dst in self.graph.edges:
                resources = self.graph.edges[src, dst]['resources']
                resources.bandwidth_used = 0.0

        logger.info("Reset all resource allocations")

    def __len__(self) -> int:
        """Return number of nodes in the network."""
        return len(self.graph.nodes)

    def __contains__(self, node_id: int) -> bool:
        """Check if node exists in the network."""
        return node_id in self.graph.nodes

    def __str__(self) -> str:
        """String representation of the substrate network."""
        stats = self.get_network_statistics()
        config = self.get_constraint_configuration()
        enabled_constraints = [k.replace('_constraints', '') for k, v in config.items() if v]
        return (f"SubstrateNetwork({stats['node_count']} nodes, {stats['link_count']} links, "
               f"constraints: {'+'.join(enabled_constraints)})")
