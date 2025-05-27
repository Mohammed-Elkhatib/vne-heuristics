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
    memory_capacity: float
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


@dataclass
class LinkResources:
    """Represents link resource attributes."""
    bandwidth_capacity: float
    delay: float
    cost: float = 1.0
    bandwidth_used: float = 0.0
    reliability: float = 1.0
    
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


class SubstrateNetwork:
    """
    Represents a physical substrate network for Virtual Network Embedding.
    
    This class manages the substrate network topology, node and link resources,
    and provides methods for resource allocation/deallocation and network I/O.
    
    Attributes:
        graph: NetworkX graph representing the network topology
        _lock: Threading lock for resource allocation safety
        
    Example:
        >>> substrate = SubstrateNetwork()
        >>> substrate.add_node(1, cpu_capacity=100, memory_capacity=200)
        >>> substrate.add_link(1, 2, bandwidth_capacity=1000, delay=5.0)
        >>> success = substrate.allocate_node_resources(1, cpu=50, memory=100)
        >>> substrate.save_to_csv("nodes.csv", "links.csv")
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize substrate network.
        
        Args:
            directed: Whether to use directed graph (default: False)
        """
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._lock = threading.Lock()
        logger.info(f"Initialized {'directed' if directed else 'undirected'} substrate network")
    
    def add_node(self, node_id: int, cpu_capacity: float, memory_capacity: float,
                 x_coord: float = 0.0, y_coord: float = 0.0, 
                 node_type: str = "default") -> None:
        """
        Add a node to the substrate network.
        
        Args:
            node_id: Unique identifier for the node
            cpu_capacity: CPU capacity of the node
            memory_capacity: Memory capacity of the node
            x_coord: X coordinate for visualization
            y_coord: Y coordinate for visualization
            node_type: Type of the node (e.g., "server", "switch")
            
        Raises:
            ValueError: If capacity values are negative
        """
        if cpu_capacity < 0 or memory_capacity < 0:
            raise ValueError("Capacity values must be non-negative")
        
        resources = NodeResources(
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
            x_coord=x_coord,
            y_coord=y_coord,
            node_type=node_type
        )
        
        self.graph.add_node(node_id, resources=resources)
        logger.debug(f"Added node {node_id} with CPU={cpu_capacity}, Memory={memory_capacity}")
    
    def add_link(self, src: int, dst: int, bandwidth_capacity: float, 
                 delay: float, cost: float = 1.0, reliability: float = 1.0) -> None:
        """
        Add a link to the substrate network.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth_capacity: Bandwidth capacity of the link
            delay: Link delay
            cost: Link cost (default: 1.0)
            reliability: Link reliability (default: 1.0)
            
        Raises:
            ValueError: If capacity or delay values are invalid
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
        logger.debug(f"Added link ({src}, {dst}) with bandwidth={bandwidth_capacity}, delay={delay}")
    
    def allocate_node_resources(self, node_id: int, cpu: float, memory: float) -> bool:
        """
        Allocate resources from a substrate node.
        
        Args:
            node_id: ID of the node to allocate resources from
            cpu: Amount of CPU to allocate
            memory: Amount of memory to allocate
            
        Returns:
            True if allocation successful, False otherwise
            
        Raises:
            ResourceAllocationError: If node doesn't exist or insufficient resources
        """
        with self._lock:
            if node_id not in self.graph.nodes:
                raise ResourceAllocationError(f"Node {node_id} does not exist")
            
            resources = self.graph.nodes[node_id]['resources']
            
            if cpu < 0 or memory < 0:
                raise ResourceAllocationError("Cannot allocate negative resources")
            
            if resources.cpu_available < cpu or resources.memory_available < memory:
                logger.warning(f"Insufficient resources on node {node_id}: "
                             f"requested CPU={cpu} (available={resources.cpu_available}), "
                             f"memory={memory} (available={resources.memory_available})")
                return False
            
            resources.cpu_used += cpu
            resources.memory_used += memory
            
            logger.debug(f"Allocated resources on node {node_id}: CPU={cpu}, Memory={memory}")
            return True
    
    def deallocate_node_resources(self, node_id: int, cpu: float, memory: float) -> bool:
        """
        Deallocate resources from a substrate node.
        
        Args:
            node_id: ID of the node to deallocate resources from
            cpu: Amount of CPU to deallocate
            memory: Amount of memory to deallocate
            
        Returns:
            True if deallocation successful, False otherwise
            
        Raises:
            ResourceAllocationError: If node doesn't exist
        """
        with self._lock:
            if node_id not in self.graph.nodes:
                raise ResourceAllocationError(f"Node {node_id} does not exist")
            
            resources = self.graph.nodes[node_id]['resources']
            
            if cpu < 0 or memory < 0:
                raise ResourceAllocationError("Cannot deallocate negative resources")
            
            # Ensure we don't deallocate more than allocated
            cpu_to_free = min(cpu, resources.cpu_used)
            memory_to_free = min(memory, resources.memory_used)
            
            resources.cpu_used = max(0, resources.cpu_used - cpu_to_free)
            resources.memory_used = max(0, resources.memory_used - memory_to_free)
            
            logger.debug(f"Deallocated resources on node {node_id}: CPU={cpu_to_free}, Memory={memory_to_free}")
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
            ResourceAllocationError: If link doesn't exist or insufficient bandwidth
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
            ResourceAllocationError: If link doesn't exist
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
    
    def check_node_resources(self, node_id: int, cpu: float, memory: float) -> bool:
        """
        Check if a node has sufficient available resources.
        
        Args:
            node_id: ID of the node to check
            cpu: Required CPU resources
            memory: Required memory resources
            
        Returns:
            True if sufficient resources available, False otherwise
        """
        if node_id not in self.graph.nodes:
            return False
        
        resources = self.graph.nodes[node_id]['resources']
        return (resources.cpu_available >= cpu and 
                resources.memory_available >= memory)
    
    def check_link_resources(self, src: int, dst: int, bandwidth: float) -> bool:
        """
        Check if a link has sufficient available bandwidth.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            bandwidth: Required bandwidth
            
        Returns:
            True if sufficient bandwidth available, False otherwise
        """
        if not self.graph.has_edge(src, dst):
            return False
        
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
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive network statistics.
        
        Returns:
            Dictionary containing network statistics
        """
        nodes = self.graph.nodes
        edges = self.graph.edges
        
        total_cpu = sum(nodes[n]['resources'].cpu_capacity for n in nodes)
        used_cpu = sum(nodes[n]['resources'].cpu_used for n in nodes)
        total_memory = sum(nodes[n]['resources'].memory_capacity for n in nodes)
        used_memory = sum(nodes[n]['resources'].memory_used for n in nodes)
        
        total_bandwidth = sum(edges[e]['resources'].bandwidth_capacity for e in edges)
        used_bandwidth = sum(edges[e]['resources'].bandwidth_used for e in edges)
        
        stats = {
            'node_count': len(nodes),
            'link_count': len(edges),
            'is_connected': nx.is_connected(self.graph) if not self.graph.is_directed() else nx.is_weakly_connected(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(nodes) if nodes else 0,
            'total_cpu': total_cpu,
            'used_cpu': used_cpu,
            'available_cpu': total_cpu - used_cpu,
            'cpu_utilization': used_cpu / total_cpu if total_cpu > 0 else 0,
            'total_memory': total_memory,
            'used_memory': used_memory,
            'available_memory': total_memory - used_memory,
            'memory_utilization': used_memory / total_memory if total_memory > 0 else 0,
            'total_bandwidth': total_bandwidth,
            'used_bandwidth': used_bandwidth,
            'available_bandwidth': total_bandwidth - used_bandwidth,
            'bandwidth_utilization': used_bandwidth / total_bandwidth if total_bandwidth > 0 else 0,
        }
        
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
            if resources.memory_used > resources.memory_capacity:
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
                        memory_capacity=float(row['memory_capacity']),
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
                        delay=float(row['delay']),
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