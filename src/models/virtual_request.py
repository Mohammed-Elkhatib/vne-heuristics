"""
Virtual Network Request Model for Virtual Network Embedding.

This module provides classes for representing and managing Virtual Network 
Requests (VNRs) including resource requirements and batch processing capabilities.
"""

import logging
import csv
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx


logger = logging.getLogger(__name__)


class VNRError(Exception):
    """Base exception for VNR operations."""
    pass


class VNRValidationError(VNRError):
    """Exception raised when VNR validation fails."""
    pass


class VNRFileFormatError(VNRError):
    """Exception raised when VNR file format is invalid."""
    pass


@dataclass
class VirtualNodeRequirement:
    """Represents resource requirements for a virtual node."""
    node_id: int
    cpu_requirement: float
    memory_requirement: float
    node_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node requirements after initialization."""
        if self.cpu_requirement < 0:
            raise VNRValidationError("CPU requirement cannot be negative")
        if self.memory_requirement < 0:
            raise VNRValidationError("Memory requirement cannot be negative")


@dataclass
class VirtualLinkRequirement:
    """Represents resource requirements for a virtual link."""
    src_node: int
    dst_node: int
    bandwidth_requirement: float
    delay_constraint: float = float('inf')
    reliability_requirement: float = 0.0
    link_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate link requirements after initialization."""
        if self.bandwidth_requirement < 0:
            raise VNRValidationError("Bandwidth requirement cannot be negative")
        if self.delay_constraint < 0:
            raise VNRValidationError("Delay constraint cannot be negative")
        if not (0 <= self.reliability_requirement <= 1):
            raise VNRValidationError("Reliability requirement must be between 0 and 1")


class VirtualNetworkRequest:
    """
    Represents a Virtual Network Request (VNR) for embedding.
    
    A VNR contains virtual nodes with resource requirements, virtual links
    with bandwidth and delay constraints, and metadata like arrival time
    and lifetime.
    
    Attributes:
        vnr_id: Unique identifier for the VNR
        virtual_nodes: Dictionary of virtual node requirements
        virtual_links: Dictionary of virtual link requirements
        arrival_time: Time when VNR arrives (simulation time)
        lifetime: Duration the VNR should remain active
        priority: Priority level (higher = more important)
        graph: NetworkX graph representation of the VNR topology
        
    Example:
        >>> vnr = VirtualNetworkRequest(vnr_id=1, arrival_time=0, lifetime=100)
        >>> vnr.add_virtual_node(1, cpu_requirement=50, memory_requirement=100)
        >>> vnr.add_virtual_link(1, 2, bandwidth_requirement=100)
        >>> revenue = vnr.calculate_revenue()
        >>> can_embed = vnr.check_feasibility(substrate_network)
    """
    
    def __init__(self, vnr_id: int, arrival_time: float = 0.0, 
                 lifetime: float = float('inf'), priority: int = 1):
        """
        Initialize a Virtual Network Request.
        
        Args:
            vnr_id: Unique identifier for the VNR
            arrival_time: Arrival time in simulation (default: 0.0)
            lifetime: How long VNR should remain active (default: infinite)
            priority: Priority level (default: 1)
            
        Raises:
            VNRValidationError: If parameters are invalid
        """
        if lifetime <= 0 and lifetime != float('inf'):
            raise VNRValidationError("Lifetime must be positive or infinite")
        if priority < 0:
            raise VNRValidationError("Priority must be non-negative")
        
        self.vnr_id = vnr_id
        self.arrival_time = arrival_time
        self.lifetime = lifetime
        self.priority = priority
        
        self.virtual_nodes: Dict[int, VirtualNodeRequirement] = {}
        self.virtual_links: Dict[Tuple[int, int], VirtualLinkRequirement] = {}
        self.graph = nx.Graph()
        
        # Additional metadata
        self.metadata: Dict[str, Any] = {}
        
        logger.debug(f"Initialized VNR {vnr_id} with arrival_time={arrival_time}, "
                    f"lifetime={lifetime}, priority={priority}")
    
    def add_virtual_node(self, node_id: int, cpu_requirement: float, 
                        memory_requirement: float, 
                        node_constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a virtual node to the VNR.
        
        Args:
            node_id: Unique identifier for the virtual node
            cpu_requirement: Required CPU resources
            memory_requirement: Required memory resources
            node_constraints: Additional constraints (optional)
            
        Raises:
            VNRValidationError: If node already exists or requirements are invalid
        """
        if node_id in self.virtual_nodes:
            raise VNRValidationError(f"Virtual node {node_id} already exists in VNR {self.vnr_id}")
        
        constraints = node_constraints or {}
        node_req = VirtualNodeRequirement(
            node_id=node_id,
            cpu_requirement=cpu_requirement,
            memory_requirement=memory_requirement,
            node_constraints=constraints
        )
        
        self.virtual_nodes[node_id] = node_req
        self.graph.add_node(node_id, requirements=node_req)
        
        logger.debug(f"Added virtual node {node_id} to VNR {self.vnr_id}: "
                    f"CPU={cpu_requirement}, Memory={memory_requirement}")
    
    def add_virtual_link(self, src_node: int, dst_node: int, 
                        bandwidth_requirement: float, delay_constraint: float = float('inf'),
                        reliability_requirement: float = 0.0,
                        link_constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a virtual link to the VNR.
        
        Args:
            src_node: Source virtual node ID
            dst_node: Destination virtual node ID
            bandwidth_requirement: Required bandwidth
            delay_constraint: Maximum acceptable delay
            reliability_requirement: Minimum reliability requirement
            link_constraints: Additional constraints (optional)
            
        Raises:
            VNRValidationError: If link already exists, nodes don't exist, or requirements are invalid
        """
        if src_node not in self.virtual_nodes:
            raise VNRValidationError(f"Source node {src_node} does not exist in VNR {self.vnr_id}")
        if dst_node not in self.virtual_nodes:
            raise VNRValidationError(f"Destination node {dst_node} does not exist in VNR {self.vnr_id}")
        
        link_key = (src_node, dst_node)
        if link_key in self.virtual_links:
            raise VNRValidationError(f"Virtual link {link_key} already exists in VNR {self.vnr_id}")
        
        constraints = link_constraints or {}
        link_req = VirtualLinkRequirement(
            src_node=src_node,
            dst_node=dst_node,
            bandwidth_requirement=bandwidth_requirement,
            delay_constraint=delay_constraint,
            reliability_requirement=reliability_requirement,
            link_constraints=constraints
        )
        
        self.virtual_links[link_key] = link_req
        self.graph.add_edge(src_node, dst_node, requirements=link_req)
        
        logger.debug(f"Added virtual link ({src_node}, {dst_node}) to VNR {self.vnr_id}: "
                    f"Bandwidth={bandwidth_requirement}, Delay≤{delay_constraint}")
    
    def remove_virtual_node(self, node_id: int) -> None:
        """
        Remove a virtual node and all its connected links.
        
        Args:
            node_id: ID of the virtual node to remove
            
        Raises:
            VNRValidationError: If node doesn't exist
        """
        if node_id not in self.virtual_nodes:
            raise VNRValidationError(f"Virtual node {node_id} does not exist in VNR {self.vnr_id}")
        
        # Remove all links connected to this node
        links_to_remove = [(src, dst) for src, dst in self.virtual_links.keys() 
                          if src == node_id or dst == node_id]
        
        for src, dst in links_to_remove:
            del self.virtual_links[(src, dst)]
        
        # Remove the node
        del self.virtual_nodes[node_id]
        self.graph.remove_node(node_id)
        
        logger.debug(f"Removed virtual node {node_id} and {len(links_to_remove)} connected links from VNR {self.vnr_id}")
    
    def remove_virtual_link(self, src_node: int, dst_node: int) -> None:
        """
        Remove a virtual link from the VNR.
        
        Args:
            src_node: Source virtual node ID
            dst_node: Destination virtual node ID
            
        Raises:
            VNRValidationError: If link doesn't exist
        """
        link_key = (src_node, dst_node)
        if link_key not in self.virtual_links:
            raise VNRValidationError(f"Virtual link {link_key} does not exist in VNR {self.vnr_id}")
        
        del self.virtual_links[link_key]
        self.graph.remove_edge(src_node, dst_node)
        
        logger.debug(f"Removed virtual link ({src_node}, {dst_node}) from VNR {self.vnr_id}")
    
    def calculate_total_requirements(self) -> Dict[str, float]:
        """
        Calculate total resource requirements for the VNR.
        
        Returns:
            Dictionary with total CPU, memory, and bandwidth requirements
        """
        total_cpu = sum(node.cpu_requirement for node in self.virtual_nodes.values())
        total_memory = sum(node.memory_requirement for node in self.virtual_nodes.values())
        total_bandwidth = sum(link.bandwidth_requirement for link in self.virtual_links.values())
        
        return {
            'total_cpu': total_cpu,
            'total_memory': total_memory,
            'total_bandwidth': total_bandwidth,
            'node_count': len(self.virtual_nodes),
            'link_count': len(self.virtual_links)
        }
    
    def calculate_revenue(self, cpu_weight: float = 1.0, memory_weight: float = 1.0, 
                         bandwidth_weight: float = 1.0, duration_weight: float = 1.0) -> float:
        """
        Calculate potential revenue for the VNR based on resource requirements.
        
        Revenue is calculated as a weighted sum of resource requirements multiplied
        by duration and priority.
        
        Args:
            cpu_weight: Weight for CPU requirements (default: 1.0)
            memory_weight: Weight for memory requirements (default: 1.0)
            bandwidth_weight: Weight for bandwidth requirements (default: 1.0)
            duration_weight: Weight for duration (default: 1.0)
            
        Returns:
            Calculated revenue value
        """
        requirements = self.calculate_total_requirements()
        
        resource_value = (
            requirements['total_cpu'] * cpu_weight +
            requirements['total_memory'] * memory_weight +
            requirements['total_bandwidth'] * bandwidth_weight
        )
        
        # Consider duration (finite lifetime gets full weight, infinite gets reduced weight)
        if self.lifetime == float('inf'):
            duration_factor = 100.0  # Default duration for infinite lifetime
        else:
            duration_factor = self.lifetime
        
        revenue = resource_value * duration_factor * duration_weight * self.priority
        
        logger.debug(f"Calculated revenue for VNR {self.vnr_id}: {revenue}")
        return revenue
    
    def validate_request(self) -> List[str]:
        """
        Validate the VNR for consistency and completeness.
        
        Returns:
            List of validation error messages (empty if no issues)
        """
        issues = []
        
        # Check if VNR has nodes
        if not self.virtual_nodes:
            issues.append("VNR has no virtual nodes")
        
        # Check if graph is connected (for multi-node VNRs)
        if len(self.virtual_nodes) > 1:
            if not nx.is_connected(self.graph):
                issues.append("Virtual network topology is not connected")
        
        # Validate node requirements
        for node_id, node_req in self.virtual_nodes.items():
            if node_req.cpu_requirement < 0:
                issues.append(f"Node {node_id} has negative CPU requirement")
            if node_req.memory_requirement < 0:
                issues.append(f"Node {node_id} has negative memory requirement")
        
        # Validate link requirements
        for (src, dst), link_req in self.virtual_links.items():
            if src not in self.virtual_nodes:
                issues.append(f"Link ({src}, {dst}) references non-existent source node {src}")
            if dst not in self.virtual_nodes:
                issues.append(f"Link ({src}, {dst}) references non-existent destination node {dst}")
            if link_req.bandwidth_requirement < 0:
                issues.append(f"Link ({src}, {dst}) has negative bandwidth requirement")
            if link_req.delay_constraint < 0:
                issues.append(f"Link ({src}, {dst}) has negative delay constraint")
        
        # Check for self-loops
        for src, dst in self.virtual_links.keys():
            if src == dst:
                issues.append(f"Self-loop detected on node {src}")
        
        return issues
    
    def check_feasibility(self, substrate_network) -> Tuple[bool, List[str]]:
        """
        Check if the VNR can potentially be embedded in the substrate network.
        
        This performs a basic feasibility check without actually attempting embedding.
        
        Args:
            substrate_network: SubstrateNetwork instance to check against
            
        Returns:
            Tuple of (is_feasible, list_of_issues)
        """
        issues = []
        
        # First validate the VNR itself
        vnr_issues = self.validate_request()
        if vnr_issues:
            return False, vnr_issues
        
        # Check if substrate has enough total resources
        requirements = self.calculate_total_requirements()
        substrate_stats = substrate_network.get_network_statistics()
        
        if requirements['total_cpu'] > substrate_stats['available_cpu']:
            issues.append(f"Insufficient total CPU: required={requirements['total_cpu']}, "
                         f"available={substrate_stats['available_cpu']}")
        
        if requirements['total_memory'] > substrate_stats['available_memory']:
            issues.append(f"Insufficient total memory: required={requirements['total_memory']}, "
                         f"available={substrate_stats['available_memory']}")
        
        if requirements['total_bandwidth'] > substrate_stats['available_bandwidth']:
            issues.append(f"Insufficient total bandwidth: required={requirements['total_bandwidth']}, "
                         f"available={substrate_stats['available_bandwidth']}")
        
        # Check if substrate has enough nodes
        if len(self.virtual_nodes) > len(substrate_network.graph.nodes):
            issues.append(f"Not enough substrate nodes: required={len(self.virtual_nodes)}, "
                         f"available={len(substrate_network.graph.nodes)}")
        
        # Check connectivity (basic check)
        if not substrate_stats['is_connected'] and len(self.virtual_links) > 0:
            issues.append("Substrate network is not connected but VNR requires links")
        
        is_feasible = len(issues) == 0
        return is_feasible, issues
    
    def get_departure_time(self) -> float:
        """
        Get the departure time of the VNR.
        
        Returns:
            Departure time (arrival_time + lifetime)
        """
        if self.lifetime == float('inf'):
            return float('inf')
        return self.arrival_time + self.lifetime
    
    def is_active(self, current_time: float) -> bool:
        """
        Check if the VNR is currently active.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if VNR is active, False otherwise
        """
        return (self.arrival_time <= current_time < self.get_departure_time())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert VNR to dictionary representation.
        
        Returns:
            Dictionary representation of the VNR
        """
        return {
            'vnr_id': self.vnr_id,
            'arrival_time': self.arrival_time,
            'lifetime': self.lifetime if self.lifetime != float('inf') else -1,
            'priority': self.priority,
            'virtual_nodes': {
                node_id: {
                    'cpu_requirement': node.cpu_requirement,
                    'memory_requirement': node.memory_requirement,
                    'node_constraints': node.node_constraints
                }
                for node_id, node in self.virtual_nodes.items()
            },
            'virtual_links': {
                f"{src}-{dst}": {
                    'bandwidth_requirement': link.bandwidth_requirement,
                    'delay_constraint': link.delay_constraint if link.delay_constraint != float('inf') else -1,
                    'reliability_requirement': link.reliability_requirement,
                    'link_constraints': link.link_constraints
                }
                for (src, dst), link in self.virtual_links.items()
            },
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VirtualNetworkRequest':
        """
        Create VNR from dictionary representation.
        
        Args:
            data: Dictionary containing VNR data
            
        Returns:
            VirtualNetworkRequest instance
            
        Raises:
            VNRValidationError: If data format is invalid
        """
        try:
            lifetime = data['lifetime'] if data['lifetime'] != -1 else float('inf')
            vnr = cls(
                vnr_id=data['vnr_id'],
                arrival_time=data['arrival_time'],
                lifetime=lifetime,
                priority=data['priority']
            )
            
            # Add virtual nodes
            for node_id, node_data in data['virtual_nodes'].items():
                vnr.add_virtual_node(
                    node_id=int(node_id),
                    cpu_requirement=node_data['cpu_requirement'],
                    memory_requirement=node_data['memory_requirement'],
                    node_constraints=node_data.get('node_constraints', {})
                )
            
            # Add virtual links
            for link_key, link_data in data['virtual_links'].items():
                src, dst = map(int, link_key.split('-'))
                delay_constraint = link_data['delay_constraint'] if link_data['delay_constraint'] != -1 else float('inf')
                vnr.add_virtual_link(
                    src_node=src,
                    dst_node=dst,
                    bandwidth_requirement=link_data['bandwidth_requirement'],
                    delay_constraint=delay_constraint,
                    reliability_requirement=link_data.get('reliability_requirement', 0.0),
                    link_constraints=link_data.get('link_constraints', {})
                )
            
            vnr.metadata = data.get('metadata', {})
            return vnr
            
        except (KeyError, ValueError, TypeError) as e:
            raise VNRValidationError(f"Invalid VNR data format: {e}")
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save VNR to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved VNR {self.vnr_id} to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'VirtualNetworkRequest':
        """
        Load VNR from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            VirtualNetworkRequest instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            VNRValidationError: If file format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            vnr = cls.from_dict(data)
            logger.info(f"Loaded VNR {vnr.vnr_id} from {filepath}")
            return vnr
            
        except FileNotFoundError:
            raise FileNotFoundError(f"VNR file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise VNRValidationError(f"Invalid JSON format: {e}")
    
    def __str__(self) -> str:
        """String representation of the VNR."""
        requirements = self.calculate_total_requirements()
        return (f"VNR({self.vnr_id}: {len(self.virtual_nodes)} nodes, "
                f"{len(self.virtual_links)} links, "
                f"CPU={requirements['total_cpu']}, "
                f"Memory={requirements['total_memory']}, "
                f"Bandwidth={requirements['total_bandwidth']})")
    
    def __repr__(self) -> str:
        """Detailed representation of the VNR."""
        return (f"VirtualNetworkRequest(vnr_id={self.vnr_id}, "
                f"arrival_time={self.arrival_time}, lifetime={self.lifetime}, "
                f"priority={self.priority}, nodes={len(self.virtual_nodes)}, "
                f"links={len(self.virtual_links)})")


class VNRBatch:
    """
    Manages a batch of Virtual Network Requests for experiments.
    
    This class provides utilities for loading, generating, and managing
    multiple VNRs for batch processing and online simulation scenarios.
    
    Attributes:
        vnrs: List of VirtualNetworkRequest instances
        batch_id: Unique identifier for the batch
        
    Example:
        >>> batch = VNRBatch.generate_random_batch(count=50, batch_id="experiment_1")
        >>> batch.save_to_csv("vnr_batch.csv")
        >>> loaded_batch = VNRBatch.load_from_csv("vnr_batch.csv")
        >>> feasible_vnrs = batch.filter_feasible(substrate_network)
    """
    
    def __init__(self, vnrs: Optional[List[VirtualNetworkRequest]] = None, 
                 batch_id: str = "default"):
        """
        Initialize VNR batch.
        
        Args:
            vnrs: List of VNR instances (default: empty list)
            batch_id: Identifier for the batch
        """
        self.vnrs = vnrs or []
        self.batch_id = batch_id
        logger.info(f"Initialized VNR batch {batch_id} with {len(self.vnrs)} VNRs")
    
    def add_vnr(self, vnr: VirtualNetworkRequest) -> None:
        """
        Add a VNR to the batch.
        
        Args:
            vnr: VirtualNetworkRequest to add
        """
        self.vnrs.append(vnr)
        logger.debug(f"Added VNR {vnr.vnr_id} to batch {self.batch_id}")
    
    def remove_vnr(self, vnr_id: int) -> bool:
        """
        Remove a VNR from the batch.
        
        Args:
            vnr_id: ID of the VNR to remove
            
        Returns:
            True if VNR was removed, False if not found
        """
        for i, vnr in enumerate(self.vnrs):
            if vnr.vnr_id == vnr_id:
                del self.vnrs[i]
                logger.debug(f"Removed VNR {vnr_id} from batch {self.batch_id}")
                return True
        return False
    
    def get_vnr(self, vnr_id: int) -> Optional[VirtualNetworkRequest]:
        """
        Get a VNR by ID.
        
        Args:
            vnr_id: ID of the VNR to retrieve
            
        Returns:
            VirtualNetworkRequest instance or None if not found
        """
        for vnr in self.vnrs:
            if vnr.vnr_id == vnr_id:
                return vnr
        return None
    
    def sort_by_arrival_time(self) -> None:
        """Sort VNRs by arrival time (for online simulation)."""
        self.vnrs.sort(key=lambda vnr: vnr.arrival_time)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by arrival time")
    
    def sort_by_priority(self, descending: bool = True) -> None:
        """
        Sort VNRs by priority.
        
        Args:
            descending: If True, sort by descending priority (default: True)
        """
        self.vnrs.sort(key=lambda vnr: vnr.priority, reverse=descending)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by priority ({'desc' if descending else 'asc'})")
    
    def filter_by_time_range(self, start_time: float, end_time: float) -> 'VNRBatch':
        """
        Filter VNRs by arrival time range.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (exclusive)
            
        Returns:
            New VNRBatch with filtered VNRs
        """
        filtered_vnrs = [vnr for vnr in self.vnrs 
                        if start_time <= vnr.arrival_time < end_time]
        
        return VNRBatch(filtered_vnrs, f"{self.batch_id}_filtered")
    
    def filter_feasible(self, substrate_network) -> 'VNRBatch':
        """
        Filter VNRs that are potentially feasible for the substrate network.
        
        Args:
            substrate_network: SubstrateNetwork instance
            
        Returns:
            New VNRBatch with only feasible VNRs
        """
        feasible_vnrs = []
        for vnr in self.vnrs:
            is_feasible, _ = vnr.check_feasibility(substrate_network)
            if is_feasible:
                feasible_vnrs.append(vnr)
        
        logger.info(f"Filtered {len(feasible_vnrs)}/{len(self.vnrs)} feasible VNRs")
        return VNRBatch(feasible_vnrs, f"{self.batch_id}_feasible")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get batch statistics.
        
        Returns:
            Dictionary with batch statistics
        """
        if not self.vnrs:
            return {'count': 0}
        
        total_requirements = [vnr.calculate_total_requirements() for vnr in self.vnrs]
        revenues = [vnr.calculate_revenue() for vnr in self.vnrs]
        
        stats = {
            'count': len(self.vnrs),
            'avg_nodes_per_vnr': sum(req['node_count'] for req in total_requirements) / len(self.vnrs),
            'avg_links_per_vnr': sum(req['link_count'] for req in total_requirements) / len(self.vnrs),
            'total_cpu_demand': sum(req['total_cpu'] for req in total_requirements),
            'total_memory_demand': sum(req['total_memory'] for req in total_requirements),
            'total_bandwidth_demand': sum(req['total_bandwidth'] for req in total_requirements),
            'avg_cpu_per_vnr': sum(req['total_cpu'] for req in total_requirements) / len(self.vnrs),
            'avg_memory_per_vnr': sum(req['total_memory'] for req in total_requirements) / len(self.vnrs),
            'avg_bandwidth_per_vnr': sum(req['total_bandwidth'] for req in total_requirements) / len(self.vnrs),
            'total_revenue': sum(revenues),
            'avg_revenue_per_vnr': sum(revenues) / len(self.vnrs),
            'arrival_time_range': (min(vnr.arrival_time for vnr in self.vnrs),
                                 max(vnr.arrival_time for vnr in self.vnrs)),
            'priority_range': (min(vnr.priority for vnr in self.vnrs),
                             max(vnr.priority for vnr in self.vnrs))
        }
        
        return stats
    
    def save_to_csv(self, base_filename: str) -> None:
        """
        Save VNR batch to CSV files.
        
        Creates three CSV files:
        - {base}_nodes.csv: Virtual node requirements
        - {base}_links.csv: Virtual link requirements  
        - {base}_meta.csv: VNR metadata
        
        Args:
            base_filename: Base filename (without extension)
        """
        base_path = Path(base_filename)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        nodes_file = f"{base_filename}_nodes.csv"
        links_file = f"{base_filename}_links.csv"
        meta_file = f"{base_filename}_meta.csv"
        
        # Save virtual nodes
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'node_id', 'cpu_requirement', 'memory_requirement', 'constraints'])
            
            for vnr in self.vnrs:
                for node_id, node_req in vnr.virtual_nodes.items():
                    writer.writerow([
                        vnr.vnr_id,
                        node_id,
                        node_req.cpu_requirement,
                        node_req.memory_requirement,
                        json.dumps(node_req.node_constraints)
                    ])
        
        # Save virtual links
        with open(links_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'src_node', 'dst_node', 'bandwidth_requirement', 
                           'delay_constraint', 'reliability_requirement', 'constraints'])
            
            for vnr in self.vnrs:
                for (src, dst), link_req in vnr.virtual_links.items():
                    delay = link_req.delay_constraint if link_req.delay_constraint != float('inf') else -1
                    writer.writerow([
                        vnr.vnr_id,
                        src,
                        dst,
                        link_req.bandwidth_requirement,
                        delay,
                        link_req.reliability_requirement,
                        json.dumps(link_req.link_constraints)
                    ])
        
        # Save metadata
        with open(meta_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'arrival_time', 'lifetime', 'priority', 'metadata'])
            
            for vnr in self.vnrs:
                lifetime = vnr.lifetime if vnr.lifetime != float('inf') else -1
                writer.writerow([
                    vnr.vnr_id,
                    vnr.arrival_time,
                    lifetime,
                    vnr.priority,
                    json.dumps(vnr.metadata)
                ])
        
        logger.info(f"Saved VNR batch {self.batch_id} to {nodes_file}, {links_file}, {meta_file}")
    
    @classmethod
    def load_from_csv(cls, base_filename: str, batch_id: Optional[str] = None) -> 'VNRBatch':
        """
        Load VNR batch from CSV files.
        
        Args:
            base_filename: Base filename (without extension)
            batch_id: Batch identifier (default: derived from filename)
            
        Returns:
            VNRBatch instance
            
        Raises:
            FileNotFoundError: If CSV files don't exist
            VNRFileFormatError: If file format is invalid
        """
        if batch_id is None:
            batch_id = Path(base_filename).stem
        
        nodes_file = f"{base_filename}_nodes.csv"
        links_file = f"{base_filename}_links.csv"
        meta_file = f"{base_filename}_meta.csv"
        
        try:
            # Load metadata first
            vnr_data = {}
            with open(meta_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    lifetime = float(row['lifetime']) if float(row['lifetime']) != -1 else float('inf')
                    vnr_data[vnr_id] = {
                        'arrival_time': float(row['arrival_time']),
                        'lifetime': lifetime,
                        'priority': int(row['priority']),
                        'metadata': json.loads(row['metadata'])
                    }
            
            # Create VNR instances
            vnrs_dict = {}
            for vnr_id, data in vnr_data.items():
                vnr = VirtualNetworkRequest(
                    vnr_id=vnr_id,
                    arrival_time=data['arrival_time'],
                    lifetime=data['lifetime'],
                    priority=data['priority']
                )
                vnr.metadata = data['metadata']
                vnrs_dict[vnr_id] = vnr
            
            # Load virtual nodes
            with open(nodes_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnrs_dict:
                        constraints = json.loads(row['constraints']) if row['constraints'] else {}
                        vnrs_dict[vnr_id].add_virtual_node(
                            node_id=int(row['node_id']),
                            cpu_requirement=float(row['cpu_requirement']),
                            memory_requirement=float(row['memory_requirement']),
                            node_constraints=constraints
                        )
            
            # Load virtual links
            with open(links_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnrs_dict:
                        constraints = json.loads(row['constraints']) if row['constraints'] else {}
                        delay = float(row['delay_constraint']) if float(row['delay_constraint']) != -1 else float('inf')
                        vnrs_dict[vnr_id].add_virtual_link(
                            src_node=int(row['src_node']),
                            dst_node=int(row['dst_node']),
                            bandwidth_requirement=float(row['bandwidth_requirement']),
                            delay_constraint=delay,
                            reliability_requirement=float(row.get('reliability_requirement', 0.0)),
                            link_constraints=constraints
                        )
            
            # Create batch
            batch = cls(list(vnrs_dict.values()), batch_id)
            logger.info(f"Loaded VNR batch {batch_id} with {len(batch.vnrs)} VNRs")
            return batch
            
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
            raise VNRFileFormatError(f"Error loading VNR batch: {e}")
    
    def __len__(self) -> int:
        """Return number of VNRs in the batch."""
        return len(self.vnrs)
    
    def __iter__(self):
        """Iterate over VNRs in the batch."""
        return iter(self.vnrs)
    
    def __getitem__(self, index: int) -> VirtualNetworkRequest:
        """Get VNR by index."""
        return self.vnrs[index]