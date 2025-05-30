"""
Virtual Network Request Model for Virtual Network Embedding.

This module provides classes for representing and managing Virtual Network 
Requests (VNRs) with optional constraint support.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import networkx as nx


logger = logging.getLogger(__name__)


class VNRError(Exception):
    """Base exception for VNR operations."""
    pass


class VNRValidationError(VNRError):
    """Exception raised when VNR validation fails."""
    pass


@dataclass
class VirtualNodeRequirement:
    """Represents resource requirements for a virtual node."""
    node_id: int
    cpu_requirement: float
    memory_requirement: float = 0.0  # Default: no memory requirement
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
    delay_constraint: float = 0.0  # Default: no delay constraint
    reliability_requirement: float = 0.0  # Default: no reliability requirement
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
    with bandwidth and optional delay/reliability constraints, and metadata
    like arrival time and holding_time.

    Constraints are optional and default to zero (no constraint):
    - CPU and bandwidth are typically always required
    - Memory, delay, and reliability are optional based on problem requirements

    Attributes:
        vnr_id: Unique identifier for the VNR
        virtual_nodes: Dictionary of virtual node requirements
        virtual_links: Dictionary of virtual link requirements
        arrival_time: Time when VNR arrives (simulation time)
        holding_time: Duration the VNR should remain active
        priority: Priority level (higher = more important)
        graph: NetworkX graph representation of the VNR topology

    Example:
        >>> vnr = VirtualNetworkRequest(vnr_id=1, arrival_time=0, holding_time=100)
        >>> vnr.add_virtual_node(1, cpu_requirement=50)  # No memory requirement
        >>> vnr.add_virtual_node(2, cpu_requirement=30, memory_requirement=100)  # With memory
        >>> vnr.add_virtual_link(1, 2, bandwidth_requirement=100)  # No delay constraint
        >>> vnr.add_virtual_link(1, 3, bandwidth_requirement=100, delay_constraint=10.0)  # With delay
    """

    def __init__(self, vnr_id: int, arrival_time: float = 0.0,
                 holding_time: float = float('inf'), priority: int = 1):
        """
        Initialize a Virtual Network Request.

        Args:
            vnr_id: Unique identifier for the VNR
            arrival_time: Arrival time in simulation (default: 0.0)
            holding_time: How long VNR should remain active (default: infinite)
            priority: Priority level (default: 1)

        Raises:
            VNRValidationError: If parameters are invalid
        """
        if holding_time <= 0 and holding_time != float('inf'):
            raise VNRValidationError("holding_time must be positive or infinite")
        if priority < 0:
            raise VNRValidationError("Priority must be non-negative")

        self.vnr_id = vnr_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.priority = priority

        self.virtual_nodes: Dict[int, VirtualNodeRequirement] = {}
        self.virtual_links: Dict[Tuple[int, int], VirtualLinkRequirement] = {}
        self.graph = nx.Graph()

        # Additional metadata
        self.metadata: Dict[str, Any] = {}

        logger.debug(f"Initialized VNR {vnr_id} with arrival_time={arrival_time}, "
                    f"holding_time={holding_time}, priority={priority}")

    def add_virtual_node(self, node_id: int, cpu_requirement: float,
                        memory_requirement: float = 0.0,
                        node_constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a virtual node to the VNR.

        Args:
            node_id: Unique identifier for the virtual node
            cpu_requirement: Required CPU resources
            memory_requirement: Required memory resources (default: 0.0 = no requirement)
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

        if memory_requirement > 0:
            logger.debug(f"Added virtual node {node_id} to VNR {self.vnr_id}: "
                        f"CPU={cpu_requirement}, Memory={memory_requirement}")
        else:
            logger.debug(f"Added virtual node {node_id} to VNR {self.vnr_id}: "
                        f"CPU={cpu_requirement}, Memory=none")

    def add_virtual_link(self, src_node: int, dst_node: int,
                        bandwidth_requirement: float, delay_constraint: float = 0.0,
                        reliability_requirement: float = 0.0,
                        link_constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a virtual link to the VNR.

        Args:
            src_node: Source virtual node ID
            dst_node: Destination virtual node ID
            bandwidth_requirement: Required bandwidth
            delay_constraint: Maximum acceptable delay (default: 0.0 = no constraint)
            reliability_requirement: Minimum reliability requirement (default: 0.0 = no requirement)
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

        # Build log message with constraints
        constraints_info = []
        if delay_constraint > 0:
            constraints_info.append(f"Delay≤{delay_constraint}")
        if reliability_requirement > 0:
            constraints_info.append(f"Reliability≥{reliability_requirement}")

        constraint_str = f" ({', '.join(constraints_info)})" if constraints_info else ""

        logger.debug(f"Added virtual link ({src_node}, {dst_node}) to VNR {self.vnr_id}: "
                    f"Bandwidth={bandwidth_requirement}{constraint_str}")

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

    def calculate_total_requirements(self, include_memory: bool = True,
                                   include_delay: bool = True,
                                   include_reliability: bool = True) -> Dict[str, float]:
        """
        Calculate total resource requirements for the VNR.

        Args:
            include_memory: Whether to include memory requirements in calculation
            include_delay: Whether to include delay constraints in calculation
            include_reliability: Whether to include reliability requirements in calculation

        Returns:
            Dictionary with total requirements (only includes requested constraint types)
        """
        requirements = {
            'total_cpu': sum(node.cpu_requirement for node in self.virtual_nodes.values()),
            'total_bandwidth': sum(link.bandwidth_requirement for link in self.virtual_links.values()),
            'node_count': len(self.virtual_nodes),
            'link_count': len(self.virtual_links)
        }

        if include_memory:
            requirements['total_memory'] = sum(node.memory_requirement for node in self.virtual_nodes.values())

        if include_delay:
            # Only include links that actually have delay constraints
            delay_constrained_links = [link for link in self.virtual_links.values() if link.delay_constraint > 0]
            requirements['total_delay_constraint'] = sum(link.delay_constraint for link in delay_constrained_links)
            requirements['delay_constrained_links'] = len(delay_constrained_links)

        if include_reliability:
            # Only include links that actually have reliability requirements
            reliability_constrained_links = [link for link in self.virtual_links.values() if link.reliability_requirement > 0]
            requirements['avg_reliability_requirement'] = (
                sum(link.reliability_requirement for link in reliability_constrained_links) / len(reliability_constrained_links)
                if reliability_constrained_links else 0.0
            )
            requirements['reliability_constrained_links'] = len(reliability_constrained_links)

        return requirements

    def validate_request(self, validate_memory: bool = True,
                        validate_delay: bool = True,
                        validate_reliability: bool = True) -> List[str]:
        """
        Validate the VNR for consistency and completeness.

        Args:
            validate_memory: Whether to validate memory requirements
            validate_delay: Whether to validate delay constraints
            validate_reliability: Whether to validate reliability requirements

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

            if validate_memory and node_req.memory_requirement < 0:
                issues.append(f"Node {node_id} has negative memory requirement")

        # Validate link requirements
        for (src, dst), link_req in self.virtual_links.items():
            if src not in self.virtual_nodes:
                issues.append(f"Link ({src}, {dst}) references non-existent source node {src}")
            if dst not in self.virtual_nodes:
                issues.append(f"Link ({src}, {dst}) references non-existent destination node {dst}")
            if link_req.bandwidth_requirement < 0:
                issues.append(f"Link ({src}, {dst}) has negative bandwidth requirement")

            if validate_delay and link_req.delay_constraint < 0:
                issues.append(f"Link ({src}, {dst}) has negative delay constraint")

            if validate_reliability and not (0 <= link_req.reliability_requirement <= 1):
                issues.append(f"Link ({src}, {dst}) has invalid reliability requirement")

        # Check for self-loops
        for src, dst in self.virtual_links.keys():
            if src == dst:
                issues.append(f"Self-loop detected on node {src}")

        return issues

    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        Get summary of which constraints are actually used in this VNR.

        Returns:
            Dictionary with constraint usage information
        """
        memory_nodes = sum(1 for node in self.virtual_nodes.values() if node.memory_requirement > 0)
        delay_links = sum(1 for link in self.virtual_links.values() if link.delay_constraint > 0)
        reliability_links = sum(1 for link in self.virtual_links.values() if link.reliability_requirement > 0)

        return {
            'uses_memory_constraints': memory_nodes > 0,
            'uses_delay_constraints': delay_links > 0,
            'uses_reliability_constraints': reliability_links > 0,
            'memory_constrained_nodes': memory_nodes,
            'delay_constrained_links': delay_links,
            'reliability_constrained_links': reliability_links,
            'total_nodes': len(self.virtual_nodes),
            'total_links': len(self.virtual_links)
        }

    def get_departure_time(self, embedding_time: float) -> float:
        """
        Get the departure time of the VNR.

        Returns:
            Departure time (embedding_time + holding_time)
        """
        if self.holding_time == float('inf'):
            return float('inf')
        return embedding_time + self.holding_time

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert VNR to dictionary representation.

        Returns:
            Dictionary representation of the VNR
        """
        return {
            'vnr_id': self.vnr_id,
            'arrival_time': self.arrival_time,
            'holding_time': self.holding_time if self.holding_time != float('inf') else -1,
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
                    'delay_constraint': link.delay_constraint,
                    'reliability_requirement': link.reliability_requirement,
                    'link_constraints': link.link_constraints
                }
                for (src, dst), link in self.virtual_links.items()
            },
            'metadata': self.metadata,
            'constraint_summary': self.get_constraint_summary()
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
            holding_time = data['holding_time'] if data['holding_time'] != -1 else float('inf')
            vnr = cls(
                vnr_id=data['vnr_id'],
                arrival_time=data['arrival_time'],
                holding_time=holding_time,
                priority=data.get('priority', 1)
            )

            # Add virtual nodes
            for node_id, node_data in data['virtual_nodes'].items():
                vnr.add_virtual_node(
                    node_id=int(node_id),
                    cpu_requirement=node_data['cpu_requirement'],
                    memory_requirement=node_data.get('memory_requirement', 0.0),
                    node_constraints=node_data.get('node_constraints', {})
                )

            # Add virtual links
            for link_key, link_data in data['virtual_links'].items():
                src, dst = map(int, link_key.split('-'))
                vnr.add_virtual_link(
                    src_node=src,
                    dst_node=dst,
                    bandwidth_requirement=link_data['bandwidth_requirement'],
                    delay_constraint=link_data.get('delay_constraint', 0.0),
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
        from pathlib import Path
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
        constraint_summary = self.get_constraint_summary()

        constraint_info = []
        if constraint_summary['uses_memory_constraints']:
            constraint_info.append(f"Memory={requirements.get('total_memory', 0)}")
        if constraint_summary['uses_delay_constraints']:
            constraint_info.append(f"DelayLinks={constraint_summary['delay_constrained_links']}")
        if constraint_summary['uses_reliability_constraints']:
            constraint_info.append(f"ReliabilityLinks={constraint_summary['reliability_constrained_links']}")

        constraint_str = f", {', '.join(constraint_info)}" if constraint_info else ""

        return (f"VNR({self.vnr_id}: {len(self.virtual_nodes)} nodes, "
                f"{len(self.virtual_links)} links, "
                f"CPU={requirements['total_cpu']}, "
                f"Bandwidth={requirements['total_bandwidth']}{constraint_str})")

    def __repr__(self) -> str:
        """Detailed representation of the VNR."""
        return (f"VirtualNetworkRequest(vnr_id={self.vnr_id}, "
                f"arrival_time={self.arrival_time}, holding_time={self.holding_time}, "
                f"priority={self.priority}, nodes={len(self.virtual_nodes)}, "
                f"links={len(self.virtual_links)})")
