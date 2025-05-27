"""
Base algorithm interface for Virtual Network Embedding (VNE) algorithms.

This module provides the abstract base class that all VNE algorithms must inherit from,
along with standardized result structures and common functionality.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import threading

# Import existing models
from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork
from src.utils.metrics import EmbeddingResult as MetricsEmbeddingResult


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Standardized result structure for VNE embedding attempts.
    
    This dataclass represents the outcome of attempting to embed a VNR
    onto a substrate network, including success status, resource mappings,
    costs, and performance metrics.
    
    Attributes:
        vnr_id: Unique identifier for the VNR
        success: Whether the embedding was successful
        node_mapping: Dictionary mapping virtual nodes to substrate nodes
        link_mapping: Dictionary mapping virtual links to substrate paths
        revenue: Revenue generated if successful, 0 otherwise
        cost: Cost incurred for the embedding attempt
        execution_time: Time taken for the embedding attempt (seconds)
        failure_reason: Reason for failure if unsuccessful
        timestamp: When the embedding was attempted
        algorithm_name: Name of the algorithm used
        metadata: Additional algorithm-specific information
    """
    vnr_id: str
    success: bool
    node_mapping: Dict[str, str]  # virtual_node -> substrate_node
    link_mapping: Dict[Tuple[str, str], List[str]]  # (v_src, v_dst) -> path
    revenue: float
    cost: float
    execution_time: float
    failure_reason: Optional[str] = None
    timestamp: Optional[float] = None
    algorithm_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_metrics_result(self) -> MetricsEmbeddingResult:
        """
        Convert to MetricsEmbeddingResult for compatibility with metrics utilities.
        
        Returns:
            MetricsEmbeddingResult instance
        """
        return MetricsEmbeddingResult(
            vnr_id=self.vnr_id,
            success=self.success,
            revenue=self.revenue,
            cost=self.cost,
            execution_time=self.execution_time,
            node_mapping=self.node_mapping,
            link_mapping=self.link_mapping,
            timestamp=self.timestamp,
            failure_reason=self.failure_reason
        )


class BaseAlgorithm(ABC):
    """
    Abstract base class for all VNE algorithms.
    
    This class provides the standard interface that all VNE algorithms must implement,
    along with common functionality for metrics collection, logging, and resource tracking.
    
    Subclasses must implement the core embedding logic while inheriting standardized
    result handling, logging, and performance measurement capabilities.
    
    Attributes:
        name: Human-readable name of the algorithm
        logger: Logger instance for the algorithm
        _lock: Threading lock for resource allocation safety
        _stats: Internal statistics tracking
        
    Example:
        >>> class MyAlgorithm(BaseAlgorithm):
        ...     def __init__(self):
        ...         super().__init__("My Algorithm")
        ...     
        ...     def _embed_single_vnr(self, vnr, substrate):
        ...         # Implementation here
        ...         return EmbeddingResult(...)
        >>> 
        >>> algorithm = MyAlgorithm()
        >>> result = algorithm.embed_vnr(vnr, substrate)
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the base algorithm.
        
        Args:
            name: Human-readable name of the algorithm
            **kwargs: Additional algorithm-specific parameters
        """
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_execution_time': 0.0,
            'total_revenue': 0.0,
            'total_cost': 0.0
        }
        
        # Store algorithm parameters
        self.parameters = kwargs
        
        self.logger.info(f"Initialized {self.name} algorithm with parameters: {kwargs}")
    
    @abstractmethod
    def _embed_single_vnr(self, vnr: VirtualNetworkRequest, 
                         substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Core embedding logic for a single VNR.
        
        This method must be implemented by all subclasses to define the specific
        algorithm for embedding a virtual network request onto the substrate.
        
        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto
            
        Returns:
            EmbeddingResult with the outcome of the embedding attempt
            
        Note:
            This method should NOT modify substrate resources directly.
            Resource allocation/deallocation is handled by the base class.
        """
        pass
    
    def embed_vnr(self, vnr: VirtualNetworkRequest, 
                  substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Embed a single VNR with full resource management and metrics collection.
        
        This method wraps the core embedding logic with standardized functionality
        including timing, logging, resource allocation, and result formatting.
        
        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto
            
        Returns:
            EmbeddingResult with complete embedding outcome
        """
        start_time = time.time()
        self.logger.info(f"Starting embedding of VNR {vnr.vnr_id}")
        
        try:
            # Validate inputs
            self._validate_inputs(vnr, substrate)
            
            # Perform the embedding
            result = self._embed_single_vnr(vnr, substrate)
            
            # Measure execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.timestamp = start_time
            result.algorithm_name = self.name
            
            # Handle resource allocation if successful
            if result.success:
                success = self._allocate_resources(vnr, substrate, result)
                if not success:
                    # Resource allocation failed, mark as unsuccessful
                    result.success = False
                    result.failure_reason = "Resource allocation failed"
                    result.revenue = 0.0
                    self.logger.warning(f"VNR {vnr.vnr_id} embedding failed during resource allocation")
                else:
                    # Calculate revenue and cost
                    result.revenue = self._calculate_revenue(vnr, result)
                    result.cost = self._calculate_cost(vnr, result)
                    self.logger.info(f"VNR {vnr.vnr_id} embedded successfully: "
                                   f"revenue={result.revenue:.2f}, cost={result.cost:.2f}")
            else:
                result.revenue = 0.0
                result.cost = self._calculate_cost(vnr, result)
                self.logger.info(f"VNR {vnr.vnr_id} embedding failed: {result.failure_reason}")
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Exception during VNR {vnr.vnr_id} embedding: {e}")
            
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=False,
                node_mapping={},
                link_mapping={},
                revenue=0.0,
                cost=0.0,
                execution_time=execution_time,
                failure_reason=f"Algorithm exception: {str(e)}",
                timestamp=start_time,
                algorithm_name=self.name
            )
    
    def embed_batch(self, vnrs: List[VirtualNetworkRequest], 
                   substrate: SubstrateNetwork) -> List[EmbeddingResult]:
        """
        Embed a batch of VNRs in sequence.
        
        This method processes a list of VNRs one by one, maintaining the substrate
        network state between embeddings. VNRs are processed in the order provided.
        
        Args:
            vnrs: List of VNRs to embed
            substrate: Substrate network to embed onto
            
        Returns:
            List of EmbeddingResult objects for each VNR
        """
        self.logger.info(f"Starting batch embedding of {len(vnrs)} VNRs")
        results = []
        
        for i, vnr in enumerate(vnrs):
            self.logger.debug(f"Processing VNR {i+1}/{len(vnrs)}: {vnr.vnr_id}")
            result = self.embed_vnr(vnr, substrate)
            results.append(result)
            
            # Log progress periodically
            if (i + 1) % 10 == 0 or (i + 1) == len(vnrs):
                success_count = sum(1 for r in results if r.success)
                self.logger.info(f"Batch progress: {i+1}/{len(vnrs)} processed, "
                               f"{success_count} successful ({success_count/(i+1)*100:.1f}%)")
        
        self.logger.info(f"Batch embedding completed: {len(results)} results generated")
        return results
    
    def embed_online(self, vnrs: List[VirtualNetworkRequest], 
                    substrate: SubstrateNetwork,
                    simulate_time: bool = True) -> List[EmbeddingResult]:
        """
        Embed VNRs in online mode with arrival times and lifetimes.
        
        This method simulates the online VNE scenario where VNRs arrive over time
        and depart after their lifetime expires. Resources are deallocated when
        VNRs depart.
        
        Args:
            vnrs: List of VNRs to embed (should be sorted by arrival time)
            substrate: Substrate network to embed onto
            simulate_time: Whether to simulate actual time passage
            
        Returns:
            List of EmbeddingResult objects for each VNR
        """
        self.logger.info(f"Starting online embedding of {len(vnrs)} VNRs")
        
        # Sort VNRs by arrival time
        sorted_vnrs = sorted(vnrs, key=lambda v: v.arrival_time)
        results = []
        active_vnrs = []  # Track currently active VNRs for resource deallocation
        
        current_time = 0.0
        
        for i, vnr in enumerate(sorted_vnrs):
            # Advance time to VNR arrival
            current_time = vnr.arrival_time
            
            # Remove expired VNRs and deallocate their resources
            active_vnrs = self._process_departures(active_vnrs, results, substrate, current_time)
            
            self.logger.debug(f"Processing VNR {i+1}/{len(vnrs)}: {vnr.vnr_id} "
                            f"at time {current_time:.2f}")
            
            # Attempt to embed the new VNR
            result = self.embed_vnr(vnr, substrate)
            results.append(result)
            
            # If successful, add to active VNRs for future deallocation
            if result.success:
                active_vnrs.append((vnr, result))
            
            # Log progress periodically
            if (i + 1) % 10 == 0 or (i + 1) == len(vnrs):
                success_count = sum(1 for r in results if r.success)
                self.logger.info(f"Online progress: {i+1}/{len(vnrs)} processed, "
                               f"{success_count} successful, {len(active_vnrs)} active")
        
        # Final cleanup - deallocate all remaining active VNRs
        for vnr, result in active_vnrs:
            self._deallocate_resources(vnr, substrate, result)
        
        self.logger.info(f"Online embedding completed: {len(results)} results generated")
        return results
    
    def _validate_inputs(self, vnr: VirtualNetworkRequest, 
                        substrate: SubstrateNetwork) -> None:
        """
        Validate VNR and substrate network inputs.
        
        Args:
            vnr: VNR to validate
            substrate: Substrate network to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not vnr.virtual_nodes:
            raise ValueError(f"VNR {vnr.vnr_id} has no virtual nodes")
        
        if len(substrate.graph.nodes) == 0:
            raise ValueError("Substrate network has no nodes")
        
        # Check if VNR is potentially feasible
        feasible, issues = vnr.check_feasibility(substrate)
        if not feasible:
            self.logger.warning(f"VNR {vnr.vnr_id} may not be feasible: {issues}")
    
    def _allocate_resources(self, vnr: VirtualNetworkRequest, 
                          substrate: SubstrateNetwork,
                          result: EmbeddingResult) -> bool:
        """
        Allocate substrate resources based on embedding result.
        
        Args:
            vnr: VNR being embedded
            substrate: Substrate network
            result: Embedding result with mappings
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self._lock:
            try:
                # Allocate node resources
                for virtual_node_id, node_req in vnr.virtual_nodes.items():
                    substrate_node_id = result.node_mapping[str(virtual_node_id)]
                    
                    success = substrate.allocate_node_resources(
                        int(substrate_node_id),
                        node_req.cpu_requirement,
                        node_req.memory_requirement
                    )
                    
                    if not success:
                        # Rollback previous allocations
                        self._partial_rollback_nodes(vnr, substrate, result, virtual_node_id)
                        return False
                
                # Allocate link resources
                for (src_virtual, dst_virtual), link_req in vnr.virtual_links.items():
                    substrate_path = result.link_mapping[(str(src_virtual), str(dst_virtual))]
                    
                    # Allocate bandwidth on each link in the path
                    for i in range(len(substrate_path) - 1):
                        src_substrate = int(substrate_path[i])
                        dst_substrate = int(substrate_path[i + 1])
                        
                        success = substrate.allocate_link_resources(
                            src_substrate,
                            dst_substrate,
                            link_req.bandwidth_requirement
                        )
                        
                        if not success:
                            # Rollback previous allocations
                            self._partial_rollback_links(vnr, substrate, result, 
                                                       (src_virtual, dst_virtual), i)
                            self._deallocate_all_nodes(vnr, substrate, result)
                            return False
                
                return True
                
            except Exception as e:
                self.logger.error(f"Resource allocation error: {e}")
                # Attempt to rollback any partial allocations
                self._deallocate_resources(vnr, substrate, result)
                return False
    
    def _deallocate_resources(self, vnr: VirtualNetworkRequest, 
                            substrate: SubstrateNetwork,
                            result: EmbeddingResult) -> None:
        """
        Deallocate substrate resources for a VNR.
        
        Args:
            vnr: VNR being deallocated
            substrate: Substrate network
            result: Embedding result with mappings
        """
        with self._lock:
            try:
                # Deallocate node resources
                for virtual_node_id, node_req in vnr.virtual_nodes.items():
                    if str(virtual_node_id) in result.node_mapping:
                        substrate_node_id = result.node_mapping[str(virtual_node_id)]
                        substrate.deallocate_node_resources(
                            int(substrate_node_id),
                            node_req.cpu_requirement,
                            node_req.memory_requirement
                        )
                
                # Deallocate link resources
                for (src_virtual, dst_virtual), link_req in vnr.virtual_links.items():
                    if (str(src_virtual), str(dst_virtual)) in result.link_mapping:
                        substrate_path = result.link_mapping[(str(src_virtual), str(dst_virtual))]
                        
                        # Deallocate bandwidth on each link in the path
                        for i in range(len(substrate_path) - 1):
                            src_substrate = int(substrate_path[i])
                            dst_substrate = int(substrate_path[i + 1])
                            substrate.deallocate_link_resources(
                                src_substrate,
                                dst_substrate,
                                link_req.bandwidth_requirement
                            )
                            
            except Exception as e:
                self.logger.error(f"Resource deallocation error: {e}")
    
    def _partial_rollback_nodes(self, vnr: VirtualNetworkRequest,
                              substrate: SubstrateNetwork,
                              result: EmbeddingResult,
                              failed_node_id: str) -> None:
        """Rollback partial node resource allocations."""
        for virtual_node_id, node_req in vnr.virtual_nodes.items():
            if str(virtual_node_id) == failed_node_id:
                break
            substrate_node_id = result.node_mapping[str(virtual_node_id)]
            substrate.deallocate_node_resources(
                int(substrate_node_id),
                node_req.cpu_requirement,
                node_req.memory_requirement
            )
    
    def _partial_rollback_links(self, vnr: VirtualNetworkRequest,
                              substrate: SubstrateNetwork,
                              result: EmbeddingResult,
                              failed_link: Tuple[str, str],
                              failed_hop: int) -> None:
        """Rollback partial link resource allocations."""
        for (src_virtual, dst_virtual), link_req in vnr.virtual_links.items():
            if (src_virtual, dst_virtual) == failed_link:
                # Rollback partial path allocation
                substrate_path = result.link_mapping[(str(src_virtual), str(dst_virtual))]
                for i in range(failed_hop):
                    src_substrate = int(substrate_path[i])
                    dst_substrate = int(substrate_path[i + 1])
                    substrate.deallocate_link_resources(
                        src_substrate, dst_substrate, link_req.bandwidth_requirement
                    )
                break
            else:
                # Rollback complete previous link allocations
                substrate_path = result.link_mapping[(str(src_virtual), str(dst_virtual))]
                for i in range(len(substrate_path) - 1):
                    src_substrate = int(substrate_path[i])
                    dst_substrate = int(substrate_path[i + 1])
                    substrate.deallocate_link_resources(
                        src_substrate, dst_substrate, link_req.bandwidth_requirement
                    )
    
    def _deallocate_all_nodes(self, vnr: VirtualNetworkRequest,
                            substrate: SubstrateNetwork,
                            result: EmbeddingResult) -> None:
        """Deallocate all node resources for rollback."""
        for virtual_node_id, node_req in vnr.virtual_nodes.items():
            substrate_node_id = result.node_mapping[str(virtual_node_id)]
            substrate.deallocate_node_resources(
                int(substrate_node_id),
                node_req.cpu_requirement,
                node_req.memory_requirement
            )
    
    def _process_departures(self, active_vnrs: List[Tuple[VirtualNetworkRequest, EmbeddingResult]],
                          results: List[EmbeddingResult],
                          substrate: SubstrateNetwork,
                          current_time: float) -> List[Tuple[VirtualNetworkRequest, EmbeddingResult]]:
        """
        Process VNR departures and deallocate resources.
        
        Args:
            active_vnrs: List of currently active VNRs
            results: List of all results (for logging)
            substrate: Substrate network
            current_time: Current simulation time
            
        Returns:
            Updated list of active VNRs
        """
        still_active = []
        
        for vnr, result in active_vnrs:
            departure_time = vnr.get_departure_time()
            
            if current_time >= departure_time:
                # VNR has expired, deallocate resources
                self._deallocate_resources(vnr, substrate, result)
                self.logger.debug(f"VNR {vnr.vnr_id} departed at time {current_time:.2f}")
            else:
                # VNR is still active
                still_active.append((vnr, result))
        
        return still_active
    
    def _calculate_revenue(self, vnr: VirtualNetworkRequest, 
                         result: EmbeddingResult) -> float:
        """
        Calculate revenue for successful VNR embedding.
        
        Args:
            vnr: VNR that was embedded
            result: Embedding result
            
        Returns:
            Revenue value
        """
        # Use the VNR's built-in revenue calculation if available
        if hasattr(vnr, 'calculate_revenue'):
            return vnr.calculate_revenue()
        
        # Simple revenue calculation based on resource requirements
        requirements = vnr.calculate_total_requirements()
        base_revenue = (
            requirements['total_cpu'] * 1.0 +
            requirements['total_memory'] * 1.0 +
            requirements['total_bandwidth'] * 1.0
        )
        
        # Consider lifetime for revenue calculation
        if vnr.lifetime != float('inf'):
            return base_revenue * vnr.lifetime / 1000.0  # Normalize by expected lifetime
        else:
            return base_revenue
    
    def _calculate_cost(self, vnr: VirtualNetworkRequest, 
                       result: EmbeddingResult) -> float:
        """
        Calculate cost for VNR embedding attempt.
        
        Args:
            vnr: VNR that was processed
            result: Embedding result
            
        Returns:
            Cost value
        """
        if not result.success:
            # Cost for failed embedding (computational cost)
            return 1.0
        
        # Cost based on substrate resources used
        node_cost = len(result.node_mapping) * 10.0  # Cost per node used
        
        # Cost based on total path length
        total_path_length = sum(
            len(path) - 1 for path in result.link_mapping.values()
        )
        link_cost = total_path_length * 5.0  # Cost per hop
        
        return node_cost + link_cost
    
    def _update_stats(self, result: EmbeddingResult) -> None:
        """Update internal algorithm statistics."""
        with self._lock:
            self._stats['total_requests'] += 1
            if result.success:
                self._stats['successful_requests'] += 1
                self._stats['total_revenue'] += result.revenue
            self._stats['total_execution_time'] += result.execution_time
            self._stats['total_cost'] += result.cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get algorithm performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            stats = self._stats.copy()
            
            if stats['total_requests'] > 0:
                stats['acceptance_ratio'] = stats['successful_requests'] / stats['total_requests']
                stats['average_execution_time'] = stats['total_execution_time'] / stats['total_requests']
            else:
                stats['acceptance_ratio'] = 0.0
                stats['average_execution_time'] = 0.0
            
            if stats['total_cost'] > 0:
                stats['revenue_to_cost_ratio'] = stats['total_revenue'] / stats['total_cost']
            else:
                stats['revenue_to_cost_ratio'] = 0.0
            
            return stats
    
    def reset_statistics(self) -> None:
        """Reset algorithm statistics."""
        with self._lock:
            self._stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_execution_time': 0.0,
                'total_revenue': 0.0,
                'total_cost': 0.0
            }
            self.logger.info(f"{self.name} statistics reset")
    
    def __str__(self) -> str:
        """String representation of the algorithm."""
        return f"{self.name} (requests: {self._stats['total_requests']}, " \
               f"success: {self._stats['successful_requests']})"
    
    def __repr__(self) -> str:
        """Detailed representation of the algorithm."""
        return f"{self.__class__.__name__}(name='{self.name}', " \
               f"parameters={self.parameters})"