"""
Base algorithm interface for Virtual Network Embedding (VNE) algorithms.

This module provides the abstract base class that all VNE algorithms must inherit from,
following VNE literature standards and best practices.
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork
from src.utils.metrics import (
    calculate_vnr_revenue,
    calculate_vnr_cost,
    generate_comprehensive_metrics_summary
)


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
    node_mapping: Dict[str, str]  # virtual_node_id -> substrate_node_id
    link_mapping: Dict[Tuple[str, str], List[str]]  # (v_src, v_dst) -> substrate_path
    revenue: float
    cost: float
    execution_time: float
    failure_reason: Optional[str] = None
    timestamp: Optional[float] = None
    algorithm_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_metrics_result(self):
        """
        Convert to MetricsEmbeddingResult for compatibility with metrics utilities.

        Returns:
            MetricsEmbeddingResult instance
        """
        from src.utils.metrics import EmbeddingResult as MetricsEmbeddingResult
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


class VNEConstraintError(Exception):
    """Exception raised when VNE constraints are violated."""
    pass


class BaseAlgorithm(ABC):
    """
    Abstract base class for all VNE algorithms following literature standards.

    This class provides the standard interface that all VNE algorithms must implement,
    along with common functionality for metrics collection, logging, and constraint
    validation following VNE research literature.

    Key VNE Literature Principles:
    1. Algorithms handle resource allocation during embedding
    2. Intra-VNR separation enforced as fundamental constraint
    3. Standard VNE metrics calculation using literature formulas
    4. Online simulation with proper temporal handling
    5. Batch processing for statistical analysis

    Subclasses must implement:
    - _embed_single_vnr(): Core embedding logic with resource allocation
    - _cleanup_failed_embedding(): Rollback mechanism for failed embeddings

    Base class provides:
    - Standard VNE workflow and constraint validation
    - Metrics calculation using standard formulas
    - Batch and online processing capabilities
    - Statistics tracking and logging

    Attributes:
        name: Human-readable name of the algorithm
        logger: Logger instance for the algorithm
        parameters: Algorithm-specific parameters

    Example:
        >>> class MyAlgorithm(BaseAlgorithm):
        ...     def __init__(self):
        ...         super().__init__("My Algorithm")
        ...
        ...     def _embed_single_vnr(self, vnr, substrate):
        ...         # Implementation with resource allocation
        ...         return EmbeddingResult(...)
        ...
        ...     def _cleanup_failed_embedding(self, vnr, substrate, result):
        ...         # Cleanup allocated resources
        ...         pass
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
        self.parameters = kwargs

        # Thread-safe statistics tracking
        self._stats_lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_execution_time': 0.0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'constraint_violations': 0
        }

        self.logger.info(f"Initialized {self.name} algorithm with parameters: {kwargs}")

    def embed_vnr(self, vnr: VirtualNetworkRequest,
                  substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Embed a single VNR following standard VNE literature workflow.

        Standard VNE Literature Flow:
        1. Pre-embedding validation (VNR-substrate compatibility)
        2. Algorithm-specific embedding (with resource allocation)
        3. Post-embedding validation (VNE constraints)
        4. Standard metrics calculation

        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto

        Returns:
            EmbeddingResult with complete embedding outcome
        """
        start_time = time.time()
        self.logger.info(f"Starting VNE embedding: VNR {vnr.vnr_id}")

        try:
            # Phase 1: Pre-embedding validation
            self._validate_vnr_substrate_compatibility(vnr, substrate)

            # Phase 2: Algorithm-specific embedding (handles resource allocation)
            result = self._embed_single_vnr(vnr, substrate)

            # Phase 3: Post-embedding VNE constraint validation
            if result.success:
                constraint_violations = self._validate_vne_constraints(vnr, substrate, result)
                if constraint_violations:
                    # Critical VNE constraint violated - must fail embedding
                    self.logger.warning(f"VNR {vnr.vnr_id} violates VNE constraints: {constraint_violations}")

                    # Algorithm must clean up its own allocations
                    self._request_algorithm_cleanup(vnr, substrate, result)

                    result.success = False
                    result.failure_reason = f"VNE constraint violation: {'; '.join(constraint_violations)}"
                    result.node_mapping = {}
                    result.link_mapping = {}

                    with self._stats_lock:
                        self._stats['constraint_violations'] += 1

            # Phase 4: Standard VNE metrics calculation (literature formulas)
            result.execution_time = time.time() - start_time
            result.timestamp = start_time
            result.algorithm_name = self.name

            if result.success:
                # Use standard VNE literature formulas from metrics module
                result.revenue = calculate_vnr_revenue(vnr)
                result.cost = calculate_vnr_cost(vnr, result.node_mapping, result.link_mapping, substrate)

                self.logger.info(f"VNR {vnr.vnr_id} embedded successfully: "
                               f"revenue={result.revenue:.2f}, cost={result.cost:.2f}")
            else:
                result.revenue = 0.0
                result.cost = self._calculate_failure_cost(vnr)

                self.logger.info(f"VNR {vnr.vnr_id} embedding failed: {result.failure_reason}")

            # Update statistics
            self._update_statistics(result)

            return result

        except Exception as e:
            # Handle unexpected exceptions
            execution_time = time.time() - start_time
            self.logger.error(f"Unexpected error embedding VNR {vnr.vnr_id}: {e}")

            error_result = EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=False,
                node_mapping={},
                link_mapping={},
                revenue=0.0,
                cost=self._calculate_failure_cost(vnr),
                execution_time=execution_time,
                failure_reason=f"Algorithm exception: {str(e)}",
                timestamp=start_time,
                algorithm_name=self.name
            )

            self._update_statistics(error_result)
            return error_result

    @abstractmethod
    def _embed_single_vnr(self, vnr: VirtualNetworkRequest,
                         substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Core algorithm-specific embedding logic.

        CRITICAL: Algorithm must handle resource allocation during embedding.

        Algorithm Responsibilities:
        1. Check resource availability during embedding process
        2. Allocate resources on successful mapping
        3. Provide rollback mechanism via _cleanup_failed_embedding()
        4. Return EmbeddingResult with mappings and success status

        VNE Literature Standard:
        - Algorithms decide their own resource allocation strategy
        - Resource allocation happens DURING embedding, not after
        - Failed embeddings should not leave resources allocated

        Args:
            vnr: VNR to embed
            substrate: Substrate network

        Returns:
            EmbeddingResult with success status and mappings

        Note:
            Base class will validate VNE constraints (like Intra-VNR separation)
            and call _cleanup_failed_embedding() if constraints are violated.
        """
        pass

    @abstractmethod
    def _cleanup_failed_embedding(self, vnr: VirtualNetworkRequest,
                                 substrate: SubstrateNetwork,
                                 result: EmbeddingResult) -> None:
        """
        Clean up resources for failed embedding.

        Called by base class when:
        - VNE constraints are violated (e.g., Intra-VNR separation)
        - Algorithm needs to rollback partial allocations

        Algorithm Responsibilities:
        - Deallocate any resources allocated during _embed_single_vnr()
        - Reset substrate state to pre-embedding condition
        - Handle partial allocation rollback

        Args:
            vnr: VNR that failed embedding
            substrate: Substrate network to clean up
            result: Embedding result with current mappings
        """
        pass

    def embed_batch(self, vnrs: List[VirtualNetworkRequest],
                   substrate: SubstrateNetwork) -> List[EmbeddingResult]:
        """
        Embed a batch of VNRs sequentially for statistical analysis.

        Standard VNE batch processing following literature patterns.
        Used for algorithm evaluation and comparison.

        Args:
            vnrs: List of VNRs to embed
            substrate: Substrate network

        Returns:
            List of EmbeddingResult objects
        """
        self.logger.info(f"Starting batch embedding: {len(vnrs)} VNRs")

        results = []
        for i, vnr in enumerate(vnrs):
            result = self.embed_vnr(vnr, substrate)
            results.append(result)

            # Progress logging for large batches
            if (i + 1) % 10 == 0 or (i + 1) == len(vnrs):
                success_count = sum(1 for r in results if r.success)
                acceptance_ratio = success_count / (i + 1)
                self.logger.info(f"Batch progress: {i+1}/{len(vnrs)}, "
                               f"AR={acceptance_ratio:.3f}")

        self.logger.info(f"Batch embedding completed: {len(results)} results")
        return results

    def embed_online(self, vnrs: List[VirtualNetworkRequest],
                    substrate: SubstrateNetwork,
                    simulation_duration: Optional[float] = None) -> List[EmbeddingResult]:
        """
        Online VNE simulation with temporal constraints.

        Standard online VNE following literature:
        1. VNRs arrive according to arrival_time
        2. Resources allocated for holding_time duration
        3. Resources automatically deallocated on departure
        4. Supports time windows and advanced temporal constraints

        Args:
            vnrs: List of VNRs (will be sorted by arrival_time)
            substrate: Substrate network
            simulation_duration: Optional maximum simulation time

        Returns:
            List of EmbeddingResult objects for all VNRs
        """
        self.logger.info(f"Starting online VNE simulation: {len(vnrs)} VNRs")

        # Sort VNRs by arrival time (standard online VNE)
        sorted_vnrs = sorted(vnrs, key=lambda v: v.arrival_time)

        results = []
        active_embeddings = []  # (vnr, result, departure_time)
        current_time = 0.0

        for vnr in sorted_vnrs:
            # Advance simulation time to VNR arrival
            current_time = vnr.arrival_time

            # Check simulation duration limit
            if simulation_duration and current_time > simulation_duration:
                self.logger.info(f"Simulation duration exceeded at time {current_time}")
                break

            # Process VNR departures
            active_embeddings = self._process_vnr_departures(
                active_embeddings, substrate, current_time
            )

            # Attempt to embed arriving VNR
            self.logger.debug(f"Processing VNR {vnr.vnr_id} at time {current_time}")
            result = self.embed_vnr(vnr, substrate)
            results.append(result)

            # Track successful embeddings for later departure
            if result.success:
                departure_time = vnr.arrival_time + vnr.holding_time
                if vnr.holding_time != float('inf'):
                    active_embeddings.append((vnr, result, departure_time))
                # Note: Infinite holding time VNRs never depart

        # Final cleanup: deallocate all remaining active VNRs
        for vnr, result, _ in active_embeddings:
            self.logger.debug(f"Final cleanup: deallocating VNR {vnr.vnr_id}")
            self._cleanup_failed_embedding(vnr, substrate, result)

        self.logger.info(f"Online simulation completed: {len(results)} results")
        return results

    def calculate_metrics(self, results: List[EmbeddingResult],
                         substrate: Optional[SubstrateNetwork] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive VNE metrics using standard literature formulas.

        Uses the metrics module for standard calculations.

        Args:
            results: List of embedding results
            substrate: Optional substrate for utilization metrics

        Returns:
            Comprehensive metrics dictionary
        """
        # Convert to metrics module format for compatibility
        metrics_results = []
        for result in results:
            metrics_result = result.to_metrics_result()
            metrics_results.append(metrics_result)

        # Use existing comprehensive metrics calculation
        return generate_comprehensive_metrics_summary(metrics_results, substrate)

    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """
        Get algorithm-specific performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()

            # Calculate derived metrics
            if stats['total_requests'] > 0:
                stats['acceptance_ratio'] = stats['successful_requests'] / stats['total_requests']
                stats['average_execution_time'] = stats['total_execution_time'] / stats['total_requests']
                stats['constraint_violation_rate'] = stats['constraint_violations'] / stats['total_requests']
            else:
                stats['acceptance_ratio'] = 0.0
                stats['average_execution_time'] = 0.0
                stats['constraint_violation_rate'] = 0.0

            if stats['total_cost'] > 0:
                stats['revenue_to_cost_ratio'] = stats['total_revenue'] / stats['total_cost']
            else:
                stats['revenue_to_cost_ratio'] = 0.0

            # Add algorithm-specific info
            stats['algorithm_name'] = self.name
            stats['algorithm_parameters'] = self.parameters

            return stats

    def reset_statistics(self) -> None:
        """Reset algorithm statistics."""
        with self._stats_lock:
            self._stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_execution_time': 0.0,
                'total_revenue': 0.0,
                'total_cost': 0.0,
                'constraint_violations': 0
            }

        self.logger.info(f"Reset statistics for {self.name}")

    # =========================================================================
    # VNE CONSTRAINT VALIDATION (Literature Standards)
    # =========================================================================

    def _validate_vnr_substrate_compatibility(self, vnr: VirtualNetworkRequest,
                                            substrate: SubstrateNetwork) -> None:
        """
        Validate VNR-substrate compatibility before embedding.

        Checks basic feasibility conditions from VNE literature.

        Raises:
            VNEConstraintError: If fundamental incompatibility detected
        """
        # Check basic requirements
        if len(vnr.virtual_nodes) == 0:
            raise VNEConstraintError("VNR has no virtual nodes")

        if len(substrate.graph.nodes) == 0:
            raise VNEConstraintError("Substrate network has no nodes")

        # Check Intra-VNR separation feasibility
        if len(vnr.virtual_nodes) > len(substrate.graph.nodes):
            raise VNEConstraintError(
                f"VNR requires {len(vnr.virtual_nodes)} nodes but substrate only has "
                f"{len(substrate.graph.nodes)} nodes - Intra-VNR separation impossible"
            )

        # Check constraint compatibility
        substrate_constraints = substrate.get_constraint_configuration()
        vnr_constraints = vnr.get_constraint_summary()

        # Warn about unsupported constraint usage
        if (vnr_constraints['uses_memory_constraints'] and
            not substrate_constraints['memory_constraints']):
            self.logger.warning(f"VNR {vnr.vnr_id} requires memory constraints "
                              "but substrate doesn't support them")

        if (vnr_constraints['uses_delay_constraints'] and
            not substrate_constraints['delay_constraints']):
            self.logger.warning(f"VNR {vnr.vnr_id} requires delay constraints "
                              "but substrate doesn't support them")

    def _validate_vne_constraints(self, vnr: VirtualNetworkRequest,
                                substrate: SubstrateNetwork,
                                result: EmbeddingResult) -> List[str]:
        """
        Validate fundamental VNE constraints after embedding.

        Critical VNE Literature Constraints:
        1. Intra-VNR separation (most important)
        2. Resource capacity constraints
        3. Network connectivity constraints

        Args:
            vnr: VNR that was embedded
            substrate: Substrate network
            result: Embedding result to validate

        Returns:
            List of constraint violations (empty if valid)
        """
        violations = []

        # CRITICAL: Intra-VNR separation constraint
        if not self._check_intra_vnr_separation(result.node_mapping):
            violations.append("Intra-VNR separation violated - multiple virtual nodes mapped to same substrate node")

        # Validate node mapping completeness
        if len(result.node_mapping) != len(vnr.virtual_nodes):
            violations.append(f"Incomplete node mapping: {len(result.node_mapping)}/{len(vnr.virtual_nodes)} nodes mapped")

        # Validate link mapping completeness
        if len(result.link_mapping) != len(vnr.virtual_links):
            violations.append(f"Incomplete link mapping: {len(result.link_mapping)}/{len(vnr.virtual_links)} links mapped")

        # Validate mapped nodes exist in substrate
        for vnode, snode in result.node_mapping.items():
            if int(snode) not in substrate.graph.nodes:
                violations.append(f"Virtual node {vnode} mapped to non-existent substrate node {snode}")

        # Validate mapped paths exist in substrate
        for (vsrc, vdst), path in result.link_mapping.items():
            if len(path) > 1:  # Skip single-node paths
                for i in range(len(path) - 1):
                    src, dst = int(path[i]), int(path[i + 1])
                    if not substrate.graph.has_edge(src, dst):
                        violations.append(f"Virtual link ({vsrc}, {vdst}) uses non-existent substrate link ({src}, {dst})")

        return violations

    @staticmethod
    def _check_intra_vnr_separation(node_mapping: Dict[str, str]) -> bool:
        """
        Check Intra-VNR separation constraint.

        VNE Literature Requirement:
        No two virtual nodes from the same VNR can be mapped to the same substrate node.

        Args:
            node_mapping: Virtual node to substrate node mapping

        Returns:
            True if constraint satisfied, False otherwise
        """
        substrate_nodes = list(node_mapping.values())
        unique_substrate_nodes = set(substrate_nodes)

        # If all mapped substrate nodes are unique, constraint is satisfied
        return len(substrate_nodes) == len(unique_substrate_nodes)

    def _process_vnr_departures(self, active_embeddings: List[Tuple],
                               substrate: SubstrateNetwork,
                               current_time: float) -> List[Tuple]:
        """
        Process VNR departures for online simulation.

        Args:
            active_embeddings: List of (vnr, result, departure_time) tuples
            substrate: Substrate network
            current_time: Current simulation time

        Returns:
            Updated list with departed VNRs removed
        """
        still_active = []

        for vnr, result, departure_time in active_embeddings:
            if current_time >= departure_time:
                # VNR has departed - deallocate resources
                self.logger.debug(f"VNR {vnr.vnr_id} departing at time {current_time}")
                self._cleanup_failed_embedding(vnr, substrate, result)
            else:
                # VNR still active
                still_active.append((vnr, result, departure_time))

        return still_active

    def _request_algorithm_cleanup(self, vnr: VirtualNetworkRequest,
                                 substrate: SubstrateNetwork,
                                 result: EmbeddingResult) -> None:
        """
        Request algorithm to clean up failed embedding.

        Called when VNE constraints are violated after successful algorithm embedding.
        """
        try:
            self._cleanup_failed_embedding(vnr, substrate, result)
        except Exception as e:
            self.logger.error(f"Algorithm cleanup failed for VNR {vnr.vnr_id}: {e}")

    def _calculate_failure_cost(self, vnr: VirtualNetworkRequest) -> float:
        """
        Calculate computational cost for failed embedding.

        Simple cost model based on VNR complexity.
        """
        return len(vnr.virtual_nodes) * 0.1 + len(vnr.virtual_links) * 0.05

    def _update_statistics(self, result: EmbeddingResult) -> None:
        """Update algorithm statistics thread-safely."""
        with self._stats_lock:
            self._stats['total_requests'] += 1
            self._stats['total_execution_time'] += result.execution_time
            self._stats['total_cost'] += result.cost

            if result.success:
                self._stats['successful_requests'] += 1
                self._stats['total_revenue'] += result.revenue

    def __str__(self) -> str:
        """String representation of algorithm."""
        stats = self.get_algorithm_statistics()
        return (f"{self.name} (requests: {stats['total_requests']}, "
               f"AR: {stats['acceptance_ratio']:.3f})")

    def __repr__(self) -> str:
        """Detailed representation of algorithm."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


# =============================================================================
# EXPORT FOR CLEAN MODULE INTERFACE
# =============================================================================

__all__ = [
    'BaseAlgorithm',
    'EmbeddingResult',
    'VNEConstraintError'
]
