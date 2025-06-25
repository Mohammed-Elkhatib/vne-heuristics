"""
Yu et al. (2008) Two-Stage Virtual Network Embedding Algorithm.

Literature-compliant implementation of the seminal two-stage VNE algorithm
from Yu et al.'s 2008 paper "Rethinking Virtual Network Embedding: Substrate
Support for Path Splitting and Migration".

This implementation follows the exact algorithm description from the paper
and uses only primary constraints (CPU + Bandwidth) as specified.

Reference:
    Yu, M., Yi, Y., Rexford, J., & Chiang, M. (2008). Rethinking virtual network
    embedding: substrate support for path splitting and migration. ACM SIGCOMM
    Computer Communication Review, 38(2), 17-29.
"""

import logging
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass

from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork


logger = logging.getLogger(__name__)


@dataclass
class NodeRankingInfo:
    """Information for ranking virtual nodes following Yu 2008."""
    node_id: int
    cpu_requirement: float
    total_requirement: float  # CPU only in Yu 2008
    degree: int


@dataclass
class CandidateNodeInfo:
    """Information about candidate substrate nodes."""
    node_id: int
    cpu_available: float
    utilization: float


@dataclass
class PathInfo:
    """Information about candidate paths for link mapping."""
    path: List[int]
    hop_count: int
    min_bandwidth: float


class YuAlgorithm(BaseAlgorithm):
    """
    Yu et al. (2008) Two-Stage Virtual Network Embedding Algorithm.

    Literature-Compliant Implementation Features:
    1. Two-stage approach: Node mapping then link mapping
    2. Node ranking by resource requirements (CPU) and connectivity
    3. Node selection using load balancing strategy
    4. K-shortest path link mapping with bandwidth constraints
    5. Resource allocation during embedding (not after)
    6. Only uses primary constraints (CPU + Bandwidth)

    Algorithm Flow (Yu 2008):
    Stage 1 - Node Mapping:
        1. Rank virtual nodes by CPU requirements (decreasing order)
        2. For each virtual node, find substrate nodes with sufficient CPU
        3. Select substrate node with highest available CPU (load balancing)
        4. Allocate CPU resources immediately

    Stage 2 - Link Mapping:
        1. For each virtual link, find k-shortest paths between mapped nodes
        2. Select path with sufficient bandwidth
        3. Allocate bandwidth resources on selected path
        4. If no suitable path found, fail entire embedding

    Attributes:
        k_paths: Number of shortest paths to consider for link mapping
        path_selection_strategy: Strategy for selecting among k paths
        enable_path_caching: Whether to cache shortest paths for performance

    Example:
        >>> algorithm = YuAlgorithm(k_paths=3)
        >>> result = algorithm.embed_vnr(vnr, substrate)
        >>> print(f"Embedding success: {result.success}")
    """

    def __init__(self, k_paths: int = 1,
                 path_selection_strategy: str = "shortest",
                 enable_path_caching: bool = True, **kwargs):
        """
        Initialize Yu algorithm with specified parameters.

        Args:
            k_paths: Number of shortest paths to consider (default: 1)
            path_selection_strategy: Path selection strategy
                ("shortest", "bandwidth") (default: "shortest")
            enable_path_caching: Enable path caching for performance (default: True)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__("Yu et al. (2008) Two-Stage Algorithm", **kwargs)

        # Algorithm parameters
        self.k_paths = max(1, k_paths)
        self.path_selection_strategy = path_selection_strategy
        self.enable_path_caching = enable_path_caching

        # Path cache for performance optimization
        self._path_cache: Dict[Tuple[int, int], List[PathInfo]] = {}

        # Validate parameters
        self._validate_parameters()

        # Warn about non-Yu 2008 usage
        self._validate_yu2008_compliance()

        self.logger.info(f"Initialized Yu algorithm: k_paths={k_paths}, "
                        f"strategy={path_selection_strategy}, caching={enable_path_caching}")

    def _validate_parameters(self) -> None:
        """Validate algorithm parameters."""
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")

        valid_strategies = ["shortest", "bandwidth"]
        if self.path_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid path selection strategy. Must be one of: {valid_strategies}")

    def _validate_yu2008_compliance(self) -> None:
        """Warn about deviations from Yu 2008 paper."""
        # Yu 2008 uses only primary constraints (CPU + Bandwidth)
        if any(param in self.parameters for param in
               ['enable_memory', 'enable_delay', 'enable_reliability']):
            self.logger.warning("Yu 2008 algorithm only uses CPU and Bandwidth constraints. "
                              "Secondary constraints will be ignored for literature compliance.")

    def _embed_single_vnr(self, vnr: VirtualNetworkRequest,
                         substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Yu 2008 Two-Stage Embedding with proper resource management.

        Implements the exact algorithm from the Yu 2008 paper:
        1. Stage 1: Node mapping with immediate resource allocation
        2. Stage 2: Link mapping with immediate bandwidth allocation
        3. Rollback on any failure

        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto

        Returns:
            EmbeddingResult with the outcome of the embedding attempt
        """
        self.logger.debug(f"Starting Yu 2008 two-stage embedding for VNR {vnr.vnr_id}")

        try:
            # Stage 1: Node Mapping with Resource Allocation
            node_mapping, allocated_nodes = self._node_mapping_stage_yu2008(vnr, substrate)

            if not node_mapping:
                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id),
                    success=False,
                    node_mapping={},
                    link_mapping={},
                    revenue=0.0,
                    cost=0.0,
                    execution_time=0.0,
                    failure_reason="Node mapping failed - insufficient CPU resources"
                )

            self.logger.debug(f"Stage 1 completed: {len(node_mapping)} nodes mapped")

            # Stage 2: Link Mapping with Resource Allocation
            link_mapping, allocated_links = self._link_mapping_stage_yu2008(
                vnr, substrate, node_mapping
            )

            if not link_mapping and vnr.virtual_links:
                # Link mapping failed - rollback node allocations
                self._rollback_node_allocations(substrate, allocated_nodes)

                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id),
                    success=False,
                    node_mapping={},
                    link_mapping={},
                    revenue=0.0,
                    cost=0.0,
                    execution_time=0.0,
                    failure_reason="Link mapping failed - insufficient bandwidth or no paths"
                )

            self.logger.debug(f"Stage 2 completed: {len(link_mapping)} links mapped")

            # Successful embedding
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=True,
                node_mapping=node_mapping,
                link_mapping=link_mapping,
                revenue=0.0,  # Will be calculated by base class using standard formula
                cost=0.0,     # Will be calculated by base class using standard formula
                execution_time=0.0,  # Will be set by base class
                metadata={
                    'allocated_nodes': len(allocated_nodes),
                    'allocated_links': len(allocated_links),
                    'algorithm_stages': 2
                }
            )

        except Exception as e:
            self.logger.error(f"Exception during Yu 2008 embedding: {e}")
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=False,
                node_mapping={},
                link_mapping={},
                revenue=0.0,
                cost=0.0,
                execution_time=0.0,
                failure_reason=f"Algorithm exception: {str(e)}"
            )

    def _cleanup_embedding(self, vnr: VirtualNetworkRequest,
                                 substrate: SubstrateNetwork,
                                 result: EmbeddingResult) -> None:
        """
        Clean up resources for failed embedding (Yu 2008 rollback).

        Deallocates all resources that were allocated during the embedding process.
        This is called by the base class when VNE constraints are violated.

        Args:
            vnr: VNR that failed embedding
            substrate: Substrate network to clean up
            result: Embedding result with current mappings
        """
        self.logger.debug(f"Cleaning up failed embedding for VNR {vnr.vnr_id}")

        try:
            # Deallocate node resources
            for vnode_id, vnode_req in vnr.virtual_nodes.items():
                if str(vnode_id) in result.node_mapping:
                    snode_id = int(result.node_mapping[str(vnode_id)])
                    substrate.deallocate_node_resources(
                        snode_id,
                        vnode_req.cpu_requirement,
                        0.0  # Yu 2008: no memory
                    )

            # Deallocate link resources
            for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
                if (str(vsrc), str(vdst)) in result.link_mapping:
                    path = [int(n) for n in result.link_mapping[(str(vsrc), str(vdst))]]

                    # Deallocate bandwidth on each link in the path
                    for i in range(len(path) - 1):
                        substrate.deallocate_link_resources(
                            path[i],
                            path[i + 1],
                            vlink_req.bandwidth_requirement
                        )

            self.logger.debug(f"Cleanup completed for VNR {vnr.vnr_id}")

        except Exception as e:
            self.logger.error(f"Error during cleanup for VNR {vnr.vnr_id}: {e}")

    def _node_mapping_stage_yu2008(self, vnr: VirtualNetworkRequest,
                                  substrate: SubstrateNetwork) -> Tuple[Dict[str, str], List[Tuple[int, float]]]:
        """
        Stage 1: Node mapping with immediate resource allocation (Yu 2008).

        Yu 2008 Algorithm:
        1. Rank virtual nodes by CPU requirements (decreasing order)
        2. For each virtual node, find substrate nodes with sufficient CPU
        3. Select substrate node with highest available CPU (load balancing)
        4. Allocate CPU resources immediately
        5. If any allocation fails, rollback all and return failure

        Args:
            vnr: Virtual network request
            substrate: Substrate network

        Returns:
            Tuple of (node_mapping, allocated_nodes) where allocated_nodes tracks
            allocations for rollback: [(node_id, cpu_allocated), ...]
        """
        self.logger.debug("Starting Yu 2008 node mapping with allocation")

        # Step 1: Rank virtual nodes by CPU requirements (Yu 2008 standard)
        ranked_vnodes = self._rank_virtual_nodes_yu2008(vnr)

        node_mapping = {}
        allocated_nodes = []  # Track for rollback: [(node_id, cpu_allocated), ...]
        mapped_substrate_nodes = set()

        # Step 2: Map each virtual node in ranked order
        for vnode_info in ranked_vnodes:
            vnode_id = vnode_info.node_id
            vnode_req = vnr.virtual_nodes[vnode_id]

            self.logger.debug(f"Mapping virtual node {vnode_id} "
                            f"(CPU: {vnode_req.cpu_requirement}, degree: {vnode_info.degree})")

            # Find best substrate node using Yu 2008 strategy
            candidate_node = self._find_best_substrate_node_yu2008(
                substrate, vnode_req.cpu_requirement, mapped_substrate_nodes
            )

            if candidate_node is None:
                self.logger.debug(f"No suitable substrate node for virtual node {vnode_id}")
                # Rollback previous allocations
                self._rollback_node_allocations(substrate, allocated_nodes)
                return {}, []

            # Allocate CPU resources immediately (Yu 2008 approach)
            success = substrate.allocate_node_resources(
                candidate_node.node_id,
                vnode_req.cpu_requirement,
                0.0  # Yu 2008: only CPU, no memory
            )

            if not success:
                self.logger.debug(f"CPU allocation failed for node {candidate_node.node_id}")
                self._rollback_node_allocations(substrate, allocated_nodes)
                return {}, []

            # Record successful mapping and allocation
            node_mapping[str(vnode_id)] = str(candidate_node.node_id)
            allocated_nodes.append((candidate_node.node_id, vnode_req.cpu_requirement))
            mapped_substrate_nodes.add(candidate_node.node_id)

            self.logger.debug(f"Mapped virtual node {vnode_id} to substrate node {candidate_node.node_id}")

        return node_mapping, allocated_nodes

    def _rank_virtual_nodes_yu2008(self, vnr: VirtualNetworkRequest) -> List[NodeRankingInfo]:
        """
        Rank virtual nodes following Yu 2008 literature.

        Yu 2008 Ranking Strategy:
        1. Primary: CPU requirements (decreasing order)
        2. Secondary: Node degree/connectivity (decreasing order)

        Args:
            vnr: Virtual network request

        Returns:
            List of NodeRankingInfo sorted by Yu 2008 criteria
        """
        node_rankings = []

        for vnode_id, vnode_req in vnr.virtual_nodes.items():
            # Yu 2008: Only CPU requirements (primary constraint)
            cpu_requirement = vnode_req.cpu_requirement
            total_requirement = cpu_requirement  # Only CPU in Yu 2008

            # Secondary ranking: node degree (connectivity)
            degree = vnr.graph.degree(vnode_id) if vnode_id in vnr.graph else 0

            node_rankings.append(NodeRankingInfo(
                node_id=vnode_id,
                cpu_requirement=cpu_requirement,
                total_requirement=total_requirement,
                degree=degree
            ))

        # Sort by total requirement (decreasing), then by degree (decreasing)
        node_rankings.sort(key=lambda x: (x.total_requirement, x.degree), reverse=True)

        self.logger.debug(f"Virtual node ranking: {[(n.node_id, n.total_requirement, n.degree) for n in node_rankings]}")
        return node_rankings

    def _find_best_substrate_node_yu2008(self, substrate: SubstrateNetwork,
                                        cpu_requirement: float,
                                        excluded_nodes: Set[int]) -> Optional[CandidateNodeInfo]:
        """
        Find best substrate node following Yu 2008 selection strategy.

        Yu 2008 Selection Strategy:
        1. Find all substrate nodes with sufficient CPU capacity
        2. Among candidates, select node with highest available CPU (load balancing)
        3. Excluded nodes ensure Intra-VNR separation

        Args:
            substrate: Substrate network
            cpu_requirement: Required CPU capacity
            excluded_nodes: Already mapped substrate nodes (Intra-VNR separation)

        Returns:
            Best candidate node or None if no suitable node found
        """
        candidates = []

        for node_id in substrate.graph.nodes:
            if node_id in excluded_nodes:
                continue  # Intra-VNR separation constraint

            node_resources = substrate.get_node_resources(node_id)
            if not node_resources:
                continue

            # Check if node has sufficient CPU (Yu 2008: only CPU constraint)
            if node_resources.cpu_available >= cpu_requirement:
                total_capacity = node_resources.cpu_capacity
                utilization = (node_resources.cpu_used / total_capacity) if total_capacity > 0 else 0.0

                candidates.append(CandidateNodeInfo(
                    node_id=node_id,
                    cpu_available=node_resources.cpu_available,
                    utilization=utilization
                ))

        if not candidates:
            return None

        # Yu 2008 Strategy: Select node with highest available CPU (load balancing)
        candidates.sort(key=lambda x: x.cpu_available, reverse=True)
        selected = candidates[0]

        self.logger.debug(f"Selected substrate node {selected.node_id} "
                        f"(available CPU: {selected.cpu_available:.2f}, "
                        f"utilization: {selected.utilization:.2f})")

        return selected

    def _link_mapping_stage_yu2008(self, vnr: VirtualNetworkRequest,
                                  substrate: SubstrateNetwork,
                                  node_mapping: Dict[str, str]) -> Tuple[Dict[Tuple[str, str], List[str]], List[Tuple[int, int, float]]]:
        """
        Stage 2: Link mapping with immediate bandwidth allocation (Yu 2008).

        Yu 2008 Algorithm:
        1. For each virtual link, find k-shortest paths between mapped nodes
        2. Select path that satisfies bandwidth requirements
        3. Allocate bandwidth immediately on selected path
        4. If no suitable path found, fail entire embedding

        Args:
            vnr: Virtual network request
            substrate: Substrate network
            node_mapping: Node mapping from stage 1

        Returns:
            Tuple of (link_mapping, allocated_links) for rollback tracking
        """
        self.logger.debug("Starting Yu 2008 link mapping with allocation")

        if not vnr.virtual_links:
            return {}, []

        link_mapping = {}
        allocated_links = []  # Track for rollback: [(src, dst, bandwidth), ...]

        for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
            # Get mapped substrate nodes
            ssrc = int(node_mapping[str(vsrc)])
            sdst = int(node_mapping[str(vdst)])

            if ssrc == sdst:
                # Virtual link between nodes mapped to same substrate node
                # This violates Intra-VNR separation and should not happen with correct base class
                self.logger.warning(f"Virtual link ({vsrc}, {vdst}) maps to same substrate node {ssrc}")
                link_mapping[(str(vsrc), str(vdst))] = [str(ssrc)]
                continue

            # Find k-shortest paths with bandwidth constraints (Yu 2008)
            candidate_paths = self._find_k_shortest_paths_yu2008(
                substrate, ssrc, sdst, vlink_req.bandwidth_requirement
            )

            if not candidate_paths:
                self.logger.debug(f"No suitable paths for virtual link ({vsrc}, {vdst})")
                # Rollback previous link allocations
                self._rollback_link_allocations(substrate, allocated_links)
                return {}, []

            # Select best path using Yu 2008 strategy
            selected_path = self._select_best_path_yu2008(candidate_paths)

            # Allocate bandwidth immediately on selected path
            allocation_success = self._allocate_path_bandwidth_yu2008(
                substrate, selected_path.path, vlink_req.bandwidth_requirement
            )

            if not allocation_success:
                self.logger.debug(f"Bandwidth allocation failed for path {selected_path.path}")
                self._rollback_link_allocations(substrate, allocated_links)
                return {}, []

            # Record successful mapping and allocation
            link_mapping[(str(vsrc), str(vdst))] = [str(node) for node in selected_path.path]

            # Track allocations for potential rollback
            for i in range(len(selected_path.path) - 1):
                allocated_links.append((
                    selected_path.path[i],
                    selected_path.path[i + 1],
                    vlink_req.bandwidth_requirement
                ))

        return link_mapping, allocated_links

    def _find_k_shortest_paths_yu2008(self, substrate: SubstrateNetwork,
                                     src: int, dst: int,
                                     bandwidth_requirement: float) -> List[PathInfo]:
        """
        Find k-shortest paths with bandwidth constraints (Yu 2008).

        Yu 2008 approach: Use shortest paths that satisfy bandwidth requirements.

        Args:
            substrate: Substrate network
            src: Source substrate node
            dst: Destination substrate node
            bandwidth_requirement: Required bandwidth

        Returns:
            List of candidate paths with sufficient bandwidth
        """
        # Check cache first
        cache_key = (src, dst)
        if self.enable_path_caching and cache_key in self._path_cache:
            cached_paths = self._path_cache[cache_key]
            # Filter cached paths by current bandwidth availability
            return [path for path in cached_paths if path.min_bandwidth >= bandwidth_requirement]

        try:
            candidate_paths = []

            if self.k_paths == 1:
                # Single shortest path
                try:
                    path = nx.shortest_path(substrate.graph, src, dst)
                    all_paths = [path]
                except nx.NetworkXNoPath:
                    return []
            else:
                # Multiple shortest paths using nx.shortest_simple_paths
                try:
                    path_generator = nx.shortest_simple_paths(substrate.graph, src, dst)
                    all_paths = []
                    for i, path in enumerate(path_generator):
                        if i >= self.k_paths:
                            break
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    return []

            # Analyze each path for bandwidth availability
            for path in all_paths:
                path_info = self._analyze_path_yu2008(substrate, path)
                if path_info and path_info.min_bandwidth >= bandwidth_requirement:
                    candidate_paths.append(path_info)

            # Cache results if enabled
            if self.enable_path_caching:
                all_path_infos = [self._analyze_path_yu2008(substrate, path) for path in all_paths]
                self._path_cache[cache_key] = [p for p in all_path_infos if p is not None]

            self.logger.debug(f"Found {len(candidate_paths)} suitable paths from {src} to {dst}")
            return candidate_paths

        except Exception as e:
            self.logger.error(f"Error finding paths from {src} to {dst}: {e}")
            return []

    def _analyze_path_yu2008(self, substrate: SubstrateNetwork, path: List[int]) -> Optional[PathInfo]:
        """
        Analyze a path and collect its properties for Yu 2008.

        Args:
            substrate: Substrate network
            path: List of substrate node IDs forming the path

        Returns:
            PathInfo object with path analysis, None if path is invalid
        """
        if len(path) < 2:
            return None

        try:
            min_bandwidth = float('inf')

            # Analyze each link in the path
            for i in range(len(path) - 1):
                src_node = path[i]
                dst_node = path[i + 1]

                link_resources = substrate.get_link_resources(src_node, dst_node)
                if not link_resources:
                    return None  # Link doesn't exist

                min_bandwidth = min(min_bandwidth, link_resources.bandwidth_available)

            return PathInfo(
                path=path,
                hop_count=len(path) - 1,
                min_bandwidth=min_bandwidth
            )

        except Exception as e:
            self.logger.error(f"Error analyzing path {path}: {e}")
            return None

    def _select_best_path_yu2008(self, candidates: List[PathInfo]) -> PathInfo:
        """
        Select the best path from candidates using Yu 2008 strategy.

        Yu 2008 Strategy:
        - "shortest": Minimum hop count, then maximum bandwidth
        - "bandwidth": Maximum available bandwidth, then minimum hop count

        Args:
            candidates: List of candidate paths

        Returns:
            Selected path
        """
        if len(candidates) == 1:
            return candidates[0]

        if self.path_selection_strategy == "shortest":
            # Select path with minimum hop count, then maximum bandwidth
            candidates.sort(key=lambda p: (p.hop_count, -p.min_bandwidth))
        elif self.path_selection_strategy == "bandwidth":
            # Select path with maximum available bandwidth, then minimum hop count
            candidates.sort(key=lambda p: (-p.min_bandwidth, p.hop_count))

        selected = candidates[0]
        self.logger.debug(f"Selected path {selected.path} "
                        f"(hops: {selected.hop_count}, bandwidth: {selected.min_bandwidth:.2f})")

        return selected

    def _allocate_path_bandwidth_yu2008(self, substrate: SubstrateNetwork,
                                       path: List[int], bandwidth: float) -> bool:
        """
        Allocate bandwidth along a path with rollback on failure.

        Args:
            substrate: Substrate network
            path: Path as list of substrate node IDs
            bandwidth: Bandwidth to allocate

        Returns:
            True if allocation successful, False otherwise
        """
        allocated_links = []

        try:
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                success = substrate.allocate_link_resources(src, dst, bandwidth)

                if not success:
                    # Rollback partial allocation
                    for rollback_src, rollback_dst in allocated_links:
                        substrate.deallocate_link_resources(rollback_src, rollback_dst, bandwidth)
                    return False

                allocated_links.append((src, dst))

            return True

        except Exception as e:
            self.logger.error(f"Error allocating bandwidth on path {path}: {e}")
            # Rollback partial allocation
            for rollback_src, rollback_dst in allocated_links:
                substrate.deallocate_link_resources(rollback_src, rollback_dst, bandwidth)
            return False

    def _rollback_node_allocations(self, substrate: SubstrateNetwork,
                                  allocated_nodes: List[Tuple[int, float]]) -> None:
        """Rollback node resource allocations."""
        for node_id, cpu_allocated in allocated_nodes:
            substrate.deallocate_node_resources(node_id, cpu_allocated, 0.0)

        self.logger.debug(f"Rolled back {len(allocated_nodes)} node allocations")

    def _rollback_link_allocations(self, substrate: SubstrateNetwork,
                                  allocated_links: List[Tuple[int, int, float]]) -> None:
        """Rollback link resource allocations."""
        for src, dst, bandwidth in allocated_links:
            substrate.deallocate_link_resources(src, dst, bandwidth)

        self.logger.debug(f"Rolled back {len(allocated_links)} link allocations")

    def clear_path_cache(self) -> None:
        """Clear the path cache to free memory."""
        cache_size = len(self._path_cache)
        self._path_cache.clear()
        self.logger.debug(f"Cleared path cache ({cache_size} entries)")

    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """
        Get Yu 2008 algorithm-specific statistics.

        Returns:
            Dictionary with algorithm statistics
        """
        base_stats = super().get_algorithm_statistics()

        yu_stats = {
            'algorithm_type': 'Two-Stage (Yu 2008)',
            'k_paths': self.k_paths,
            'path_selection_strategy': self.path_selection_strategy,
            'path_cache_size': len(self._path_cache) if self.enable_path_caching else 0,
            'path_cache_enabled': self.enable_path_caching,
            'constraint_types': ['CPU', 'Bandwidth'],  # Yu 2008 only
            'literature_reference': 'Yu et al. (2008) ACM SIGCOMM'
        }

        # Merge with base statistics
        base_stats.update(yu_stats)
        return base_stats

    def __str__(self) -> str:
        """String representation of the algorithm."""
        stats = self.get_algorithm_statistics()
        return (f"Yu2008Algorithm(k_paths={self.k_paths}, "
                f"AR={stats['acceptance_ratio']:.3f}, "
                f"requests={stats['total_requests']})")

    def __repr__(self) -> str:
        """Detailed representation of the algorithm."""
        return (f"YuAlgorithm(k_paths={self.k_paths}, "
                f"path_selection_strategy='{self.path_selection_strategy}', "
                f"enable_path_caching={self.enable_path_caching})")


# =============================================================================
# EXPORT FOR CLEAN MODULE INTERFACE
# =============================================================================

__all__ = ['YuAlgorithm']
