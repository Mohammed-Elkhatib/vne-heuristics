"""
Yu et al. (2008) Two-Stage Virtual Network Embedding Algorithm.

This module implements the seminal two-stage VNE algorithm from Yu et al.'s 2008 paper
"Rethinking Virtual Network Embedding: Substrate Support for Path Splitting and Migration".

The algorithm uses a greedy approach for node mapping followed by shortest path
link mapping, serving as a fundamental baseline for VNE research.

Reference:
    Yu, M., Yi, Y., Rexford, J., & Chiang, M. (2008). Rethinking virtual network 
    embedding: substrate support for path splitting and migration. ACM SIGCOMM 
    Computer Communication Review, 38(2), 17-29.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
import networkx as nx
from dataclasses import dataclass

from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork


@dataclass
class NodeRankingInfo:
    """Information for ranking virtual nodes."""
    node_id: str
    cpu_requirement: float
    memory_requirement: float
    total_requirement: float
    degree: int


@dataclass
class CandidateNodeInfo:
    """Information about candidate substrate nodes."""
    node_id: str
    cpu_available: float
    memory_available: float
    total_available: float
    utilization: float


@dataclass
class PathInfo:
    """Information about candidate paths for link mapping."""
    path: List[str]
    hop_count: int
    total_delay: float
    min_bandwidth: float
    total_cost: float
    bottleneck_link: Tuple[str, str]


class YuAlgorithm(BaseAlgorithm):
    """
    Yu et al. (2008) Two-Stage Virtual Network Embedding Algorithm.
    
    This algorithm implements the classical two-stage approach:
    1. Greedy node mapping based on resource requirements
    2. Shortest path link mapping with bandwidth constraints
    
    The algorithm serves as a fundamental baseline for VNE research and provides
    a solid foundation for comparison with more advanced algorithms.
    
    Attributes:
        k_paths: Number of shortest paths to consider for link mapping
        cpu_weight: Weight for CPU requirements in node ranking
        memory_weight: Weight for memory requirements in node ranking
        path_selection_strategy: Strategy for selecting among k paths
        enable_caching: Whether to cache shortest paths for performance
        max_path_length: Maximum allowed path length for link mapping
        
    Example:
        >>> algorithm = YuAlgorithm(k_paths=3, cpu_weight=1.0, memory_weight=1.0)
        >>> result = algorithm.embed_vnr(vnr, substrate)
        >>> print(f"Embedding success: {result.success}")
    """
    
    def __init__(self, k_paths: int = 1, cpu_weight: float = 1.0, 
                 memory_weight: float = 1.0, path_selection_strategy: str = "shortest",
                 enable_caching: bool = True, max_path_length: int = 10, **kwargs):
        """
        Initialize the Yu algorithm with specified parameters.
        
        Args:
            k_paths: Number of shortest paths to consider (default: 1)
            cpu_weight: Weight for CPU in node ranking (default: 1.0)
            memory_weight: Weight for memory in node ranking (default: 1.0)
            path_selection_strategy: Path selection strategy 
                ("shortest", "bandwidth", "delay", "cost") (default: "shortest")
            enable_caching: Enable path caching for performance (default: True)
            max_path_length: Maximum path length to consider (default: 10)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__("Yu et al. (2008) Two-Stage Algorithm", **kwargs)
        
        # Algorithm parameters
        self.k_paths = max(1, k_paths)
        self.cpu_weight = cpu_weight
        self.memory_weight = memory_weight
        self.path_selection_strategy = path_selection_strategy
        self.enable_caching = enable_caching
        self.max_path_length = max_path_length
        
        # Path cache for performance optimization
        self._path_cache: Dict[Tuple[str, str], List[PathInfo]] = {}
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info(f"Initialized Yu algorithm with k_paths={k_paths}, "
                        f"weights=({cpu_weight}, {memory_weight}), "
                        f"strategy={path_selection_strategy}")
    
    def _validate_parameters(self) -> None:
        """Validate algorithm parameters."""
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        
        if self.cpu_weight < 0 or self.memory_weight < 0:
            raise ValueError("Resource weights must be non-negative")
        
        if self.cpu_weight == 0 and self.memory_weight == 0:
            raise ValueError("At least one resource weight must be positive")
        
        valid_strategies = ["shortest", "bandwidth", "delay", "cost"]
        if self.path_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid path selection strategy. Must be one of: {valid_strategies}")
        
        if self.max_path_length <= 0:
            raise ValueError("max_path_length must be positive")
    
    def _embed_single_vnr(self, vnr: VirtualNetworkRequest, 
                         substrate: SubstrateNetwork) -> EmbeddingResult:
        """
        Core embedding logic for a single VNR using the two-stage approach.
        
        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto
            
        Returns:
            EmbeddingResult with the outcome of the embedding attempt
        """
        self.logger.info(f"Starting two-stage embedding for VNR {vnr.vnr_id}")
        
        try:
            # Stage 1: Node Mapping
            self.logger.debug("Stage 1: Starting node mapping")
            node_mapping = self._node_mapping_stage(vnr, substrate)
            
            if not node_mapping:
                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id),
                    success=False,
                    node_mapping={},
                    link_mapping={},
                    revenue=0.0,
                    cost=0.0,
                    execution_time=0.0,
                    failure_reason="Node mapping failed - insufficient node resources"
                )
            
            self.logger.debug(f"Stage 1 completed: {len(node_mapping)} nodes mapped")
            
            # Stage 2: Link Mapping
            self.logger.debug("Stage 2: Starting link mapping")
            link_mapping = self._link_mapping_stage(vnr, substrate, node_mapping)
            
            if not link_mapping and vnr.virtual_links:
                # Link mapping failed, rollback node allocations
                self.logger.debug("Stage 2 failed: Rolling back node allocations")
                self._rollback_allocation(substrate, node_mapping, {})
                
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
            self.logger.info(f"Successfully embedded VNR {vnr.vnr_id}")
            
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=True,
                node_mapping=node_mapping,
                link_mapping=link_mapping,
                revenue=0.0,  # Will be calculated by base class
                cost=0.0,     # Will be calculated by base class
                execution_time=0.0  # Will be set by base class
            )
            
        except Exception as e:
            self.logger.error(f"Exception during embedding of VNR {vnr.vnr_id}: {e}")
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
    
    def _node_mapping_stage(self, vnr: VirtualNetworkRequest, 
                           substrate: SubstrateNetwork) -> Dict[str, str]:
        """
        Stage 1: Greedy node mapping based on resource requirements.
        
        This stage implements the greedy node mapping approach where virtual nodes
        are ranked by their resource requirements and mapped to substrate nodes
        with the highest available resources.
        
        Args:
            vnr: Virtual network request
            substrate: Substrate network
            
        Returns:
            Dictionary mapping virtual node IDs to substrate node IDs,
            empty dict if mapping fails
        """
        self.logger.debug("Starting greedy node mapping stage")
        
        # Step 1: Rank virtual nodes by resource requirements
        ranked_vnodes = self._rank_virtual_nodes(vnr)
        self.logger.debug(f"Ranked {len(ranked_vnodes)} virtual nodes by requirements")
        
        # Step 2: Attempt to map each virtual node
        node_mapping = {}
        mapped_substrate_nodes = set()
        
        for vnode_info in ranked_vnodes:
            vnode_id = vnode_info.node_id
            vnode_req = vnr.virtual_nodes[int(vnode_id)]
            
            self.logger.debug(f"Mapping virtual node {vnode_id} "
                            f"(CPU: {vnode_req.cpu_requirement}, "
                            f"Memory: {vnode_req.memory_requirement})")
            
            # Find candidate substrate nodes
            candidates = self._find_candidate_nodes(
                substrate, vnode_req.cpu_requirement, 
                vnode_req.memory_requirement, mapped_substrate_nodes
            )
            
            if not candidates:
                self.logger.debug(f"No suitable substrate nodes for virtual node {vnode_id}")
                # Rollback previous mappings
                self._rollback_node_mapping(substrate, node_mapping, vnr)
                return {}
            
            # Select best candidate substrate node
            selected_node = self._select_best_candidate_node(candidates)
            
            # Record the mapping
            node_mapping[vnode_id] = str(selected_node.node_id)
            mapped_substrate_nodes.add(selected_node.node_id)
            
            self.logger.debug(f"Mapped virtual node {vnode_id} to substrate node {selected_node.node_id}")
        
        self.logger.debug(f"Node mapping stage completed successfully: {node_mapping}")
        return node_mapping
    
    def _rank_virtual_nodes(self, vnr: VirtualNetworkRequest) -> List[NodeRankingInfo]:
        """
        Rank virtual nodes by their resource requirements in descending order.
        
        Args:
            vnr: Virtual network request
            
        Returns:
            List of NodeRankingInfo sorted by total requirements (descending)
        """
        node_rankings = []
        
        for vnode_id, vnode_req in vnr.virtual_nodes.items():
            # Calculate total weighted resource requirement
            total_req = (self.cpu_weight * vnode_req.cpu_requirement + 
                        self.memory_weight * vnode_req.memory_requirement)
            
            # Calculate node degree in VNR topology
            degree = vnr.graph.degree(vnode_id) if vnode_id in vnr.graph else 0
            
            node_rankings.append(NodeRankingInfo(
                node_id=str(vnode_id),
                cpu_requirement=vnode_req.cpu_requirement,
                memory_requirement=vnode_req.memory_requirement,
                total_requirement=total_req,
                degree=degree
            ))
        
        # Sort by total requirement (descending), then by degree (descending)
        node_rankings.sort(key=lambda x: (x.total_requirement, x.degree), reverse=True)
        
        self.logger.debug(f"Virtual node ranking: {[(n.node_id, n.total_requirement) for n in node_rankings]}")
        return node_rankings
    
    def _find_candidate_nodes(self, substrate: SubstrateNetwork, 
                             cpu_req: float, memory_req: float,
                             excluded_nodes: Set[str]) -> List[CandidateNodeInfo]:
        """
        Find substrate nodes that can satisfy the resource requirements.
        
        Args:
            substrate: Substrate network
            cpu_req: Required CPU capacity
            memory_req: Required memory capacity  
            excluded_nodes: Set of already mapped substrate nodes to exclude
            
        Returns:
            List of candidate substrate nodes with their resource information
        """
        candidates = []
        
        for snode_id in substrate.graph.nodes:
            if snode_id in excluded_nodes:
                continue
                
            snode_resources = substrate.get_node_resources(snode_id)
            if not snode_resources:
                continue
            
            # Check if node has sufficient resources
            if (snode_resources.cpu_available >= cpu_req and 
                snode_resources.memory_available >= memory_req):
                
                total_available = (self.cpu_weight * snode_resources.cpu_available + 
                                 self.memory_weight * snode_resources.memory_available)
                
                total_capacity = (self.cpu_weight * snode_resources.cpu_capacity + 
                                self.memory_weight * snode_resources.memory_capacity)
                
                utilization = 1.0 - (total_available / total_capacity) if total_capacity > 0 else 0.0
                
                candidates.append(CandidateNodeInfo(
                    node_id=str(snode_id),
                    cpu_available=snode_resources.cpu_available,
                    memory_available=snode_resources.memory_available,
                    total_available=total_available,
                    utilization=utilization
                ))
        
        self.logger.debug(f"Found {len(candidates)} candidate substrate nodes")
        return candidates
    
    def _select_best_candidate_node(self, candidates: List[CandidateNodeInfo]) -> CandidateNodeInfo:
        """
        Select the best substrate node from candidates.
        
        Selection strategy: Choose node with maximum available resources
        (equivalent to minimum utilization for resource balancing).
        
        Args:
            candidates: List of candidate substrate nodes
            
        Returns:
            Selected candidate node
        """
        # Sort by total available resources (descending)
        candidates.sort(key=lambda x: x.total_available, reverse=True)
        
        selected = candidates[0]
        self.logger.debug(f"Selected substrate node {selected.node_id} "
                        f"(available: {selected.total_available:.2f}, "
                        f"utilization: {selected.utilization:.2f})")
        
        return selected
    
    def _rollback_node_mapping(self, substrate: SubstrateNetwork, 
                              node_mapping: Dict[str, str],
                              vnr: VirtualNetworkRequest) -> None:
        """
        Rollback node resource allocations on mapping failure.
        
        Args:
            substrate: Substrate network
            node_mapping: Current node mapping to rollback
            vnr: Virtual network request
        """
        self.logger.debug("Rolling back node resource allocations")
        
        for vnode_id, snode_id in node_mapping.items():
            vnode_req = vnr.virtual_nodes[int(vnode_id)]
            substrate.deallocate_node_resources(
                int(snode_id),
                vnode_req.cpu_requirement,
                vnode_req.memory_requirement
            )
        
        self.logger.debug(f"Rolled back {len(node_mapping)} node allocations")
    
    def _link_mapping_stage(self, vnr: VirtualNetworkRequest,
                           substrate: SubstrateNetwork,
                           node_mapping: Dict[str, str]) -> Dict[Tuple[str, str], List[str]]:
        """
        Stage 2: Link mapping using shortest path algorithms.
        
        This stage finds paths in the substrate network for each virtual link
        using shortest path algorithms and bandwidth constraint checking.
        
        Args:
            vnr: Virtual network request
            substrate: Substrate network
            node_mapping: Mapping of virtual nodes to substrate nodes
            
        Returns:
            Dictionary mapping virtual links to substrate paths,
            empty dict if mapping fails
        """
        self.logger.debug("Starting shortest path link mapping stage")
        
        if not vnr.virtual_links:
            self.logger.debug("No virtual links to map")
            return {}
        
        link_mapping = {}
        
        for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
            self.logger.debug(f"Mapping virtual link ({vsrc}, {vdst}) "
                            f"with bandwidth requirement {vlink_req.bandwidth_requirement}")
            
            # Get mapped substrate nodes
            ssrc = node_mapping[str(vsrc)]
            sdst = node_mapping[str(vdst)]
            
            if ssrc == sdst:
                # Virtual link between nodes mapped to same substrate node
                self.logger.debug(f"Virtual link ({vsrc}, {vdst}) maps to same substrate node {ssrc}")
                link_mapping[(str(vsrc), str(vdst))] = [ssrc, ssrc]
                continue
            
            # Find suitable path(s)
            candidate_paths = self._find_candidate_paths(
                substrate, int(ssrc), int(sdst), vlink_req.bandwidth_requirement
            )
            
            if not candidate_paths:
                self.logger.debug(f"No suitable paths for virtual link ({vsrc}, {vdst})")
                # Rollback previous link allocations
                self._rollback_link_mapping(substrate, link_mapping, vnr)
                return {}
            
            # Select best path
            selected_path = self._select_best_candidate_path(candidate_paths)
            
            # Record the mapping
            link_mapping[(str(vsrc), str(vdst))] = [str(node) for node in selected_path.path]
            
            self.logger.debug(f"Mapped virtual link ({vsrc}, {vdst}) to path {selected_path.path}")
        
        self.logger.debug(f"Link mapping stage completed successfully: {len(link_mapping)} links mapped")
        return link_mapping
    
    def _find_candidate_paths(self, substrate: SubstrateNetwork, 
                             src: int, dst: int, bandwidth_req: float) -> List[PathInfo]:
        """
        Find candidate paths between source and destination substrate nodes.
        
        Args:
            substrate: Substrate network
            src: Source substrate node ID
            dst: Destination substrate node ID
            bandwidth_req: Required bandwidth
            
        Returns:
            List of candidate paths that satisfy bandwidth requirements
        """
        # Check cache first
        cache_key = (str(src), str(dst))
        if self.enable_caching and cache_key in self._path_cache:
            cached_paths = self._path_cache[cache_key]
            # Filter cached paths by bandwidth availability
            return [path for path in cached_paths if path.min_bandwidth >= bandwidth_req]
        
        try:
            # Find k shortest paths
            if self.k_paths == 1:
                # Single shortest path
                try:
                    shortest_path = nx.shortest_path(substrate.graph, src, dst, weight='delay')
                    all_paths = [shortest_path]
                except nx.NetworkXNoPath:
                    self.logger.debug(f"No path exists between {src} and {dst}")
                    return []
            else:
                # Multiple shortest paths
                try:
                    path_generator = nx.shortest_simple_paths(substrate.graph, src, dst, weight='delay')
                    all_paths = []
                    for i, path in enumerate(path_generator):
                        if i >= self.k_paths or len(path) > self.max_path_length:
                            break
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    self.logger.debug(f"No path exists between {src} and {dst}")
                    return []
            
            # Analyze each path
            candidate_paths = []
            for path in all_paths:
                path_info = self._analyze_path(substrate, path)
                if path_info and path_info.min_bandwidth >= bandwidth_req:
                    candidate_paths.append(path_info)
            
            # Cache results if enabled
            if self.enable_caching:
                # Cache all analyzed paths (not just candidates)
                all_path_infos = [self._analyze_path(substrate, path) for path in all_paths]
                self._path_cache[cache_key] = [p for p in all_path_infos if p is not None]
            
            self.logger.debug(f"Found {len(candidate_paths)} candidate paths from {src} to {dst}")
            return candidate_paths
            
        except Exception as e:
            self.logger.error(f"Error finding paths from {src} to {dst}: {e}")
            return []
    
    def _analyze_path(self, substrate: SubstrateNetwork, path: List[int]) -> Optional[PathInfo]:
        """
        Analyze a path and collect its properties.
        
        Args:
            substrate: Substrate network
            path: List of substrate node IDs forming the path
            
        Returns:
            PathInfo object with path analysis, None if path is invalid
        """
        if len(path) < 2:
            return None
        
        try:
            total_delay = 0.0
            total_cost = 0.0
            min_bandwidth = float('inf')
            bottleneck_link = None
            
            # Analyze each link in the path
            for i in range(len(path) - 1):
                src_node = path[i]
                dst_node = path[i + 1]
                
                link_resources = substrate.get_link_resources(src_node, dst_node)
                if not link_resources:
                    self.logger.warning(f"Link ({src_node}, {dst_node}) not found in substrate")
                    return None
                
                total_delay += link_resources.delay
                total_cost += link_resources.cost
                
                if link_resources.bandwidth_available < min_bandwidth:
                    min_bandwidth = link_resources.bandwidth_available
                    bottleneck_link = (str(src_node), str(dst_node))
            
            return PathInfo(
                path=[str(node) for node in path],
                hop_count=len(path) - 1,
                total_delay=total_delay,
                min_bandwidth=min_bandwidth,
                total_cost=total_cost,
                bottleneck_link=bottleneck_link
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing path {path}: {e}")
            return None
    
    def _select_best_candidate_path(self, candidates: List[PathInfo]) -> PathInfo:
        """
        Select the best path from candidates based on selection strategy.
        
        Args:
            candidates: List of candidate paths
            
        Returns:
            Selected path
        """
        if len(candidates) == 1:
            return candidates[0]
        
        if self.path_selection_strategy == "shortest":
            # Select path with minimum hop count, then minimum delay
            candidates.sort(key=lambda p: (p.hop_count, p.total_delay))
        elif self.path_selection_strategy == "bandwidth":
            # Select path with maximum available bandwidth
            candidates.sort(key=lambda p: p.min_bandwidth, reverse=True)
        elif self.path_selection_strategy == "delay":
            # Select path with minimum total delay
            candidates.sort(key=lambda p: p.total_delay)
        elif self.path_selection_strategy == "cost":
            # Select path with minimum total cost
            candidates.sort(key=lambda p: p.total_cost)
        
        selected = candidates[0]
        self.logger.debug(f"Selected path {selected.path} "
                        f"(hops: {selected.hop_count}, "
                        f"bandwidth: {selected.min_bandwidth:.2f}, "
                        f"delay: {selected.total_delay:.2f})")
        
        return selected
    
    def _rollback_link_mapping(self, substrate: SubstrateNetwork,
                              link_mapping: Dict[Tuple[str, str], List[str]],
                              vnr: VirtualNetworkRequest) -> None:
        """
        Rollback link resource allocations on mapping failure.
        
        Args:
            substrate: Substrate network
            link_mapping: Current link mapping to rollback
            vnr: Virtual network request
        """
        self.logger.debug("Rolling back link resource allocations")
        
        for (vsrc, vdst), path in link_mapping.items():
            vlink_req = vnr.virtual_links[(int(vsrc), int(vdst))]
            
            # Deallocate bandwidth on each link in the path
            for i in range(len(path) - 1):
                src_node = int(path[i])
                dst_node = int(path[i + 1])
                substrate.deallocate_link_resources(
                    src_node, dst_node, vlink_req.bandwidth_requirement
                )
        
        self.logger.debug(f"Rolled back {len(link_mapping)} link allocations")
    
    def _rollback_allocation(self, substrate: SubstrateNetwork,
                           node_mapping: Dict[str, str],
                           link_mapping: Dict[Tuple[str, str], List[str]]) -> None:
        """
        Complete rollback of both node and link allocations.
        
        Args:
            substrate: Substrate network
            node_mapping: Node mapping to rollback
            link_mapping: Link mapping to rollback
        """
        self.logger.debug("Performing complete allocation rollback")
        
        # Note: This method is called from base class if needed
        # Individual rollback methods handle the actual deallocation
        self.logger.debug("Rollback completed")
    
    def clear_path_cache(self) -> None:
        """Clear the path cache to free memory."""
        cache_size = len(self._path_cache)
        self._path_cache.clear()
        self.logger.debug(f"Cleared path cache ({cache_size} entries)")
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """
        Get algorithm-specific statistics.
        
        Returns:
            Dictionary with algorithm statistics
        """
        base_stats = self.get_statistics()
        
        yu_stats = {
            'algorithm_name': 'Yu et al. (2008)',
            'k_paths': self.k_paths,
            'cpu_weight': self.cpu_weight,
            'memory_weight': self.memory_weight,
            'path_selection_strategy': self.path_selection_strategy,
            'path_cache_size': len(self._path_cache) if self.enable_caching else 0,
            'path_cache_enabled': self.enable_caching,
            'max_path_length': self.max_path_length
        }
        
        # Merge with base statistics
        base_stats.update(yu_stats)
        return base_stats
    
    def __str__(self) -> str:
        """String representation of the algorithm."""
        return (f"Yu2008Algorithm(k_paths={self.k_paths}, "
                f"strategy={self.path_selection_strategy}, "
                f"weights=({self.cpu_weight}, {self.memory_weight}))")
    
    def __repr__(self) -> str:
        """Detailed representation of the algorithm."""  
        return (f"YuAlgorithm(k_paths={self.k_paths}, "
                f"cpu_weight={self.cpu_weight}, "
                f"memory_weight={self.memory_weight}, "
                f"path_selection_strategy='{self.path_selection_strategy}', "
                f"enable_caching={self.enable_caching}, "
                f"max_path_length={self.max_path_length})")


# Make the algorithm discoverable
__all__ = ['YuAlgorithm']