"""
Performance metrics calculations for Virtual Network Embedding (VNE).

This module provides functions to calculate VNE performance metrics according to
standard formulas from VNE literature, accounting for primary (CPU + bandwidth)
and secondary (memory, delay, reliability) constraints.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from statistics import mean, stdev
import time

# Import VNE model classes
try:
    from src.models.substrate import SubstrateNetwork
    from src.models.virtual_request import VirtualNetworkRequest
except ImportError:
    # Handle cases where models might not be available
    SubstrateNetwork = None
    VirtualNetworkRequest = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Represents the result of a VNR embedding attempt.

    Attributes:
        vnr_id: Unique identifier for the VNR
        success: Whether the embedding was successful
        revenue: Revenue generated (calculated from VNR requirements)
        cost: Cost incurred (calculated from substrate consumption)
        execution_time: Time taken for the embedding attempt (seconds)
        node_mapping: Dictionary mapping virtual nodes to substrate nodes
        link_mapping: Dictionary mapping virtual links to substrate paths
        timestamp: When the embedding was attempted
        failure_reason: Reason for failure if unsuccessful
        vnr: Reference to the original VNR (for revenue calculation)
        path_lengths: Dictionary of path lengths for cost calculation
    """
    vnr_id: Union[str, int]
    success: bool
    revenue: float = 0.0
    cost: float = 0.0
    execution_time: float = 0.0
    node_mapping: Optional[Dict[Union[str, int], Union[str, int]]] = None
    link_mapping: Optional[Dict[Tuple[Union[str, int], Union[str, int]], List[Tuple[Union[str, int], Union[str, int]]]]] = None
    timestamp: Optional[float] = None
    failure_reason: Optional[str] = None
    vnr: Optional['VirtualNetworkRequest'] = None
    path_lengths: Optional[Dict[Tuple[Union[str, int], Union[str, int]], int]] = field(default_factory=dict)


class MetricsError(Exception):
    """Exception raised for metrics calculation errors."""
    pass


# =============================================================================
# STANDARD VNE REVENUE CALCULATION (Based on VNR Requirements)
# =============================================================================

def calculate_vnr_revenue(vnr: 'VirtualNetworkRequest') -> float:
    """
    Calculate revenue for a VNR based on standard VNE formula.

    Standard Formula:
    Revenue = Σ(CPU_requirements) + Σ(Bandwidth_requirements)

    Args:
        vnr: VirtualNetworkRequest instance

    Returns:
        Total revenue for the VNR

    Note:
        - Primary constraints (CPU + Bandwidth) always contribute to revenue
        - Secondary constraints (Memory, Delay, Reliability) are optional
    """
    if not vnr:
        return 0.0

    # Primary constraints revenue (always included)
    node_revenue = sum(node.cpu_requirement for node in vnr.virtual_nodes.values())
    link_revenue = sum(link.bandwidth_requirement for link in vnr.virtual_links.values())

    primary_revenue = node_revenue + link_revenue

    # Secondary constraints revenue (optional - only if constraints are used)
    secondary_revenue = 0.0
    constraint_summary = vnr.get_constraint_summary()

    # Add memory revenue if memory constraints are used
    if constraint_summary['uses_memory_constraints']:
        memory_revenue = sum(node.memory_requirement for node in vnr.virtual_nodes.values())
        secondary_revenue += memory_revenue

    # Note: Delay and reliability are constraints, not resource requirements,
    # so they don't directly contribute to revenue in standard VNE formulations

    total_revenue = primary_revenue + secondary_revenue
    logger.debug(f"VNR {vnr.vnr_id} revenue: nodes={node_revenue}, links={link_revenue}, "
                f"memory={secondary_revenue}, total={total_revenue}")

    return total_revenue


def calculate_vnr_cost(vnr: 'VirtualNetworkRequest',
                      node_mapping: Dict[Union[str, int], Union[str, int]],
                      link_mapping: Dict[Tuple[Union[str, int], Union[str, int]], List[Tuple[Union[str, int], Union[str, int]]]],
                      substrate_network: Optional['SubstrateNetwork'] = None) -> float:
    """
    Calculate embedding cost based on standard VNE formula.

    Standard Formula:
    Cost = Σ(Node_CPU_allocated) + Σ(Link_bandwidth_allocated × path_length)

    Note: Memory costs are only added if the substrate network explicitly
    enables memory constraints, not just if the VNR has memory requirements.

    Args:
        vnr: VirtualNetworkRequest instance
        node_mapping: Virtual to substrate node mapping
        link_mapping: Virtual to substrate path mapping
        substrate_network: Optional substrate network for resource validation

    Returns:
        Total embedding cost
    """
    if not vnr or not node_mapping:
        return 0.0

    # Node costs (CPU allocation - primary constraint)
    node_cost = sum(node.cpu_requirement for node in vnr.virtual_nodes.values())

    # Link costs (Bandwidth × path length - primary constraint)
    link_cost = 0.0
    for (src_vnode, dst_vnode), substrate_path in link_mapping.items():
        if (src_vnode, dst_vnode) in vnr.virtual_links:
            virtual_link = vnr.virtual_links[(src_vnode, dst_vnode)]
            path_length = len(substrate_path)  # Number of substrate links in path
            link_cost += virtual_link.bandwidth_requirement * path_length

    # Secondary constraint costs (only if substrate network enables them)
    secondary_cost = 0.0

    # Add memory cost ONLY if substrate network explicitly enables memory constraints
    if (substrate_network and
        hasattr(substrate_network, 'enable_memory_constraints') and
        substrate_network.enable_memory_constraints):
        memory_cost = sum(node.memory_requirement for node in vnr.virtual_nodes.values())
        secondary_cost += memory_cost
    elif substrate_network is None:
        # Fallback: check if VNR uses memory constraints (for backward compatibility)
        constraint_summary = vnr.get_constraint_summary()
        if constraint_summary['uses_memory_constraints']:
            memory_cost = sum(node.memory_requirement for node in vnr.virtual_nodes.values())
            secondary_cost += memory_cost

    total_cost = node_cost + link_cost + secondary_cost
    logger.debug(f"VNR {vnr.vnr_id} cost: nodes={node_cost}, links={link_cost}, "
                f"memory={secondary_cost}, total={total_cost}")

    return total_cost


# =============================================================================
# PRIMARY VNE METRICS (Standard Literature Formulas)
# =============================================================================

def calculate_acceptance_ratio(results: List[EmbeddingResult]) -> float:
    """
    Calculate the acceptance ratio of VNR embedding attempts.

    Standard Formula: AR = |Successfully_embedded_VNRs| / |Total_VNRs|

    Args:
        results: List of embedding results

    Returns:
        Acceptance ratio (0.0 to 1.0)

    Raises:
        ValueError: If results list is empty
    """
    if not results:
        raise ValueError("Results list cannot be empty")

    successful = sum(1 for result in results if result.success)
    total = len(results)

    ratio = successful / total
    logger.debug(f"Acceptance ratio: {successful}/{total} = {ratio:.4f}")

    return ratio


def calculate_blocking_probability(results: List[EmbeddingResult]) -> float:
    """
    Calculate the blocking probability.

    Standard Formula: BP = 1 - Acceptance_Ratio

    Args:
        results: List of embedding results

    Returns:
        Blocking probability (0.0 to 1.0)
    """
    if not results:
        return 0.0

    return 1.0 - calculate_acceptance_ratio(results)


def calculate_total_revenue(results: List[EmbeddingResult]) -> float:
    """
    Calculate total revenue from successful VNR embeddings.

    Standard Formula: Revenue = Σ(VNR_revenues) for successful embeddings

    Args:
        results: List of embedding results

    Returns:
        Total revenue from successful embeddings
    """
    # Calculate revenue from VNR requirements for successful embeddings
    total_revenue = 0.0

    for result in results:
        if result.success:
            if result.vnr:
                # Calculate revenue from VNR requirements (standard approach)
                vnr_revenue = calculate_vnr_revenue(result.vnr)
                total_revenue += vnr_revenue
            else:
                # Fallback to stored revenue value
                total_revenue += result.revenue

    logger.debug(f"Total revenue: {total_revenue:.2f}")
    return total_revenue


def calculate_total_cost(results: List[EmbeddingResult]) -> float:
    """
    Calculate total cost from all VNR embedding attempts.

    Standard Formula: Cost = Σ(Embedding_costs) for all attempts

    Args:
        results: List of embedding results

    Returns:
        Total cost from all embedding attempts
    """
    # Calculate cost from actual resource consumption
    total_cost = 0.0

    for result in results:
        if result.vnr and result.node_mapping and result.link_mapping:
            # Calculate cost from substrate consumption (standard approach)
            embedding_cost = calculate_vnr_cost(result.vnr, result.node_mapping, result.link_mapping)
            total_cost += embedding_cost
        else:
            # Fallback to stored cost value
            total_cost += result.cost

    logger.debug(f"Total cost: {total_cost:.2f}")
    return total_cost


def calculate_revenue_to_cost_ratio(results: List[EmbeddingResult]) -> float:
    """
    Calculate the revenue-to-cost ratio.

    Standard Formula: R/C = Total_Revenue / Total_Cost

    Args:
        results: List of embedding results

    Returns:
        Revenue-to-cost ratio, or 0.0 if total cost is 0
    """
    total_revenue = calculate_total_revenue(results)
    total_cost = calculate_total_cost(results)

    if total_cost == 0:
        logger.warning("Total cost is zero, returning 0.0 for revenue-to-cost ratio")
        return 0.0

    ratio = total_revenue / total_cost
    logger.debug(f"Revenue-to-cost ratio: {ratio:.4f}")

    return ratio


# =============================================================================
# RESOURCE UTILIZATION METRICS
# =============================================================================

def calculate_utilization(substrate_network: Union['SubstrateNetwork', Any]) -> Dict[str, float]:
    """
    Calculate average resource utilization for substrate network.

    Standard Formula:
    - CPU_utilization = Σ(CPU_used) / Σ(CPU_capacity)
    - Bandwidth_utilization = Σ(Bandwidth_used) / Σ(Bandwidth_capacity)

    Args:
        substrate_network: SubstrateNetwork instance

    Returns:
        Dictionary with utilization metrics:
        - avg_node_cpu_util: Average CPU utilization across nodes
        - avg_node_memory_util: Average memory utilization across nodes
        - avg_link_bandwidth_util: Average bandwidth utilization across links

    Raises:
        MetricsError: If substrate network structure is invalid
    """
    try:
        utilization_metrics = {}

        # Handle NetworkX-based structure (correct for our SubstrateNetwork)
        if hasattr(substrate_network, 'graph'):
            nodes = substrate_network.graph.nodes
            edges = substrate_network.graph.edges
        else:
            raise MetricsError("Substrate network has invalid structure")

        # Calculate node utilization (PRIMARY: CPU, SECONDARY: Memory)
        node_cpu_utils = []
        node_memory_utils = []

        for node_id in nodes:
            try:
                node_resources = substrate_network.get_node_resources(node_id)
                if not node_resources:
                    continue

                # Primary constraint: CPU (always calculated)
                if node_resources.cpu_capacity > 0:
                    cpu_util = node_resources.cpu_used / node_resources.cpu_capacity
                    node_cpu_utils.append(cpu_util)

                # Secondary constraint: Memory (only if enabled)
                if (substrate_network.enable_memory_constraints and
                    node_resources.memory_capacity > 0):
                    memory_util = node_resources.memory_used / node_resources.memory_capacity
                    node_memory_utils.append(memory_util)

            except Exception as e:
                logger.warning(f"Error calculating utilization for node {node_id}: {e}")
                continue

        utilization_metrics['avg_node_cpu_util'] = mean(node_cpu_utils) if node_cpu_utils else 0.0
        utilization_metrics['avg_node_memory_util'] = mean(node_memory_utils) if node_memory_utils else 0.0

        # Calculate link utilization (PRIMARY: Bandwidth)
        link_bandwidth_utils = []

        for src, dst in edges:
            try:
                link_resources = substrate_network.get_link_resources(src, dst)
                if not link_resources:
                    continue

                # Primary constraint: Bandwidth (always calculated)
                if link_resources.bandwidth_capacity > 0:
                    bandwidth_util = link_resources.bandwidth_used / link_resources.bandwidth_capacity
                    link_bandwidth_utils.append(bandwidth_util)

            except Exception as e:
                logger.warning(f"Error calculating utilization for link ({src}, {dst}): {e}")
                continue

        utilization_metrics['avg_link_bandwidth_util'] = mean(link_bandwidth_utils) if link_bandwidth_utils else 0.0

        logger.debug(f"Utilization metrics calculated: {utilization_metrics}")
        return utilization_metrics

    except Exception as e:
        raise MetricsError(f"Failed to calculate utilization metrics: {e}")


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_throughput(results: List[EmbeddingResult], time_duration: Optional[float] = None) -> float:
    """
    Calculate the throughput (successful embeddings per time unit).

    Standard Formula: Throughput = |Successful_embeddings| / Time_duration

    Args:
        results: List of embedding results
        time_duration: Total time duration, if None calculated from timestamps

    Returns:
        Throughput (successful embeddings per time unit)
    """
    successful_results = [r for r in results if r.success]

    if not successful_results:
        return 0.0

    if time_duration is None:
        # Calculate from timestamps if available
        timestamped_results = [r for r in results if r.timestamp is not None]
        if len(timestamped_results) < 2:
            return 0.0
        timestamps = [r.timestamp for r in timestamped_results]
        time_duration = max(timestamps) - min(timestamps)

    if time_duration <= 0:
        return 0.0

    return len(successful_results) / time_duration


def calculate_average_execution_time(results: List[EmbeddingResult]) -> float:
    """
    Calculate average execution time per VNR embedding attempt.

    Args:
        results: List of embedding results

    Returns:
        Average execution time in seconds
    """
    if not results:
        return 0.0

    execution_times = [result.execution_time for result in results if result.execution_time > 0]
    avg_time = mean(execution_times) if execution_times else 0.0

    logger.debug(f"Average execution time: {avg_time:.4f} seconds")
    return avg_time


# =============================================================================
# ADVANCED METRICS
# =============================================================================

def calculate_resource_efficiency(results: List[EmbeddingResult]) -> Dict[str, float]:
    """
    Calculate resource efficiency metrics.

    Args:
        results: List of embedding results

    Returns:
        Dictionary with efficiency metrics
    """
    successful_results = [r for r in results if r.success]

    if not successful_results:
        return {
            'revenue_per_unit_cost': 0.0,
            'successful_revenue_rate': 0.0,
            'cost_efficiency': 0.0
        }

    total_revenue = calculate_total_revenue(results)
    total_cost = calculate_total_cost(results)
    successful_cost = sum(r.cost for r in successful_results)

    revenue_per_unit_cost = total_revenue / total_cost if total_cost > 0 else 0.0
    successful_revenue_rate = total_revenue / len(successful_results)
    cost_efficiency = total_revenue / successful_cost if successful_cost > 0 else 0.0

    efficiency_metrics = {
        'revenue_per_unit_cost': revenue_per_unit_cost,
        'successful_revenue_rate': successful_revenue_rate,
        'cost_efficiency': cost_efficiency
    }

    logger.debug(f"Resource efficiency metrics: {efficiency_metrics}")
    return efficiency_metrics


def calculate_long_term_average_revenue(results: List[EmbeddingResult], time_duration: Optional[float] = None) -> float:
    """
    Calculate long-term average revenue (standard VNE metric).

    Standard Formula: LTR = Total_Revenue / Time_duration

    Args:
        results: List of embedding results
        time_duration: Time duration, calculated from timestamps if None

    Returns:
        Long-term average revenue
    """
    total_revenue = calculate_total_revenue(results)

    if time_duration is None:
        timestamped_results = [r for r in results if r.timestamp is not None]
        if len(timestamped_results) < 2:
            return 0.0
        timestamps = [r.timestamp for r in timestamped_results]
        time_duration = max(timestamps) - min(timestamps)

    if time_duration <= 0:
        return 0.0

    return total_revenue / time_duration


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_embedding_result_from_vnr(vnr: 'VirtualNetworkRequest',
                                    success: bool,
                                    node_mapping: Optional[Dict] = None,
                                    link_mapping: Optional[Dict] = None,
                                    execution_time: float = 0.0,
                                    failure_reason: Optional[str] = None) -> EmbeddingResult:
    """
    Create an EmbeddingResult with proper revenue and cost calculation.

    Args:
        vnr: VirtualNetworkRequest instance
        success: Whether embedding was successful
        node_mapping: Node mapping (if successful)
        link_mapping: Link mapping (if successful)
        execution_time: Time taken for embedding
        failure_reason: Reason for failure (if unsuccessful)

    Returns:
        EmbeddingResult with calculated revenue and cost
    """
    # Calculate revenue from VNR requirements
    revenue = calculate_vnr_revenue(vnr) if success else 0.0

    # Calculate cost from substrate consumption
    cost = 0.0
    path_lengths = {}

    if success and node_mapping and link_mapping:
        cost = calculate_vnr_cost(vnr, node_mapping, link_mapping)
        # Calculate path lengths for analysis
        for virtual_link, substrate_path in link_mapping.items():
            path_lengths[virtual_link] = len(substrate_path)

    return EmbeddingResult(
        vnr_id=vnr.vnr_id,
        success=success,
        revenue=revenue,
        cost=cost,
        execution_time=execution_time,
        node_mapping=node_mapping,
        link_mapping=link_mapping,
        timestamp=time.time(),
        failure_reason=failure_reason,
        vnr=vnr,
        path_lengths=path_lengths
    )


# =============================================================================
# COMPREHENSIVE METRICS SUMMARY
# =============================================================================

def generate_comprehensive_metrics_summary(results: List[EmbeddingResult],
                                          substrate_network: Optional['SubstrateNetwork'] = None,
                                          time_duration: Optional[float] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of all VNE performance metrics.

    Args:
        results: List of embedding results
        substrate_network: Optional substrate network for utilization metrics
        time_duration: Optional time duration for rate calculations

    Returns:
        Dictionary containing all VNE metrics organized by category
    """
    if not results:
        logger.warning("No results provided for metrics summary")
        return {
            'basic_stats': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0
            }
        }

    # Basic statistics
    basic_stats = {
        'total_requests': len(results),
        'successful_requests': sum(1 for r in results if r.success),
        'failed_requests': sum(1 for r in results if not r.success),
    }

    # Primary VNE metrics (standard literature metrics)
    primary_metrics = {
        'acceptance_ratio': calculate_acceptance_ratio(results),
        'blocking_probability': calculate_blocking_probability(results),
        'total_revenue': calculate_total_revenue(results),
        'total_cost': calculate_total_cost(results),
        'revenue_to_cost_ratio': calculate_revenue_to_cost_ratio(results),
    }

    # Performance metrics
    performance_metrics = {
        'average_execution_time': calculate_average_execution_time(results),
        'throughput': calculate_throughput(results, time_duration),
        'long_term_average_revenue': calculate_long_term_average_revenue(results, time_duration),
    }

    # Resource utilization (if substrate network provided)
    utilization_metrics = {}
    if substrate_network:
        try:
            utilization_metrics = calculate_utilization(substrate_network)
        except MetricsError as e:
            logger.warning(f"Could not calculate utilization metrics: {e}")

    # Resource efficiency
    efficiency_metrics = calculate_resource_efficiency(results)

    # Failure analysis
    failure_analysis = {}
    failed_results = [r for r in results if not r.success]
    if failed_results:
        failure_reasons = [r.failure_reason for r in failed_results if r.failure_reason]
        if failure_reasons:
            from collections import Counter
            failure_counts = Counter(failure_reasons)
            failure_analysis = {
                'failure_distribution': dict(failure_counts),
                'most_common_failure': failure_counts.most_common(1)[0] if failure_counts else None
            }

    # Constraint usage analysis
    constraint_analysis = {}
    if any(r.vnr for r in results):
        constraint_usage = {
            'memory_constrained_vnrs': 0,
            'delay_constrained_vnrs': 0,
            'reliability_constrained_vnrs': 0
        }

        for result in results:
            if result.vnr:
                summary = result.vnr.get_constraint_summary()
                if summary['uses_memory_constraints']:
                    constraint_usage['memory_constrained_vnrs'] += 1
                if summary['uses_delay_constraints']:
                    constraint_usage['delay_constrained_vnrs'] += 1
                if summary['uses_reliability_constraints']:
                    constraint_usage['reliability_constrained_vnrs'] += 1

        constraint_analysis = constraint_usage

    summary = {
        'basic_stats': basic_stats,
        'primary_metrics': primary_metrics,
        'performance_metrics': performance_metrics,
        'utilization_metrics': utilization_metrics,
        'efficiency_metrics': efficiency_metrics,
        'failure_analysis': failure_analysis,
        'constraint_analysis': constraint_analysis,
        'metadata': {
            'calculation_timestamp': time.time(),
            'metrics_version': '1.0_standard_vne',
            'primary_constraints': ['cpu', 'bandwidth'],
            'secondary_constraints': ['memory', 'delay', 'reliability']
        }
    }

    logger.info(f"Generated comprehensive metrics summary with {len([k for v in summary.values() for k in (v if isinstance(v, dict) else {})])} total metrics")

    return summary


# =============================================================================
# METRICS LISTING FOR REFERENCE
# =============================================================================

def list_available_metrics() -> Dict[str, List[str]]:
    """
    List all available VNE metrics organized by category.

    Returns:
        Dictionary of metric categories and their metrics
    """
    return {
        'Primary VNE Metrics (Literature Standard)': [
            'acceptance_ratio',
            'blocking_probability',
            'total_revenue',
            'total_cost',
            'revenue_to_cost_ratio'
        ],
        'Performance Metrics': [
            'average_execution_time',
            'throughput',
            'long_term_average_revenue'
        ],
        'Resource Utilization': [
            'avg_node_cpu_util',
            'avg_node_memory_util',
            'avg_link_bandwidth_util'
        ],
        'Efficiency Metrics': [
            'revenue_per_unit_cost',
            'successful_revenue_rate',
            'cost_efficiency'
        ],
        'Analysis Metrics': [
            'failure_analysis',
            'constraint_analysis',
            'time_series_metrics'
        ]
    }
