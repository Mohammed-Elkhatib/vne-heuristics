"""
Performance metrics calculations for Virtual Network Embedding (VNE).

This module provides functions to calculate various VNE performance metrics
including acceptance ratios, revenue, costs, and resource utilization.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from statistics import mean, stdev
import time

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Represents the result of a VNR embedding attempt.
    
    Attributes:
        vnr_id: Unique identifier for the VNR
        success: Whether the embedding was successful
        revenue: Revenue generated if successful, 0 otherwise
        cost: Cost incurred for the embedding
        execution_time: Time taken for the embedding attempt (seconds)
        node_mapping: Dictionary mapping virtual nodes to substrate nodes
        link_mapping: Dictionary mapping virtual links to substrate paths
        timestamp: When the embedding was attempted
        failure_reason: Reason for failure if unsuccessful
    """
    vnr_id: str
    success: bool
    revenue: float = 0.0
    cost: float = 0.0
    execution_time: float = 0.0
    node_mapping: Optional[Dict[str, str]] = None
    link_mapping: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None
    timestamp: Optional[float] = None
    failure_reason: Optional[str] = None


def calculate_acceptance_ratio(results: List[EmbeddingResult]) -> float:
    """
    Calculate the acceptance ratio of VNR embedding attempts.
    
    Args:
        results: List of embedding results
        
    Returns:
        Acceptance ratio (0.0 to 1.0)
        
    Raises:
        ValueError: If results list is empty
        
    Example:
        >>> results = [EmbeddingResult("vnr1", True), EmbeddingResult("vnr2", False)]
        >>> ratio = calculate_acceptance_ratio(results)
        >>> print(f"Acceptance ratio: {ratio:.2%}")
    """
    if not results:
        raise ValueError("Results list cannot be empty")
    
    successful = sum(1 for result in results if result.success)
    total = len(results)
    
    ratio = successful / total
    logger.info(f"Acceptance ratio: {successful}/{total} = {ratio:.4f}")
    
    return ratio


def calculate_total_revenue(results: List[EmbeddingResult]) -> float:
    """
    Calculate total revenue from successful VNR embeddings.
    
    Args:
        results: List of embedding results
        
    Returns:
        Total revenue from successful embeddings
        
    Example:
        >>> results = [EmbeddingResult("vnr1", True, revenue=100.0)]
        >>> revenue = calculate_total_revenue(results)
    """
    total_revenue = sum(result.revenue for result in results if result.success)
    logger.info(f"Total revenue: {total_revenue:.2f}")
    
    return total_revenue


def calculate_total_cost(results: List[EmbeddingResult]) -> float:
    """
    Calculate total cost from all VNR embedding attempts.
    
    Args:
        results: List of embedding results
        
    Returns:
        Total cost from all embedding attempts
        
    Example:
        >>> results = [EmbeddingResult("vnr1", True, cost=50.0)]
        >>> cost = calculate_total_cost(results)
    """
    total_cost = sum(result.cost for result in results)
    logger.info(f"Total cost: {total_cost:.2f}")
    
    return total_cost


def calculate_revenue_to_cost_ratio(results: List[EmbeddingResult]) -> float:
    """
    Calculate the revenue-to-cost ratio.
    
    Args:
        results: List of embedding results
        
    Returns:
        Revenue-to-cost ratio, or 0.0 if total cost is 0
        
    Example:
        >>> results = [EmbeddingResult("vnr1", True, revenue=100.0, cost=50.0)]
        >>> ratio = calculate_revenue_to_cost_ratio(results)
    """
    total_revenue = calculate_total_revenue(results)
    total_cost = calculate_total_cost(results)
    
    if total_cost == 0:
        logger.warning("Total cost is zero, returning 0.0 for revenue-to-cost ratio")
        return 0.0
    
    ratio = total_revenue / total_cost
    logger.info(f"Revenue-to-cost ratio: {ratio:.4f}")
    
    return ratio


def calculate_utilization(substrate_network) -> Dict[str, float]:
    """
    Calculate average resource utilization for substrate network.
    
    Args:
        substrate_network: SubstrateNetwork instance
        
    Returns:
        Dictionary with utilization metrics:
        - avg_node_cpu_util: Average CPU utilization across nodes
        - avg_node_memory_util: Average memory utilization across nodes
        - avg_link_bandwidth_util: Average bandwidth utilization across links
        
    Example:
        >>> utilization = calculate_utilization(substrate)
        >>> print(f"CPU utilization: {utilization['avg_node_cpu_util']:.2%}")
    """
    utilization_metrics = {}
    
    # Calculate node utilization
    node_cpu_utils = []
    node_memory_utils = []
    
    for node_id in substrate_network.nodes:
        node_resources = substrate_network.nodes[node_id]
        
        if node_resources.cpu_capacity > 0:
            cpu_util = (node_resources.cpu_capacity - node_resources.available_cpu) / node_resources.cpu_capacity
            node_cpu_utils.append(cpu_util)
        
        if node_resources.memory_capacity > 0:
            memory_util = (node_resources.memory_capacity - node_resources.available_memory) / node_resources.memory_capacity
            node_memory_utils.append(memory_util)
    
    utilization_metrics['avg_node_cpu_util'] = mean(node_cpu_utils) if node_cpu_utils else 0.0
    utilization_metrics['avg_node_memory_util'] = mean(node_memory_utils) if node_memory_utils else 0.0
    
    # Calculate link utilization
    link_bandwidth_utils = []
    
    for link_id in substrate_network.links:
        link_resources = substrate_network.links[link_id]
        
        if link_resources.bandwidth_capacity > 0:
            bandwidth_util = (link_resources.bandwidth_capacity - link_resources.available_bandwidth) / link_resources.bandwidth_capacity
            link_bandwidth_utils.append(bandwidth_util)
    
    utilization_metrics['avg_link_bandwidth_util'] = mean(link_bandwidth_utils) if link_bandwidth_utils else 0.0
    
    logger.info(f"Utilization metrics calculated: {utilization_metrics}")
    
    return utilization_metrics


def calculate_average_execution_time(results: List[EmbeddingResult]) -> float:
    """
    Calculate average execution time per VNR embedding attempt.
    
    Args:
        results: List of embedding results
        
    Returns:
        Average execution time in seconds
        
    Example:
        >>> avg_time = calculate_average_execution_time(results)
        >>> print(f"Average execution time: {avg_time:.4f} seconds")
    """
    if not results:
        return 0.0
    
    execution_times = [result.execution_time for result in results if result.execution_time > 0]
    avg_time = mean(execution_times) if execution_times else 0.0
    
    logger.info(f"Average execution time: {avg_time:.4f} seconds")
    
    return avg_time


def calculate_resource_efficiency(results: List[EmbeddingResult]) -> Dict[str, float]:
    """
    Calculate resource efficiency metrics.
    
    Args:
        results: List of embedding results
        
    Returns:
        Dictionary with efficiency metrics:
        - revenue_per_unit_cost: Revenue generated per unit of cost
        - successful_revenue_rate: Revenue rate from successful embeddings only
        
    Example:
        >>> efficiency = calculate_resource_efficiency(results)
    """
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        return {
            'revenue_per_unit_cost': 0.0,
            'successful_revenue_rate': 0.0
        }
    
    total_revenue = sum(r.revenue for r in successful_results)
    total_cost = sum(r.cost for r in successful_results)
    
    revenue_per_unit_cost = total_revenue / total_cost if total_cost > 0 else 0.0
    successful_revenue_rate = total_revenue / len(successful_results)
    
    efficiency_metrics = {
        'revenue_per_unit_cost': revenue_per_unit_cost,
        'successful_revenue_rate': successful_revenue_rate
    }
    
    logger.info(f"Resource efficiency metrics: {efficiency_metrics}")
    
    return efficiency_metrics


def calculate_moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average over a specified window.
    
    Args:
        values: List of numerical values
        window_size: Size of the moving window
        
    Returns:
        List of moving averages
        
    Raises:
        ValueError: If window_size is invalid
        
    Example:
        >>> values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> moving_avg = calculate_moving_average(values, 3)
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    if window_size > len(values):
        raise ValueError("Window size cannot be larger than the number of values")
    
    moving_averages = []
    
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        moving_averages.append(mean(window))
    
    return moving_averages


def aggregate_metrics_across_runs(metric_lists: List[List[float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple experimental runs.
    
    Args:
        metric_lists: List of metric lists from different runs
        
    Returns:
        Dictionary with aggregated statistics:
        - mean: Mean across all runs
        - std: Standard deviation across all runs
        - min: Minimum value across all runs
        - max: Maximum value across all runs
        - count: Total number of data points
        
    Example:
        >>> run1_metrics = [0.8, 0.9, 0.7]
        >>> run2_metrics = [0.85, 0.88, 0.75]
        >>> aggregated = aggregate_metrics_across_runs([run1_metrics, run2_metrics])
    """
    if not metric_lists:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    
    # Flatten all metrics into a single list
    all_metrics = [metric for run_metrics in metric_lists for metric in run_metrics]
    
    if not all_metrics:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    
    aggregated = {
        'mean': mean(all_metrics),
        'std': stdev(all_metrics) if len(all_metrics) > 1 else 0.0,
        'min': min(all_metrics),
        'max': max(all_metrics),
        'count': len(all_metrics)
    }
    
    logger.info(f"Aggregated metrics: {aggregated}")
    
    return aggregated


def generate_metrics_summary(results: List[EmbeddingResult], 
                           substrate_network=None) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of VNE performance metrics.
    
    Args:
        results: List of embedding results
        substrate_network: Optional substrate network for utilization metrics
        
    Returns:
        Dictionary containing all calculated metrics
        
    Example:
        >>> summary = generate_metrics_summary(results, substrate)
        >>> print(f"Summary: {summary}")
    """
    if not results:
        logger.warning("No results provided for metrics summary")
        return {}
    
    summary = {
        'total_requests': len(results),
        'successful_requests': sum(1 for r in results if r.success),
        'failed_requests': sum(1 for r in results if not r.success),
        'acceptance_ratio': calculate_acceptance_ratio(results),
        'total_revenue': calculate_total_revenue(results),
        'total_cost': calculate_total_cost(results),
        'revenue_to_cost_ratio': calculate_revenue_to_cost_ratio(results),
        'average_execution_time': calculate_average_execution_time(results),
    }
    
    # Add resource efficiency metrics
    efficiency = calculate_resource_efficiency(results)
    summary.update(efficiency)
    
    # Add utilization metrics if substrate network is provided
    if substrate_network:
        utilization = calculate_utilization(substrate_network)
        summary.update(utilization)
    
    # Add time-based metrics if timestamps are available
    timestamped_results = [r for r in results if r.timestamp is not None]
    if timestamped_results:
        timestamps = [r.timestamp for r in timestamped_results]
        summary['experiment_duration'] = max(timestamps) - min(timestamps)
        summary['average_inter_arrival_time'] = summary['experiment_duration'] / len(timestamped_results) if len(timestamped_results) > 1 else 0.0
    
    logger.info(f"Generated metrics summary with {len(summary)} metrics")
    
    return summary


def calculate_time_series_metrics(results: List[EmbeddingResult], 
                                time_window: float = 3600.0) -> Dict[str, List[float]]:
    """
    Calculate metrics over time windows for time series analysis.
    
    Args:
        results: List of embedding results with timestamps
        time_window: Time window size in seconds (default: 1 hour)
        
    Returns:
        Dictionary with time series metrics:
        - timestamps: Time window start times
        - acceptance_ratios: Acceptance ratio for each window
        - revenues: Total revenue for each window
        - costs: Total cost for each window
        
    Example:
        >>> time_series = calculate_time_series_metrics(results, 1800.0)  # 30 min windows
    """
    # Filter results with timestamps
    timestamped_results = [r for r in results if r.timestamp is not None]
    
    if not timestamped_results:
        logger.warning("No timestamped results available for time series analysis")
        return {'timestamps': [], 'acceptance_ratios': [], 'revenues': [], 'costs': []}
    
    # Sort by timestamp
    timestamped_results.sort(key=lambda x: x.timestamp)
    
    start_time = min(r.timestamp for r in timestamped_results)
    end_time = max(r.timestamp for r in timestamped_results)
    
    # Create time windows
    time_series = {
        'timestamps': [],
        'acceptance_ratios': [],
        'revenues': [],
        'costs': []
    }
    
    current_time = start_time
    while current_time < end_time:
        window_end = current_time + time_window
        
        # Get results in current window
        window_results = [r for r in timestamped_results 
                         if current_time <= r.timestamp < window_end]
        
        if window_results:
            time_series['timestamps'].append(current_time)
            time_series['acceptance_ratios'].append(calculate_acceptance_ratio(window_results))
            time_series['revenues'].append(calculate_total_revenue(window_results))
            time_series['costs'].append(calculate_total_cost(window_results))
        
        current_time = window_end
    
    logger.info(f"Generated time series with {len(time_series['timestamps'])} windows")
    
    return time_series
