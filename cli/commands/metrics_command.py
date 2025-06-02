"""
Metrics command implementation using the metrics module.
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from .base_command import BaseCommand
from cli.exceptions import CommandError, FileError

logger = logging.getLogger(__name__)


class MetricsCommand(BaseCommand):
    """Command for calculating comprehensive VNE metrics from algorithm results."""

    def execute(self, args) -> int:
        """Execute the metrics command with full implementation."""
        try:
            # Validate required arguments
            self._validate_metrics_arguments(args)

            # Load results data
            results_data = self._load_results_file(args.results)

            # Load substrate network if provided (for utilization metrics)
            substrate_network = None
            if getattr(args, 'substrate', None):
                substrate_network = self._load_substrate_network(args.substrate)

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                results_data, substrate_network, args
            )

            # Save metrics to output file
            output_path = self._save_metrics(metrics, args)

            # Print summary
            self._print_metrics_summary(metrics, output_path)

            return 0

        except (CommandError, FileError):
            # These are already handled by the error handler
            raise
        except Exception as e:
            self.error_handler.handle_unexpected_error(e) if self.error_handler else print(f"Error: {e}")
            return 1

    def _validate_metrics_arguments(self, args) -> None:
        """Validate arguments for metrics calculation."""
        if not getattr(args, 'results', None):
            raise CommandError("--results is required for metrics calculation")
        if not getattr(args, 'output', None):
            raise CommandError("--output is required for metrics calculation")

        # Check if results file exists
        results_path = Path(args.results)
        if not results_path.exists():
            raise FileError(f"Results file not found: {args.results}")

    def _load_results_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load results from JSON file with proper error handling."""
        try:
            results_path = Path(filepath)

            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different result file formats
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                logger.info(f"Loaded {len(results)} results from {filepath}")
            elif isinstance(data, list):
                results = data
                logger.info(f"Loaded {len(results)} results from {filepath}")
            else:
                raise FileError("Invalid results file format")

            return results

        except json.JSONDecodeError as e:
            raise FileError(f"Invalid JSON format in results file: {e}")
        except Exception as e:
            raise FileError(f"Failed to load results file {filepath}: {e}")

    def _load_substrate_network(self, filepath: str):
        """Load substrate network for utilization metrics."""
        try:
            from src.models.substrate import SubstrateNetwork

            substrate = SubstrateNetwork()

            # Handle different file naming conventions
            base_name = filepath.replace('.csv', '') if filepath.endswith('.csv') else filepath
            nodes_file = f"{base_name}_nodes.csv"
            links_file = f"{base_name}_links.csv"

            if not Path(nodes_file).exists() or not Path(links_file).exists():
                raise FileError(f"Substrate files not found: {nodes_file}, {links_file}")

            substrate.load_from_csv(nodes_file, links_file)
            logger.info(f"Loaded substrate network for utilization metrics: {substrate}")

            return substrate

        except Exception as e:
            logger.warning(f"Could not load substrate network: {e}")
            return None

    def _calculate_comprehensive_metrics(self, results_data: List[Dict[str, Any]],
                                       substrate_network, args) -> Dict[str, Any]:
        """Calculate comprehensive VNE metrics using the metrics module."""
        try:
            # Import metrics utilities
            from src.utils.metrics import (
                EmbeddingResult as MetricsEmbeddingResult,
                generate_comprehensive_metrics_summary
            )

            # Convert results to metrics module format
            metrics_results = []
            for result_dict in results_data:
                # Convert link mapping back to proper format
                link_mapping = {}
                if 'link_mapping' in result_dict:
                    for key, path in result_dict['link_mapping'].items():
                        if '-' in key:
                            src, dst = key.split('-', 1)
                            link_mapping[(src, dst)] = path
                        else:
                            link_mapping[key] = path

                metrics_result = MetricsEmbeddingResult(
                    vnr_id=result_dict.get('vnr_id', ''),
                    success=result_dict.get('success', False),
                    revenue=float(result_dict.get('revenue', 0.0)),
                    cost=float(result_dict.get('cost', 0.0)),
                    execution_time=float(result_dict.get('execution_time', 0.0)),
                    node_mapping=result_dict.get('node_mapping', {}),
                    link_mapping=link_mapping,
                    timestamp=result_dict.get('timestamp'),
                    failure_reason=result_dict.get('failure_reason')
                )
                metrics_results.append(metrics_result)

            # Calculate time duration for rate metrics
            time_duration = None
            if getattr(args, 'time_series', False):
                timestamps = [r.timestamp for r in metrics_results if r.timestamp is not None]
                if len(timestamps) >= 2:
                    time_duration = max(timestamps) - min(timestamps)

            # Generate comprehensive metrics
            metrics = generate_comprehensive_metrics_summary(
                metrics_results, substrate_network, time_duration
            )

            # Add time series metrics if requested
            if getattr(args, 'time_series', False):
                metrics['time_series_metrics'] = self._calculate_time_series_metrics(
                    metrics_results, getattr(args, 'window_size', 3600.0)
                )

            logger.info("Comprehensive metrics calculated successfully")
            return metrics

        except ImportError as e:
            raise CommandError(f"Failed to import metrics module: {e}")
        except Exception as e:
            raise CommandError(f"Failed to calculate metrics: {e}")

    def _calculate_time_series_metrics(self, results: List, window_size: float) -> Dict[str, Any]:
        """Calculate time series metrics with sliding windows."""
        try:
            # Sort results by timestamp
            timestamped_results = [r for r in results if r.timestamp is not None]
            timestamped_results.sort(key=lambda x: x.timestamp)

            if len(timestamped_results) < 2:
                return {'error': 'Insufficient timestamped data for time series analysis'}

            # Calculate sliding window metrics
            windows = []
            start_time = timestamped_results[0].timestamp
            end_time = timestamped_results[-1].timestamp

            current_time = start_time
            while current_time < end_time:
                window_end = current_time + window_size
                window_results = [r for r in timestamped_results
                                if current_time <= r.timestamp < window_end]

                if window_results:
                    window_metrics = {
                        'window_start': current_time,
                        'window_end': window_end,
                        'total_requests': len(window_results),
                        'successful_requests': sum(1 for r in window_results if r.success),
                        'acceptance_ratio': sum(1 for r in window_results if r.success) / len(window_results),
                        'total_revenue': sum(r.revenue for r in window_results if r.success),
                        'avg_execution_time': sum(r.execution_time for r in window_results) / len(window_results)
                    }
                    windows.append(window_metrics)

                current_time += window_size / 2  # 50% overlap

            return {
                'window_size_seconds': window_size,
                'num_windows': len(windows),
                'windows': windows
            }

        except Exception as e:
            logger.error(f"Error calculating time series metrics: {e}")
            return {'error': str(e)}

    def _save_metrics(self, metrics: Dict[str, Any], args) -> Path:
        """Save metrics to output file in specified format."""
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = getattr(args, 'format', 'csv').lower()

            if format_type == 'json':
                self._save_metrics_json(metrics, output_path)
            elif format_type == 'csv':
                self._save_metrics_csv(metrics, output_path)
            else:
                raise CommandError(f"Unsupported output format: {format_type}")

            logger.info(f"Saved metrics to {output_path}")
            return output_path

        except Exception as e:
            raise FileError(f"Failed to save metrics: {e}")

    def _save_metrics_json(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics in JSON format."""
        # Add metadata
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'metrics_version': '1.0_standard_vne'
            },
            'metrics': metrics
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    def _save_metrics_csv(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics in CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['metric_category', 'metric_name', 'value', 'description'])

            # Write basic stats
            basic_stats = metrics.get('basic_stats', {})
            for key, value in basic_stats.items():
                writer.writerow(['basic_stats', key, value, f'Basic statistic: {key}'])

            # Write primary metrics
            primary_metrics = metrics.get('primary_metrics', {})
            for key, value in primary_metrics.items():
                writer.writerow(['primary_metrics', key, value, f'Primary VNE metric: {key}'])

            # Write performance metrics
            performance_metrics = metrics.get('performance_metrics', {})
            for key, value in performance_metrics.items():
                writer.writerow(['performance_metrics', key, value, f'Performance metric: {key}'])

            # Write utilization metrics
            utilization_metrics = metrics.get('utilization_metrics', {})
            for key, value in utilization_metrics.items():
                writer.writerow(['utilization_metrics', key, value, f'Resource utilization: {key}'])

            # Write efficiency metrics
            efficiency_metrics = metrics.get('efficiency_metrics', {})
            for key, value in efficiency_metrics.items():
                writer.writerow(['efficiency_metrics', key, value, f'Efficiency metric: {key}'])

    def _print_metrics_summary(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Print a summary of calculated metrics."""
        print(f"\n{'=' * 60}")
        print(f"METRICS CALCULATION SUMMARY")
        print(f"{'=' * 60}")

        # Basic statistics
        basic_stats = metrics.get('basic_stats', {})
        if basic_stats:
            print(f"Total VNRs processed: {basic_stats.get('total_requests', 0)}")
            print(f"Successful embeddings: {basic_stats.get('successful_requests', 0)}")
            print(f"Failed embeddings: {basic_stats.get('failed_requests', 0)}")

        # Primary metrics
        primary_metrics = metrics.get('primary_metrics', {})
        if primary_metrics:
            print(f"\nPrimary VNE Metrics:")
            print(f"  Acceptance Ratio: {primary_metrics.get('acceptance_ratio', 0):.3f}")
            print(f"  Blocking Probability: {primary_metrics.get('blocking_probability', 0):.3f}")
            print(f"  Total Revenue: {primary_metrics.get('total_revenue', 0):.2f}")
            print(f"  Total Cost: {primary_metrics.get('total_cost', 0):.2f}")
            print(f"  Revenue-to-Cost Ratio: {primary_metrics.get('revenue_to_cost_ratio', 0):.3f}")

        # Performance metrics
        performance_metrics = metrics.get('performance_metrics', {})
        if performance_metrics:
            print(f"\nPerformance Metrics:")
            print(f"  Average Execution Time: {performance_metrics.get('average_execution_time', 0):.4f}s")
            print(f"  Throughput: {performance_metrics.get('throughput', 0):.2f} VNRs/time_unit")
            print(f"  Long-term Average Revenue: {performance_metrics.get('long_term_average_revenue', 0):.2f}")

        # Utilization metrics (if available)
        utilization_metrics = metrics.get('utilization_metrics', {})
        if utilization_metrics:
            print(f"\nResource Utilization:")
            print(f"  Average CPU Utilization: {utilization_metrics.get('avg_node_cpu_util', 0):.3f}")
            print(f"  Average Memory Utilization: {utilization_metrics.get('avg_node_memory_util', 0):.3f}")
            print(f"  Average Bandwidth Utilization: {utilization_metrics.get('avg_link_bandwidth_util', 0):.3f}")

        # Time series info (if available)
        time_series = metrics.get('time_series_metrics', {})
        if time_series and 'windows' in time_series:
            print(f"\nTime Series Analysis:")
            print(f"  Number of time windows: {time_series.get('num_windows', 0)}")
            print(f"  Window size: {time_series.get('window_size_seconds', 0):.1f} seconds")

        print(f"\nMetrics saved to: {output_path}")
        print(f"{'=' * 60}")
