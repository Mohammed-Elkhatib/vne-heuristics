#!/usr/bin/env python3
"""
Main command-line interface for the VNE (Virtual Network Embedding) project.

This module provides the primary entry point for the VNE heuristics project,
offering commands for network generation, algorithm execution, and metrics calculation.

Usage Examples:
    # Generate networks
    python main.py generate substrate --nodes 20 --topology erdos_renyi --save data/substrate_20.csv
    python main.py generate vnrs --count 50 --substrate data/substrate_20.csv --save data/vnrs_batch1.csv

    # Run algorithms
    python main.py run --algorithm yu2008 --substrate data/substrate_20.csv --vnrs data/vnrs_batch1.csv --mode batch
    python main.py run --algorithm yu2008 --substrate data/substrate_20.csv --vnrs data/vnrs_batch1.csv --mode online

    # Calculate metrics
    python main.py metrics --results data/output/results.json --output data/output/metrics.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import importlib
import pkgutil

# Import project modules
from .config_management import ConfigurationManager, VNEConfig, ConfigurationError, load_config_from_args
from src.models.substrate import SubstrateNetwork
from src.models.virtual_request import VirtualNetworkRequest, VNRBatch
from src.utils.generators import (
    generate_substrate_network, generate_vnr_batch,
    NetworkGenerationConfig, set_random_seed
)
from src.utils.metrics import generate_metrics_summary, calculate_time_series_metrics
from src.utils.io_utils import (
    save_substrate_to_csv, load_substrate_from_csv,
    save_vnrs_to_csv, load_vnrs_from_csv,
    save_results_to_file, load_results_from_file,
    export_metrics_to_csv, create_experiment_directory
)
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult


class VNECLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass


class VNECommandLineInterface:
    """
    Main command-line interface for the VNE project.

    This class handles argument parsing, command routing, and provides
    the primary user interface for all VNE operations.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.config_manager = ConfigurationManager()
        self.config: Optional[VNEConfig] = None
        self.logger = logging.getLogger(__name__)
        self.available_algorithms: Dict[str, type] = {}

        # Discover available algorithms
        self._discover_algorithms()

    def _discover_algorithms(self) -> None:
        """Discover available algorithm implementations."""
        try:
            # Import and register available algorithms
            from src.algorithms.baseline.yu_2008 import YuAlgorithm

            self.available_algorithms = {
                'yu2008': YuAlgorithm,
            }

            self.logger.debug(f"Discovered {len(self.available_algorithms)} algorithms")

        except ImportError as e:
            self.logger.warning(f"Could not import algorithms: {e}")
            self.available_algorithms = {}
        except Exception as e:
            self.logger.warning(f"Algorithm discovery failed: {e}")
            self.available_algorithms = {}

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create the main argument parser with all subcommands.

        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            prog='VNE Heuristics',
            description='Virtual Network Embedding Heuristics Toolkit',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate a substrate network
  %(prog)s generate substrate --nodes 20 --save data/substrate_20.csv

  # Generate VNRs for a substrate
  %(prog)s generate vnrs --count 50 --substrate data/substrate_20.csv --save data/vnrs_batch1.csv

  # Run algorithm in batch mode
  %(prog)s run --algorithm yu2008 --substrate data/substrate_20.csv --vnrs data/vnrs_batch1.csv --mode batch

  # Calculate metrics from results
  %(prog)s metrics --results data/output/results.json --output data/output/metrics.csv

  # Create default configuration file
  %(prog)s config --create-default config.yaml
            """
        )

        # Global arguments
        parser.add_argument('--config', '-c', type=str, metavar='FILE',
                            help='Configuration file path (YAML/JSON)')
        parser.add_argument('--verbose', '-v', action='store_true',
                            help='Enable verbose output')
        parser.add_argument('--quiet', '-q', action='store_true',
                            help='Suppress non-essential output')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        parser.add_argument('--data-dir', type=str, metavar='DIR',
                            help='Base data directory')

        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Generate command
        self._add_generate_command(subparsers)

        # Run command
        self._add_run_command(subparsers)

        # Metrics command
        self._add_metrics_command(subparsers)

        # Config command
        self._add_config_command(subparsers)

        return parser

    def _add_generate_command(self, subparsers) -> None:
        """Add generate command and subcommands."""
        generate_parser = subparsers.add_parser('generate', help='Generate networks and VNRs')
        generate_subparsers = generate_parser.add_subparsers(dest='generate_type',
                                                             help='What to generate')

        # Generate substrate network
        substrate_parser = generate_subparsers.add_parser('substrate',
                                                          help='Generate substrate network')
        substrate_parser.add_argument('--nodes', '-n', type=int, default=50,
                                      help='Number of substrate nodes (default: 50)')
        substrate_parser.add_argument('--topology', '-t', type=str, default='erdos_renyi',
                                      choices=['erdos_renyi', 'barabasi_albert', 'grid'],
                                      help='Network topology (default: erdos_renyi)')
        substrate_parser.add_argument('--edge-prob', type=float, default=0.15,
                                      help='Edge probability for Erdős-Rényi (default: 0.15)')
        substrate_parser.add_argument('--attachment-count', type=int, default=3,
                                      help='Attachment count for Barabási-Albert (default: 3)')
        substrate_parser.add_argument('--cpu-range', nargs=2, type=int, default=[50, 200],
                                      metavar=('MIN', 'MAX'), help='CPU capacity range')
        substrate_parser.add_argument('--memory-range', nargs=2, type=int, default=[50, 200],
                                      metavar=('MIN', 'MAX'), help='Memory capacity range')
        substrate_parser.add_argument('--bandwidth-range', nargs=2, type=int, default=[50, 200],
                                      metavar=('MIN', 'MAX'), help='Bandwidth capacity range')
        substrate_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        substrate_parser.add_argument('--save', '-s', type=str, required=True,
                                      help='Output file path (CSV format)')

        # Generate VNRs
        vnr_parser = generate_subparsers.add_parser('vnrs', help='Generate VNR batch')
        vnr_parser.add_argument('--count', '-c', type=int, default=100,
                                help='Number of VNRs to generate (default: 100)')
        vnr_parser.add_argument('--substrate', type=str, required=True,
                                help='Substrate network file (CSV format)')
        vnr_parser.add_argument('--nodes-range', nargs=2, type=int, default=[2, 10],
                                metavar=('MIN', 'MAX'), help='VNR node count range')
        vnr_parser.add_argument('--topology', type=str, default='random',
                                choices=['random', 'star', 'linear', 'tree'],
                                help='VNR topology (default: random)')
        vnr_parser.add_argument('--edge-prob', type=float, default=0.5,
                                help='Edge probability for random topology (default: 0.5)')
        vnr_parser.add_argument('--cpu-ratio', nargs=2, type=float, default=[0.1, 0.3],
                                metavar=('MIN', 'MAX'), help='CPU requirement ratio range')
        vnr_parser.add_argument('--memory-ratio', nargs=2, type=float, default=[0.1, 0.3],
                                metavar=('MIN', 'MAX'), help='Memory requirement ratio range')
        vnr_parser.add_argument('--bandwidth-ratio', nargs=2, type=float, default=[0.1, 0.3],
                                metavar=('MIN', 'MAX'), help='Bandwidth requirement ratio range')
        vnr_parser.add_argument('--arrival-rate', type=float, default=10.0,
                                help='VNR arrival rate (default: 10.0)')
        vnr_parser.add_argument('--lifetime-mean', type=float, default=1000.0,
                                help='Mean VNR lifetime (default: 1000.0)')
        vnr_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        vnr_parser.add_argument('--save', '-s', type=str, required=True,
                                help='Output file path (CSV format)')

    def _add_run_command(self, subparsers) -> None:
        """Add run command for algorithm execution."""
        run_parser = subparsers.add_parser('run', help='Run VNE algorithms')
        run_parser.add_argument('--algorithm', '-a', type=str, required=True,
                                help='Algorithm to use')
        run_parser.add_argument('--list-algorithms', action='store_true',
                                help='List available algorithms and exit')
        run_parser.add_argument('--substrate', type=str, required=True,
                                help='Substrate network file (CSV format)')
        run_parser.add_argument('--vnrs', type=str, required=True,
                                help='VNR batch file (CSV format)')
        run_parser.add_argument('--mode', type=str, default='batch',
                                choices=['batch', 'online'],
                                help='Execution mode (default: batch)')
        run_parser.add_argument('--output', '-o', type=str,
                                help='Output file for results (default: auto-generated)')
        run_parser.add_argument('--format', type=str, default='json',
                                choices=['json', 'csv'],
                                help='Output format (default: json)')
        run_parser.add_argument('--progress', action='store_true',
                                help='Show progress during execution')
        run_parser.add_argument('--timeout', type=float, default=300.0,
                                help='Timeout per VNR in seconds (default: 300)')

    def _add_metrics_command(self, subparsers) -> None:
        """Add metrics command for result analysis."""
        metrics_parser = subparsers.add_parser('metrics', help='Calculate metrics from results')
        metrics_parser.add_argument('--results', '-r', type=str, required=True,
                                    help='Results file to analyze')
        metrics_parser.add_argument('--output', '-o', type=str, required=True,
                                    help='Output file for metrics')
        metrics_parser.add_argument('--format', type=str, default='csv',
                                    choices=['csv', 'json'],
                                    help='Output format (default: csv)')
        metrics_parser.add_argument('--substrate', type=str,
                                    help='Substrate network file for utilization metrics')
        metrics_parser.add_argument('--time-series', action='store_true',
                                    help='Generate time series metrics')
        metrics_parser.add_argument('--window-size', type=float, default=3600.0,
                                    help='Time window size for time series (default: 3600)')

    def _add_config_command(self, subparsers) -> None:
        """Add config command for configuration management."""
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_parser.add_argument('--create-default', type=str, metavar='FILE',
                                   help='Create default configuration file')
        config_parser.add_argument('--validate', type=str, metavar='FILE',
                                   help='Validate configuration file')
        config_parser.add_argument('--show', action='store_true',
                                   help='Show current configuration')

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point for the CLI.

        Args:
            args: Command-line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parser = self.create_parser()
            parsed_args = parser.parse_args(args)

            # Handle case where no command is provided
            if not parsed_args.command:
                parser.print_help()
                return 1

            # Load configuration
            self.config = load_config_from_args(parsed_args)

            # Create necessary directories
            self.config_manager.create_directories()

            # Route to appropriate command handler
            if parsed_args.command == 'generate':
                return self._handle_generate_command(parsed_args)
            elif parsed_args.command == 'run':
                return self._handle_run_command(parsed_args)
            elif parsed_args.command == 'metrics':
                return self._handle_metrics_command(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config_command(parsed_args)
            else:
                self.logger.error(f"Unknown command: {parsed_args.command}")
                return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            return 1
        except VNECLIError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"Unexpected error: {e}")
            return 1

    def _handle_generate_command(self, args) -> int:
        """Handle generate command and subcommands."""
        if not args.generate_type:
            print("Error: Must specify what to generate (substrate or vnrs)")
            return 1

        if args.generate_type == 'substrate':
            return self._generate_substrate(args)
        elif args.generate_type == 'vnrs':
            return self._generate_vnrs(args)
        else:
            print(f"Error: Unknown generate type: {args.generate_type}")
            return 1

    def _generate_substrate(self, args) -> int:
        """Generate substrate network."""
        try:
            # Set random seed if provided
            if args.seed:
                set_random_seed(args.seed)
                self.logger.info(f"Random seed set to {args.seed}")

            # Generate substrate network
            print(f"Generating substrate network with {args.nodes} nodes...")

            substrate = generate_substrate_network(
                nodes=args.nodes,
                topology=args.topology,
                edge_probability=args.edge_prob,
                attachment_count=args.attachment_count,
                cpu_range=tuple(args.cpu_range),
                memory_range=tuple(args.memory_range),
                bandwidth_range=tuple(args.bandwidth_range)
            )

            # Save to file
            print(f"Saving substrate network to {args.save}...")
            save_substrate_to_csv(substrate, args.save)

            print(f"✓ Successfully generated substrate network: {args.save}")
            print(f"  - Nodes: {args.nodes}")
            print(f"  - Topology: {args.topology}")
            print(f"  - CPU range: {args.cpu_range[0]}-{args.cpu_range[1]}")
            print(f"  - Memory range: {args.memory_range[0]}-{args.memory_range[1]}")
            print(f"  - Bandwidth range: {args.bandwidth_range[0]}-{args.bandwidth_range[1]}")

            return 0

        except Exception as e:
            self.logger.error(f"Substrate generation failed: {e}")
            print(f"Error generating substrate network: {e}")
            return 1

    def _generate_vnrs(self, args) -> int:
        """Generate VNR batch."""
        try:
            # Load substrate network
            print(f"Loading substrate network from {args.substrate}...")
            substrate_data = load_substrate_from_csv(args.substrate)

            # Extract substrate node IDs (simplified for now)
            substrate_nodes = [f"s{i}" for i in range(len(substrate_data['nodes']))]

            # Set random seed if provided
            if args.seed:
                set_random_seed(args.seed)
                self.logger.info(f"Random seed set to {args.seed}")

            # Create generation config
            config = NetworkGenerationConfig(
                vnr_nodes_range=tuple(args.nodes_range),
                vnr_topology=args.topology,
                vnr_edge_probability=args.edge_prob,
                vnr_cpu_ratio_range=tuple(args.cpu_ratio),
                vnr_memory_ratio_range=tuple(args.memory_ratio),
                vnr_bandwidth_ratio_range=tuple(args.bandwidth_ratio),
                arrival_rate=args.arrival_rate,
                lifetime_mean=args.lifetime_mean
            )

            # Generate VNRs
            print(f"Generating {args.count} VNRs...")
            vnrs = generate_vnr_batch(
                count=args.count,
                substrate_nodes=substrate_nodes,
                config=config
            )

            # Save to file
            print(f"Saving VNRs to {args.save}...")
            save_vnrs_to_csv(vnrs, args.save)

            print(f"✓ Successfully generated VNR batch: {args.save}")
            print(f"  - Count: {args.count}")
            print(f"  - Node range: {args.nodes_range[0]}-{args.nodes_range[1]}")
            print(f"  - Topology: {args.topology}")
            print(f"  - Arrival rate: {args.arrival_rate}")
            print(f"  - Mean lifetime: {args.lifetime_mean}")

            return 0

        except Exception as e:
            self.logger.error(f"VNR generation failed: {e}")
            print(f"Error generating VNRs: {e}")
            return 1

    def _handle_run_command(self, args) -> int:
        """Handle algorithm execution."""
        try:
            # Handle list algorithms
            if args.list_algorithms:
                return self._list_algorithms()

            # Validate algorithm
            if args.algorithm not in self.available_algorithms:
                print(f"Error: Algorithm '{args.algorithm}' not available")
                print("Available algorithms:")
                for name in self.available_algorithms.keys():
                    print(f"  - {name}")
                if not self.available_algorithms:
                    print("  (No algorithms currently implemented)")
                return 1

            # Load networks
            print(f"Loading substrate network from {args.substrate}...")
            substrate = self._load_substrate_network(args.substrate)

            print(f"Loading VNRs from {args.vnrs}...")
            vnrs = self._load_vnr_batch(args.vnrs)

            # Initialize algorithm
            algorithm_class = self.available_algorithms[args.algorithm]
            algorithm = algorithm_class()

            # Execute algorithm
            print(f"Running {args.algorithm} algorithm in {args.mode} mode...")
            start_time = time.time()

            if args.mode == 'batch':
                results = algorithm.embed_batch(vnrs, substrate)
            elif args.mode == 'online':
                results = algorithm.embed_online(vnrs, substrate)
            else:
                raise VNECLIError(f"Unknown mode: {args.mode}")

            execution_time = time.time() - start_time

            # Generate output filename if not provided
            if not args.output:
                timestamp = int(time.time())
                args.output = f"results_{args.algorithm}_{timestamp}.{args.format}"
                output_path = Path(self.config.file_paths.results_dir) / args.output
            else:
                output_path = Path(args.output)

            # Save results
            print(f"Saving results to {output_path}...")
            save_results_to_file(results, output_path, args.format)

            # Print summary
            self._print_execution_summary(results, execution_time, args.algorithm)

            return 0

        except Exception as e:
            self.logger.error(f"Algorithm execution failed: {e}")
            print(f"Error running algorithm: {e}")
            return 1

    def _handle_metrics_command(self, args) -> int:
        """Handle metrics calculation."""
        try:
            # Load results
            print(f"Loading results from {args.results}...")
            results_data = load_results_from_file(args.results)

            # Convert to EmbeddingResult objects if needed
            if results_data and isinstance(results_data[0], dict):
                results = [self._dict_to_embedding_result(r) for r in results_data]
            else:
                results = results_data

            # Load substrate network if provided for utilization metrics
            substrate = None
            if args.substrate:
                print(f"Loading substrate network from {args.substrate}...")
                substrate = self._load_substrate_network(args.substrate)

            # Calculate metrics
            print("Calculating metrics...")
            metrics = generate_metrics_summary(results, substrate)

            # Calculate time series metrics if requested
            if args.time_series:
                print("Calculating time series metrics...")
                time_series = calculate_time_series_metrics(results, args.window_size)
                metrics['time_series'] = time_series

            # Save metrics
            print(f"Saving metrics to {args.output}...")
            if args.format == 'csv':
                export_metrics_to_csv(metrics, args.output)
            else:
                save_results_to_file([metrics], args.output, 'json')

            # Print summary
            self._print_metrics_summary(metrics)

            return 0

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            print(f"Error calculating metrics: {e}")
            return 1

    def _handle_config_command(self, args) -> int:
        """Handle configuration management."""
        try:
            if args.create_default:
                print(f"Creating default configuration file: {args.create_default}")
                self.config_manager.save_config(args.create_default)
                print("✓ Default configuration file created")
                return 0

            elif args.validate:
                print(f"Validating configuration file: {args.validate}")
                config = self.config_manager.load_config(args.validate)
                print("✓ Configuration file is valid")
                return 0

            elif args.show:
                print("Current configuration:")
                if self.config:
                    print(json.dumps(self.config.__dict__, indent=2, default=str))
                else:
                    # Load default config
                    config = self.config_manager.load_config()
                    print(json.dumps(config.__dict__, indent=2, default=str))
                return 0

            else:
                print("Error: Must specify a config action")
                return 1

        except Exception as e:
            self.logger.error(f"Config command failed: {e}")
            print(f"Error: {e}")
            return 1

    def _list_algorithms(self) -> int:
        """List available algorithms."""
        print("Available algorithms:")
        if self.available_algorithms:
            for name in sorted(self.available_algorithms.keys()):
                print(f"  - {name}")
        else:
            print("  (No algorithms currently implemented)")
            print("\nNote: Algorithms will be available after implementation.")
            print("Expected algorithms:")
            print("  - yu2008: Yu et al. (2008) two-stage heuristic")
            print("  - baseline1: First baseline algorithm from Fischer et al.")
            print("  - baseline2: Second baseline algorithm from Fischer et al.")

        return 0

    def _load_substrate_network(self, filepath: str) -> SubstrateNetwork:
        """Load substrate network from file."""
        # This is a placeholder - in real implementation,
        # we'd create a proper SubstrateNetwork from the CSV data
        substrate_data = load_substrate_from_csv(filepath)

        # For now, return a mock substrate network
        # In actual implementation, we'd properly construct the SubstrateNetwork
        substrate = SubstrateNetwork()

        # Add nodes from loaded data
        for node_data in substrate_data['nodes']:
            substrate.add_node(
                node_id=int(node_data['node_id'].replace('s', '')),
                cpu_capacity=node_data['cpu_capacity'],
                memory_capacity=node_data['memory_capacity'],
                x_coord=node_data.get('x_coordinate', 0.0),
                y_coord=node_data.get('y_coordinate', 0.0)
            )

        # Add links from loaded data
        for link_data in substrate_data['links']:
            substrate.add_link(
                src=int(link_data['source_node'].replace('s', '')),
                dst=int(link_data['target_node'].replace('s', '')),
                bandwidth_capacity=link_data['bandwidth_capacity'],
                delay=link_data.get('delay', 1.0),
                cost=link_data.get('cost', 1.0)
            )

        return substrate

    def _load_vnr_batch(self, filepath: str) -> List[VirtualNetworkRequest]:
        """Load VNR batch from file."""
        # This is a placeholder - in real implementation,
        # we'd create proper VirtualNetworkRequest objects from the CSV data
        vnr_data = load_vnrs_from_csv(filepath)

        vnrs = []
        for vnr_info in vnr_data:
            vnr = VirtualNetworkRequest(
                vnr_id=int(vnr_info['vnr_id']),
                arrival_time=vnr_info['arrival_time'],
                lifetime=vnr_info['lifetime'],
                priority=vnr_info.get('priority', 1)
            )

            # Add virtual nodes
            for node_id, node_data in vnr_info['virtual_nodes'].items():
                vnr.add_virtual_node(
                    node_id=int(node_id),
                    cpu_requirement=node_data['cpu_requirement'],
                    memory_requirement=node_data['memory_requirement']
                )

            # Add virtual links
            for link_id, link_data in vnr_info['virtual_links'].items():
                vnr.add_virtual_link(
                    src_node=int(link_data['source_node']),
                    dst_node=int(link_data['target_node']),
                    bandwidth_requirement=link_data['bandwidth_requirement']
                )

            vnrs.append(vnr)

        return vnrs

    def _dict_to_embedding_result(self, data: Dict[str, Any]) -> EmbeddingResult:
        """Convert dictionary to EmbeddingResult object."""
        return EmbeddingResult(
            vnr_id=str(data['vnr_id']),
            success=data['success'],
            node_mapping=data.get('node_mapping', {}),
            link_mapping=data.get('link_mapping', {}),
            revenue=data.get('revenue', 0.0),
            cost=data.get('cost', 0.0),
            execution_time=data.get('execution_time', 0.0),
            failure_reason=data.get('failure_reason'),
            timestamp=data.get('timestamp'),
            algorithm_name=data.get('algorithm_name')
        )

    def _print_execution_summary(self, results: List[EmbeddingResult],
                                 execution_time: float, algorithm_name: str) -> None:
        """Print execution summary."""
        successful = sum(1 for r in results if r.success)
        total = len(results)

        print(f"\n{'=' * 60}")
        print(f"EXECUTION SUMMARY - {algorithm_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Total VNRs processed: {total}")
        print(f"Successful embeddings: {successful}")
        print(f"Failed embeddings: {total - successful}")
        print(f"Acceptance ratio: {successful / total * 100:.1f}%")

        if successful > 0:
            total_revenue = sum(r.revenue for r in results if r.success)
            total_cost = sum(r.cost for r in results)
            avg_revenue = total_revenue / successful

            print(f"Total revenue: {total_revenue:.2f}")
            print(f"Total cost: {total_cost:.2f}")
            print(f"Average revenue per successful VNR: {avg_revenue:.2f}")
            if total_cost > 0:
                print(f"Revenue-to-cost ratio: {total_revenue / total_cost:.2f}")

        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Average time per VNR: {execution_time / total:.4f} seconds")

        # Show sample results
        print(f"\nSample results:")
        for i, result in enumerate(results[:5]):
            status = "✓ accepted" if result.success else "✗ rejected"
            reason = f", reason={result.failure_reason}" if result.failure_reason else ""
            print(f"  VNR {result.vnr_id}: {status}, revenue={result.revenue:.2f}, "
                  f"cost={result.cost:.2f}{reason}")

        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")

        print(f"{'=' * 60}")

    def _print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Print metrics summary."""
        print(f"\n{'=' * 60}")
        print("METRICS SUMMARY")
        print(f"{'=' * 60}")

        # Essential metrics
        if 'acceptance_ratio' in metrics:
            print(f"Acceptance ratio: {metrics['acceptance_ratio'] * 100:.1f}%")
        if 'total_revenue' in metrics:
            print(f"Total revenue: {metrics['total_revenue']:.2f}")
        if 'total_cost' in metrics:
            print(f"Total cost: {metrics['total_cost']:.2f}")
        if 'revenue_to_cost_ratio' in metrics:
            print(f"Revenue-to-cost ratio: {metrics['revenue_to_cost_ratio']:.2f}")

        # Performance metrics
        if 'average_execution_time' in metrics:
            print(f"Average execution time: {metrics['average_execution_time']:.4f} seconds")

        # Utilization metrics
        utilization_metrics = ['avg_node_cpu_util', 'avg_node_memory_util', 'avg_link_bandwidth_util']
        for metric in utilization_metrics:
            if metric in metrics:
                print(f"{metric.replace('_', ' ').title()}: {metrics[metric] * 100:.1f}%")

        print(f"{'=' * 60}")


def main() -> int:
    """Main entry point."""
    cli = VNECommandLineInterface()
    return cli.run()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
