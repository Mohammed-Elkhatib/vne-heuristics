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
from config_management import ConfigurationManager, VNEConfig, ConfigurationError, load_config_from_args
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
            from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm

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
        run_parser.add_argument('--list-algorithms', action='store_true',
                                help='List available algorithms and exit')
        run_parser.add_argument('--algorithm', '-a', type=str,
                                help='Algorithm to use')
        run_parser.add_argument('--substrate', type=str,
                                help='Substrate network file (CSV format)')
        run_parser.add_argument('--vnrs', type=str,
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

            # Create actual SubstrateNetwork instead of using placeholder generator
            from src.models.substrate import SubstrateNetwork
            import random

            substrate = SubstrateNetwork()

            # Generate nodes with resources
            for i in range(args.nodes):
                node_id = i
                cpu_capacity = random.randint(*args.cpu_range)
                memory_capacity = random.randint(*args.memory_range)
                x_coord = random.uniform(0.0, 100.0)
                y_coord = random.uniform(0.0, 100.0)

                substrate.add_node(
                    node_id=node_id,
                    cpu_capacity=cpu_capacity,
                    memory_capacity=memory_capacity,
                    x_coord=x_coord,
                    y_coord=y_coord
                )

            # Generate edges based on topology
            if args.topology == "erdos_renyi":
                edges_created = 0
                for i in range(args.nodes):
                    for j in range(i + 1, args.nodes):
                        if random.random() < args.edge_prob:
                            bandwidth = random.randint(*args.bandwidth_range)
                            delay = random.uniform(1.0, 10.0)
                            substrate.add_link(
                                src=i, dst=j,
                                bandwidth_capacity=bandwidth,
                                delay=delay
                            )
                            edges_created += 1
                print(f"Created {edges_created} edges for Erdos-Renyi topology")

            # Save to file
            print(f"Saving substrate network to {args.save}...")
            substrate.save_to_csv(
                nodes_file=args.save.replace('.csv', '_nodes.csv'),
                links_file=args.save.replace('.csv', '_links.csv')
            )

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
            substrate = self._load_substrate_network(args.substrate)

            # Get substrate node IDs
            substrate_nodes = [str(node_id) for node_id in substrate.graph.nodes()]
            print(f"Found {len(substrate_nodes)} substrate nodes")

            # Set random seed if provided
            if args.seed:
                set_random_seed(args.seed)
                self.logger.info(f"Random seed set to {args.seed}")

            # Generate VNRs using our VNR creation logic
            print(f"Generating {args.count} VNRs...")
            vnrs = []

            import random
            from src.models.virtual_request import VirtualNetworkRequest

            for i in range(args.count):
                # Generate VNR parameters
                vnr_nodes_count = random.randint(*args.nodes_range)
                arrival_time = random.expovariate(1.0 / args.arrival_rate) * i
                lifetime = random.expovariate(1.0 / args.lifetime_mean)

                # Create VNR
                vnr = VirtualNetworkRequest(
                    vnr_id=i,
                    arrival_time=arrival_time,
                    lifetime=lifetime
                )

                # Add virtual nodes
                for j in range(vnr_nodes_count):
                    cpu_req = random.randint(
                        int(50 * args.cpu_ratio[0]),
                        int(100 * args.cpu_ratio[1])
                    )
                    memory_req = random.randint(
                        int(50 * args.memory_ratio[0]),
                        int(100 * args.memory_ratio[1])
                    )

                    vnr.add_virtual_node(
                        node_id=j,
                        cpu_requirement=cpu_req,
                        memory_requirement=memory_req
                    )

                # Add virtual links based on topology
                if args.topology == "random" and vnr_nodes_count > 1:
                    for src in range(vnr_nodes_count):
                        for dst in range(src + 1, vnr_nodes_count):
                            if random.random() < args.edge_prob:
                                bandwidth_req = random.randint(
                                    int(50 * args.bandwidth_ratio[0]),
                                    int(100 * args.bandwidth_ratio[1])
                                )
                                vnr.add_virtual_link(
                                    src_node=src,
                                    dst_node=dst,
                                    bandwidth_requirement=bandwidth_req
                                )
                elif args.topology == "star" and vnr_nodes_count > 1:
                    # Star topology: connect all nodes to node 0
                    for i in range(1, vnr_nodes_count):
                        bandwidth_req = random.randint(
                            int(50 * args.bandwidth_ratio[0]),
                            int(100 * args.bandwidth_ratio[1])
                        )
                        vnr.add_virtual_link(
                            src_node=0,
                            dst_node=i,
                            bandwidth_requirement=bandwidth_req
                        )
                elif args.topology == "linear" and vnr_nodes_count > 1:
                    # Linear topology: chain of nodes
                    for i in range(vnr_nodes_count - 1):
                        bandwidth_req = random.randint(
                            int(50 * args.bandwidth_ratio[0]),
                            int(100 * args.bandwidth_ratio[1])
                        )
                        vnr.add_virtual_link(
                            src_node=i,
                            dst_node=i + 1,
                            bandwidth_requirement=bandwidth_req
                        )

                vnrs.append(vnr)

            # Save VNRs using VNRBatch
            print(f"Saving VNRs to {args.save}...")
            from src.models.virtual_request import VNRBatch

            batch = VNRBatch(vnrs, "generated_batch")
            batch.save_to_csv(args.save.replace('.csv', ''))

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
            # Handle list algorithms first (doesn't need other arguments)
            if args.list_algorithms:
                return self._list_algorithms()

            # Validate required arguments for algorithm execution
            if not args.algorithm:
                print("Error: --algorithm is required for algorithm execution")
                return 1
            if not args.substrate:
                print("Error: --substrate is required for algorithm execution")
                return 1
            if not args.vnrs:
                print("Error: --vnrs is required for algorithm execution")
                return 1

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

            # Convert results to JSON-serializable format
            serializable_results = []
            for result in results:
                # Convert tuple keys in link_mapping to strings
                link_mapping_serializable = {}
                for (src, dst), path in result.link_mapping.items():
                    key = f"{src}-{dst}"
                    link_mapping_serializable[key] = path

                serializable_result = {
                    'vnr_id': result.vnr_id,
                    'success': result.success,
                    'node_mapping': result.node_mapping,
                    'link_mapping': link_mapping_serializable,
                    'revenue': result.revenue,
                    'cost': result.cost,
                    'execution_time': result.execution_time,
                    'failure_reason': result.failure_reason,
                    'timestamp': result.timestamp,
                    'algorithm_name': result.algorithm_name,
                    'metadata': result.metadata
                }
                serializable_results.append(serializable_result)

            # Save results
            print(f"Saving results to {output_path}...")
            save_results_to_file(serializable_results, output_path, args.format)

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
        try:
            substrate = SubstrateNetwork()

            # Handle different file naming conventions
            if filepath.endswith('.csv'):
                # Try the naming from our generation first
                base_name = filepath.replace('.csv', '')
                nodes_file = f"{base_name}_nodes.csv"
                links_file = f"{base_name}_links.csv"

                # Check if files exist
                from pathlib import Path
                if not Path(nodes_file).exists() or not Path(links_file).exists():
                    # Try alternative naming (for backward compatibility)
                    nodes_file = f"{filepath}_nodes.csv"
                    links_file = f"{filepath}_links.csv"

                    if not Path(nodes_file).exists() or not Path(links_file).exists():
                        raise FileNotFoundError(f"Could not find substrate files. Expected either:\n"
                                                f"  - {base_name}_nodes.csv and {base_name}_links.csv\n"
                                                f"  - {filepath}_nodes.csv and {filepath}_links.csv")
            else:
                nodes_file = f"{filepath}_nodes.csv"
                links_file = f"{filepath}_links.csv"

            print(f"Loading nodes from: {nodes_file}")
            print(f"Loading links from: {links_file}")

            substrate.load_from_csv(nodes_file, links_file)
            return substrate

        except Exception as e:
            self.logger.error(f"Failed to load substrate network: {e}")
            raise Exception(f"Failed to load substrate network: {e}")

    def _load_vnr_batch(self, filepath: str) -> List[VirtualNetworkRequest]:
        """Load VNR batch from file."""
        try:
            from src.models.virtual_request import VNRBatch

            # Determine base filename
            if filepath.endswith('.csv'):
                base_name = filepath.replace('.csv', '')
            else:
                base_name = filepath

            # Load using VNRBatch
            batch = VNRBatch.load_from_csv(base_name)
            return batch.vnrs

        except Exception as e:
            self.logger.error(f"Failed to load VNR batch: {e}")
            raise Exception(f"Failed to load VNR batch: {e}")

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

        # Show sample results in the specified format
        print(f"\nSample results:")
        for i, result in enumerate(results[:5]):
            if result.success:
                # Format node mapping
                node_mapping_str = "{" + ", ".join([f"{k}->{v}" for k, v in result.node_mapping.items()]) + "}"

                # Format link mapping
                link_mapping_parts = []
                for (src, dst), path in result.link_mapping.items():
                    if len(path) > 1:
                        path_str = "->".join(path)
                        link_mapping_parts.append(f"({src},{dst})->[{path_str}]")
                    else:
                        # Single node path (virtual link collapses to same substrate node)
                        link_mapping_parts.append(f"({src},{dst})->[{path[0]}]")

                link_mapping_str = "{" + ", ".join(link_mapping_parts) + "}"

                print(f"  VNR {result.vnr_id}: accepted, nodes mapped as {node_mapping_str}, "
                      f"links mapped as {link_mapping_str}, revenue={result.revenue:.2f}, cost={result.cost:.2f}")
            else:
                reason = f", reason={result.failure_reason}" if result.failure_reason else ""
                print(f"  VNR {result.vnr_id}: rejected{reason}")

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