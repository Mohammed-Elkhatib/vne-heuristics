"""
Enhanced argument parsing for VNE CLI with all original arguments.
"""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.algorithm_registry import AlgorithmRegistry


def create_main_parser(algorithm_registry: 'AlgorithmRegistry') -> argparse.ArgumentParser:
    """Create the main argument parser with all original arguments."""
    parser = argparse.ArgumentParser(
        prog='VNE Heuristics',
        description='Virtual Network Embedding Heuristics Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    generate_parser = subparsers.add_parser('generate', help='Generate networks and VNRs')
    generate_subparsers = generate_parser.add_subparsers(dest='generate_type', help='What to generate')

    # Substrate generation - WITH ALL ORIGINAL ARGUMENTS
    substrate_parser = generate_subparsers.add_parser('substrate', help='Generate substrate network')
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

    # VNR generation - WITH ALL ORIGINAL ARGUMENTS
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

    # Run command - WITH ALL ORIGINAL ARGUMENTS
    run_parser = subparsers.add_parser('run', help='Run VNE algorithms')
    run_parser.add_argument('--list-algorithms', action='store_true',
                            help='List available algorithms and exit')

    # Get available algorithms dynamically
    available_algorithms = list(algorithm_registry.get_algorithms().keys())
    run_parser.add_argument('--algorithm', '-a', type=str,
                            choices=available_algorithms,
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

    # Metrics command - WITH ALL ORIGINAL ARGUMENTS
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

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create-default', type=str, metavar='FILE',
                               help='Create default configuration file')
    config_parser.add_argument('--validate', type=str, metavar='FILE',
                               help='Validate configuration file')
    config_parser.add_argument('--show', action='store_true',
                               help='Show current configuration')

    return parser
