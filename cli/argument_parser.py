"""
Enhanced argument parsing for VNE CLI with dynamic algorithm discovery.
"""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.algorithm_registry import AlgorithmRegistry


def create_main_parser(algorithm_registry: 'AlgorithmRegistry') -> argparse.ArgumentParser:
    """Create the main argument parser with dynamic algorithm discovery."""
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

    # Substrate generation
    substrate_parser = generate_subparsers.add_parser('substrate', help='Generate substrate network')
    substrate_parser.add_argument('--nodes', '-n', type=int, default=50, help='Number of substrate nodes')
    substrate_parser.add_argument('--topology', '-t', type=str, default='erdos_renyi',
                                  choices=['erdos_renyi', 'barabasi_albert', 'grid'], help='Network topology')
    substrate_parser.add_argument('--save', '-s', type=str, required=True, help='Output file path')

    # VNR generation  
    vnr_parser = generate_subparsers.add_parser('vnrs', help='Generate VNR batch')
    vnr_parser.add_argument('--count', '-c', type=int, default=100, help='Number of VNRs to generate')
    vnr_parser.add_argument('--substrate', type=str, required=True, help='Substrate network file')
    vnr_parser.add_argument('--save', '-s', type=str, required=True, help='Output file path')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run VNE algorithms')
    run_parser.add_argument('--list-algorithms', action='store_true', help='List available algorithms')
    
    # Get available algorithms dynamically
    available_algorithms = list(algorithm_registry.get_algorithms().keys())
    run_parser.add_argument('--algorithm', '-a', type=str, choices=available_algorithms, help='Algorithm to use')
    run_parser.add_argument('--substrate', type=str, help='Substrate network file')
    run_parser.add_argument('--vnrs', type=str, help='VNR batch file')
    run_parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'online'], help='Execution mode')

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Calculate metrics from results')
    metrics_parser.add_argument('--results', '-r', type=str, required=True, help='Results file to analyze')
    metrics_parser.add_argument('--output', '-o', type=str, required=True, help='Output file for metrics')

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create-default', type=str, metavar='FILE', help='Create default configuration file')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')

    return parser
