"""
Generate command implementation using proper generator modules.

This refactored version properly uses the generator modules from src.utils.generators
instead of reimplementing generation logic, following DRY principles.
"""

import logging
from pathlib import Path
from .base_command import BaseCommand
from cli.exceptions import CommandError, FileError

# Import the generator modules instead of reimplementing
from src.utils.generators import (
    generate_substrate_network,
    generate_vnr_batch,
    NetworkGenerationConfig,
    set_random_seed
)

logger = logging.getLogger(__name__)


class GenerateCommand(BaseCommand):
    """Command for generating substrate networks and VNR batches using generator modules."""

    def execute(self, args) -> int:
        """Execute the generate command."""
        if not args.generate_type:
            raise CommandError("Must specify what to generate (substrate or vnrs)")

        if args.generate_type == 'substrate':
            return self._generate_substrate(args)
        elif args.generate_type == 'vnrs':
            return self._generate_vnrs(args)
        else:
            raise CommandError(f"Unknown generate type: {args.generate_type}")

    def _generate_substrate(self, args) -> int:
        """Generate substrate network using the generator modules."""
        try:
            # Set random seed if provided
            if args.seed:
                set_random_seed(args.seed)
                logger.info(f"Random seed set to {args.seed}")

            print(f"Generating substrate network with {args.nodes} nodes...")

            # Map CLI arguments to generator parameters
            # The generator handles all the complexity internally
            topology_params = {}
            if args.topology == "grid":
                # For grid topology, calculate grid size
                import math
                topology_params['grid_size'] = int(math.sqrt(args.nodes))

            # Use the generator module instead of manual implementation
            substrate = generate_substrate_network(
                nodes=args.nodes,
                topology=args.topology,
                edge_probability=getattr(args, 'edge_prob', 0.15),
                attachment_count=getattr(args, 'attachment_count', 3),
                enable_memory_constraints=True,  # Always generate memory for compatibility
                enable_delay_constraints=False,   # Match original behavior
                enable_cost_constraints=False,
                enable_reliability_constraints=False,
                cpu_range=args.cpu_range,
                memory_range=args.memory_range,
                bandwidth_range=args.bandwidth_range,
                delay_range=(1.0, 10.0),  # Match original behavior
                coordinate_range=(0.0, 100.0),  # Match original behavior
                **topology_params
            )

            # Get edge count for reporting
            edges_created = len(substrate.graph.edges())

            # Save to file using the same naming convention
            print(f"Saving substrate network to {args.save}...")
            base_name = args.save.replace('.csv', '') if args.save.endswith('.csv') else args.save
            substrate.save_to_csv(
                nodes_file=f"{base_name}_nodes.csv",
                links_file=f"{base_name}_links.csv"
            )

            print(f"✓ Successfully generated substrate network: {args.save}")
            print(f"  - Nodes: {args.nodes}")
            print(f"  - Links: {edges_created}")
            print(f"  - Topology: {args.topology}")
            print(f"  - CPU range: {args.cpu_range[0]}-{args.cpu_range[1]}")
            print(f"  - Memory range: {args.memory_range[0]}-{args.memory_range[1]}")
            print(f"  - Bandwidth range: {args.bandwidth_range[0]}-{args.bandwidth_range[1]}")

            return 0

        except Exception as e:
            logger.error(f"Substrate generation failed: {e}")
            print(f"Error generating substrate network: {e}")
            return 1

    def _generate_vnrs(self, args) -> int:
        """Generate VNR batch using the generator modules."""
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
                logger.info(f"Random seed set to {args.seed}")

            print(f"Generating {args.count} VNRs...")

            # Create configuration for VNR generation
            config = NetworkGenerationConfig(
                # VNR parameters
                vnr_nodes_range=args.nodes_range,
                vnr_topology=args.topology,
                vnr_edge_probability=getattr(args, 'edge_prob', 0.5),

                # Resource ratios (matching original behavior)
                vnr_cpu_ratio_range=args.cpu_ratio,
                vnr_memory_ratio_range=args.memory_ratio,
                vnr_bandwidth_ratio_range=args.bandwidth_ratio,

                # Temporal parameters
                arrival_pattern="poisson",
                arrival_rate=args.arrival_rate,
                holding_time_distribution="exponential",
                holding_time_mean=args.lifetime_mean,

                # Substrate resource ranges (for ratio calculations)
                cpu_range=(50, 100),  # These will be overridden by actual substrate values
                memory_range=(50, 100),
                bandwidth_range=(50, 100),

                # Inherit constraint settings from substrate
                enable_memory_constraints=substrate.enable_memory_constraints,
                enable_delay_constraints=substrate.enable_delay_constraints,
                enable_cost_constraints=substrate.enable_cost_constraints,
                enable_reliability_constraints=substrate.enable_reliability_constraints
            )

            # Use the generator module to create VNR batch
            vnr_batch = generate_vnr_batch(
                count=args.count,
                substrate_nodes=substrate_nodes,
                config=config,
                substrate_network=substrate  # Pass substrate for constraint inheritance
            )

            # Save VNRs using the same file naming convention
            print(f"Saving VNRs to {args.save}...")
            base_name = args.save.replace('.csv', '') if args.save.endswith('.csv') else args.save
            vnr_batch.save_to_csv(base_name)

            print(f"✓ Successfully generated VNR batch: {args.save}")
            print(f"  - Count: {args.count}")
            print(f"  - Node range: {args.nodes_range[0]}-{args.nodes_range[1]}")
            print(f"  - Topology: {args.topology}")
            print(f"  - Arrival rate: {args.arrival_rate}")
            print(f"  - Mean lifetime: {args.lifetime_mean}")

            return 0

        except Exception as e:
            logger.error(f"VNR generation failed: {e}")
            print(f"Error generating VNRs: {e}")
            return 1

    def _load_substrate_network(self, filepath: str):
        """Load substrate network from file."""
        try:
            from src.models.substrate import SubstrateNetwork

            substrate = SubstrateNetwork()

            # Handle different file naming conventions
            if filepath.endswith('.csv'):
                base_name = filepath.replace('.csv', '')
                nodes_file = f"{base_name}_nodes.csv"
                links_file = f"{base_name}_links.csv"

                if not Path(nodes_file).exists() or not Path(links_file).exists():
                    raise FileError(f"Could not find substrate files: {nodes_file}, {links_file}")
            else:
                nodes_file = f"{filepath}_nodes.csv"
                links_file = f"{filepath}_links.csv"

            substrate.load_from_csv(nodes_file, links_file)
            return substrate

        except Exception as e:
            raise FileError(f"Failed to load substrate network: {e}")
