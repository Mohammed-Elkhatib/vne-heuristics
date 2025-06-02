"""
Generate command implementation using proper generator modules.
"""

import logging
import random
from pathlib import Path
from .base_command import BaseCommand
from cli.exceptions import CommandError, FileError

logger = logging.getLogger(__name__)


class GenerateCommand(BaseCommand):
    """Command for generating substrate networks and VNR batches."""

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
        """Generate substrate network using the original logic."""
        try:
            # Set random seed if provided
            if args.seed:
                random.seed(args.seed)
                logger.info(f"Random seed set to {args.seed}")

            # Generate substrate network using original logic from main.py
            print(f"Generating substrate network with {args.nodes} nodes...")

            # Import and use SubstrateNetwork directly
            from src.models.substrate import SubstrateNetwork

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
            edges_created = 0
            if args.topology == "erdos_renyi":
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

            elif args.topology == "barabasi_albert":
                # Simplified Barabási-Albert implementation
                # Start with a small complete graph
                initial_nodes = min(args.attachment_count, args.nodes)
                for i in range(initial_nodes):
                    for j in range(i + 1, initial_nodes):
                        bandwidth = random.randint(*args.bandwidth_range)
                        delay = random.uniform(1.0, 10.0)
                        substrate.add_link(i, j, bandwidth, delay)
                        edges_created += 1

                # Add remaining nodes with preferential attachment
                for new_node in range(initial_nodes, args.nodes):
                    # Simple preferential attachment: connect to random existing nodes
                    targets = random.sample(range(new_node), min(args.attachment_count, new_node))
                    for target in targets:
                        bandwidth = random.randint(*args.bandwidth_range)
                        delay = random.uniform(1.0, 10.0)
                        substrate.add_link(new_node, target, bandwidth, delay)
                        edges_created += 1

                print(f"Created {edges_created} edges for Barabási-Albert topology")

            elif args.topology == "grid":
                # Grid topology implementation
                grid_size = int(args.nodes ** 0.5)
                for i in range(args.nodes):
                    row = i // grid_size
                    col = i % grid_size

                    # Connect to right neighbor
                    if col < grid_size - 1 and (i + 1) < args.nodes:
                        bandwidth = random.randint(*args.bandwidth_range)
                        delay = random.uniform(1.0, 10.0)
                        substrate.add_link(i, i + 1, bandwidth, delay)
                        edges_created += 1

                    # Connect to bottom neighbor
                    if row < grid_size - 1 and (i + grid_size) < args.nodes:
                        bandwidth = random.randint(*args.bandwidth_range)
                        delay = random.uniform(1.0, 10.0)
                        substrate.add_link(i, i + grid_size, bandwidth, delay)
                        edges_created += 1

                print(f"Created {edges_created} edges for grid topology")

            # Save to file using the original naming convention
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
        """Generate VNR batch using the original logic."""
        try:
            # Load substrate network
            print(f"Loading substrate network from {args.substrate}...")
            substrate = self._load_substrate_network(args.substrate)

            # Get substrate node IDs
            substrate_nodes = [str(node_id) for node_id in substrate.graph.nodes()]
            print(f"Found {len(substrate_nodes)} substrate nodes")

            # Set random seed if provided
            if args.seed:
                random.seed(args.seed)
                logger.info(f"Random seed set to {args.seed}")

            # Generate VNRs using original logic from main.py
            print(f"Generating {args.count} VNRs...")
            vnrs = []

            from src.models.virtual_request import VirtualNetworkRequest

            for i in range(args.count):
                # Generate VNR parameters
                vnr_nodes_count = random.randint(*args.nodes_range)
                arrival_time = random.expovariate(1.0 / args.arrival_rate) * i
                holding_time = random.expovariate(1.0 / args.lifetime_mean)

                # Create VNR
                vnr = VirtualNetworkRequest(
                    vnr_id=i,
                    arrival_time=arrival_time,
                    holding_time=holding_time
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
            from src.models.vnr_batch import VNRBatch

            batch = VNRBatch(vnrs, "generated_batch")
            base_name = args.save.replace('.csv', '') if args.save.endswith('.csv') else args.save
            batch.save_to_csv(base_name)

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
