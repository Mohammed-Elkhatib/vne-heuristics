"""
Run command implementation with real algorithm execution.
"""

import time
import logging
from pathlib import Path
from datetime import datetime

from .base_command import BaseCommand
from cli.exceptions import CommandError, AlgorithmError, FileError

logger = logging.getLogger(__name__)


class RunCommand(BaseCommand):
    """Command for running VNE algorithms with real implementation."""

    def execute(self, args) -> int:
        """Execute the run command."""
        try:
            # Handle list algorithms request
            if getattr(args, 'list_algorithms', False):
                return self._list_algorithms()

            # Validate required arguments
            self._validate_run_arguments(args)

            # Load networks
            substrate = self._load_substrate_network(args.substrate)
            vnrs = self._load_vnr_batch(args.vnrs)

            # Get algorithm
            algorithm = self._get_algorithm(args.algorithm)

            # Execute algorithm
            results = self._execute_algorithm(algorithm, vnrs, substrate, args)

            # Save results
            output_path = self._save_results(results, args)

            # Print summary
            self._print_execution_summary(results, args.algorithm, output_path)

            return 0

        except (CommandError, AlgorithmError, FileError):
            # These are already handled by the error handler
            raise
        except Exception as e:
            self.error_handler.handle_unexpected_error(e) if self.error_handler else print(f"Error: {e}")
            return 1

    def _list_algorithms(self) -> int:
        """List available algorithms."""
        if self.algorithm_registry:
            algorithms = self.algorithm_registry.get_algorithms()
            print("Available algorithms:")
            if algorithms:
                for name in sorted(algorithms.keys()):
                    metadata = self.algorithm_registry.get_algorithm_metadata(name)
                    class_name = metadata.get('class_name', 'Unknown') if metadata else 'Unknown'
                    print(f"  - {name} ({class_name})")
            else:
                print("  (No algorithms currently available)")
        else:
            print("Algorithm registry not available")
        return 0

    def _validate_run_arguments(self, args) -> None:
        """Validate arguments for algorithm execution."""
        if not getattr(args, 'algorithm', None):
            raise CommandError("--algorithm is required for algorithm execution")
        if not getattr(args, 'substrate', None):
            raise CommandError("--substrate is required for algorithm execution")
        if not getattr(args, 'vnrs', None):
            raise CommandError("--vnrs is required for algorithm execution")

    def _load_substrate_network(self, filepath: str):
        """Load substrate network with proper error handling."""
        try:
            from src.models.substrate import SubstrateNetwork

            substrate = SubstrateNetwork()

            # Handle different file naming conventions
            base_name = filepath.replace('.csv', '') if filepath.endswith('.csv') else filepath
            nodes_file = f"{base_name}_nodes.csv"
            links_file = f"{base_name}_links.csv"

            # Check if files exist
            if not Path(nodes_file).exists() or not Path(links_file).exists():
                raise FileError(f"Substrate files not found: {nodes_file}, {links_file}")

            substrate.load_from_csv(nodes_file, links_file)
            logger.info(f"Loaded substrate network: {substrate}")

            return substrate

        except Exception as e:
            raise FileError(f"Failed to load substrate network from {filepath}: {e}")

    def _load_vnr_batch(self, filepath: str):
        """Load VNR batch with proper error handling."""
        try:
            # Try to load using VNRBatch if available
            try:
                from src.models.virtual_request import VNRBatch
                base_name = filepath.replace('.csv', '') if filepath.endswith('.csv') else filepath
                vnr_batch = VNRBatch.load_from_csv(base_name)
                logger.info(f"Loaded VNR batch: {vnr_batch}")
                return vnr_batch.vnrs  # Return the list of VNRs

            except (ImportError, AttributeError):
                # Fallback: Load VNRs manually from CSV files
                return self._load_vnrs_from_csv(filepath)

        except Exception as e:
            raise FileError(f"Failed to load VNR batch from {filepath}: {e}")

    def _load_vnrs_from_csv(self, filepath: str):
        """Fallback method to load VNRs from CSV files."""
        import csv
        from src.models.virtual_request import VirtualNetworkRequest

        base_name = filepath.replace('.csv', '') if filepath.endswith('.csv') else filepath

        # Load VNR metadata
        vnrs = []
        vnr_data = {}

        try:
            # Read metadata
            with open(f"{base_name}_metadata.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    vnr = VirtualNetworkRequest(
                        vnr_id=vnr_id,
                        arrival_time=float(row['arrival_time']),
                        holding_time=float(row['holding_time'])
                    )
                    vnrs.append(vnr)
                    vnr_data[vnr_id] = vnr

            # Read nodes
            with open(f"{base_name}_nodes.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnr_data:
                        vnr_data[vnr_id].add_virtual_node(
                            node_id=int(row['node_id']),
                            cpu_requirement=float(row['cpu_requirement']),
                            memory_requirement=float(row['memory_requirement'])
                        )

            # Read links
            with open(f"{base_name}_links.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnr_data:
                        vnr_data[vnr_id].add_virtual_link(
                            src_node=int(row['src_node']),
                            dst_node=int(row['dst_node']),
                            bandwidth_requirement=float(row['bandwidth_requirement'])
                        )

            logger.info(f"Loaded {len(vnrs)} VNRs from CSV files")
            return vnrs

        except Exception as e:
            raise FileError(f"Failed to load VNRs from CSV files: {e}")

    def _get_algorithm(self, algorithm_name: str):
        """Get algorithm instance with proper error handling."""
        if not self.algorithm_registry or not self.algorithm_registry.is_available(algorithm_name):
            available = ', '.join(self.algorithm_registry.list_algorithms()) if self.algorithm_registry else "none"
            raise AlgorithmError(
                f"Algorithm '{algorithm_name}' not available. Available: {available}",
                algorithm=algorithm_name
            )

        try:
            algorithm_class = self.algorithm_registry.get_algorithm(algorithm_name)
            algorithm = algorithm_class()
            logger.info(f"Initialized algorithm: {algorithm}")

            return algorithm

        except Exception as e:
            raise AlgorithmError(
                f"Failed to initialize algorithm '{algorithm_name}': {e}",
                algorithm=algorithm_name
            )

    def _execute_algorithm(self, algorithm, vnrs, substrate, args):
        """Execute algorithm with progress reporting."""
        mode = getattr(args, 'mode', 'batch')

        logger.info(f"Running {algorithm.name} in {mode} mode on {len(vnrs)} VNRs")
        print(f"Running {algorithm.name} algorithm in {mode} mode...")
        print(f"Processing {len(vnrs)} VNRs on substrate with {len(substrate.graph.nodes)} nodes...")

        # Setup progress reporting
        if getattr(args, 'progress', False) and self.progress_reporter:
            self.progress_reporter.start(len(vnrs), f"Running {algorithm.name}")

        try:
            start_time = time.time()

            if mode == 'batch':
                results = algorithm.embed_batch(vnrs, substrate)
                if getattr(args, 'progress', False) and self.progress_reporter:
                    # Manual progress update for batch mode
                    for i, result in enumerate(results):
                        if (i + 1) % 5 == 0 or (i + 1) == len(results):
                            self.progress_reporter.update(i + 1, len(results), f"Running {algorithm.name}")
            elif mode == 'online':
                results = algorithm.embed_online(vnrs, substrate)
            else:
                raise AlgorithmError(f"Unknown execution mode: {mode}")

            execution_time = time.time() - start_time

            if getattr(args, 'progress', False) and self.progress_reporter:
                self.progress_reporter.finish(len(vnrs), f"Running {algorithm.name}")

            logger.info(f"Algorithm execution completed in {execution_time:.2f} seconds")
            print(f"Algorithm execution completed in {execution_time:.2f} seconds")

            return results

        except Exception as e:
            raise AlgorithmError(
                f"Algorithm execution failed: {e}",
                algorithm=algorithm.name,
                details=f"Mode: {mode}, VNRs: {len(vnrs)}"
            )

    def _save_results(self, results, args):
        """Save algorithm results with proper error handling."""
        try:
            # Generate output filename if not provided
            if getattr(args, 'output', None):
                output_path = Path(args.output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{args.algorithm}_{timestamp}.{getattr(args, 'format', 'json')}"
                if self.config and hasattr(self.config, 'file_paths'):
                    output_path = Path(self.config.file_paths.results_dir) / filename
                else:
                    output_path = Path("data/output/results") / filename

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert results to JSON-serializable format
            serializable_results = self._convert_results_to_serializable(results)

            # Save results
            self._save_results_to_file(serializable_results, output_path, getattr(args, 'format', 'json'))
            logger.info(f"Saved {len(results)} results to {output_path}")

            return output_path

        except Exception as e:
            raise FileError(f"Failed to save results: {e}")

    def _convert_results_to_serializable(self, results):
        """Convert EmbeddingResult objects to JSON-serializable format."""
        serializable_results = []

        for result in results:
            # Handle link mapping tuple keys
            link_mapping_serializable = {}
            if hasattr(result, 'link_mapping') and result.link_mapping:
                for (src, dst), path in result.link_mapping.items():
                    key = f"{src}-{dst}"
                    link_mapping_serializable[key] = path

            # Create serializable result
            serializable_result = {
                'vnr_id': str(result.vnr_id),
                'success': result.success,
                'node_mapping': result.node_mapping or {},
                'link_mapping': link_mapping_serializable,
                'revenue': getattr(result, 'revenue', 0.0),
                'cost': getattr(result, 'cost', 0.0),
                'execution_time': getattr(result, 'execution_time', 0.0),
                'failure_reason': getattr(result, 'failure_reason', None),
                'timestamp': getattr(result, 'timestamp', None),
                'algorithm_name': getattr(result, 'algorithm_name', None),
                'metadata': getattr(result, 'metadata', {})
            }
            serializable_results.append(serializable_result)

        return serializable_results

    def _save_results_to_file(self, results, filepath, format):
        """Save results to file in specified format."""
        import json

        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'result_count': len(results)
                    },
                    'results': results
                }, f, indent=2, ensure_ascii=False)
        else:
            # CSV format - simplified
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if results:
                    fieldnames = ['vnr_id', 'success', 'revenue', 'cost', 'execution_time', 'failure_reason']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in results:
                        writer.writerow({
                            'vnr_id': result['vnr_id'],
                            'success': result['success'],
                            'revenue': result['revenue'],
                            'cost': result['cost'],
                            'execution_time': result['execution_time'],
                            'failure_reason': result['failure_reason']
                        })

    def _print_execution_summary(self, results, algorithm_name, output_path):
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
            total_revenue = sum(getattr(r, 'revenue', 0) for r in results if r.success)
            total_cost = sum(getattr(r, 'cost', 0) for r in results)
            avg_revenue = total_revenue / successful if successful > 0 else 0

            print(f"Total revenue: {total_revenue:.2f}")
            print(f"Total cost: {total_cost:.2f}")
            print(f"Average revenue per successful VNR: {avg_revenue:.2f}")
            if total_cost > 0:
                print(f"Revenue-to-cost ratio: {total_revenue / total_cost:.2f}")

        print(f"\nResults saved to: {output_path}")

        # Show sample results
        print(f"\nSample results:")
        for i, result in enumerate(results[:3]):  # Show first 3
            status = "accepted" if result.success else "rejected"
            reason = f" ({result.failure_reason})" if not result.success and hasattr(result, 'failure_reason') and result.failure_reason else ""
            print(f"  VNR {result.vnr_id}: {status}{reason}")

        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more")

        print(f"{'=' * 60}")
