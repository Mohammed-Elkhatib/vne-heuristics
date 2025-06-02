"""
Unit tests for VNE algorithms including base algorithm and Yu 2008 implementation.

These tests ensure VNE literature compliance and validate our clean architecture.
"""

import unittest
import logging

# Import our VNE modules
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult, VNEConstraintError
from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm
from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork
from src.utils.metrics import calculate_vnr_revenue, calculate_vnr_cost


class TestVNEAlgorithmBase(unittest.TestCase):
    """Base test class with common setup for VNE algorithm tests."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test substrate network
        self.substrate = SubstrateNetwork()
        
        # Add substrate nodes (ensure enough for Intra-VNR separation)
        for i in range(10):
            self.substrate.add_node(
                node_id=i,
                cpu_capacity=100.0,
                memory_capacity=100.0,
                x_coord=float(i),
                y_coord=float(i)
            )
        
        # Add substrate links
        for i in range(9):
            self.substrate.add_link(
                src=i, dst=i+1,
                bandwidth_capacity=100.0,
                delay=1.0
            )
        
        # Add some cross-connections for better connectivity
        self.substrate.add_link(src=0, dst=2, bandwidth_capacity=100.0, delay=2.0)
        self.substrate.add_link(src=1, dst=3, bandwidth_capacity=100.0, delay=2.0)
        self.substrate.add_link(src=2, dst=4, bandwidth_capacity=100.0, delay=2.0)
        
        # Create test VNRs
        self.simple_vnr = self._create_simple_vnr()
        self.complex_vnr = self._create_complex_vnr()
        self.large_vnr = self._create_large_vnr()
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def _create_simple_vnr(self):
        """Create a simple 2-node VNR."""
        vnr = VirtualNetworkRequest(vnr_id=1, arrival_time=0.0, holding_time=100.0)
        
        vnr.add_virtual_node(node_id=0, cpu_requirement=20.0, memory_requirement=20.0)
        vnr.add_virtual_node(node_id=1, cpu_requirement=30.0, memory_requirement=30.0)
        
        vnr.add_virtual_link(
            src_node=0, dst_node=1,
            bandwidth_requirement=20.0
        )
        
        return vnr
    
    def _create_complex_vnr(self):
        """Create a more complex 4-node VNR."""
        vnr = VirtualNetworkRequest(vnr_id=2, arrival_time=10.0, holding_time=200.0)
        
        # Add nodes
        for i in range(4):
            vnr.add_virtual_node(
                node_id=i,
                cpu_requirement=15.0 + i * 5,
                memory_requirement=10.0 + i * 5
            )
        
        # Add links (star topology)
        for i in range(1, 4):
            vnr.add_virtual_link(
                src_node=0, dst_node=i,
                bandwidth_requirement=10.0 + i * 2
            )
        
        return vnr
    
    def _create_large_vnr(self):
        """Create a large VNR that exceeds substrate capacity."""
        vnr = VirtualNetworkRequest(vnr_id=3, arrival_time=20.0, holding_time=300.0)
        
        # Add more nodes than substrate has
        for i in range(15):  # More than our 10 substrate nodes
            vnr.add_virtual_node(
                node_id=i,
                cpu_requirement=10.0,
                memory_requirement=10.0
            )
        
        return vnr


class TestBaseAlgorithm(TestVNEAlgorithmBase):
    """Test the BaseAlgorithm class for VNE literature compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create mock algorithm for testing base class
        self.mock_algorithm = MockAlgorithm()
    
    def test_base_algorithm_initialization(self):
        """Test base algorithm initialization."""
        algorithm = MockAlgorithm(test_param="test_value")
        
        self.assertEqual(algorithm.name, "Mock Algorithm")
        self.assertEqual(algorithm.parameters["test_param"], "test_value")
        self.assertIsNotNone(algorithm.logger)
        
        # Check statistics initialization
        stats = algorithm.get_algorithm_statistics()
        self.assertEqual(stats['total_requests'], 0)
        self.assertEqual(stats['successful_requests'], 0)
        self.assertEqual(stats['acceptance_ratio'], 0.0)
    
    def test_vnr_substrate_compatibility_validation(self):
        """Test VNR-substrate compatibility validation."""
        # Valid case should not raise exception
        self.mock_algorithm._validate_vnr_substrate_compatibility(self.simple_vnr, self.substrate)
        
        # Empty VNR should raise exception
        empty_vnr = VirtualNetworkRequest(vnr_id=99)
        with self.assertRaises(VNEConstraintError):
            self.mock_algorithm._validate_vnr_substrate_compatibility(empty_vnr, self.substrate)
        
        # VNR with too many nodes should raise exception
        with self.assertRaises(VNEConstraintError):
            self.mock_algorithm._validate_vnr_substrate_compatibility(self.large_vnr, self.substrate)
    
    def test_intra_vnr_separation_constraint(self):
        """Test Intra-VNR separation constraint enforcement."""
        # Valid mapping (all different substrate nodes)
        valid_mapping = {"0": "1", "1": "2", "2": "3"}
        self.assertTrue(BaseAlgorithm._check_intra_vnr_separation(valid_mapping))
        
        # Invalid mapping (two virtual nodes mapped to same substrate node)
        invalid_mapping = {"0": "1", "1": "1", "2": "3"}  # Both 0 and 1 map to substrate node 1
        self.assertFalse(BaseAlgorithm._check_intra_vnr_separation(invalid_mapping))
        
        # Single node mapping should be valid
        single_mapping = {"0": "1"}
        self.assertTrue(BaseAlgorithm._check_intra_vnr_separation(single_mapping))
        
        # Empty mapping should be valid
        empty_mapping = {}
        self.assertTrue(BaseAlgorithm._check_intra_vnr_separation(empty_mapping))
    
    def test_vne_constraint_validation(self):
        """Test VNE constraint validation after embedding."""
        # Create valid embedding result
        valid_result = EmbeddingResult(
            vnr_id="1",
            success=True,
            node_mapping={"0": "1", "1": "2"},
            link_mapping={("0", "1"): ["1", "2"]},
            revenue=100.0,
            cost=50.0,
            execution_time=0.1
        )
        
        violations = self.mock_algorithm._validate_vne_constraints(
            self.simple_vnr, self.substrate, valid_result
        )
        self.assertEqual(len(violations), 0)
        
        # Create invalid embedding result (Intra-VNR violation)
        invalid_result = EmbeddingResult(
            vnr_id="1",
            success=True,
            node_mapping={"0": "1", "1": "1"},  # Both map to same substrate node
            link_mapping={("0", "1"): ["1"]},
            revenue=100.0,
            cost=50.0,
            execution_time=0.1
        )
        
        violations = self.mock_algorithm._validate_vne_constraints(
            self.simple_vnr, self.substrate, invalid_result
        )
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("Intra-VNR separation" in v for v in violations))
    
    def test_embedding_workflow(self):
        """Test complete embedding workflow."""
        # Test successful embedding
        result = self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.vnr_id, "1")
        self.assertTrue(result.success)
        self.assertGreater(result.revenue, 0)
        self.assertGreater(result.cost, 0)
        self.assertGreater(result.execution_time, 0)
        self.assertEqual(result.algorithm_name, "Mock Algorithm")
        
        # Check statistics were updated
        stats = self.mock_algorithm.get_algorithm_statistics()
        self.assertEqual(stats['total_requests'], 1)
        self.assertEqual(stats['successful_requests'], 1)
        self.assertEqual(stats['acceptance_ratio'], 1.0)
    
    def test_failed_embedding_handling(self):
        """Test handling of failed embeddings."""
        # Force algorithm to fail
        self.mock_algorithm.force_failure = True
        
        result = self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        self.assertFalse(result.success)
        self.assertEqual(result.revenue, 0.0)
        self.assertGreater(result.cost, 0)  # Should have failure cost
        self.assertIsNotNone(result.failure_reason)
        
        # Check statistics
        stats = self.mock_algorithm.get_algorithm_statistics()
        self.assertEqual(stats['total_requests'], 1)
        self.assertEqual(stats['successful_requests'], 0)
        self.assertEqual(stats['acceptance_ratio'], 0.0)
    
    def test_batch_embedding(self):
        """Test batch embedding functionality."""
        vnrs = [self.simple_vnr, self.complex_vnr]
        
        results = self.mock_algorithm.embed_batch(vnrs, self.substrate)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, EmbeddingResult) for r in results))
        
        # Check statistics
        stats = self.mock_algorithm.get_algorithm_statistics()
        self.assertEqual(stats['total_requests'], 2)
    
    def test_online_embedding(self):
        """Test online embedding with temporal constraints."""
        # Create VNRs with different arrival times
        vnr1 = VirtualNetworkRequest(vnr_id=1, arrival_time=0.0, holding_time=50.0)
        vnr1.add_virtual_node(0, 20.0)
        vnr1.add_virtual_node(1, 20.0)
        vnr1.add_virtual_link(0, 1, 20.0)
        
        vnr2 = VirtualNetworkRequest(vnr_id=2, arrival_time=10.0, holding_time=30.0)
        vnr2.add_virtual_node(0, 20.0)
        vnr2.add_virtual_node(1, 20.0)
        vnr2.add_virtual_link(0, 1, 20.0)
        
        vnrs = [vnr1, vnr2]
        
        results = self.mock_algorithm.embed_online(vnrs, self.substrate)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, EmbeddingResult) for r in results))
    
    def test_constraint_violation_handling(self):
        """Test handling when algorithm succeeds but violates VNE constraints."""
        # Force algorithm to return invalid mapping
        self.mock_algorithm.return_invalid_mapping = True
        
        result = self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        # Should be marked as failed due to constraint violation
        self.assertFalse(result.success)
        self.assertTrue("VNE constraint violation" in result.failure_reason)
        self.assertEqual(result.revenue, 0.0)
        
        # Check that constraint violations are tracked
        stats = self.mock_algorithm.get_algorithm_statistics()
        self.assertEqual(stats['constraint_violations'], 1)
        self.assertGreater(stats['constraint_violation_rate'], 0)
    
    def test_metrics_integration(self):
        """Test integration with metrics module."""
        result = self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        # Revenue should be calculated using standard VNE formula
        expected_revenue = calculate_vnr_revenue(self.simple_vnr)
        self.assertEqual(result.revenue, expected_revenue)
        
        # Cost should be calculated using standard VNE formula
        expected_cost = calculate_vnr_cost(
            self.simple_vnr, result.node_mapping, result.link_mapping, self.substrate
        )
        self.assertEqual(result.cost, expected_cost)
    
    def test_statistics_tracking(self):
        """Test algorithm statistics tracking."""
        # Process several VNRs
        self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        self.mock_algorithm.force_failure = True
        self.mock_algorithm.embed_vnr(self.complex_vnr, self.substrate)
        
        self.mock_algorithm.force_failure = False
        self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        stats = self.mock_algorithm.get_algorithm_statistics()
        
        self.assertEqual(stats['total_requests'], 3)
        self.assertEqual(stats['successful_requests'], 2)
        self.assertAlmostEqual(stats['acceptance_ratio'], 2/3, places=2)
        self.assertGreater(stats['total_revenue'], 0)
        self.assertGreater(stats['total_cost'], 0)
        self.assertGreater(stats['average_execution_time'], 0)
    
    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        # Generate some statistics
        self.mock_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        stats_before = self.mock_algorithm.get_algorithm_statistics()
        self.assertGreater(stats_before['total_requests'], 0)
        
        # Reset statistics
        self.mock_algorithm.reset_statistics()
        
        stats_after = self.mock_algorithm.get_algorithm_statistics()
        self.assertEqual(stats_after['total_requests'], 0)
        self.assertEqual(stats_after['successful_requests'], 0)
        self.assertEqual(stats_after['total_revenue'], 0.0)


class TestYuAlgorithm(TestVNEAlgorithmBase):
    """Test the Yu 2008 algorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.yu_algorithm = YuAlgorithm(k_paths=1)
    
    def test_yu_algorithm_initialization(self):
        """Test Yu algorithm initialization."""
        # Test with different parameters
        algorithm = YuAlgorithm(k_paths=3, path_selection_strategy="bandwidth")
        
        self.assertEqual(algorithm.name, "Yu et al. (2008) Two-Stage Algorithm")
        self.assertEqual(algorithm.k_paths, 3)
        self.assertEqual(algorithm.path_selection_strategy, "bandwidth")
        
        # Test that k_paths=0 gets corrected to 1 (current implementation behavior)
        algorithm_corrected = YuAlgorithm(k_paths=0)
        self.assertEqual(algorithm_corrected.k_paths, 1)

        # Test valid path selection strategies
        valid_strategies = ["shortest", "bandwidth"]
        for strategy in valid_strategies:
            algo = YuAlgorithm(path_selection_strategy=strategy)
            self.assertEqual(algo.path_selection_strategy, strategy)

        # Test invalid path selection strategy
        with self.assertRaises(ValueError):
            YuAlgorithm(path_selection_strategy="invalid_strategy")

    def test_yu2008_literature_compliance(self):
        """Test Yu 2008 literature compliance features."""
        # Yu 2008 should only use CPU and bandwidth constraints
        stats = self.yu_algorithm.get_algorithm_statistics()
        self.assertEqual(stats['constraint_types'], ['CPU', 'Bandwidth'])
        self.assertEqual(stats['algorithm_type'], 'Two-Stage (Yu 2008)')
        self.assertIn('Yu et al. (2008)', stats['literature_reference'])

    def test_node_ranking_yu2008(self):
        """Test Yu 2008 node ranking strategy."""
        ranked_nodes = self.yu_algorithm._rank_virtual_nodes_yu2008(self.complex_vnr)

        # Should return NodeRankingInfo objects
        self.assertTrue(all(hasattr(node, 'cpu_requirement') for node in ranked_nodes))
        self.assertTrue(all(hasattr(node, 'degree') for node in ranked_nodes))

        # Should be sorted by CPU requirement (descending)
        cpu_requirements = [node.cpu_requirement for node in ranked_nodes]
        self.assertEqual(cpu_requirements, sorted(cpu_requirements, reverse=True))

    def test_substrate_node_selection_yu2008(self):
        """Test Yu 2008 substrate node selection strategy."""
        excluded_nodes = {0, 1}  # Simulate Intra-VNR separation

        candidate = self.yu_algorithm._find_best_substrate_node_yu2008(
            self.substrate, cpu_requirement=20.0, excluded_nodes=excluded_nodes
        )

        self.assertIsNotNone(candidate)
        self.assertNotIn(candidate.node_id, excluded_nodes)
        self.assertGreaterEqual(candidate.cpu_available, 20.0)

    def test_path_finding_yu2008(self):
        """Test Yu 2008 k-shortest path finding."""
        paths = self.yu_algorithm._find_k_shortest_paths_yu2008(
            self.substrate, src=0, dst=2, bandwidth_requirement=20.0
        )

        self.assertGreater(len(paths), 0)

        # All paths should have sufficient bandwidth
        for path_info in paths:
            self.assertGreaterEqual(path_info.min_bandwidth, 20.0)
            self.assertGreaterEqual(len(path_info.path), 2)

    def test_successful_embedding_yu2008(self):
        """Test successful Yu 2008 embedding."""
        result = self.yu_algorithm.embed_vnr(self.simple_vnr, self.substrate)

        self.assertTrue(result.success)
        self.assertEqual(len(result.node_mapping), 2)
        self.assertEqual(len(result.link_mapping), 1)

        # Check Intra-VNR separation
        substrate_nodes = set(result.node_mapping.values())
        self.assertEqual(len(substrate_nodes), len(result.node_mapping))

        # Check that resources were actually allocated
        for vnode_id, snode_id in result.node_mapping.items():
            snode_resources = self.substrate.get_node_resources(int(snode_id))
            self.assertGreater(snode_resources.cpu_used, 0)

    def test_failed_embedding_yu2008(self):
        """Test Yu 2008 embedding failure handling."""
        # Create substrate with insufficient resources
        small_substrate = SubstrateNetwork()
        small_substrate.add_node(0, cpu_capacity=10.0)  # Insufficient for our VNR
        small_substrate.add_node(1, cpu_capacity=10.0)
        small_substrate.add_link(0, 1, bandwidth_capacity=100.0)

        result = self.yu_algorithm.embed_vnr(self.simple_vnr, small_substrate)

        self.assertFalse(result.success)
        self.assertIn("insufficient", result.failure_reason.lower())

        # Check that no resources were left allocated
        for node_id in small_substrate.graph.nodes:
            node_resources = small_substrate.get_node_resources(node_id)
            self.assertEqual(node_resources.cpu_used, 0.0)

    def test_resource_allocation_during_embedding(self):
        """Test that Yu 2008 allocates resources during embedding."""
        # Check initial resource state
        initial_cpu_used = sum(
            self.substrate.get_node_resources(i).cpu_used
            for i in range(10)
        )
        initial_bandwidth_used = sum(
            self.substrate.get_link_resources(i, i+1).bandwidth_used
            for i in range(9)
        )

        # Embed VNR
        result = self.yu_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        self.assertTrue(result.success)

        # Check that resources are now allocated
        final_cpu_used = sum(
            self.substrate.get_node_resources(i).cpu_used
            for i in range(10)
        )
        final_bandwidth_used = sum(
            self.substrate.get_link_resources(i, i+1).bandwidth_used
            for i in range(9)
            if self.substrate.get_link_resources(i, i+1) is not None
        )

        self.assertGreater(final_cpu_used, initial_cpu_used)
        self.assertGreater(final_bandwidth_used, initial_bandwidth_used)

    def test_rollback_on_failure(self):
        """Test that Yu 2008 properly rolls back on failure."""
        # Create a substrate that will cause link mapping to fail
        limited_substrate = SubstrateNetwork()

        # Add nodes with sufficient CPU
        for i in range(5):
            limited_substrate.add_node(i, cpu_capacity=100.0)

        # Add only one link with insufficient bandwidth
        limited_substrate.add_link(0, 1, bandwidth_capacity=5.0)  # Too little bandwidth

        # Create VNR that will fail in link mapping stage
        failing_vnr = VirtualNetworkRequest(vnr_id=99, arrival_time=0.0, holding_time=100.0)
        failing_vnr.add_virtual_node(0, cpu_requirement=20.0)
        failing_vnr.add_virtual_node(1, cpu_requirement=20.0)
        failing_vnr.add_virtual_link(0, 1, bandwidth_requirement=50.0)  # Too much bandwidth

        result = self.yu_algorithm.embed_vnr(failing_vnr, limited_substrate)

        self.assertFalse(result.success)

        # Check that node resources were rolled back
        for i in range(5):
            node_resources = limited_substrate.get_node_resources(i)
            self.assertEqual(node_resources.cpu_used, 0.0)

    def test_path_caching(self):
        """Test Yu 2008 path caching functionality."""
        algorithm_with_cache = YuAlgorithm(k_paths=2, enable_path_caching=True)

        # First call should populate cache
        paths1 = algorithm_with_cache._find_k_shortest_paths_yu2008(
            self.substrate, 0, 2, 20.0
        )

        # Second call should use cache
        paths2 = algorithm_with_cache._find_k_shortest_paths_yu2008(
            self.substrate, 0, 2, 20.0
        )

        # Results should be identical
        self.assertEqual(len(paths1), len(paths2))

        # Clear cache and verify
        initial_cache_size = len(algorithm_with_cache._path_cache)
        self.assertGreater(initial_cache_size, 0)

        algorithm_with_cache.clear_path_cache()
        self.assertEqual(len(algorithm_with_cache._path_cache), 0)

    def test_cleanup_functionality(self):
        """Test Yu 2008 cleanup functionality."""
        # Embed VNR successfully first
        result = self.yu_algorithm.embed_vnr(self.simple_vnr, self.substrate)
        self.assertTrue(result.success)

        # Manually call cleanup (simulating base class constraint violation)
        self.yu_algorithm._cleanup_failed_embedding(self.simple_vnr, self.substrate, result)

        # Check that resources were deallocated
        for node_id in self.substrate.graph.nodes:
            node_resources = self.substrate.get_node_resources(node_id)
            self.assertEqual(node_resources.cpu_used, 0.0)

        for src, dst in self.substrate.graph.edges:
            link_resources = self.substrate.get_link_resources(src, dst)
            self.assertEqual(link_resources.bandwidth_used, 0.0)


class MockAlgorithm(BaseAlgorithm):
    """Mock algorithm for testing BaseAlgorithm functionality."""

    def __init__(self, **kwargs):
        super().__init__("Mock Algorithm", **kwargs)
        self.force_failure = False
        self.return_invalid_mapping = False

    def _embed_single_vnr(self, vnr, substrate):
        """Mock embedding implementation."""
        # Add small delay to ensure execution time is measurable
        import time
        time.sleep(0.001)

        if self.force_failure:
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id),
                success=False,
                node_mapping={},
                link_mapping={},
                revenue=0.0,
                cost=0.0,
                execution_time=0.001,
                failure_reason="Forced failure for testing"
            )

        # Simple mock mapping
        node_mapping = {}
        link_mapping = {}

        # Map virtual nodes to substrate nodes
        substrate_nodes = list(substrate.graph.nodes)
        for i, vnode_id in enumerate(vnr.virtual_nodes.keys()):
            if self.return_invalid_mapping and i > 0:
                # Create invalid mapping (multiple virtual nodes to same substrate node)
                node_mapping[str(vnode_id)] = str(substrate_nodes[0])
            else:
                node_mapping[str(vnode_id)] = str(substrate_nodes[i])

        # Map virtual links to simple paths
        for (vsrc, vdst) in vnr.virtual_links.keys():
            ssrc = node_mapping[str(vsrc)]
            sdst = node_mapping[str(vdst)]
            if ssrc != sdst:
                link_mapping[(str(vsrc), str(vdst))] = [ssrc, sdst]
            else:
                link_mapping[(str(vsrc), str(vdst))] = [ssrc]

        return EmbeddingResult(
            vnr_id=str(vnr.vnr_id),
            success=True,
            node_mapping=node_mapping,
            link_mapping=link_mapping,
            revenue=0.0,  # Will be calculated by base class
            cost=0.0,     # Will be calculated by base class
            execution_time=0.001
        )

    def _cleanup_failed_embedding(self, vnr, substrate, result):
        """Mock cleanup implementation."""
        # In a real algorithm, this would deallocate resources
        pass


class TestVNEAlgorithmIntegration(TestVNEAlgorithmBase):
    """Integration tests for VNE algorithms with models and metrics."""

    def test_metrics_module_integration(self):
        """Test integration with metrics module."""
        yu_algorithm = YuAlgorithm()

        # Embed several VNRs
        vnrs = [self.simple_vnr, self.complex_vnr]
        results = yu_algorithm.embed_batch(vnrs, self.substrate)

        # Calculate metrics using base class method
        metrics = yu_algorithm.calculate_metrics(results, self.substrate)

        # Verify metrics structure
        self.assertIn('primary_metrics', metrics)
        self.assertIn('acceptance_ratio', metrics['primary_metrics'])
        self.assertIn('revenue_to_cost_ratio', metrics['primary_metrics'])

        # Verify utilization metrics are included
        self.assertIn('utilization_metrics', metrics)

    def test_generator_integration(self):
        """Test integration with network generators."""
        from src.utils.generators import generate_substrate_network, generate_vnr_batch

        # Generate networks using your generators
        generated_substrate = generate_substrate_network(
            nodes=20,
            topology="erdos_renyi",
            edge_probability=0.2,
            enable_memory_constraints=False,  # Yu 2008 compatible
            cpu_range=(50, 100),
            bandwidth_range=(50, 100)
        )

        substrate_nodes = [str(i) for i in generated_substrate.graph.nodes]
        generated_vnrs = generate_vnr_batch(
            count=10,
            substrate_nodes=substrate_nodes
        )

        # Test algorithm with generated networks
        yu_algorithm = YuAlgorithm()
        results = yu_algorithm.embed_batch(generated_vnrs.vnrs, generated_substrate)

        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(r, EmbeddingResult) for r in results))

    def test_constraint_configuration_compatibility(self):
        """Test algorithm compatibility with different constraint configurations."""
        # Test with memory constraints disabled (Yu 2008 compatible)
        substrate_no_memory = SubstrateNetwork(enable_memory_constraints=False)

        for i in range(5):
            substrate_no_memory.add_node(i, cpu_capacity=100.0, memory_capacity=0.0)

        for i in range(4):
            substrate_no_memory.add_link(i, i+1, bandwidth_capacity=100.0)

        yu_algorithm = YuAlgorithm()
        result = yu_algorithm.embed_vnr(self.simple_vnr, substrate_no_memory)

        # Should work fine with Yu 2008 algorithm
        self.assertTrue(result.success)


def run_algorithm_tests():
    """Run all VNE algorithm tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestBaseAlgorithm,
        TestYuAlgorithm,
        TestVNEAlgorithmIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.CRITICAL)

    print("Running VNE Algorithm Unit Tests...")
    print("=" * 60)

    success = run_algorithm_tests()

    print("=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    exit(0 if success else 1)
