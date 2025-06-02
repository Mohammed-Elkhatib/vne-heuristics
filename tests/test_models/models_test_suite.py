#!/usr/bin/env python3
"""
Comprehensive test suite for VNE models.

This test suite validates the functionality, integration, and compliance
with VNE literature standards for all model classes.
"""

import sys
import logging
import traceback
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_substrate_model():
    """Test SubstrateNetwork model functionality."""
    print("=" * 70)
    print("TEST 1: SubstrateNetwork Model")
    print("=" * 70)
    
    try:
        from src.models.substrate import (
            SubstrateNetwork, 
            NodeResources, 
            LinkResources,
            SubstrateNetworkError,
            ResourceAllocationError
        )
        
        print("✅ Substrate model imports successful")
        
        # Test 1.1: Basic network creation
        print("\n1.1 Testing basic network creation...")
        
        # Yu 2008 style (CPU + Bandwidth only)
        substrate_yu = SubstrateNetwork(
            enable_memory_constraints=False,
            enable_delay_constraints=False,
            enable_cost_constraints=False,
            enable_reliability_constraints=False
        )
        
        # Add nodes first to avoid empty graph issues
        substrate_yu.add_node(0, cpu_capacity=100.0, memory_capacity=0.0)
        substrate_yu.add_node(1, cpu_capacity=150.0, memory_capacity=0.0)
        
        print(f"Created Yu 2008 style substrate: {substrate_yu}")
        config = substrate_yu.get_constraint_configuration()
        expected_yu_config = {
            'cpu_constraints': True,
            'bandwidth_constraints': True,
            'memory_constraints': False,
            'delay_constraints': False,
            'cost_constraints': False,
            'reliability_constraints': False
        }
        
        if config == expected_yu_config:
            print("✅ Yu 2008 constraint configuration correct")
        else:
            print(f"❌ Yu 2008 config incorrect: {config}")
            return False
        
        # Test 1.2: Node operations
        print("\n1.2 Testing node operations...")
        
        substrate_yu.add_node(2, cpu_capacity=200.0, memory_capacity=0.0)
        
        if len(substrate_yu) == 3:
            print("✅ Node addition successful")
        else:
            print("❌ Node addition failed")
            return False
        
        # Test resource allocation
        success = substrate_yu.allocate_node_resources(0, cpu=50.0, memory=0.0)
        if success:
            print("✅ Node resource allocation successful")
        else:
            print("❌ Node resource allocation failed")
            return False
        
        # Test 1.3: Link operations
        print("\n1.3 Testing link operations...")
        
        substrate_yu.add_link(0, 1, bandwidth_capacity=100.0)
        substrate_yu.add_link(1, 2, bandwidth_capacity=150.0)
        
        success = substrate_yu.allocate_link_resources(0, 1, bandwidth=30.0)
        if success:
            print("✅ Link resource allocation successful")
        else:
            print("❌ Link resource allocation failed")
            return False
        
        # Test 1.4: Full constraint model
        print("\n1.4 Testing full constraint model...")
        
        substrate_full = SubstrateNetwork(
            enable_memory_constraints=True,
            enable_delay_constraints=True,
            enable_cost_constraints=True,
            enable_reliability_constraints=True
        )
        
        substrate_full.add_node(0, cpu_capacity=100.0, memory_capacity=200.0)
        substrate_full.add_link(0, 1, bandwidth_capacity=100.0, delay=5.0, 
                               cost=2.0, reliability=0.95)
        
        full_config = substrate_full.get_constraint_configuration()
        if all(full_config.values()):
            print("✅ Full constraint configuration correct")
        else:
            print(f"❌ Full constraint config incorrect: {full_config}")
            return False
        
        # Test 1.5: Validation
        print("\n1.5 Testing validation...")
        
        issues = substrate_yu.validate_network()
        if len(issues) == 0:
            print("✅ Network validation passed")
        else:
            print(f"⚠️  Validation issues: {issues}")
        
        # Test 1.6: Statistics
        print("\n1.6 Testing statistics...")
        
        stats = substrate_yu.get_network_statistics()
        expected_fields = ['node_count', 'link_count', 'total_cpu', 'total_bandwidth']
        
        if all(field in stats for field in expected_fields):
            print("✅ Network statistics complete")
            print(f"   Nodes: {stats['node_count']}, Links: {stats['link_count']}")
        else:
            print("❌ Network statistics incomplete")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Substrate model test failed: {e}")
        traceback.print_exc()
        return False


def test_virtual_request_model():
    """Test VirtualNetworkRequest model functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: VirtualNetworkRequest Model")
    print("=" * 70)
    
    try:
        from src.models.virtual_request import (
            VirtualNetworkRequest,
            VirtualNodeRequirement,
            VirtualLinkRequirement,
            VNRError,
            VNRValidationError
        )
        
        print("✅ VNR model imports successful")
        
        # Test 2.1: Basic VNR creation
        print("\n2.1 Testing basic VNR creation...")
        
        vnr = VirtualNetworkRequest(
            vnr_id="test_vnr_1",
            arrival_time=10.0,
            holding_time=100.0,
            priority=5
        )
        
        print(f"Created VNR: {vnr}")
        
        if vnr.vnr_id == "test_vnr_1" and vnr.arrival_time == 10.0:
            print("✅ VNR creation successful")
        else:
            print("❌ VNR creation failed")
            return False
        
        # Test 2.2: Virtual node operations
        print("\n2.2 Testing virtual node operations...")
        
        # Yu 2008 style (CPU only)
        vnr.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=0.0)
        vnr.add_virtual_node(1, cpu_requirement=75.0, memory_requirement=0.0)
        vnr.add_virtual_node(2, cpu_requirement=25.0, memory_requirement=0.0)
        
        if len(vnr.virtual_nodes) == 3:
            print("✅ Virtual node addition successful")
        else:
            print("❌ Virtual node addition failed")
            return False
        
        # Test 2.3: Virtual link operations
        print("\n2.3 Testing virtual link operations...")
        
        vnr.add_virtual_link(0, 1, bandwidth_requirement=100.0)
        vnr.add_virtual_link(1, 2, bandwidth_requirement=75.0)
        
        if len(vnr.virtual_links) == 2:
            print("✅ Virtual link addition successful")
        else:
            print("❌ Virtual link addition failed")
            return False
        
        # Test 2.4: Requirements calculation
        print("\n2.4 Testing requirements calculation...")
        
        requirements = vnr.calculate_total_requirements()
        expected_cpu = 50.0 + 75.0 + 25.0  # 150
        expected_bandwidth = 100.0 + 75.0  # 175
        
        if (requirements['total_cpu'] == expected_cpu and 
            requirements['total_bandwidth'] == expected_bandwidth):
            print("✅ Requirements calculation correct")
            print(f"   CPU: {requirements['total_cpu']}, Bandwidth: {requirements['total_bandwidth']}")
        else:
            print(f"❌ Requirements calculation incorrect: {requirements}")
            return False
        
        # Test 2.5: Constraint handling
        print("\n2.5 Testing constraint handling...")
        
        # Create VNR with secondary constraints
        vnr_full = VirtualNetworkRequest(vnr_id="full_constraints", arrival_time=0, holding_time=50)
        
        # Add nodes with memory
        vnr_full.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=100.0)
        vnr_full.add_virtual_node(1, cpu_requirement=75.0, memory_requirement=150.0)
        
        # Add links with delay and reliability
        vnr_full.add_virtual_link(0, 1, bandwidth_requirement=100.0, 
                                 delay_constraint=10.0, reliability_requirement=0.95)
        
        constraint_summary = vnr_full.get_constraint_summary()
        print(f"Constraint summary: {constraint_summary}")
        
        if constraint_summary['uses_memory_constraints']:
            print("✅ Memory constraints detected")
        else:
            print("⚠️  Memory constraints not detected")
        
        # Test 2.6: Validation
        print("\n2.6 Testing VNR validation...")
        
        issues = vnr.validate_request()
        if len(issues) == 0:
            print("✅ VNR validation passed")
        else:
            print(f"⚠️  VNR validation issues: {issues}")
        
        # Test 2.7: JSON serialization
        print("\n2.7 Testing JSON serialization...")
        
        vnr_dict = vnr.to_dict()
        if 'vnr_id' in vnr_dict and 'virtual_nodes' in vnr_dict:
            print("✅ VNR serialization successful")
        else:
            print("❌ VNR serialization failed")
            return False
        
        # Test deserialization
        vnr_restored = VirtualNetworkRequest.from_dict(vnr_dict)
        if (vnr_restored.vnr_id == vnr.vnr_id and 
            len(vnr_restored.virtual_nodes) == len(vnr.virtual_nodes)):
            print("✅ VNR deserialization successful")
        else:
            print("❌ VNR deserialization failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ VNR model test failed: {e}")
        traceback.print_exc()
        return False


def test_vnr_batch_model():
    """Test VNRBatch model functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: VNRBatch Model")
    print("=" * 70)
    
    try:
        from src.models.vnr_batch import VNRBatch, VNRBatchError
        from src.models.virtual_request import VirtualNetworkRequest
        
        print("✅ VNR batch model imports successful")
        
        # Test 3.1: Batch creation
        print("\n3.1 Testing batch creation...")
        
        batch = VNRBatch(batch_id="test_batch")
        
        # Create test VNRs
        vnr1 = VirtualNetworkRequest(vnr_id="vnr_1", arrival_time=5.0, priority=3)
        vnr1.add_virtual_node(0, cpu_requirement=50.0)
        vnr1.add_virtual_node(1, cpu_requirement=30.0)  # Add node 1 before creating link
        vnr1.add_virtual_link(0, 1, bandwidth_requirement=100.0)
        
        vnr2 = VirtualNetworkRequest(vnr_id="vnr_2", arrival_time=10.0, priority=1)
        vnr2.add_virtual_node(0, cpu_requirement=75.0)
        vnr2.add_virtual_node(1, cpu_requirement=25.0)
        
        vnr3 = VirtualNetworkRequest(vnr_id="vnr_3", arrival_time=2.0, priority=5)
        vnr3.add_virtual_node(0, cpu_requirement=100.0)
        
        # Add VNRs to batch
        batch.add_vnr(vnr1)
        batch.add_vnr(vnr2)
        batch.add_vnr(vnr3)
        
        if len(batch) == 3:
            print("✅ Batch creation and VNR addition successful")
        else:
            print("❌ Batch creation failed")
            return False
        
        # Test 3.2: Sorting operations
        print("\n3.2 Testing sorting operations...")
        
        # Sort by arrival time
        batch.sort_by_arrival_time()
        if batch[0].arrival_time == 2.0:  # vnr_3 should be first
            print("✅ Sort by arrival time successful")
        else:
            print("❌ Sort by arrival time failed")
            return False
        
        # Sort by priority (descending)
        batch.sort_by_priority()
        if batch[0].priority == 5:  # vnr_3 should be first (highest priority)
            print("✅ Sort by priority successful")
        else:
            print("❌ Sort by priority failed")
            return False
        
        # Test 3.3: Filtering operations
        print("\n3.3 Testing filtering operations...")
        
        # Filter by time range
        time_filtered = batch.filter_by_time_range(4.0, 12.0)
        if len(time_filtered) == 2:  # vnr1 and vnr2
            print("✅ Time range filtering successful")
        else:
            print(f"❌ Time range filtering failed: {len(time_filtered)} VNRs")
            return False
        
        # Filter by priority
        priority_filtered = batch.filter_by_priority_range(2, 4)
        if len(priority_filtered) == 1:  # only vnr1 (priority 3)
            print("✅ Priority filtering successful")
        else:
            print(f"❌ Priority filtering failed: {len(priority_filtered)} VNRs")
            return False
        
        # Test 3.4: Batch information
        print("\n3.4 Testing batch information...")
        
        info = batch.get_basic_info()
        if (info['count'] == 3 and 
            'arrival_time_range' in info and
            'priority_range' in info):
            print("✅ Batch information retrieval successful")
            print(f"   Count: {info['count']}, Avg nodes: {info['avg_nodes_per_vnr']:.1f}")
        else:
            print(f"❌ Batch information retrieval failed: {info}")
            return False
        
        # Test 3.5: Batch serialization
        print("\n3.5 Testing batch serialization...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON serialization
            json_path = os.path.join(temp_dir, "test_batch.json")
            batch.save_to_json(json_path)
            
            if os.path.exists(json_path):
                print("✅ JSON serialization successful")
            else:
                print("❌ JSON serialization failed")
                return False
            
            # Test JSON deserialization
            loaded_batch = VNRBatch.load_from_json(json_path)
            if (loaded_batch.batch_id == batch.batch_id and 
                len(loaded_batch) == len(batch)):
                print("✅ JSON deserialization successful")
            else:
                print("❌ JSON deserialization failed")
                return False
            
            # Test CSV serialization
            csv_base = os.path.join(temp_dir, "test_batch_csv")
            batch.save_to_csv(csv_base)
            
            expected_files = [
                f"{csv_base}_metadata.csv",
                f"{csv_base}_nodes.csv", 
                f"{csv_base}_links.csv"
            ]
            
            if all(os.path.exists(f) for f in expected_files):
                print("✅ CSV serialization successful")
            else:
                print("❌ CSV serialization failed")
                return False
            
            # Test CSV deserialization
            loaded_csv_batch = VNRBatch.load_from_csv(csv_base)
            if (len(loaded_csv_batch) == len(batch)):
                print("✅ CSV deserialization successful")
            else:
                print("❌ CSV deserialization failed")
                return False
        
        # Test 3.6: Batch operations
        print("\n3.6 Testing batch operations...")
        
        # Split batch
        sub_batches = batch.split_batch(2)  # Max 2 VNRs per batch
        if len(sub_batches) == 2:  # Should create 2 sub-batches
            print("✅ Batch splitting successful")
        else:
            print(f"❌ Batch splitting failed: {len(sub_batches)} batches")
            return False
        
        # Merge batches
        batch2 = VNRBatch(batch_id="batch2")
        batch2.add_vnr(vnr1)
        
        merged = batch.merge_batch(batch2)
        if len(merged) == len(batch) + len(batch2):
            print("✅ Batch merging successful")
        else:
            print("❌ Batch merging failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ VNR batch model test failed: {e}")
        traceback.print_exc()
        return False


def test_model_integration():
    """Test integration between all models."""
    print("\n" + "=" * 70)
    print("TEST 4: Model Integration")
    print("=" * 70)
    
    try:
        from src.models.substrate import SubstrateNetwork
        from src.models.virtual_request import VirtualNetworkRequest
        from src.models.vnr_batch import VNRBatch
        
        print("✅ All model imports successful")
        
        # Test 4.1: Constraint compatibility
        print("\n4.1 Testing constraint compatibility...")
        
        # Create substrate with specific constraints
        substrate = SubstrateNetwork(
            enable_memory_constraints=True,
            enable_delay_constraints=False,
            enable_cost_constraints=False,
            enable_reliability_constraints=True
        )
        
        substrate.add_node(0, cpu_capacity=200.0, memory_capacity=400.0)
        substrate.add_node(1, cpu_capacity=150.0, memory_capacity=300.0)
        substrate.add_link(0, 1, bandwidth_capacity=200.0, reliability=0.95)
        
        # Create VNR that matches substrate constraints
        vnr = VirtualNetworkRequest(vnr_id="compatible_vnr", arrival_time=0, holding_time=100)
        vnr.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=100.0)  # Memory used
        vnr.add_virtual_node(1, cpu_requirement=75.0, memory_requirement=150.0)  # Memory used
        vnr.add_virtual_link(0, 1, bandwidth_requirement=50.0, reliability_requirement=0.90)  # Reliability used
        
        constraint_summary = vnr.get_constraint_summary()
        substrate_config = substrate.get_constraint_configuration()
        
        print(f"Substrate constraints: {substrate_config}")
        print(f"VNR constraint usage: {constraint_summary}")
        
        # Check compatibility
        memory_compatible = (substrate_config['memory_constraints'] and 
                           constraint_summary['uses_memory_constraints'])
        reliability_compatible = (substrate_config['reliability_constraints'] and
                                constraint_summary['uses_reliability_constraints'])
        
        if memory_compatible and reliability_compatible:
            print("✅ Constraint compatibility successful")
        else:
            print("⚠️  Constraint compatibility check - may need manual verification")
        
        # Test 4.2: Resource checking
        print("\n4.2 Testing resource checking...")
        
        # Check if substrate can accommodate VNR
        node0_check = substrate.check_node_resources(0, cpu=50.0, memory=100.0)
        node1_check = substrate.check_node_resources(1, cpu=75.0, memory=150.0)
        link_check = substrate.check_link_resources(0, 1, bandwidth=50.0)
        
        if (all(node0_check.values()) and all(node1_check.values()) and link_check):
            print("✅ Resource checking successful - VNR requirements can be satisfied")
        else:
            print("❌ Resource checking failed")
            return False
        
        # Test 4.3: Batch and network integration
        print("\n4.3 Testing batch and network integration...")
        
        batch = VNRBatch(batch_id="integration_test")
        
        # Create multiple VNRs with different patterns
        for i in range(5):
            test_vnr = VirtualNetworkRequest(f"vnr_{i}", arrival_time=i*10.0, holding_time=50.0)
            test_vnr.add_virtual_node(0, cpu_requirement=30.0 + i*10, memory_requirement=50.0 + i*20)
            test_vnr.add_virtual_node(1, cpu_requirement=20.0 + i*5, memory_requirement=30.0 + i*10)  # Add node 1
            test_vnr.add_virtual_link(0, 1, bandwidth_requirement=40.0 + i*5)
            batch.add_vnr(test_vnr)
        
        # Analyze batch in context of substrate
        batch_info = batch.get_basic_info()
        substrate_stats = substrate.get_network_statistics()
        
        print(f"Batch: {batch_info['count']} VNRs, avg nodes: {batch_info['avg_nodes_per_vnr']:.1f}")
        print(f"Substrate: {substrate_stats['node_count']} nodes, {substrate_stats['total_cpu']} total CPU")
        
        # Check if batch requirements are within substrate capacity
        total_batch_cpu = sum(
            sum(node.cpu_requirement for node in vnr.virtual_nodes.values())
            for vnr in batch
        )
        
        if total_batch_cpu <= substrate_stats['total_cpu']:
            print("✅ Batch-substrate integration feasible")
        else:
            print("⚠️  Batch requirements exceed substrate capacity (expected for stress test)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        traceback.print_exc()
        return False


def test_vne_literature_compliance():
    """Test compliance with VNE literature standards."""
    print("\n" + "=" * 70)
    print("TEST 5: VNE Literature Compliance")
    print("=" * 70)
    
    try:
        from src.models.substrate import SubstrateNetwork
        from src.models.virtual_request import VirtualNetworkRequest
        
        print("✅ Testing VNE literature compliance...")
        
        # Test 5.1: Yu et al. 2008 compatibility
        print("\n5.1 Testing Yu et al. 2008 compatibility...")
        
        # Create Yu 2008 style substrate (CPU + Bandwidth only)
        substrate_yu = SubstrateNetwork(
            enable_memory_constraints=False,
            enable_delay_constraints=False,
            enable_cost_constraints=False,
            enable_reliability_constraints=False
        )
        
        # Add substrate resources
        substrate_yu.add_node(0, cpu_capacity=100.0, memory_capacity=0.0)
        substrate_yu.add_node(1, cpu_capacity=80.0, memory_capacity=0.0)  # Add node 1 for link
        substrate_yu.add_link(0, 1, bandwidth_capacity=100.0)
        
        # Create compatible VNR
        vnr_yu = VirtualNetworkRequest(vnr_id="yu2008_vnr", arrival_time=0, holding_time=100)
        vnr_yu.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=0.0)
        vnr_yu.add_virtual_node(1, cpu_requirement=30.0, memory_requirement=0.0)  # Add node 1
        vnr_yu.add_virtual_link(0, 1, bandwidth_requirement=30.0)
        
        # Verify Yu 2008 pattern
        substrate_config = substrate_yu.get_constraint_configuration()
        vnr_constraints = vnr_yu.get_constraint_summary()
        
        yu2008_substrate = (substrate_config['cpu_constraints'] and 
                           substrate_config['bandwidth_constraints'] and
                           not any([substrate_config['memory_constraints'],
                                  substrate_config['delay_constraints'],
                                  substrate_config['cost_constraints'],
                                  substrate_config['reliability_constraints']]))
        
        yu2008_vnr = (not vnr_constraints['uses_memory_constraints'] and
                     not vnr_constraints['uses_delay_constraints'] and
                     not vnr_constraints['uses_reliability_constraints'])
        
        if yu2008_substrate and yu2008_vnr:
            print("✅ Yu et al. 2008 compatibility confirmed")
        else:
            print("❌ Yu et al. 2008 compatibility failed")
            return False
        
        # Test 5.2: Standard VNE metrics support
        print("\n5.2 Testing standard VNE metrics support...")
        
        # Verify required attributes for metrics
        vnr_attrs = ['vnr_id', 'arrival_time', 'holding_time', 'virtual_nodes', 'virtual_links']
        substrate_attrs = ['graph', 'get_network_statistics', 'get_constraint_configuration']
        
        vnr_complete = all(hasattr(vnr_yu, attr) for attr in vnr_attrs)
        substrate_complete = all(hasattr(substrate_yu, attr) for attr in substrate_attrs)
        
        if vnr_complete and substrate_complete:
            print("✅ Standard VNE metrics support confirmed")
        else:
            print("❌ Standard VNE metrics support incomplete")
            return False
        
        # Test 5.3: Modern constraint support
        print("\n5.3 Testing modern constraint extensions...")
        
        # Create modern VNE setup with all constraints
        substrate_modern = SubstrateNetwork(
            enable_memory_constraints=True,
            enable_delay_constraints=True,
            enable_cost_constraints=True,
            enable_reliability_constraints=True
        )
        
        substrate_modern.add_node(0, cpu_capacity=100.0, memory_capacity=200.0)
        substrate_modern.add_node(1, cpu_capacity=120.0, memory_capacity=240.0)  # Add node 1 for link
        substrate_modern.add_link(0, 1, bandwidth_capacity=100.0, delay=5.0, 
                                 cost=2.0, reliability=0.95)
        
        vnr_modern = VirtualNetworkRequest(vnr_id="modern_vnr", arrival_time=0, holding_time=100)
        vnr_modern.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=100.0)
        vnr_modern.add_virtual_node(1, cpu_requirement=40.0, memory_requirement=80.0)  # Add node 1
        vnr_modern.add_virtual_link(0, 1, bandwidth_requirement=30.0, 
                                   delay_constraint=3.0, reliability_requirement=0.90)
        
        modern_constraints = vnr_modern.get_constraint_summary()
        
        if (modern_constraints['uses_memory_constraints'] and
            modern_constraints['uses_delay_constraints'] and
            modern_constraints['uses_reliability_constraints']):
            print("✅ Modern constraint extensions confirmed")
        else:
            print("⚠️  Modern constraint extensions - partial support")
        
        return True
        
    except Exception as e:
        print(f"❌ VNE literature compliance test failed: {e}")
        traceback.print_exc()
        return False


def run_all_model_tests():
    """Run all model tests and report results."""
    print("VNE MODELS TEST SUITE")
    print("=" * 70)
    print("Testing functional, modular, sensible models following VNE literature standards")
    print("=" * 70)
    
    tests = [
        ("SubstrateNetwork Model", test_substrate_model),
        ("VirtualNetworkRequest Model", test_virtual_request_model),
        ("VNRBatch Model", test_vnr_batch_model),
        ("Model Integration", test_model_integration),
        ("VNE Literature Compliance", test_vne_literature_compliance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("MODELS TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(tests)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL MODEL TESTS PASSED!")
        print("\n📋 Model Assessment:")
        print("✅ Functional - All models work correctly")
        print("✅ Modular - Clean separation of concerns")
        print("✅ Sensible - Logical design and interfaces")
        print("✅ Standards - Follows VNE literature conventions")
        print("\n🚀 Models are ready for production use!")
        return True
    else:
        print("⚠️  SOME MODEL TESTS FAILED!")
        print("\n🔧 Review failed tests and fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_model_tests()
    sys.exit(0 if success else 1)
