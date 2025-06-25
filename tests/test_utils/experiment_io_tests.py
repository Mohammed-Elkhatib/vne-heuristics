#!/usr/bin/env python3
"""
Comprehensive test suite for experiment I/O utilities.

This module tests the experiment I/O functionality with real VNE models
following the same pattern as other test suites in the project.
"""

import sys
import logging
import traceback
import tempfile
import json
from pathlib import Path
from src.utils.metrics import generate_comprehensive_metrics_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_io_utils_imports():
    """Test that all experiment I/O functions can be imported."""
    print("=" * 70)
    print("TEST 1: Experiment I/O Imports")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import (
            ExperimentIOError,
            create_experiment_directory,
            save_experiment_data,
            load_experiment_data,
            save_algorithm_results,
            load_algorithm_results,
            save_metrics_summary,
            save_experiment_config,
            load_experiment_config,
            ExperimentRunner
        )
        print("✅ All experiment I/O functions imported successfully")
        
        # Test model imports
        from src.models.substrate import SubstrateNetwork
        from src.models.virtual_request import VirtualNetworkRequest
        from src.models.vnr_batch import VNRBatch
        print("✅ VNE model imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_directory_creation():
    """Test experiment directory creation."""
    print("\n" + "=" * 70)
    print("TEST 2: Directory Creation")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import create_experiment_directory
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with timestamp
            exp_dir1 = create_experiment_directory(temp_dir, "test_experiment")
            print(f"Created experiment directory: {exp_dir1}")
            
            if exp_dir1.exists():
                print("✅ Experiment directory created successfully")
            else:
                print("❌ Experiment directory creation failed")
                return False
                
            # Check subdirectories
            expected_subdirs = ["data", "results", "logs"]
            for subdir in expected_subdirs:
                subdir_path = exp_dir1 / subdir
                if subdir_path.exists():
                    print(f"✅ Subdirectory '{subdir}' created")
                else:
                    print(f"❌ Subdirectory '{subdir}' missing")
                    return False
            
            # Test without timestamp
            exp_dir2 = create_experiment_directory(temp_dir, "test_no_timestamp", include_timestamp=False)
            if exp_dir2.name == "test_no_timestamp":
                print("✅ Directory created without timestamp")
            else:
                print("❌ Directory created with unexpected name")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory creation test failed: {e}")
        traceback.print_exc()
        return False


def test_experiment_data_io():
    """Test saving and loading experiment data with real models."""
    print("\n" + "=" * 70)
    print("TEST 3: Experiment Data I/O")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import save_experiment_data, load_experiment_data
        from src.utils.generators import generate_substrate_network, generate_vnr_batch, NetworkGenerationConfig
        
        print("Generating test data with real models...")
        
        # Generate substrate network
        substrate = generate_substrate_network(
            nodes=10,
            topology="erdos_renyi",
            edge_probability=0.3,
            enable_memory_constraints=True
        )
        print(f"Generated substrate: {substrate}")
        
        # Generate VNR batch
        config = NetworkGenerationConfig(vnr_nodes_range=(2, 4))
        substrate_nodes = [str(i) for i in range(10)]
        vnr_batch = generate_vnr_batch(
            count=20,
            substrate_nodes=substrate_nodes,
            config=config
        )
        print(f"Generated VNR batch: {vnr_batch}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "test_experiment"
            
            # Test saving
            print("Testing save_experiment_data...")
            save_experiment_data(substrate, vnr_batch, exp_dir)
            
            # Check if files were created
            data_dir = exp_dir / "data"
            expected_files = [
                "substrate_nodes.csv",
                "substrate_links.csv", 
                "vnr_batch_metadata.csv",
                "vnr_batch_nodes.csv",
                "vnr_batch_links.csv"
            ]
            
            for filename in expected_files:
                file_path = data_dir / filename
                if file_path.exists():
                    print(f"✅ File created: {filename}")
                else:
                    print(f"❌ File missing: {filename}")
                    return False
            
            # Test loading
            print("Testing load_experiment_data...")
            loaded_substrate, loaded_vnr_batch = load_experiment_data(exp_dir)
            
            print(f"Loaded substrate: {loaded_substrate}")
            print(f"Loaded VNR batch: {loaded_vnr_batch}")
            
            # Basic validation
            if len(loaded_substrate) == len(substrate):
                print("✅ Substrate node count matches")
            else:
                print("❌ Substrate node count mismatch")
                return False
                
            if len(loaded_vnr_batch) == len(vnr_batch):
                print("✅ VNR batch count matches")
            else:
                print("❌ VNR batch count mismatch")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment data I/O test failed: {e}")
        traceback.print_exc()
        return False


def test_results_io():
    """Test saving and loading algorithm results."""
    print("\n" + "=" * 70)
    print("TEST 4: Algorithm Results I/O")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import save_algorithm_results, load_algorithm_results
        from src.utils.metrics import EmbeddingResult
        
        # Create mock algorithm results
        mock_results = [
            EmbeddingResult("vnr_1", success=True, revenue=100.0, cost=50.0, execution_time=0.01),
            EmbeddingResult("vnr_2", success=False, revenue=0.0, cost=25.0, execution_time=0.005),
            EmbeddingResult("vnr_3", success=True, revenue=150.0, cost=75.0, execution_time=0.012),
            EmbeddingResult("vnr_4", success=True, revenue=200.0, cost=100.0, execution_time=0.008)
        ]
        
        print(f"Created {len(mock_results)} mock algorithm results")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "test_experiment"
            
            # Test saving results
            print("Testing save_algorithm_results...")
            results_file = save_algorithm_results(mock_results, exp_dir, "test_algorithm")
            
            if results_file.exists():
                print(f"✅ Results file created: {results_file}")
            else:
                print("❌ Results file not created")
                return False
            
            # Verify file content
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            if 'metadata' in data and 'results' in data:
                print("✅ Results file has correct structure")
            else:
                print("❌ Results file has incorrect structure")
                return False
                
            if len(data['results']) == len(mock_results):
                print("✅ All results saved")
            else:
                print("❌ Results count mismatch")
                return False
            
            # Test loading results
            print("Testing load_algorithm_results...")
            loaded_results = load_algorithm_results(results_file)
            
            if len(loaded_results) == len(mock_results):
                print("✅ All results loaded")
            else:
                print("❌ Loaded results count mismatch")
                return False
            
            # Verify a sample result
            first_result = loaded_results[0]
            if first_result.get('vnr_id') == 'vnr_1' and first_result.get('success') is True:
                print("✅ Results content verified")
            else:
                print("❌ Results content verification failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Results I/O test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics_io():
    """Test saving metrics summary."""
    print("\n" + "=" * 70)
    print("TEST 5: Metrics I/O")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import save_metrics_summary
        
        # Create mock metrics
        mock_metrics = {
            'basic_stats': {
                'total_requests': 4,
                'successful_requests': 3,
                'failed_requests': 1
            },
            'primary_metrics': {
                'acceptance_ratio': 0.75,
                'total_revenue': 450.0,
                'total_cost': 250.0,
                'revenue_to_cost_ratio': 1.8
            },
            'performance_metrics': {
                'average_execution_time': 0.00875
            }
        }
        
        print("Created mock metrics")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "test_experiment"
            
            # Test saving metrics
            print("Testing save_metrics_summary...")
            metrics_file = save_metrics_summary(mock_metrics, exp_dir, "test_algorithm")
            
            if metrics_file.exists():
                print(f"✅ Metrics file created: {metrics_file}")
            else:
                print("❌ Metrics file not created")
                return False
            
            # Verify file content
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            if 'metadata' in data and 'metrics' in data:
                print("✅ Metrics file has correct structure")
            else:
                print("❌ Metrics file has incorrect structure")
                return False
                
            if data['metrics']['primary_metrics']['acceptance_ratio'] == 0.75:
                print("✅ Metrics content verified")
            else:
                print("❌ Metrics content verification failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics I/O test failed: {e}")
        traceback.print_exc()
        return False


def test_experiment_runner():
    """Test the ExperimentRunner class."""
    print("\n" + "=" * 70)
    print("TEST 6: ExperimentRunner")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import ExperimentRunner
        from src.utils.generators import generate_substrate_network, generate_vnr_batch, NetworkGenerationConfig
        from src.utils.metrics import EmbeddingResult
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize runner
            print("Testing ExperimentRunner initialization...")
            runner = ExperimentRunner("test_runner_experiment", base_dir=temp_dir)
            
            print(f"Created experiment runner: {runner}")
            
            if runner.get_experiment_path().exists():
                print("✅ Experiment directory created")
            else:
                print("❌ Experiment directory not created")
                return False
            
            # Setup experiment
            print("Testing experiment setup...")
            substrate = generate_substrate_network(8, "erdos_renyi", edge_probability=0.25)
            config = NetworkGenerationConfig(vnr_nodes_range=(2, 3))
            substrate_nodes = [str(i) for i in range(8)]
            vnr_batch = generate_vnr_batch(10, substrate_nodes, config)
            
            runner.setup_experiment(substrate, vnr_batch)
            print("✅ Experiment setup completed")
            
            # Test saving results
            print("Testing results saving...")
            mock_results = [
                EmbeddingResult("vnr_1", success=True, revenue=80.0, cost=40.0),
                EmbeddingResult("vnr_2", success=False, revenue=0.0, cost=20.0)
            ]
            
            results_file = runner.save_results(mock_results, "test_algorithm")
            if results_file.exists():
                print("✅ Results saved successfully")
            else:
                print("❌ Results saving failed")
                return False
            
            # Test metrics calculation
            print("Testing metrics calculation...")
            metrics = generate_comprehensive_metrics_summary(mock_results)
            
            if 'primary_metrics' in metrics:
                print("✅ Metrics calculated successfully")
            else:
                print("❌ Metrics calculation failed")
                return False
            
            # Test finishing experiment
            print("Testing experiment finish...")
            summary = runner.finish_experiment()
            
            if 'status' in summary and summary['status'] == 'completed':
                print("✅ Experiment finished successfully")
            else:
                print("❌ Experiment finish failed")
                return False
            
            # Check final files
            exp_path = runner.get_experiment_path()
            expected_files = [
                "experiment_config.json",
                "experiment_summary.json",
                "data/substrate_nodes.csv",
                "data/vnr_batch_metadata.csv"
            ]
            
            for file_path in expected_files:
                full_path = exp_path / file_path
                if full_path.exists():
                    print(f"✅ File exists: {file_path}")
                else:
                    print(f"❌ File missing: {file_path}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ ExperimentRunner test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in experiment I/O."""
    print("\n" + "=" * 70)
    print("TEST 7: Error Handling")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import (
            ExperimentIOError, 
            load_experiment_data, 
            load_algorithm_results,
            load_experiment_config
        )
        
        # Test loading from non-existent directory
        print("Testing load from non-existent directory...")
        try:
            load_experiment_data("non_existent_directory")
            print("❌ Should have raised ExperimentIOError")
            return False
        except ExperimentIOError:
            print("✅ Correctly raised ExperimentIOError for missing directory")
        
        # Test loading non-existent results file
        print("Testing load from non-existent results file...")
        try:
            load_algorithm_results("non_existent_file.json")
            print("❌ Should have raised ExperimentIOError")
            return False
        except ExperimentIOError:
            print("✅ Correctly raised ExperimentIOError for missing file")
        
        # Test loading invalid JSON
        print("Testing load from invalid JSON...")
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, 'w') as f:
                f.write("invalid json content {")
            
            try:
                load_algorithm_results(invalid_file)
                print("❌ Should have raised ExperimentIOError")
                return False
            except ExperimentIOError:
                print("✅ Correctly raised ExperimentIOError for invalid JSON")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_integration_with_existing_models():
    """Test integration with the existing model ecosystem."""
    print("\n" + "=" * 70)
    print("TEST 8: Integration with Existing Models")
    print("=" * 70)
    
    try:
        from src.utils.io_utils import ExperimentRunner
        from src.utils.generators import generate_substrate_network, generate_vnr_batch, NetworkGenerationConfig
        from src.utils.metrics import generate_comprehensive_metrics_summary, EmbeddingResult
        
        print("Testing full integration workflow...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Create experiment
            runner = ExperimentRunner("integration_test", base_dir=temp_dir)
            
            # 2. Generate data using existing generators
            substrate = generate_substrate_network(
                nodes=15,
                topology="barabasi_albert",
                attachment_count=2,
                enable_memory_constraints=True,
                enable_delay_constraints=True
            )
            
            config = NetworkGenerationConfig(
                vnr_nodes_range=(2, 4),
                enable_memory_constraints=True,
                enable_delay_constraints=True
            )
            substrate_nodes = [str(i) for i in range(15)]
            vnr_batch = generate_vnr_batch(30, substrate_nodes, config)
            
            # 3. Setup experiment
            runner.setup_experiment(substrate, vnr_batch)
            
            # 4. Simulate algorithm results
            algorithm_results = []
            for i, vnr in enumerate(vnr_batch[:10]):  # Process first 10 VNRs
                success = i % 3 != 0  # 2/3 success rate
                revenue = sum(node.cpu_requirement for node in vnr.virtual_nodes.values()) if success else 0
                cost = revenue * 0.6 if success else revenue * 0.2
                
                result = EmbeddingResult(
                    vnr_id=vnr.vnr_id,
                    success=success,
                    revenue=revenue,
                    cost=cost,
                    execution_time=0.01 + i * 0.001,
                    vnr=vnr
                )
                algorithm_results.append(result)
            
            # 5. Save results and calculate metrics
            runner.save_results(algorithm_results, "integration_test_algorithm")
            metrics = generate_comprehensive_metrics_summary(algorithm_results)

            # 6. Finish experiment
            runner.finish_experiment()
            
            # 7. Verify integration
            exp_path = runner.get_experiment_path()
            
            # Check substrate constraint compatibility
            config_file = exp_path / "experiment_config.json"
            with open(config_file, 'r') as f:
                exp_config = json.load(f)
            
            substrate_constraints = exp_config['constraint_config']
            if (substrate_constraints['memory_constraints'] and 
                substrate_constraints['delay_constraints']):
                print("✅ Constraint configuration preserved")
            else:
                print("❌ Constraint configuration lost")
                return False
            
            # Check metrics calculation
            if 'primary_metrics' in metrics and 'acceptance_ratio' in metrics['primary_metrics']:
                acceptance_ratio = metrics['primary_metrics']['acceptance_ratio']
                expected_ratio = len([r for r in algorithm_results if r.success]) / len(algorithm_results)
                
                if abs(acceptance_ratio - expected_ratio) < 0.001:
                    print("✅ Metrics calculation integrated correctly")
                else:
                    print("❌ Metrics calculation integration failed")
                    return False
            else:
                print("❌ Metrics structure incorrect")
                return False
            
            print("✅ Full integration workflow completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_io_utils_tests():
    """Run all experiment I/O tests and report results."""
    print("VNE EXPERIMENT I/O TEST SUITE")
    print("=" * 70)
    print("Testing functional, modular, sensible I/O following VNE literature standards")
    print("=" * 70)
    
    tests = [
        ("Experiment I/O Imports", test_io_utils_imports),
        ("Directory Creation", test_directory_creation),
        ("Experiment Data I/O", test_experiment_data_io),
        ("Algorithm Results I/O", test_results_io),
        ("Metrics I/O", test_metrics_io),
        ("ExperimentRunner", test_experiment_runner),
        ("Error Handling", test_error_handling),
        ("Model Integration", test_integration_with_existing_models),
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
    print("EXPERIMENT I/O TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(tests)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL EXPERIMENT I/O TESTS PASSED!")
        print("\n📋 I/O Assessment:")
        print("✅ Functional - All I/O operations work correctly")
        print("✅ Modular - Clean separation using model built-ins")
        print("✅ Sensible - Logical interface design")
        print("✅ Standards - Integrates with VNE framework")
        print("\n🚀 Ready for algorithm development!")
        return True
    else:
        print("⚠️  SOME EXPERIMENT I/O TESTS FAILED!")
        print("\n🔧 Review failed tests and fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_io_utils_tests()
    sys.exit(0 if success else 1)
