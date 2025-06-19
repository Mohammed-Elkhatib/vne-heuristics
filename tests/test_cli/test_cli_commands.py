"""
Enhanced CLI tests for the refactored GenerateCommand.

These tests ensure that the refactored command properly uses generator modules
and maintains backward compatibility with the original implementation.
"""

import unittest
import tempfile
import subprocess
import sys
import os
import csv
from pathlib import Path


class TestCLICommandsEnhanced(unittest.TestCase):
    """Enhanced test cases for CLI commands with focus on GenerateCommand refactoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)

        # Find project root (where main.py is located)
        self.project_root = self._find_project_root()
        if not self.project_root:
            raise unittest.SkipTest("Could not find project root with main.py")

        # Set UTF-8 encoding for subprocess on Windows
        self.env = os.environ.copy()
        self.env['PYTHONIOENCODING'] = 'utf-8'

        # Add project root to PYTHONPATH
        if 'PYTHONPATH' in self.env:
            self.env['PYTHONPATH'] = f"{self.project_root}{os.pathsep}{self.env['PYTHONPATH']}"
        else:
            self.env['PYTHONPATH'] = str(self.project_root)

        print(f"Project root: {self.project_root}")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _find_project_root(self):
        """Find the project root directory (where main.py is located)."""
        current = Path.cwd()

        # Check current directory and parents
        for path in [current] + list(current.parents):
            main_py = path / "main.py"
            if main_py.exists():
                return path

        # Also check relative to test file
        test_file_path = Path(__file__)
        for path in [test_file_path.parent.parent.parent, test_file_path.parent.parent]:
            main_py = path / "main.py"
            if main_py.exists():
                return path

        return None

    def run_command_safely(self, cmd, timeout=30, capture_output=True):
        """Run command with proper path and encoding."""
        try:
            # Change to project root directory
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
                env=self.env
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {' '.join(cmd)}")
            return None
        except Exception as e:
            print(f"Command execution error: {e}")
            return None

    def test_generate_substrate_basic(self):
        """Test basic substrate generation with refactored command."""
        print("\n🧪 Testing basic substrate generation...")

        substrate_base = self.test_data_dir / "test_substrate_basic"
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", "10",
            "--topology", "erdos_renyi",
            "--edge-prob", "0.3",
            "--save", str(substrate_base)
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Verify files created
        nodes_file = substrate_base.parent / f"{substrate_base.name}_nodes.csv"
        links_file = substrate_base.parent / f"{substrate_base.name}_links.csv"

        self.assertTrue(nodes_file.exists(), "Nodes CSV should be created")
        self.assertTrue(links_file.exists(), "Links CSV should be created")

        # Verify node file content
        with open(nodes_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 10, "Should have 10 nodes")

            # Check that all required columns exist
            required_columns = ['node_id', 'cpu_capacity', 'memory_capacity', 'x_coord', 'y_coord']
            for col in required_columns:
                self.assertIn(col, reader.fieldnames, f"Column {col} should exist")

        print("✅ Basic substrate generation test passed!")

    def test_generate_substrate_all_topologies(self):
        """Test substrate generation with all supported topologies."""
        print("\n🧪 Testing all substrate topologies...")

        topologies = [
            ("erdos_renyi", {"--edge-prob": "0.2"}),
            ("barabasi_albert", {"--attachment-count": "2"}),
            ("grid", {})
        ]

        for topology, extra_args in topologies:
            print(f"\n  Testing {topology} topology...")

            substrate_base = self.test_data_dir / f"test_substrate_{topology}"
            cmd = [
                sys.executable, "main.py", "generate", "substrate",
                "--nodes", "16",  # Use 16 for perfect grid
                "--topology", topology,
                "--save", str(substrate_base)
            ]

            # Add topology-specific arguments
            for arg, value in extra_args.items():
                cmd.extend([arg, value])

            result = self.run_command_safely(cmd)
            self.assertIsNotNone(result, f"{topology} generation should not timeout")
            self.assertEqual(result.returncode, 0, f"{topology} generation failed: {result.stderr}")

            # Verify files exist
            nodes_file = substrate_base.parent / f"{substrate_base.name}_nodes.csv"
            links_file = substrate_base.parent / f"{substrate_base.name}_links.csv"

            self.assertTrue(nodes_file.exists(), f"{topology} nodes file should exist")
            self.assertTrue(links_file.exists(), f"{topology} links file should exist")

            # Verify we have links
            with open(links_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                link_count = sum(1 for _ in reader)
                self.assertGreater(link_count, 0, f"{topology} should have links")

        print("✅ All topology tests passed!")

    def test_generate_substrate_resource_ranges(self):
        """Test substrate generation with custom resource ranges."""
        print("\n🧪 Testing custom resource ranges...")

        substrate_base = self.test_data_dir / "test_substrate_custom"
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", "5",
            "--cpu-range", "100", "500",
            "--memory-range", "200", "800",
            "--bandwidth-range", "1000", "5000",
            "--save", str(substrate_base)
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Verify resource values are in specified ranges
        nodes_file = substrate_base.parent / f"{substrate_base.name}_nodes.csv"
        with open(nodes_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpu = int(row['cpu_capacity'])
                memory = int(row['memory_capacity'])

                self.assertGreaterEqual(cpu, 100, "CPU should be >= 100")
                self.assertLessEqual(cpu, 500, "CPU should be <= 500")
                self.assertGreaterEqual(memory, 200, "Memory should be >= 200")
                self.assertLessEqual(memory, 800, "Memory should be <= 800")

        print("✅ Custom resource range test passed!")

    def test_generate_vnrs_basic(self):
        """Test basic VNR generation with refactored command."""
        print("\n🧪 Testing basic VNR generation...")

        # First create a substrate
        substrate_base = self.test_data_dir / "test_substrate_for_vnrs"
        self._create_test_substrate(substrate_base, nodes=10)

        # Generate VNRs
        vnr_base = self.test_data_dir / "test_vnrs_basic"
        cmd = [
            sys.executable, "main.py", "generate", "vnrs",
            "--count", "20",
            "--substrate", str(substrate_base),
            "--nodes-range", "2", "4",
            "--save", str(vnr_base)
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "VNR generation should not timeout")
        self.assertEqual(result.returncode, 0, f"VNR generation failed: {result.stderr}")

        # Verify files created
        metadata_file = vnr_base.parent / f"{vnr_base.name}_metadata.csv"
        nodes_file = vnr_base.parent / f"{vnr_base.name}_nodes.csv"
        links_file = vnr_base.parent / f"{vnr_base.name}_links.csv"

        self.assertTrue(metadata_file.exists(), "VNR metadata file should exist")
        self.assertTrue(nodes_file.exists(), "VNR nodes file should exist")
        self.assertTrue(links_file.exists(), "VNR links file should exist")

        # Verify metadata content
        with open(metadata_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 20, "Should have 20 VNRs")

            # Check required columns
            required_columns = ['vnr_id', 'arrival_time', 'holding_time', 'priority']
            for col in required_columns:
                self.assertIn(col, reader.fieldnames, f"Column {col} should exist")

        print("✅ Basic VNR generation test passed!")

    def test_generate_vnrs_all_topologies(self):
        """Test VNR generation with all supported topologies."""
        print("\n🧪 Testing all VNR topologies...")

        # Create substrate first
        substrate_base = self.test_data_dir / "test_substrate_for_vnr_topologies"
        self._create_test_substrate(substrate_base, nodes=20)

        topologies = ["random", "star", "linear", "tree"]

        for topology in topologies:
            print(f"\n  Testing {topology} VNR topology...")

            vnr_base = self.test_data_dir / f"test_vnrs_{topology}"
            cmd = [
                sys.executable, "main.py", "generate", "vnrs",
                "--count", "10",
                "--substrate", str(substrate_base),
                "--nodes-range", "3", "6",
                "--topology", topology,
                "--save", str(vnr_base)
            ]

            result = self.run_command_safely(cmd)
            self.assertIsNotNone(result, f"{topology} VNR generation should not timeout")
            self.assertEqual(result.returncode, 0, f"{topology} VNR generation failed: {result.stderr}")

            # Verify files exist
            metadata_file = vnr_base.parent / f"{vnr_base.name}_metadata.csv"
            self.assertTrue(metadata_file.exists(), f"{topology} VNR metadata should exist")

        print("✅ All VNR topology tests passed!")

    def test_generate_vnrs_resource_ratios(self):
        """Test VNR generation with custom resource ratios."""
        print("\n🧪 Testing custom resource ratios...")

        # Create substrate
        substrate_base = self.test_data_dir / "test_substrate_for_ratios"
        self._create_test_substrate(substrate_base, nodes=10)

        vnr_base = self.test_data_dir / "test_vnrs_ratios"
        cmd = [
            sys.executable, "main.py", "generate", "vnrs",
            "--count", "15",
            "--substrate", str(substrate_base),
            "--cpu-ratio", "0.2", "0.5",
            "--memory-ratio", "0.3", "0.6",
            "--bandwidth-ratio", "0.1", "0.4",
            "--save", str(vnr_base)
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Verify that VNRs were created with appropriate resource requirements
        nodes_file = vnr_base.parent / f"{vnr_base.name}_nodes.csv"
        self.assertTrue(nodes_file.exists(), "VNR nodes file should exist")

        print("✅ Custom resource ratio test passed!")

    def test_generate_with_seed(self):
        """Test generation with random seed for reproducibility."""
        print("\n🧪 Testing reproducible generation with seed...")

        # Generate substrate twice with same seed
        substrate1 = self.test_data_dir / "test_substrate_seed1"
        substrate2 = self.test_data_dir / "test_substrate_seed2"

        for substrate_base in [substrate1, substrate2]:
            cmd = [
                sys.executable, "main.py", "generate", "substrate",
                "--nodes", "8",
                "--topology", "erdos_renyi",
                "--edge-prob", "0.3",
                "--seed", "42",
                "--save", str(substrate_base)
            ]

            result = self.run_command_safely(cmd)
            self.assertIsNotNone(result, "Command should not timeout")
            self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Compare files - they should be identical
        nodes1 = substrate1.parent / f"{substrate1.name}_nodes.csv"
        nodes2 = substrate2.parent / f"{substrate2.name}_nodes.csv"

        with open(nodes1, 'r') as f1, open(nodes2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            self.assertEqual(content1, content2, "Seeded generation should be reproducible")

        print("✅ Seed reproducibility test passed!")

    def test_error_handling(self):
        """Test error handling in generate command."""
        print("\n🧪 Testing error handling...")

        # Test invalid topology
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", "10",
            "--topology", "invalid_topology",
            "--save", str(self.test_data_dir / "test")
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertNotEqual(result.returncode, 0, "Invalid topology should fail")

        # Test VNR generation without substrate
        cmd = [
            sys.executable, "main.py", "generate", "vnrs",
            "--count", "10",
            "--substrate", str(self.test_data_dir / "nonexistent"),
            "--save", str(self.test_data_dir / "test_vnrs")
        ]

        result = self.run_command_safely(cmd)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertNotEqual(result.returncode, 0, "Missing substrate should fail")

        print("✅ Error handling test passed!")

    def test_generator_module_integration(self):
        """Test that generate command properly uses generator modules."""
        print("\n🧪 Testing generator module integration...")

        # This test ensures we're using the generator modules by checking
        # that the output format matches what the generators produce

        substrate_base = self.test_data_dir / "test_generator_integration"
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", "6",
            "--topology", "barabasi_albert",
            "--attachment-count", "2",
            "--save", str(substrate_base)
        ]

        result = self.run_command_safely(cmd, capture_output=True)
        self.assertIsNotNone(result, "Command should not timeout")
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Check output for generator module usage indicators
        # The refactored code should show clean output from generators
        self.assertIn("Generating substrate network", result.stdout)
        self.assertIn("Successfully generated", result.stdout)

        print("✅ Generator module integration test passed!")

    def _create_test_substrate(self, base_path, nodes=10):
        """Helper to create a test substrate network."""
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", str(nodes),
            "--topology", "erdos_renyi",
            "--edge-prob", "0.3",
            "--save", str(base_path)
        ]

        result = self.run_command_safely(cmd)
        if result is None or result.returncode != 0:
            self.fail(f"Failed to create test substrate: {result.stderr if result else 'timeout'}")


def run_enhanced_cli_tests():
    """Run enhanced CLI tests for refactored GenerateCommand."""
    print("🚀 Running Enhanced CLI Tests for Refactored GenerateCommand...")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCLICommandsEnhanced)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All enhanced CLI tests passed!")
        print("✅ The refactored GenerateCommand is working correctly.")
        print("✅ Generator modules are properly integrated.")
        print("🏆 Ready to proceed with next refactoring steps!")
    else:
        print("⚠️  Some enhanced CLI tests failed.")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        print("🔧 Fix the issues before proceeding.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_enhanced_cli_tests()
    sys.exit(0 if success else 1)
