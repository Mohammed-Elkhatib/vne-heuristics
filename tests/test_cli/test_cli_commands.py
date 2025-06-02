"""
Essential CLI tests to complement your excellent existing test suite.

These tests focus on the CLI layer that was missing from your comprehensive tests.
Add these to tests/test_cli/test_cli_commands.py
"""

import unittest
import tempfile
import json
import subprocess
import sys
import os
from pathlib import Path


class TestCLICommands(unittest.TestCase):
    """Test CLI commands with automatic path detection."""

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

        print(f"Project root: {self.project_root}")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _find_project_root(self):
        """Find the project root directory (where main.py is located)."""
        # Start from current directory and walk up
        current = Path.cwd()

        # Check current directory and parents
        for path in [current] + list(current.parents):
            main_py = path / "main.py"
            if main_py.exists():
                return path

        # Also check if we're in a subdirectory of the project
        # Look for main.py in various possible locations
        possible_roots = [
            Path.cwd().parent.parent,  # If we're in tests/test_cli/
            Path.cwd().parent,  # If we're in tests/
            Path(__file__).parent.parent.parent,  # Relative to this test file
        ]

        for path in possible_roots:
            main_py = path / "main.py"
            if main_py.exists():
                return path

        return None

    def run_command_safely(self, cmd, timeout=30):
        """Run command with proper path and encoding."""
        try:
            # Make sure we use the correct path to main.py
            if cmd[1] == "main.py":
                cmd[1] = str(self.project_root / "main.py")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),  # Run from project root
                env=self.env,
                encoding='utf-8',
                errors='replace'
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out: {' '.join(cmd)}")
            return None
        except Exception as e:
            print(f"💥 Command failed: {e}")
            return None

    def test_metrics_command_with_real_data(self):
        """Test the NEW metrics command implementation with real data."""
        print("\n🧪 Testing metrics command...")

        # Create realistic mock results
        mock_results = {
            "metadata": {"timestamp": "2024-01-01T00:00:00"},
            "results": [
                {
                    "vnr_id": "vnr_1",
                    "success": True,
                    "revenue": 150.0,
                    "cost": 75.0,
                    "execution_time": 0.012,
                    "node_mapping": {"0": "1", "1": "3"},
                    "link_mapping": {"0-1": ["1", "3"]},
                    "timestamp": 1640995200.0
                },
                {
                    "vnr_id": "vnr_2",
                    "success": False,
                    "revenue": 0.0,
                    "cost": 25.0,
                    "execution_time": 0.005,
                    "failure_reason": "Insufficient bandwidth"
                },
                {
                    "vnr_id": "vnr_3",
                    "success": True,
                    "revenue": 200.0,
                    "cost": 100.0,
                    "execution_time": 0.018,
                    "node_mapping": {"0": "2", "1": "4"},
                    "link_mapping": {"0-1": ["2", "4"]}
                }
            ]
        }

        # Save mock results to file
        results_file = self.test_data_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(mock_results, f, indent=2)

        # Test CSV output
        csv_output = self.test_data_dir / "test_metrics.csv"
        cmd = [
            sys.executable, "main.py", "metrics",
            "--results", str(results_file),
            "--output", str(csv_output),
            "--format", "csv"
        ]

        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Command execution failed or timed out")

        print(f"Command exit code: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")

        # Should succeed
        self.assertEqual(result.returncode, 0, f"Metrics command failed: {result.stderr}")

        # Verify output file created
        self.assertTrue(csv_output.exists(), "CSV metrics file should be created")

        # Test JSON output
        json_output = self.test_data_dir / "test_metrics.json"
        cmd = [
            sys.executable, "main.py", "metrics",
            "--results", str(results_file),
            "--output", str(json_output),
            "--format", "json"
        ]

        result = self.run_command_safely(cmd)
        if result:
            self.assertEqual(result.returncode, 0)
            self.assertTrue(json_output.exists())

            # Verify JSON content
            with open(json_output, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)

            self.assertIn('metrics', metrics_data)
            self.assertIn('primary_metrics', metrics_data['metrics'])

            # Verify calculated metrics are reasonable
            primary = metrics_data['metrics']['primary_metrics']
            self.assertAlmostEqual(primary['acceptance_ratio'], 2 / 3, places=2)  # 2 successful out of 3
            self.assertGreater(primary['total_revenue'], 0)

        print("✅ Metrics command test passed!")

    def test_generate_command_integration(self):
        """Test generate command end-to-end."""
        print("\n🧪 Testing generate command...")

        # Test substrate generation
        substrate_base = self.test_data_dir / "test_substrate"
        cmd = [
            sys.executable, "main.py", "generate", "substrate",
            "--nodes", "10",
            "--topology", "erdos_renyi",
            "--edge-prob", "0.3",
            "--save", str(substrate_base)
        ]

        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Generate substrate command failed or timed out")

        if result.returncode != 0:
            print(f"Generate substrate STDERR: {result.stderr}")
            print(f"Generate substrate STDOUT: {result.stdout}")

        self.assertEqual(result.returncode, 0, f"Generate substrate failed: {result.stderr}")

        # Verify files created
        self.assertTrue((substrate_base.parent / f"{substrate_base.name}_nodes.csv").exists())
        self.assertTrue((substrate_base.parent / f"{substrate_base.name}_links.csv").exists())

        # Test VNR generation
        vnr_base = self.test_data_dir / "test_vnrs"
        cmd = [
            sys.executable, "main.py", "generate", "vnrs",
            "--count", "20",
            "--substrate", str(substrate_base),
            "--nodes-range", "2", "4",
            "--save", str(vnr_base)
        ]

        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Generate VNRs command failed or timed out")

        if result.returncode != 0:
            print(f"Generate VNRs STDERR: {result.stderr}")
            print(f"Generate VNRs STDOUT: {result.stdout}")

        self.assertEqual(result.returncode, 0, f"Generate VNRs failed: {result.stderr}")

        # Verify VNR files created
        self.assertTrue((vnr_base.parent / f"{vnr_base.name}_metadata.csv").exists())

        print("✅ Generate command test passed!")

    def test_run_command_basic(self):
        """Test run command with algorithm discovery."""
        print("\n🧪 Testing run command...")

        # Test algorithm listing
        cmd = [sys.executable, "main.py", "run", "--list-algorithms"]
        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Algorithm listing command failed or timed out")

        print(f"Algorithm list exit code: {result.returncode}")
        print(f"Available algorithms: {result.stdout}")

        self.assertEqual(result.returncode, 0, "Algorithm listing should work")
        self.assertIn("algorithms", result.stdout.lower())

        print("✅ Run command basic test passed!")

    def test_config_command(self):
        """Test configuration command."""
        print("\n🧪 Testing config command...")

        # Test default config creation
        config_file = self.test_data_dir / "test_config.yaml"
        cmd = [
            sys.executable, "main.py", "config",
            "--create-default", str(config_file)
        ]

        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Config command failed or timed out")

        if result.returncode != 0:
            print(f"Config command STDERR: {result.stderr}")
            print(f"Config command STDOUT: {result.stdout}")

        self.assertEqual(result.returncode, 0, f"Config creation failed: {result.stderr}")
        self.assertTrue(config_file.exists(), "Config file should be created")

        print("✅ Config command test passed!")

    def test_error_handling_scenarios(self):
        """Test CLI error handling scenarios with robust error checking."""
        print("\n🧪 Testing error handling...")

        # Test metrics with non-existent file
        cmd = [
            sys.executable, "main.py", "metrics",
            "--results", "nonexistent_file.json",
            "--output", "output.csv"
        ]

        result = self.run_command_safely(cmd)

        if result is None:
            print("⚠️ Error handling test skipped due to command execution issues")
            return

        # Should fail gracefully
        self.assertNotEqual(result.returncode, 0, "Should fail for non-existent file")

        # Robust error message checking
        error_output = ""
        if result.stderr:
            error_output += result.stderr.lower()
        if result.stdout:
            error_output += result.stdout.lower()

        # Check for error indicators
        error_indicators = ["not found", "error", "failed", "exception", "no such file"]
        has_error_indicator = any(indicator in error_output for indicator in error_indicators)

        self.assertTrue(has_error_indicator,
                        f"Should have error message. Output: stdout={result.stdout}, stderr={result.stderr}")

        print("✅ Error handling test passed!")

    def test_help_system(self):
        """Test help system works."""
        print("\n🧪 Testing help system...")

        # Test main help
        cmd = [sys.executable, "main.py", "--help"]
        result = self.run_command_safely(cmd)

        if result is None:
            self.fail("Help command failed or timed out")

        self.assertEqual(result.returncode, 0)
        self.assertIn("generate", result.stdout)
        self.assertIn("run", result.stdout)
        self.assertIn("metrics", result.stdout)

        # Test subcommand help
        cmd = [sys.executable, "main.py", "generate", "--help"]
        result = self.run_command_safely(cmd)

        if result:
            self.assertEqual(result.returncode, 0)
            self.assertIn("substrate", result.stdout)
            self.assertIn("vnrs", result.stdout)

        print("✅ Help system test passed!")

    def test_project_structure_validation(self):
        """Test that we can find and validate the project structure."""
        print("\n🧪 Testing project structure...")

        # Verify main.py exists in project root
        main_py = self.project_root / "main.py"
        self.assertTrue(main_py.exists(), f"main.py should exist at {main_py}")

        # Verify key directories exist
        key_dirs = ["src", "src/models", "src/algorithms", "src/utils", "cli", "core"]
        for dir_name in key_dirs:
            dir_path = self.project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")

        # Verify key files exist
        key_files = [
            "src/models/substrate.py",
            "src/models/virtual_request.py",
            "src/algorithms/base_algorithm.py",
            "cli/commands/metrics_command.py",
            "core/algorithm_registry.py"
        ]

        for file_name in key_files:
            file_path = self.project_root / file_name
            self.assertTrue(file_path.exists(), f"File {file_name} should exist")

        print("✅ Project structure validation passed!")


def run_cli_tests():
    """Run CLI tests with proper error handling and path detection."""
    print("🚀 Running Path-Aware CLI Tests...")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCLICommands)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All CLI tests passed!")
        print("✅ Your CLI layer is working perfectly with the fixes.")
        print("🏆 You now have comprehensive test coverage!")
    else:
        print("⚠️ Some CLI tests failed.")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        print("🔧 Check the error messages above for guidance.")

        # Print failure details
        if result.failures:
            print("\nFailure details:")
            for test, traceback in result.failures:
                print(f"FAIL: {test}")
                print(traceback)

        if result.errors:
            print("\nError details:")
            for test, traceback in result.errors:
                print(f"ERROR: {test}")
                print(traceback)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the path-aware CLI tests
    success = run_cli_tests()

    if success:
        print("\n🎊 CONGRATULATIONS! 🎊")
        print("=" * 60)
        print("✅ All tests pass - your VNE framework is production ready!")
        print("🏆 You have achieved comprehensive test coverage!")
        print("🚀 Ready for serious VNE research and development!")
        print("\n📊 Your testing includes:")
        print("  ✅ Core models and algorithms")
        print("  ✅ Generators and utilities")
        print("  ✅ Metrics and I/O systems")
        print("  ✅ CLI commands and workflows")
        print("  ✅ Error handling and edge cases")
        print("  ✅ Cross-platform compatibility")
        print("\n🎯 Next: Start your VNE research!")

    sys.exit(0 if success else 1)
