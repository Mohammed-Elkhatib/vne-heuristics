# VNE Heuristics

A comprehensive framework for Virtual Network Embedding (VNE) algorithm implementation and evaluation.

## Overview

This project implements and evaluates various heuristic algorithms for solving the Virtual Network Embedding (VNE) problem. The VNE problem involves embedding virtual network requests (VNRs) onto a substrate network while optimizing metrics such as acceptance ratio, cost, revenue, and resource utilization.

The framework provides a complete toolkit for VNE research, including network generation, algorithm implementation, performance evaluation, and result analysis.

## Features

- **Complete VNE Framework**: End-to-end solution for VNE research
- **Command-Line Interface**: Professional CLI for all operations
- **Algorithm Implementation**: Extensible framework with baseline algorithms
- **Network Generation**: Configurable substrate and VNR generation
- **Performance Metrics**: Comprehensive evaluation and analysis
- **Data Management**: Robust file I/O and result storage
- **Configuration System**: Flexible parameter management
- **Professional Logging**: Detailed execution tracking

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohammed-Elkhatib/vne-heuristics.git
   cd vne-heuristics
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

1. **Generate a substrate network:**
   ```bash
   python main.py generate substrate --nodes 20 --save data/substrate_20
   ```

2. **Generate VNRs:**
   ```bash
   python main.py generate vnrs --count 50 --substrate data/substrate_20 --save data/vnrs_batch1
   ```

3. **Run algorithm:**
   ```bash
   python main.py run --algorithm yu --substrate data/substrate_20 --vnrs data/vnrs_batch1 --mode batch
   ```

4. **Analyze results:**
   ```bash
   python main.py metrics --results data/output/results/results_yu2008_*.json --output data/metrics.csv
   ```

## Project Structure

```
vne-heuristics/
│
├── src/                           # Source code
│   ├── algorithms/                # Algorithm implementations
│   │   ├── base_algorithm.py      # Abstract base class for all algorithms
│   │   └── baseline/              # Baseline algorithms
│   │       └── yu_2008_algorithm.py  # Yu et al. (2008) two-stage algorithm
│   ├── models/                    # Core data models
│   │   ├── substrate.py           # Physical network representation
│   │   ├── virtual_request.py     # Virtual network request representation
│   │   └── vnr_batch.py           # VNR batch management
│   └── utils/                     # Utility modules
│       ├── generators/            # Network and VNR generators
│       │   ├── generation_config.py      # Generation configuration
│       │   ├── substrate_generators.py   # Substrate network generators
│       │   └── vnr_generators.py         # VNR generators
│       ├── metrics.py             # Performance metrics calculation
│       └── io_utils.py            # File I/O operations
│
├── cli/                           # Command-line interface
│   ├── argument_parser.py         # CLI argument parsing
│   ├── commands/                  # Command implementations
│   │   ├── generate_command.py    # Generate networks and VNRs
│   │   ├── run_command.py         # Run algorithms
│   │   ├── metrics_command.py     # Calculate metrics
│   │   └── config_command.py      # Configuration management
│   └── exceptions.py              # CLI exception classes
│
├── core/                          # Core framework components
│   ├── algorithm_registry.py      # Algorithm discovery and registration
│   ├── error_handler.py           # Centralized error handling
│   └── progress_reporter.py       # Progress reporting system
│
├── data/                          # Data directory (gitignored)
│   ├── input/                     # Generated networks and VNRs
│   └── output/                    # Algorithm results and metrics
│
├── main.py                        # CLI entry point
├── config_management.py           # Configuration management system
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Implemented Algorithms

### Baseline Algorithms

1. **Yu et al. (2008)** - Two-Stage Heuristic (`yu2008`)
   - **Stage 1**: Greedy node mapping based on resource requirements
   - **Stage 2**: K-shortest path link mapping with bandwidth constraints
   - **Features**: k-shortest paths, multiple selection strategies, path caching
   - **Reference**: "Rethinking Virtual Network Embedding: Substrate Support for Path Splitting and Migration"
   - **Class**: `YuAlgorithm` in `src.algorithms.baseline.yu_2008_algorithm`

*More algorithms will be added as development continues...*

## Performance Metrics

The framework evaluates algorithms using comprehensive metrics:

### Essential Metrics
- **Acceptance Ratio**: Ratio of successfully embedded VNRs
- **Total Revenue**: Revenue from successful embeddings
- **Total Cost**: Cost of embedding attempts
- **Revenue-to-Cost Ratio**: Efficiency measure

### Additional Metrics
- **Resource Utilization**: CPU, memory, and bandwidth usage
- **Execution Time**: Algorithm performance
- **Path Length Statistics**: Network efficiency metrics
- **Time Series Analysis**: Performance over time

## Command Reference

### Generate Networks

```bash
# Generate substrate network
python main.py generate substrate --nodes N --topology TYPE --save FILE

# Examples:
python main.py generate substrate --nodes 20 --topology erdos_renyi --save data/substrate_20
python main.py generate substrate --nodes 50 --topology barabasi_albert --attachment-count 3 --save data/substrate_50

# Generate VNR batch  
python main.py generate vnrs --count N --substrate FILE --save FILE

# Examples:
python main.py generate vnrs --count 50 --substrate data/substrate_20 --save data/vnrs_50
python main.py generate vnrs --count 100 --substrate data/substrate_50 --topology star --save data/star_vnrs
```

### Run Algorithms

```bash
# List available algorithms
python main.py run --list-algorithms

# Run in batch mode
python main.py run --algorithm NAME --substrate FILE --vnrs FILE --mode batch

# Examples:
python main.py run --algorithm yu --substrate data/substrate_20 --vnrs data/vnrs_50 --mode batch
python main.py run --algorithm yu --substrate data/substrate_20 --vnrs data/vnrs_50 --mode online
```

### Analyze Results

```bash
# Calculate metrics
python main.py metrics --results FILE --output FILE

# Examples:
python main.py metrics --results data/output/results/results_yu2008_*.json --output data/metrics.csv
python main.py metrics --results results.json --output metrics.json --format json

# Generate time series analysis
python main.py metrics --results FILE --output FILE --time-series --window-size 3600
```

### Configuration

```bash
# Create default config file
python main.py config --create-default config.yaml

# Show current configuration
python main.py config --show
```

## Configuration

The framework uses a hierarchical configuration system supporting:

- **Configuration files** (YAML/JSON)
- **Environment variables** (prefix: `VNE_`)
- **Command-line overrides**

### Example Configuration

```yaml
network_generation:
  substrate_nodes: 50
  substrate_topology: "erdos_renyi"
  substrate_edge_probability: 0.15
  vnr_count: 100
  vnr_nodes_range: [2, 10]

algorithm:
  timeout_seconds: 300.0
  k_shortest_paths: 3
  cpu_weight: 1.0
  memory_weight: 1.0

file_paths:
  data_dir: "data"
  results_dir: "data/output/results"
  networks_dir: "data/input/networks"

logging:
  root_level: "INFO"
  log_file: "vne.log"
```

## Development and Testing

The VNE Heuristics framework includes comprehensive development tools and testing infrastructure to ensure code quality and algorithm correctness.

### Development Environment Setup

#### Prerequisites
- Python 3.8 or higher
- Git for version control
- Virtual environment (recommended)

#### Quick Development Setup
```bash
# Clone and set up the project
git clone https://github.com/Mohammed-Elkhatib/vne-heuristics.git
cd vne-heuristics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest>=7.0.0 black>=22.0.0 flake8>=4.0.0

# Verify installation
python main.py --help
```

### Running Tests

The framework includes comprehensive test coverage across all components. Tests are organized by module and functionality.

#### Test Structure
```
tests/
├── test_models/                 # Core model testing
│   ├── models_test_suite.py     # Comprehensive model tests
│   └── individual test files
├── test_algorithms/             # Algorithm testing
│   ├── vne_algorithm_unit_tests.py  # Complete algorithm tests
│   └── individual algorithm tests
├── test_utils/                  # Utility testing
│   ├── metrics_test_suite.py    # Metrics calculation tests
│   ├── test_generators.py       # Network generation tests
│   └── experiment_io_tests.py   # I/O functionality tests
└── test_cli/                    # CLI interface testing
    └── test_cli_commands.py     # Command-line interface tests
```

#### Running Test Suites

**Quick Test Verification**:
```bash
# Run all main test suites (recommended for development)
python tests/test_models/models_test_suite.py
python tests/test_algorithms/vne_algorithm_unit_tests.py
python tests/test_utils/metrics_test_suite.py
python tests/test_cli/test_cli_commands.py
```

**Comprehensive Testing with pytest**:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models/ -v           # Model tests only
python -m pytest tests/test_algorithms/ -v      # Algorithm tests only
python -m pytest tests/test_utils/ -v           # Utility tests only
python -m pytest tests/test_cli/ -v             # CLI tests only

# Run with coverage reporting
python -m pytest tests/ --cov=src --cov-report=html
```

**Performance and Integration Tests**:
```bash
# Run integration tests (end-to-end workflows)
python -m pytest tests/integration/ -v

# Run performance benchmarks
python -m pytest tests/performance/ -v -s
```

#### Test Categories and Coverage

**Core Model Tests** (`test_models/`):
- SubstrateNetwork resource management and thread safety
- VirtualNetworkRequest validation and constraint handling
- VNRBatch operations and serialization
- Integration between models

**Algorithm Tests** (`test_algorithms/`):
- BaseAlgorithm framework compliance
- Yu 2008 algorithm implementation verification
- Resource allocation and rollback mechanisms
- Constraint validation and Intra-VNR separation
- Performance benchmarking

**Utility Tests** (`test_utils/`):
- Network and VNR generation with various configurations
- Metrics calculation using standard VNE formulas
- File I/O operations and data persistence
- Configuration management and validation

**CLI Tests** (`test_cli/`):
- Command-line interface functionality
- Argument parsing and validation
- Error handling and user feedback
- Integration with core framework components

#### Expected Test Results

When running the test suites, you should see output like:
```
VNE MODELS TEST SUITE
======================================================================
✅ SubstrateNetwork Model         PASS
✅ VirtualNetworkRequest Model    PASS
✅ VNRBatch Model                 PASS
✅ Model Integration              PASS
✅ VNE Literature Compliance     PASS

Total: 5/5 tests passed
🎉 ALL MODEL TESTS PASSED!
```

### Code Quality

The framework maintains high code quality standards through automated checking and consistent style guidelines.

#### Code Style and Formatting

**Automatic Code Formatting**:
```bash
# Format all code (modifies files)
black src/ cli/ tests/

# Check formatting without making changes
black --check src/ cli/ tests/

# Format specific files
black src/algorithms/baseline/yu_2008_algorithm.py
```

**Linting and Style Checks**:
```bash
# Run flake8 linting
flake8 src/ cli/ tests/

# Run with specific configuration
flake8 src/ --max-line-length=100 --ignore=E203,W503
```

**Type Checking**:
```bash
# Run mypy type checking
mypy src/ cli/

# Check specific modules
mypy src/models/substrate.py
```

#### Code Quality Standards

The framework follows these quality standards:

**Python Style**:
- PEP 8 compliance with 100-character line limit
- Comprehensive type hints for all public functions
- Google-style docstrings with examples
- Structured exception handling

**VNE-Specific Standards**:
- Standard VNE terminology and variable naming
- Literature-compliant algorithm implementations
- Proper resource management and thread safety
- Comprehensive constraint validation

**Testing Requirements**:
- Minimum 85% code coverage
- Tests for both success and failure scenarios
- Performance benchmarking for algorithms
- Integration tests for complete workflows

#### Pre-commit Quality Checks

Set up automatic quality checks before commits:
```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running code quality checks..."

# Format check
black --check src/ cli/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run: black src/ cli/ tests/"
    exit 1
fi

# Linting
flake8 src/ cli/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Linting check failed."
    exit 1
fi

# Type checking
mypy src/ cli/
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed."
    exit 1
fi

echo "✅ All quality checks passed!"
EOF

chmod +x .git/hooks/pre-commit
```

### Development Workflow

#### Adding New Features

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/new-algorithm
   ```

2. **Implement with Tests**:
   ```bash
   # Implement your feature
   # Add comprehensive tests
   # Update documentation
   ```

3. **Quality Checks**:
   ```bash
   # Run all tests
   python -m pytest tests/ -v
   
   # Check code quality
   black --check src/ cli/ tests/
   flake8 src/ cli/ tests/
   mypy src/ cli/
   ```

4. **Integration Testing**:
   ```bash
   # Test CLI integration
   python main.py run --algorithm your_algorithm --substrate data/test_substrate --vnrs data/test_vnrs
   
   # Verify metrics calculation
   python main.py metrics --results results.json --output metrics.csv
   ```

#### Algorithm Development

For implementing new VNE algorithms:

1. **Study the Literature**: Understand the algorithm's theoretical foundation
2. **Plan Implementation**: Design the algorithm structure and parameters
3. **Inherit from BaseAlgorithm**: Use the provided framework
4. **Implement Core Methods**: `_embed_single_vnr()` and `_cleanup_failed_embedding()`
5. **Add Comprehensive Tests**: Cover success, failure, and edge cases
6. **Document Thoroughly**: Include literature references and examples

See `CONTRIBUTING.md` for detailed algorithm development guidelines.

---

## Troubleshooting

This section covers common issues and their solutions when using the VNE Heuristics framework.

### Installation Issues

#### Python Version Compatibility
**Problem**: Import errors or unexpected behavior
```
ModuleNotFoundError: No module named 'typing_extensions'
```

**Solution**: Ensure Python 3.8+ is being used
```bash
# Check Python version
python --version  # Should be 3.8 or higher

# Use specific Python version if needed
python3.8 -m venv venv
python3.8 main.py --help
```

#### Missing Dependencies
**Problem**: Import errors for required packages
```
ModuleNotFoundError: No module named 'networkx'
```

**Solution**: Install all required dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Install specific missing package
pip install networkx>=2.8.0
```

#### Virtual Environment Issues
**Problem**: Dependencies installed globally but not in virtual environment

**Solution**: Ensure virtual environment is activated
```bash
# Create new virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation (should show venv in prompt)
which python  # Should point to venv/bin/python
```

### CLI Command Issues

#### File Not Found Errors
**Problem**: CLI cannot find input files
```
Error: Substrate files not found: substrate_20_nodes.csv, substrate_20_links.csv
```

**Solution**: Check file naming convention and paths
```bash
# Verify file structure (substrate networks need paired files)
ls data/substrate_20*
# Expected: substrate_20_nodes.csv, substrate_20_links.csv

# Generate properly formatted files
python main.py generate substrate --nodes 20 --save data/substrate_20

# Use absolute paths if needed
python main.py run --substrate /full/path/to/data/substrate_20 --vnrs /full/path/to/data/vnrs_50
```

#### Algorithm Not Found
**Problem**: Algorithm not available or typo in name
```
Error: Algorithm 'yu2009' not available. Available: yu2008
```

**Solution**: Check available algorithms and correct spelling
```bash
# List all available algorithms
python main.py run --list-algorithms

# Use correct algorithm name (case-sensitive)
python main.py run --algorithm yu2008 --substrate data/substrate_20 --vnrs data/vnrs_50
```

#### Permission Errors
**Problem**: Cannot write to output directories
```
PermissionError: [Errno 13] Permission denied: 'data/output/results'
```

**Solution**: Check directory permissions and create directories
```bash
# Create output directories
mkdir -p data/output/results data/output/metrics

# Fix permissions (Linux/Mac)
chmod 755 data/output/results

# Use alternative output location
python main.py run --output /tmp/results.json --algorithm yu2008 --substrate data/substrate_20 --vnrs data/vnrs_50
```

### Algorithm and Embedding Issues

#### All VNRs Rejected
**Problem**: Algorithm rejects all or most VNRs
```
Acceptance Ratio: 0.00 (0 successful out of 100)
```

**Potential Causes and Solutions**:

1. **Resource Constraints Too Tight**:
   ```bash
   # Generate substrate with higher capacities
   python main.py generate substrate --nodes 20 --cpu-range 100 300 --bandwidth-range 200 500 --save data/high_capacity
   
   # Generate VNRs with lower requirements
   python main.py generate vnrs --count 50 --substrate data/high_capacity --cpu-ratio 0.05 0.15 --save data/low_demand_vnrs
   ```

2. **Network Connectivity Issues**:
   ```bash
   # Generate more connected substrate
   python main.py generate substrate --nodes 20 --topology erdos_renyi --edge-prob 0.3 --save data/connected_substrate
   
   # Use different topology
   python main.py generate substrate --nodes 20 --topology barabasi_albert --attachment-count 4 --save data/scale_free
   ```

3. **VNR Size Too Large**:
   ```bash
   # Generate smaller VNRs
   python main.py generate vnrs --count 50 --nodes-range 2 4 --substrate data/substrate_20 --save data/small_vnrs
   ```

#### Poor Algorithm Performance
**Problem**: Algorithm takes too long or uses too much memory

**Solutions**:

1. **Optimize Algorithm Parameters**:
   ```bash
   # For Yu algorithm, reduce k-paths
   python main.py run --algorithm yu2008 --substrate data/substrate_20 --vnrs data/vnrs_50
   # (Yu algorithm uses k_paths=1 by default for performance)
   ```

2. **Use Smaller Test Cases**:
   ```bash
   # Test with smaller networks first
   python main.py generate substrate --nodes 10 --save data/test_substrate
   python main.py generate vnrs --count 20 --substrate data/test_substrate --save data/test_vnrs
   ```

3. **Monitor Resource Usage**:
   ```bash
   # Run with progress reporting
   python main.py run --algorithm yu2008 --substrate data/substrate_20 --vnrs data/vnrs_50 --progress
   ```

### Data and File Format Issues

#### CSV Format Errors
**Problem**: Generated files cannot be loaded
```
Error: Invalid CSV format in substrate_nodes.csv
```

**Solution**: Regenerate files with proper format
```bash
# Remove corrupted files
rm data/substrate_20*

# Regenerate with verbose output
python main.py generate substrate --nodes 20 --save data/substrate_20 --verbose

# Verify file contents
head -5 data/substrate_20_nodes.csv
head -5 data/substrate_20_links.csv
```

#### Encoding Issues
**Problem**: Special characters in file paths or data
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution**: Use proper UTF-8 encoding and avoid special characters
```bash
# Use simple ASCII filenames
python main.py generate substrate --nodes 20 --save data/substrate_20

# Avoid special characters in paths
# Good: data/substrate_20
# Bad: data/sübstrate_20, data/substrate (with spaces)
```

### Memory and Performance Issues

#### Memory Usage Too High
**Problem**: Framework uses too much memory for large networks

**Solutions**:

1. **Process in Smaller Batches**:
   ```python
   # Split large VNR batches into smaller chunks
   python main.py generate vnrs --count 100 --substrate data/substrate_100 --save data/vnrs_part1
   python main.py generate vnrs --count 100 --substrate data/substrate_100 --save data/vnrs_part2
   ```

2. **Disable Caching**:
   ```python
   # For algorithms that support it, disable path caching
   # (This is algorithm-specific; check algorithm documentation)
   ```

3. **Use Simpler Topologies**:
   ```bash
   # Grid topologies use less memory than random graphs
   python main.py generate substrate --nodes 100 --topology grid --save data/grid_substrate
   ```

#### Slow Performance
**Problem**: Commands take too long to execute

**Solutions**:

1. **Profile Performance**:
   ```bash
   # Run with timing information
   time python main.py run --algorithm yu2008 --substrate data/substrate_50 --vnrs data/vnrs_100
   ```

2. **Optimize Network Size**:
   ```bash
   # Start with smaller networks for testing
   python main.py generate substrate --nodes 20 --save data/small_substrate
   python main.py generate vnrs --count 50 --substrate data/small_substrate --save data/small_vnrs
   ```

3. **Use Efficient Algorithms**:
   ```bash
   # Yu 2008 is typically the fastest baseline algorithm
   python main.py run --algorithm yu2008 --substrate data/substrate_20 --vnrs data/vnrs_50
   ```

### Testing Issues

#### Test Failures
**Problem**: Tests fail when running the test suite

**Common Solutions**:

1. **Missing Test Dependencies**:
   ```bash
   pip install pytest>=7.0.0
   ```

2. **Path Issues in Tests**:
   ```bash
   # Run tests from project root directory
   cd /path/to/vne-heuristics
   python -m pytest tests/ -v
   ```

3. **Resource Conflicts**:
   ```bash
   # Run tests individually if there are conflicts
   python tests/test_models/models_test_suite.py
   python tests/test_algorithms/vne_algorithm_unit_tests.py
   ```

#### Test Coverage Issues
**Problem**: Coverage reports show unexpected low coverage

**Solution**: Run coverage with proper source specification
```bash
# Install coverage tools
pip install pytest-cov

# Run with explicit source specification
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# View detailed coverage report
open htmlcov/index.html  # Opens coverage report in browser
```

### Getting Help

If you encounter issues not covered here:

1. **Check Log Files**: Look for detailed error messages in console output
2. **Use Debug Mode**: Run commands with `--debug` or `--verbose` flags
3. **Verify Installation**: Re-run the installation steps
4. **Check Dependencies**: Ensure all required packages are installed correctly
5. **Review Documentation**: Check API documentation for correct usage patterns

**Debug Commands**:
```bash
# Enable verbose logging
python main.py --verbose generate substrate --nodes 10 --save data/debug_substrate

# Enable debug mode
python main.py --debug run --algorithm yu2008 --substrate data/debug_substrate --vnrs data/debug_vnrs

# Check framework status
python main.py run --list-algorithms  # Should show available algorithms
python main.py config --show          # Should show current configuration
```

For persistent issues or bug reports, please refer to the project's issue tracking system with detailed information about your environment, command executed, and error messages received.

## File Formats

### Substrate Networks
Generated as CSV pairs:
- `substrate_NAME_nodes.csv` - Node resources and coordinates
- `substrate_NAME_links.csv` - Link capacities and attributes

### VNR Batches
Generated as CSV triplets:
- `vnrs_NAME_metadata.csv` - VNR timing and priorities
- `vnrs_NAME_nodes.csv` - Virtual node requirements
- `vnrs_NAME_links.csv` - Virtual link requirements

### Results
Saved as JSON with comprehensive embedding information:
```json
{
  "vnr_id": "0",
  "success": true,
  "node_mapping": {"0": "3", "1": "7"},
  "link_mapping": {"0-1": ["3", "2", "7"]},
  "revenue": 250.75,
  "cost": 45.50,
  "execution_time": 0.0023,
  "algorithm_name": "Yu et al. (2008) Two-Stage Algorithm"
}
```

## Expected Output Format

```
VNR 0: accepted, nodes mapped as {0->3, 1->7}, links mapped as {(0,1)->[3,2,7]}, revenue=150, cost=80
VNR 1: rejected, reason=insufficient_bandwidth
...
Summary: Acceptance ratio=0.85, Total revenue=1200, Total cost=800, Revenue/Cost=1.5
```

## Development

### Adding New Algorithms

1. **Inherit from BaseAlgorithm:**
   ```python
   from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
   
   class MyAlgorithm(BaseAlgorithm):
       def __init__(self):
           super().__init__("My Algorithm Name")
       
       def _embed_single_vnr(self, vnr, substrate):
           # Implementation with resource allocation
           return EmbeddingResult(...)
       
       def _cleanup_failed_embedding(self, vnr, substrate, result):
           # Cleanup allocated resources
           pass
   ```

2. **Place in algorithm directory:**
   - Create file in `src/algorithms/` or appropriate subdirectory
   - The algorithm registry will automatically discover it

### Running Tests

```bash
# Run all tests (when implemented)
python -m pytest

# Run specific test
python -m pytest tests/test_algorithms.py
```

## Documentation

- **[Technical Documentation](docs/technical_documentation.md)** - Detailed code architecture
- **[Algorithm Guide](docs/ALGORITHMS.md)** - Algorithm implementation details
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Data Format Reference](data/data_documentation.md)** - File format specifications

## Requirements

- Python 3.8+
- NetworkX 2.8+
- NumPy 1.21+
- PyYAML 6.0+
- See `requirements.txt` for complete list

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Research Applications

This framework is designed for:

- **Algorithm Comparison**: Benchmarking different VNE approaches
- **Parameter Studies**: Analyzing algorithm sensitivity
- **Performance Evaluation**: Comprehensive metrics collection
- **Research Publications**: Professional-grade implementation
- **Educational Use**: Learning VNE concepts and algorithms

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vne_heuristics,
  title={VNE Heuristics: A Framework for Virtual Network Embedding Research},
  author={Mohammed Elkhatib},
  year={2025},
  url={https://github.com/Mohammed-Elkhatib/vne-heuristics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yu et al. for their foundational VNE algorithm (2008)
- Fischer et al. for their comprehensive VNE survey (2013)
- NetworkX team for the excellent graph library
- Contributors to the VNE research community

## Support

- 📖 Check the [documentation](docs/)
- 🐛 Report bugs via [GitHub Issues](https://github.com/Mohammed-Elkhatib/vne-heuristics/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/Mohammed-Elkhatib/vne-heuristics/discussions)
- 📧 Contact: elkhatibmuhammad@gmail.com