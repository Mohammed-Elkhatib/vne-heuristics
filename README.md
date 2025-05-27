# VNE Heuristics

A comprehensive framework for Virtual Network Embedding (VNE) algorithm implementation and evaluation.

## Overview

This project implements and evaluates various heuristic algorithms for solving the Virtual Network Embedding (VNE) problem. The VNE problem involves embedding virtual network requests (VNRs) onto a substrate network while optimizing metrics such as acceptance ratio, cost, revenue, and resource utilization.

The framework provides a complete toolkit for VNE research, including network generation, algorithm implementation, performance evaluation, and result analysis.

## Features

- 🎯 **Complete VNE Framework**: End-to-end solution for VNE research
- 🚀 **Command-Line Interface**: Professional CLI for all operations
- 📊 **Algorithm Implementation**: Extensible framework with baseline algorithms
- 🔧 **Network Generation**: Configurable substrate and VNR generation
- 📈 **Performance Metrics**: Comprehensive evaluation and analysis
- 💾 **Data Management**: Robust file I/O and result storage
- 🎛️ **Configuration System**: Flexible parameter management
- 📝 **Professional Logging**: Detailed execution tracking

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
   python main.py generate substrate --nodes 20 --save data/substrate_20.csv
   ```

2. **Generate VNRs:**
   ```bash
   python main.py generate vnrs --count 50 --substrate data/substrate_20.csv --save data/vnrs_batch1.csv
   ```

3. **Run algorithm:**
   ```bash
   python main.py run --algorithm yu2008 --substrate data/substrate_20.csv --vnrs data/vnrs_batch1.csv --mode batch
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
│   │       └── yu_2008.py         # Yu et al. (2008) two-stage algorithm
│   ├── models/                    # Core data models
│   │   ├── substrate.py           # Physical network representation
│   │   └── virtual_request.py     # Virtual network request representation
│   ├── utils/                     # Utility modules
│   │   ├── generators.py          # Network and VNR generators
│   │   ├── metrics.py             # Performance metrics calculation
│   │   └── io_utils.py            # File I/O operations
│   └── config.py                  # Configuration management
│
├── data/                          # Data directory (gitignored)
│   ├── input/                     # Generated networks and VNRs
│   └── output/                    # Algorithm results and metrics
│
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Implemented Algorithms

### Baseline Algorithms

1. **Yu et al. (2008)** - Two-Stage Heuristic (`yu2008`)
   - **Stage 1**: Greedy node mapping based on resource requirements
   - **Stage 2**: Shortest path link mapping with bandwidth constraints
   - **Features**: k-shortest paths, multiple selection strategies, path caching
   - **Reference**: "Rethinking Virtual Network Embedding: Substrate Support for Path Splitting and Migration"

*More algorithms coming soon...*

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

# Generate VNR batch  
python main.py generate vnrs --count N --substrate FILE --save FILE
```

### Run Algorithms

```bash
# List available algorithms
python main.py run --list-algorithms

# Run in batch mode
python main.py run --algorithm NAME --substrate FILE --vnrs FILE --mode batch

# Run in online mode (with arrival times)
python main.py run --algorithm NAME --substrate FILE --vnrs FILE --mode online
```

### Analyze Results

```bash
# Calculate metrics
python main.py metrics --results FILE --output FILE

# Generate time series analysis
python main.py metrics --results FILE --output FILE --time-series
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
  vnr_count: 100
  vnr_nodes_range: [2, 10]

algorithm:
  timeout_seconds: 300.0
  k_paths: 3
  cpu_weight: 1.0
  memory_weight: 1.0

file_paths:
  data_dir: "data"
  results_dir: "data/output/results"
```

## Expected Output Format

```
VNR 1: accepted, nodes mapped as {v1->s3, v2->s7}, links mapped as {(v1,v2)->[s3,s1,s7]}, revenue=150, cost=80
VNR 2: rejected, reason=insufficient_bandwidth
...
Summary: Acceptance ratio=0.85, Total revenue=1200, Total cost=800, Revenue/Cost=1.5
```

## Development

### Adding New Algorithms

1. **Inherit from BaseAlgorithm:**
   ```python
   from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
   
   class MyAlgorithm(BaseAlgorithm):
       def _embed_single_vnr(self, vnr, substrate):
           # Implementation here
           return EmbeddingResult(...)
   ```

2. **Register in CLI:**
   ```python
   # In main.py _discover_algorithms()
   self.available_algorithms['myalgorithm'] = MyAlgorithm
   ```

### Running Tests

```bash
# Run all tests
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