# VNE Heuristics

A comparative analysis of heuristic algorithms for Virtual Network Embedding (VNE) problems.

## Overview

This repository implements and evaluates various heuristic algorithms for solving the Virtual Network Embedding (VNE) problem. The VNE problem involves embedding virtual network requests (VNRs) onto a substrate network while optimizing various metrics such as acceptance ratio, cost, revenue, and revenue-to-cost ratio.

The implemented algorithms include baseline approaches from Fischer et al.'s 2013 survey paper ["Virtual Network Embedding: A Survey"](https://ieeexplore.ieee.org/abstract/document/6463372) as well as more recent algorithms from the literature.

## Project Structure

```
vne-heuristics/
│
├── src/                         # Source code directory
│   ├── algorithms/              # Algorithm implementations
│   ├── models/                  # Data models
│   ├── utils/                   # Utility functions
│   └── experiments/             # Experiment configurations and runners
│
├── data/                        # Data directory
│   ├── input/                   # Input data (substrate networks, VNRs)
│   └── output/                  # Experiment results
│
└── tests/                       # Unit tests
```

## Algorithms

This project implements some of the following VNE heuristic algorithms:

### Baseline Algorithms (Fischer et al., 2013)

1. **[Algorithm 1]** - Brief description
2. **[Algorithm 2]** - Brief description

### Recent Algorithms

3. **[Algorithm 3]** - Brief description
4. **[Algorithm 4]** - Brief description

## Metrics

The performance of each algorithm is evaluated using the following metrics:

- **Acceptance Ratio**: The ratio of accepted VNRs to the total number of VNRs.
- **Average Revenue**: The average revenue generated from successful embeddings.
- **Average Cost**: The average cost of embedding VNRs.
- **Revenue-to-Cost Ratio**: The ratio of revenue to cost, representing the efficiency of the embedding.
- **Node Utilization**: The utilization of substrate nodes.
- **Link Utilization**: The utilization of substrate links.
- **Running Time**: The computational time required for embedding.

## Getting Started

### Prerequisites

- Python 3.8+
- NetworkX
- NumPy
- Matplotlib
- (Add other dependencies as needed)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vne-heuristics.git
   cd vne-heuristics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run an experiment with a specific algorithm:

```bash
python -m src.experiments.runner --algorithm [algorithm_name] --substrate [substrate_file] --vnrs [vnrs_file]
```

Example:

```bash
python -m src.experiments.runner --algorithm baseline1 --substrate data/input/substrate1.json --vnrs data/input/vnrs1.json
```

## Results

Results will be saved in the `data/output/` directory, including:
- CSV files with metrics for each algorithm
- Visualization of the embedding
- Performance comparison plots

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fischer et al. for their comprehensive survey on VNE algorithms
- (Add other acknowledgments as needed)
