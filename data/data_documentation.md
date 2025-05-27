# Data Directory

This directory contains generated network data, algorithm results, and experimental datasets for the VNE Heuristics framework.

## Directory Structure

```
data/
├── input/                     # Generated networks and VNR batches
│   ├── networks/             # Substrate network files
│   │   ├── substrate_N_topology.csv        # Network metadata
│   │   ├── substrate_N_topology_nodes.csv  # Node resources
│   │   └── substrate_N_topology_links.csv  # Link resources
│   └── vnrs/                 # Virtual network request batches
│       ├── vnrs_COUNT_substrate.csv        # Batch metadata
│       ├── vnrs_COUNT_substrate_nodes.csv  # VNR node requirements
│       ├── vnrs_COUNT_substrate_links.csv  # VNR link requirements
│       └── vnrs_COUNT_substrate_meta.csv   # VNR timing and metadata
│
├── output/                   # Algorithm results and analysis
│   ├── results/             # Raw embedding results
│   │   └── results_ALGORITHM_TIMESTAMP.json
│   └── metrics/             # Performance metrics and analysis
│       └── metrics_ALGORITHM_TIMESTAMP.csv
│
└── data_documentation.md               # This file
```

## File Format Specifications

### Substrate Network Files

#### Substrate Nodes (`*_nodes.csv`)
```csv
node_id,cpu_capacity,memory_capacity,cpu_used,memory_used,x_coord,y_coord,node_type
0,150,200,0.0,0.0,25.5,75.2,server
1,100,150,0.0,0.0,45.1,30.8,server
```

**Columns**:
- `node_id`: Unique integer identifier for the substrate node
- `cpu_capacity`: Total CPU capacity (arbitrary units)
- `memory_capacity`: Total memory capacity (arbitrary units) 
- `cpu_used`: Currently allocated CPU (managed by algorithms)
- `memory_used`: Currently allocated memory (managed by algorithms)
- `x_coord`, `y_coord`: Coordinates for visualization (optional)
- `node_type`: Node type classification (default: "server")

#### Substrate Links (`*_links.csv`)
```csv
src_node,dst_node,bandwidth_capacity,bandwidth_used,delay,cost,reliability
0,1,1000,0.0,5.5,1.0,0.99
1,2,500,0.0,2.1,1.0,0.95
```

**Columns**:
- `src_node`, `dst_node`: Connected substrate node IDs
- `bandwidth_capacity`: Total bandwidth capacity (arbitrary units)
- `bandwidth_used`: Currently allocated bandwidth (managed by algorithms)
- `delay`: Link propagation delay (milliseconds)
- `cost`: Link usage cost (arbitrary units, default: 1.0)
- `reliability`: Link reliability (0.0-1.0, default: 1.0)

### Virtual Network Request Files

#### VNR Metadata (`*_meta.csv`)
```csv
vnr_id,arrival_time,lifetime,priority,metadata
0,0.0,1000.0,1,"{}"
1,5.2,850.5,2,"{\"special\": true}"
```

**Columns**:
- `vnr_id`: Unique integer identifier for the VNR
- `arrival_time`: When the VNR arrives (simulation time units)
- `lifetime`: How long the VNR remains active (simulation time units)
- `priority`: VNR priority level (higher = more important)
- `metadata`: JSON string with additional VNR properties

#### VNR Node Requirements (`*_nodes.csv`)
```csv
vnr_id,node_id,cpu_requirement,memory_requirement,constraints
0,0,25,50,"{}"
0,1,30,60,"{}"
1,0,15,30,"{}"
```

**Columns**:
- `vnr_id`: VNR identifier (foreign key to metadata)
- `node_id`: Virtual node identifier within the VNR
- `cpu_requirement`: Required CPU capacity
- `memory_requirement`: Required memory capacity
- `constraints`: JSON string with additional node constraints

#### VNR Link Requirements (`*_links.csv`)
```csv
vnr_id,src_node,dst_node,bandwidth_requirement,delay_constraint,reliability_requirement,constraints
0,0,1,100,10.0,0.95,"{}"
1,0,1,75,5.0,0.99,"{}"
```

**Columns**:
- `vnr_id`: VNR identifier (foreign key to metadata)
- `src_node`, `dst_node`: Connected virtual node IDs within the VNR
- `bandwidth_requirement`: Required bandwidth capacity
- `delay_constraint`: Maximum acceptable delay (-1 for unlimited)
- `reliability_requirement`: Minimum required reliability (0.0-1.0)
- `constraints`: JSON string with additional link constraints

### Algorithm Results (`results_*.json`)

```json
[
  {
    "vnr_id": "0",
    "success": true,
    "node_mapping": {
      "0": "3",
      "1": "7"
    },
    "link_mapping": {
      "0-1": ["3", "2", "7"]
    },
    "revenue": 250.75,
    "cost": 45.50,
    "execution_time": 0.0023,
    "failure_reason": null,
    "timestamp": 1748379950.123,
    "algorithm_name": "Yu et al. (2008) Two-Stage Algorithm",
    "metadata": {
      "path_cache_hits": 2,
      "rollback_count": 0
    }
  }
]
```

**Fields**:
- `vnr_id`: Identifier of the processed VNR
- `success`: Whether embedding was successful
- `node_mapping`: Map of virtual node IDs to substrate node IDs
- `link_mapping`: Map of virtual links to substrate paths (as node sequences)
- `revenue`: Revenue generated from successful embedding
- `cost`: Cost incurred for embedding attempt
- `execution_time`: Algorithm execution time (seconds)
- `failure_reason`: Reason for failure (if unsuccessful)
- `timestamp`: Unix timestamp of embedding attempt
- `algorithm_name`: Name of algorithm used
- `metadata`: Algorithm-specific additional information

### Performance Metrics (`metrics_*.csv`)

```csv
metric,value
acceptance_ratio,0.85
total_revenue,1250.75
total_cost,450.25
revenue_to_cost_ratio,2.78
average_execution_time,0.0045
avg_node_cpu_util,0.67
avg_node_memory_util,0.72
avg_link_bandwidth_util,0.43
```

## Data Generation

### Generate Substrate Networks

```bash
# Basic substrate network
python main.py generate substrate --nodes 20 --save data/substrate_20.csv

# Erdős-Rényi random graph
python main.py generate substrate --nodes 50 --topology erdos_renyi --edge-prob 0.15 --save data/erdos_50.csv

# Barabási-Albert scale-free network
python main.py generate substrate --nodes 100 --topology barabasi_albert --attachment-count 3 --save data/barabasi_100.csv

# Custom resource ranges
python main.py generate substrate --nodes 30 \
  --cpu-range 50 200 --memory-range 100 300 --bandwidth-range 500 2000 \
  --save data/high_capacity_30.csv

# With random seed for reproducibility
python main.py generate substrate --nodes 25 --seed 42 --save data/reproducible_25.csv
```

### Generate VNR Batches

```bash
# Basic VNR batch
python main.py generate vnrs --count 50 --substrate data/substrate_20.csv --save data/vnrs_50.csv

# High-demand VNRs
python main.py generate vnrs --count 100 --substrate data/substrate_50.csv \
  --cpu-ratio 0.3 0.7 --memory-ratio 0.4 0.8 --bandwidth-ratio 0.2 0.6 \
  --save data/high_demand_vnrs.csv

# Different topologies
python main.py generate vnrs --count 30 --substrate data/substrate_20.csv \
  --topology star --nodes-range 3 8 --save data/star_vnrs.csv

python main.py generate vnrs --count 40 --substrate data/substrate_20.csv \
  --topology linear --nodes-range 2 6 --save data/linear_vnrs.csv

# Custom arrival patterns
python main.py generate vnrs --count 75 --substrate data/substrate_30.csv \
  --arrival-rate 15.0 --lifetime-mean 500.0 --save data/dense_arrival_vnrs.csv
```

## Data Analysis Examples

### Load and Analyze Results

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load embedding results
with open('data/output/results/results_yu2008_1748379950.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Basic statistics
print(f"Acceptance ratio: {df['success'].mean():.2%}")
print(f"Average revenue: {df[df['success']]['revenue'].mean():.2f}")
print(f"Average execution time: {df['execution_time'].mean():.4f}s")

# Plot revenue distribution
successful_results = df[df['success']]
plt.hist(successful_results['revenue'], bins=20)
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.title('Revenue Distribution for Successful Embeddings')
plt.show()
```

### Network Statistics

```python
import pandas as pd

# Load substrate network
nodes_df = pd.read_csv('data/substrate_50_nodes.csv')
links_df = pd.read_csv('data/substrate_50_links.csv')

# Network statistics
print(f"Network size: {len(nodes_df)} nodes, {len(links_df)} links")
print(f"Average degree: {2 * len(links_df) / len(nodes_df):.2f}")
print(f"Total CPU capacity: {nodes_df['cpu_capacity'].sum()}")
print(f"Total memory capacity: {nodes_df['memory_capacity'].sum()}")
print(f"Total bandwidth capacity: {links_df['bandwidth_capacity'].sum()}")
```

## Common Data Scenarios

### Small Test Networks
```bash
# Quick testing with small networks
python main.py generate substrate --nodes 5 --save data/test_small.csv
python main.py generate vnrs --count 3 --substrate data/test_small.csv --save data/test_vnrs.csv
```

### Realistic Evaluation Networks
```bash
# Medium-scale realistic networks
python main.py generate substrate --nodes 50 --topology barabasi_albert --save data/realistic_50.csv
python main.py generate vnrs --count 200 --substrate data/realistic_50.csv \
  --cpu-ratio 0.1 0.4 --memory-ratio 0.1 0.4 --save data/realistic_vnrs.csv
```

### Stress Testing Networks
```bash
# High-load scenario
python main.py generate substrate --nodes 30 --cpu-range 20 60 --memory-range 30 80 --save data/constrained.csv
python main.py generate vnrs --count 100 --substrate data/constrained.csv \
  --cpu-ratio 0.4 0.8 --memory-ratio 0.4 0.8 --save data/demanding_vnrs.csv
```

### Comparative Studies
```bash
# Generate multiple substrate topologies for comparison
python main.py generate substrate --nodes 40 --topology erdos_renyi --save data/compare_erdos.csv
python main.py generate substrate --nodes 40 --topology barabasi_albert --save data/compare_barabasi.csv

# Same VNR batch for both
python main.py generate vnrs --count 100 --substrate data/compare_erdos.csv --seed 123 --save data/compare_vnrs.csv
```

## Data Quality Guidelines

### Substrate Networks
- **Connectivity**: Ensure networks are connected for meaningful embedding
- **Resource Balance**: Balance CPU, memory, and bandwidth capacities
- **Topology Realism**: Use appropriate topologies for your research scenario
- **Scale Appropriateness**: Match network size to computational resources

### VNR Batches
- **Feasibility**: Ensure at least some VNRs can be embedded
- **Diversity**: Include various VNR sizes and resource requirements
- **Arrival Patterns**: Use realistic arrival distributions for online scenarios
- **Lifetime Variation**: Include both short and long-lived VNRs

### Result Validation
- **Consistency Checks**: Verify resource allocations don't exceed capacities
- **Mapping Validity**: Ensure all mapped nodes and links exist in substrate
- **Revenue/Cost Sanity**: Check that revenue and cost calculations are reasonable

## Storage Recommendations

### File Organization
```
data/
├── experiments/
│   ├── experiment_1/
│   │   ├── substrate_50_erdos.csv
│   │   ├── vnrs_100_*.csv
│   │   └── results_*/
│   └── experiment_2/
│       └── ...
├── benchmarks/
│   ├── standard_substrate_100.csv
│   └── standard_vnrs_500.csv
└── archive/
    └── old_experiments/
```

### Backup Strategy
- **Version Control**: Use git for code, not large data files
- **Compression**: Compress large result files
- **Documentation**: Keep detailed records of experimental parameters
- **Replication**: Store generation parameters for reproducibility

## Troubleshooting

### Common Issues

**File Not Found Errors**:
```bash
# Check file naming convention
ls data/*substrate*
# Expected: substrate_NAME_nodes.csv, substrate_NAME_links.csv
```

**Empty Results**:
- Check VNR feasibility with `--debug` flag
- Verify substrate network connectivity
- Ensure resource requirements are not too high

**Memory Issues with Large Networks**:
- Use batch processing for large VNR sets
- Consider network size limits based on available RAM
- Monitor resource usage during generation

**Inconsistent Results**:
- Set random seeds for reproducibility
- Check for concurrent access to shared resources
- Verify configuration consistency across runs

## Performance Notes

### File Size Estimates
- **Small networks** (10-20 nodes): ~1-5 KB per file
- **Medium networks** (50-100 nodes): ~10-50 KB per file  
- **Large networks** (200+ nodes): ~100+ KB per file
- **Result files**: ~1-10 MB for 100-1000 VNRs

### Generation Time
- **Substrate networks**: Nearly instantaneous for <1000 nodes
- **VNR batches**: Seconds for <1000 VNRs
- **Algorithm execution**: Varies by algorithm complexity and network size

This data directory serves as the foundation for all VNE experiments and analysis. Proper data management ensures reproducible research and meaningful comparisons between algorithms.
#