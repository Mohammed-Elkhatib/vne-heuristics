# Technical Documentation

This document provides detailed technical information about the VNE Heuristics framework architecture, implementation details, and development guidelines.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Data Models](#data-models)
- [Algorithm Framework](#algorithm-framework)
- [Configuration System](#configuration-system)
- [File I/O System](#file-io-system)
- [CLI Implementation](#cli-implementation)
- [Development Guidelines](#development-guidelines)

## Architecture Overview

The VNE Heuristics framework follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────────────────────────────────────────┐
│                  CLI Layer (main.py)                │
├─────────────────────────────────────────────────────┤
│              Algorithm Framework                    │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │  BaseAlgorithm  │ │    Concrete Algorithms      ││
│  │  (Abstract)     │ │  (Yu2008, Baseline1, ...)   ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                  Core Models                        │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │ SubstrateNetwork│ │ VirtualNetworkRequest       ││
│  │                 │ │                             ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                 Utility Layer                       │
│ ┌──────────┐┌─────────┐┌─────────┐┌────────────────┐│
│ │Generators││ Metrics ││I/O Utils││ Configuration  ││
│ │          ││         ││         ││                ││
│ └──────────┘└─────────┘└─────────┘└────────────────┘│
├─────────────────────────────────────────────────────┤
│             External Dependencies                   │
│        NetworkX │ NumPy │ PyYAML │ Python Std       │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Models (`src/models/`)

#### SubstrateNetwork (`substrate.py`)

**Purpose**: Represents the physical infrastructure network where VNRs are embedded.

**Key Features**:
- Thread-safe resource allocation/deallocation
- NetworkX integration for graph operations
- CSV-based persistence
- Comprehensive resource tracking

**Core Classes**:
```python
@dataclass
class NodeResources:
    cpu_capacity: float
    memory_capacity: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    x_coord: float = 0.0
    y_coord: float = 0.0
    node_type: str = "default"

@dataclass  
class LinkResources:
    bandwidth_capacity: float
    delay: float
    cost: float = 1.0
    bandwidth_used: float = 0.0
    reliability: float = 1.0

class SubstrateNetwork:
    # Main substrate network implementation
```

**Key Methods**:
- `add_node()` / `add_link()`: Network construction
- `allocate_node_resources()` / `deallocate_node_resources()`: Resource management
- `get_network_statistics()`: Performance monitoring
- `load_from_csv()` / `save_to_csv()`: Persistence

#### VirtualNetworkRequest (`virtual_request.py`)

**Purpose**: Represents virtual network requests that need to be embedded.

**Key Features**:
- NetworkX-based topology representation
- Resource requirement specification
- Arrival time and lifetime management
- Batch processing capabilities

**Core Classes**:
```python
@dataclass
class VirtualNodeRequirement:
    node_id: int
    cpu_requirement: float
    memory_requirement: float
    node_constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VirtualLinkRequirement:
    src_node: int
    dst_node: int
    bandwidth_requirement: float
    delay_constraint: float = float('inf')
    reliability_requirement: float = 0.0

class VirtualNetworkRequest:
    # Main VNR implementation

class VNRBatch:
    # Batch processing for multiple VNRs
```

### 2. Algorithm Framework (`src/algorithms/`)

#### BaseAlgorithm (`base_algorithm.py`)

**Purpose**: Abstract base class providing standardized algorithm interface.

**Key Features**:
- Consistent embedding interface
- Resource management integration
- Performance metrics collection
- Error handling and logging

**Core Architecture**:
```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def _embed_single_vnr(self, vnr, substrate) -> EmbeddingResult:
        # Core algorithm implementation
        
    def embed_vnr(self, vnr, substrate) -> EmbeddingResult:
        # Wrapper with resource management
        
    def embed_batch(self, vnrs, substrate) -> List[EmbeddingResult]:
        # Batch processing
        
    def embed_online(self, vnrs, substrate) -> List[EmbeddingResult]:
        # Online simulation with departures
```

**EmbeddingResult Structure**:
```python
@dataclass
class EmbeddingResult:
    vnr_id: str
    success: bool
    node_mapping: Dict[str, str]  # virtual_node -> substrate_node
    link_mapping: Dict[Tuple[str, str], List[str]]  # virtual_link -> path
    revenue: float
    cost: float
    execution_time: float
    failure_reason: Optional[str] = None
    timestamp: Optional[float] = None
    algorithm_name: Optional[str] = None
```

#### Yu et al. (2008) Algorithm (`baseline/yu_2008_algorithm.py`)

**Purpose**: Implementation of the foundational two-stage VNE algorithm.

**Algorithm Stages**:

1. **Node Mapping Stage**:
   ```python
   def _node_mapping_stage(self, vnr, substrate):
       # 1. Rank virtual nodes by resource requirements
       ranked_vnodes = self._rank_virtual_nodes(vnr)
       
       # 2. Greedily map to substrate nodes with most resources
       for vnode in ranked_vnodes:
           candidates = self._find_candidate_nodes(...)
           selected = self._select_best_candidate_node(candidates)
   ```

2. **Link Mapping Stage**:
   ```python
   def _link_mapping_stage(self, vnr, substrate, node_mapping):
       # 1. For each virtual link, find k-shortest paths
       for (vsrc, vdst), vlink in vnr.virtual_links.items():
           paths = self._find_candidate_paths(...)
           selected_path = self._select_best_candidate_path(paths)
   ```

**Advanced Features**:
- k-shortest path computation
- Multiple path selection strategies (shortest, bandwidth, delay, cost)
- Path caching for performance
- Comprehensive rollback mechanisms

### 3. Utilities (`src/utils/`)

#### Network Generators (`generators.py`)

**Purpose**: Generate synthetic substrate networks and VNR batches for experiments.

**Key Functions**:
- `generate_substrate_network()`: Creates substrate with various topologies
- `generate_vnr_batch()`: Creates batch of VNRs with arrival patterns
- `set_random_seed()`: Reproducible generation

**Supported Topologies**:
- Erdős-Rényi random graphs
- Barabási-Albert scale-free networks
- Grid topologies
- Custom topologies

#### Metrics Calculator (`metrics.py`)

**Purpose**: Comprehensive performance evaluation of embedding results.

**Core Metrics**:
```python
def calculate_acceptance_ratio(results) -> float
def calculate_total_revenue(results) -> float  
def calculate_total_cost(results) -> float
def calculate_utilization(substrate_network) -> Dict[str, float]
def generate_metrics_summary(results, substrate=None) -> Dict[str, Any]
```

#### File I/O System (`io_utils.py`)

**Purpose**: Robust file operations for all data formats.

**Key Features**:
- CSV-based network storage
- JSON result serialization
- Backup and compression utilities
- Error handling and validation

### 4. Configuration System (`config.py`)

**Purpose**: Centralized, hierarchical configuration management.

**Configuration Hierarchy** (highest to lowest precedence):
1. Command-line arguments
2. Environment variables (`VNE_*`)
3. Configuration files (YAML/JSON)
4. Default values

**Configuration Sections**:
```python
@dataclass
class VNEConfig:
    network_generation: NetworkGenerationConfig
    algorithm: AlgorithmConfig  
    file_paths: FilePathConfig
    logging: LoggingConfig
    experiment: ExperimentConfig
```

### 5. CLI Implementation (`main.py`)

**Purpose**: Professional command-line interface for all framework operations.

**Command Structure**:
```
main.py
├── generate
│   ├── substrate  # Generate substrate networks
│   └── vnrs       # Generate VNR batches
├── run
│   ├── --algorithm    # Run specific algorithm
│   └── --list-algorithms  # List available algorithms
├── metrics        # Calculate performance metrics
└── config         # Configuration management
```

**Key Classes**:
```python
class VNECommandLineInterface:
    def create_parser(self) -> argparse.ArgumentParser
    def _handle_generate_command(self, args) -> int
    def _handle_run_command(self, args) -> int
    def _handle_metrics_command(self, args) -> int
```

## Data Flow

### 1. Network Generation Flow
```
CLI Command → Configuration Loading → Network Generation → CSV Storage
```

### 2. Algorithm Execution Flow
```
CLI Command → Network Loading → Algorithm Initialization → 
Embedding Execution → Result Storage → Metrics Display
```

### 3. Resource Management Flow
```
Embedding Attempt → Resource Allocation Check → 
Success: Allocate Resources | Failure: Rollback → Result Recording
```

## Key Design Patterns

### 1. Template Method Pattern
- **BaseAlgorithm** defines the embedding workflow
- Concrete algorithms implement `_embed_single_vnr()`
- Common functionality (timing, logging, resource management) handled by base class

### 2. Strategy Pattern
- **Path selection strategies** in Yu algorithm
- **Network topologies** in generators
- **Configuration sources** in config system

### 3. Factory Pattern
- **Algorithm discovery** and instantiation
- **Network generation** based on topology type
- **File format handling** based on extension

### 4. Observer Pattern
- **Logging system** throughout the framework
- **Progress reporting** during long operations
- **Metrics collection** during algorithm execution

## Thread Safety

### Critical Sections
- **Resource allocation/deallocation** in SubstrateNetwork
- **Path cache access** in algorithms
- **Statistics updates** in BaseAlgorithm

### Synchronization Mechanisms
```python
# Resource allocation lock
with self._lock:
    substrate.allocate_node_resources(...)

# Statistics update lock  
with self._lock:
    self._stats['total_requests'] += 1
```

## Error Handling Strategy

### 1. Exception Hierarchy
```python
VNEError (Base)
├── VNRError
│   ├── VNRValidationError
│   └── VNRFileFormatError
├── SubstrateNetworkError
│   ├── ResourceAllocationError
│   └── FileFormatError
├── ConfigurationError
└── VNECLIError
```

### 2. Error Recovery
- **Graceful degradation**: Continue processing other VNRs if one fails
- **Resource rollback**: Automatic cleanup on embedding failure
- **User-friendly messages**: Clear error reporting in CLI

### 3. Logging Strategy
```python
# Debug: Algorithm decisions
self.logger.debug(f"Mapping virtual node {vnode_id} to substrate node {snode_id}")

# Info: Major operations
self.logger.info(f"Successfully embedded VNR {vnr_id}")

# Warning: Potential issues
self.logger.warning(f"VNR {vnr_id} may not be feasible: {issues}")

# Error: Failures
self.logger.error(f"Exception during embedding: {e}")
```

## Performance Considerations

### 1. Algorithm Optimization
- **Path caching**: Avoid recomputing shortest paths
- **Early termination**: Stop when embedding is impossible
- **Efficient data structures**: Use appropriate containers

### 2. Memory Management
- **Lazy loading**: Load data only when needed
- **Cache clearing**: Periodic cleanup of path cache
- **Generator patterns**: Stream processing for large datasets

### 3. I/O Optimization
- **Batch operations**: Minimize file system calls
- **Compression**: Optional result compression
- **Streaming**: Process large files without loading entirely

## Testing Strategy

### 1. Unit Tests
- **Model validation**: Test data model constraints
- **Algorithm correctness**: Verify embedding validity
- **Utility functions**: Test generators and metrics

### 2. Integration Tests
- **CLI commands**: End-to-end command testing
- **File I/O**: Round-trip data persistence
- **Configuration**: Multi-source config loading

### 3. Performance Tests
- **Scalability**: Large network handling
- **Memory usage**: Resource consumption monitoring
- **Execution time**: Algorithm performance benchmarks

## Future Extensions

### 1. Algorithm Framework
- **Multi-objective optimization**: Pareto-optimal solutions
- **Machine learning integration**: Learning-based algorithms
- **Reinforcement Learning algorithms**: Adaptive embedding strategies
- **Quantom Inspired algorithms**: Quantum-inspired heuristics for VNE
- **Distributed algorithms**: Multi-substrate embedding

### 2. Visualization
- **Network topology visualization**: Interactive graph display
- **Performance dashboards**: Real-time metrics
- **Algorithm animation**: Step-by-step embedding visualization

### 3. Advanced Features
- **REST API**: Web service interface
- **Database integration**: Persistent data storage
- **Cloud deployment**: Distributed execution

## Development Guidelines

### 1. Code Style
- **PEP 8 compliance**: Standard Python formatting
- **Type hints**: Complete type annotations
- **Docstrings**: Google-style documentation

### 2. Testing Requirements
- **Test coverage**: Minimum 80% coverage
- **Test isolation**: Independent test cases
- **Performance regression**: Benchmark tracking

### 3. Documentation Standards
- **API documentation**: Complete function documentation
- **Architecture diagrams**: Visual system overview
- **Usage examples**: Comprehensive examples

This technical documentation provides the foundation for understanding, extending, and maintaining the VNE Heuristics framework.