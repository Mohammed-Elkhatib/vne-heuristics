# Technical Documentation

This document provides detailed technical information about the VNE Heuristics framework architecture, implementation details, and development guidelines.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Data Models](#data-models)
- [Algorithm Framework](#algorithm-framework)
- [Configuration System](#configuration-system)
- [CLI Implementation](#cli-implementation)
- [Development Guidelines](#development-guidelines)

## Architecture Overview

The VNE Heuristics framework follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────────────────────────────────────────┐
│              CLI Layer (main.py)                    │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │ Argument Parser │ │    Command Implementations  ││
│  │                 │ │  (Generate, Run, Metrics)   ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                 Core Framework                      │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │ Algorithm       │ │    Error Handler &          ││
│  │ Registry        │ │    Progress Reporter        ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│              Algorithm Framework                    │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │  BaseAlgorithm  │ │    Concrete Algorithms      ││
│  │  (Abstract)     │ │     (YuAlgorithm)           ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                  Core Models                        │
│  ┌─────────────────┐ ┌─────────────────────────────┐│
│  │ SubstrateNetwork│ │ VirtualNetworkRequest       ││
│  │                 │ │ VNRBatch                    ││
│  └─────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│                 Utility Layer                       │
│ ┌──────────┐┌─────────┐┌─────────┐┌────────────────┐│
│ │Generators││ Metrics ││I/O Utils││ Configuration  ││
│ │          ││         ││         ││ (config_mgmt)  ││
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
- Thread-safe resource allocation/deallocation using `threading.Lock()`
- NetworkX integration for graph operations
- CSV-based persistence with robust I/O
- Comprehensive resource tracking and constraint management

**Core Classes**:
```python
@dataclass
class NodeResources:
    cpu_capacity: float
    memory_capacity: float = 0.0  # Always present, but may be ignored
    cpu_used: float = 0.0
    memory_used: float = 0.0
    x_coord: float = 0.0
    y_coord: float = 0.0
    node_type: str = "default"

@dataclass  
class LinkResources:
    bandwidth_capacity: float
    bandwidth_used: float = 0.0
    delay: float = 0.0           # Always present, but may be ignored
    cost: float = 1.0            # Always present, but may be ignored
    reliability: float = 1.0     # Always present, but may be ignored

class SubstrateNetwork:
    def __init__(self, directed: bool = False,
                 enable_memory_constraints: bool = False,
                 enable_delay_constraints: bool = False,
                 enable_cost_constraints: bool = False,
                 enable_reliability_constraints: bool = False):
```

**Constraint Management**:
- **Primary constraints**: CPU and Bandwidth (always enforced)
- **Secondary constraints**: Memory, delay, cost, reliability (optional)
- **Network-level configuration**: Constraints enabled/disabled for entire network

**Key Methods**:
- `add_node()` / `add_link()`: Network construction with constraint awareness
- `allocate_node_resources()` / `deallocate_node_resources()`: Thread-safe resource management
- `get_network_statistics()`: Performance monitoring with constraint filtering
- `load_from_csv()` / `save_to_csv()`: Persistence with proper error handling

#### VirtualNetworkRequest (`virtual_request.py`)

**Purpose**: Represents virtual network requests that need to be embedded.

**Key Features**:
- NetworkX-based topology representation
- Resource requirement specification with constraint support
- Arrival time and lifetime management for online simulation
- Validation and consistency checking

**Core Classes**:
```python
@dataclass
class VirtualNodeRequirement:
    node_id: int
    cpu_requirement: float
    memory_requirement: float = 0.0  # Default: no memory requirement
    node_constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VirtualLinkRequirement:
    src_node: int
    dst_node: int
    bandwidth_requirement: float
    delay_constraint: float = 0.0  # Default: no delay constraint
    reliability_requirement: float = 0.0  # Default: no reliability requirement
    link_constraints: Dict[str, Any] = field(default_factory=dict)

class VirtualNetworkRequest:
    def __init__(self, vnr_id: int, arrival_time: float = 0.0,
                 holding_time: float = float('inf'), priority: int = 1):
```

**Advanced Features**:
- **Constraint analysis**: `get_constraint_summary()` identifies which constraints are actually used
- **Resource calculation**: `calculate_total_requirements()` with selective constraint inclusion
- **Validation**: `validate_request()` with constraint-specific validation
- **JSON serialization**: Complete persistence with `to_dict()` / `from_dict()`

#### VNRBatch (`vnr_batch.py`)

**Purpose**: Manages collections of VNRs for batch processing and experiments.

**Key Features**:
- Batch operations (sorting, filtering, splitting)
- CSV-based persistence with multiple file format
- Basic information and statistics collection
- Integration with experiment workflows

### 2. Algorithm Framework (`src/algorithms/`)

#### BaseAlgorithm (`base_algorithm.py`)

**Purpose**: Abstract base class providing standardized algorithm interface following VNE literature standards.

**Key Features**:
- Consistent embedding interface with resource management
- Performance metrics collection using standard formulas
- Online simulation with VNR departures
- Comprehensive error handling and constraint validation

**Core Architecture**:
```python
class BaseAlgorithm(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._stats_lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_execution_time': 0.0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'constraint_violations': 0
        }
    
    @abstractmethod
    def _embed_single_vnr(self, vnr, substrate) -> EmbeddingResult:
        # Core algorithm implementation with resource allocation
        pass
        
    @abstractmethod  
    def _cleanup_failed_embedding(self, vnr, substrate, result) -> None:
        # Rollback mechanism for failed embeddings
        pass
        
    def embed_vnr(self, vnr, substrate) -> EmbeddingResult:
        # Standard workflow wrapper with constraint validation
        pass
        
    def embed_batch(self, vnrs, substrate) -> List[EmbeddingResult]:
        # Batch processing with progress tracking
        pass
        
    def embed_online(self, vnrs, substrate) -> List[EmbeddingResult]:
        # Online simulation with temporal constraints
        pass
```

**EmbeddingResult Structure**:
```python
@dataclass
class EmbeddingResult:
    vnr_id: str
    success: bool
    node_mapping: Dict[str, str]  # virtual_node_id -> substrate_node_id
    link_mapping: Dict[Tuple[str, str], List[str]]  # (v_src, v_dst) -> substrate_path
    revenue: float
    cost: float
    execution_time: float
    failure_reason: Optional[str] = None
    timestamp: Optional[float] = None
    algorithm_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

**VNE Constraint Validation**:
```python
def _validate_vne_constraints(self, vnr, substrate, result) -> List[str]:
    violations = []
    
    # CRITICAL: Intra-VNR separation constraint
    if not self._check_intra_vnr_separation(result.node_mapping):
        violations.append("Intra-VNR separation violated")
    
    # Additional constraint checks...
    return violations
```

#### YuAlgorithm (`baseline/yu_2008_algorithm.py`)

**Purpose**: Literature-compliant implementation of Yu et al. (2008) two-stage algorithm.

**Algorithm Architecture**:

```python
class YuAlgorithm(BaseAlgorithm):
    def __init__(self, k_paths: int = 1,
                 path_selection_strategy: str = "shortest",
                 enable_path_caching: bool = True, **kwargs):
        super().__init__("Yu et al. (2008) Two-Stage Algorithm", **kwargs)
        self.k_paths = max(1, k_paths)
        self.path_selection_strategy = path_selection_strategy
        self._path_cache: Dict[Tuple[int, int], List[PathInfo]] = {}
```

**Stage 1 - Node Mapping**:
```python
def _node_mapping_stage_yu2008(self, vnr, substrate):
    # 1. Rank virtual nodes by CPU requirements (Yu 2008 standard)
    ranked_vnodes = self._rank_virtual_nodes_yu2008(vnr)
    
    # 2. Greedily map with immediate resource allocation
    for vnode_info in ranked_vnodes:
        candidate_node = self._find_best_substrate_node_yu2008(...)
        success = substrate.allocate_node_resources(...)
        # Track allocations for rollback
```

**Stage 2 - Link Mapping**:
```python
def _link_mapping_stage_yu2008(self, vnr, substrate, node_mapping):
    # 1. Find k-shortest paths with bandwidth constraints
    for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
        candidate_paths = self._find_k_shortest_paths_yu2008(...)
        selected_path = self._select_best_path_yu2008(candidate_paths)
        # Allocate bandwidth with rollback support
```

**Advanced Features**:
- **k-shortest path computation** using NetworkX
- **Multiple path selection strategies**: shortest, bandwidth
- **Path caching** for performance optimization
- **Comprehensive rollback** mechanisms for resource cleanup
- **Literature compliance validation** with constraint warnings

### 3. Generators (`src/utils/generators/`)

#### Generation Configuration (`generation_config.py`)

**Purpose**: Centralized configuration for network and VNR generation.

```python
@dataclass
class NetworkGenerationConfig:
    # Substrate parameters
    substrate_nodes: int = 100
    substrate_topology: str = "erdos_renyi"
    substrate_edge_probability: float = 0.1
    
    # Constraint configuration
    enable_memory_constraints: bool = False
    enable_delay_constraints: bool = False
    enable_cost_constraints: bool = False
    enable_reliability_constraints: bool = False
    
    # Resource parameters
    cpu_range: Tuple[int, int] = (50, 100)
    bandwidth_range: Tuple[int, int] = (50, 100)
    # ... additional ranges
    
    # VNR parameters
    vnr_nodes_range: Tuple[int, int] = (2, 10)
    vnr_topology: str = "random"
    arrival_rate: float = 10.0
```

#### Substrate Generators (`substrate_generators.py`)

**Purpose**: Generate substrate networks with various topologies and constraint configurations.

**Key Functions**:
```python
def generate_substrate_network(nodes: int, 
                             topology: str = "erdos_renyi",
                             edge_probability: float = 0.1,
                             enable_memory_constraints: bool = False,
                             enable_delay_constraints: bool = False,
                             # ... additional parameters
                             ) -> SubstrateNetwork:

def generate_realistic_substrate_network(nodes: int,
                                       geographic_area: str = "metro",
                                       # ... parameters
                                       ) -> SubstrateNetwork:

def create_predefined_scenarios() -> Dict[str, NetworkGenerationConfig]:
    # Returns pre-configured scenarios for research
```

**Supported Topologies**:
- **Erdős-Rényi**: Random graphs with configurable edge probability
- **Barabási-Albert**: Scale-free networks with preferential attachment
- **Grid**: Regular grid topologies
- **Realistic**: Geographic area-based realistic networks

#### VNR Generators (`vnr_generators.py`)

**Purpose**: Generate VNR batches with configurable topologies and temporal patterns.

**Key Functions**:
```python
def generate_vnr(substrate_nodes: List[str],
                vnr_nodes_count: Optional[int] = None,
                topology: str = "random",
                enable_memory_constraints: bool = False,
                # ... constraint parameters
                ) -> VirtualNetworkRequest:

def generate_vnr_batch(count: int,
                      substrate_nodes: List[str],
                      config: Optional[NetworkGenerationConfig] = None,
                      ) -> VNRBatch:

def generate_arrival_times(count: int,
                          pattern: str = "poisson",
                          rate: float = 10.0) -> List[float]:
```

### 4. Metrics System (`src/utils/metrics.py`)

**Purpose**: Comprehensive performance evaluation using standard VNE literature formulas.

**Core Metrics Functions**:
```python
# Standard VNE Revenue Calculation
def calculate_vnr_revenue(vnr: VirtualNetworkRequest) -> float:
    # Revenue = Σ(CPU_requirements) + Σ(Bandwidth_requirements)
    # + Σ(Memory_requirements) if memory constraints used

# Standard VNE Cost Calculation  
def calculate_vnr_cost(vnr: VirtualNetworkRequest,
                      node_mapping: Dict, link_mapping: Dict,
                      substrate_network: Optional[SubstrateNetwork]) -> float:
    # Cost = Σ(Node_CPU_allocated) + Σ(Link_bandwidth_allocated × path_length)
    # + Σ(Memory_allocated) if substrate enables memory constraints

# Primary VNE Metrics
def calculate_acceptance_ratio(results: List[EmbeddingResult]) -> float
def calculate_total_revenue(results: List[EmbeddingResult]) -> float
def calculate_total_cost(results: List[EmbeddingResult]) -> float
def calculate_revenue_to_cost_ratio(results: List[EmbeddingResult]) -> float

# Resource Utilization
def calculate_utilization(substrate_network: SubstrateNetwork) -> Dict[str, float]

# Comprehensive Analysis
def generate_comprehensive_metrics_summary(results: List[EmbeddingResult],
                                          substrate_network: Optional[SubstrateNetwork] = None,
                                          time_duration: Optional[float] = None) -> Dict[str, Any]
```

### 5. Configuration System (`config_management.py`)

**Purpose**: Centralized, hierarchical configuration management with enhanced error handling.

**Configuration Hierarchy** (highest to lowest precedence):
1. Command-line arguments
2. Environment variables (`VNE_*`)
3. Configuration files (YAML/JSON)
4. Default values

**Configuration Structure**:
```python
@dataclass
class VNEConfig:
    network_generation: NetworkGenerationConfig
    algorithm: AlgorithmConfig  
    file_paths: FilePathConfig
    logging: LoggingConfig
    experiment: ExperimentConfig
    
    # Global settings
    version: str = "1.0.0"
    debug_mode: bool = False
    verbose: bool = False

class ConfigurationManager:
    def load_config(self, config_file: Optional[Path] = None,
                   env_prefix: str = "VNE_",
                   **overrides) -> VNEConfig:
        # Multi-source configuration loading with validation
```

### 6. CLI Implementation (`main.py` and `cli/`)

**Purpose**: Professional command-line interface using the Command pattern.

**Main CLI Class**:
```python
class VNECommandLineInterface:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.algorithm_registry = AlgorithmRegistry()
        self.progress_reporter = ProgressReporter()
        self.error_handler = ErrorHandler()
        
        # Command registry using Command pattern
        self.commands = {
            'generate': GenerateCommand(self),
            'run': RunCommand(self),
            'metrics': MetricsCommand(self),
            'config': ConfigCommand(self)
        }
```

**Command Implementations**:
- **GenerateCommand**: Network and VNR generation using generator modules
- **RunCommand**: Algorithm execution with real algorithm discovery
- **MetricsCommand**: Comprehensive metrics calculation using metrics module
- **ConfigCommand**: Configuration file management

**Algorithm Registry**:
```python
class AlgorithmRegistry:
    def _discover_algorithms(self) -> None:
        # Automatic discovery from src/algorithms/ packages
        # Fallback discovery for known algorithms
        # Error tracking and dummy algorithm registration
    
    def get_algorithm(self, name: str) -> Optional[Type]:
        # Case-insensitive algorithm lookup
```

## Data Flow

### 1. Network Generation Flow
```
CLI Command → Configuration Loading → Generator Module Selection → 
Network Generation → CSV Storage (paired files)
```

### 2. Algorithm Execution Flow
```
CLI Command → Network Loading (CSV parsing) → Algorithm Registry Lookup → 
Algorithm Initialization → Embedding Execution → Result Storage (JSON) → 
Metrics Display
```

### 3. Resource Management Flow
```
Embedding Attempt → Pre-allocation Check → Algorithm Execution → 
Success: Resource Allocation | Failure: Rollback → Post-validation → 
Result Recording
```

### 4. Constraint Handling Flow
```
Network Creation → Constraint Configuration → VNR Generation → 
Constraint Validation → Algorithm Execution → Constraint Compliance Check
```

## Key Design Patterns

### 1. Template Method Pattern
- **BaseAlgorithm** defines the standard VNE workflow
- Concrete algorithms implement `_embed_single_vnr()` and `_cleanup_failed_embedding()`
- Common functionality handled by base class (timing, logging, constraint validation)

### 2. Strategy Pattern
- **Path selection strategies** in YuAlgorithm
- **Network topologies** in generators
- **Configuration sources** in config system
- **Constraint handling** strategies

### 3. Command Pattern
- **CLI commands** with execute() interface
- **Modular command system** for easy extension
- **Error handling** and **progress reporting** integration

### 4. Factory Pattern
- **Algorithm discovery** and instantiation via registry
- **Network generation** based on topology type
- **Generator selection** based on configuration

### 5. Observer Pattern
- **Logging system** throughout the framework
- **Progress reporting** during long operations
- **Metrics collection** during algorithm execution

## Thread Safety

### Critical Sections
- **Resource allocation/deallocation** in SubstrateNetwork using `self._lock`
- **Path cache access** in algorithms
- **Statistics updates** in BaseAlgorithm using `self._stats_lock`

### Synchronization Mechanisms
```python
# Resource allocation lock
with self._lock:
    substrate.allocate_node_resources(...)

# Statistics update lock  
with self._stats_lock:
    self._stats['total_requests'] += 1
```

## Error Handling Strategy

### 1. Exception Hierarchy
```python
# CLI Exceptions
VNECLIError (Base)
├── CommandError
├── ValidationError
├── FileError
└── AlgorithmError

# Model Exceptions  
VNRError
├── VNRValidationError
└── VNRFileFormatError

SubstrateNetworkError
├── ResourceAllocationError
└── FileFormatError

# Configuration Exceptions
ConfigurationError
├── FileHandlingError
├── ValidationError
└── EnvironmentError
```

### 2. Error Recovery
- **Graceful degradation**: Continue processing other VNRs if one fails
- **Resource rollback**: Automatic cleanup via `_cleanup_failed_embedding()`
- **User-friendly messages**: Clear error reporting with contextual help

### 3. Enhanced Error Handler
```python
class ErrorHandler:
    def handle_algorithm_error(self, error, algorithm_name=None, details=None):
        # Algorithm-specific guidance and troubleshooting
        
    def handle_file_error(self, error, filepath=None):
        # File analysis and similar file suggestions
        
    def handle_configuration_error(self, error):
        # Configuration validation and examples
```

## Performance Considerations

### 1. Algorithm Optimization
- **Path caching**: Avoid recomputing shortest paths in YuAlgorithm
- **Early termination**: Stop when embedding is clearly impossible
- **Efficient data structures**: NetworkX graphs with proper indexing

### 2. Memory Management
- **Lazy loading**: VNRBatch loads data only when needed
- **Cache clearing**: `clear_path_cache()` in algorithms
- **Generator patterns**: Stream processing for large datasets

### 3. I/O Optimization
- **Batch operations**: CSV files minimize file system calls
- **UTF-8 encoding**: Consistent encoding across all file operations
- **Path handling**: Pathlib for robust file operations

## Configuration Management

### 1. Multi-Source Configuration
```python
# Configuration precedence (highest to lowest)
1. Command-line arguments (--verbose, --debug)
2. Environment variables (VNE_SUBSTRATE_NODES=50)
3. Configuration files (config.yaml)
4. Default values (built into dataclasses)
```

### 2. Constraint Configuration
```python
# Network-level constraint configuration
substrate = SubstrateNetwork(
    enable_memory_constraints=True,
    enable_delay_constraints=True,
    enable_cost_constraints=False,
    enable_reliability_constraints=False
)

# VNR automatically adapts to substrate constraints
vnr = generate_vnr_from_config(substrate_nodes, config, substrate_network)
```

### 3. Environment Integration
```python
# Environment variable mapping
VNE_SUBSTRATE_NODES=50
VNE_ALGORITHM_TIMEOUT=300.0
VNE_DATA_DIR="custom_data"
VNE_DEBUG=true
```

## Future Extensions

### 1. Algorithm Framework
- **Multi-objective optimization**: Pareto-optimal solutions
- **Machine learning integration**: Learning-based algorithms
- **Reinforcement learning**: Adaptive embedding strategies
- **Distributed algorithms**: Multi-substrate embedding

### 2. Advanced Features
- **REST API**: Web service interface using FastAPI
- **Database integration**: PostgreSQL/MongoDB for persistent storage
- **Visualization**: Interactive network and performance visualization
- **Cloud deployment**: Kubernetes-based distributed execution

### 3. Research Extensions
- **Security constraints**: Trust and isolation requirements
- **Energy optimization**: Power-aware embedding
- **Fault tolerance**: Resilient embedding strategies
- **Network slicing**: 5G/NFV integration

## Development Guidelines

### 1. Code Style
- **PEP 8 compliance**: Using black formatter
- **Type hints**: Complete type annotations throughout
- **Docstrings**: Google-style documentation with examples

### 2. Testing Strategy
```python
# Unit tests for core functionality
tests/test_models/test_substrate.py
tests/test_algorithms/test_yu_algorithm.py

# Integration tests for CLI
tests/test_cli/test_commands.py

# Performance tests
tests/test_performance/test_scalability.py
```

### 3. Documentation Standards
- **API documentation**: Complete function and class documentation
- **Architecture diagrams**: Visual system overview
- **Usage examples**: Comprehensive examples with expected output

This technical documentation provides the foundation for understanding, extending, and maintaining the VNE Heuristics framework, accurately reflecting the current implementation.