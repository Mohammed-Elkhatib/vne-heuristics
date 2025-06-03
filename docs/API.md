# VNE Heuristics Framework API Reference

This document provides comprehensive API documentation for the VNE Heuristics framework, designed for developers who want to integrate with or extend the framework.

## Table of Contents

- [Overview](#overview)
- [Core Models](#core-models)
  - [SubstrateNetwork](#substratenetwork)
  - [VirtualNetworkRequest](#virtualnetworkrequest)
- [Algorithm Framework](#algorithm-framework)
  - [BaseAlgorithm](#basealgorithm)
  - [EmbeddingResult](#embeddingresult)
  - [YuAlgorithm](#yualgorithm)
- [Metrics and Analysis](#metrics-and-analysis)
- [Common Use Cases](#common-use-cases)
- [Error Handling](#error-handling)

## Overview

The VNE Heuristics framework provides a complete toolkit for Virtual Network Embedding research and development. The API is designed around several core principles:

- **Thread Safety**: All resource operations are thread-safe
- **Constraint Flexibility**: Support for both primary (CPU, bandwidth) and optional secondary constraints
- **Standard Compliance**: Follows VNE literature standards for metrics and algorithms
- **Extensibility**: Clean inheritance patterns for adding new algorithms

### Import Structure

```python
# Core models
from src.models.substrate import SubstrateNetwork, NodeResources, LinkResources
from src.models.virtual_request import VirtualNetworkRequest, VirtualNodeRequirement, VirtualLinkRequirement

# Algorithm framework  
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm

# Metrics and analysis
from src.utils.metrics import (
    calculate_vnr_revenue, calculate_vnr_cost, calculate_acceptance_ratio,
    generate_comprehensive_metrics_summary, EmbeddingResult as MetricsEmbeddingResult
)
```

---

## Core Models

### SubstrateNetwork

**Module**: `src.models.substrate`

Represents the physical infrastructure network where Virtual Network Requests (VNRs) are embedded. Provides thread-safe resource management and constraint-aware operations.

#### Class Definition

```python
class SubstrateNetwork:
    def __init__(self, 
                 directed: bool = False,
                 enable_memory_constraints: bool = False,
                 enable_delay_constraints: bool = False,
                 enable_cost_constraints: bool = False,
                 enable_reliability_constraints: bool = False)
```

**Parameters**:
- `directed` (bool): Whether to use directed graph topology. Default: `False`
- `enable_memory_constraints` (bool): Enable memory resource constraints. Default: `False`  
- `enable_delay_constraints` (bool): Enable delay constraints for links. Default: `False`
- `enable_cost_constraints` (bool): Enable cost constraints for links. Default: `False`
- `enable_reliability_constraints` (bool): Enable reliability constraints for links. Default: `False`

**Example**:
```python
# Create substrate with only primary constraints (CPU + Bandwidth)
substrate = SubstrateNetwork()

# Create substrate with all constraints enabled
substrate_full = SubstrateNetwork(
    enable_memory_constraints=True,
    enable_delay_constraints=True,
    enable_cost_constraints=True,
    enable_reliability_constraints=True
)
```

#### Network Construction

##### add_node()

```python
def add_node(self, 
             node_id: int, 
             cpu_capacity: float, 
             memory_capacity: float = 0.0,
             x_coord: float = 0.0, 
             y_coord: float = 0.0,
             node_type: str = "default") -> None
```

Add a substrate node with specified resources.

**Parameters**:
- `node_id` (int): Unique identifier for the node
- `cpu_capacity` (float): CPU capacity (must be ≥ 0)
- `memory_capacity` (float): Memory capacity (ignored if memory constraints disabled)
- `x_coord`, `y_coord` (float): Coordinates for visualization
- `node_type` (str): Node type classification

**Raises**:
- `ValueError`: If capacity values are negative

**Example**:
```python
substrate = SubstrateNetwork()

# Add nodes with different capacities
substrate.add_node(0, cpu_capacity=100, memory_capacity=200, x_coord=10.0, y_coord=20.0)
substrate.add_node(1, cpu_capacity=150, memory_capacity=300, x_coord=30.0, y_coord=40.0)
substrate.add_node(2, cpu_capacity=80, memory_capacity=160, x_coord=50.0, y_coord=10.0)
```

##### add_link()

```python
def add_link(self, 
             src: int, 
             dst: int, 
             bandwidth_capacity: float,
             delay: float = 0.0, 
             cost: float = 1.0, 
             reliability: float = 1.0) -> None
```

Add a substrate link with specified resources and attributes.

**Parameters**:
- `src`, `dst` (int): Source and destination node IDs
- `bandwidth_capacity` (float): Bandwidth capacity (must be ≥ 0)
- `delay` (float): Link delay (ignored if delay constraints disabled)
- `cost` (float): Link usage cost (ignored if cost constraints disabled)
- `reliability` (float): Link reliability 0.0-1.0 (ignored if reliability constraints disabled)

**Example**:
```python
# Add links between nodes
substrate.add_link(0, 1, bandwidth_capacity=1000, delay=5.0, cost=2.0, reliability=0.99)
substrate.add_link(1, 2, bandwidth_capacity=500, delay=3.0, cost=1.5, reliability=0.95)
substrate.add_link(0, 2, bandwidth_capacity=750, delay=8.0, cost=3.0, reliability=0.98)
```

#### Resource Management

##### allocate_node_resources()

```python
def allocate_node_resources(self, 
                           node_id: int, 
                           cpu: float, 
                           memory: float = 0.0) -> bool
```

Allocate resources from a substrate node (thread-safe).

**Parameters**:
- `node_id` (int): ID of the node to allocate from
- `cpu` (float): Amount of CPU to allocate
- `memory` (float): Amount of memory to allocate (ignored if memory constraints disabled)

**Returns**:
- `bool`: `True` if allocation successful, `False` if insufficient resources

**Raises**:
- `ResourceAllocationError`: If node doesn't exist or parameters are invalid

**Example**:
```python
# Allocate resources for VNR embedding
success = substrate.allocate_node_resources(node_id=0, cpu=50.0, memory=100.0)
if success:
    print("Resources allocated successfully")
else:
    print("Insufficient resources available")
```

##### deallocate_node_resources()

```python
def deallocate_node_resources(self, 
                             node_id: int, 
                             cpu: float, 
                             memory: float = 0.0) -> bool
```

Deallocate resources from a substrate node (thread-safe).

**Parameters**: Same as `allocate_node_resources()`

**Returns**: `bool` indicating success

**Example**:
```python
# Clean up after VNR departure  
substrate.deallocate_node_resources(node_id=0, cpu=50.0, memory=100.0)
```

##### allocate_link_resources() / deallocate_link_resources()

```python
def allocate_link_resources(self, src: int, dst: int, bandwidth: float) -> bool
def deallocate_link_resources(self, src: int, dst: int, bandwidth: float) -> bool
```

Allocate/deallocate bandwidth from substrate links (thread-safe).

#### Resource Inspection

##### get_node_resources()

```python
def get_node_resources(self, node_id: int) -> Optional[NodeResources]
```

Get resource information for a specific node.

**Returns**: `NodeResources` object or `None` if node doesn't exist

**Example**:
```python
resources = substrate.get_node_resources(0)
if resources:
    print(f"CPU: {resources.cpu_available}/{resources.cpu_capacity}")
    print(f"Memory: {resources.memory_available}/{resources.memory_capacity}")
    print(f"Utilization: {resources.cpu_used/resources.cpu_capacity:.2%}")
```

##### get_network_statistics()

```python
def get_network_statistics(self) -> Dict[str, Any]
```

Get comprehensive network statistics and resource utilization.

**Returns**: Dictionary containing:
- `node_count`, `link_count`: Network size
- `is_connected`: Connectivity status
- `total_cpu`, `used_cpu`, `available_cpu`: CPU statistics
- `total_bandwidth`, `used_bandwidth`, `available_bandwidth`: Bandwidth statistics
- `cpu_utilization`, `bandwidth_utilization`: Utilization ratios
- `constraint_configuration`: Which constraints are enabled

**Example**:
```python
stats = substrate.get_network_statistics()
print(f"Network: {stats['node_count']} nodes, {stats['link_count']} links")
print(f"CPU utilization: {stats['cpu_utilization']:.2%}")
print(f"Bandwidth utilization: {stats['bandwidth_utilization']:.2%}")
print(f"Connected: {stats['is_connected']}")
```

#### Persistence

##### save_to_csv() / load_from_csv()

```python
def save_to_csv(self, nodes_file: str, links_file: str) -> None
def load_from_csv(self, nodes_file: str, links_file: str) -> None
```

Save/load substrate network to/from CSV files.

**Example**:
```python
# Save substrate network
substrate.save_to_csv("substrate_nodes.csv", "substrate_links.csv")

# Load substrate network
new_substrate = SubstrateNetwork()
new_substrate.load_from_csv("substrate_nodes.csv", "substrate_links.csv")
```

#### Data Classes

##### NodeResources

```python
@dataclass
class NodeResources:
    cpu_capacity: float
    memory_capacity: float = 0.0
    cpu_used: float = 0.0
    memory_used: float = 0.0
    x_coord: float = 0.0
    y_coord: float = 0.0
    node_type: str = "default"
    
    @property
    def cpu_available(self) -> float
    @property  
    def memory_available(self) -> float
```

##### LinkResources

```python
@dataclass
class LinkResources:
    bandwidth_capacity: float
    bandwidth_used: float = 0.0
    delay: float = 0.0
    cost: float = 1.0
    reliability: float = 1.0
    
    @property
    def bandwidth_available(self) -> float
    @property
    def utilization(self) -> float
```

---

### VirtualNetworkRequest

**Module**: `src.models.virtual_request`

Represents a Virtual Network Request that needs to be embedded onto the substrate network.

#### Class Definition

```python
class VirtualNetworkRequest:
    def __init__(self, 
                 vnr_id: int, 
                 arrival_time: float = 0.0,
                 holding_time: float = float('inf'), 
                 priority: int = 1)
```

**Parameters**:
- `vnr_id` (int): Unique identifier for the VNR
- `arrival_time` (float): When VNR arrives in simulation
- `holding_time` (float): How long VNR remains active (use `float('inf')` for infinite)
- `priority` (int): Priority level (higher = more important)

**Example**:
```python
# Create VNR with finite lifetime
vnr = VirtualNetworkRequest(
    vnr_id=1, 
    arrival_time=10.0, 
    holding_time=1000.0, 
    priority=2
)

# Create VNR with infinite lifetime
persistent_vnr = VirtualNetworkRequest(vnr_id=2, holding_time=float('inf'))
```

#### Virtual Network Construction

##### add_virtual_node()

```python
def add_virtual_node(self, 
                    node_id: int, 
                    cpu_requirement: float,
                    memory_requirement: float = 0.0,
                    node_constraints: Optional[Dict[str, Any]] = None) -> None
```

Add a virtual node with resource requirements.

**Parameters**:
- `node_id` (int): Unique identifier within the VNR
- `cpu_requirement` (float): Required CPU resources (must be > 0)
- `memory_requirement` (float): Required memory resources (0.0 = no requirement)
- `node_constraints` (dict): Additional constraints (optional)

**Example**:
```python
# Add virtual nodes with different requirements
vnr.add_virtual_node(0, cpu_requirement=50.0, memory_requirement=100.0)
vnr.add_virtual_node(1, cpu_requirement=30.0, memory_requirement=80.0)
vnr.add_virtual_node(2, cpu_requirement=40.0)  # No memory requirement
```

##### add_virtual_link()

```python
def add_virtual_link(self, 
                    src_node: int, 
                    dst_node: int,
                    bandwidth_requirement: float, 
                    delay_constraint: float = 0.0,
                    reliability_requirement: float = 0.0,
                    link_constraints: Optional[Dict[str, Any]] = None) -> None
```

Add a virtual link with requirements and constraints.

**Parameters**:
- `src_node`, `dst_node` (int): Connected virtual node IDs
- `bandwidth_requirement` (float): Required bandwidth (must be > 0)
- `delay_constraint` (float): Maximum acceptable delay (0.0 = no constraint)
- `reliability_requirement` (float): Minimum reliability 0.0-1.0 (0.0 = no requirement)
- `link_constraints` (dict): Additional constraints (optional)

**Example**:
```python
# Add virtual links with different constraints
vnr.add_virtual_link(0, 1, bandwidth_requirement=100.0, delay_constraint=10.0, reliability_requirement=0.95)
vnr.add_virtual_link(1, 2, bandwidth_requirement=150.0)  # No delay/reliability constraints
```

#### Analysis and Validation

##### calculate_total_requirements()

```python
def calculate_total_requirements(self, 
                               include_memory: bool = True,
                               include_delay: bool = True,
                               include_reliability: bool = True) -> Dict[str, float]
```

Calculate total resource requirements for the VNR.

**Returns**: Dictionary with requirements:
- `total_cpu`, `total_bandwidth`: Always included
- `total_memory`: If `include_memory=True` and memory constraints used
- `delay_constrained_links`, `total_delay_constraint`: If `include_delay=True`
- `reliability_constrained_links`, `avg_reliability_requirement`: If `include_reliability=True`

**Example**:
```python
requirements = vnr.calculate_total_requirements()
print(f"Total CPU: {requirements['total_cpu']}")
print(f"Total bandwidth: {requirements['total_bandwidth']}")
print(f"Links with delay constraints: {requirements.get('delay_constrained_links', 0)}")
```

##### get_constraint_summary()

```python
def get_constraint_summary(self) -> Dict[str, Any]
```

Get summary of which constraints are actually used in this VNR.

**Returns**: Dictionary indicating constraint usage:
- `uses_memory_constraints`, `uses_delay_constraints`, `uses_reliability_constraints`
- Counts of nodes/links using each constraint type

**Example**:
```python
summary = vnr.get_constraint_summary()
if summary['uses_memory_constraints']:
    print(f"VNR uses memory: {summary['memory_constrained_nodes']} nodes")
if summary['uses_delay_constraints']:
    print(f"VNR uses delay: {summary['delay_constrained_links']} links")
```

##### validate_request()

```python
def validate_request(self, 
                    validate_memory: bool = True,
                    validate_delay: bool = True,
                    validate_reliability: bool = True) -> List[str]
```

Validate VNR for consistency and completeness.

**Returns**: List of validation error messages (empty if valid)

**Example**:
```python
issues = vnr.validate_request()
if issues:
    print("VNR validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("VNR is valid")
```

#### Persistence

##### to_dict() / from_dict()

```python
def to_dict(self) -> Dict[str, Any]

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'VirtualNetworkRequest'
```

Convert VNR to/from dictionary representation for serialization.

**Example**:
```python
# Serialize VNR
vnr_dict = vnr.to_dict()

# Deserialize VNR
restored_vnr = VirtualNetworkRequest.from_dict(vnr_dict)
```

##### save_to_json() / load_from_json()

```python
def save_to_json(self, filepath: str) -> None

@classmethod  
def load_from_json(cls, filepath: str) -> 'VirtualNetworkRequest'
```

**Example**:
```python
# Save VNR to file
vnr.save_to_json("vnr_001.json")

# Load VNR from file
loaded_vnr = VirtualNetworkRequest.load_from_json("vnr_001.json")
```

#### Data Classes

##### VirtualNodeRequirement

```python
@dataclass
class VirtualNodeRequirement:
    node_id: int
    cpu_requirement: float
    memory_requirement: float = 0.0
    node_constraints: Dict[str, Any] = field(default_factory=dict)
```

##### VirtualLinkRequirement  

```python
@dataclass
class VirtualLinkRequirement:
    src_node: int
    dst_node: int
    bandwidth_requirement: float
    delay_constraint: float = 0.0
    reliability_requirement: float = 0.0
    link_constraints: Dict[str, Any] = field(default_factory=dict)
```

---

## Algorithm Framework

### BaseAlgorithm

**Module**: `src.algorithms.base_algorithm`

Abstract base class for all VNE algorithms, providing standardized interface and common functionality.

#### Class Definition

```python
class BaseAlgorithm(ABC):
    def __init__(self, name: str, **kwargs)
```

**Parameters**:
- `name` (str): Human-readable algorithm name
- `**kwargs`: Algorithm-specific parameters

**Example**:
```python
# Subclass implementation
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, custom_param: float = 1.0):
        super().__init__("My Custom Algorithm", custom_param=custom_param)
        self.custom_param = custom_param
    
    def _embed_single_vnr(self, vnr, substrate):
        # Implementation
        pass
        
    def _cleanup_failed_embedding(self, vnr, substrate, result):
        # Cleanup implementation
        pass
```

#### Core Embedding Interface

##### embed_vnr()

```python
def embed_vnr(self, 
              vnr: VirtualNetworkRequest,
              substrate: SubstrateNetwork) -> EmbeddingResult
```

Embed a single VNR with complete workflow including validation and metrics.

**Parameters**:
- `vnr`: Virtual network request to embed
- `substrate`: Substrate network to embed onto

**Returns**: `EmbeddingResult` with complete embedding outcome

**Example**:
```python
algorithm = YuAlgorithm()
result = algorithm.embed_vnr(vnr, substrate)

if result.success:
    print(f"VNR {result.vnr_id} embedded successfully")
    print(f"Revenue: {result.revenue:.2f}, Cost: {result.cost:.2f}")
    print(f"Node mapping: {result.node_mapping}")
else:
    print(f"Embedding failed: {result.failure_reason}")
```

##### embed_batch()

```python
def embed_batch(self, 
                vnrs: List[VirtualNetworkRequest],
                substrate: SubstrateNetwork) -> List[EmbeddingResult]
```

Embed a batch of VNRs sequentially for statistical analysis.

**Example**:
```python
# Process multiple VNRs
vnr_list = [vnr1, vnr2, vnr3, vnr4, vnr5]
results = algorithm.embed_batch(vnr_list, substrate)

# Analyze results
successful = [r for r in results if r.success]
print(f"Acceptance ratio: {len(successful)/len(results):.2%}")
```

##### embed_online()

```python
def embed_online(self, 
                 vnrs: List[VirtualNetworkRequest],
                 substrate: SubstrateNetwork,
                 simulation_duration: Optional[float] = None) -> List[EmbeddingResult]
```

Online VNE simulation with temporal constraints and automatic VNR departures.

**Parameters**:
- `simulation_duration`: Optional maximum simulation time

**Example**:
```python
# Online simulation with VNR arrivals and departures
results = algorithm.embed_online(vnrs, substrate, simulation_duration=10000.0)
```

#### Metrics and Statistics

##### calculate_metrics()

```python
def calculate_metrics(self, 
                     results: List[EmbeddingResult],
                     substrate: Optional[SubstrateNetwork] = None) -> Dict[str, Any]
```

Calculate comprehensive VNE metrics using standard literature formulas.

**Example**:
```python
metrics = algorithm.calculate_metrics(results, substrate)
print(f"Acceptance ratio: {metrics['primary_metrics']['acceptance_ratio']:.3f}")
print(f"Revenue/Cost ratio: {metrics['primary_metrics']['revenue_to_cost_ratio']:.2f}")
```

##### get_algorithm_statistics()

```python
def get_algorithm_statistics(self) -> Dict[str, Any]
```

Get algorithm-specific performance statistics.

**Returns**: Dictionary with:
- `total_requests`, `successful_requests`: Counts
- `acceptance_ratio`, `average_execution_time`: Performance metrics  
- `revenue_to_cost_ratio`: Efficiency metrics
- `algorithm_name`, `algorithm_parameters`: Algorithm info

#### Abstract Methods (Must Implement)

##### _embed_single_vnr()

```python
@abstractmethod
def _embed_single_vnr(self, 
                     vnr: VirtualNetworkRequest,
                     substrate: SubstrateNetwork) -> EmbeddingResult
```

Core algorithm implementation with resource allocation.

**Responsibilities**:
- Check resource availability
- Perform node and link mapping
- Allocate resources during embedding
- Return embedding result with mappings

##### _cleanup_failed_embedding()

```python
@abstractmethod  
def _cleanup_failed_embedding(self, 
                             vnr: VirtualNetworkRequest,
                             substrate: SubstrateNetwork,
                             result: EmbeddingResult) -> None
```

Clean up resources for failed embeddings.

**Responsibilities**:
- Deallocate any resources allocated during `_embed_single_vnr()`
- Reset substrate state to pre-embedding condition

---

### EmbeddingResult

**Module**: `src.algorithms.base_algorithm`

Standardized result structure for VNE embedding attempts.

#### Class Definition

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

**Example**:
```python
# Successful embedding result
success_result = EmbeddingResult(
    vnr_id="1",
    success=True,
    node_mapping={"0": "3", "1": "7", "2": "1"},
    link_mapping={("0", "1"): ["3", "2", "7"], ("1", "2"): ["7", "1"]},
    revenue=250.75,
    cost=145.50,
    execution_time=0.0023,
    algorithm_name="Yu et al. (2008) Two-Stage Algorithm",
    metadata={"allocated_nodes": 3, "allocated_links": 2}
)

# Failed embedding result
failed_result = EmbeddingResult(
    vnr_id="2",
    success=False,
    node_mapping={},
    link_mapping={},
    revenue=0.0,
    cost=0.0,
    execution_time=0.0015,
    failure_reason="Insufficient CPU resources",
    algorithm_name="Yu et al. (2008) Two-Stage Algorithm"
)
```

---

### YuAlgorithm

**Module**: `src.algorithms.baseline.yu_2008_algorithm`

Literature-compliant implementation of Yu et al. (2008) two-stage VNE algorithm.

#### Class Definition

```python
class YuAlgorithm(BaseAlgorithm):
    def __init__(self, 
                 k_paths: int = 1,
                 path_selection_strategy: str = "shortest",
                 enable_path_caching: bool = True, 
                 **kwargs)
```

**Parameters**:
- `k_paths` (int): Number of shortest paths to consider for link mapping
- `path_selection_strategy` (str): Path selection strategy ("shortest" or "bandwidth")  
- `enable_path_caching` (bool): Enable path caching for performance optimization

**Example**:
```python
# Basic Yu algorithm
yu_basic = YuAlgorithm()

# Advanced configuration
yu_advanced = YuAlgorithm(
    k_paths=3,
    path_selection_strategy="bandwidth",
    enable_path_caching=True
)
```

#### Algorithm-Specific Methods

##### clear_path_cache()

```python
def clear_path_cache(self) -> None
```

Clear the internal path cache to free memory.

**Example**:
```python
# After processing many VNRs, clear cache
algorithm.clear_path_cache()
```

#### Algorithm Workflow

The Yu algorithm implements a two-stage approach:

1. **Stage 1 - Node Mapping**: 
   - Rank virtual nodes by CPU requirements (decreasing)
   - Select substrate nodes with highest available CPU (load balancing)
   - Allocate CPU resources immediately

2. **Stage 2 - Link Mapping**:
   - Find k-shortest paths between mapped nodes
   - Select path with sufficient bandwidth  
   - Allocate bandwidth resources immediately

**Example Usage**:
```python
# Create algorithm instance
yu_algorithm = YuAlgorithm(k_paths=3, path_selection_strategy="shortest")

# Single VNR embedding
result = yu_algorithm.embed_vnr(vnr, substrate)

# Batch processing
results = yu_algorithm.embed_batch(vnr_list, substrate)

# Get algorithm statistics
stats = yu_algorithm.get_algorithm_statistics()
print(f"Yu Algorithm - AR: {stats['acceptance_ratio']:.3f}")
```

---

## Metrics and Analysis

**Module**: `src.utils.metrics`

Comprehensive performance evaluation using standard VNE literature formulas.

### Core Metrics Functions

#### Revenue and Cost Calculation

##### calculate_vnr_revenue()

```python
def calculate_vnr_revenue(vnr: VirtualNetworkRequest) -> float
```

Calculate revenue using standard VNE formula: Revenue = Σ(CPU) + Σ(Bandwidth) + Σ(Memory if used).

**Example**:
```python
revenue = calculate_vnr_revenue(vnr)
print(f"VNR revenue: {revenue:.2f}")
```

##### calculate_vnr_cost()

```python
def calculate_vnr_cost(vnr: VirtualNetworkRequest,
                      node_mapping: Dict[str, str],
                      link_mapping: Dict[Tuple[str, str], List[str]],
                      substrate_network: Optional[SubstrateNetwork] = None) -> float
```

Calculate embedding cost using standard formula: Cost = Σ(CPU allocated) + Σ(Bandwidth × path_length).

**Example**:
```python
cost = calculate_vnr_cost(vnr, result.node_mapping, result.link_mapping, substrate)
print(f"Embedding cost: {cost:.2f}")
```

#### Primary VNE Metrics

##### calculate_acceptance_ratio()

```python
def calculate_acceptance_ratio(results: List[EmbeddingResult]) -> float
```

Standard VNE acceptance ratio: AR = |Successful_VNRs| / |Total_VNRs|.

##### calculate_revenue_to_cost_ratio()

```python
def calculate_revenue_to_cost_ratio(results: List[EmbeddingResult]) -> float
```

Efficiency metric: R/C = Total_Revenue / Total_Cost.

**Example**:
```python
# Calculate key metrics
acceptance_ratio = calculate_acceptance_ratio(results)
rc_ratio = calculate_revenue_to_cost_ratio(results)

print(f"Acceptance Ratio: {acceptance_ratio:.3f}")
print(f"Revenue/Cost Ratio: {rc_ratio:.2f}")
```

#### Resource Utilization

##### calculate_utilization()

```python
def calculate_utilization(substrate_network: SubstrateNetwork) -> Dict[str, float]
```

Calculate average resource utilization for substrate network.

**Returns**: Dictionary with:
- `avg_node_cpu_util`: Average CPU utilization across nodes
- `avg_node_memory_util`: Average memory utilization (if enabled)
- `avg_link_bandwidth_util`: Average bandwidth utilization across links

**Example**:
```python
utilization = calculate_utilization(substrate)
print(f"CPU utilization: {utilization['avg_node_cpu_util']:.2%}")
print(f"Bandwidth utilization: {utilization['avg_link_bandwidth_util']:.2%}")
```

#### Comprehensive Analysis

##### generate_comprehensive_metrics_summary()

```python
def generate_comprehensive_metrics_summary(
    results: List[EmbeddingResult],
    substrate_network: Optional[SubstrateNetwork] = None,
    time_duration: Optional[float] = None
) -> Dict[str, Any]
```

Generate complete metrics report with all VNE performance indicators.

**Returns**: Dictionary with sections:
- `basic_stats`: Counts and basic statistics
- `primary_metrics`: Standard VNE metrics (AR, revenue, cost, etc.)
- `performance_metrics`: Execution time and throughput
- `utilization_metrics`: Resource utilization (if substrate provided)
- `efficiency_metrics`: Advanced efficiency measures
- `failure_analysis`: Failure reason distribution

**Example**:
```python
# Generate complete metrics report
metrics = generate_comprehensive_metrics_summary(results, substrate, time_duration=1000.0)

# Access different metric categories
print("=== Basic Statistics ===")
basic = metrics['basic_stats']
print(f"Total requests: {basic['total_requests']}")
print(f"Successful: {basic['successful_requests']}")

print("\n=== Primary VNE Metrics ===")
primary = metrics['primary_metrics']
print(f"Acceptance ratio: {primary['acceptance_ratio']:.3f}")
print(f"Total revenue: {primary['total_revenue']:.2f}")
print(f"Revenue/Cost ratio: {primary['revenue_to_cost_ratio']:.2f}")

print("\n=== Performance Metrics ===")
performance = metrics['performance_metrics']
print(f"Average execution time: {performance['average_execution_time']:.4f}s")
print(f"Throughput: {performance['throughput']:.2f} VNRs/time_unit")

if 'utilization_metrics' in metrics:
    print("\n=== Resource Utilization ===")
    util = metrics['utilization_metrics']
    print(f"CPU utilization: {util['avg_node_cpu_util']:.2%}")
    print(f"Bandwidth utilization: {util['avg_link_bandwidth_util']:.2%}")
```

### Utility Functions

##### create_embedding_result_from_vnr()

```python
def create_embedding_result_from_vnr(vnr: VirtualNetworkRequest,
                                    success: bool,
                                    node_mapping: Optional[Dict] = None,
                                    link_mapping: Optional[Dict] = None,
                                    execution_time: float = 0.0,
                                    failure_reason: Optional[str] = None) -> EmbeddingResult
```

Create properly formatted EmbeddingResult with automatic revenue/cost calculation.

**Example**:
```python
# Create result for successful embedding
result = create_embedding_result_from_vnr(
    vnr=vnr,
    success=True,
    node_mapping={"0": "3", "1": "7"},
    link_mapping={("0", "1"): ["3", "2", "7"]},
    execution_time=0.005
)
```

##### list_available_metrics()

```python
def list_available_metrics() -> Dict[str, List[str]]
```

List all available metrics organized by category.

**Example**:
```python
available = list_available_metrics()
for category, metric_list in available.items():
    print(f"{category}:")
    for metric in metric_list:
        print(f"  - {metric}")
```

---

## Common Use Cases

### 1. Basic Algorithm Development

```python
from src.models.substrate import SubstrateNetwork
from src.models.virtual_request import VirtualNetworkRequest
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult

class SimpleGreedyAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__("Simple Greedy Algorithm")
    
    def _embed_single_vnr(self, vnr, substrate):
        # Simple greedy node mapping
        node_mapping = {}
        allocated_nodes = []
        
        for vnode_id, vnode_req in vnr.virtual_nodes.items():
            # Find first substrate node with sufficient CPU
            for snode_id in substrate.graph.nodes:
                if substrate.check_node_resources(snode_id, vnode_req.cpu_requirement):
                    success = substrate.allocate_node_resources(
                        snode_id, vnode_req.cpu_requirement
                    )
                    if success:
                        node_mapping[str(vnode_id)] = str(snode_id)
                        allocated_nodes.append((snode_id, vnode_req.cpu_requirement))
                        break
            else:
                # Rollback and fail
                self._rollback_allocations(substrate, allocated_nodes, [])
                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id), success=False,
                    node_mapping={}, link_mapping={},
                    revenue=0.0, cost=0.0, execution_time=0.0,
                    failure_reason="No suitable substrate node found"
                )
        
        # Simplified link mapping (direct links only)
        link_mapping = {}
        allocated_links = []
        
        for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
            ssrc = int(node_mapping[str(vsrc)])
            sdst = int(node_mapping[str(vdst)])
            
            if substrate.graph.has_edge(ssrc, sdst):
                success = substrate.allocate_link_resources(
                    ssrc, sdst, vlink_req.bandwidth_requirement
                )
                if success:
                    link_mapping[(str(vsrc), str(vdst))] = [str(ssrc), str(sdst)]
                    allocated_links.append((ssrc, sdst, vlink_req.bandwidth_requirement))
                else:
                    # Rollback and fail
                    self._rollback_allocations(substrate, allocated_nodes, allocated_links)
                    return EmbeddingResult(
                        vnr_id=str(vnr.vnr_id), success=False,
                        node_mapping={}, link_mapping={},
                        revenue=0.0, cost=0.0, execution_time=0.0,
                        failure_reason="Insufficient bandwidth"
                    )
        
        return EmbeddingResult(
            vnr_id=str(vnr.vnr_id), success=True,
            node_mapping=node_mapping, link_mapping=link_mapping,
            revenue=0.0, cost=0.0, execution_time=0.0  # Will be calculated by base class
        )
    
    def _cleanup_failed_embedding(self, vnr, substrate, result):
        # Cleanup implementation
        for vnode_id, vnode_req in vnr.virtual_nodes.items():
            if str(vnode_id) in result.node_mapping:
                snode_id = int(result.node_mapping[str(vnode_id)])
                substrate.deallocate_node_resources(snode_id, vnode_req.cpu_requirement)
        
        for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
            if (str(vsrc), str(vdst)) in result.link_mapping:
                path = [int(n) for n in result.link_mapping[(str(vsrc), str(vdst))]]
                for i in range(len(path) - 1):
                    substrate.deallocate_link_resources(
                        path[i], path[i+1], vlink_req.bandwidth_requirement
                    )
```

### 2. Network Generation and Algorithm Evaluation

```python
from src.models.substrate import SubstrateNetwork
from src.models.virtual_request import VirtualNetworkRequest
from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm
from src.utils.metrics import generate_comprehensive_metrics_summary
import random

# Generate substrate network
substrate = SubstrateNetwork()
for i in range(20):
    substrate.add_node(i, cpu_capacity=random.randint(50, 150))

for i in range(20):
    for j in range(i+1, 20):
        if random.random() < 0.3:  # 30% edge probability
            substrate.add_link(i, j, bandwidth_capacity=random.randint(100, 500))

# Generate VNR batch
vnrs = []
for i in range(50):
    vnr = VirtualNetworkRequest(vnr_id=i, arrival_time=i*10.0, holding_time=1000.0)
    
    # Add 3-5 virtual nodes
    num_nodes = random.randint(3, 5)
    for j in range(num_nodes):
        vnr.add_virtual_node(j, cpu_requirement=random.randint(10, 30))
    
    # Add virtual links (star topology)
    for j in range(1, num_nodes):
        vnr.add_virtual_link(0, j, bandwidth_requirement=random.randint(20, 80))
    
    vnrs.append(vnr)

# Run algorithm
algorithm = YuAlgorithm(k_paths=3)
results = algorithm.embed_batch(vnrs, substrate)

# Analyze results
metrics = generate_comprehensive_metrics_summary(results, substrate)
print(f"Acceptance Ratio: {metrics['primary_metrics']['acceptance_ratio']:.3f}")
print(f"Revenue/Cost Ratio: {metrics['primary_metrics']['revenue_to_cost_ratio']:.2f}")
```

### 3. Online Simulation with Temporal Constraints

```python
from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm
import random

# Create VNRs with realistic arrival times and lifetimes
vnrs = []
current_time = 0.0

for i in range(100):
    # Poisson arrivals
    inter_arrival = random.expovariate(0.1)  # Average 10 time units between arrivals
    current_time += inter_arrival
    
    # Exponential holding times
    holding_time = random.expovariate(1.0/1000.0)  # Average 1000 time units
    
    vnr = VirtualNetworkRequest(
        vnr_id=i, 
        arrival_time=current_time, 
        holding_time=holding_time
    )
    
    # Add virtual nodes and links...
    vnrs.append(vnr)

# Run online simulation
algorithm = YuAlgorithm()
results = algorithm.embed_online(vnrs, substrate, simulation_duration=5000.0)

# The algorithm automatically handles VNR departures
print(f"Processed {len(results)} VNRs in online simulation")
```

### 4. Advanced Metrics Analysis

```python
from src.utils.metrics import *
import matplotlib.pyplot as plt

# Calculate individual metrics
acceptance_ratio = calculate_acceptance_ratio(results)
total_revenue = calculate_total_revenue(results) 
total_cost = calculate_total_cost(results)
utilization = calculate_utilization(substrate)

# Time series analysis (if timestamps available)
successful_results = [r for r in results if r.success]
timestamps = [r.timestamp for r in successful_results if r.timestamp]

if timestamps:
    # Plot acceptance ratio over time
    time_windows = []
    window_size = 100.0  # 100 time units
    
    start_time = min(timestamps)
    end_time = max(timestamps)
    current = start_time
    
    while current < end_time:
        window_results = [r for r in results 
                         if current <= r.timestamp < current + window_size]
        if window_results:
            window_ar = sum(1 for r in window_results if r.success) / len(window_results)
            time_windows.append((current, window_ar))
        current += window_size
    
    # Plot results
    times, ars = zip(*time_windows) if time_windows else ([], [])
    plt.plot(times, ars)
    plt.xlabel('Time')
    plt.ylabel('Acceptance Ratio')
    plt.title('Acceptance Ratio Over Time')
    plt.show()
```

### 5. Algorithm Comparison Framework

```python
from src.algorithms.baseline.yu_2008_algorithm import YuAlgorithm

def compare_algorithms(algorithms, vnrs, substrate):
    """Compare multiple algorithms on the same dataset."""
    results = {}
    
    for name, algorithm in algorithms.items():
        # Reset substrate before each algorithm
        substrate.reset_allocations()
        
        # Run algorithm
        algo_results = algorithm.embed_batch(vnrs.copy(), substrate)
        
        # Calculate metrics
        metrics = generate_comprehensive_metrics_summary(algo_results, substrate)
        results[name] = {
            'results': algo_results,
            'metrics': metrics,
            'algorithm_stats': algorithm.get_algorithm_statistics()
        }
        
        print(f"{name}:")
        print(f"  Acceptance Ratio: {metrics['primary_metrics']['acceptance_ratio']:.3f}")
        print(f"  Revenue/Cost: {metrics['primary_metrics']['revenue_to_cost_ratio']:.2f}")
        print(f"  Avg Execution Time: {metrics['performance_metrics']['average_execution_time']:.4f}s")
        print()
    
    return results

# Usage
algorithms = {
    'Yu2008_k1': YuAlgorithm(k_paths=1),
    'Yu2008_k3': YuAlgorithm(k_paths=3),
    'Yu2008_bandwidth': YuAlgorithm(k_paths=3, path_selection_strategy="bandwidth")
}

comparison_results = compare_algorithms(algorithms, vnrs, substrate)
```

---

## Error Handling

The framework provides comprehensive error handling with specific exception types:

### Model Exceptions

```python
from src.models.substrate import ResourceAllocationError, SubstrateNetworkError
from src.models.virtual_request import VNRValidationError, VNRError

try:
    substrate.allocate_node_resources(node_id=999, cpu=50.0)
except ResourceAllocationError as e:
    print(f"Resource allocation failed: {e}")

try:
    vnr.add_virtual_node(-1, cpu_requirement=-10.0)
except VNRValidationError as e:
    print(f"VNR validation failed: {e}")
```

### Algorithm Exceptions

```python
from src.algorithms.base_algorithm import VNEConstraintError

# Constraint violations are detected automatically
try:
    result = algorithm.embed_vnr(vnr, substrate)
    if not result.success:
        print(f"Embedding failed: {result.failure_reason}")
except VNEConstraintError as e:
    print(f"VNE constraint violation: {e}")
```

### Best Practices

1. **Always check return values**: Methods like `allocate_node_resources()` return boolean success indicators
2. **Handle resource cleanup**: Implement proper `_cleanup_failed_embedding()` methods
3. **Validate inputs**: Use validation methods before processing
4. **Use try-catch blocks**: Wrap resource allocation in exception handling
5. **Check constraint compatibility**: Ensure VNR constraints match substrate capabilities

---

This API documentation provides complete coverage of the VNE Heuristics framework's public interface. For implementation examples and advanced usage patterns, refer to the algorithm implementations and test cases in the framework.