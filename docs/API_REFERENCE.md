# VNE Heuristics Framework - Comprehensive API Reference

**Version**: 1.0  
**Last Updated**: June 2025  
**Purpose**: Complete technical reference for all functions, methods, and classes in the VNE Heuristics framework.

---

## Table of Contents

1. [Core Models](#core-models)
   - [SubstrateNetwork](#substratenetwork)
   - [VirtualNetworkRequest](#virtualnetworkrequest)
   - [VNRBatch](#vnrbatch)
2. [Algorithm Framework](#algorithm-framework)
   - [BaseAlgorithm](#basealgorithm)
   - [YuAlgorithm](#yualgorithm)
   - [EmbeddingResult](#embeddingresult)
3. [Generators](#generators)
   - [Substrate Generators](#substrate-generators)
   - [VNR Generators](#vnr-generators)
   - [Generation Configuration](#generation-configuration)
4. [Metrics and Analysis](#metrics-and-analysis)
5. [I/O and Experiment Management](#io-and-experiment-management)
6. [CLI System](#cli-system)
   - [VNECommandLineInterface](#vnecommandlineinterface)
   - [Commands](#commands)
   - [Argument Parser](#argument-parser)
7. [Core Infrastructure](#core-infrastructure)
   - [AlgorithmRegistry](#algorithmregistry)
   - [ErrorHandler](#errorhandler)
   - [ProgressReporter](#progressreporter)
8. [Configuration Management](#configuration-management)
9. [Exception Classes](#exception-classes)

---

## Core Models

### SubstrateNetwork

**Module**: `src.models.substrate`  
**Purpose**: Represents the physical infrastructure network with thread-safe resource management.

#### Constructor

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

#### Node Management Methods

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

Add a node to the substrate network.

**Parameters**:
- `node_id` (int): Unique identifier for the node
- `cpu_capacity` (float): CPU capacity of the node (must be >= 0)
- `memory_capacity` (float): Memory capacity of the node (ignored if memory constraints disabled)
- `x_coord` (float): X coordinate for visualization
- `y_coord` (float): Y coordinate for visualization
- `node_type` (str): Type of the node (e.g., "server", "switch")

**Raises**: `ValueError` if capacity values are invalid

#### Resource Management Methods

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

**Returns**: `bool` - `True` if allocation successful, `False` if insufficient resources

**Raises**: `ResourceAllocationError` if node doesn't exist or invalid parameters

##### deallocate_node_resources()

```python
def deallocate_node_resources(self, 
                             node_id: int,
                             cpu: float,
                             memory: float = 0.0) -> bool
```

Deallocate resources from a substrate node (thread-safe).

**Returns**: `bool` - `True` if deallocation successful

**Raises**: `ResourceAllocationError` if node doesn't exist or invalid parameters

##### check_node_resources()

```python
def check_node_resources(self, 
                        node_id: int,
                        cpu: float,
                        memory: float = 0.0) -> Dict[str, bool]
```

Check if node has sufficient resources without allocating.

##### get_node_resources()

```python
def get_node_resources(self, node_id: int) -> Optional[NodeResources]
```

Get resource information for a specific node.

**Returns**: `NodeResources` dataclass or `None` if node doesn't exist

#### Link Management Methods

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

Add a link to the substrate network.

**Parameters**:
- `src`, `dst` (int): Source and destination node IDs
- `bandwidth_capacity` (float): Bandwidth capacity of the link (must be >= 0)
- `delay` (float): Link delay (ignored if delay constraints disabled)
- `cost` (float): Link cost (ignored if cost constraints disabled)
- `reliability` (float): Link reliability between 0 and 1 (ignored if reliability constraints disabled)

**Raises**: `ValueError` if capacity or parameter values are invalid

##### allocate_link_resources()

```python
def allocate_link_resources(self, 
                           src: int,
                           dst: int,
                           bandwidth: float) -> bool
```

Allocate bandwidth from a substrate link (thread-safe).

**Returns**: `bool` - `True` if allocation successful, `False` if insufficient bandwidth

**Raises**: `ResourceAllocationError` if link doesn't exist or invalid parameters

##### deallocate_link_resources()

```python
def deallocate_link_resources(self, 
                             src: int,
                             dst: int,
                             bandwidth: float) -> bool
```

Deallocate bandwidth from a substrate link (thread-safe).

**Returns**: `bool` - `True` if deallocation successful

**Raises**: `ResourceAllocationError` if link doesn't exist or invalid parameters

##### check_link_resources()

```python
def check_link_resources(self, 
                        src: int,
                        dst: int,
                        bandwidth: float) -> bool
```

Check if link has sufficient bandwidth without allocating.

##### get_link_resources()

```python
def get_link_resources(self, src: int, dst: int) -> Optional[LinkResources]
```

Get resource information for a specific link.

**Returns**: `LinkResources` dataclass or `None` if link doesn't exist

#### Network Analysis Methods

##### get_network_statistics()

```python
def get_network_statistics(self) -> Dict[str, Any]
```

Get comprehensive network statistics.

**Returns**: Dictionary containing:
- `node_count`: Number of nodes in the network
- `link_count`: Number of links in the network
- `average_degree`: Average Degree of the network
- `total_cpu`: Sum of all CPU capacities
- `used_cpu`: Sum of all allocated CPU
- `available_cpu`: Sum of all available CPU
- `total_bandwidth`: Sum of all bandwidth capacities
- `used_bandwidth`: Sum of all allocated bandwidth
- `available_bandwidth`: Sum of all available bandwidth
- `cpu_utilization`: Overall CPU utilization ratio
- `bandwidth_utilization`: Overall bandwidth utilization ratio
- `is_connected`: Whether the network is connected
- `constraint_configuration`: Dictionary of enabled constraints
- Additional memory statistics if memory constraints enabled

##### get_constraint_configuration()

```python
def get_constraint_configuration(self) -> Dict[str, bool]
```

Get current constraint configuration.

**Returns**: Dictionary with:
- `cpu_constraints`: Always `True`
- `bandwidth_constraints`: Always `True`
- `memory_constraints`: Whether memory constraints are enabled
- `delay_constraints`: Whether delay constraints are enabled
- `cost_constraints`: Whether cost constraints are enabled
- `reliability_constraints`: Whether reliability constraints are enabled

##### reset_allocations()

```python
def reset_allocations(self) -> None
```

Reset all resource allocations to zero.

#### Persistence Methods

##### save_to_csv()

```python
def save_to_csv(self, nodes_file: str, links_file: str) -> None
```

Save substrate network to CSV files.

**Parameters**:
- `nodes_file` (str): Path for nodes CSV file
- `links_file` (str): Path for links CSV file

##### load_from_csv()

```python
def load_from_csv(self, nodes_file: str, links_file: str) -> None
```

Load substrate network from CSV files.

**Raises**: 
- `FileNotFoundError` if files don't exist
- `FileFormatError` if file format is invalid

#### Validation Methods

##### validate_network()

```python
def validate_network(self) -> List[str]
```

Validate network consistency.

**Returns**: List of validation error messages (empty if valid)

#### Special Methods

```python
def __len__(self) -> int  # Returns number of nodes
def __contains__(self, node_id: int) -> bool  # Check if node exists
def __str__(self) -> str  # String representation
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
```

---

### VirtualNetworkRequest

**Module**: `src.models.virtual_request`  
**Purpose**: Represents a virtual network request that needs to be embedded.

#### Constructor

```python
class VirtualNetworkRequest:
    def __init__(self, 
                 vnr_id: int,
                 arrival_time: float = 0.0,
                 holding_time: float = float('inf'),
                 priority: int = 1)
```
**Attributes**:
- virtual_nodes: Dict[int, VirtualNodeRequirement]  # Dictionary of virtual nodes
- virtual_links: Dict[Tuple[int, int], VirtualLinkRequirement]  # Dictionary of virtual links
- graph: nx.Graph()  # NetworkX graph representation of the VNR topology

**Parameters**:
- `vnr_id` (int): Unique identifier for the VNR.
- `arrival_time` (float): Arrival time for temporal simulation. Default: `0.0`
- `holding_time` (float): Duration before departure. Default: `float('inf')`
- `priority` (int): Priority level (higher = more important). Default: `1`


#### Virtual Node Management

##### add_virtual_node()

```python
def add_virtual_node(self, 
                     node_id: int,
                     cpu_requirement: float,
                     memory_requirement: float = 0.0,
                     node_constraints: Optional[Dict[str, Any]] = None) -> None
```

Add a virtual node to the VNR.

**Parameters**:
- `node_id` (int): Unique identifier for the virtual node
- `cpu_requirement` (float): Required CPU resources
- `memory_requirement` (float): Required memory resources (default: 0.0 = no requirement)
- `node_constraints` (Dict): Additional constraints (optional)

**Raises**: `VNRValidationError` if node already exists or requirements are invalid

##### remove_virtual_node()

```python
def remove_virtual_node(self, node_id: int) -> None
```

Remove a virtual node and all incident links.

**Raises**: `VNRValidationError` if node doesn't exist

#### Virtual Link Management

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

Add a virtual link to the VNR.

**Parameters**:
- `src_node`, `dst_node` (int): Source and destination virtual node IDs
- `bandwidth_requirement` (float): Required bandwidth
- `delay_constraint` (float): Maximum acceptable delay (ignored if 0.0)
- `reliability_requirement` (float): Minimum required reliability (ignored if 0.0)
- `link_constraints` (Dict): Additional constraints (optional)

**Raises**: `VNRValidationError` if nodes don't exist or requirements are invalid

##### remove_virtual_link()

```python
def remove_virtual_link(self, src_node: int, dst_node: int) -> None
```

Remove a virtual link.

**Raises**: `VNRValidationError` if link doesn't exist

#### Resource Analysis Methods

##### calculate_total_requirements()

```python
def calculate_total_requirements(self, 
                                include_memory: bool = True,
                                include_delay: bool = True,
                                include_reliability: bool = True) -> Dict[str, float]
```

Calculate total resource requirements for this VNR.

**Returns**: Dictionary with:
- `total_cpu`: Sum of all CPU requirements
- `total_bandwidth`: Sum of all bandwidth requirements
- `node_count`: Number of virtual nodes
- `link_count`: Number of virtual links
- `total_memory`: Sum of all memory requirements (if included)
- `total_delay_constraint`: Sum of delay constraints for constrained links (if included)
- `delay_constrained_links`: Number of links with delay constraints (if included)
- `avg_reliability_requirement`: Average reliability requirement for constrained links (if included)
- `reliability_constrained_links`: Number of links with reliability requirements (if included)

##### get_constraint_summary()

```python
def get_constraint_summary(self) -> Dict[str, Any]
```

Get summary of which constraint types are used by this VNR.

**Returns**: Dictionary with:
- `uses_memory_constraints`: True if any node has memory_requirement > 0
- `uses_delay_constraints`: True if any link has delay_constraint > 0
- `uses_reliability_constraints`: True if any link has reliability_requirement > 0
- `memory_constrained_nodes`: Count of nodes with memory requirements
- `delay_constrained_links`: Count of links with delay constraints
- `reliability_constrained_links`: Count of links with reliability requirements
- `total_nodes`: Total number of virtual nodes
- `total_links`: Total number of virtual links

#### Temporal Methods

##### get_departure_time()

```python
def get_departure_time(self, embedding_time: float) -> float
```

Calculate departure time based on embedding time and holding time.

**Parameters**:
- `embedding_time` (float): Time when VNR was embedded

**Returns**: Departure time, or `float('inf')` if infinite holding time

#### Validation Methods

##### validate_request()

```python
def validate_request(self, 
                    validate_memory: bool = True,
                    validate_delay: bool = True,
                    validate_reliability: bool = True) -> List[str]
```

Validate VNR consistency.

**Parameters**:
- `validate_memory` (bool): Whether to validate memory constraints
- `validate_delay` (bool): Whether to validate delay constraints  
- `validate_reliability` (bool): Whether to validate reliability constraints

**Returns**: List of validation error messages (empty if valid)

#### Persistence Methods

##### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

Convert VNR to dictionary representation.

##### from_dict()

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'VirtualNetworkRequest'
```

Create VNR from dictionary representation.

##### save_to_json()

```python
def save_to_json(self, filepath: str) -> None
```

Save VNR to JSON file.

##### load_from_json()

```python
@classmethod
def load_from_json(cls, filepath: str) -> 'VirtualNetworkRequest'
```

Load VNR from JSON file.

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

### VNRBatch

**Module**: `src.models.vnr_batch`  
**Purpose**: Manages collections of VNRs for batch processing and experiments.

#### Constructor

```python
class VNRBatch:
    def __init__(self, 
                 vnrs: Optional[List[VirtualNetworkRequest]] = None,
                 batch_id: str = "default")
```

**Parameters**:
- `vnrs` (List): Initial list of VNRs (default: empty list)
- `batch_id` (str): Unique batch identifier

#### Batch Management Methods

##### add_vnr()

```python
def add_vnr(self, vnr: VirtualNetworkRequest) -> None
```

Add a VNR to the batch.

##### remove_vnr()

```python
def remove_vnr(self, vnr_id: int) -> bool
```

Remove VNR by ID.

**Returns**: `True` if removed, `False` if not found

##### get_vnr()

```python
def get_vnr(self, vnr_id: int) -> Optional[VirtualNetworkRequest]
```

Get VNR by ID.

**Returns**: VNR instance or `None` if not found

#### Batch Operations

##### sort_by_arrival_time()

```python
def sort_by_arrival_time(self, reverse: bool = False) -> None
```

Sort VNRs by arrival time.

**Parameters**:
- `reverse` (bool): Sort in descending order if `True`

##### sort_by_priority()

```python
def sort_by_priority(self, reverse: bool = True) -> None
```

Sort VNRs by priority.

**Parameters**:
- `reverse` (bool): Sort in descending order (highest priority first). Default: `True`

##### sort_by_holding_time()

```python
def sort_by_holding_time(self, reverse: bool = False) -> None
```

Sort VNRs by holding time.

**Parameters**:
- `reverse` (bool): Sort in descending order if `True`

##### sort_by_node_count()

```python
def sort_by_node_count(self, reverse: bool = False) -> None
```

Sort VNRs by number of virtual nodes.

**Parameters**:
- `reverse` (bool): Sort in descending order if `True`

##### filter_by_time_range()

```python
def filter_by_time_range(self, start_time: float, end_time: float) -> 'VNRBatch'
```

Get VNRs arriving within time range.

**Parameters**:
- `start_time` (float): Start time (inclusive)
- `end_time` (float): End time (exclusive)

**Returns**: VNRBatch with filtered VNRs

##### filter_by_priority_range()

```python
def filter_by_priority_range(self, min_priority: int, max_priority: int) -> 'VNRBatch'
```

Filter VNRs by priority range.

**Parameters**:
- `min_priority` (int): Minimum priority (inclusive)
- `max_priority` (int): Maximum priority (inclusive)

**Returns**: VNRBatch with filtered VNRs

##### filter_by_node_count_range()

```python
def filter_by_node_count_range(self, min_nodes: int, max_nodes: int) -> 'VNRBatch'
```

Filter VNRs by node count.

**Parameters**:
- `min_nodes` (int): Minimum number of nodes (inclusive)
- `max_nodes` (int): Maximum number of nodes (inclusive)

**Returns**: VNRBatch with filtered VNRs

##### merge_batch()

```python
def merge_batch(self, other: 'VNRBatch') -> 'VNRBatch'
```

Merge with another batch.

**Parameters**:
- `other` (VNRBatch): Another VNRBatch to merge with

**Returns**: VNRBatch containing all VNRs

##### split_batch()

```python
def split_batch(self, max_vnrs_per_batch: int) -> List['VNRBatch']
```

Split batch into smaller batches.

**Parameters**:
- `max_vnrs_per_batch` (int): Maximum number of VNRs per batch

**Returns**: List of smaller VNRBatch instances

#### Analysis Methods

##### get_basic_info()

```python
def get_basic_info(self) -> Dict[str, Any]
```

Get batch statistics.

**Returns**: Dictionary with:
- `batch_id`: Batch identifier
- `count`: Number of VNRs
- `is_empty`: Whether batch is empty
- `arrival_time_range`: (min, max) arrival times
- `priority_range`: (min, max) priorities
- `node_count_range`: (min, max) node counts
- `link_count_range`: (min, max) link counts
- `holding_time_range`: (min, max) holding time
- `avg_nodes_per_vnr`: Average nodes per VNR
- `avg_links_per_vnr`: Average links per VNR

#### Persistence Methods

##### save_to_csv()

```python
def save_to_csv(self, base_filename: str) -> None
```

Save batch to CSV files.

**Parameters**:
- `base_filename` (str): Base filename (without extension)

**Creates**:
- `{base_filename}_metadata.csv`: VNR timing and metadata
- `{base_filename}_nodes.csv`: Virtual node requirements
- `{base_filename}_links.csv`: Virtual link requirements

##### load_from_csv()

```python
@classmethod
def load_from_csv(cls, 
                  base_filename: str,
                  batch_id: Optional[str] = None) -> 'VNRBatch'
```

Load batch from CSV files.

**Parameters**:
- `base_filename` (str): Base filename (without extension)
- `batch_id` (str): Optional batch identifier

**Returns**: VNRBatch instance

##### save_to_json()

```python
def save_to_json(self, filepath: str) -> None
```

Save VNR batch to JSON file.

**Parameters**:
- `filepath` (str): Path to save the JSON file

##### load_from_json()

```python
@classmethod
def load_from_json(cls, filepath: str) -> 'VNRBatch'
```

Create batch from JSON file.

**Parameters**:
- `filepath` (str): Path to the JSON file

**Returns**: VNRBatch instance

#### Special Methods

```python
def __len__(self) -> int # Return number of VNRs in the batch
def __iter__(self) -> Iterator[VirtualNetworkRequest] # Iterate over VNRs in the batch
def __getitem__(self, index: int) -> VirtualNetworkRequest # Get VNR by index
def __bool__(self) -> bool # Return True if batch contains VNRs
def __str__(self) -> str # String representation of the batch
def __repr__(self) -> str # Simple representation of the batch
```
