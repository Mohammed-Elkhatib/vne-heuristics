# Contributing to VNE Heuristics

Thank you for your interest in contributing to the VNE Heuristics framework! This document provides guidelines for contributing code, documentation, algorithms, and improvements to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Adding New Algorithms](#adding-new-algorithms)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Issue Reporting](#issue-reporting)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Familiarity with Virtual Network Embedding concepts
- Basic understanding of graph theory and network algorithms

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/vne-heuristics.git
   cd vne-heuristics
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install core dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (for testing and code quality)
   pip install pytest>=7.0.0 black>=22.0.0 flake8>=4.0.0 mypy>=0.900
   ```

4. **Verify Installation**
   ```bash
   # Run basic functionality test
   python main.py --help
   
   # Run test suite
   python -m pytest tests/ -v
   ```

5. **Set Up Git Hooks (Optional but Recommended)**
   ```bash
   # Create pre-commit hook for code formatting
   cat > .git/hooks/pre-commit << 'EOF'
   #!/bin/bash
   black --check src/ cli/ tests/
   flake8 src/ cli/ tests/
   EOF
   chmod +x .git/hooks/pre-commit
   ```

## Code Standards

The VNE Heuristics framework follows established Python conventions with specific guidelines for research code quality and reproducibility.

### Python Code Style

#### General Guidelines
- **PEP 8 Compliance**: Follow Python Enhancement Proposal 8 for style
- **Line Length**: Maximum 100 characters (slightly longer than PEP 8 for readability)
- **Imports**: Organize imports in three groups (standard library, third-party, local)
- **Naming Conventions**: Use descriptive names that reflect VNE terminology

#### Type Hints
All public functions and methods must include comprehensive type hints:

```python
from typing import Dict, List, Optional, Tuple, Union
from src.models.virtual_request import VirtualNetworkRequest
from src.models.substrate import SubstrateNetwork

def embed_vnr(self, 
              vnr: VirtualNetworkRequest,
              substrate: SubstrateNetwork) -> EmbeddingResult:
    """Embed a VNR onto substrate network with proper type annotations."""
    pass

# Complex type annotations
def process_batch(self, 
                  vnrs: List[VirtualNetworkRequest],
                  config: Optional[Dict[str, Any]] = None) -> Tuple[List[EmbeddingResult], Dict[str, float]]:
    """Process VNR batch with comprehensive type information."""
    pass
```

#### Documentation Strings
Use Google-style docstrings with complete parameter and return documentation:

```python
def calculate_acceptance_ratio(results: List[EmbeddingResult]) -> float:
    """
    Calculate the acceptance ratio of VNR embedding attempts.

    Implements the standard VNE literature formula:
    AR = |Successfully_embedded_VNRs| / |Total_VNRs|

    Args:
        results: List of embedding results from algorithm execution.
                Each result contains success status and performance metrics.

    Returns:
        Acceptance ratio as float between 0.0 and 1.0, where 1.0 indicates
        all VNRs were successfully embedded.

    Raises:
        ValueError: If results list is empty or contains invalid data.

    Example:
        >>> results = [EmbeddingResult("1", True, ...), EmbeddingResult("2", False, ...)]
        >>> ratio = calculate_acceptance_ratio(results)
        >>> print(f"Acceptance ratio: {ratio:.2%}")
        Acceptance ratio: 50.00%
    """
```

#### Error Handling
Follow the framework's exception hierarchy and provide meaningful error messages:

```python
from src.models.substrate import ResourceAllocationError
from src.algorithms.base_algorithm import VNEConstraintError

def allocate_resources(self, node_id: int, cpu: float) -> bool:
    """Allocate resources with proper error handling."""
    try:
        if cpu <= 0:
            raise ValueError(f"CPU requirement must be positive, got {cpu}")
        
        if node_id not in self.graph.nodes:
            raise ResourceAllocationError(f"Node {node_id} does not exist in substrate")
        
        # Allocation logic here
        return True
        
    except Exception as e:
        self.logger.error(f"Resource allocation failed for node {node_id}: {e}")
        raise
```

#### Logging
Use structured logging throughout the codebase:

```python
import logging

class MyAlgorithm:
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def embed_vnr(self, vnr, substrate):
        self.logger.info(f"Starting embedding for VNR {vnr.vnr_id}")
        self.logger.debug(f"VNR details: {len(vnr.virtual_nodes)} nodes, {len(vnr.virtual_links)} links")
        
        try:
            # Algorithm implementation
            result = self._perform_embedding(vnr, substrate)
            self.logger.info(f"Embedding {'successful' if result.success else 'failed'}")
            return result
        except Exception as e:
            self.logger.error(f"Embedding failed with exception: {e}")
            raise
```

### Code Formatting

#### Automatic Formatting with Black
```bash
# Format all code
black src/ cli/ tests/

# Check formatting without changes
black --check src/ cli/ tests/

# Format specific files
black src/algorithms/baseline/yu_2008_algorithm.py
```

#### Import Organization
```python
# Standard library imports
import logging
import time
from typing import Dict, List, Optional

# Third-party imports
import networkx as nx
import numpy as np

# Local imports
from src.models.substrate import SubstrateNetwork
from src.models.virtual_request import VirtualNetworkRequest
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
```

## Testing Guidelines

The VNE Heuristics framework uses comprehensive testing to ensure reliability and correctness of VNE algorithms and supporting infrastructure.

### Test Structure and Organization

```
tests/
├── test_models/                    # Core model testing
│   ├── test_substrate.py          # SubstrateNetwork tests
│   ├── test_virtual_request.py    # VirtualNetworkRequest tests
│   └── models_test_suite.py       # Comprehensive model tests
├── test_algorithms/                # Algorithm testing
│   ├── test_base_algorithm.py     # BaseAlgorithm framework tests
│   ├── test_yu_algorithm.py       # Yu 2008 algorithm tests
│   └── vne_algorithm_unit_tests.py # Complete algorithm test suite
├── test_utils/                     # Utility module testing
│   ├── test_generators.py         # Network/VNR generation tests
│   ├── test_metrics.py            # Metrics calculation tests
│   ├── metrics_test_suite.py      # Comprehensive metrics tests
│   └── experiment_io_tests.py     # I/O and experiment tests
├── test_cli/                       # CLI interface testing
│   └── test_cli_commands.py       # Command-line interface tests
└── integration/                    # Integration and end-to-end tests
    ├── test_full_workflow.py      # Complete workflow tests
    └── test_algorithm_comparison.py # Algorithm comparison tests
```

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models/ -v           # Model tests only
python -m pytest tests/test_algorithms/ -v      # Algorithm tests only
python -m pytest tests/test_utils/ -v           # Utility tests only

# Run specific test files
python -m pytest tests/test_algorithms/test_yu_algorithm.py -v

# Run with coverage reporting
python -m pytest tests/ --cov=src --cov-report=html
```

#### Test Suite Runners
```bash
# Run comprehensive test suites (these are the main test files)
python tests/test_models/models_test_suite.py
python tests/test_algorithms/vne_algorithm_unit_tests.py
python tests/test_utils/metrics_test_suite.py
python tests/test_cli/test_cli_commands.py
```

### Writing Tests

#### Test Classes and Structure
```python
import unittest
from src.models.substrate import SubstrateNetwork
from src.models.virtual_request import VirtualNetworkRequest

class TestNewAlgorithm(unittest.TestCase):
    """Test cases for NewAlgorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create standard test substrate
        self.substrate = SubstrateNetwork()
        for i in range(10):
            self.substrate.add_node(i, cpu_capacity=100.0, memory_capacity=200.0)
        for i in range(9):
            self.substrate.add_link(i, i+1, bandwidth_capacity=100.0)
        
        # Create standard test VNR
        self.vnr = VirtualNetworkRequest(vnr_id=1, arrival_time=0.0, holding_time=100.0)
        self.vnr.add_virtual_node(0, cpu_requirement=20.0, memory_requirement=40.0)
        self.vnr.add_virtual_node(1, cpu_requirement=30.0, memory_requirement=60.0)
        self.vnr.add_virtual_link(0, 1, bandwidth_requirement=50.0)
        
        # Create algorithm instance
        self.algorithm = NewAlgorithm()
    
    def test_successful_embedding(self):
        """Test successful VNR embedding scenario."""
        result = self.algorithm.embed_vnr(self.vnr, self.substrate)
        
        # Verify embedding success
        self.assertTrue(result.success)
        self.assertEqual(len(result.node_mapping), 2)
        self.assertEqual(len(result.link_mapping), 1)
        
        # Verify resource allocation
        total_cpu_used = sum(
            self.substrate.get_node_resources(i).cpu_used 
            for i in range(10)
        )
        self.assertEqual(total_cpu_used, 50.0)  # 20 + 30
    
    def test_resource_constraint_validation(self):
        """Test that resource constraints are properly enforced."""
        # Create VNR with excessive requirements
        large_vnr = VirtualNetworkRequest(vnr_id=2)
        large_vnr.add_virtual_node(0, cpu_requirement=150.0)  # Exceeds capacity
        
        result = self.algorithm.embed_vnr(large_vnr, self.substrate)
        self.assertFalse(result.success)
        self.assertIn("insufficient", result.failure_reason.lower())
    
    def test_intra_vnr_separation(self):
        """Test that Intra-VNR separation constraint is enforced."""
        result = self.algorithm.embed_vnr(self.vnr, self.substrate)
        
        if result.success:
            # Verify no two virtual nodes map to same substrate node
            mapped_nodes = set(result.node_mapping.values())
            self.assertEqual(len(mapped_nodes), len(result.node_mapping))
```

#### Test Data and Fixtures
```python
# Create reusable test fixtures
def create_test_substrate(nodes=10, links_per_node=2):
    """Create standardized test substrate network."""
    substrate = SubstrateNetwork()
    
    # Add nodes with varied capacities
    for i in range(nodes):
        cpu_cap = 50 + (i % 3) * 25  # 50, 75, 100 pattern
        memory_cap = cpu_cap * 2
        substrate.add_node(i, cpu_capacity=cpu_cap, memory_capacity=memory_cap)
    
    # Add links in ring + random topology
    for i in range(nodes):
        # Ring connectivity
        substrate.add_link(i, (i + 1) % nodes, bandwidth_capacity=100)
        
        # Additional random links
        for j in range(links_per_node - 1):
            target = (i + 2 + j) % nodes
            if not substrate.graph.has_edge(i, target):
                substrate.add_link(i, target, bandwidth_capacity=75)
    
    return substrate
```

### Test Quality Requirements

#### Coverage Requirements
- **Minimum Coverage**: 85% line coverage for all modules
- **Critical Paths**: 100% coverage for resource allocation and constraint validation
- **Algorithm Tests**: Must test both successful and failed embedding scenarios
- **Edge Cases**: Test boundary conditions and error scenarios

#### Performance Testing
```python
import time
import pytest

def test_algorithm_performance():
    """Test algorithm performance meets acceptable thresholds."""
    substrate = create_test_substrate(nodes=50)
    vnrs = [create_test_vnr(nodes=3) for _ in range(100)]
    
    algorithm = YuAlgorithm()
    
    start_time = time.time()
    results = algorithm.embed_batch(vnrs, substrate)
    execution_time = time.time() - start_time
    
    # Performance requirements
    assert execution_time < 10.0  # Should complete within 10 seconds
    assert len(results) == 100   # Should process all VNRs
    
    # Quality requirements
    acceptance_ratio = sum(1 for r in results if r.success) / len(results)
    assert acceptance_ratio > 0.3  # Minimum 30% acceptance ratio
```

## Adding New Algorithms

The framework provides a structured approach for implementing new VNE algorithms while maintaining consistency and quality.

### Algorithm Development Process

#### 1. Literature Review and Planning
Before implementing, thoroughly understand the algorithm:
- Read the original research paper
- Understand the theoretical foundation and assumptions
- Identify computational complexity and expected performance
- Determine constraint requirements and limitations

#### 2. Algorithm Design
Plan the implementation approach:
- Decompose the algorithm into logical phases
- Identify data structures and state management needs
- Plan resource allocation and rollback strategies
- Design configuration parameters and tuning options

#### 3. Implementation Structure

Create your algorithm by inheriting from `BaseAlgorithm`:

```python
from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult
from typing import Dict, List, Optional
import logging

class NewAlgorithm(BaseAlgorithm):
    """
    Implementation of [Algorithm Name] from [Citation].
    
    This algorithm implements [brief description of approach] for solving
    the Virtual Network Embedding problem.
    
    Reference:
        [Author], [Year]. "[Paper Title]". [Journal/Conference].
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
    
    Example:
        >>> algorithm = NewAlgorithm(param1=2.0, param2="strategy")
        >>> result = algorithm.embed_vnr(vnr, substrate)
    """
    
    def __init__(self, param1: float = 1.0, param2: str = "default", **kwargs):
        super().__init__("New Algorithm Name", **kwargs)
        
        # Store algorithm parameters
        self.param1 = param1
        self.param2 = param2
        
        # Initialize algorithm state
        self._internal_state = {}
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info(f"Initialized {self.name} with param1={param1}, param2={param2}")
    
    def _validate_parameters(self):
        """Validate algorithm parameters."""
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")
        
        valid_strategies = ["default", "alternative"]
        if self.param2 not in valid_strategies:
            raise ValueError(f"param2 must be one of {valid_strategies}")
    
    def _embed_single_vnr(self, vnr, substrate) -> EmbeddingResult:
        """
        Core algorithm implementation with resource allocation.
        
        This method must:
        1. Implement the algorithm's node and link mapping logic
        2. Allocate resources during the embedding process
        3. Handle failures gracefully with proper rollback
        4. Return a complete EmbeddingResult
        
        Args:
            vnr: Virtual network request to embed
            substrate: Substrate network to embed onto
            
        Returns:
            EmbeddingResult with success status and mappings
        """
        self.logger.debug(f"Starting embedding for VNR {vnr.vnr_id}")
        
        try:
            # Phase 1: Node Mapping
            node_mapping, allocated_nodes = self._perform_node_mapping(vnr, substrate)
            
            if not node_mapping:
                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id), success=False,
                    node_mapping={}, link_mapping={},
                    revenue=0.0, cost=0.0, execution_time=0.0,
                    failure_reason="Node mapping failed"
                )
            
            # Phase 2: Link Mapping
            link_mapping, allocated_links = self._perform_link_mapping(
                vnr, substrate, node_mapping
            )
            
            if not link_mapping and vnr.virtual_links:
                # Rollback node allocations on link mapping failure
                self._rollback_node_allocations(substrate, allocated_nodes)
                return EmbeddingResult(
                    vnr_id=str(vnr.vnr_id), success=False,
                    node_mapping={}, link_mapping={},
                    revenue=0.0, cost=0.0, execution_time=0.0,
                    failure_reason="Link mapping failed"
                )
            
            # Success
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id), success=True,
                node_mapping=node_mapping, link_mapping=link_mapping,
                revenue=0.0, cost=0.0, execution_time=0.0,  # Calculated by base class
                metadata=self._get_algorithm_metadata()
            )
            
        except Exception as e:
            self.logger.error(f"Exception during embedding: {e}")
            return EmbeddingResult(
                vnr_id=str(vnr.vnr_id), success=False,
                node_mapping={}, link_mapping={},
                revenue=0.0, cost=0.0, execution_time=0.0,
                failure_reason=f"Algorithm exception: {str(e)}"
            )
    
    def _cleanup_failed_embedding(self, vnr, substrate, result):
        """
        Clean up resources for failed embedding.
        
        This method is called by the base class when VNE constraints are
        violated after successful algorithm execution. It must deallocate
        all resources that were allocated during _embed_single_vnr().
        """
        self.logger.debug(f"Cleaning up failed embedding for VNR {vnr.vnr_id}")
        
        try:
            # Deallocate node resources
            for vnode_id, vnode_req in vnr.virtual_nodes.items():
                if str(vnode_id) in result.node_mapping:
                    snode_id = int(result.node_mapping[str(vnode_id)])
                    substrate.deallocate_node_resources(
                        snode_id, vnode_req.cpu_requirement, vnode_req.memory_requirement
                    )
            
            # Deallocate link resources
            for (vsrc, vdst), vlink_req in vnr.virtual_links.items():
                if (str(vsrc), str(vdst)) in result.link_mapping:
                    path = [int(n) for n in result.link_mapping[(str(vsrc), str(vdst))]]
                    for i in range(len(path) - 1):
                        substrate.deallocate_link_resources(
                            path[i], path[i + 1], vlink_req.bandwidth_requirement
                        )
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Helper methods (implement according to your algorithm)
    def _perform_node_mapping(self, vnr, substrate):
        """Implement algorithm-specific node mapping logic."""
        # Your node mapping implementation here
        pass
    
    def _perform_link_mapping(self, vnr, substrate, node_mapping):
        """Implement algorithm-specific link mapping logic."""
        # Your link mapping implementation here
        pass
```

#### 4. Testing Your Algorithm

Create comprehensive tests for your algorithm:

```python
# tests/test_algorithms/test_new_algorithm.py
import unittest
from src.algorithms.new_algorithm import NewAlgorithm
from tests.test_algorithms.test_base_algorithm import TestVNEAlgorithmBase

class TestNewAlgorithm(TestVNEAlgorithmBase):
    """Test cases for NewAlgorithm implementation."""
    
    def setUp(self):
        super().setUp()
        self.algorithm = NewAlgorithm(param1=2.0, param2="default")
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization and parameters."""
        self.assertEqual(self.algorithm.name, "New Algorithm Name")
        self.assertEqual(self.algorithm.param1, 2.0)
        self.assertEqual(self.algorithm.param2, "default")
    
    def test_successful_embedding(self):
        """Test successful embedding scenario."""
        result = self.algorithm.embed_vnr(self.simple_vnr, self.substrate)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.node_mapping), 2)
        
        # Verify Intra-VNR separation
        mapped_nodes = set(result.node_mapping.values())
        self.assertEqual(len(mapped_nodes), len(result.node_mapping))
    
    def test_algorithm_specific_features(self):
        """Test algorithm-specific functionality."""
        # Test algorithm-specific behavior
        pass
```

#### 5. Documentation Requirements

Add comprehensive documentation:

```python
"""
New Algorithm Implementation for VNE

This module implements the [Algorithm Name] approach for Virtual Network
Embedding as described in [Citation].

Key Features:
- [Feature 1]: Description
- [Feature 2]: Description
- [Constraint Support]: Which constraints are supported

Performance Characteristics:
- Time Complexity: O(...)
- Space Complexity: O(...)
- Typical Acceptance Ratio: X-Y% on standard benchmarks

Example Usage:
    >>> from src.algorithms.new_algorithm import NewAlgorithm
    >>> algorithm = NewAlgorithm(param1=2.0)
    >>> result = algorithm.embed_vnr(vnr, substrate)
    >>> print(f"Success: {result.success}")

Literature Reference:
    [Author], [Year]. "[Title]". [Journal/Conference], [Volume]([Issue]), [Pages].
    DOI: [DOI if available]
"""
```

### Algorithm Integration Checklist

Before submitting your algorithm implementation:

- [ ] **Implementation Complete**: All abstract methods implemented
- [ ] **Resource Management**: Proper allocation and rollback mechanisms
- [ ] **Error Handling**: Comprehensive exception handling
- [ ] **Testing**: Full test suite with >90% coverage
- [ ] **Documentation**: Complete docstrings and usage examples
- [ ] **Performance**: Meets reasonable performance benchmarks
- [ ] **Code Quality**: Passes all code quality checks
- [ ] **Literature Compliance**: Follows original algorithm specification

## Documentation Standards

### Code Documentation

#### Docstring Requirements
All public classes, methods, and functions must have comprehensive docstrings:

```python
def calculate_vnr_revenue(vnr: VirtualNetworkRequest) -> float:
    """
    Calculate revenue for a VNR using standard VNE literature formula.

    Implements the widely-used revenue calculation from VNE research:
    Revenue = Σ(CPU_requirements) + Σ(Bandwidth_requirements) + Σ(Memory_requirements)

    The calculation includes primary constraints (CPU, bandwidth) and optional
    secondary constraints (memory) based on the VNR's constraint usage.

    Args:
        vnr: Virtual network request with resource requirements.
             Must contain at least one virtual node.

    Returns:
        Total revenue as positive float. Returns 0.0 for empty VNRs.

    Raises:
        ValueError: If VNR is None or contains invalid resource requirements.

    Example:
        >>> vnr = VirtualNetworkRequest(vnr_id=1)
        >>> vnr.add_virtual_node(0, cpu_requirement=50.0)
        >>> vnr.add_virtual_link(0, 1, bandwidth_requirement=100.0)
        >>> revenue = calculate_vnr_revenue(vnr)
        >>> print(f"VNR revenue: {revenue}")
        VNR revenue: 150.0

    Literature Reference:
        Standard formula used across VNE literature, including:
        Yu et al. (2008), Fischer et al. (2013), and others.
    """
```

#### README and Guide Updates
When adding new features, update relevant documentation:
- Add algorithm to README.md algorithm list
- Update ALGORITHMS.md with detailed algorithm description
- Add usage examples to appropriate sections
- Update API.md if new public interfaces are added

### Commit Message Standards

Use clear, descriptive commit messages following this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New features or algorithms
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring without feature changes
- `perf`: Performance improvements
- `style`: Code style/formatting changes

**Examples**:
```
feat(algorithms): implement NSGA-II multi-objective VNE algorithm

Add comprehensive implementation of NSGA-II for multi-objective VNE
optimization. Includes Pareto ranking, crowding distance calculation,
and standard NSGA-II selection mechanisms.

- Supports multiple objectives (acceptance ratio, cost, revenue)
- Includes extensive test suite with benchmark comparisons
- Documentation updated with algorithm details and usage examples

Closes #123

fix(substrate): resolve thread safety issue in resource allocation

Fixed race condition in SubstrateNetwork.allocate_node_resources()
that could cause resource over-allocation under concurrent access.

- Added proper locking mechanism
- Updated tests to verify thread safety
- Performance impact minimal (<5% overhead)

Fixes #456

docs(api): update BaseAlgorithm documentation with new examples

Added comprehensive usage examples for algorithm development and
updated parameter documentation for clarity.
```

## Pull Request Process

### Before Submitting

1. **Code Quality Checks**
   ```bash
   # Run formatting
   black src/ cli/ tests/
   
   # Run linting
   flake8 src/ cli/ tests/
   
   # Run type checking
   mypy src/ cli/
   
   # Run tests
   python -m pytest tests/ -v --cov=src
   ```

2. **Documentation Updates**
   - Update relevant documentation files
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md with your changes

3. **Test Coverage**
   - Ensure >85% test coverage for new code
   - Add tests for new features and algorithms
   - Verify existing tests still pass

### Pull Request Template

Use this template for your pull request description:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] New algorithm implementation
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- [ ] List specific changes made
- [ ] Include any new dependencies
- [ ] Note any configuration changes

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Test coverage maintained/improved

## Algorithm Implementation (if applicable)
- [ ] Literature reference provided
- [ ] Algorithm complexity documented
- [ ] Performance benchmarks included
- [ ] Comparison with existing algorithms

## Documentation
- [ ] Code documentation updated
- [ ] README.md updated (if needed)
- [ ] API.md updated (if needed)
- [ ] ALGORITHMS.md updated (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex algorithms
- [ ] Documentation reflects changes
- [ ] Tests added/updated and passing
```

### Review Process

1. **Automated Checks**: GitHub Actions will run tests and quality checks
2. **Code Review**: Project maintainers will review for:
   - Code quality and adherence to standards
   - Algorithm correctness and efficiency
   - Test coverage and quality
   - Documentation completeness
3. **Feedback Incorporation**: Address review comments
4. **Final Approval**: Maintainer approval required for merge

## Project Structure

Understanding the project structure helps contributors know where to make changes:

```
vne-heuristics/
├── src/                           # Source code
│   ├── algorithms/                # Algorithm implementations
│   │   ├── base_algorithm.py      # Abstract base class
│   │   └── baseline/              # Baseline algorithms
│   │       └── yu_2008_algorithm.py
│   ├── models/                    # Core data models
│   │   ├── substrate.py           # Physical network representation
│   │   ├── virtual_request.py     # VNR representation
│   │   └── vnr_batch.py           # VNR batch management
│   └── utils/                     # Utility modules
│       ├── generators/            # Network/VNR generators
│       ├── metrics.py             # Performance metrics
│       └── io_utils.py            # File I/O operations
│
├── cli/                           # Command-line interface
│   ├── argument_parser.py         # CLI argument parsing
│   ├── commands/                  # Command implementations
│   └── exceptions.py              # CLI exception classes
│
├── core/                          # Core framework components
│   ├── algorithm_registry.py      # Algorithm discovery
│   ├── error_handler.py           # Error handling
│   └── progress_reporter.py       # Progress reporting
│
├── tests/                         # Test suite
│   ├── test_models/               # Model tests
│   ├── test_algorithms/           # Algorithm tests
│   ├── test_utils/                # Utility tests
│   └── test_cli/                  # CLI tests
│
├── docs/                          # Documentation
├── data/                          # Data directory (gitignored)
├── main.py                        # CLI entry point
├── config_management.py           # Configuration system
└── requirements.txt               # Dependencies
```

### Where to Make Changes

- **New algorithms**: `src/algorithms/` (create appropriate subdirectory)
- **Core model improvements**: `src/models/`
- **New metrics or utilities**: `src/utils/`
- **CLI enhancements**: `cli/commands/`
- **Tests**: `tests/` (mirror source structure)
- **Documentation**: `docs/` and relevant `.md` files

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS]
- Python version: [e.g., 3.8.5]
- Framework version: [e.g., 1.0.0]

**Additional Context**
- Log output (if applicable)
- Network sizes and configurations
- Algorithm used
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why this feature would be valuable.

**Proposed Implementation**
How you think this should be implemented.

**Alternatives Considered**
Alternative approaches you've considered.

**Literature Reference**
Any relevant research papers or algorithms.
```

### Algorithm Requests

For requesting new algorithm implementations:

```markdown
**Algorithm Details**
- Name: [Algorithm name]
- Paper: [Full citation]
- DOI/Link: [If available]

**Algorithm Description**
Brief description of the algorithm approach.

**Expected Benefits**
Why this algorithm would be valuable for the framework.

**Implementation Complexity**
Your assessment of implementation difficulty.

**Volunteer Implementation**
[ ] I would like to implement this algorithm myself
[ ] I need someone else to implement this
```

---

Thank you for contributing to the VNE Heuristics framework! Your contributions help advance Virtual Network Embedding research and provide valuable tools for the research community.