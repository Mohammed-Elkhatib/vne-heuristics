"""
Algorithm discovery with robust error handling.

This module provides automatic discovery and registration of VNE algorithms
from the algorithms package, with fallback mechanisms for known algorithms.
"""

import logging
import importlib
import inspect
from typing import Dict, Type, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AlgorithmRegistryError(Exception):
    """Exception raised for algorithm registry errors."""
    pass


class AlgorithmRegistry:
    """Enhanced registry for VNE algorithms with robust discovery."""

    def __init__(self):
        """Initialize the algorithm registry with enhanced error handling."""
        self._algorithms: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict] = {}
        self._discovery_errors: List[str] = []
        self.logger = logging.getLogger(__name__)

        # Perform initial discovery with error tracking
        self._discover_algorithms()

    def _discover_algorithms(self) -> None:
        """Automatically discover and register available algorithms with error tracking."""
        self.logger.debug("Starting enhanced algorithm discovery")

        # Standard algorithm locations with fallback support
        algorithm_packages = [
            'src.algorithms.baseline',
            'src.algorithms.heuristic',
            'src.algorithms.metaheuristic',
            'src.algorithms',  # Fallback to main algorithms package
        ]

        discovered_count = 0
        for package_name in algorithm_packages:
            try:
                count = self._discover_package_algorithms(package_name)
                discovered_count += count
                if count > 0:
                    self.logger.debug(f"Discovered {count} algorithms in {package_name}")
            except ImportError as e:
                self._discovery_errors.append(f"Package {package_name} not found: {e}")
                self.logger.debug(f"Package {package_name} not available")
            except Exception as e:
                self._discovery_errors.append(f"Error in package {package_name}: {e}")
                self.logger.warning(f"Error discovering algorithms in {package_name}: {e}")

        # Fallback: Try known algorithms directly
        if discovered_count == 0:
            self.logger.info("No algorithms found via package discovery, trying fallback discovery")
            discovered_count = self._discover_fallback_algorithms()

        if discovered_count == 0:
            self.logger.warning("No algorithms discovered. Check if algorithm modules are properly installed.")
            # Register a dummy algorithm for testing
            self._register_dummy_algorithm()
        else:
            self.logger.info(f"Algorithm discovery completed: {discovered_count} algorithms registered")

    def _discover_package_algorithms(self, package_name: str) -> int:
        """Discover algorithms in a specific package with enhanced error handling."""
        try:
            package = importlib.import_module(package_name)
            discovered = 0

            # Try to get package path
            package_path = None
            if hasattr(package, '__file__') and package.__file__:
                package_path = Path(package.__file__).parent
            elif hasattr(package, '__path__'):
                # Handle namespace packages
                package_path = Path(next(iter(package.__path__)))

            if package_path and package_path.exists():
                # Scan directory for Python files
                for py_file in package_path.glob("*.py"):
                    if py_file.name.startswith('_'):
                        continue

                    module_name = f"{package_name}.{py_file.stem}"
                    try:
                        discovered += self._discover_module_algorithms(module_name)
                    except Exception as e:
                        self.logger.debug(f"Could not discover algorithms in {module_name}: {e}")
            else:
                # Try to discover from module attributes directly
                discovered = self._discover_module_algorithms(package_name)

            return discovered

        except ImportError:
            raise  # Re-raise for caller to handle
        except Exception as e:
            self.logger.debug(f"Error discovering package {package_name}: {e}")
            return 0

    def _discover_module_algorithms(self, module_name: str) -> int:
        """Discover algorithms in a specific module with enhanced error handling."""
        try:
            module = importlib.import_module(module_name)
            discovered = 0

            for name, obj in inspect.getmembers(module):
                try:
                    if self._is_algorithm_class(obj):
                        algorithm_name = self._extract_algorithm_name(name, obj)
                        self._register_discovered_algorithm(algorithm_name, obj, module_name)
                        discovered += 1
                except Exception as e:
                    self.logger.debug(f"Could not register {name} from {module_name}: {e}")

            return discovered

        except ImportError as e:
            self.logger.debug(f"Could not import module {module_name}: {e}")
            return 0
        except Exception as e:
            self.logger.debug(f"Error discovering algorithms in {module_name}: {e}")
            return 0

    def _discover_fallback_algorithms(self) -> int:
        """Fallback discovery for known algorithms with enhanced error handling."""
        fallback_algorithms = [
            ('yu2008', 'src.algorithms.baseline.yu_2008_algorithm', 'YuAlgorithm'),
            ('greedy', 'src.algorithms.heuristic.greedy_algorithm', 'GreedyAlgorithm'),
            ('random', 'src.algorithms.baseline.random_algorithm', 'RandomAlgorithm'),
        ]

        discovered = 0
        for name, module_name, class_name in fallback_algorithms:
            try:
                module = importlib.import_module(module_name)
                algorithm_class = getattr(module, class_name)

                if self._is_algorithm_class(algorithm_class):
                    self._register_discovered_algorithm(name, algorithm_class, module_name)
                    discovered += 1
                    self.logger.debug(f"Fallback discovery successful for {name}")

            except (ImportError, AttributeError) as e:
                self.logger.debug(f"Could not load fallback algorithm {name}: {e}")
            except Exception as e:
                self.logger.debug(f"Error loading fallback algorithm {name}: {e}")

        return discovered

    def _register_dummy_algorithm(self) -> None:
        """Register a dummy algorithm for testing when no real algorithms are found."""
        try:
            # Create a minimal dummy algorithm class
            from src.algorithms.base_algorithm import BaseAlgorithm, EmbeddingResult

            class DummyAlgorithm(BaseAlgorithm):
                def __init__(self):
                    super().__init__("Dummy Algorithm (Testing)")

                def _embed_single_vnr(self, vnr, substrate):
                    return EmbeddingResult(
                        vnr_id=str(vnr.vnr_id),
                        success=False,
                        node_mapping={},
                        link_mapping={},
                        revenue=0.0,
                        cost=0.0,
                        execution_time=0.0,
                        failure_reason="Dummy algorithm - always fails"
                    )

                def _cleanup_failed_embedding(self, vnr, substrate, result):
                    pass  # No cleanup needed for dummy

            self._algorithms['dummy'] = DummyAlgorithm
            self._metadata['dummy'] = {
                'class_name': 'DummyAlgorithm',
                'module': 'built-in',
                'discovery_method': 'fallback_dummy',
                'description': 'Dummy algorithm for testing when no real algorithms are available'
            }

            self.logger.info("Registered dummy algorithm for testing")

        except Exception as e:
            self.logger.warning(f"Could not register dummy algorithm: {e}")

    def _is_algorithm_class(self, obj) -> bool:
        """Check if an object is a valid algorithm class with enhanced validation."""
        try:
            from src.algorithms.base_algorithm import BaseAlgorithm

            return (
                inspect.isclass(obj) and
                issubclass(obj, BaseAlgorithm) and
                obj is not BaseAlgorithm and
                not inspect.isabstract(obj) and
                hasattr(obj, '_embed_single_vnr') and
                hasattr(obj, '_cleanup_failed_embedding')
            )
        except ImportError:
            self.logger.debug("BaseAlgorithm not available for validation")
            return False
        except Exception as e:
            self.logger.debug(f"Error validating algorithm class: {e}")
            return False

    def _extract_algorithm_name(self, class_name: str, algorithm_class: Type) -> str:
        """Extract algorithm name from class with enhanced name handling."""
        # Try to get name from class attribute first
        if hasattr(algorithm_class, 'ALGORITHM_NAME'):
            return algorithm_class.ALGORITHM_NAME

        if hasattr(algorithm_class, 'algorithm_name'):
            return algorithm_class.algorithm_name

        # Convert class name to algorithm name
        name = class_name.lower()

        # Remove common suffixes
        suffixes = ['algorithm', 'algo', 'embedding', 'embedder']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # Handle empty names
        if not name:
            name = class_name.lower()

        return name

    def _register_discovered_algorithm(self, name: str, algorithm_class: Type, module_name: str) -> None:
        """Register a discovered algorithm with enhanced metadata."""
        # Handle name conflicts
        if name in self._algorithms:
            original_name = name
            counter = 1
            while name in self._algorithms:
                name = f"{original_name}_{counter}"
                counter += 1
            self.logger.warning(f"Algorithm name conflict: renamed {original_name} to {name}")

        self._algorithms[name] = algorithm_class
        self._metadata[name] = {
            'class_name': algorithm_class.__name__,
            'module': module_name,
            'discovery_method': 'automatic',
            'description': getattr(algorithm_class, '__doc__', '').split('\n')[0] if algorithm_class.__doc__ else 'No description available'
        }

        self.logger.debug(f"Registered algorithm: {name} ({algorithm_class.__name__})")

    def get_algorithms(self) -> Dict[str, Type]:
        """Get all registered algorithms."""
        return self._algorithms.copy()

    def get_algorithm(self, name: str) -> Optional[Type]:
        """Get a specific algorithm by name with case-insensitive lookup."""
        # Try exact match first
        if name in self._algorithms:
            return self._algorithms[name]

        # Try case-insensitive match
        for algo_name, algo_class in self._algorithms.items():
            if algo_name.lower() == name.lower():
                return algo_class

        return None

    def list_algorithms(self) -> List[str]:
        """Get list of algorithm names."""
        return list(self._algorithms.keys())

    def is_available(self, name: str) -> bool:
        """Check if an algorithm is available with case-insensitive lookup."""
        return self.get_algorithm(name) is not None

    def get_algorithm_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a specific algorithm."""
        # Handle case-insensitive lookup
        for algo_name, metadata in self._metadata.items():
            if algo_name.lower() == name.lower():
                return metadata.copy()
        return None

    def get_discovery_errors(self) -> List[str]:
        """Get list of errors encountered during discovery."""
        return self._discovery_errors.copy()

    def refresh_algorithms(self) -> int:
        """Refresh algorithm discovery and return count of newly discovered algorithms."""
        old_count = len(self._algorithms)
        self._algorithms.clear()
        self._metadata.clear()
        self._discovery_errors.clear()

        self._discover_algorithms()

        new_count = len(self._algorithms)
        self.logger.info(f"Algorithm refresh completed: {new_count} algorithms (was {old_count})")
        return new_count
