"""
Dynamic algorithm discovery and registration system for VNE algorithms.
Fixed version with proper error handling for packages without __file__.
"""

import logging
import importlib
import inspect
from typing import Dict, Type, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """Registry for VNE algorithms with dynamic discovery capabilities."""

    def __init__(self):
        """Initialize the algorithm registry."""
        self._algorithms: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

        # Perform initial discovery
        self._discover_algorithms()

    def _discover_algorithms(self) -> None:
        """Automatically discover and register available algorithms."""
        self.logger.debug("Starting algorithm discovery")

        # Standard algorithm locations
        algorithm_packages = [
            'src.algorithms.baseline',
            'src.algorithms.heuristic',
            'src.algorithms.metaheuristic',
        ]

        discovered_count = 0

        for package_name in algorithm_packages:
            try:
                discovered_count += self._discover_package_algorithms(package_name)
            except ImportError:
                self.logger.debug(f"Package {package_name} not found")
            except Exception as e:
                self.logger.warning(f"Error discovering algorithms in {package_name}: {e}")

        # Fallback: Discover known algorithms directly
        if discovered_count == 0:
            self._discover_fallback_algorithms()

        self.logger.info(f"Discovered {len(self._algorithms)} algorithms")

    def _discover_package_algorithms(self, package_name: str) -> int:
        """Discover algorithms in a specific package."""
        try:
            package = importlib.import_module(package_name)

            # Check if package has a valid __file__ attribute
            if not hasattr(package, '__file__') or package.__file__ is None:
                self.logger.debug(f"Package {package_name} has no __file__ attribute, skipping directory scan")
                return 0

            package_path = Path(package.__file__).parent

            # Verify the path exists
            if not package_path.exists():
                self.logger.debug(f"Package path {package_path} does not exist")
                return 0

            discovered = 0
            for py_file in package_path.glob("*.py"):
                if py_file.name.startswith('_'):
                    continue

                module_name = f"{package_name}.{py_file.stem}"
                discovered += self._discover_module_algorithms(module_name)

            return discovered

        except ImportError:
            self.logger.debug(f"Could not import package {package_name}")
            return 0
        except Exception as e:
            self.logger.debug(f"Error discovering package {package_name}: {e}")
            return 0

    def _discover_module_algorithms(self, module_name: str) -> int:
        """Discover algorithms in a specific module."""
        try:
            module = importlib.import_module(module_name)
            discovered = 0

            for name, obj in inspect.getmembers(module):
                if self._is_algorithm_class(obj):
                    algorithm_name = self._extract_algorithm_name(name, obj)
                    self._register_discovered_algorithm(algorithm_name, obj, module_name)
                    discovered += 1

            return discovered

        except ImportError as e:
            self.logger.debug(f"Could not import module {module_name}: {e}")
            return 0
        except Exception as e:
            self.logger.debug(f"Error discovering algorithms in {module_name}: {e}")
            return 0

    def _discover_fallback_algorithms(self) -> None:
        """Fallback discovery for known algorithms."""
        fallback_algorithms = [
            ('yu2008', 'src.algorithms.baseline.yu_2008_algorithm', 'YuAlgorithm'),
        ]

        for name, module_name, class_name in fallback_algorithms:
            try:
                module = importlib.import_module(module_name)
                algorithm_class = getattr(module, class_name)

                if self._is_algorithm_class(algorithm_class):
                    self._register_discovered_algorithm(name, algorithm_class, module_name)
                    self.logger.debug(f"Fallback discovery successful for {name}")

            except (ImportError, AttributeError) as e:
                self.logger.debug(f"Could not load fallback algorithm {name}: {e}")

    def _is_algorithm_class(self, obj) -> bool:
        """Check if an object is a valid algorithm class."""
        try:
            from src.algorithms.base_algorithm import BaseAlgorithm
            return (
                inspect.isclass(obj) and
                issubclass(obj, BaseAlgorithm) and
                obj is not BaseAlgorithm and
                not inspect.isabstract(obj)
            )
        except ImportError:
            return False

    def _extract_algorithm_name(self, class_name: str, algorithm_class: Type) -> str:
        """Extract algorithm name from class."""
        # Try to get name from class attribute
        if hasattr(algorithm_class, 'ALGORITHM_NAME'):
            return algorithm_class.ALGORITHM_NAME

        # Convert class name to algorithm name
        name = class_name.lower()
        if name.endswith('algorithm'):
            name = name[:-9]  # Remove 'algorithm' suffix
        return name

    def _register_discovered_algorithm(self, name: str, algorithm_class: Type, module_name: str) -> None:
        """Register a discovered algorithm."""
        self._algorithms[name] = algorithm_class
        self._metadata[name] = {
            'class_name': algorithm_class.__name__,
            'module': module_name,
            'discovery_method': 'automatic'
        }

        self.logger.debug(f"Registered algorithm: {name} ({algorithm_class.__name__})")

    def get_algorithms(self) -> Dict[str, Type]:
        """Get all registered algorithms."""
        return self._algorithms.copy()

    def get_algorithm(self, name: str) -> Optional[Type]:
        """Get a specific algorithm by name."""
        return self._algorithms.get(name)

    def list_algorithms(self) -> List[str]:
        """Get list of algorithm names."""
        return list(self._algorithms.keys())

    def is_available(self, name: str) -> bool:
        """Check if an algorithm is available."""
        return name in self._algorithms
