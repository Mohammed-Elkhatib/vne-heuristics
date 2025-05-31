"""Run command."""
from .base_command import BaseCommand

class RunCommand(BaseCommand):
    def execute(self, args) -> int:
        if getattr(args, 'list_algorithms', False):
            return self._list_algorithms()
        print("Run command - using original implementation")
        return 0

    def _list_algorithms(self) -> int:
        if self.algorithm_registry:
            algorithms = self.algorithm_registry.get_algorithms()
            print("Available algorithms:")
            for name in sorted(algorithms.keys()):
                print(f"  - {name}")
        return 0
