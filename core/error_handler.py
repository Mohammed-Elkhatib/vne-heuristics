"""
Centralized error handling system for VNE CLI with context and guidance.
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Enhanced centralized error handling for VNE CLI with contextual guidance."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the error handler with debug mode support."""
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode

        # Error context for better user guidance
        self.error_contexts = {
            'configuration': {
                'common_fixes': [
                    "Check configuration file syntax (YAML/JSON)",
                    "Verify all required parameters are set",
                    "Ensure file paths exist and are accessible",
                    "Check parameter value ranges and types"
                ],
                'examples': [
                    "substrate_nodes: 50",
                    "vnr_count: 100",
                    "timeout_seconds: 300.0"
                ]
            },
            'file_operations': {
                'common_fixes': [
                    "Verify file paths are correct",
                    "Check file permissions (read/write access)",
                    "Ensure directories exist",
                    "Verify file formats (CSV structure, JSON syntax)"
                ],
                'examples': [
                    "data/substrate_20_nodes.csv",
                    "data/substrate_20_links.csv",
                    "results/algorithm_results.json"
                ]
            },
            'algorithm_execution': {
                'common_fixes': [
                    "Check if algorithm is properly installed",
                    "Verify substrate network and VNR compatibility",
                    "Ensure sufficient resources for embedding",
                    "Check algorithm-specific parameters"
                ],
                'examples': [
                    "--algorithm yu2008",
                    "--substrate data/substrate.csv",
                    "--vnrs data/vnr_batch.csv"
                ]
            }
        }

    def handle_configuration_error(self, error) -> None:
        """Handle configuration-related errors with detailed guidance."""
        self.logger.error(f"Configuration error: {error}")

        print(f"\n❌ Configuration Error: {error}", file=sys.stderr)
        self._print_contextual_help('configuration')

        if self.debug_mode:
            print(f"\nDebug trace:", file=sys.stderr)
            traceback.print_exc()

    def handle_file_error(self, error, filepath: Optional[str] = None) -> None:
        """Handle file operation errors with specific guidance."""
        self.logger.error(f"File error: {error}")

        print(f"\n❌ File Error: {error}", file=sys.stderr)

        if filepath:
            print(f"📄 Problem file: {filepath}", file=sys.stderr)
            self._analyze_file_issue(filepath)

        self._print_contextual_help('file_operations')

        if self.debug_mode:
            print(f"\nDebug trace:", file=sys.stderr)
            traceback.print_exc()

    def handle_algorithm_error(self, error, algorithm_name: Optional[str] = None,
                             details: Optional[str] = None) -> None:
        """Handle algorithm execution errors with specific guidance."""
        self.logger.error(f"Algorithm error: {error}")

        print(f"\n❌ Algorithm Error: {error}", file=sys.stderr)

        if algorithm_name:
            print(f"🔧 Algorithm: {algorithm_name}", file=sys.stderr)

        if details:
            print(f"📊 Details: {details}", file=sys.stderr)

        self._print_contextual_help('algorithm_execution')

        # Algorithm-specific guidance
        if algorithm_name:
            self._print_algorithm_specific_help(algorithm_name)

        if self.debug_mode:
            print(f"\nDebug trace:", file=sys.stderr)
            traceback.print_exc()

    def handle_cli_error(self, error) -> None:
        """Handle general CLI-specific errors."""
        self.logger.error(f"CLI error: {error}")

        print(f"\n❌ CLI Error: {error}", file=sys.stderr)
        print(f"💡 Try: python main.py --help for usage information", file=sys.stderr)

        if self.debug_mode:
            print(f"\nDebug trace:", file=sys.stderr)
            traceback.print_exc()

    def handle_validation_error(self, error, context: Optional[str] = None) -> None:
        """Handle validation errors with context-specific guidance."""
        self.logger.error(f"Validation error: {error}")

        print(f"\n❌ Validation Error: {error}", file=sys.stderr)

        if context:
            print(f"🎯 Context: {context}", file=sys.stderr)

        print(f"💡 Suggestions:", file=sys.stderr)
        print(f"  - Check input data format and ranges", file=sys.stderr)
        print(f"  - Verify required parameters are provided", file=sys.stderr)
        print(f"  - Ensure data consistency (node IDs, link references)", file=sys.stderr)

        if self.debug_mode:
            print(f"\nDebug trace:", file=sys.stderr)
            traceback.print_exc()

    def handle_user_interruption(self) -> None:
        """Handle user interruption with cleanup guidance."""
        self.logger.info("Operation cancelled by user")

        print(f"\n🛑 Operation cancelled by user", file=sys.stderr)
        print(f"💡 Note: Partial results may have been saved", file=sys.stderr)
        print(f"   Check output directories for intermediate files", file=sys.stderr)

    def handle_unexpected_error(self, error: Exception) -> None:
        """Handle unexpected errors with comprehensive guidance."""
        self.logger.error(f"Unexpected error: {error}", exc_info=True)

        print(f"\n💥 Unexpected Error: {error}", file=sys.stderr)
        print(f"🔍 This appears to be an internal error.", file=sys.stderr)

        # Provide general troubleshooting steps
        print(f"\n🛠️ Troubleshooting steps:", file=sys.stderr)
        print(f"  1. Check if all required files exist", file=sys.stderr)
        print(f"  2. Verify input data format", file=sys.stderr)
        print(f"  3. Try with smaller datasets", file=sys.stderr)
        print(f"  4. Run with --debug flag for more details", file=sys.stderr)
        print(f"  5. Check available memory and disk space", file=sys.stderr)

        # Always show full traceback for unexpected errors
        print(f"\nFull error trace:", file=sys.stderr)
        traceback.print_exc()

    def _print_contextual_help(self, context: str) -> None:
        """Print contextual help for specific error types."""
        if context not in self.error_contexts:
            return

        help_info = self.error_contexts[context]

        print(f"\n💡 Common fixes for {context} issues:", file=sys.stderr)
        for i, fix in enumerate(help_info.get('common_fixes', []), 1):
            print(f"  {i}. {fix}", file=sys.stderr)

        if 'examples' in help_info:
            print(f"\n📋 Examples:", file=sys.stderr)
            for example in help_info['examples']:
                print(f"  {example}", file=sys.stderr)

    def _analyze_file_issue(self, filepath: str) -> None:
        """Analyze specific file issues and provide targeted guidance."""
        file_path = Path(filepath)

        print(f"\n🔍 File analysis:", file=sys.stderr)

        # Check existence
        if not file_path.exists():
            print(f"  ❌ File does not exist: {filepath}", file=sys.stderr)

            # Suggest similar files
            if file_path.parent.exists():
                similar_files = list(file_path.parent.glob(f"*{file_path.stem}*"))
                if similar_files:
                    print(f"  🔍 Similar files found:", file=sys.stderr)
                    for similar in similar_files[:3]:  # Show max 3
                        print(f"    - {similar}", file=sys.stderr)
        else:
            # Check readability
            try:
                if file_path.is_file():
                    with open(file_path, 'r') as f:
                        f.read(100)  # Try to read first 100 chars
                    print(f"  ✅ File exists and is readable", file=sys.stderr)
                else:
                    print(f"  ❌ Path exists but is not a file", file=sys.stderr)
            except PermissionError:
                print(f"  ❌ Permission denied - check file permissions", file=sys.stderr)
            except UnicodeDecodeError:
                print(f"  ⚠️ File encoding issue - ensure UTF-8 encoding", file=sys.stderr)
            except Exception as e:
                print(f"  ❌ File read error: {e}", file=sys.stderr)

    def _print_algorithm_specific_help(self, algorithm_name: str) -> None:
        """Print algorithm-specific troubleshooting help."""
        algorithm_help = {
            'yu2008': {
                'description': 'Yu et al. (2008) Two-Stage Algorithm',
                'requirements': [
                    'Substrate network with CPU and bandwidth resources',
                    'VNRs with CPU and bandwidth requirements',
                    'No memory/delay constraints (Yu 2008 is CPU+Bandwidth only)'
                ],
                'common_issues': [
                    'Check substrate has sufficient CPU capacity',
                    'Verify substrate connectivity (no isolated nodes)',
                    'Ensure VNR requirements are reasonable ratios of substrate capacity'
                ]
            }
        }

        if algorithm_name in algorithm_help:
            help_info = algorithm_help[algorithm_name]
            print(f"\n🔧 {help_info['description']} specific guidance:", file=sys.stderr)

            print(f"\n📋 Requirements:", file=sys.stderr)
            for req in help_info['requirements']:
                print(f"  - {req}", file=sys.stderr)

            print(f"\n🐛 Common issues:", file=sys.stderr)
            for issue in help_info['common_issues']:
                print(f"  - {issue}", file=sys.stderr)

    def set_debug_mode(self, debug_mode: bool) -> None:
        """Enable or disable debug mode."""
        self.debug_mode = debug_mode
        self.logger.debug(f"Debug mode {'enabled' if debug_mode else 'disabled'}")

    def log_operation_context(self, operation: str, context: Dict[str, Any]) -> None:
        """Log operation context for debugging purposes."""
        if self.debug_mode:
            self.logger.debug(f"Operation: {operation}")
            for key, value in context.items():
                self.logger.debug(f"  {key}: {value}")

    def suggest_next_steps(self, error_type: str) -> None:
        """Suggest next steps based on error type."""
        suggestions = {
            'configuration': [
                "Run 'python main.py config --create-default config.yaml' to create a template",
                "Validate your config with 'python main.py config --validate config.yaml'"
            ],
            'file_missing': [
                "Generate test data with 'python main.py generate substrate --nodes 20'",
                "Check the examples/ directory for sample data files"
            ],
            'algorithm_failure': [
                "Try with smaller datasets first",
                "Check algorithm compatibility with your data",
                "Review algorithm documentation for parameter requirements"
            ]
        }

        if error_type in suggestions:
            print(f"\n➡️ Suggested next steps:", file=sys.stderr)
            for i, suggestion in enumerate(suggestions[error_type], 1):
                print(f"  {i}. {suggestion}", file=sys.stderr)
