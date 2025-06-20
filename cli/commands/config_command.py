"""
Config command implementation.
"""

import yaml
import json
from pathlib import Path
from .base_command import BaseCommand


class ConfigCommand(BaseCommand):
    """Command for configuration management with full functionality."""

    def execute(self, args) -> int:
        """Execute the config command."""
        if getattr(args, 'create_default', None):
            return self._create_default_config(args)
        elif getattr(args, 'show', False):
            return self._show_current_config(args)
        elif getattr(args, 'validate', None):
            return self._validate_config_file(args)
        else:
            print("Must specify a config action (--create-default, --show, or --validate)")
            return 1

    def _create_default_config(self, args) -> int:
        """Create default configuration file."""
        try:
            from config_management import create_default_config_file
            create_default_config_file(args.create_default)
            print(f"✓ Created default configuration file: {args.create_default}")
            return 0
        except Exception as e:
            print(f"✗ Failed to create config: {e}")
            return 1

    def _show_current_config(self, args) -> int:
        """Show current configuration in readable format."""
        try:
            # Load current configuration (either from file or defaults)
            from config_management import load_config_from_args

            print("🔧 Current VNE Configuration")
            print("=" * 50)

            # Load configuration
            config = load_config_from_args(args)

            # Convert config to dictionary for display
            config_dict = self._config_to_dict(config)

            # Convert tuples to lists for better YAML display
            config_dict = self._convert_tuples_to_lists(config_dict)

            # Display in YAML format for readability
            yaml_output = yaml.dump(config_dict,
                                    default_flow_style=False,
                                    sort_keys=True,
                                    indent=2)
            print(yaml_output)

            # Show configuration sources
            print("\n📋 Configuration Sources (in precedence order):")
            print("1. Command-line arguments (highest priority)")
            print("2. Environment variables (VNE_ prefix)")
            print("3. Configuration file (if specified)")
            print("4. Default values (lowest priority)")

            if hasattr(args, 'config') and args.config:
                print(f"\n📁 Configuration file: {args.config}")
            else:
                print("\n📁 No configuration file specified (using defaults)")

            return 0

        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()
            return 1

    def _validate_config_file(self, args) -> int:
        """Validate a configuration file."""
        try:
            config_file = Path(args.validate)

            print(f"🔍 Validating configuration file: {config_file}")
            print("-" * 50)

            # Check if file exists
            if not config_file.exists():
                print(f"✗ Configuration file not found: {config_file}")
                return 1

            # First, try to load the YAML/JSON file directly
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.suffix.lower() in ['.yaml', '.yml']:
                        # For YAML files, use safe_load to avoid Python-specific tags
                        data = yaml.safe_load(f)
                    elif config_file.suffix.lower() == '.json':
                        data = json.load(f)
                    else:
                        # Try YAML first, then JSON
                        f.seek(0)
                        content = f.read()
                        try:
                            data = yaml.safe_load(content)
                        except yaml.YAMLError:
                            data = json.loads(content)

                print("✓ Configuration file syntax is valid")

            except (yaml.YAMLError, json.JSONDecodeError) as e:
                print(f"✗ Configuration file has syntax errors: {e}")
                return 1
            except UnicodeDecodeError as e:
                print(f"✗ Configuration file encoding error: {e}")
                print("💡 Ensure the file is saved in UTF-8 encoding")
                return 1

            # Now try to load it through the configuration manager
            try:
                from config_management import ConfigurationManager

                config_manager = ConfigurationManager()
                config = config_manager.load_config(str(config_file))

                print("✓ All required sections are present")
                print("✓ Configuration values are valid")

                # Show summary of loaded configuration
                print(f"\n📊 Configuration Summary:")
                print(f"  • Substrate nodes: {config.network_generation.substrate_nodes}")
                print(f"  • VNR count: {config.network_generation.vnr_count}")
                print(f"  • Data directory: {config.file_paths.data_dir}")
                print(f"  • Log level: {config.logging.root_level}")

                print(f"\n✅ Configuration file '{config_file}' is valid!")
                return 0

            except Exception as e:
                print(f"⚠️  Configuration structure validation failed: {e}")
                print("📋 File has valid syntax but may have structural issues")
                print("💡 Check that all required sections and values are present")
                return 1

        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            print(f"\n💡 Common issues:")
            print(f"  • Check YAML/JSON syntax")
            print(f"  • Verify all required sections exist")
            print(f"  • Ensure values are in correct format")
            print(f"  • Make sure file is UTF-8 encoded")
            return 1

    def _config_to_dict(self, config) -> dict:
        """Convert VNEConfig object to dictionary for display."""
        result = {}

        # Get all attributes that don't start with underscore
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                attr_value = getattr(config, attr_name)

                # Skip methods
                if callable(attr_value):
                    continue

                # Handle nested config objects
                if hasattr(attr_value, '__dict__'):
                    result[attr_name] = self._nested_config_to_dict(attr_value)
                else:
                    result[attr_name] = attr_value

        return result

    def _nested_config_to_dict(self, nested_config) -> dict:
        """Convert nested config objects to dictionaries."""
        result = {}

        for attr_name in dir(nested_config):
            if not attr_name.startswith('_'):
                attr_value = getattr(nested_config, attr_name)

                # Skip methods
                if callable(attr_value):
                    continue

                # Handle deeply nested objects
                if hasattr(attr_value, '__dict__') and not isinstance(attr_value,
                                                                      (str, int, float, bool, list, dict, tuple)):
                    result[attr_name] = self._nested_config_to_dict(attr_value)
                else:
                    result[attr_name] = attr_value

        return result

    def _convert_tuples_to_lists(self, data):
        """Convert tuples to lists recursively for better YAML serialization."""
        if isinstance(data, dict):
            return {key: self._convert_tuples_to_lists(value) for key, value in data.items()}
        elif isinstance(data, tuple):
            return list(data)
        elif isinstance(data, list):
            return [self._convert_tuples_to_lists(item) for item in data]
        else:
            return data
