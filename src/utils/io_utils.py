#!/usr/bin/env python3
"""
Experiment I/O utilities for Virtual Network Embedding (VNE).

This module provides simple experiment management using model built-in methods
for saving and loading substrate networks, VNR batches, and results.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class ExperimentIOError(Exception):
    """Exception raised for experiment I/O operations."""
    pass


def create_experiment_directory(base_dir: Union[str, Path],
                                experiment_name: str,
                                include_timestamp: bool = True) -> Path:
    """
    Create experiment directory structure.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        include_timestamp: Whether to include timestamp in directory name

    Returns:
        Path to experiment directory

    Example:
        >>> exp_dir = create_experiment_directory("experiments", "test_algorithm")
    """
    base_path = Path(base_dir)

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_path / f"{experiment_name}_{timestamp}"
    else:
        exp_dir = base_path / experiment_name

    # Create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def save_experiment_data(substrate_network,
                         vnr_batch,
                         experiment_dir: Union[str, Path]) -> None:
    """
    Save experiment data using model built-in methods.

    Args:
        substrate_network: SubstrateNetwork instance
        vnr_batch: VNRBatch instance
        experiment_dir: Experiment directory path

    Raises:
        ExperimentIOError: If save operation fails

    Example:
        >>> save_experiment_data(substrate, vnr_batch, "experiments/test_001")
    """
    try:
        exp_path = Path(experiment_dir)
        data_dir = exp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Use model built-in CSV export methods
        substrate_network.save_to_csv(
            str(data_dir / "substrate_nodes.csv"),
            str(data_dir / "substrate_links.csv")
        )

        vnr_batch.save_to_csv(str(data_dir / "vnr_batch"))

        logger.info(f"Saved experiment data to {data_dir}")

    except Exception as e:
        raise ExperimentIOError(f"Failed to save experiment data: {e}")


def load_experiment_data(experiment_dir: Union[str, Path]) -> Tuple[object, object]:
    """
    Load experiment data using model built-in methods.

    Args:
        experiment_dir: Experiment directory path

    Returns:
        Tuple of (substrate_network, vnr_batch)

    Raises:
        ExperimentIOError: If load operation fails

    Example:
        >>> substrate, vnr_batch = load_experiment_data("experiments/test_001")
    """
    try:
        from src.models.substrate import SubstrateNetwork
        from src.models.vnr_batch import VNRBatch

        exp_path = Path(experiment_dir)
        data_dir = exp_path / "data"

        if not data_dir.exists():
            raise ExperimentIOError(f"Data directory not found: {data_dir}")

        # Use model built-in CSV import methods
        substrate = SubstrateNetwork()
        substrate.load_from_csv(
            str(data_dir / "substrate_nodes.csv"),
            str(data_dir / "substrate_links.csv")
        )

        vnr_batch = VNRBatch.load_from_csv(str(data_dir / "vnr_batch"))

        logger.info(f"Loaded experiment data from {data_dir}")
        return substrate, vnr_batch

    except ImportError as e:
        raise ExperimentIOError(f"Failed to import models: {e}")
    except Exception as e:
        raise ExperimentIOError(f"Failed to load experiment data: {e}")


def save_algorithm_results(results: List[Dict[str, Any]],
                           experiment_dir: Union[str, Path],
                           algorithm_name: str,
                           include_timestamp: bool = True) -> Path:
    """
    Save algorithm results to JSON.

    Args:
        results: List of embedding results
        experiment_dir: Experiment directory path
        algorithm_name: Name of the algorithm
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Path to saved results file

    Raises:
        ExperimentIOError: If save operation fails

    Example:
        >>> file_path = save_algorithm_results(results, exp_dir, "greedy_algorithm")
    """
    try:
        exp_path = Path(experiment_dir)
        results_dir = exp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name}_results_{timestamp}.json"
        else:
            filename = f"{algorithm_name}_results.json"

        results_file = results_dir / filename

        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, '__dict__'):
                result_dict = result.__dict__.copy()
            else:
                result_dict = dict(result) if result else {}

            # Handle non-serializable values
            clean_result = {}
            for key, value in result_dict.items():
                if value is None or isinstance(value, (int, float, str, bool, list, dict)):
                    clean_result[key] = value
                else:
                    clean_result[key] = str(value)

            serializable_results.append(clean_result)

        # Save with metadata
        output_data = {
            'metadata': {
                'algorithm': algorithm_name,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(results),
                'experiment_dir': str(exp_path)
            },
            'results': serializable_results
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {len(results)} algorithm results to {results_file}")
        return results_file

    except Exception as e:
        raise ExperimentIOError(f"Failed to save algorithm results: {e}")


def load_algorithm_results(results_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load algorithm results from JSON.

    Args:
        results_file: Path to results JSON file

    Returns:
        List of result dictionaries

    Raises:
        ExperimentIOError: If load operation fails

    Example:
        >>> results = load_algorithm_results("experiments/test_001/results/greedy_results.json")
    """
    try:
        results_path = Path(results_file)

        if not results_path.exists():
            raise ExperimentIOError(f"Results file not found: {results_path}")

        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'results' in data:
            logger.info(f"Loaded {len(data['results'])} results from {results_path}")
            return data['results']
        elif isinstance(data, list):
            logger.info(f"Loaded {len(data)} results from {results_path}")
            return data
        else:
            raise ExperimentIOError("Invalid results file format")

    except json.JSONDecodeError as e:
        raise ExperimentIOError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ExperimentIOError(f"Failed to load algorithm results: {e}")


def save_metrics_summary(metrics: Dict[str, Any],
                         experiment_dir: Union[str, Path],
                         algorithm_name: str,
                         include_timestamp: bool = True) -> Path:
    """
    Save metrics summary to JSON.

    Args:
        metrics: Metrics dictionary
        experiment_dir: Experiment directory path
        algorithm_name: Name of the algorithm
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Path to saved metrics file

    Raises:
        ExperimentIOError: If save operation fails

    Example:
        >>> metrics_file = save_metrics_summary(metrics, exp_dir, "greedy_algorithm")
    """
    try:
        exp_path = Path(experiment_dir)
        results_dir = exp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name}_metrics_{timestamp}.json"
        else:
            filename = f"{algorithm_name}_metrics.json"

        metrics_file = results_dir / filename

        metrics_data = {
            'metadata': {
                'algorithm': algorithm_name,
                'timestamp': datetime.now().isoformat(),
                'experiment_dir': str(exp_path)
            },
            'metrics': metrics
        }

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Saved metrics summary to {metrics_file}")
        return metrics_file

    except Exception as e:
        raise ExperimentIOError(f"Failed to save metrics summary: {e}")


def save_experiment_config(config: Dict[str, Any],
                           experiment_dir: Union[str, Path]) -> Path:
    """
    Save experiment configuration.

    Args:
        config: Configuration dictionary
        experiment_dir: Experiment directory path

    Returns:
        Path to saved config file

    Raises:
        ExperimentIOError: If save operation fails

    Example:
        >>> config_file = save_experiment_config(config, exp_dir)
    """
    try:
        exp_path = Path(experiment_dir)
        config_file = exp_path / "experiment_config.json"

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved experiment config to {config_file}")
        return config_file

    except Exception as e:
        raise ExperimentIOError(f"Failed to save experiment config: {e}")


def load_experiment_config(experiment_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment configuration.

    Args:
        experiment_dir: Experiment directory path

    Returns:
        Configuration dictionary

    Raises:
        ExperimentIOError: If load operation fails

    Example:
        >>> config = load_experiment_config("experiments/test_001")
    """
    try:
        exp_path = Path(experiment_dir)
        config_file = exp_path / "experiment_config.json"

        if not config_file.exists():
            raise ExperimentIOError(f"Config file not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"Loaded experiment config from {config_file}")
        return config

    except json.JSONDecodeError as e:
        raise ExperimentIOError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ExperimentIOError(f"Failed to load experiment config: {e}")


class ExperimentRunner:
    """
    Simple experiment runner for algorithm development.

    This class provides a clean interface for managing VNE experiments
    using the model's built-in I/O capabilities.

    Example:
        >>> runner = ExperimentRunner("my_algorithm_test")
        >>> runner.setup_experiment(substrate, vnr_batch)
        >>>
        >>> # Your algorithm development here
        >>> results = my_algorithm.run(substrate, vnr_batch)
        >>>
        >>> runner.save_results(results, "my_algorithm")
        >>> runner.finish_experiment()
    """

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment runner.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = create_experiment_directory(base_dir, experiment_name)
        self.start_time = time.time()

        logger.info(f"Started experiment: {experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")

    def setup_experiment(self, substrate_network, vnr_batch) -> None:
        """
        Save experiment setup data and configuration.

        Args:
            substrate_network: SubstrateNetwork instance
            vnr_batch: VNRBatch instance

        Raises:
            ExperimentIOError: If setup fails
        """
        try:
            # Save experiment data
            save_experiment_data(substrate_network, vnr_batch, self.experiment_dir)

            # Save experiment configuration
            config = {
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'substrate_stats': substrate_network.get_network_statistics(),
                'vnr_batch_info': vnr_batch.get_basic_info(),
                'constraint_config': substrate_network.get_constraint_configuration()
            }

            save_experiment_config(config, self.experiment_dir)

            logger.info("Experiment setup completed")

        except Exception as e:
            raise ExperimentIOError(f"Failed to setup experiment: {e}")

    def save_results(self, results: List, algorithm_name: str) -> Path:
        """
        Save algorithm results.

        Args:
            results: List of embedding results
            algorithm_name: Name of the algorithm

        Returns:
            Path to saved results file

        Raises:
            ExperimentIOError: If save fails
        """
        return save_algorithm_results(results, self.experiment_dir, algorithm_name)

    def finish_experiment(self) -> Dict[str, Any]:
        """
        Finish experiment and save summary.

        Returns:
            Experiment summary dictionary

        Raises:
            ExperimentIOError: If finish fails
        """
        try:
            duration = time.time() - self.start_time

            summary = {
                'experiment_name': self.experiment_name,
                'experiment_dir': str(self.experiment_dir),
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'status': 'completed'
            }

            summary_file = self.experiment_dir / "experiment_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Experiment completed in {duration:.1f} seconds")
            logger.info(f"Results saved in: {self.experiment_dir}")

            return summary

        except Exception as e:
            raise ExperimentIOError(f"Failed to finish experiment: {e}")

    def get_experiment_path(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir

    def __str__(self) -> str:
        """String representation of the experiment runner."""
        return f"ExperimentRunner(name='{self.experiment_name}', dir='{self.experiment_dir}')"
