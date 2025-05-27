"""
File I/O utilities for Virtual Network Embedding (VNE) experiments.

This module provides functions for saving and loading substrate networks,
VNRs, and experimental results in various formats.
"""

import logging
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class VNEFileError(Exception):
    """Custom exception for VNE file operations."""
    pass


def ensure_directory_exists(filepath: Union[str, Path]) -> None:
    """
    Ensure the directory for a file path exists.
    
    Args:
        filepath: Path to file or directory
        
    Example:
        >>> ensure_directory_exists("results/experiment1/data.csv")
    """
    path = Path(filepath)
    directory = path.parent if path.suffix else path
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def validate_file_path(filepath: Union[str, Path], 
                      check_exists: bool = False,
                      expected_extension: Optional[str] = None) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        filepath: Path to validate
        check_exists: Whether to check if file exists
        expected_extension: Expected file extension (e.g., '.csv')
        
    Returns:
        Validated Path object
        
    Raises:
        VNEFileError: If validation fails
        
    Example:
        >>> path = validate_file_path("data.csv", expected_extension=".csv")
    """
    path = Path(filepath)
    
    if expected_extension and path.suffix.lower() != expected_extension.lower():
        raise VNEFileError(f"Expected {expected_extension} file, got {path.suffix}")
    
    if check_exists and not path.exists():
        raise VNEFileError(f"File does not exist: {path}")
    
    return path


def save_substrate_to_csv(substrate_network, 
                         filepath: Union[str, Path],
                         include_metadata: bool = True) -> None:
    """
    Save substrate network to CSV files.
    
    Creates two files:
    - {filepath}_nodes.csv: Node information and resources
    - {filepath}_links.csv: Link information and resources
    
    Args:
        substrate_network: SubstrateNetwork instance
        filepath: Base filepath (without extension)
        include_metadata: Whether to include metadata file
        
    Raises:
        VNEFileError: If save operation fails
        
    Example:
        >>> save_substrate_to_csv(substrate, "networks/substrate_100")
    """
    base_path = Path(filepath)
    ensure_directory_exists(base_path.parent)
    
    try:
        # Save nodes
        nodes_file = base_path.with_name(f"{base_path.name}_nodes.csv")
        _save_substrate_nodes_csv(substrate_network, nodes_file)
        
        # Save links  
        links_file = base_path.with_name(f"{base_path.name}_links.csv")
        _save_substrate_links_csv(substrate_network, links_file)
        
        # Save metadata if requested
        if include_metadata:
            metadata_file = base_path.with_name(f"{base_path.name}_metadata.json")
            _save_substrate_metadata(substrate_network, metadata_file)
        
        logger.info(f"Created experiment directory: {exp_dir}")
    
    return exp_dir


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        VNEFileError: If file doesn't exist
        
    Example:
        >>> size = get_file_size("data.csv")
    """
    path = validate_file_path(filepath, check_exists=True)
    return path.stat().st_size


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
        
    Raises:
        VNEFileError: If file doesn't exist
        
    Example:
        >>> info = get_file_info("data.csv")
        >>> print(f"Size: {info['size_bytes']} bytes")
    """
    path = validate_file_path(filepath, check_exists=True)
    stat = path.stat()
    
    return {
        'filename': path.name,
        'filepath': str(path.absolute()),
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'extension': path.suffix,
        'is_file': path.is_file(),
        'is_directory': path.is_dir()
    }


def backup_file(filepath: Union[str, Path], 
                backup_suffix: str = '.bak') -> Path:
    """
    Create backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_suffix: Suffix for backup file
        
    Returns:
        Path to backup file
        
    Raises:
        VNEFileError: If backup fails
        
    Example:
        >>> backup_path = backup_file("important_data.csv")
    """
    source_path = validate_file_path(filepath, check_exists=True)
    backup_path = source_path.with_suffix(source_path.suffix + backup_suffix)
    
    try:
        import shutil
        shutil.copy2(source_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        raise VNEFileError(f"Failed to create backup: {e}")


def cleanup_old_files(directory: Union[str, Path], 
                     pattern: str = "*.csv",
                     max_age_days: int = 30) -> List[Path]:
    """
    Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age in days
        
    Returns:
        List of deleted file paths
        
    Example:
        >>> deleted = cleanup_old_files("temp", "*.csv", 7)
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    import time
    from glob import glob
    
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    deleted_files = []
    
    try:
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_files.append(file_path)
                logger.debug(f"Deleted old file: {file_path}")
        
        logger.info(f"Cleaned up {len(deleted_files)} old files")
        return deleted_files
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return deleted_files


def compress_file(filepath: Union[str, Path], 
                 compression: str = 'gzip') -> Path:
    """
    Compress a file.
    
    Args:
        filepath: Path to file to compress
        compression: Compression method ('gzip', 'zip')
        
    Returns:
        Path to compressed file
        
    Raises:
        VNEFileError: If compression fails
        
    Example:
        >>> compressed = compress_file("large_data.csv", "gzip")
    """
    source_path = validate_file_path(filepath, check_exists=True)
    
    try:
        if compression.lower() == 'gzip':
            import gzip
            compressed_path = source_path.with_suffix(source_path.suffix + '.gz')
            
            with open(source_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
        elif compression.lower() == 'zip':
            import zipfile
            compressed_path = source_path.with_suffix('.zip')
            
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(source_path, source_path.name)
        else:
            raise VNEFileError(f"Unsupported compression: {compression}")
        
        logger.info(f"Compressed {source_path} to {compressed_path}")
        return compressed_path
        
    except Exception as e:
        raise VNEFileError(f"Compression failed: {e}")


def decompress_file(filepath: Union[str, Path]) -> Path:
    """
    Decompress a file.
    
    Args:
        filepath: Path to compressed file
        
    Returns:
        Path to decompressed file
        
    Raises:
        VNEFileError: If decompression fails
        
    Example:
        >>> decompressed = decompress_file("data.csv.gz")
    """
    source_path = validate_file_path(filepath, check_exists=True)
    
    try:
        if source_path.suffix.lower() == '.gz':
            import gzip
            decompressed_path = source_path.with_suffix('')
            
            with gzip.open(source_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
        elif source_path.suffix.lower() == '.zip':
            import zipfile
            with zipfile.ZipFile(source_path, 'r') as zipf:
                # Extract first file
                names = zipf.namelist()
                if not names:
                    raise VNEFileError("Empty zip file")
                
                decompressed_path = source_path.parent / names[0]
                zipf.extract(names[0], source_path.parent)
        else:
            raise VNEFileError(f"Unsupported compression format: {source_path.suffix}")
        
        logger.info(f"Decompressed {source_path} to {decompressed_path}")
        return decompressed_path
        
    except Exception as e:
        raise VNEFileError(f"Decompression failed: {e}")


def merge_csv_files(file_paths: List[Union[str, Path]], 
                   output_path: Union[str, Path],
                   include_source_column: bool = True) -> None:
    """
    Merge multiple CSV files into one.
    
    Args:
        file_paths: List of CSV file paths to merge
        output_path: Output CSV file path
        include_source_column: Whether to add source filename column
        
    Raises:
        VNEFileError: If merge fails
        
    Example:
        >>> merge_csv_files(["data1.csv", "data2.csv"], "merged.csv")
    """
    if not file_paths:
        raise VNEFileError("No files to merge")
    
    output = Path(output_path)
    ensure_directory_exists(output.parent)
    
    try:
        first_file = validate_file_path(file_paths[0], check_exists=True)
        
        # Get headers from first file
        with open(first_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        if include_source_column:
            headers.append('source_file')
        
        # Write merged file
        with open(output, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            
            for file_path in file_paths:
                current_file = validate_file_path(file_path, check_exists=True)
                
                with open(current_file, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    next(reader)  # Skip header
                    
                    for row in reader:
                        if include_source_column:
                            row.append(current_file.name)
                        writer.writerow(row)
        
        logger.info(f"Merged {len(file_paths)} files into {output}")
        
    except Exception as e:
        raise VNEFileError(f"File merge failed: {e}")


def split_csv_file(filepath: Union[str, Path], 
                  output_dir: Union[str, Path],
                  rows_per_file: int = 1000,
                  prefix: str = "split") -> List[Path]:
    """
    Split large CSV file into smaller files.
    
    Args:
        filepath: Path to CSV file to split
        output_dir: Directory for output files
        rows_per_file: Number of rows per output file
        prefix: Prefix for output filenames
        
    Returns:
        List of created file paths
        
    Raises:
        VNEFileError: If split fails
        
    Example:
        >>> files = split_csv_file("large_data.csv", "split_files", 500)
    """
    source_path = validate_file_path(filepath, check_exists=True, expected_extension='.csv')
    output_path = Path(output_dir)
    ensure_directory_exists(output_path)
    
    try:
        created_files = []
        
        with open(source_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            
            file_count = 0
            row_count = 0
            current_writer = None
            current_file = None
            
            for row in reader:
                if row_count % rows_per_file == 0:
                    # Close previous file
                    if current_file:
                        current_file.close()
                    
                    # Open new file
                    file_count += 1
                    filename = f"{prefix}_{file_count:04d}.csv"
                    current_path = output_path / filename
                    current_file = open(current_path, 'w', newline='', encoding='utf-8')
                    current_writer = csv.writer(current_file)
                    current_writer.writerow(headers)
                    created_files.append(current_path)
                
                current_writer.writerow(row)
                row_count += 1
            
            # Close last file
            if current_file:
                current_file.close()
        
        logger.info(f"Split {source_path} into {len(created_files)} files")
        return created_files
        
    except Exception as e:
        raise VNEFileError(f"File split failed: {e}")


# Example usage and testing functions
def example_usage():
    """
    Example usage of the I/O utilities.
    
    This function demonstrates how to use the various I/O functions.
    """
    logger.info("=== VNE I/O Utilities Example Usage ===")
    
    try:
        # Create experiment directory
        exp_dir = create_experiment_directory("examples", "io_test", timestamp=False)
        
        # Example file operations
        test_file = exp_dir / "test_data.csv"
        
        # Create sample data
        sample_data = [
            ["name", "value", "timestamp"],
            ["test1", "100", "2024-01-01"],
            ["test2", "200", "2024-01-02"]
        ]
        
        with open(test_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)
        
        # Demonstrate file info
        info = get_file_info(test_file)
        print(f"File info: {info}")
        
        # Demonstrate backup
        backup_path = backup_file(test_file)
        print(f"Backup created: {backup_path}")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_usage()Saved substrate network to {base_path}")
        
    except Exception as e:
        raise VNEFileError(f"Failed to save substrate network: {e}")


def _save_substrate_nodes_csv(substrate_network, filepath: Path) -> None:
    """Save substrate nodes to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'node_id', 'cpu_capacity', 'available_cpu', 'memory_capacity', 
            'available_memory', 'x_coordinate', 'y_coordinate'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for node_id, node_resources in substrate_network.nodes.items():
            writer.writerow({
                'node_id': node_id,
                'cpu_capacity': node_resources.cpu_capacity,
                'available_cpu': node_resources.available_cpu,
                'memory_capacity': node_resources.memory_capacity,
                'available_memory': node_resources.available_memory,
                'x_coordinate': getattr(node_resources, 'x_coordinate', 0.0),
                'y_coordinate': getattr(node_resources, 'y_coordinate', 0.0)
            })


def _save_substrate_links_csv(substrate_network, filepath: Path) -> None:
    """Save substrate links to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'source_node', 'target_node', 'bandwidth_capacity', 
            'available_bandwidth', 'delay', 'cost'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for link_id, link_resources in substrate_network.links.items():
            # Extract source and target nodes from link_id
            if isinstance(link_id, tuple):
                source, target = link_id
            else:
                # Assume format like "node1-node2"
                source, target = str(link_id).split('-', 1)
            
            writer.writerow({
                'source_node': source,
                'target_node': target,
                'bandwidth_capacity': link_resources.bandwidth_capacity,
                'available_bandwidth': link_resources.available_bandwidth,
                'delay': getattr(link_resources, 'delay', 1.0),
                'cost': getattr(link_resources, 'cost', 1.0)
            })


def _save_substrate_metadata(substrate_network, filepath: Path) -> None:
    """Save substrate network metadata to JSON file."""
    metadata = {
        'created_at': datetime.now().isoformat(),
        'node_count': len(substrate_network.nodes),
        'link_count': len(substrate_network.links),
        'total_cpu_capacity': sum(n.cpu_capacity for n in substrate_network.nodes.values()),
        'total_memory_capacity': sum(n.memory_capacity for n in substrate_network.nodes.values()),
        'total_bandwidth_capacity': sum(l.bandwidth_capacity for l in substrate_network.links.values()),
        'network_statistics': substrate_network.get_network_statistics() if hasattr(substrate_network, 'get_network_statistics') else {}
    }
    
    with open(filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump(metadata, jsonfile, indent=2)


def load_substrate_from_csv(filepath: Union[str, Path]) -> object:
    """
    Load substrate network from CSV files.
    
    Expects files:
    - {filepath}_nodes.csv: Node information
    - {filepath}_links.csv: Link information
    
    Args:
        filepath: Base filepath (without extension)
        
    Returns:
        SubstrateNetwork instance
        
    Raises:
        VNEFileError: If load operation fails
        
    Example:
        >>> substrate = load_substrate_from_csv("networks/substrate_100")
    """
    base_path = Path(filepath)
    
    try:
        # Check required files exist
        nodes_file = base_path.with_name(f"{base_path.name}_nodes.csv")
        links_file = base_path.with_name(f"{base_path.name}_links.csv")
        
        if not nodes_file.exists():
            raise VNEFileError(f"Nodes file not found: {nodes_file}")
        if not links_file.exists():
            raise VNEFileError(f"Links file not found: {links_file}")
        
        # Load nodes and links data
        nodes_data = _load_substrate_nodes_csv(nodes_file)
        links_data = _load_substrate_links_csv(links_file)
        
        logger.info(f"Loaded substrate network from {base_path}")
        
        # Return data structure for now - in real implementation would create SubstrateNetwork
        return {
            'nodes': nodes_data,
            'links': links_data,
            'source_path': str(base_path)
        }
        
    except Exception as e:
        raise VNEFileError(f"Failed to load substrate network: {e}")


def _load_substrate_nodes_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load substrate nodes from CSV file."""
    nodes_data = []
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            node_data = {
                'node_id': row['node_id'],
                'cpu_capacity': int(row['cpu_capacity']),
                'available_cpu': int(row['available_cpu']),
                'memory_capacity': int(row['memory_capacity']),
                'available_memory': int(row['available_memory']),
                'x_coordinate': float(row.get('x_coordinate', 0.0)),
                'y_coordinate': float(row.get('y_coordinate', 0.0))
            }
            nodes_data.append(node_data)
    
    return nodes_data


def _load_substrate_links_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load substrate links from CSV file."""
    links_data = []
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            link_data = {
                'source_node': row['source_node'],
                'target_node': row['target_node'],
                'bandwidth_capacity': int(row['bandwidth_capacity']),
                'available_bandwidth': int(row['available_bandwidth']),
                'delay': float(row.get('delay', 1.0)),
                'cost': float(row.get('cost', 1.0))
            }
            links_data.append(link_data)
    
    return links_data


def save_vnrs_to_csv(vnrs: List, filepath: Union[str, Path]) -> None:
    """
    Save VNR batch to CSV files.
    
    Creates two files:
    - {filepath}_vnrs.csv: VNR metadata
    - {filepath}_vnr_details.csv: Detailed node and link requirements
    
    Args:
        vnrs: List of VirtualNetworkRequest instances
        filepath: Base filepath (without extension)
        
    Raises:
        VNEFileError: If save operation fails
        
    Example:
        >>> save_vnrs_to_csv(vnr_batch, "experiments/vnrs_batch1")
    """
    base_path = Path(filepath)
    ensure_directory_exists(base_path.parent)
    
    try:
        # Save VNR metadata
        vnrs_file = base_path.with_name(f"{base_path.name}_vnrs.csv")
        _save_vnrs_metadata_csv(vnrs, vnrs_file)
        
        # Save VNR details
        details_file = base_path.with_name(f"{base_path.name}_vnr_details.csv")
        _save_vnrs_details_csv(vnrs, details_file)
        
        logger.info(f"Saved {len(vnrs)} VNRs to {base_path}")
        
    except Exception as e:
        raise VNEFileError(f"Failed to save VNRs: {e}")


def _save_vnrs_metadata_csv(vnrs: List, filepath: Path) -> None:
    """Save VNR metadata to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'vnr_id', 'arrival_time', 'lifetime', 'departure_time',
            'node_count', 'link_count', 'total_cpu_req', 'total_memory_req',
            'total_bandwidth_req', 'priority', 'revenue'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for vnr in vnrs:
            writer.writerow({
                'vnr_id': vnr.vnr_id,
                'arrival_time': vnr.arrival_time,
                'lifetime': vnr.lifetime,
                'departure_time': vnr.arrival_time + vnr.lifetime,
                'node_count': len(vnr.virtual_nodes),
                'link_count': len(vnr.virtual_links),
                'total_cpu_req': sum(n.cpu_requirement for n in vnr.virtual_nodes.values()),
                'total_memory_req': sum(n.memory_requirement for n in vnr.virtual_nodes.values()),
                'total_bandwidth_req': sum(l.bandwidth_requirement for l in vnr.virtual_links.values()),
                'priority': getattr(vnr, 'priority', 1),
                'revenue': vnr.calculate_revenue() if hasattr(vnr, 'calculate_revenue') else 0.0
            })


def _save_vnrs_details_csv(vnrs: List, filepath: Path) -> None:
    """Save detailed VNR requirements to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'vnr_id', 'element_type', 'element_id', 'source_node', 'target_node',
            'cpu_requirement', 'memory_requirement', 'bandwidth_requirement'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for vnr in vnrs:
            # Write virtual nodes
            for node_id, node_req in vnr.virtual_nodes.items():
                writer.writerow({
                    'vnr_id': vnr.vnr_id,
                    'element_type': 'node',
                    'element_id': node_id,
                    'source_node': '',
                    'target_node': '',
                    'cpu_requirement': node_req.cpu_requirement,
                    'memory_requirement': node_req.memory_requirement,
                    'bandwidth_requirement': 0
                })
            
            # Write virtual links
            for link_id, link_req in vnr.virtual_links.items():
                if isinstance(link_id, tuple):
                    source, target = link_id
                else:
                    source, target = str(link_id).split('-', 1)
                
                writer.writerow({
                    'vnr_id': vnr.vnr_id,
                    'element_type': 'link',
                    'element_id': str(link_id),
                    'source_node': source,
                    'target_node': target,
                    'cpu_requirement': 0,
                    'memory_requirement': 0,
                    'bandwidth_requirement': link_req.bandwidth_requirement
                })


def load_vnrs_from_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load VNR batch from CSV files.
    
    Expects files:
    - {filepath}_vnrs.csv: VNR metadata
    - {filepath}_vnr_details.csv: Detailed requirements
    
    Args:
        filepath: Base filepath (without extension)
        
    Returns:
        List of VNR data dictionaries
        
    Raises:
        VNEFileError: If load operation fails
        
    Example:
        >>> vnrs = load_vnrs_from_csv("experiments/vnrs_batch1")
    """
    base_path = Path(filepath)
    
    try:
        # Check required files exist
        vnrs_file = base_path.with_name(f"{base_path.name}_vnrs.csv")
        details_file = base_path.with_name(f"{base_path.name}_vnr_details.csv")
        
        if not vnrs_file.exists():
            raise VNEFileError(f"VNRs file not found: {vnrs_file}")
        if not details_file.exists():
            raise VNEFileError(f"VNR details file not found: {details_file}")
        
        # Load VNR metadata
        vnrs_metadata = _load_vnrs_metadata_csv(vnrs_file)
        
        # Load VNR details
        vnrs_details = _load_vnrs_details_csv(details_file)
        
        # Combine metadata and details
        vnrs_data = _combine_vnr_data(vnrs_metadata, vnrs_details)
        
        logger.info(f"Loaded {len(vnrs_data)} VNRs from {base_path}")
        
        return vnrs_data
        
    except Exception as e:
        raise VNEFileError(f"Failed to load VNRs: {e}")


def _load_vnrs_metadata_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load VNR metadata from CSV file."""
    vnrs_metadata = []
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            vnr_data = {
                'vnr_id': row['vnr_id'],
                'arrival_time': float(row['arrival_time']),
                'lifetime': float(row['lifetime']),
                'departure_time': float(row['departure_time']),
                'node_count': int(row['node_count']),
                'link_count': int(row['link_count']),
                'total_cpu_req': int(row['total_cpu_req']),
                'total_memory_req': int(row['total_memory_req']),
                'total_bandwidth_req': int(row['total_bandwidth_req']),
                'priority': int(row.get('priority', 1)),
                'revenue': float(row.get('revenue', 0.0))
            }
            vnrs_metadata.append(vnr_data)
    
    return vnrs_metadata


def _load_vnrs_details_csv(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Load VNR details from CSV file."""
    vnrs_details = {}
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            vnr_id = row['vnr_id']
            element_type = row['element_type']
            
            if vnr_id not in vnrs_details:
                vnrs_details[vnr_id] = {'nodes': {}, 'links': {}}
            
            if element_type == 'node':
                vnrs_details[vnr_id]['nodes'][row['element_id']] = {
                    'cpu_requirement': int(row['cpu_requirement']),
                    'memory_requirement': int(row['memory_requirement'])
                }
            elif element_type == 'link':
                vnrs_details[vnr_id]['links'][row['element_id']] = {
                    'source_node': row['source_node'],
                    'target_node': row['target_node'],
                    'bandwidth_requirement': int(row['bandwidth_requirement'])
                }
    
    return vnrs_details


def _combine_vnr_data(metadata: List[Dict[str, Any]], 
                     details: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine VNR metadata and details into complete VNR data."""
    combined_data = []
    
    for vnr_meta in metadata:
        vnr_id = vnr_meta['vnr_id']
        vnr_detail = details.get(vnr_id, {'nodes': {}, 'links': {}})
        
        combined_vnr = {
            **vnr_meta,
            'virtual_nodes': vnr_detail['nodes'],
            'virtual_links': vnr_detail['links']
        }
        combined_data.append(combined_vnr)
    
    return combined_data


def save_results_to_file(results: List, 
                        filepath: Union[str, Path],
                        format: str = 'json') -> None:
    """
    Save embedding results to file.
    
    Args:
        results: List of EmbeddingResult instances or result dictionaries
        filepath: Output file path
        format: Output format ('json', 'csv')
        
    Raises:
        VNEFileError: If save operation fails
        
    Example:
        >>> save_results_to_file(results, "results/experiment1.json")
        >>> save_results_to_file(results, "results/experiment1.csv", format='csv')
    """
    path = Path(filepath)
    ensure_directory_exists(path.parent)
    
    try:
        if format.lower() == 'json':
            _save_results_to_json(results, path)
        elif format.lower() == 'csv':
            _save_results_to_csv(results, path)
        else:
            raise VNEFileError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(results)} results to {path}")
        
    except Exception as e:
        raise VNEFileError(f"Failed to save results: {e}")


def _save_results_to_json(results: List, filepath: Path) -> None:
    """Save results to JSON file."""
    # Convert results to serializable format
    serializable_results = []
    
    for result in results:
        if hasattr(result, '__dict__'):
            # Convert object to dictionary
            result_dict = result.__dict__.copy()
        else:
            result_dict = result
        
        # Ensure all values are JSON serializable
        cleaned_result = _clean_for_json(result_dict)
        serializable_results.append(cleaned_result)
    
    with open(filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump({
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'result_count': len(results),
                'format_version': '1.0'
            },
            'results': serializable_results
        }, jsonfile, indent=2, default=str)


def _save_results_to_csv(results: List, filepath: Path) -> None:
    """Save results to CSV file."""
    if not results:
        raise VNEFileError("No results to save")
    
    # Extract field names from first result
    first_result = results[0]
    if hasattr(first_result, '__dict__'):
        fieldnames = list(first_result.__dict__.keys())
    else:
        fieldnames = list(first_result.keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            if hasattr(result, '__dict__'):
                row_data = result.__dict__.copy()
            else:
                row_data = result.copy()
            
            # Convert complex types to strings for CSV
            for key, value in row_data.items():
                if isinstance(value, (dict, list)):
                    row_data[key] = json.dumps(value)
                elif value is None:
                    row_data[key] = ''
            
            writer.writerow(row_data)


def _clean_for_json(obj: Any) -> Any:
    """Clean object for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _clean_for_json(obj.__dict__)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def load_results_from_file(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load results from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of result dictionaries
        
    Raises:
        VNEFileError: If load operation fails
        
    Example:
        >>> results = load_results_from_file("results/experiment1.json")
    """
    path = validate_file_path(filepath, check_exists=True)
    
    try:
        if path.suffix.lower() == '.json':
            return _load_results_from_json(path)
        elif path.suffix.lower() == '.csv':
            return _load_results_from_csv(path)
        else:
            raise VNEFileError(f"Unsupported file format: {path.suffix}")
        
    except Exception as e:
        raise VNEFileError(f"Failed to load results: {e}")


def _load_results_from_json(filepath: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise VNEFileError("Invalid JSON format for results file")


def _load_results_from_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load results from CSV file."""
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Convert string values back to appropriate types
            cleaned_row = {}
            for key, value in row.items():
                if value == '':
                    cleaned_row[key] = None
                elif value.startswith('{') or value.startswith('['):
                    try:
                        cleaned_row[key] = json.loads(value)
                    except json.JSONDecodeError:
                        cleaned_row[key] = value
                else:
                    # Try to convert to number if possible
                    try:
                        if '.' in value:
                            cleaned_row[key] = float(value)
                        else:
                            cleaned_row[key] = int(value)
                    except ValueError:
                        cleaned_row[key] = value
            
            results.append(cleaned_row)
    
    return results


def validate_csv_format(filepath: Union[str, Path], 
                       expected_headers: List[str]) -> bool:
    """
    Validate CSV file format.
    
    Args:
        filepath: Path to CSV file
        expected_headers: List of expected column headers
        
    Returns:
        True if format is valid
        
    Raises:
        VNEFileError: If validation fails
        
    Example:
        >>> validate_csv_format("data.csv", ["vnr_id", "arrival_time"])
    """
    path = validate_file_path(filepath, check_exists=True, expected_extension='.csv')
    
    try:
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, [])
            
            # Check if all expected headers are present
            missing_headers = set(expected_headers) - set(headers)
            if missing_headers:
                raise VNEFileError(f"Missing headers: {missing_headers}")
            
            return True
            
    except Exception as e:
        raise VNEFileError(f"CSV validation failed: {e}")


def export_metrics_to_csv(metrics: Dict[str, Any], 
                         filepath: Union[str, Path]) -> None:
    """
    Export metrics dictionary to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output CSV file path
        
    Raises:
        VNEFileError: If export fails
        
    Example:
        >>> metrics = {"acceptance_ratio": 0.85, "total_revenue": 1250.0}
        >>> export_metrics_to_csv(metrics, "results/metrics.csv")
    """
    path = Path(filepath)
    ensure_directory_exists(path.parent)
    
    try:
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])
            
            for metric_name, value in metrics.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        writer.writerow([f"{metric_name}.{sub_metric}", sub_value])
                else:
                    writer.writerow([metric_name, value])
        
        logger.info(f"Exported metrics to {path}")
        
    except Exception as e:
        raise VNEFileError(f"Failed to export metrics: {e}")


def create_experiment_directory(base_path: Union[str, Path], 
                              experiment_name: str,
                              timestamp: bool = True) -> Path:
    """
    Create directory structure for experiment results.
    
    Args:
        base_path: Base directory path
        experiment_name: Name of the experiment
        timestamp: Whether to include timestamp in directory name
        
    Returns:
        Path to created experiment directory
        
    Example:
        >>> exp_dir = create_experiment_directory("results", "test_experiment")
    """
    base = Path(base_path)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base / f"{experiment_name}_{timestamp_str}"
    else:
        exp_dir = base / experiment_name
    
    # Create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "networks").mkdir(exist_ok=True)
    (exp_dir / "vnrs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    
    # Create experiment info file
    info_file = exp_dir / "experiment_info.json"
    experiment_info = {
        'experiment_name': experiment_name,
        'created_at': datetime.now().isoformat(),
        'directory_structure': {
            'networks': 'Substrate network files',
            'vnrs': 'Virtual network request files', 
            'results': 'Embedding results',
            'metrics': 'Performance metrics'
        }
    }
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2)
    
    logger.info(f"