"""
VNR Batch Management for Virtual Network Embedding.

This module provides the VNRBatch class for managing collections of Virtual Network 
Requests (VNRs) with batch processing and organization capabilities.
"""

import logging
import csv
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from src.models.virtual_request import VirtualNetworkRequest, VNRValidationError


logger = logging.getLogger(__name__)


class VNRBatchError(Exception):
    """Base exception for VNR batch operations."""
    pass


class VNRFileFormatError(VNRBatchError):
    """Exception raised when VNR file format is invalid."""
    pass


class VNRBatch:
    """
    Manages a batch of Virtual Network Requests for experiments.
    
    This class provides utilities for organizing and managing multiple VNRs
    for batch processing scenarios. It focuses on collection management
    operations like sorting, filtering, and serialization.
    
    Note: Analysis operations like statistics and feasibility checking
    should be handled by dedicated utility modules.
    
    Attributes:
        vnrs: List of VirtualNetworkRequest instances
        batch_id: Unique identifier for the batch
        
    Example:
        >>> batch = VNRBatch(batch_id="experiment_1")
        >>> batch.add_vnr(vnr1)
        >>> batch.add_vnr(vnr2)
        >>> batch.sort_by_arrival_time()
        >>> batch.save_to_csv("vnr_batch")
        >>> loaded_batch = VNRBatch.load_from_csv("vnr_batch")
    """
    
    def __init__(self, vnrs: Optional[List[VirtualNetworkRequest]] = None, 
                 batch_id: str = "default"):
        """
        Initialize VNR batch.
        
        Args:
            vnrs: List of VNR instances (default: empty list)
            batch_id: Identifier for the batch
        """
        self.vnrs = vnrs or []
        self.batch_id = batch_id
        logger.info(f"Initialized VNR batch {batch_id} with {len(self.vnrs)} VNRs")
    
    def add_vnr(self, vnr: VirtualNetworkRequest) -> None:
        """
        Add a VNR to the batch.
        
        Args:
            vnr: VirtualNetworkRequest to add
        """
        self.vnrs.append(vnr)
        logger.debug(f"Added VNR {vnr.vnr_id} to batch {self.batch_id}")
    
    def remove_vnr(self, vnr_id: int) -> bool:
        """
        Remove a VNR from the batch.
        
        Args:
            vnr_id: ID of the VNR to remove
            
        Returns:
            True if VNR was removed, False if not found
        """
        for i, vnr in enumerate(self.vnrs):
            if vnr.vnr_id == vnr_id:
                del self.vnrs[i]
                logger.debug(f"Removed VNR {vnr_id} from batch {self.batch_id}")
                return True
        return False
    
    def get_vnr(self, vnr_id: int) -> Optional[VirtualNetworkRequest]:
        """
        Get a VNR by ID.
        
        Args:
            vnr_id: ID of the VNR to retrieve
            
        Returns:
            VirtualNetworkRequest instance or None if not found
        """
        for vnr in self.vnrs:
            if vnr.vnr_id == vnr_id:
                return vnr
        return None
    
    def sort_by_arrival_time(self, reverse: bool = False) -> None:
        """
        Sort VNRs by arrival time.
        
        Args:
            reverse: If True, sort in descending order (default: False)
        """
        self.vnrs.sort(key=lambda vnr: vnr.arrival_time, reverse=reverse)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by arrival time ({'desc' if reverse else 'asc'})")
    
    def sort_by_priority(self, reverse: bool = True) -> None:
        """
        Sort VNRs by priority.
        
        Args:
            reverse: If True, sort by descending priority (default: True)
        """
        self.vnrs.sort(key=lambda vnr: vnr.priority, reverse=reverse)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by priority ({'desc' if reverse else 'asc'})")
    
    def sort_by_holding_time(self, reverse: bool = False) -> None:
        """
        Sort VNRs by holding_time.
        
        Args:
            reverse: If True, sort by descending holding_time (default: False)
        """
        # Handle infinite holding_times by treating them as very large values for sorting
        def holding_time_key(vnr):
            return vnr.holding_time if vnr.holding_time != float('inf') else float('1e10')
        
        self.vnrs.sort(key=holding_time_key, reverse=reverse)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by holding_time ({'desc' if reverse else 'asc'})")
    
    def sort_by_node_count(self, reverse: bool = False) -> None:
        """
        Sort VNRs by number of virtual nodes.
        
        Args:
            reverse: If True, sort by descending node count (default: False)
        """
        self.vnrs.sort(key=lambda vnr: len(vnr.virtual_nodes), reverse=reverse)
        logger.debug(f"Sorted {len(self.vnrs)} VNRs by node count ({'desc' if reverse else 'asc'})")
    
    def filter_by_time_range(self, start_time: float, end_time: float) -> 'VNRBatch':
        """
        Filter VNRs by arrival time range.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (exclusive)
            
        Returns:
            New VNRBatch with filtered VNRs
        """
        filtered_vnrs = [vnr for vnr in self.vnrs 
                        if start_time <= vnr.arrival_time < end_time]
        
        new_batch = VNRBatch(filtered_vnrs, f"{self.batch_id}_time_filtered")
        logger.debug(f"Filtered {len(filtered_vnrs)}/{len(self.vnrs)} VNRs by time range [{start_time}, {end_time})")
        return new_batch
    
    def filter_by_priority_range(self, min_priority: int, max_priority: int) -> 'VNRBatch':
        """
        Filter VNRs by priority range.
        
        Args:
            min_priority: Minimum priority (inclusive)
            max_priority: Maximum priority (inclusive)
            
        Returns:
            New VNRBatch with filtered VNRs
        """
        filtered_vnrs = [vnr for vnr in self.vnrs 
                        if min_priority <= vnr.priority <= max_priority]
        
        new_batch = VNRBatch(filtered_vnrs, f"{self.batch_id}_priority_filtered")
        logger.debug(f"Filtered {len(filtered_vnrs)}/{len(self.vnrs)} VNRs by priority range [{min_priority}, {max_priority}]")
        return new_batch
    
    def filter_by_node_count_range(self, min_nodes: int, max_nodes: int) -> 'VNRBatch':
        """
        Filter VNRs by number of virtual nodes.
        
        Args:
            min_nodes: Minimum number of nodes (inclusive)
            max_nodes: Maximum number of nodes (inclusive)
            
        Returns:
            New VNRBatch with filtered VNRs
        """
        filtered_vnrs = [vnr for vnr in self.vnrs 
                        if min_nodes <= len(vnr.virtual_nodes) <= max_nodes]
        
        new_batch = VNRBatch(filtered_vnrs, f"{self.batch_id}_nodes_filtered")
        logger.debug(f"Filtered {len(filtered_vnrs)}/{len(self.vnrs)} VNRs by node count range [{min_nodes}, {max_nodes}]")
        return new_batch
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the batch.
        
        Returns:
            Dictionary with basic batch information
        """
        if not self.vnrs:
            return {
                'batch_id': self.batch_id,
                'count': 0,
                'is_empty': True
            }
        
        arrival_times = [vnr.arrival_time for vnr in self.vnrs]
        priorities = [vnr.priority for vnr in self.vnrs]
        node_counts = [len(vnr.virtual_nodes) for vnr in self.vnrs]
        link_counts = [len(vnr.virtual_links) for vnr in self.vnrs]
        holding_times = [vnr.holding_time for vnr in self.vnrs if vnr.holding_time != float('inf')]

        return {
            'batch_id': self.batch_id,
            'count': len(self.vnrs),
            'is_empty': False,
            'arrival_time_range': (min(arrival_times), max(arrival_times)),
            'priority_range': (min(priorities), max(priorities)),
            'node_count_range': (min(node_counts), max(node_counts)),
            'link_count_range': (min(link_counts), max(link_counts)),
            'holding_time_range': (min(holding_times), max(holding_times)) if holding_times else (0, 0),
            'avg_nodes_per_vnr': sum(node_counts) / len(node_counts),
            'avg_links_per_vnr': sum(link_counts) / len(link_counts)
        }
    
    def save_to_csv(self, base_filename: str) -> None:
        """
        Save VNR batch to CSV files.
        
        Creates three CSV files:
        - {base}_metadata.csv: VNR metadata (id, arrival_time, holding_time, etc.)
        - {base}_nodes.csv: Virtual node requirements
        - {base}_links.csv: Virtual link requirements  
        
        Args:
            base_filename: Base filename (without extension)
        """
        base_path = Path(base_filename)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_file = f"{base_filename}_metadata.csv"
        nodes_file = f"{base_filename}_nodes.csv"
        links_file = f"{base_filename}_links.csv"
        
        # Save VNR metadata
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'arrival_time', 'holding_time', 'priority',
                           'node_count', 'link_count'])
            
            for vnr in self.vnrs:
                writer.writerow([
                    vnr.vnr_id,
                    vnr.arrival_time,
                    vnr.holding_time if vnr.holding_time != float('inf') else -1,
                    vnr.priority,
                    len(vnr.virtual_nodes),
                    len(vnr.virtual_links)
                ])
        
        # Save virtual nodes
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'node_id', 'cpu_requirement', 'memory_requirement', 'constraints'])
            
            for vnr in self.vnrs:
                for node_id, node_req in vnr.virtual_nodes.items():
                    writer.writerow([
                        vnr.vnr_id,
                        node_id,
                        node_req.cpu_requirement,
                        node_req.memory_requirement,
                        json.dumps(node_req.node_constraints)
                    ])
        
        # Save virtual links
        with open(links_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnr_id', 'src_node', 'dst_node', 'bandwidth_requirement', 
                           'delay_constraint', 'reliability_requirement', 'constraints'])
            
            for vnr in self.vnrs:
                for (src, dst), link_req in vnr.virtual_links.items():
                    writer.writerow([
                        vnr.vnr_id,
                        src,
                        dst,
                        link_req.bandwidth_requirement,
                        link_req.delay_constraint,
                        link_req.reliability_requirement,
                        json.dumps(link_req.link_constraints)
                    ])
        
        logger.info(f"Saved VNR batch {self.batch_id} to {metadata_file}, {nodes_file}, {links_file}")
    
    @classmethod
    def load_from_csv(cls, base_filename: str, batch_id: Optional[str] = None) -> 'VNRBatch':
        """
        Load VNR batch from CSV files.
        
        Args:
            base_filename: Base filename (without extension)
            batch_id: Batch identifier (default: derived from filename)
            
        Returns:
            VNRBatch instance
            
        Raises:
            FileNotFoundError: If CSV files don't exist
            VNRFileFormatError: If file format is invalid
        """
        if batch_id is None:
            batch_id = Path(base_filename).stem
        
        metadata_file = f"{base_filename}_metadata.csv"
        nodes_file = f"{base_filename}_nodes.csv"
        links_file = f"{base_filename}_links.csv"
        
        try:
            # Load metadata first
            vnr_data = {}
            with open(metadata_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    holding_time = float(row['holding_time']) if float(row['holding_time']) != -1 else float('inf')
                    vnr_data[vnr_id] = {
                        'arrival_time': float(row['arrival_time']),
                        'holding_time': holding_time,
                        'priority': int(row['priority'])
                    }
            
            # Create VNR instances
            vnrs_dict = {}
            for vnr_id, data in vnr_data.items():
                vnr = VirtualNetworkRequest(
                    vnr_id=vnr_id,
                    arrival_time=data['arrival_time'],
                    holding_time=data['holding_time'],
                    priority=data['priority']
                )
                vnrs_dict[vnr_id] = vnr
            
            # Load virtual nodes
            with open(nodes_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnrs_dict:
                        constraints = json.loads(row['constraints']) if row['constraints'] else {}
                        vnrs_dict[vnr_id].add_virtual_node(
                            node_id=int(row['node_id']),
                            cpu_requirement=float(row['cpu_requirement']),
                            memory_requirement=float(row.get('memory_requirement', 0.0)),
                            node_constraints=constraints
                        )
            
            # Load virtual links
            with open(links_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnr_id = int(row['vnr_id'])
                    if vnr_id in vnrs_dict:
                        constraints = json.loads(row['constraints']) if row['constraints'] else {}
                        vnrs_dict[vnr_id].add_virtual_link(
                            src_node=int(row['src_node']),
                            dst_node=int(row['dst_node']),
                            bandwidth_requirement=float(row['bandwidth_requirement']),
                            delay_constraint=float(row.get('delay_constraint', 0.0)),
                            reliability_requirement=float(row.get('reliability_requirement', 0.0)),
                            link_constraints=constraints
                        )
            
            # Create batch
            batch = cls(list(vnrs_dict.values()), batch_id)
            logger.info(f"Loaded VNR batch {batch_id} with {len(batch.vnrs)} VNRs")
            return batch
            
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
            raise VNRFileFormatError(f"Error loading VNR batch: {e}")
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save VNR batch to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        batch_data = {
            'batch_id': self.batch_id,
            'created_at': datetime.now().isoformat(),
            'vnr_count': len(self.vnrs),
            'vnrs': [vnr.to_dict() for vnr in self.vnrs]
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        logger.info(f"Saved VNR batch {self.batch_id} to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'VNRBatch':
        """
        Load VNR batch from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            VNRBatch instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            VNRFileFormatError: If file format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            vnrs = [VirtualNetworkRequest.from_dict(vnr_data) for vnr_data in data['vnrs']]
            batch = cls(vnrs, data['batch_id'])
            
            logger.info(f"Loaded VNR batch {batch.batch_id} from {filepath}")
            return batch
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise VNRFileFormatError(f"Error loading VNR batch: {e}")
    
    def split_batch(self, max_vnrs_per_batch: int) -> List['VNRBatch']:
        """
        Split large batch into smaller batches.
        
        Args:
            max_vnrs_per_batch: Maximum number of VNRs per batch
            
        Returns:
            List of smaller VNRBatch instances
        """
        if max_vnrs_per_batch <= 0:
            raise ValueError("max_vnrs_per_batch must be positive")
        
        sub_batches = []
        for i in range(0, len(self.vnrs), max_vnrs_per_batch):
            sub_vnrs = self.vnrs[i:i + max_vnrs_per_batch]
            sub_batch_id = f"{self.batch_id}_part_{i // max_vnrs_per_batch + 1}"
            sub_batches.append(VNRBatch(sub_vnrs, sub_batch_id))
        
        logger.info(f"Split batch {self.batch_id} into {len(sub_batches)} sub-batches")
        return sub_batches
    
    def merge_batch(self, other: 'VNRBatch') -> 'VNRBatch':
        """
        Merge with another VNR batch.
        
        Args:
            other: Another VNRBatch to merge with
            
        Returns:
            New VNRBatch with combined VNRs
        """
        merged_vnrs = self.vnrs + other.vnrs
        merged_id = f"{self.batch_id}_merged_{other.batch_id}"
        
        merged_batch = VNRBatch(merged_vnrs, merged_id)
        logger.info(f"Merged batches {self.batch_id} and {other.batch_id} into {merged_id}")
        return merged_batch
    
    def __len__(self) -> int:
        """Return number of VNRs in the batch."""
        return len(self.vnrs)
    
    def __iter__(self):
        """Iterate over VNRs in the batch."""
        return iter(self.vnrs)
    
    def __getitem__(self, index: int) -> VirtualNetworkRequest:
        """Get VNR by index."""
        return self.vnrs[index]
    
    def __bool__(self) -> bool:
        """Return True if batch contains VNRs."""
        return len(self.vnrs) > 0
    
    def __str__(self) -> str:
        """String representation of the batch."""
        info = self.get_basic_info()
        if info['is_empty']:
            return f"VNRBatch({self.batch_id}: empty)"
        
        return (f"VNRBatch({self.batch_id}: {info['count']} VNRs, "
                f"arrival_time={info['arrival_time_range'][0]:.1f}-{info['arrival_time_range'][1]:.1f}, "
                f"avg_nodes={info['avg_nodes_per_vnr']:.1f})")
    
    def __repr__(self) -> str:
        """Detailed representation of the batch."""
        return f"VNRBatch(batch_id='{self.batch_id}', vnr_count={len(self.vnrs)})"
