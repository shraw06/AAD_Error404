#!/usr/bin/env python3
"""
Network Dataset Loader
======================

Unified loader for various network dataset formats:
- SNAP format (edge lists)
- CSV format (source, target, capacity)
- VFOA temporal networks (Visual Focus of Attention)

Supports auto-detection and conversion to FlowGraph format.
"""

import os
import csv
from typing import Tuple, List, Dict, Optional

try:
    import numpy as np
    import pandas as pd
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    np = None
    pd = None

from .graph import FlowGraph


# =============================================================================
# METADATA AND DISCOVERY
# =============================================================================

def discover_all_datasets(data_dir: str = "data") -> Dict[str, List[str]]:
    """
    Discover all available datasets in the data directory.
    
    Returns:
        Dictionary with dataset types as keys and lists of paths as values.
    """
    datasets = {
        'snap': [],
        'csv': [],
        'vfoa': [],
        'other': []
    }
    
    if not os.path.exists(data_dir):
        return datasets
    
    # Find regular CSV and TXT files
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        if os.path.isfile(filepath):
            if filename.endswith('.txt'):
                datasets['snap'].append(filepath)
            elif filename.endswith('.csv'):
                datasets['csv'].append(filepath)
    
    # Find VFOA networks
    vfoa_base = os.path.join(data_dir, "comm-f2f-Resistance-network", "comm-f2f-Resistance")
    if os.path.exists(vfoa_base):
        network_dir = os.path.join(vfoa_base, "network")
        if os.path.exists(network_dir):
            # Get unique network IDs
            network_files = [f for f in os.listdir(network_dir) if f.startswith('network') and f.endswith('.csv')]
            network_ids = set()
            for f in network_files:
                # Extract network ID (e.g., "network0" from "network0.csv" or "network0_weighted.csv")
                if '_weighted' in f:
                    net_id = f.replace('_weighted.csv', '')
                else:
                    net_id = f.replace('.csv', '')
                network_ids.add(net_id)
            
            datasets['vfoa'] = sorted(list(network_ids))
    
    return datasets


def get_dataset_metadata(filepath: str, dataset_type: str = None) -> Dict:
    """
    Get metadata about a dataset without fully loading it.
    
    Args:
        filepath: Path to the dataset
        dataset_type: Type of dataset ('snap', 'csv', 'vfoa', or None for auto-detect)
    
    Returns:
        Dictionary with metadata (nodes, edges, format, etc.)
    """
    metadata = {
        'path': filepath,
        'type': dataset_type or detect_format(filepath),
        'nodes': 0,
        'edges': 0,
        'weighted': False,
        'temporal': False,
        'description': ''
    }
    
    try:
        if metadata['type'] == 'snap':
            # Count lines for edge estimate
            with open(filepath, 'r') as f:
                edges = sum(1 for line in f if line.strip() and not line.startswith('#'))
            metadata['edges'] = edges
            metadata['description'] = 'SNAP edge list format'
            
        elif metadata['type'] == 'csv':
            if not HAS_DEPENDENCIES:
                metadata['description'] = 'CSV format (requires pandas)'
                return metadata
            df = pd.read_csv(filepath, nrows=5)
            metadata['weighted'] = len(df.columns) >= 3
            metadata['description'] = 'CSV format (source, target' + (', capacity)' if metadata['weighted'] else ')')
            
        elif metadata['type'] == 'vfoa':
            # VFOA networks - read metadata from network_list.csv
            if not HAS_DEPENDENCIES:
                metadata['description'] = 'VFOA temporal network (requires pandas)'
                return metadata
            vfoa_base = os.path.dirname(os.path.dirname(filepath))
            meta_file = os.path.join(vfoa_base, "network_list.csv")
            if os.path.exists(meta_file):
                net_id = os.path.basename(filepath)
                meta_df = pd.read_csv(meta_file)
                # Extract network number
                net_num = int(''.join(filter(str.isdigit, net_id)))
                row = meta_df[meta_df['NETWORK'] == net_num]
                if not row.empty:
                    metadata['nodes'] = int(row['NUMBER_OF_PARTICIPANTS'].values[0])
            
            metadata['temporal'] = True
            metadata['weighted'] = '_weighted' in filepath
            metadata['description'] = 'VFOA temporal network (Visual Focus of Attention)'
    
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata


def list_vfoa_networks(data_dir: str = "data"):
    """
    List all VFOA networks with their metadata.
    
    Returns:
        DataFrame with network information (or empty DataFrame if pandas not available)
    """
    if not HAS_DEPENDENCIES:
        print("[ERROR] This function requires pandas and numpy")
        return None
    
    vfoa_base = os.path.join(data_dir, "comm-f2f-Resistance-network", "comm-f2f-Resistance")
    meta_file = os.path.join(vfoa_base, "network_list.csv")
    
    if not os.path.exists(meta_file):
        return pd.DataFrame()
    
    df = pd.read_csv(meta_file)
    df['network_path'] = df['NETWORK'].apply(
        lambda x: os.path.join(vfoa_base, "network", f"network{x}")
    )
    
    return df


# =============================================================================
# FORMAT DETECTION
# =============================================================================

def detect_format(filepath: str) -> str:
    """
    Auto-detect the format of a network dataset.
    
    Returns:
        'snap', 'csv', 'vfoa', or 'unknown'
    """
    if not os.path.exists(filepath):
        return 'unknown'
    
    # Check if it's a VFOA network
    if 'comm-f2f-Resistance' in filepath and 'network' in filepath:
        return 'vfoa'
    
    # Check file extension
    if filepath.endswith('.txt'):
        return 'snap'
    
    if filepath.endswith('.csv'):
        # Peek at the file to determine if it's a standard CSV or VFOA
        try:
            df = pd.read_csv(filepath, nrows=1)
            if 'TIME' in df.columns and 'P1_TO_LAPTOP' in df.columns:
                return 'vfoa'
            else:
                return 'csv'
        except:
            return 'csv'
    
    return 'unknown'


# =============================================================================
# LOADERS
# =============================================================================

def load_snap_network(filepath: str, max_capacity: int = 100) -> Tuple[FlowGraph, int, int]:
    """
    Load a SNAP-format edge list (simple edge pairs).
    
    Args:
        filepath: Path to the SNAP file
        max_capacity: Default capacity for edges (default: 100)
    
    Returns:
        (FlowGraph, source, sink)
    """
    edges = []
    nodes = set()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
    
    # Create node mapping (in case nodes aren't 0-indexed)
    node_list = sorted(list(nodes))
    node_map = {old: new for new, old in enumerate(node_list)}
    n = len(node_list)
    
    # Create graph
    g = FlowGraph(n)
    for u, v in edges:
        g.add_edge(node_map[u], node_map[v], max_capacity)
    
    # Use first and last nodes as source and sink
    source = 0
    sink = n - 1
    
    return g, source, sink


def load_csv_network(filepath: str, default_capacity: int = 100) -> Tuple[FlowGraph, int, int]:
    """
    Load a CSV-format network (source, target, [capacity]).
    
    Args:
        filepath: Path to the CSV file
        default_capacity: Default capacity if not specified
    
    Returns:
        (FlowGraph, source, sink)
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("This function requires pandas. Install with: pip install pandas")
    
    df = pd.read_csv(filepath, header=None)
    
    # Determine if weighted
    has_capacity = len(df.columns) >= 3
    
    nodes = set()
    edges = []
    
    for _, row in df.iterrows():
        u, v = int(row[0]), int(row[1])
        cap = int(row[2]) if has_capacity else default_capacity
        edges.append((u, v, cap))
        nodes.add(u)
        nodes.add(v)
    
    # Create node mapping
    node_list = sorted(list(nodes))
    node_map = {old: new for new, old in enumerate(node_list)}
    n = len(node_list)
    
    # Create graph
    g = FlowGraph(n)
    for u, v, cap in edges:
        g.add_edge(node_map[u], node_map[v], cap)
    
    # Use first and last nodes as source and sink
    source = 0
    sink = n - 1
    
    return g, source, sink


def load_vfoa_network(network_id: str, data_dir: str = "data", 
                      weighted: bool = True, capacity_scale: float = 100.0) -> Tuple[FlowGraph, int, int, Dict]:
    """
    Load a VFOA temporal network and aggregate into a static network.
    
    Args:
        network_id: Network identifier (e.g., "network0" or "0")
        data_dir: Base data directory
        weighted: Use weighted version (probabilities) or unweighted (binary)
        capacity_scale: Scale factor for converting probabilities to capacities
    
    Returns:
        (FlowGraph, source, sink, metadata)
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("This function requires pandas and numpy. Install with: pip install pandas numpy")
    # Normalize network_id
    if not network_id.startswith('network'):
        network_id = f'network{network_id}'
    
    # Build path
    vfoa_base = os.path.join(data_dir, "comm-f2f-Resistance-network", "comm-f2f-Resistance")
    network_dir = os.path.join(vfoa_base, "network")
    
    suffix = '_weighted.csv' if weighted else '.csv'
    filepath = os.path.join(network_dir, network_id + suffix)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VFOA network not found: {filepath}")
    
    # Load the temporal data
    df = pd.read_csv(filepath)
    
    # Extract number of participants from column count
    # Columns: TIME, P1_TO_LAPTOP, P1_TO_P1, ..., P1_TO_PN, P2_TO_LAPTOP, ...
    # Each participant has (N+1) columns (laptop + N participants)
    num_cols = len(df.columns) - 1  # Exclude TIME column
    n_participants = int(np.sqrt(num_cols))
    
    # Get metadata
    meta_file = os.path.join(vfoa_base, "network_list.csv")
    metadata = {'participants': n_participants, 'timesteps': len(df), 'weighted': weighted}
    
    if os.path.exists(meta_file):
        meta_df = pd.read_csv(meta_file)
        net_num = int(''.join(filter(str.isdigit, network_id)))
        row = meta_df[meta_df['NETWORK'] == net_num]
        if not row.empty:
            metadata['participants'] = int(row['NUMBER_OF_PARTICIPANTS'].values[0])
            n_participants = metadata['participants']
    
    # Aggregate temporal data into static network
    # Strategy: Average attention probabilities over all timesteps
    # Create adjacency matrix
    n = n_participants
    capacity_matrix = np.zeros((n, n))
    
    for t_idx in range(len(df)):
        row = df.iloc[t_idx]
        
        # Parse each participant's attention
        for i in range(n):
            start_col = 1 + i * (n + 1)  # Skip TIME, then skip laptop column
            
            for j in range(n):
                col_idx = start_col + 1 + j  # +1 to skip laptop
                
                if col_idx < len(row):
                    value = float(row.iloc[col_idx])
                    if i != j:  # No self-loops
                        capacity_matrix[i][j] += value
    
    # Average over timesteps
    capacity_matrix /= len(df)
    
    # Scale to integer capacities
    capacity_matrix = (capacity_matrix * capacity_scale).astype(int)
    
    # Create FlowGraph
    g = FlowGraph(n)
    for i in range(n):
        for j in range(n):
            if capacity_matrix[i][j] > 0:
                g.add_edge(i, j, int(capacity_matrix[i][j]))
    
    # Use first and last participants as source and sink
    source = 0
    sink = n - 1
    
    return g, source, sink, metadata


# =============================================================================
# UNIFIED LOADER
# =============================================================================

def load_network(filepath: str, **kwargs) -> Tuple[FlowGraph, int, int, Dict]:
    """
    Unified loader that auto-detects format and loads the network.
    
    Args:
        filepath: Path to the network file or VFOA network ID
        **kwargs: Additional arguments passed to specific loaders
    
    Returns:
        (FlowGraph, source, sink, metadata)
    """
    format_type = detect_format(filepath)
    metadata = {'path': filepath, 'type': format_type}
    
    if format_type == 'snap':
        g, s, t = load_snap_network(filepath, kwargs.get('max_capacity', 100))
        metadata['nodes'] = g.n
        return g, s, t, metadata
    
    elif format_type == 'csv':
        g, s, t = load_csv_network(filepath, kwargs.get('default_capacity', 100))
        metadata['nodes'] = g.n
        return g, s, t, metadata
    
    elif format_type == 'vfoa':
        # Extract network ID from filepath
        basename = os.path.basename(filepath)
        network_id = basename.replace('_weighted', '').replace('.csv', '')
        
        data_dir = kwargs.get('data_dir', 'data')
        weighted = kwargs.get('weighted', True)
        capacity_scale = kwargs.get('capacity_scale', 100.0)
        
        g, s, t, vfoa_meta = load_vfoa_network(network_id, data_dir, weighted, capacity_scale)
        metadata.update(vfoa_meta)
        metadata['nodes'] = g.n
        return g, s, t, metadata
    
    else:
        raise ValueError(f"Unknown or unsupported format: {format_type}")


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def load_all_vfoa_networks(data_dir: str = "data", weighted: bool = True, 
                           capacity_scale: float = 100.0, 
                           progress_callback=None) -> List[Tuple[str, FlowGraph, int, int, Dict]]:
    """
    Load all VFOA networks in batch.
    
    Args:
        data_dir: Base data directory
        weighted: Use weighted version
        capacity_scale: Scale factor for capacities
        progress_callback: Optional callback function for progress updates
    
    Returns:
        List of (network_id, FlowGraph, source, sink, metadata) tuples
    """
    meta_df = list_vfoa_networks(data_dir)
    
    if meta_df.empty:
        return []
    
    results = []
    total = len(meta_df)
    
    for idx, row in meta_df.iterrows():
        network_id = f"network{row['NETWORK']}"
        
        if progress_callback:
            progress_callback(idx + 1, total, network_id)
        
        try:
            g, s, t, metadata = load_vfoa_network(
                network_id, data_dir, weighted, capacity_scale
            )
            results.append((network_id, g, s, t, metadata))
        except Exception as e:
            print(f"Warning: Failed to load {network_id}: {e}")
            continue
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_dataset_info(filepath: str):
    """Print detailed information about a dataset."""
    metadata = get_dataset_metadata(filepath)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    print(f"Type:        {metadata['type']}")
    print(f"Path:        {filepath}")
    
    if metadata.get('nodes'):
        print(f"Nodes:       {metadata['nodes']}")
    if metadata.get('edges'):
        print(f"Edges:       {metadata['edges']}")
    if metadata.get('weighted'):
        print(f"Weighted:    Yes")
    if metadata.get('temporal'):
        print(f"Temporal:    Yes")
    
    print(f"Description: {metadata['description']}")
    print(f"{'='*60}\n")
