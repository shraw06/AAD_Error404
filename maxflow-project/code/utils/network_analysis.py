#!/usr/bin/env python3
"""
Network Analysis and Comparison
================================

Tools for analyzing and comparing network structures and max-flow results.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .graph import FlowGraph


# =============================================================================
# BASIC NETWORK STATISTICS
# =============================================================================

def compute_network_stats(g: FlowGraph) -> Dict:
    """
    Compute basic statistics about a network.
    
    Args:
        g: FlowGraph instance
    
    Returns:
        Dictionary with network statistics
    """
    stats = {
        'nodes': g.n,
        'edges': 0,
        'total_capacity': 0,
        'avg_capacity': 0,
        'max_capacity': 0,
        'min_capacity': float('inf'),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0
    }
    
    # Count edges and capacities
    edge_count = 0
    capacities = []
    out_degrees = [0] * g.n
    in_degrees = [0] * g.n
    
    for u in range(g.n):
        for v, cap, rev in g.adj[u]:
            if cap > 0:  # Only count forward edges
                edge_count += 1
                capacities.append(cap)
                out_degrees[u] += 1
                in_degrees[v] += 1
    
    stats['edges'] = edge_count
    
    if capacities:
        stats['total_capacity'] = sum(capacities)
        stats['avg_capacity'] = np.mean(capacities)
        stats['max_capacity'] = max(capacities)
        stats['min_capacity'] = min(capacities)
    
    if g.n > 0:
        stats['avg_degree'] = edge_count / g.n
        stats['max_out_degree'] = max(out_degrees) if out_degrees else 0
        stats['max_in_degree'] = max(in_degrees) if in_degrees else 0
    
    return stats


def compute_degree_distribution(g: FlowGraph) -> Tuple[List[int], List[int]]:
    """
    Compute in-degree and out-degree distributions.
    
    Returns:
        (in_degrees, out_degrees) as lists
    """
    in_degrees = [0] * g.n
    out_degrees = [0] * g.n
    
    for u in range(g.n):
        for v, cap, rev in g.adj[u]:
            if cap > 0:
                out_degrees[u] += 1
                in_degrees[v] += 1
    
    return in_degrees, out_degrees


def compute_capacity_distribution(g: FlowGraph) -> List[int]:
    """
    Get list of all edge capacities for distribution analysis.
    
    Returns:
        List of capacities
    """
    capacities = []
    
    for u in range(g.n):
        for v, cap, rev in g.adj[u]:
            if cap > 0:
                capacities.append(cap)
    
    return capacities


# =============================================================================
# FLOW ANALYSIS
# =============================================================================

def analyze_flow(original: FlowGraph, residual: FlowGraph) -> Dict:
    """
    Analyze the flow on a network after max-flow computation.
    
    Args:
        original: Original FlowGraph
        residual: Residual graph after max-flow
    
    Returns:
        Dictionary with flow statistics
    """
    stats = {
        'total_flow': 0,
        'saturated_edges': 0,
        'active_edges': 0,
        'flow_distribution': []
    }
    
    flows = []
    
    for u in range(original.n):
        for i, (v, orig_cap, rev) in enumerate(original.adj[u]):
            if orig_cap > 0:
                res_cap = residual.adj[u][i][1]
                flow = orig_cap - res_cap
                
                if flow > 0:
                    stats['active_edges'] += 1
                    flows.append(flow)
                    
                    if flow == orig_cap:  # Saturated edge
                        stats['saturated_edges'] += 1
    
    stats['flow_distribution'] = flows
    
    return stats


# =============================================================================
# NETWORK COMPARISON
# =============================================================================

def compare_networks(networks: List[Tuple[str, FlowGraph]]) -> pd.DataFrame:
    """
    Compare multiple networks based on their structural properties.
    
    Args:
        networks: List of (name, FlowGraph) tuples
    
    Returns:
        DataFrame with comparison statistics
    """
    comparison_data = []
    
    for name, g in networks:
        stats = compute_network_stats(g)
        stats['name'] = name
        comparison_data.append(stats)
    
    df = pd.DataFrame(comparison_data)
    
    # Reorder columns
    cols = ['name', 'nodes', 'edges', 'avg_degree', 'total_capacity', 
            'avg_capacity', 'max_capacity', 'min_capacity',
            'max_out_degree', 'max_in_degree']
    
    return df[cols]


def compare_flow_results(results: List[Tuple[str, int, float, Dict]]) -> pd.DataFrame:
    """
    Compare max-flow results across multiple networks.
    
    Args:
        results: List of (name, max_flow, runtime, metadata) tuples
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for name, max_flow, runtime, metadata in results:
        row = {
            'Network': name,
            'Max Flow': max_flow,
            'Runtime (s)': runtime,
            'Nodes': metadata.get('nodes', 'N/A'),
            'Edges': metadata.get('edges', 'N/A')
        }
        
        if 'nodes' in metadata and metadata['nodes'] > 0:
            row['Flow/Node'] = max_flow / metadata['nodes']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    return df


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def print_network_summary(name: str, g: FlowGraph, s: int, t: int):
    """Print a formatted summary of a network."""
    stats = compute_network_stats(g)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Nodes:           {stats['nodes']}")
    print(f"Edges:           {stats['edges']}")
    print(f"Source:          {s}")
    print(f"Sink:            {t}")
    print(f"Avg Degree:      {stats['avg_degree']:.2f}")
    print(f"Total Capacity:  {stats['total_capacity']}")
    print(f"Avg Capacity:    {stats['avg_capacity']:.2f}")
    print(f"Max Capacity:    {stats['max_capacity']}")
    print(f"Min Capacity:    {stats['min_capacity']}")
    print(f"{'='*60}\n")


def print_comparison_table(df: pd.DataFrame, title: str = "Network Comparison"):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def compute_structural_similarity(g1: FlowGraph, g2: FlowGraph) -> float:
    """
    Compute structural similarity between two networks.
    Simple metric based on node/edge count similarity.
    
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    stats1 = compute_network_stats(g1)
    stats2 = compute_network_stats(g2)
    
    # Normalize differences
    node_sim = 1.0 - abs(stats1['nodes'] - stats2['nodes']) / max(stats1['nodes'], stats2['nodes'])
    edge_sim = 1.0 - abs(stats1['edges'] - stats2['edges']) / max(stats1['edges'], stats2['edges'])
    
    # Average similarity
    return (node_sim + edge_sim) / 2.0


def rank_networks_by_size(networks: List[Tuple[str, FlowGraph]]) -> List[Tuple[str, int, int]]:
    """
    Rank networks by size (nodes and edges).
    
    Returns:
        List of (name, nodes, edges) sorted by size
    """
    sizes = []
    
    for name, g in networks:
        stats = compute_network_stats(g)
        sizes.append((name, stats['nodes'], stats['edges']))
    
    # Sort by nodes, then edges
    sizes.sort(key=lambda x: (x[1], x[2]))
    
    return sizes


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_comparison_csv(df: pd.DataFrame, filepath: str):
    """Export comparison results to CSV."""
    df.to_csv(filepath, index=False)
    print(f"[OK] Comparison saved to: {filepath}")


def export_network_stats_csv(networks: List[Tuple[str, FlowGraph]], filepath: str):
    """Export network statistics to CSV."""
    df = compare_networks(networks)
    df.to_csv(filepath, index=False)
    print(f"[OK] Network statistics saved to: {filepath}")
