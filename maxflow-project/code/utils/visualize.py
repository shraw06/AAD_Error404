import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


# -----------------------------------------------------------
# 1. VISUALIZE GRAPH STRUCTURE (WITH CAPACITIES)
# -----------------------------------------------------------
def visualize_graph(graph, title="Graph Structure"):
    G = nx.DiGraph()

    # add edges with capacities
    for u in range(graph.n):
        for (v, cap, rev) in graph.adj[u]:
            if cap > 0:  # only forward edges
                G.add_edge(u, v, capacity=cap)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))

    nx.draw(G, pos, with_labels=True, node_size=600, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, "capacity")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()


# -----------------------------------------------------------
# 2. VISUALIZE FLOW AFTER MAX-FLOW
# -----------------------------------------------------------
def visualize_flow(original_graph, residual_graph, title="Flow Visualization"):

    """
    Flow = original capacity - residual forward capacity
    """
    G = nx.DiGraph()

    for u in range(original_graph.n):
        for i, (v, cap, rev) in enumerate(original_graph.adj[u]):
            orig_cap = cap
            res_cap = residual_graph.adj[u][i][1]
            flow = orig_cap - res_cap

            if flow > 0:
                G.add_edge(u, v, flow=flow)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))

    # edge thickness proportional to flow
    flows = [G[u][v]["flow"] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=[f/2 for f in flows], arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, "flow")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()


# -----------------------------------------------------------
# 3. VISUALIZE MIN-CUT
# -----------------------------------------------------------
def visualize_min_cut(graph, S, cut_edges, title="Min-Cut Visualization"):
    G = nx.DiGraph()

    for u in range(graph.n):
        for (v, cap, rev) in graph.adj[u]:
            if cap > 0:
                G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))

    # Color nodes: S = green, T = red
    node_colors = ["lightgreen" if S[u] else "lightcoral" for u in range(graph.n)]

    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        node_size=600, arrowsize=20
    )

    # Highlight cut edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cut_edges,
        edge_color="red",
        width=3
    )

    plt.title(title)
    plt.show()


# -----------------------------------------------------------
# 4. VISUALIZE NETWORK COMPARISON (SIDE-BY-SIDE)
# -----------------------------------------------------------
def visualize_network_comparison(networks: List[Tuple[str, object]], max_networks: int = 4):
    """
    Visualize multiple networks side-by-side for comparison.
    
    Args:
        networks: List of (name, FlowGraph) tuples
        max_networks: Maximum number of networks to display (default: 4)
    """
    n_networks = min(len(networks), max_networks)
    
    if n_networks == 0:
        print("No networks to visualize")
        return
    
    # Calculate subplot layout
    cols = min(2, n_networks)
    rows = (n_networks + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
    
    # Handle single subplot case
    if n_networks == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_networks > 1 else axes
    
    for idx, (name, graph) in enumerate(networks[:max_networks]):
        ax = axes[idx] if n_networks > 1 else axes[0]
        
        G = nx.DiGraph()
        
        # Add edges with capacities
        for u in range(graph.n):
            for (v, cap, rev) in graph.adj[u]:
                if cap > 0:  # only forward edges
                    G.add_edge(u, v, capacity=cap)
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw on specific subplot
        plt.sca(ax)
        nx.draw(G, pos, with_labels=True, node_size=400, 
                arrowsize=15, ax=ax, node_color='lightblue')
        
        # Add edge labels only if not too many edges
        if G.number_of_edges() <= 30:
            edge_labels = nx.get_edge_attributes(G, "capacity")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        ax.set_title(f"{name}\n({graph.n} nodes, {G.number_of_edges()} edges)", 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_networks, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 5. VISUALIZE NETWORK STATISTICS
# -----------------------------------------------------------
def visualize_network_statistics(networks: List[Tuple[str, object]]):
    """
    Visualize statistical comparison of multiple networks.
    
    Args:
        networks: List of (name, FlowGraph) tuples
    """
    if not networks:
        print("No networks to analyze")
        return
    
    # Compute statistics for each network
    stats_data = {
        'names': [],
        'nodes': [],
        'edges': [],
        'avg_degree': [],
        'total_capacity': [],
        'avg_capacity': []
    }
    
    for name, graph in networks:
        stats_data['names'].append(name)
        stats_data['nodes'].append(graph.n)
        
        # Count edges and capacities
        edge_count = 0
        capacities = []
        
        for u in range(graph.n):
            for (v, cap, rev) in graph.adj[u]:
                if cap > 0:
                    edge_count += 1
                    capacities.append(cap)
        
        stats_data['edges'].append(edge_count)
        stats_data['avg_degree'].append(edge_count / graph.n if graph.n > 0 else 0)
        stats_data['total_capacity'].append(sum(capacities))
        stats_data['avg_capacity'].append(np.mean(capacities) if capacities else 0)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Network Statistics Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Node count
    axes[0, 0].bar(stats_data['names'], stats_data['nodes'], color='skyblue')
    axes[0, 0].set_title('Number of Nodes')
    axes[0, 0].set_ylabel('Nodes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Edge count
    axes[0, 1].bar(stats_data['names'], stats_data['edges'], color='lightcoral')
    axes[0, 1].set_title('Number of Edges')
    axes[0, 1].set_ylabel('Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average degree
    axes[0, 2].bar(stats_data['names'], stats_data['avg_degree'], color='lightgreen')
    axes[0, 2].set_title('Average Degree')
    axes[0, 2].set_ylabel('Avg Degree')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Total capacity
    axes[1, 0].bar(stats_data['names'], stats_data['total_capacity'], color='gold')
    axes[1, 0].set_title('Total Capacity')
    axes[1, 0].set_ylabel('Total Capacity')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Average capacity
    axes[1, 1].bar(stats_data['names'], stats_data['avg_capacity'], color='orchid')
    axes[1, 1].set_title('Average Capacity')
    axes[1, 1].set_ylabel('Avg Capacity')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Node vs Edge scatter
    axes[1, 2].scatter(stats_data['nodes'], stats_data['edges'], 
                       s=100, c=range(len(stats_data['names'])), cmap='viridis')
    for i, name in enumerate(stats_data['names']):
        axes[1, 2].annotate(name, (stats_data['nodes'][i], stats_data['edges'][i]),
                           fontsize=8, ha='right')
    axes[1, 2].set_title('Nodes vs Edges')
    axes[1, 2].set_xlabel('Nodes')
    axes[1, 2].set_ylabel('Edges')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 6. VISUALIZE VFOA ATTENTION HEATMAP
# -----------------------------------------------------------
def plot_vfoa_heatmap(capacity_matrix: np.ndarray, title: str = "VFOA Attention Matrix"):
    """
    Visualize VFOA attention patterns as a heatmap.
    
    Args:
        capacity_matrix: NxN matrix of attention probabilities/capacities
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(capacity_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    n = capacity_matrix.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f'P{i+1}' for i in range(n)])
    ax.set_yticklabels([f'P{i+1}' for i in range(n)])
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Capacity', rotation=270, labelpad=20)
    
    # Add title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Participant')
    ax.set_ylabel('Source Participant')
    
    # Add text annotations for non-zero values
    if n <= 10:  # Only annotate if matrix is small enough
        for i in range(n):
            for j in range(n):
                if capacity_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{capacity_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 7. VISUALIZE FLOW COMPARISON
# -----------------------------------------------------------
def visualize_flow_comparison(results: List[Tuple[str, int, float]]):
    """
    Visualize comparison of max-flow results across networks.
    
    Args:
        results: List of (network_name, max_flow, runtime) tuples
    """
    if not results:
        print("No results to visualize")
        return
    
    names = [r[0] for r in results]
    flows = [r[1] for r in results]
    runtimes = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Max-Flow Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Max flow values
    bars1 = ax1.bar(names, flows, color='steelblue')
    ax1.set_title('Maximum Flow Values')
    ax1.set_ylabel('Max Flow')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Plot 2: Runtime
    bars2 = ax2.bar(names, runtimes, color='coral')
    ax2.set_title('Computation Runtime')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
