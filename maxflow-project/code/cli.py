#!/usr/bin/env python3
"""
Interactive CLI for the MaxFlow Project
========================================

A comprehensive menu-driven interface for:
- Visualizing random graphs and real datasets
- Running algorithms on custom graphs
- Executing benchmark experiments
- Plotting results

Usage:
    python3 code/cli.py
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Algorithm imports
from algos.ford_fulkerson import ford_fulkerson
from algos.edmonds_karp import edmonds_karp
from algos.dinic import dinic
from algos.push_relabel import push_relabel
from algos.capacity_scaling import capacity_scaling
from algos.greedy_fattest import greedy_fattest

# Utility imports
from utils.graph import FlowGraph
from utils.generators import (
    random_sparse_graph,
    random_dense_graph,
    bipartite_graph,
    grid_graph
)
from utils.datasets import load_csv, load_snap
from utils.mincut import extract_min_cut

# Lazy imports for visualization (requires networkx/matplotlib)
visualize_graph = None
visualize_flow = None
visualize_min_cut = None

# Lazy imports for benchmark plots
plot_runtime_bar = None
plot_runtime_vs_n = None
plot_runtime_vs_density = None

# Lazy imports for benchmarking
benchmark_single_graph = None
save_results = None

# Lazy imports for experiments
experiment_sparse_sizes = None
experiment_density = None
experiment_bipartite = None
experiment_grid = None
experiment_capacity_distribution = None

# Lazy imports for network datasets
discover_all_datasets = None
load_network = None
list_vfoa_networks = None
load_vfoa_network = None
load_all_vfoa_networks = None
print_dataset_info = None

# Lazy imports for network analysis
compare_networks = None
compare_flow_results = None
print_network_summary = None
print_comparison_table = None

# Lazy imports for comparison visualizations
visualize_network_comparison = None
visualize_network_statistics = None
plot_vfoa_heatmap = None
visualize_flow_comparison = None


def _load_visualization():
    """Lazy load visualization functions."""
    global visualize_graph, visualize_flow, visualize_min_cut
    if visualize_graph is None:
        try:
            from utils.visualize import (
                visualize_graph as vg,
                visualize_flow as vf,
                visualize_min_cut as vmc
            )
            visualize_graph = vg
            visualize_flow = vf
            visualize_min_cut = vmc
        except ImportError as e:
            print(f"\n[ERROR] Visualization requires additional packages: {e}")
            print("  Install with: pip install networkx matplotlib")
            return False
    return True


def _load_plotting():
    """Lazy load plotting functions."""
    global plot_runtime_bar, plot_runtime_vs_n, plot_runtime_vs_density
    if plot_runtime_bar is None:
        try:
            from benchmark.plots import (
                plot_runtime_bar as prb,
                plot_runtime_vs_n as prvn,
                plot_runtime_vs_density as prvd
            )
            plot_runtime_bar = prb
            plot_runtime_vs_n = prvn
            plot_runtime_vs_density = prvd
        except ImportError as e:
            print(f"\n[ERROR] Plotting requires additional packages: {e}")
            print("  Install with: pip install pandas matplotlib")
            return False
    return True


def _load_benchmarking():
    """Lazy load benchmarking functions."""
    global benchmark_single_graph, save_results
    if benchmark_single_graph is None:
        try:
            from benchmark.runner import (
                benchmark_single_graph as bsg,
                save_results as sr
            )
            benchmark_single_graph = bsg
            save_results = sr
        except ImportError as e:
            print(f"\n[ERROR] Benchmarking requires pandas: {e}")
            print("  Install with: pip install pandas")
            return False
    return True


def _load_experiments():
    """Lazy load experiment functions."""
    global experiment_sparse_sizes, experiment_density
    global experiment_bipartite, experiment_grid, experiment_capacity_distribution
    if experiment_sparse_sizes is None:
        try:
            from run_all_experiments import (
                experiment_sparse_sizes as ess,
                experiment_density as ed,
                experiment_bipartite as eb,
                experiment_grid as eg,
                experiment_capacity_distribution as ecd
            )
            experiment_sparse_sizes = ess
            experiment_density = ed
            experiment_bipartite = eb
            experiment_grid = eg
            experiment_capacity_distribution = ecd
        except ImportError as e:
            print(f"\n[ERROR] Experiments require pandas: {e}")
            print("  Install with: pip install pandas")
            return False
    return True


def _load_network_datasets():
    """Lazy load network dataset functions."""
    global discover_all_datasets, load_network, list_vfoa_networks
    global load_vfoa_network, load_all_vfoa_networks, print_dataset_info
    if discover_all_datasets is None:
        try:
            from utils.network_datasets import (
                discover_all_datasets as dad,
                load_network as ln,
                list_vfoa_networks as lvn,
                load_vfoa_network as lvfoa,
                load_all_vfoa_networks as lavfoa,
                print_dataset_info as pdi
            )
            discover_all_datasets = dad
            load_network = ln
            list_vfoa_networks = lvn
            load_vfoa_network = lvfoa
            load_all_vfoa_networks = lavfoa
            print_dataset_info = pdi
        except ImportError as e:
            print(f"\n[ERROR] Network datasets require pandas: {e}")
            print("  Install with: pip install pandas numpy")
            return False
    return True


def _load_network_analysis():
    """Lazy load network analysis functions."""
    global compare_networks, compare_flow_results
    global print_network_summary, print_comparison_table
    if compare_networks is None:
        try:
            from utils.network_analysis import (
                compare_networks as cn,
                compare_flow_results as cfr,
                print_network_summary as pns,
                print_comparison_table as pct
            )
            compare_networks = cn
            compare_flow_results = cfr
            print_network_summary = pns
            print_comparison_table = pct
        except ImportError as e:
            print(f"\n[ERROR] Network analysis requires pandas: {e}")
            print("  Install with: pip install pandas numpy")
            return False
    return True


def _load_comparison_visualizations():
    """Lazy load comparison visualization functions."""
    global visualize_network_comparison, visualize_network_statistics
    global plot_vfoa_heatmap, visualize_flow_comparison
    if visualize_network_comparison is None:
        try:
            from utils.visualize import (
                visualize_network_comparison as vnc,
                visualize_network_statistics as vns,
                plot_vfoa_heatmap as pvh,
                visualize_flow_comparison as vfc
            )
            visualize_network_comparison = vnc
            visualize_network_statistics = vns
            plot_vfoa_heatmap = pvh
            visualize_flow_comparison = vfc
        except ImportError as e:
            print(f"\n[ERROR] Comparison visualizations require additional packages: {e}")
            print("  Install with: pip install networkx matplotlib numpy")
            return False
    return True

# Available algorithms
ALGORITHMS = {
    "1": ("Ford-Fulkerson", ford_fulkerson),
    "2": ("Edmonds-Karp", edmonds_karp),
    "3": ("Dinic", dinic),
    "4": ("Push-Relabel", push_relabel),
    "5": ("Capacity-Scaling", capacity_scaling),
    "6": ("Greedy-Fattest", greedy_fattest)
}


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_int_input(prompt, min_val=None, max_val=None, default=None):
    """Get validated integer input from user."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"[ERROR] Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"[ERROR] Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")


def get_float_input(prompt, min_val=None, max_val=None, default=None):
    """Get validated float input from user."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"[ERROR] Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"[ERROR] Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")


def select_algorithm():
    """Display algorithm menu and return selected algorithm."""
    print("\n Select Algorithm:")
    for key, (name, _) in ALGORITHMS.items():
        print(f"  {key}. {name}")
    
    choice = input("\nYour choice: ").strip()
    if choice in ALGORITHMS:
        return ALGORITHMS[choice]
    print("[ERROR] Invalid choice. Using Dinic by default.")
    return ALGORITHMS["3"]


def generate_random_graph():
    """Generate a random graph based on user preferences."""
    print("\n Graph Type:")
    print("  1. Sparse Graph (m ≈ edge_factor × n)")
    print("  2. Dense Graph (m ≈ density × n²)")
    print("  3. Bipartite Graph")
    print("  4. Grid Graph")
    
    graph_type = input("\nYour choice: ").strip()
    
    if graph_type == "1":
        n = get_int_input("Number of nodes (n): ", min_val=5, default=10)
        edge_factor = get_int_input("Edge factor (edges per node): ", min_val=1, default=3)
        max_cap = get_int_input("Maximum capacity: ", min_val=1, default=20)
        seed = get_int_input("Random seed (0 for random): ", min_val=0, default=42)
        seed = seed if seed > 0 else None
        
        print(f"\n Generating sparse graph with n={n}, edge_factor={edge_factor}...")
        g, s, t = random_sparse_graph(n, edge_factor=edge_factor, max_cap=max_cap, seed=seed)
        return g, s, t, f"Sparse Graph (n={n}, m≈{edge_factor*n})"
    
    elif graph_type == "2":
        n = get_int_input("Number of nodes (n): ", min_val=5, default=10)
        density = get_float_input("Density (0-1): ", min_val=0.01, max_val=1.0, default=0.3)
        max_cap = get_int_input("Maximum capacity: ", min_val=1, default=50)
        seed = get_int_input("Random seed (0 for random): ", min_val=0, default=42)
        seed = seed if seed > 0 else None
        
        print(f"\n Generating dense graph with n={n}, density={density}...")
        g, s, t = random_dense_graph(n, density=density, max_cap=max_cap, seed=seed)
        return g, s, t, f"Dense Graph (n={n}, density={density})"
    
    elif graph_type == "3":
        left = get_int_input("Left partition size: ", min_val=2, default=5)
        right = get_int_input("Right partition size: ", min_val=2, default=5)
        p = get_float_input("Edge probability (0-1): ", min_val=0.01, max_val=1.0, default=0.3)
        max_cap = get_int_input("Maximum capacity: ", min_val=1, default=10)
        seed = get_int_input("Random seed (0 for random): ", min_val=0, default=42)
        seed = seed if seed > 0 else None
        
        print(f"\n Generating bipartite graph ({left}×{right})...")
        g, s, t = bipartite_graph(left, right, p=p, max_cap=max_cap, seed=seed)
        return g, s, t, f"Bipartite Graph ({left}×{right})"
    
    elif graph_type == "4":
        rows = get_int_input("Number of rows: ", min_val=2, default=5)
        cols = get_int_input("Number of columns: ", min_val=2, default=5)
        max_cap = get_int_input("Maximum capacity: ", min_val=1, default=20)
        seed = get_int_input("Random seed (0 for random): ", min_val=0, default=42)
        seed = seed if seed > 0 else None
        
        print(f"\n Generating grid graph ({rows}×{cols})...")
        g, s, t = grid_graph(rows, cols, max_cap=max_cap, seed=seed)
        return g, s, t, f"Grid Graph ({rows}×{cols})"
    
    else:
        print("[ERROR] Invalid choice. Using default sparse graph.")
        g, s, t = random_sparse_graph(10, edge_factor=3, max_cap=20, seed=42)
        return g, s, t, "Default Sparse Graph"


def visualize_random_graph():
    """Option 1: Visualize Random Graph."""
    print_header("VISUALIZE RANDOM GRAPH")
    
    # Check if visualization is available
    if not _load_visualization():
        return
    
    # Generate graph
    g, s, t, graph_desc = generate_random_graph()
    
    # Select algorithm
    algo_name, algo_func = select_algorithm()
    
    print(f"\n Graph generated: {graph_desc}")
    print(f"   Nodes: {g.n}, Source: {s}, Sink: {t}")
    
    # Run algorithm
    print(f"\n  Running {algo_name}...")
    start_time = time.time()
    flow, residual = algo_func(g, s, t)
    runtime = time.time() - start_time
    
    print(f"\n[OK] Max Flow: {flow}")
    print(f"   Runtime: {runtime:.4f} seconds")
    
    # Visualizations
    if g.n > 50:
        print(f"\n[WARNING]  Graph has {g.n} nodes. Visualization may be cluttered.")
        visualize = input("   Continue with visualization? (y/n): ").strip().lower()
        if visualize != 'y':
            print("Skipping visualization.")
            return
    
    print("\n Visualizing graph...")
    visualize_graph(g, f"Original {graph_desc}")
    
    print(" Visualizing flow...")
    visualize_flow(g, residual, f"Flow on {graph_desc} ({algo_name})")
    
    print(" Computing and visualizing min-cut...")
    S, cut_edges = extract_min_cut(residual, s)
    visualize_min_cut(residual, S, cut_edges, f"Min-Cut on {graph_desc}")
    
    print("\n[OK] Visualization complete!")


def visualize_real_dataset():
    """Option 2: Visualize Real Dataset."""
    print_header("VISUALIZE REAL DATASET")
    
    # Check if visualization is available
    if not _load_visualization():
        return
    
    # Get file path
    print("\n Available datasets in data/:")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.txt'))]
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")
    
    file_path = input("\nEnter dataset path (or full path): ").strip()
    
    # Auto-detect format
    if not os.path.exists(file_path):
        file_path = os.path.join("data", file_path)
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return
    
    # Load dataset
    print(f"\n Loading dataset: {file_path}")
    try:
        if file_path.endswith('.csv'):
            g, s, t = load_csv(file_path)
            print("  Format: CSV")
        else:
            g, s, t = load_snap(file_path)
            print("  Format: SNAP")
        
        print(f"   Nodes: {g.n}, Source: {s}, Sink: {t}")
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return
    
    # Select algorithm
    algo_name, algo_func = select_algorithm()
    
    # Run algorithm
    print(f"\n  Running {algo_name}...")
    start_time = time.time()
    flow, residual = algo_func(g, s, t)
    runtime = time.time() - start_time
    
    print(f"\n[OK] Max Flow: {flow}")
    print(f"   Runtime: {runtime:.4f} seconds")
    
    # Visualizations (with warning for large graphs)
    if g.n > 30:
        print(f"\n[WARNING]  Graph has {g.n} nodes. Visualization not recommended.")
        visualize = input("   Visualize anyway? (y/n): ").strip().lower()
        if visualize != 'y':
            print("Skipping visualization.")
            return
    
    print("\n Visualizing graph...")
    visualize_graph(g, f"Real Dataset: {os.path.basename(file_path)}")
    
    print(" Visualizing flow...")
    visualize_flow(g, residual, f"Flow on {os.path.basename(file_path)} ({algo_name})")
    
    print(" Computing and visualizing min-cut...")
    S, cut_edges = extract_min_cut(residual, s)
    visualize_min_cut(residual, S, cut_edges, f"Min-Cut on {os.path.basename(file_path)}")
    
    print("\n[OK] Visualization complete!")


# def run_single_algorithm():
#     """Option 3: Run Single Algorithm on Chosen Graph."""
#     print_header("RUN SINGLE ALGORITHM")
    
#     # Generate graph
#     g, s, t, graph_desc = generate_random_graph()
    
#     # Select algorithm
#     algo_name, algo_func = select_algorithm()
    
#     print(f"\n Graph: {graph_desc}")
#     print(f"   Nodes: {g.n}, Source: {s}, Sink: {t}")
    
#     # Run algorithm
#     print(f"\n  Running {algo_name}...")
#     start_time = time.time()
#     flow, residual = algo_func(g, s, t)
#     runtime = time.time() - start_time
    
#     print(f"\n{'='*50}")
#     print(f"  [OK] RESULTS")
#     print(f"{'='*50}")
#     print(f"  Algorithm:  {algo_name}")
#     print(f"  Max Flow:   {flow}")
#     print(f"  Runtime:    {runtime:.6f} seconds")
#     print(f"{'='*50}")


def run_single_algorithm():
    """Option 3: Run a single algorithm on a chosen graph WITH VISUALIZATION."""
    print_header("RUN SINGLE ALGORITHM")
    
    # Generate graph
    result = generate_random_graph()
    if result is None:
        return
    
    g, s, t, graph_desc = result
    
    # Select algorithm
    algo_name, algo_func = select_algorithm()
    
    print(f"\n Graph: {graph_desc}")
    print(f"   Nodes: {g.n}, Source: {s}, Sink: {t}")
    print(f"\n  Running {algo_name}...")
    
    # Run algorithm with timing
    start = time.time()
    flow, residual = algo_func(g, s, t)
    runtime = time.time() - start
    
    # Display results
    print("\n" + "=" * 50)
    print("  [OK] RESULTS")
    print("=" * 50)
    print(f"  Algorithm:  {algo_name}")
    print(f"  Max Flow:   {flow}")
    print(f"  Runtime:    {runtime:.6f} seconds")
    print("=" * 50)
    
    # NEW: Ask if user wants to visualize
    print("\n Visualization Options:")
    print("  1. Show original graph structure")
    print("  2. Show flow on graph")
    print("  3. Show min-cut")
    print("  4. Show all three (recommended!)")
    print("  5. Skip visualization")
    
    vis_choice = input("\nYour choice: ").strip()
    
    if vis_choice in ['1', '2', '3', '4']:
        if not _load_visualization():
            print("[ERROR] Visualization requires: pip install networkx matplotlib")
            input("\n Press ENTER to continue...")
            return
        
        try:
            if vis_choice in ['1', '4']:
                print("\n Displaying original graph...")
                visualize_graph(g, graph_desc)
            
            if vis_choice in ['2', '4']:
                print("\n Displaying flow...")
                visualize_flow(g, residual, f"Flow on {graph_desc} ({algo_name})")
            
            if vis_choice in ['3', '4']:
                print("\n Computing min-cut...")
                S, cut_edges = extract_min_cut(residual, s)
                print(f"   Min-cut size: {len(cut_edges)} edges")
                print(f"   Source side: {len(S)} nodes")
                visualize_min_cut(residual, S, cut_edges, f"Min-Cut on {graph_desc}")
            
            print("\n[OK] Visualization complete!")
            
        except Exception as e:
            print(f"[ERROR] Error during visualization: {e}")
    
    # Ask if user wants to save results
    print("\n Save Options:")
    save_choice = input("  Save results to CSV? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        try:
            import pandas as pd
            
            # Create results directory
            os.makedirs("results", exist_ok=True)
            
            # Create filename from graph description
            safe_name = graph_desc.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results/single_run_{safe_name}_{timestamp}.csv"
            
            # Save to CSV
            data = {
                'Algorithm': [algo_name],
                'Flow': [flow],
                'Runtime': [runtime],
                'Graph': [graph_desc],
                'Nodes': [g.n],
                'Source': [s],
                'Sink': [t],
                'Edges': [sum(len(g.adj[u]) for u in range(g.n))]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            print(f"   [OK] Saved to: {filename}")
            
        except ImportError:
            print("  [ERROR] Cannot save - pandas not installed (pip install pandas)")
        except Exception as e:
            print(f"   [ERROR] Error saving: {e}")
    
    input("\n Press ENTER to continue...")


def run_all_algorithms():
    """Option 4: Run All Algorithms on Chosen Graph."""
    print_header("RUN ALL ALGORITHMS")
    
    # Check if benchmarking is available
    if not _load_benchmarking():
        return
    
    # Generate graph
    g, s, t, graph_desc = generate_random_graph()
    
    print(f"\n Graph: {graph_desc}")
    print(f"   Nodes: {g.n}, Source: {s}, Sink: {t}")
    
    # Run benchmark
    print(f"\n  Running all algorithms...")
    df = benchmark_single_graph(g, s, t, trials=1)
    
    print(f"\n{'='*70}")
    print(f"  [OK] BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")
    
    # Option to save
    save = input("\n Save results to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("   Filename (default: custom_benchmark.csv): ").strip()
        if not filename:
            filename = "custom_benchmark.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        save_path = os.path.join("results", filename)
        save_results(df, save_path)
        print(f"   [OK] Saved to {save_path}")


def run_benchmark_experiments():
    """Option 5: Run Benchmark Experiments."""
    print_header("RUN BENCHMARK EXPERIMENTS")
    
    # Check if experiments are available
    if not _load_experiments():
        return
    
    print("\n Available Experiments:")
    print("  1. Size Sweep (sparse graphs with varying n)")
    print("  2. Density Sweep (varying density for fixed n)")
    print("  3. Bipartite Experiment")
    print("  4. Grid Experiment (image segmentation)")
    print("  5. Capacity Distribution (low/medium/high)")
    print("  6. Run All Experiments")
    print("  0. Back to Main Menu")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == "1":
        print("\n Running Size Sweep Experiment...")
        experiment_sparse_sizes()
        print("[OK] Complete! Results saved to results/sparse_size_experiment.csv")
    
    elif choice == "2":
        print("\n Running Density Sweep Experiment...")
        experiment_density()
        print("[OK] Complete! Results saved to results/density_experiment.csv")
    
    elif choice == "3":
        print("\n Running Bipartite Experiment...")
        experiment_bipartite()
        print("[OK] Complete! Results saved to results/bipartite_experiment.csv")
    
    elif choice == "4":
        print("\n Running Grid Experiment...")
        experiment_grid()
        print("[OK] Complete! Results saved to results/grid_experiment.csv")
    
    elif choice == "5":
        print("\n Running Capacity Distribution Experiment...")
        experiment_capacity_distribution()
        print("[OK] Complete! Results saved to results/capacity_distribution_experiment.csv")
    
    elif choice == "6":
        print("\n Running ALL Experiments (this may take a while)...")
        experiment_sparse_sizes()
        experiment_density()
        experiment_bipartite()
        experiment_grid()
        experiment_capacity_distribution()
        print("\n[OK] All experiments complete! Check results/ directory.")
    
    elif choice == "0":
        return
    
    else:
        print("[ERROR] Invalid choice.")


def plot_benchmark_results():
    """Option 6: Plot Benchmark Results."""
    print_header("PLOT BENCHMARK RESULTS")
    
    # Check if plotting is available
    if not _load_plotting():
        return
    
    # List available CSV files
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("[ERROR] No results directory found. Run experiments first.")
        return
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print("[ERROR] No CSV files found in results/. Run experiments first.")
        return
    
    print("\n Available Result Files:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    
    file_choice = get_int_input("\nSelect file number: ", min_val=1, max_val=len(csv_files))
    selected_file = os.path.join(results_dir, csv_files[file_choice - 1])
    
    print(f"\n Selected: {csv_files[file_choice - 1]}")
    print("\n Plot Type:")
    print("  1. Bar Chart (runtime comparison)")
    print("  2. Line Plot (runtime vs nodes/size)")
    print("  3. Line Plot (runtime vs density)")
    
    plot_choice = input("\nYour choice: ").strip()
    
    try:
        if plot_choice == "1":
            print("\n Generating bar chart...")
            plot_runtime_bar(selected_file, title=f"Runtime: {csv_files[file_choice - 1]}")
        
        elif plot_choice == "2":
            x_col = input("X-axis column (default: Nodes): ").strip()
            if not x_col:
                x_col = "Nodes"
            print(f"\n Generating line plot (x={x_col})...")
            plot_runtime_vs_n(selected_file, x_name=x_col)
        
        elif plot_choice == "3":
            print("\n Generating density plot...")
            plot_runtime_vs_density(selected_file)
        
        else:
            print("[ERROR] Invalid choice. Showing bar chart by default.")
            plot_runtime_bar(selected_file)
        
        print("[OK] Plot displayed!")
    
    except Exception as e:
        print(f"[ERROR] Error generating plot: {e}")


def browse_and_visualize_datasets():
    """Option 8: Browse & Visualize Network Datasets."""
    print_header("BROWSE & VISUALIZE NETWORK DATASETS")
    
    # Load network dataset functions
    if not _load_network_datasets():
        return
    if not _load_network_analysis():
        return
    
    # Discover all datasets
    print("\n Discovering datasets...")
    datasets = discover_all_datasets()
    
    print(f"\n{'='*70}")
    print(f"  AVAILABLE DATASETS")
    print(f"{'='*70}")
    
    all_datasets = []
    idx = 1
    
    # List SNAP datasets
    if datasets['snap']:
        print(f"\n SNAP Edge Lists ({len(datasets['snap'])} files):")
        for path in datasets['snap']:
            print(f"  {idx}. {os.path.basename(path)}")
            all_datasets.append(('snap', path))
            idx += 1
    
    # List CSV datasets
    if datasets['csv']:
        print(f"\n CSV Networks ({len(datasets['csv'])} files):")
        for path in datasets['csv']:
            print(f"  {idx}. {os.path.basename(path)}")
            all_datasets.append(('csv', path))
            idx += 1
    
    # List VFOA networks
    if datasets['vfoa']:
        print(f"\n VFOA Networks ({len(datasets['vfoa'])} networks):")
        print(f"  {idx}. Browse all VFOA networks (see option 10)")
        print(f"     Sample: {datasets['vfoa'][:3]} ...")
    
    print(f"{'='*70}")
    
    if not all_datasets:
        print("\n[ERROR] No datasets found in data/ directory.")
        return
    
    # Select dataset
    choice = get_int_input(f"\nSelect dataset (1-{len(all_datasets)}, 0 to cancel): ", 
                           min_val=0, max_val=len(all_datasets))
    
    if choice == 0:
        return
    
    dtype, filepath = all_datasets[choice - 1]
    
    # Show dataset info
    print_dataset_info(filepath)
    
    # Load the dataset
    print(f"\n Loading {os.path.basename(filepath)}...")
    try:
        g, s, t, metadata = load_network(filepath)
        print(f"[OK] Loaded successfully!")
        print_network_summary(os.path.basename(filepath), g, s, t)
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        return
    
    # Visualization options
    print("\n What would you like to do?")
    print("  1. Visualize network structure")
    print("  2. Run max-flow algorithm and visualize")
    print("  3. Show network statistics")
    print("  4. Back to main menu")
    
    action = input("\nYour choice: ").strip()
    
    if action == "1":
        if not _load_visualization():
            return
        print("\n Visualizing network...")
        visualize_graph(g, os.path.basename(filepath))
    
    elif action == "2":
        if not _load_visualization():
            return
        
        # Select algorithm
        algo_name, algo_func = select_algorithm()
        
        print(f"\n  Running {algo_name}...")
        start_time = time.time()
        flow, residual = algo_func(g, s, t)
        runtime = time.time() - start_time
        
        print(f"\n[OK] Max Flow: {flow}")
        print(f"   Runtime: {runtime:.4f} seconds")
        
        # Visualize
        print("\n Visualizing flow...")
        visualize_flow(g, residual, f"{os.path.basename(filepath)} - {algo_name}")
        
        from utils.mincut import extract_min_cut
        S, cut_edges = extract_min_cut(residual, s)
        visualize_min_cut(residual, S, cut_edges, f"Min-Cut: {os.path.basename(filepath)}")
    
    elif action == "3":
        if not _load_comparison_visualizations():
            return
        visualize_network_statistics([(os.path.basename(filepath), g)])


def compare_multiple_networks():
    """Option 9: Compare Multiple Networks."""
    print_header("COMPARE MULTIPLE NETWORKS")
    
    # Load required functions
    if not _load_network_datasets():
        return
    if not _load_network_analysis():
        return
    
    # Discover datasets
    datasets = discover_all_datasets()
    all_datasets = []
    
    print(f"\n{'='*70}")
    print("  SELECT NETWORKS TO COMPARE")
    print(f"{'='*70}")
    
    idx = 1
    if datasets['snap']:
        for path in datasets['snap']:
            print(f"  {idx}. {os.path.basename(path)} [SNAP]")
            all_datasets.append(path)
            idx += 1
    
    if datasets['csv']:
        for path in datasets['csv']:
            print(f"  {idx}. {os.path.basename(path)} [CSV]")
            all_datasets.append(path)
            idx += 1
    
    if datasets['vfoa']:
        # Show first few VFOA networks as examples
        for net_id in datasets['vfoa'][:5]:
            print(f"  {idx}. {net_id} [VFOA]")
            all_datasets.append(net_id)
            idx += 1
        if len(datasets['vfoa']) > 5:
            print(f"  ... and {len(datasets['vfoa']) - 5} more VFOA networks")
    
    print(f"{'='*70}")
    
    if not all_datasets:
        print("\n[ERROR] No datasets found.")
        return
    
    # Select multiple networks
    print("\nEnter network numbers to compare (comma-separated, e.g., 1,2,3)")
    print("Maximum: 6 networks")
    
    selections = input("Your selection: ").strip()
    
    try:
        indices = [int(x.strip()) - 1 for x in selections.split(',')]
        indices = [i for i in indices if 0 <= i < len(all_datasets)][:6]
    except:
        print("[ERROR] Invalid selection format.")
        return
    
    if len(indices) < 2:
        print("[ERROR] Please select at least 2 networks.")
        return
    
    # Load selected networks with progress
    networks = []
    print(f"\n Loading {len(indices)} networks...")
    
    for i, idx in enumerate(indices, 1):
        filepath = all_datasets[idx]
        name = os.path.basename(filepath) if '/' in filepath else filepath
        print(f"  [{i}/{len(indices)}] Loading {name}...")
        
        try:
            g, s, t, metadata = load_network(filepath)
            networks.append((name, g, s, t))
        except Exception as e:
            print(f"   [WARNING] Failed to load {name}: {e}")
    
    if len(networks) < 2:
        print("\n[ERROR] Need at least 2 successfully loaded networks.")
        return
    
    print(f"\n[OK] Loaded {len(networks)} networks successfully!")
    
    # Show comparison menu
    print("\n Comparison Options:")
    print("  1. Compare network structures (table)")
    print("  2. Visualize networks side-by-side")
    print("  3. Compare network statistics (charts)")
    print("  4. Run max-flow on all and compare results")
    print("  5. Do all of the above")
    
    action = input("\nYour choice: ").strip()
    
    if action in ['1', '5']:
        # Compare structures
        network_pairs = [(name, g) for name, g, s, t in networks]
        df = compare_networks(network_pairs)
        print_comparison_table(df, "Network Structure Comparison")
    
    if action in ['2', '5']:
        # Visualize side-by-side
        if not _load_comparison_visualizations():
            return
        network_pairs = [(name, g) for name, g, s, t in networks]
        visualize_network_comparison(network_pairs)
    
    if action in ['3', '5']:
        # Statistics charts
        if not _load_comparison_visualizations():
            return
        network_pairs = [(name, g) for name, g, s, t in networks]
        visualize_network_statistics(network_pairs)
    
    if action in ['4', '5']:
        # Run max-flow comparison
        algo_name, algo_func = select_algorithm()
        
        print(f"\n Running {algo_name} on all networks...")
        results = []
        
        for i, (name, g, s, t) in enumerate(networks, 1):
            print(f"  [{i}/{len(networks)}] {name}...")
            start_time = time.time()
            flow, residual = algo_func(g, s, t)
            runtime = time.time() - start_time
            results.append((name, flow, runtime, {'nodes': g.n}))
            print(f"     Flow: {flow}, Time: {runtime:.4f}s")
        
        # Show comparison
        df = compare_flow_results(results)
        print_comparison_table(df, f"Max-Flow Comparison ({algo_name})")
        
        # Visualize flow comparison
        if not _load_comparison_visualizations():
            return
        visualize_flow_comparison([(r[0], r[1], r[2]) for r in results])
    
    # Save option
    save = input("\n Save comparison results to CSV? (y/n): ").strip().lower()
    if save == 'y':
        try:
            import pandas as pd
            os.makedirs("results", exist_ok=True)
            
            network_pairs = [(name, g) for name, g, s, t in networks]
            df = compare_networks(network_pairs)
            
            filename = f"results/network_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"[OK] Saved to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save: {e}")


def explore_vfoa_networks():
    """Option 10: VFOA Network Explorer."""
    print_header("VFOA NETWORK EXPLORER")
    
    # Load required functions
    if not _load_network_datasets():
        return
    if not _load_network_analysis():
        return
    
    # List all VFOA networks
    print("\n Loading VFOA network metadata...")
    try:
        vfoa_df = list_vfoa_networks()
        
        if vfoa_df.empty:
            print("[ERROR] No VFOA networks found.")
            return
        
        print(f"\n{'='*70}")
        print(f"  VFOA NETWORKS ({len(vfoa_df)} total)")
        print(f"{'='*70}")
        print("\nFirst 10 networks:")
        print(vfoa_df.head(10).to_string(index=False))
        print(f"\n... and {len(vfoa_df) - 10} more networks")
        print(f"{'='*70}")
    
    except Exception as e:
        print(f"[ERROR] Failed to load VFOA networks: {e}")
        return
    
    # Menu options
    print("\n Options:")
    print("  1. Visualize a specific VFOA network")
    print("  2. Compare multiple VFOA networks")
    print("  3. Run max-flow on specific network")
    print("  4. Batch analyze all VFOA networks")
    print("  5. Back to main menu")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == "1":
        # Visualize specific network
        net_id = get_int_input(f"Enter network ID (0-{len(vfoa_df)-1}): ", 
                               min_val=0, max_val=len(vfoa_df)-1)
        
        print(f"\n Loading network{net_id}...")
        try:
            g, s, t, metadata = load_vfoa_network(f"network{net_id}", weighted=True)
            
            print(f"\n[OK] Loaded network{net_id}")
            print(f"   Participants: {metadata['participants']}")
            print(f"   Timesteps: {metadata['timesteps']}")
            print(f"   Nodes: {g.n}")
            
            # Visualize
            if not _load_visualization():
                return
            visualize_graph(g, f"VFOA Network {net_id}")
            
            # Show heatmap
            if not _load_comparison_visualizations():
                return
            
            # Reconstruct capacity matrix
            import numpy as np
            n = g.n
            cap_matrix = np.zeros((n, n))
            for u in range(n):
                for v, cap, rev in g.adj[u]:
                    if cap > 0:
                        cap_matrix[u][v] = cap
            
            plot_vfoa_heatmap(cap_matrix, f"VFOA Network {net_id} - Attention Matrix")
            
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
    
    elif choice == "2":
        # Compare multiple VFOA networks
        print("\nEnter network IDs to compare (comma-separated, e.g., 0,5,10)")
        print("Maximum: 6 networks")
        
        selections = input("Your selection: ").strip()
        
        try:
            net_ids = [int(x.strip()) for x in selections.split(',')]
            net_ids = [i for i in net_ids if 0 <= i < len(vfoa_df)][:6]
        except:
            print("[ERROR] Invalid selection.")
            return
        
        if len(net_ids) < 2:
            print("[ERROR] Select at least 2 networks.")
            return
        
        # Load networks
        networks = []
        print(f"\n Loading {len(net_ids)} networks...")
        for net_id in net_ids:
            print(f"  Loading network{net_id}...")
            try:
                g, s, t, metadata = load_vfoa_network(f"network{net_id}", weighted=True)
                networks.append((f"network{net_id}", g, s, t))
            except Exception as e:
                print(f"   [WARNING] Failed: {e}")
        
        if networks:
            # Visualize comparison
            if not _load_comparison_visualizations():
                return
            
            network_pairs = [(name, g) for name, g, s, t in networks]
            visualize_network_comparison(network_pairs)
            visualize_network_statistics(network_pairs)
    
    elif choice == "3":
        # Run max-flow on specific network
        net_id = get_int_input(f"Enter network ID (0-{len(vfoa_df)-1}): ",
                               min_val=0, max_val=len(vfoa_df)-1)
        
        print(f"\n Loading network{net_id}...")
        try:
            g, s, t, metadata = load_vfoa_network(f"network{net_id}", weighted=True)
            
            algo_name, algo_func = select_algorithm()
            
            print(f"\n Running {algo_name}...")
            start_time = time.time()
            flow, residual = algo_func(g, s, t)
            runtime = time.time() - start_time
            
            print(f"\n[OK] Max Flow: {flow}")
            print(f"   Runtime: {runtime:.4f} seconds")
            print(f"   Participants: {metadata['participants']}")
            
            # Visualize
            if not _load_visualization():
                return
            visualize_flow(g, residual, f"VFOA Network {net_id} - Flow")
            
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
    
    elif choice == "4":
        # Batch analyze all networks
        print(f"\n Batch analyzing all {len(vfoa_df)} VFOA networks...")
        print("This may take a few minutes...")
        
        algo_name, algo_func = select_algorithm()
        
        def progress_callback(current, total, name):
            print(f"  [{current}/{total}] {name}...")
        
        print("\n Loading all networks...")
        all_networks = load_all_vfoa_networks(progress_callback=progress_callback)
        
        print(f"\n[OK] Loaded {len(all_networks)} networks")
        
        print(f"\n Running {algo_name} on all networks...")
        results = []
        
        for i, (net_id, g, s, t, metadata) in enumerate(all_networks, 1):
            print(f"  [{i}/{len(all_networks)}] {net_id}...")
            try:
                start_time = time.time()
                flow, residual = algo_func(g, s, t)
                runtime = time.time() - start_time
                results.append((net_id, flow, runtime, metadata))
            except Exception as e:
                print(f"   [WARNING] Failed: {e}")
        
        print(f"\n[OK] Completed {len(results)} networks")
        
        # Show results
        df = compare_flow_results(results)
        print_comparison_table(df, f"VFOA Networks - Max-Flow Results ({algo_name})")
        
        # Save results
        try:
            import pandas as pd
            os.makedirs("results", exist_ok=True)
            filename = f"results/vfoa_batch_{algo_name.replace('-', '_').lower()}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\n[OK] Results saved to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save: {e}")


def main_menu():
    """Main interactive menu."""
    while True:
        clear_screen()
        print_header("MAXFLOW PROJECT - INTERACTIVE CLI")
        print("\n Main Menu:")
        print("  1. Visualize Random Graph (graph + flow + min-cut)")
        print("  2. Visualize Real Dataset (graph + flow + min-cut)")
        print("  3. Run Single Algorithm on Chosen Graph")
        print("  4. Run All Algorithms on Chosen Graph")
        print("  5. Run Benchmark Experiments (size, density, bipartite, grid, capacity)")
        print("  6. Plot Benchmark Results")
        print("  7. VFOA Network Explorer (62 networks)")
        print("\n 0. Exit")
        
        choice = input("\n➤ Your choice: ").strip()
        
        try:
            if choice == "1":
                visualize_random_graph()
            elif choice == "2":
                visualize_real_dataset()
            elif choice == "3":
                run_single_algorithm()
            elif choice == "4":
                run_all_algorithms()
            elif choice == "5":
                run_benchmark_experiments()
            elif choice == "6":
                plot_benchmark_results()
            elif choice == "7":
                explore_vfoa_networks()
            elif choice == "0":
                print("\n Goodbye!")
                sys.exit(0)
            else:
                print("\n[ERROR] Invalid choice. Please select a valid option.")
        
        except KeyboardInterrupt:
            print("\n\n[WARNING]  Operation cancelled by user.")
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n Press ENTER to continue...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        sys.exit(0)
