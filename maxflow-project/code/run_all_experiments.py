import pandas as pd
from utils.generators import (
    random_sparse_graph,
    random_dense_graph,
    bipartite_graph,
    grid_graph,
    layered_graph,
    worst_case_long_path_graph,
    unit_capacity_graph
)
from benchmark.runner import benchmark_single_graph, save_results


def experiment_sparse_sizes():
    sizes = [50, 100, 200, 400]
    all_rows = []

    for n in sizes:
        print(f"Running sparse experiment for n={n}...")
        g, s, t = random_sparse_graph(n, edge_factor=3, max_cap=20, seed=42)
        df = benchmark_single_graph(g, s, t, trials=1)

        df["Nodes"] = n
        df["Edges"] = 3 * n
        all_rows.append(df)

    final_df = pd.concat(all_rows)
    save_results(final_df, "results/sparse_size_experiment.csv")


def experiment_density():
    n = 200
    densities = [0.05, 0.1, 0.3, 0.5]

    all_rows = []

    for d in densities:
        print(f"Running density experiment for density={d}...")
        g, s, t = random_dense_graph(n, density=d, max_cap=50, seed=42)
        df = benchmark_single_graph(g, s, t, trials=1)

        df["Nodes"] = n
        df["Density"] = d
        all_rows.append(df)

    final_df = pd.concat(all_rows)
    save_results(final_df, "results/density_experiment.csv")


def experiment_bipartite():
    left = 50
    right = 50
    all_rows = []

    g, s, t = bipartite_graph(left, right, p=0.1, max_cap=10, seed=42)
    df = benchmark_single_graph(g, s, t, trials=1)
    df["Left"] = left
    df["Right"] = right
    save_results(df, "results/bipartite_experiment.csv")


def experiment_grid():
    print("Running grid experiment (image segmentation style)...")
    g, s, t = grid_graph(20, 20, max_cap=20, seed=42)
    df = benchmark_single_graph(g, s, t, trials=1)
    save_results(df, "results/grid_experiment.csv")

def experiment_capacity_distribution():
    import numpy as np
    all_rows = []

    distributions = {
        "low": lambda: random_sparse_graph(200, edge_factor=3, max_cap=10, seed=42),
        "medium": lambda: random_sparse_graph(200, edge_factor=3, max_cap=100, seed=42),
        "high": lambda: random_sparse_graph(200, edge_factor=3, max_cap=1000, seed=42)
    }

    for name, generator in distributions.items():
        print(f"Running capacity distribution experiment: {name}")
        g, s, t = generator()
        df = benchmark_single_graph(g, s, t)
        df["CapacityType"] = name
        all_rows.append(df)

    final_df = pd.concat(all_rows)
    save_results(final_df, "results/capacity_distribution_experiment.csv")


def experiment_layered_graphs():
    """Vary number of layers to show blocking flows that Dinic handles efficiently."""
    layer_counts = [3, 5, 7]
    width = 15  # nodes per layer
    all_rows = []
    for L in layer_counts:
        print(f"Running layered graph experiment for {L} layers (width={width})...")
        g, s, t = layered_graph([width] * L, cap=10, full=True, seed=42)
        df = benchmark_single_graph(g, s, t, trials=1)
        df["Layers"] = L
        df["Width"] = width
        df["Nodes"] = width * L + 2
        all_rows.append(df)
    final_df = pd.concat(all_rows)
    save_results(final_df, "results/layered_experiment.csv")


def experiment_long_path_graphs():
    """Worst-case for Ford-Fulkerson (DFS) style augmentation: long unit path."""
    lengths = [50, 100, 200, 400]
    all_rows = []
    for length in lengths:
        print(f"Running long path experiment for path length={length}...")
        g, s, t = worst_case_long_path_graph(length, cap=1)
        df = benchmark_single_graph(g, s, t, trials=1)
        df["PathLength"] = length
        df["Nodes"] = length + 2
        all_rows.append(df)
    final_df = pd.concat(all_rows)
    save_results(final_df, "results/long_path_experiment.csv")


def experiment_unit_capacity_graphs():
    """Unit capacity random graphs highlighting Edmonds-Karp performance."""
    sizes = [100, 200, 400]
    density = 0.05
    all_rows = []
    for n in sizes:
        print(f"Running unit capacity experiment for n={n}, density={density}...")
        g, s, t = unit_capacity_graph(n, density=density, seed=42)
        df = benchmark_single_graph(g, s, t, trials=1)
        df["Nodes"] = n
        df["Density"] = density
        all_rows.append(df)
    final_df = pd.concat(all_rows)
    save_results(final_df, "results/unit_capacity_experiment.csv")



if __name__ == "__main__":
    experiment_sparse_sizes()
    experiment_density()
    experiment_bipartite()
    experiment_grid()
    experiment_capacity_distribution()
    experiment_layered_graphs()
    experiment_long_path_graphs()
    experiment_unit_capacity_graphs()



# to run this. - python3 code/run_all_experiments.py
