import os
import statistics
import tracemalloc
import pandas as pd
from benchmark.runner import ALGORITHMS  # base algorithms (without variants)
from algos.push_relabel import (
    push_relabel_fifo,
    push_relabel_highest,
    push_relabel_gap,
    push_relabel_global
)

# Extended algorithms list for metrics-only benchmarking (includes variants of push-relabel)
METRICS_ALGOS = {
    **ALGORITHMS,
    "Push-Relabel-FIFO": push_relabel_fifo,
    "Push-Relabel-Highest": push_relabel_highest,
    "Push-Relabel-Gap": push_relabel_gap,
    "Push-Relabel-Global": push_relabel_global,
}

def benchmark_metrics_single_graph(graph, s, t):
    """Run algorithms (including push-relabel variants) on a single graph and collect instrumentation metrics.

    Returns a DataFrame with columns:
      Algorithm, Flow, augmenting_paths, blocking_flows, max_level_depth, relabel_ops,
      mean_increment, median_increment, max_increment, memory_peak_kb
    """
    rows = []
    for name, algo in METRICS_ALGOS.items():
        g_copy = graph.copy()
        tracemalloc.start()
        flow, residual = algo(g_copy, s, t)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = getattr(residual, "metrics", {})
        increments = metrics.get("path_flow_increments", [])
        mean_inc = statistics.mean(increments) if increments else None
        median_inc = statistics.median(increments) if increments else None
        max_inc = max(increments) if increments else None
        rows.append({
            "Algorithm": name,
            "Flow": flow,
            "augmenting_paths": metrics.get("augmenting_paths"),
            "blocking_flows": metrics.get("blocking_flows"),
            "max_level_depth": metrics.get("max_level_depth"),
            "relabel_ops": metrics.get("relabel_ops"),
            "variant": metrics.get("variant"),
            "mean_increment": mean_inc,
            "median_increment": median_inc,
            "max_increment": max_inc,
            "memory_peak_kb": peak / 1024.0
        })
    return pd.DataFrame(rows)


def save_metrics_results(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved metrics results to {filename}")
