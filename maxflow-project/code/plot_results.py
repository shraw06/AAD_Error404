"""
Extended plotting script adding analytical plots requested:

Plots implemented:
 (A) Running Time vs Number of Vertices (n) for fixed edge ratio m ≈ 3n (sparse)
	 and optionally dense ratio m ≈ n^2/4.
 (B) Running Time vs Number of Edges (m) for fixed n varying density.
 (C) Running Time vs Capacity Range (Cmax) using capacity distribution experiments.
 (D) Running Time vs Sparsity/Density Ratio (edges per node) across densities.
 (E) Throughput (Flow / MeanTime) comparison per algorithm for a chosen dataset.

Existing CSVs are reused when possible. Some sweeps will lazily generate new CSVs
if they do not exist.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import tracemalloc

# Ensure consistent results directory (project root /results)
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def _rp(name: str) -> str:
	"""Return absolute path in central results directory."""
	return os.path.join(RESULTS_DIR, name)

from benchmark.runner import benchmark_single_graph, save_results
from benchmark.metrics import benchmark_metrics_single_graph, save_metrics_results
from utils.generators import (
	random_sparse_graph,
	random_dense_graph
)


def ensure_size_sweep(edge_factor=3, sizes=None, filename=None, seed=42):
	"""Generate (if missing) a size sweep dataset with m ≈ edge_factor * n."""
	if sizes is None:
		sizes = [50, 100, 200, 400, 800, 1200, 1600, 2000]
	if filename is None:
		filename = _rp(f"size_sweep_edgefactor{edge_factor}.csv")
	if os.path.exists(filename):
		return filename

	all_rows = []
	for n in sizes:
		print(f"[size_sweep sparse] n={n}, edge_factor={edge_factor}")
		g, s, t = random_sparse_graph(n, edge_factor=edge_factor, max_cap=20, seed=seed)
		df = benchmark_single_graph(g, s, t, trials=1)
		df["Nodes"] = n
		df["Edges"] = edge_factor * n
		all_rows.append(df)
	final_df = pd.concat(all_rows)
	save_results(final_df, filename)
	return filename


def ensure_size_sweep_dense(sizes=None, filename=None, seed=42):
	"""Generate (if missing) a dense size sweep with target m ≈ n^2 / 4.
	We approximate density = (n^2/4) / (n*(n-1)) ≈ n / (4*(n-1)).
	For very large n this tends to 1/4; for small n it's a bit larger.
	Limit sizes to avoid extreme runtime.
	"""
	if sizes is None:
		sizes = [50, 100, 200, 400, 800]
	if filename is None:
		filename = _rp("size_sweep_dense.csv")
	if os.path.exists(filename):
		return filename

	all_rows = []
	for n in sizes:
		density = n / (4 * (n - 1))
		print(f"[size_sweep dense] n={n}, target density≈{density:.3f}")
		g, s, t = random_dense_graph(n, density=density, max_cap=50, seed=seed)
		df = benchmark_single_graph(g, s, t, trials=1)
		m_est = density * n * (n - 1)
		df["Nodes"] = n
		df["Edges"] = int(m_est)
		df["Density"] = density
		all_rows.append(df)
	final_df = pd.concat(all_rows)
	save_results(final_df, filename)
	return filename


def _save(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot -> {path}")
    return path


def plot_time_vs_n():
	sparse_file = ensure_size_sweep()
	dense_file = ensure_size_sweep_dense()
	df_sparse = pd.read_csv(sparse_file)
	df_dense = pd.read_csv(dense_file)

	fig = plt.figure(figsize=(10, 6))
	for algo in df_sparse["Algorithm"].unique():
		sub = df_sparse[df_sparse["Algorithm"] == algo]
		plt.plot(sub["Nodes"], sub["MeanTime"], marker="o", label=f"{algo} (m≈3n)")
	for algo in df_dense["Algorithm"].unique():
		sub = df_dense[df_dense["Algorithm"] == algo]
		plt.plot(sub["Nodes"], sub["MeanTime"], marker="s", linestyle="--", label=f"{algo} (m≈n²/4)")
	plt.xlabel("Number of Vertices (n)")
	plt.ylabel("Runtime (s)")
	plt.title("(A) Runtime vs n at Different Edge Ratios")
	plt.legend(fontsize=8)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "(A)_runtime_vs_n.png")


def plot_time_vs_edges_for_fixed_n(n_fixed=200, densities=None, filename=None, seed=42):
	if densities is None:
		densities = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
	if filename is None:
		filename = _rp(f"edge_sweep_n{n_fixed}.csv")
	if not os.path.exists(filename):
		rows = []
		for d in densities:
			print(f"[edge_sweep] n={n_fixed}, density={d}")
			g, s, t = random_dense_graph(n_fixed, density=d, max_cap=50, seed=seed)
			df = benchmark_single_graph(g, s, t, trials=1)
			m_est = int(d * n_fixed * (n_fixed - 1))
			df["Nodes"] = n_fixed
			df["Edges"] = m_est
			df["Density"] = d
			rows.append(df)
		final = pd.concat(rows)
		save_results(final, filename)

	df = pd.read_csv(filename)
	fig = plt.figure(figsize=(10, 6))
	for algo in df["Algorithm"].unique():
		sub = df[df["Algorithm"] == algo]
		plt.plot(sub["Edges"], sub["MeanTime"], marker="o", label=algo)
	plt.xlabel("Number of Edges (m)")
	plt.ylabel("Runtime (s)")
	plt.title(f"(B) Runtime vs m (n={n_fixed})")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, f"(B)_runtime_vs_m_n{n_fixed}.png")


def plot_time_vs_capacity_range(csv_file=None):
	if csv_file is None:
		# Try central results dir first, then fallback to script-local previous location
		primary = _rp("capacity_distribution_experiment.csv")
		alt = os.path.join(SCRIPT_DIR, "results", "capacity_distribution_experiment.csv")
		csv_file = primary if os.path.exists(primary) else alt
	df = pd.read_csv(csv_file)
	# Map capacity type to Cmax
	mapping = {"low": 10, "medium": 100, "high": 1000}
	df["Cmax"] = df["CapacityType"].map(mapping)
	fig = plt.figure(figsize=(10, 6))
	for algo in df["Algorithm"].unique():
		sub = df[df["Algorithm"] == algo]
		plt.plot(sub["Cmax"], sub["MeanTime"], marker="o", label=algo)
	plt.xscale("log")
	plt.xlabel("Maximum Capacity (Cmax)")
	plt.ylabel("Runtime (s)")
	plt.title("(C) Runtime vs Capacity Range (log scale Cmax)")
	plt.legend()
	plt.grid(True, which="both", alpha=0.3)
	plt.tight_layout()
	_save(fig, "(C)_runtime_vs_capacity_range.png")


def plot_time_vs_sparsity(n_fixed=200, densities=None):
	if densities is None:
		densities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
	temp_file = _rp(f"sparsity_sweep_n{n_fixed}.csv")
	if not os.path.exists(temp_file):
		rows = []
		for d in densities:
			print(f"[sparsity_sweep] n={n_fixed}, density={d}")
			g, s, t = random_dense_graph(n_fixed, density=d, max_cap=50, seed=42)
			df = benchmark_single_graph(g, s, t, trials=1)
			m = d * n_fixed * (n_fixed - 1)
			df["EdgesPerNode"] = m / n_fixed
			df["Density"] = d
			rows.append(df)
		final = pd.concat(rows)
		save_results(final, temp_file)
	df = pd.read_csv(temp_file)
	fig = plt.figure(figsize=(10, 6))
	for algo in df["Algorithm"].unique():
		sub = df[df["Algorithm"] == algo]
		plt.plot(sub["EdgesPerNode"], sub["MeanTime"], marker="o", label=algo)
	plt.xlabel("Edges per Node (m/n)")
	plt.ylabel("Runtime (s)")
	plt.title("(D) Runtime vs Sparsity/Density Ratio (m/n)")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, f"(D)_runtime_vs_sparsity_n{n_fixed}.png")


def plot_throughput(csv_file, x_col):
	df = pd.read_csv(csv_file)
	df["Throughput"] = df["Flow"] / df["MeanTime"].replace(0, float('nan'))
	fig = plt.figure(figsize=(10, 6))
	for algo in df["Algorithm"].unique():
		sub = df[df["Algorithm"] == algo]
		plt.plot(sub[x_col], sub["Throughput"], marker="o", label=algo)
	plt.xlabel(x_col)
	plt.ylabel("Flow per Second")
	plt.title(f"(E) Throughput vs {x_col}")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, f"(E)_throughput_vs_{x_col}.png")


# ---------------- New Metrics Plots (H - M) -----------------

def ensure_metrics_sample(filename=None, n=300, edge_factor=3, seed=123):
	"""Generate (if missing) a single graph metrics dataset including algorithm variants."""
	if filename is None:
		filename = _rp("algorithm_metrics_sample.csv")
	if os.path.exists(filename):
		return filename
	g, s, t = random_sparse_graph(n, edge_factor=edge_factor, max_cap=50, seed=seed)
	df = benchmark_metrics_single_graph(g, s, t)
	save_metrics_results(df, filename)
	return filename


def plot_augmenting_path_counts(n=300, edge_factor=3):
	file = ensure_metrics_sample(n=n, edge_factor=edge_factor)
	df = pd.read_csv(file)
	# Include Dinic blocking flows with distinct label
	fig = plt.figure(figsize=(10, 6))
	labels = []
	counts = []
	for _, row in df.iterrows():
		if not pd.isna(row.get("augmenting_paths")):
			labels.append(row["Algorithm"])
			counts.append(row["augmenting_paths"])
		elif not pd.isna(row.get("blocking_flows")):
			labels.append(row["Algorithm"] + " (blocking flows)")
			counts.append(row["blocking_flows"])
	plt.bar(labels, counts, color="skyblue")
	plt.ylabel("Count")
	plt.title("(H) Number of Augmenting Paths / Blocking Flows")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	_save(fig, "(H)_augmenting_paths.png")


def plot_relabel_operations(n=300, edge_factor=3):
	file = ensure_metrics_sample(n=n, edge_factor=edge_factor)
	df = pd.read_csv(file)
	subset = df[df["Algorithm"].str.contains("Push-Relabel")]
	fig = plt.figure(figsize=(10, 6))
	plt.bar(subset["Algorithm"], subset["relabel_ops"], color="salmon")
	plt.ylabel("Relabel Operations")
	plt.title("(I) Relabel Operations Across Push-Relabel Variants")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	_save(fig, "(I)_relabel_ops.png")


def plot_dinic_layer_depth(n=300, edge_factor=3):
	file = ensure_metrics_sample(n=n, edge_factor=edge_factor)
	df = pd.read_csv(file)
	dinic_row = df[df["Algorithm"] == "Dinic"].head(1)
	if dinic_row.empty:
		print("Dinic metrics not found.")
		return
	depth = dinic_row.iloc[0]["max_level_depth"]
	fig = plt.figure(figsize=(6, 5))
	plt.bar(["Dinic"], [depth], color="mediumseagreen")
	plt.ylabel("Max Layer Depth")
	plt.title("(J) Maximum Layer Depth in Level Graph (Dinic)")
	plt.tight_layout()
	_save(fig, "(J)_dinic_max_layer_depth.png")


def plot_flow_increments(n=300, edge_factor=3):
	file = ensure_metrics_sample(n=n, edge_factor=edge_factor)
	df = pd.read_csv(file)
	# Focus on algorithms with path increments (exclude push-relabel variants except maybe not needed)
	increment_df = df[~df["Algorithm"].str.contains("Push-Relabel")]
	fig = plt.figure(figsize=(10, 6))
	for _, row in increment_df.iterrows():
		algo = row["Algorithm"]
		# We don't have per-step increments stored in CSV (only summary stats). For richer plot, regenerate on the fly.
		# Re-run single algorithm to extract increments list.
		g, s, t = random_sparse_graph(150, edge_factor=3, max_cap=40, seed=42)
		from benchmark.metrics import METRICS_ALGOS
		flow, residual = METRICS_ALGOS[algo](g, s, t)
		inc_list = residual.metrics.get("path_flow_increments", [])
		plt.plot(range(1, len(inc_list) + 1), inc_list, marker="o", label=algo)
	plt.xlabel("Augmenting Step Index")
	plt.ylabel("Flow Increment (bottleneck)")
	plt.title("(K) Flow Increment per Augmenting Step")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "(K)_flow_increments.png")


def plot_memory_usage(n=300, edge_factor=3):
	file = ensure_metrics_sample(n=n, edge_factor=edge_factor)
	df = pd.read_csv(file)
	fig = plt.figure(figsize=(10, 6))
	plt.bar(df["Algorithm"], df["memory_peak_kb"], color="plum")
	plt.ylabel("Peak Memory (KB)")
	plt.title("(L) Peak Memory Usage Comparison")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	_save(fig, "(L)_memory_usage.png")


def plot_parallelism_placeholder():
	"""Placeholder for parallel scaling (M). Actual parallel implementation not present; show note."""
	fig = plt.figure(figsize=(6,4))
	plt.text(0.5, 0.5, "(M) Parallelism not implemented\nPush-Relabel best suited for parallelization.",
	         ha='center', va='center')
	plt.axis('off')
	plt.tight_layout()
	_save(fig, "(M)_parallelism_placeholder.png")


# ---------------- New Topology & Distance Plots (F, G) -----------------

def ensure_topology_runtime_sample(filename=None, seed=77):
	"""Generate (if missing) a dataset capturing runtime across different graph topologies.

	Topologies included:
	  Random-Sparse, Random-Dense, Bipartite, Grid, Layered, Long-Path, Unit-Capacity
	"""
	if filename is None:
		filename = _rp("topology_runtime_sample.csv")
	if os.path.exists(filename):
		return filename

	from utils.generators import (
		random_sparse_graph,
		random_dense_graph,
		bipartite_graph,
		grid_graph,
		layered_graph,
		worst_case_long_path_graph,
		unit_capacity_graph
	)
	rows = []
	# Choose representative sizes (balanced for runtime)
	specs = [
		("Random-Sparse", lambda: random_sparse_graph(300, edge_factor=3, max_cap=30, seed=seed)),
		("Random-Dense", lambda: random_dense_graph(250, density=0.4, max_cap=40, seed=seed)),
		("Bipartite", lambda: bipartite_graph(80, 80, p=0.08, max_cap=15, seed=seed)),
		("Grid", lambda: grid_graph(18, 18, max_cap=20, seed=seed)),
		("Layered", lambda: layered_graph([20]*5, cap=12, full=True, seed=seed)),
		("Long-Path", lambda: worst_case_long_path_graph(220, cap=1)),
		("Unit-Capacity", lambda: unit_capacity_graph(300, density=0.04, seed=seed)),
	]
	for topo_name, gen in specs:
		print(f"[topology runtime] {topo_name}")
		g, s, t = gen()
		df = benchmark_single_graph(g, s, t, trials=1)
		df["Topology"] = topo_name
		rows.append(df)
	final = pd.concat(rows)
	save_results(final, filename)
	return filename


def plot_runtime_vs_topology():
	file = ensure_topology_runtime_sample()
	df = pd.read_csv(file)
	fig = plt.figure(figsize=(12, 6))
	# Grouped bar chart: topology on x, colored by algorithm (mean)
	topologies = df["Topology"].unique()
	algos = df["Algorithm"].unique()
	import numpy as np
	x = np.arange(len(topologies))
	width = 0.8 / len(algos)
	for i, algo in enumerate(algos):
		sub = df[df["Algorithm"] == algo].set_index("Topology")
		heights = [sub.loc[topo, "MeanTime"] if topo in sub.index else float('nan') for topo in topologies]
		plt.bar(x + i*width, heights, width, label=algo)
	plt.xticks(x + (len(algos)-1)*width/2, topologies, rotation=25, ha='right')
	plt.ylabel("Runtime (s)")
	plt.title("(F) Runtime vs Graph Topology")
	plt.legend(ncol=2, fontsize=8)
	plt.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	_save(fig, "(F)_runtime_vs_topology.png")


def ensure_distance_scaling_dataset(distances=None, filename=None):
	"""Generate dataset varying guaranteed shortest path length between source and sink using long path graphs."""
	if distances is None:
		distances = [20, 50, 100, 200, 300]
	if filename is None:
		filename = _rp("distance_scaling_experiment.csv")
	if os.path.exists(filename):
		return filename
	from utils.generators import worst_case_long_path_graph
	rows = []
	for d in distances:
		print(f"[distance scaling] path length={d}")
		g, s, t = worst_case_long_path_graph(d, cap=1)
		df = benchmark_single_graph(g, s, t, trials=1)
		df["Distance"] = d  # shortest path length
		df["Nodes"] = d + 2
		rows.append(df)
	final = pd.concat(rows)
	save_results(final, filename)
	return filename


def plot_distance_scaling():
	file = ensure_distance_scaling_dataset()
	df = pd.read_csv(file)
	fig = plt.figure(figsize=(10, 6))
	for algo in df["Algorithm"].unique():
		sub = df[df["Algorithm"] == algo]
		plt.plot(sub["Distance"], sub["MeanTime"], marker='o', label=algo)
	plt.xlabel("Source–Sink Shortest Path Length")
	plt.ylabel("Runtime (s)")
	plt.title("(G) Runtime Scaling with Source–Sink Distance")
	plt.legend(fontsize=8)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "(G)_runtime_vs_source_sink_distance.png")


# ---------------- Per-Algorithm Detailed Plots -----------------

def ensure_capacity_range_dataset(sizes=None, caps=None, filename=None):
	"""Dataset varying max capacity for Ford-Fulkerson & Capacity-Scaling time comparisons."""
	if caps is None:
		caps = [10, 100, 1000, 5000]
	if filename is None:
		filename = _rp("capacity_range_ff_cs.csv")
	if os.path.exists(filename):
		return filename
	from utils.generators import random_sparse_graph
	rows = []
	for C in caps:
		g, s, t = random_sparse_graph(250, edge_factor=3, max_cap=C, seed=99)
		df = benchmark_single_graph(g, s, t, trials=1)
		df["Cmax"] = C
		rows.append(df)
	final = pd.concat(rows)
	save_results(final, filename)
	return filename


def plot_ff_time_vs_capacity():
	file = ensure_capacity_range_dataset()
	df = pd.read_csv(file)
	sub = df[df["Algorithm"].isin(["Ford-Fulkerson", "Capacity-Scaling"])]
	fig = plt.figure(figsize=(8,6))
	for algo in sub["Algorithm"].unique():
		a = sub[sub["Algorithm"] == algo]
		plt.plot(a["Cmax"], a["MeanTime"], marker='o', label=algo)
	plt.xscale('log')
	plt.xlabel('Max Capacity (Cmax)')
	plt.ylabel('Runtime (s)')
	plt.title('Ford-Fulkerson & Capacity-Scaling: Time vs Capacity Range')
	plt.grid(True, which='both', alpha=0.3)
	plt.legend()
	plt.tight_layout()
	_save(fig, "FF_CS_time_vs_capacity.png")


def plot_capacity_scaling_steps():
	# Regenerate single graph metrics for capacity scaling to extract scaling steps multiple times
	from utils.generators import random_sparse_graph
	from algos.capacity_scaling import capacity_scaling
	fig = plt.figure(figsize=(8,6))
	steps_list = []
	caps = [50, 200, 800, 3200]
	for C in caps:
		g, s, t = random_sparse_graph(300, edge_factor=3, max_cap=C, seed=7)
		flow, residual = capacity_scaling(g, s, t)
		steps = residual.metrics.get("scaling_steps")
		steps_list.append((C, steps))
	xs = [c for c,_ in steps_list]
	ys = [st for _,st in steps_list]
	plt.plot(xs, ys, marker='o')
	plt.xscale('log')
	plt.xlabel('Max Capacity (Cmax)')
	plt.ylabel('Scaling Steps')
	plt.title('Capacity-Scaling: Steps vs Capacity Range')
	plt.grid(True, which='both', alpha=0.3)
	plt.tight_layout()
	_save(fig, "CapacityScaling_steps_vs_capacity.png")


def ensure_density_dataset(filename=None):
	if filename is None:
		filename = _rp("edmonds_karp_density.csv")
	if os.path.exists(filename):
		return filename
	from utils.generators import random_dense_graph
	densities = [0.05, 0.1, 0.2, 0.3, 0.4]
	rows = []
	n = 250
	for d in densities:
		g, s, t = random_dense_graph(n, density=d, max_cap=30, seed=101)
		df = benchmark_single_graph(g, s, t, trials=1)
		df["Density"] = d
		df["Nodes"] = n
		rows.append(df)
	final = pd.concat(rows)
	save_results(final, filename)
	return filename


def plot_ek_time_vs_density():
	file = ensure_density_dataset()
	df = pd.read_csv(file)
	ek = df[df["Algorithm"] == "Edmonds-Karp"]
	ff = df[df["Algorithm"] == "Ford-Fulkerson"]
	fig = plt.figure(figsize=(8,6))
	plt.plot(ek["Density"], ek["MeanTime"], marker='o', label='Edmonds-Karp')
	if not ff.empty:
		plt.plot(ff["Density"], ff["MeanTime"], marker='s', label='Ford-Fulkerson')
	plt.xlabel('Density')
	plt.ylabel('Runtime (s)')
	plt.title('Edmonds-Karp: Time vs Density (with FF reference)')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	_save(fig, "EK_time_vs_density.png")


def plot_dinic_blocking_flows_vs_n():
	# vary layered graph size to see blocking flows scaling
	from utils.generators import layered_graph
	from algos.dinic import dinic
	widths = [10, 15, 20, 25]
	layers = 6
	fig = plt.figure(figsize=(8,6))
	xs = []
	bfs_counts = []
	for w in widths:
		g, s, t = layered_graph([w]*layers, cap=10, full=True, seed=55)
		flow, residual = dinic(g, s, t)
		xs.append(w*layers + 2)
		bfs_counts.append(residual.metrics.get("blocking_flows"))
	plt.plot(xs, bfs_counts, marker='o')
	plt.xlabel('Total Nodes (approx width*layers)')
	plt.ylabel('Blocking Flows')
	plt.title('Dinic: Blocking Flows vs n (Layered Graphs)')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "Dinic_blocking_flows_vs_n.png")


def plot_greedy_fattest_increment_sequence():
	from utils.generators import random_sparse_graph
	from algos.greedy_fattest import greedy_fattest
	g, s, t = random_sparse_graph(200, edge_factor=3, max_cap=100, seed=202)
	flow, residual = greedy_fattest(g, s, t)
	incs = residual.metrics.get("path_flow_increments", [])
	fig = plt.figure(figsize=(9,5))
	plt.plot(range(1, len(incs)+1), incs, marker='o')
	plt.xlabel('Iteration')
	plt.ylabel('Flow Increment')
	plt.title('Greedy Fattest: Flow Increment per Iteration')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "GreedyFattest_flow_increment_sequence.png")


def plot_push_relabel_queue_growth():
	from utils.generators import random_sparse_graph
	from algos.push_relabel import push_relabel_fifo, push_relabel_highest
	g1, s1, t1 = random_sparse_graph(300, edge_factor=3, max_cap=50, seed=333)
	f_flow, f_res = push_relabel_fifo(g1, s1, t1)
	g2, s2, t2 = random_sparse_graph(300, edge_factor=3, max_cap=50, seed=334)
	h_flow, h_res = push_relabel_highest(g2, s2, t2)
	fig = plt.figure(figsize=(9,5))
	plt.plot(range(len(f_res.metrics.get('queue_sizes', []))), f_res.metrics.get('queue_sizes', []), label='FIFO variant')
	plt.plot(range(len(h_res.metrics.get('queue_sizes', []))), h_res.metrics.get('queue_sizes', []), label='Highest-Label variant')
	plt.xlabel('Iteration')
	plt.ylabel('Active Structure Size')
	plt.title('Push-Relabel: Queue/Active Size Growth')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "PushRelabel_queue_growth.png")


def plot_push_relabel_relabels_vs_runtime():
	from utils.generators import random_sparse_graph
	from algos.push_relabel import push_relabel_fifo, push_relabel_highest
	variants = [("FIFO", push_relabel_fifo), ("Highest", push_relabel_highest)]
	fig = plt.figure(figsize=(8,6))
	xs = []
	ys = []
	labels = []
	for name, fn in variants:
		g, s, t = random_sparse_graph(300, edge_factor=3, max_cap=60, seed=500 if name=='FIFO' else 501)
		import time
		start = time.time(); flow, residual = fn(g, s, t); end = time.time()
		xs.append(residual.metrics.get('relabel_ops'))
		ys.append(end-start)
		labels.append(name)
	plt.scatter(xs, ys)
	for i, lbl in enumerate(labels):
		plt.annotate(lbl, (xs[i], ys[i]))
	plt.xlabel('Relabel Operations')
	plt.ylabel('Runtime (s)')
	plt.title('Push-Relabel: Relabel Ops vs Runtime (Variants)')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	_save(fig, "PushRelabel_relabels_vs_runtime.png")


def run_all_algorithm_specific_plots():
	plot_ff_time_vs_capacity()
	plot_capacity_scaling_steps()
	plot_ek_time_vs_density()
	plot_dinic_blocking_flows_vs_n()
	plot_greedy_fattest_increment_sequence()
	plot_push_relabel_queue_growth()
	plot_push_relabel_relabels_vs_runtime()


if __name__ == "__main__":
	# A
	plot_time_vs_n()
	# B
	plot_time_vs_edges_for_fixed_n(n_fixed=200)
	# C
	plot_time_vs_capacity_range()
	# D
	plot_time_vs_sparsity(n_fixed=200)
	# E (using sparsity sweep for throughput demonstration)
	plot_throughput(_rp("sparsity_sweep_n200.csv"), "EdgesPerNode")
	# F
	plot_runtime_vs_topology()
	# G
	plot_distance_scaling()
	# H
	plot_augmenting_path_counts()
	# I
	plot_relabel_operations()
	# J
	plot_dinic_layer_depth()
	# K
	plot_flow_increments()
	# L
	plot_memory_usage()
	# M (placeholder)
	plot_parallelism_placeholder()
	# Algorithm-specific suite
	run_all_algorithm_specific_plots()

