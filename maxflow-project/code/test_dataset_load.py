from utils.datasets import load_csv, load_snap
from utils.visualize import visualize_graph, visualize_flow, visualize_min_cut
from utils.mincut import extract_min_cut
from algos.dinic import dinic   # you can switch to any algorithm

# ------------------------------------------
# 1. Choose which dataset to load
# ------------------------------------------

# Example 1: Load your custom CSV real-world dataset
graph_path = "data/road_traffic_network.csv"
# we can add more diff graph paths here

g, s, t = load_csv(graph_path)
# g, s, t = load_csv("data/your_file.csv", default_cap=1)   if unweighted graph


# Example 2: If using SNAP format instead
# graph_path = "data/email-Eu-core.txt"
# g, s, t = load_snap(graph_path)

print(f"\nLoaded graph from: {graph_path}")
print(f"Number of nodes: {g.n}")
print(f"Source: {s}, Sink: {t}")

# ------------------------------------------
# 2. Run an algorithm (Dinic by default)
# ------------------------------------------

flow, residual = dinic(g, s, t)
print("\n==============================")
print(" Max Flow Computed =", flow)
print("==============================")

# ------------------------------------------
# 3. Visualize (only if graph is small)
# ------------------------------------------

do_visualize = False   # Change to True for small graphs (n < 30)

if do_visualize:
    visualize_graph(g, "Original Graph")
    visualize_flow(g, residual, "Flow on Real-World Dataset")

    S, cut_edges = extract_min_cut(residual, s)
    visualize_min_cut(residual, S, cut_edges, "Min Cut on Real Dataset")
