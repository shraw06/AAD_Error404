from utils.visualize import visualize_graph, visualize_flow, visualize_min_cut
from utils.mincut import extract_min_cut
from utils.generators import random_sparse_graph
from algos.dinic import dinic

# 1. Build small graph
g, s, t = random_sparse_graph(n=10, edge_factor=3, max_cap=10, seed=1)

visualize_graph(g, "Initial Graph")

# 2. Run the chosen algorithm (just change this line)
flow, residual = dinic(g, s, t)             # CHANGE THIS LINE
print("Flow =", flow)

# 3. Flow visualization
visualize_flow(g, residual, "Flow Visualization")

# 4. Extract and visualize Min-Cut
S, cut_edges = extract_min_cut(residual, s)
visualize_min_cut(residual, S, cut_edges, "Min-Cut")


