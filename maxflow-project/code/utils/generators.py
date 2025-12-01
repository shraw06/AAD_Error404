import random
from .graph import FlowGraph

# -------------------------------------------------------
# 1. RANDOM SPARSE GRAPH
# -------------------------------------------------------
def random_sparse_graph(n, edge_factor=3, max_cap=20, seed=None):
    """
    n = number of nodes
    edge_factor = approx edges per node (3 => ~3n edges)
    max_cap = max capacity on each edge
    """
    if seed is not None:
        random.seed(seed)

    g = FlowGraph(n)
    m = edge_factor * n
    edges = set()

    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if (u, v) in edges:
            continue

        cap = random.randint(1, max_cap)
        g.add_edge(u, v, cap)
        edges.add((u, v))

    s = 0
    t = n - 1

    return g, s, t


# -------------------------------------------------------
# 2. RANDOM DENSE GRAPH
# -------------------------------------------------------
def random_dense_graph(n, density=0.5, max_cap=50, seed=None):
    """
    density = probability of edge between u->v
    typical value: 0.3 to 0.7
    """
    if seed is not None:
        random.seed(seed)

    g = FlowGraph(n)

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if random.random() < density:
                cap = random.randint(1, max_cap)
                g.add_edge(u, v, cap)

    s = 0
    t = n - 1

    return g, s, t


# -------------------------------------------------------
# 3. BIPARTITE GRAPH (perfect for matching tests)
# -------------------------------------------------------
def bipartite_graph(left, right, p=0.1, max_cap=10, seed=None):
    """
    left = size of left partition
    right = size of right partition
    p = probability of an edge from left->right
    """
    if seed is not None:
        random.seed(seed)

    n = left + right + 2
    s = 0
    t = n - 1
    g = FlowGraph(n)

    # source -> left side
    for i in range(left):
        g.add_edge(s, 1+i, random.randint(1, max_cap))

    # left -> right edges
    for i in range(left):
        for j in range(right):
            if random.random() < p:
                g.add_edge(1+i, 1+left+j, random.randint(1, max_cap))

    # right -> sink
    for j in range(right):
        g.add_edge(1+left+j, t, random.randint(1, max_cap))

    return g, s, t


# -------------------------------------------------------
# 4. GRID GRAPH (for image segmentation-like tasks)
# -------------------------------------------------------
def grid_graph(h, w, max_cap=20, seed=None):
    """
    Creates a h x w 4-connected grid.

    Nodes:
    1 ... h*w are pixels
    0 = source
    h*w+1 = sink

    Edges:
    - between neighbors with random capacities
    - source connected to top-left region (foreground seeds)
    - sink connected to bottom-right region (background seeds)
    """
    if seed is not None:
        random.seed(seed)

    n = h*w + 2
    s = 0
    t = n - 1

    g = FlowGraph(n)

    # Function to index pixels 2D -> 1D graph nodes
    def node(i, j):
        return 1 + i*w + j

    # connect neighbors
    for i in range(h):
        for j in range(w):
            u = node(i, j)

            # 4 neighbors
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    v = node(ni, nj)
                    g.add_edge(u, v, random.randint(1, max_cap))

    # connect some pixels to source (simulate foreground seeds)
    for i in range(h//3):
        for j in range(w//3):
            g.add_edge(s, node(i, j), max_cap * 2)

    # connect some pixels to sink (simulate background seeds)
    for i in range(2*h//3, h):
        for j in range(2*w//3, w):
            g.add_edge(node(i, j), t, max_cap * 2)

    return g, s, t


# -------------------------------------------------------
# 5. HEAVY-CAPACITY GRAPH (for capacity-scaling experiments)
# -------------------------------------------------------
def heavy_capacity_graph(n, m, cap_range=(1, 10**6), seed=None):
    """
    edges have huge capacity range (1 to 1e6).
    Demonstrates capacity scaling advantage.
    """
    if seed is not None:
        random.seed(seed)

    g = FlowGraph(n)
    edges = set()

    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if (u, v) in edges:
            continue

        cap = random.randint(cap_range[0], cap_range[1])
        g.add_edge(u, v, cap)
        edges.add((u, v))

    s = 0
    t = n - 1

    return g, s, t


# -------------------------------------------------------
# 6. LAYERED GRAPH (showcases blocking flows for Dinic)
# -------------------------------------------------------
def layered_graph(layer_sizes, cap=10, full=True, seed=None):
    """
    Constructs a layered DAG from source through successive layers to sink.

    layer_sizes: list of ints, number of nodes in each internal layer.
    cap: capacity for each inter-layer edge (single value).
    full: if True connect every node in layer i to every node in layer i+1,
          else create a single random edge per node to next layer.

    Returns: (FlowGraph, source, sink)
    """
    if seed is not None:
        random.seed(seed)

    # Node indexing: 0 = source, last = sink, internal nodes packed layer by layer.
    total_internal = sum(layer_sizes)
    n = total_internal + 2
    s = 0
    t = n - 1
    g = FlowGraph(n)

    # Build list of node ids for each layer
    layers = []
    current = 1
    for size in layer_sizes:
        layer_nodes = list(range(current, current + size))
        layers.append(layer_nodes)
        current += size

    # Source to first layer
    for u in layers[0]:
        g.add_edge(s, u, cap)

    # Inter-layer edges
    for i in range(len(layers) - 1):
        A = layers[i]
        B = layers[i + 1]
        if full:
            for u in A:
                for v in B:
                    g.add_edge(u, v, cap)
        else:
            for u in A:
                v = random.choice(B)
                g.add_edge(u, v, cap)

    # Last layer to sink
    for u in layers[-1]:
        g.add_edge(u, t, cap)

    return g, s, t


# -------------------------------------------------------
# 7. WORST-CASE LONG PATH GRAPH (bad for Ford-Fulkerson DFS)
# -------------------------------------------------------
def worst_case_long_path_graph(length, cap=1):
    """
    Constructs a simple path from source to sink with 'length' internal nodes.
    All edges have unit (or given) capacity. Max flow equals cap and algorithms
    that repeatedly augment along the long path will take O(length * max_flow) steps.

    length: number of internal nodes (path length minus 1 in edges)
    cap: capacity on each edge (default 1 for unit path demonstrating worst-case)
    """
    n = length + 2  # source + internal + sink
    s = 0
    t = n - 1
    g = FlowGraph(n)
    # chain s -> 1 -> 2 -> ... -> length -> t
    for u in range(0, n - 1):
        g.add_edge(u, u + 1, cap)
    return g, s, t


# -------------------------------------------------------
# 8. UNIT CAPACITY RANDOM GRAPH (good for Edmonds-Karp)
# -------------------------------------------------------
def unit_capacity_graph(n, density=0.1, seed=None):
    """
    Random directed graph where every edge has capacity = 1.
    Useful for highlighting Edmonds-Karp performance on unit networks.

    n: number of nodes (including source & sink)
    density: probability of edge between distinct u->v
    """
    if seed is not None:
        random.seed(seed)

    g = FlowGraph(n)
    s = 0
    t = n - 1
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if random.random() < density:
                g.add_edge(u, v, 1)
    return g, s, t
