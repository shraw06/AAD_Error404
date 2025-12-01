import os
import random
from utils.graph import FlowGraph


# ------------------------------------------------------
# Helper: get new S and T
# ------------------------------------------------------
def choose_source_sink(n):
    """
    Common function to pick source and sink for real-world graphs.
    For large graphs, choose:
    - smallest-degree node as source
    - largest-degree node as sink
    """
    # For now, pick 0 as s and n-1 as t (simple, safe)
    return 0, n - 1


# ------------------------------------------------------
# 1. LOAD SNAP FORMAT (edge list)
# ------------------------------------------------------
def load_snap(path, default_cap=1):
    """
    SNAP format: each line: u  v
    Nodes are integers but may not be 0..n-1.
    """
    edges = []
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v, default_cap))
            nodes.add(u)
            nodes.add(v)

    # reindex nodes to 0..n-1
    mapping = {old: i for i, old in enumerate(sorted(nodes))}
    n = len(mapping)

    g = FlowGraph(n)
    for u, v, cap in edges:
        g.add_edge(mapping[u], mapping[v], cap)

    s, t = choose_source_sink(n)
    return g, s, t


# ------------------------------------------------------
# 2. LOAD KONECT FORMAT (u v w)
# ------------------------------------------------------
def load_konect(path):
    """
    Format:
    % comment
    u v w
    If no w, default capacity = 1
    """
    edges = []
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            w = int(parts[2]) if len(parts) >= 3 else 1
            edges.append((u, v, w))
            nodes.update([u, v])

    # reindex
    mapping = {old: i for i, old in enumerate(sorted(nodes))}
    n = len(mapping)

    g = FlowGraph(n)
    for u, v, w in edges:
        g.add_edge(mapping[u], mapping[v], w)

    s, t = choose_source_sink(n)
    return g, s, t


# ------------------------------------------------------
# 3. LOAD CSV edge list: u,v,capacity
# ------------------------------------------------------
def load_csv(path, default_cap=1):
    import csv
    from .graph import FlowGraph

    edges = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:

            # Skip empty lines
            if not row:
                continue

            # Skip header (u,v,capacity)
            if row[0].lower() in ["u", "node", "source"]:
                continue

            # If file has only 2 columns → assign default capacity
            if len(row) == 2:
                u = int(row[0])
                v = int(row[1])
                cap = default_cap

            # If file has 3 columns (u,v,capacity)
            elif len(row) == 3:
                u = int(row[0])
                v = int(row[1])
                cap = int(row[2])

            else:
                continue

            edges.append((u, v, cap))

    # Build graph WITH CORRECT CONSTRUCTOR
    max_node = max(max(u, v) for u, v, _ in edges)
    g = FlowGraph(max_node + 1)

    # Add edges
    for u, v, cap in edges:
        g.add_edge(u, v, cap)

    # Source = 0, Sink = last node by default
    s = 0
    t = max_node

    return g, s, t


# ------------------------------------------------------
# 4. LOAD TXT: "u v cap"
# ------------------------------------------------------
def load_txt(path):
    edges = []
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            cap = int(parts[2]) if len(parts) >= 3 else 1
            edges.append((u, v, cap))
            nodes.update([u, v])

    mapping = {old: i for i, old in enumerate(sorted(nodes))}
    n = len(mapping)

    g = FlowGraph(n)
    for u, v, cap in edges:
        g.add_edge(mapping[u], mapping[v], cap)

    s, t = choose_source_sink(n)
    return g, s, t


# ------------------------------------------------------
# 5. LOAD DIMACS (DTU Max-Flow Benchmark)
# ------------------------------------------------------
def load_dimacs(path):
    """
    DIMACS format:
    c comment
    p max n m
    a u v cap
    s source
    t sink
    """
    g = None
    s = None
    t = None

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue

            parts = line.strip().split()

            if parts[0] == 'p':
                # p max n m
                n = int(parts[2])
                g = FlowGraph(n)

            elif parts[0] == 'a':
                # a u v cap
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                cap = int(parts[3])
                g.add_edge(u, v, cap)

            elif parts[0] == 's':
                s = int(parts[1]) - 1

            elif parts[0] == 't':
                t = int(parts[1]) - 1

    # If s or t not explicitly set (rare), choose defaults
    if s is None or t is None:
        s, t = choose_source_sink(g.n)

    return g, s, t
