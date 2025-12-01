"""Ford-Fulkerson using iterative DFS to find augmenting paths.

Replaces the previous recursive implementation to avoid hitting Python's
recursion limit on long or deep augmenting paths (e.g., n > 1000 in chain graphs).

graph: FlowGraph
s: source node index
t: sink node index

Returns: (max_flow, residual_graph)
"""

def ford_fulkerson(graph, s, t):
    g = graph.copy()
    n = g.n

    max_flow = 0
    metrics = {
        "augmenting_paths": 0,
        "path_flow_increments": []
    }

    while True:
        # Parent tracking: parent[v] = (u, edge_index_in_u_adj)
        parent = [None] * n
        stack = [s]
        parent[s] = (-1, -1)

        # Iterative DFS
        while stack and parent[t] is None:
            u = stack.pop()
            for idx, (v, cap, rev) in enumerate(g.adj[u]):
                if cap > 0 and parent[v] is None:
                    parent[v] = (u, idx)
                    if v == t:
                        break
                    stack.append(v)

        # No augmenting path
        if parent[t] is None:
            break

        # Compute bottleneck
        bottleneck = float('inf')
        v = t
        while v != s:
            u, idx = parent[v]
            cap = g.adj[u][idx][1]
            if cap < bottleneck:
                bottleneck = cap
            v = u

        # Augment along path
        v = t
        while v != s:
            u, idx = parent[v]
            g.augment(u, idx, bottleneck)
            v = u

        max_flow += bottleneck
        metrics["augmenting_paths"] += 1
        metrics["path_flow_increments"].append(bottleneck)

    g.metrics = metrics
    return max_flow, g
