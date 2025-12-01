from collections import deque

def edmonds_karp(graph, s, t):
    g = graph.copy()
    n = g.n
    max_flow = 0
    metrics = {
        "augmenting_paths": 0,
        "path_flow_increments": []
    }

    while True:
        parent = [(-1, -1)] * n
        q = deque([s])
        parent[s] = (s, -1)

        # BFS to find shortest augmenting path
        while q and parent[t][0] == -1:
            u = q.popleft()
            for i, (v, cap, rev) in enumerate(g.adj[u]):
                if cap > 0 and parent[v][0] == -1:
                    parent[v] = (u, i)
                    q.append(v)

        # No path found
        if parent[t][0] == -1:
            break

        # Find bottleneck
        flow = float('inf')
        v = t
        while v != s:
            u, ei = parent[v]
            flow = min(flow, g.adj[u][ei][1])
            v = u

        # Augment flow
        v = t
        while v != s:
            u, ei = parent[v]
            g.augment(u, ei, flow)
            v = u

        max_flow += flow
        metrics["augmenting_paths"] += 1
        metrics["path_flow_increments"].append(flow)

    g.metrics = metrics
    return max_flow, g
