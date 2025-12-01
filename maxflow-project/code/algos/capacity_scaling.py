from collections import deque

def capacity_scaling(graph, s, t):
    """Capacity scaling algorithm with instrumentation.

    Metrics collected:
      scaling_steps: number of delta phases processed
      path_flow_increments: list of bottleneck values for augmenting paths under current delta
    """
    g = graph.copy()
    max_cap = max(cap for u in range(g.n) for _, cap, _ in g.adj[u])
    delta = 1
    while delta <= max_cap:
        delta <<= 1
    delta >>= 1

    flow = 0
    metrics = {
        "scaling_steps": 0,
        "path_flow_increments": []
    }

    while delta > 0:
        phase_augmented = False
        while True:
            parent = [(-1, -1)]*g.n
            q = deque([s])
            parent[s] = (s, -1)

            while q and parent[t][0] == -1:
                u = q.popleft()
                for i, (v, cap, rev) in enumerate(g.adj[u]):
                    if cap >= delta and parent[v][0] == -1:
                        parent[v] = (u, i)
                        q.append(v)

            if parent[t][0] == -1:
                break

            # Find bottleneck
            bottleneck = float('inf')
            v = t
            while v != s:
                u, ei = parent[v]
                bottleneck = min(bottleneck, g.adj[u][ei][1])
                v = u

            # Augment
            v = t
            while v != s:
                u, ei = parent[v]
                g.augment(u, ei, bottleneck)
                v = u

            flow += bottleneck
            metrics["path_flow_increments"].append(bottleneck)
            phase_augmented = True

        metrics["scaling_steps"] += 1
        delta //= 2

    g.metrics = metrics
    return flow, g
