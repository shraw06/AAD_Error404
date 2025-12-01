import heapq

def greedy_fattest(graph, s, t):
    g = graph.copy()
    n = g.n

    max_flow = 0
    metrics = {
        "augmenting_paths": 0,
        "path_flow_increments": []
    }

    while True:
        bottleneck = [0]*n
        parent = [(-1, -1)]*n

        bottleneck[s] = float('inf')
        pq = [(-bottleneck[s], s)]

        while pq:
            b, u = heapq.heappop(pq)
            b = -b

            if b < bottleneck[u]:
                continue

            if u == t:
                break

            for i, (v, cap, rev) in enumerate(g.adj[u]):
                if cap > 0:
                    new_bottle = min(b, cap)
                    if new_bottle > bottleneck[v]:
                        bottleneck[v] = new_bottle
                        parent[v] = (u, i)
                        heapq.heappush(pq, (-new_bottle, v))

        if bottleneck[t] == 0:
            break

        flow = bottleneck[t]
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
