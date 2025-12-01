from collections import deque

def dinic(graph, s, t):
    g = graph.copy()
    n = g.n
    INF = 10**18
    metrics = {
        "blocking_flows": 0,            # number of BFS level graph constructions that led to at least one augmentation
        "max_level_depth": 0,           # maximum layer depth observed across BFS phases
        "path_flow_increments": []      # each individual DFS augmentation flow value
    }

    # BFS to build level graph
    def bfs():
        level = [-1]*n
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in g.adj[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    # DFS to send blocking flow
    def dfs(u, pushed, level, it):
        if u == t:
            return pushed
        for i in range(it[u], len(g.adj[u])):
            it[u] = i
            v, cap, rev = g.adj[u][i]
            if cap > 0 and level[v] == level[u] + 1:
                f = dfs(v, min(pushed, cap), level, it)
                if f > 0:
                    g.augment(u, i, f)
                    return f
        return 0

    flow = 0

    while True:
        level = bfs()
        if level[t] < 0:
            break
        # Track max depth this phase
        phase_max_depth = max(l for l in level if l >= 0)
        if phase_max_depth > metrics["max_level_depth"]:
            metrics["max_level_depth"] = phase_max_depth
        it = [0]*n

        while True:
            pushed = dfs(s, INF, level, it)
            if pushed == 0:
                break
            flow += pushed
            metrics["path_flow_increments"].append(pushed)
        # If at least one path augmented in this phase, count as blocking flow
        if metrics["path_flow_increments"] and (len(metrics["path_flow_increments"]) - metrics.get("_counted_paths_prev", 0)) > 0:
            metrics["blocking_flows"] += 1
        metrics["_counted_paths_prev"] = len(metrics["path_flow_increments"])

    # cleanup temporary key
    if "_counted_paths_prev" in metrics:
        del metrics["_counted_paths_prev"]
    g.metrics = metrics
    return flow, g
