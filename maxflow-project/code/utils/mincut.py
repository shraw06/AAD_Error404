from collections import deque

def extract_min_cut(residual_graph, s):
    n = residual_graph.n
    visited = [False]*n

    # BFS from s on residual graph
    q = deque([s])
    visited[s] = True

    while q:
        u = q.popleft()
        for v, cap, rev in residual_graph.adj[u]:
            if cap > 0 and not visited[v]:
                visited[v] = True
                q.append(v)

    # Edges from S to T are min-cut edges
    cut_edges = []
    for u in range(n):
        if visited[u]:
            for v, cap, rev in residual_graph.adj[u]:
                if not visited[v]:
                    cut_edges.append((u, v))

    return visited, cut_edges
