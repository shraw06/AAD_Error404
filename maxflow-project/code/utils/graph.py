class FlowGraph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        # forward edge: [to, capacity, reverse_edge_index]
        self.adj[u].append([v, cap, len(self.adj[v])])
        # backward edge (residual)
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])

    def copy(self):
        g = FlowGraph(self.n)
        g.adj = [
            [[to, cap, rev] for (to, cap, rev) in node]
            for node in self.adj
        ]
        return g

    def augment(self, u, edge_index, flow):
        v, cap, rev = self.adj[u][edge_index]
        # reduce forward capacity
        self.adj[u][edge_index][1] -= flow
        # increase backward capacity
        self.adj[v][rev][1] += flow
