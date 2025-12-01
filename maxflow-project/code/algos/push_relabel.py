def push_relabel(graph, s, t):
    """Basic push-relabel with FIFO-style single pass (original implementation) plus instrumentation."""
    g = graph.copy()
    n = g.n
    height = [0]*n
    excess = [0]*n
    height[s] = n
    metrics = {
        "relabel_ops": 0,
        "variant": "basic"
    }

    # Pre-flow initialization
    for i, (v, cap, rev) in enumerate(g.adj[s]):
        if cap > 0:
            g.augment(s, i, cap)
            excess[v] += cap
            excess[s] -= cap

    def discharge(u):
        while excess[u] > 0:
            pushed_something = False
            for i, (v, cap, rev) in enumerate(g.adj[u]):
                if cap > 0 and height[u] == height[v] + 1:
                    pushed = min(excess[u], cap)
                    g.augment(u, i, pushed)
                    excess[u] -= pushed
                    excess[v] += pushed
                    pushed_something = True
                    if excess[u] == 0:
                        return
            if not pushed_something:
                # Relabel
                min_height = min((height[v] for v, cap, rev in g.adj[u] if cap > 0), default=None)
                if min_height is None:
                    # No admissible edges; break to avoid infinite loop
                    return
                height[u] = min_height + 1
                metrics["relabel_ops"] += 1

    active = [i for i in range(n) if i not in (s, t)]
    for u in active:
        discharge(u)

    max_flow = sum(cap for v, cap, rev in g.adj[t])
    g.metrics = metrics
    return max_flow, g


def _push_relabel_variant(graph, s, t, strategy="FIFO", gap=False, global_relabel=False, global_freq=0):
    """Generalized push-relabel with selectable strategies and instrumentation.

    strategy: FIFO | Highest
    gap: enable gap heuristic (removes nodes of empty height layer > h)
    global_relabel: perform periodic global relabel BFS from sink to recompute heights
    global_freq: perform global relabel after this many relabel operations (if >0)
    """
    from collections import deque
    g = graph.copy()
    n = g.n
    height = [0]*n
    excess = [0]*n
    height[s] = n
    metrics = {
        "relabel_ops": 0,
        "variant": strategy + ("+Gap" if gap else "") + ("+Global" if global_relabel else ""),
        "queue_sizes": []  # track growth of active structure (queue or list)
    }

    # Pre-flow
    for i, (v, cap, rev) in enumerate(g.adj[s]):
        if cap > 0:
            g.augment(s, i, cap)
            excess[v] += cap
            excess[s] -= cap

    active = [i for i in range(n) if i not in (s, t) and excess[i] > 0]
    fifo_queue = deque(active) if strategy == "FIFO" else None

    def global_relabel_bfs():
        from collections import deque
        level = [None]*n
        q = deque([t])
        level[t] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in g.adj[u]:  # reverse edges with cap>0 are residual from v->u
                if g.adj[v][rev][1] > 0 and level[v] is None:
                    level[v] = level[u] + 1
                    q.append(v)
        for i in range(n):
            height[i] = (level[i] if level[i] is not None else n*2)

    if global_relabel:
        global_relabel_bfs()

    def relabel(u):
        min_h = None
        for v, cap, rev in g.adj[u]:
            if cap > 0:
                h = height[v]
                if min_h is None or h < min_h:
                    min_h = h
        if min_h is None:
            return False
        height[u] = min_h + 1
        metrics["relabel_ops"] += 1
        if global_relabel and global_freq > 0 and metrics["relabel_ops"] % global_freq == 0:
            global_relabel_bfs()
        return True

    def discharge(u):
        i = 0
        while excess[u] > 0:
            if i >= len(g.adj[u]):
                if not relabel(u):
                    break
                i = 0
                continue
            v, cap, rev = g.adj[u][i]
            if cap > 0 and height[u] == height[v] + 1:
                pushed = min(excess[u], cap)
                g.augment(u, i, pushed)
                excess[u] -= pushed
                excess[v] += pushed
                if v not in (s, t) and excess[v] == pushed:  # newly active
                    if strategy == "FIFO":
                        fifo_queue.append(v)
                continue  # try same index again (capacity may remain)
            i += 1

        return excess[u] == 0

    iteration = 0
    while True:
        if strategy == "FIFO":
            if not fifo_queue:
                break
            u = fifo_queue.popleft()
        else:  # Highest label selection
            if not active:
                break
            u = max(active, key=lambda x: height[x])
            active.remove(u)

        prev_height = height[u]
        finished = discharge(u)
        if not finished and u not in (s, t):
            # still has excess after relabel
            if strategy == "FIFO":
                fifo_queue.append(u)
            else:
                active.append(u)

        # Gap heuristic: if a height layer becomes empty remove nodes > that height
        if gap:
            counts = {}
            for v in range(n):
                if v not in (s, t) and excess[v] > 0:
                    counts[height[v]] = counts.get(height[v], 0) + 1
            # find a height h with count==0 and h < max height
            max_h = max(height)
            present = set(counts.keys())
            for h in range(max_h):
                if h not in present:
                    for v in range(n):
                        if height[v] > h and height[v] < n*2:  # relabel unreachable
                            height[v] = n*2
                    break
        # record queue/active size
        if strategy == "FIFO":
            metrics["queue_sizes"].append(len(fifo_queue))
        else:
            metrics["queue_sizes"].append(len(active))
        iteration += 1

    max_flow = sum(cap for v, cap, rev in g.adj[t])
    g.metrics = metrics
    return max_flow, g


def push_relabel_fifo(graph, s, t):
    return _push_relabel_variant(graph, s, t, strategy="FIFO")

def push_relabel_highest(graph, s, t):
    return _push_relabel_variant(graph, s, t, strategy="Highest")

def push_relabel_gap(graph, s, t):
    return _push_relabel_variant(graph, s, t, strategy="Highest", gap=True)

def push_relabel_global(graph, s, t):
    # periodic global relabel every 50 relabel ops
    return _push_relabel_variant(graph, s, t, strategy="Highest", global_relabel=True, global_freq=50)
