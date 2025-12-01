import time
import pandas as pd
from algos.ford_fulkerson import ford_fulkerson
from algos.edmonds_karp import edmonds_karp
from algos.dinic import dinic
from algos.push_relabel import push_relabel
from algos.capacity_scaling import capacity_scaling
from algos.greedy_fattest import greedy_fattest


ALGORITHMS = {
    "Ford-Fulkerson": ford_fulkerson,
    "Edmonds-Karp": edmonds_karp,
    "Dinic": dinic,
    "Push-Relabel": push_relabel,
    "Capacity-Scaling": capacity_scaling,
    "Greedy-Fattest": greedy_fattest
}


def benchmark_single_graph(graph, s, t, trials=1, time_limit=None):
    """
    Runs all algorithms on a single graph.
    Returns a dictionary with runtime and flow values.
    """
    results = []

    for name, algo in ALGORITHMS.items():
        run_times = []
        final_flow = None

        for _ in range(trials):
            g_copy = graph.copy()
            start = time.time()
            flow, _ = algo(g_copy, s, t)
            end = time.time()

            run_times.append(end - start)
            final_flow = flow

            if time_limit is not None and (end - start) > time_limit:
                break

        results.append({
            "Algorithm": name,
            "Flow": final_flow,
            "MeanTime": sum(run_times) / len(run_times),
            "StdTime": pd.Series(run_times).std()
        })

    return pd.DataFrame(results)


def save_results(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")
