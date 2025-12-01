# Table 1: Asymptotic Complexity Summary

| Algorithm | Time Complexity | Space Complexity | Practical Notes |
|-----------|-----------------|------------------|-----------------|
| Ford–Fulkerson (DFS augment) | O(E * max_flow) | O(V + E) | Runtime depends on path choices; can degrade on long unit-capacity paths (max_flow ~ path length). |
| Edmonds–Karp (BFS shortest augmenting path) | O(V * E^2) | O(V + E) | Deterministic upper bound; slower on dense graphs but stable; good with unit capacities. |
| Dinic | O(min(V^{2/3}, √E) * E) typical; worst-case O(V^2 * E) | O(V + E) | Level graph + blocking flows; near-linear on many sparse / layered / bipartite graphs. |
| Capacity Scaling | O(E^2 * log C) (simplified bound) | O(V + E) | Improves when capacities span wide range; scales by threshold Δ halving each phase. |
| Greedy Fattest Path | O(E log V * augmentations) | O(V + E) | Chooses widest path; fewer augmentations on skewed capacity distributions. |
| Push–Relabel (Generic) | O(V^3) worst-case | O(V^2) (due to residual adjacency + height/excess arrays) | Highly efficient in practice; good candidate for parallelization; heuristics (gap, global relabel) reduce runtime. |
| Push–Relabel (Highest-Label) | O(V^3) worst-case; empirically faster | O(V^2) | Prioritizing highest label often reduces number of discharges. |
| Push–Relabel (FIFO) | O(V^3) worst-case | O(V^2) | Simple queue ordering; may yield more relabels. |
| Push–Relabel + Gap Heuristic | Same asymptotic worst-case | O(V^2) | Gap rule can prune unreachable height layers, cutting work significantly. |
| Push–Relabel + Global Relabeling | Amortized improvement | O(V^2) | Periodically recomputes exact distances to sink, accelerating admissible edge discovery. |

## Notes & Caveats
- max_flow refers to the numeric value of the maximum flow; for unit capacities this can be Θ(E) in worst-case path constructions.
- Dinic's commonly cited bound O(min(V^{2/3}, √E) * E) assumes unit capacities; for arbitrary capacities worst-case can reach O(V^2 * E).
- Capacity Scaling bound shown is a simplified heuristic; tighter bounds depend on implementation specifics (often O(E^2 log C)).
- Greedy Fattest Path doesn't guarantee optimal asymptotic improvement but empirically reduces augmentation count when capacities are heavy-tailed.
- Push–Relabel variants share worst-case upper bounds; heuristics drastically improve real performance on random and structured graphs.

## When to Use Which Algorithm
| Scenario | Recommended Algorithm | Reason |
|----------|-----------------------|--------|
| Long narrow path / adversarial augmenting path | Dinic / Push–Relabel | Avoids pathological many small augmentations. |
| Unit capacity dense graph | Edmonds–Karp / Dinic | Edmonds–Karp predictable; Dinic often faster with blocking flows. |
| Wide capacity range (1 .. 1e6) | Capacity Scaling / Push–Relabel | Capacity buckets reduce search space; push operations handle skew. |
| Need high practical performance quickly | Push–Relabel (Highest + Gap + Global) | Heuristics accelerate convergence. |
| Memory-constrained environment | Dinic | Lower memory overhead than push–relabel's larger auxiliary arrays. |

## References (Informal)
- Original analyses: Cormen et al. (CLRS), Tarjan & Goldberg (Push–Relabel), Dinic (1970s), Edmonds & Karp (1972).
- Practical performance notes derived from empirical benchmarking in this project (see generated plots in `plots/`).

*Generated automatically; extend or refine as needed for formal report.*

## Empirical Scaling (Experimental)
The following table summarizes empirically estimated scaling exponents derived from log–log least squares fits on the benchmark datasets (single trial per point). A value s in column "n (sparse)" roughly suggests time ≈ k · n^s for the sparse sweep (m ≈ 3n). Dense sweep uses m ≈ n²/4. Edge and density sweeps fix n=200. Capacity range maps capacity classes {low=10, medium=100, high=1000}. Treat these as indicative only; noise and small sample counts can skew slopes.

| Algorithm | n (sparse) | n (dense) | m @ n=200 | density @ n=200 | Cmax (dist) | Cmax (ff/cs) | MeanTime n=200 sparse (s) | MeanTime edge sweep avg (s) |
|-----------|-----------:|----------:|----------:|----------------:|------------:|-------------:|---------------------------:|----------------------------:|
| Ford-Fulkerson | 1.20 | 2.39 | 0.55 | 0.55 | 1.24 | 0.16 | 0.00057 | 0.05740 |
| Edmonds-Karp   | 1.01 | 2.74 | 1.12 | 1.12 | 0.29 | 0.05 | 0.00047 | 0.07360 |
| Dinic          | 1.13 | 2.10 | 0.55 | 0.55 | 0.49 | 0.09 | 0.00040 | 0.01317 |
| Push–Relabel   | 1.24 | 2.64 | 0.76 | 0.76 | 0.02 | 0.10 | 0.00031 | 0.00411 |
| Capacity-Scaling | 0.94 | 2.68 | 1.08 | 1.08 | 0.23 | 0.21 | 0.00098 | 0.04552 |
| Greedy-Fattest | 0.99 | 2.86 | 1.24 | 1.24 | 0.35 | 0.04 | 0.00063 | 0.18084 |

### Observations
- Push–Relabel shows a relatively low exponent vs edges/density compared to Ford–Fulkerson / Greedy Fattest, matching practical efficiency on denser graphs.
- Ford–Fulkerson’s high Cmax slope (≈1.24) highlights sensitivity to capacity range; Capacity-Scaling reduces this markedly.
- Edmonds–Karp exhibits higher sensitivity to edge growth (≈1.12) than Dinic (≈0.55) in the fixed-n edge sweep, aligning with theoretical heavier dependence on E.
- Greedy Fattest grows fastest with m/density in this setup, suggesting overhead in incremental widest-path selection as graph thickens.
- Dense exponents cluster around ~2–3, reflecting super-linear behavior as both n and implied m grow quickly.

### Caveats
- Single trial per point → variance not captured; add repetitions for confidence intervals.
- Very small runtimes (<1ms) are susceptible to timing noise; slopes for those algorithms (especially Push–Relabel at n=200) may understate true scaling.
- Capacity range experiment uses only three Cmax levels; a richer sweep would stabilize exponent estimates.
- Slopes do not directly translate to worst-case theoretical bounds; they reflect this generator mix only.

Raw data: `results/empirical_complexity_summary.csv`. Regenerate via `code/benchmark/aggregate_complexity.py` after new experiments.
