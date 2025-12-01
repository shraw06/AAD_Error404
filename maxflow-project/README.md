Activate python environment - source venv/bin/activate

## 🚀 Quick Start

```bash
python3 code/cli.py
```

### Main Features:
  1-6. Original functionality (graph generation, algorithms, experiments)
  **NEW:** 7. VFOA Network Explorer (62 networks)

### What's New?
- **VFOA Analysis**: Explore 62 Visual Focus of Attention temporal networks
- **Batch Processing**: Run max-flow on all networks with progress bars
- **Rich Visualizations**: Heatmaps, attention matrices, statistical charts
- **Network Comparison**: Compare multiple VFOA networks side-by-side

📖 **See [QUICK_START.md](QUICK_START.md) for step-by-step guide**
📚 **See [NETWORK_DATASETS_GUIDE.md](NETWORK_DATASETS_GUIDE.md) for full documentation**

This is user interactive.
We can also run a particular thing by - python3 code/test_visualization.py nd changing in code what all graph/ algo we want.


Visualize a Random Graph
  python3 code/test_visualization.py

  Shows:
  Initial graph
  Flow graph
  Min-cut graph
  To change algorithm, edit the import at the top.


Run All Experiments
  python3 code/run_all_experiments.py

  Generates CSV results for:
  Size sweep
  Density sweep
  Bipartite graphs
  Grid graphs
  Capacity distribution
  Layered graphs
  Long-path worst-case graphs
  Unit capacity graphs
  Outputs saved in results/.


Plot Comparison Graphs
  python3 code/plot_results.py

  Generates:
  Runtime comparisons
  Runtime vs size
  Runtime vs density
  Algorithm metrics


Load Real Datasets
  python3 code/test_dataset_load.py

## Credits

VFOA Dataset Reference:
- C. Bai, S. Kumar, J. Leskovec, M. Metzger, J.F. Nunamaker, V.S. Subrahmanian
- "Predicting Visual Focus of Attention in Multi-person Discussion Videos"
- International Joint Conference on Artificial Intelligence (IJCAI), 2019
- Dataset: http://snap.stanford.edu/data/comm-f2f-Resistance.html