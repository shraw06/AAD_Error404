# Maximum Flow Algorithms - Interactive CLI & Benchmarking

## 🚀 Quick Setup

### Step 1: Install Dependencies
```bash
pip install pandas numpy networkx matplotlib
```

**Required packages:**
- `pandas` - For VFOA network data loading and CSV handling
- `numpy` - For numerical computations
- `networkx` - For graph visualization
- `matplotlib` - For plotting graphs and results

### Step 2: Activate Python Environment (if using venv)
```bash
source venv/bin/activate
```

---

### Method 1: Interactive CLI (Recommended for Live Demo)

```bash
python3 code/cli.py
```

The CLI is **fully interactive** with 7 menu options. Navigate using numbers 1-7.

#### **Demo Workflow for Live Presentation:**

**Step 1: Visual Component Demo** (Option 1 or 2)
```
Select Option 1: Visualize Random Graph
→ Choose graph type (e.g., "sparse" or "dense")
→ Enter n=20, edge_factor=3 (for sparse)
→ Select algorithm (e.g., "Dinic")
→ Choose 'y' for visualization

Result: Three plots displayed:
  1. Original graph with capacities
  2. Flow on edges (thickness = flow amount)
  3. Min-cut partition (source/sink sides highlighted)
```

**Step 2: Benchmarking Demo** (Option 4)
```
Select Option 4: Run All Algorithms on Chosen Graph
→ Choose graph type (e.g., "dense")
→ Enter n=30, density=0.3
→ Watch all 6 algorithms execute in real-time

Result: Comparison table showing:
  - Algorithm names
  - Runtime (seconds)
  - Max flow value (all should match!)
  - CSV file saved to results/
```

**Step 3: VFOA Network Analysis** (Option 7)
```
Select Option 7: VFOA Network Explorer
→ Select sub-option 4: Batch analyze all networks
→ Choose algorithm (e.g., "Push-Relabel")

Result: Progress bars for 24 networks, then:
  - Summary statistics table
  - CSV with detailed results
  - Comparison across all networks
```

**Step 4: Generate Visual Comparisons** (Option 6)
```
Select Option 6: Plot Benchmark Results
→ Select a CSV file from results/ directory
→ Choose plot type (e.g., "Runtime bar chart")

Result: Publication-ready plots saved to plots/
```

---

## 📊 Method 2: Direct Benchmarking Scripts

### Option A: Quick Visual Demo
```bash
python3 code/test_visualization.py
```

**What it shows:**
- Initial graph structure
- Flow graph with edge flows
- Min-cut graph with partition coloring

**To change algorithm:** Edit the import at the top of the file
```python
# from algos.dinic import Dinic as Algorithm
from algos.push_relabel import PushRelabel as Algorithm
```

### Option B: Run Full Benchmarking Suite
```bash
python3 code/run_all_experiments.py
```

**What it generates:**
CSV results files in `results/` for:
- ✅ Size sweep (10 to 200 nodes)
- ✅ Density sweep (0.1 to 0.9 density)
- ✅ Bipartite graphs (matching problems)
- ✅ Grid graphs (image segmentation simulation)
- ✅ Capacity distribution experiments
- ✅ Layered graphs (Dinic optimization)
- ✅ Long-path worst-case graphs
- ✅ Unit capacity graphs

**Runtime:** Approximately 5-10 minutes for all experiments

**Expected output files:**
```
results/
├── size_sweep_dense.csv
├── size_sweep_edgefactor3.csv
├── density_experiment.csv
├── bipartite_experiment.csv
├── grid_experiment.csv
├── capacity_range_ff_cs.csv
├── layered_experiment.csv
├── long_path_experiment.csv
└── unit_capacity_experiment.csv
```

### Option C: Generate Visual Plots from Results
```bash
python3 code/plot_results.py
```

**What it generates:**
PNG plots in `plots/` directory:
- ✅ Runtime comparison bar charts
- ✅ Runtime vs. size (scaling analysis)
- ✅ Runtime vs. density
- ✅ Algorithm-specific metrics

**Interactive prompts guide you through:**
1. Select a CSV file from `results/`
2. Choose plot type
3. Plots automatically saved and displayed

### Option D: Test Real Dataset Loading
```bash
python3 code/test_dataset_load.py
```

**What it tests:**
- Email network loading (1,005 nodes)
- Road traffic network loading
- Dataset format validation

## 🔍 What Each Component Demonstrates

| Component | Demonstrates | Visual Output | Benchmark Output |
|-----------|--------------|---------------|------------------|
| **CLI Option 1-2** | Graph visualization, flow visualization, min-cut | ✅ 3 matplotlib plots | - |
| **CLI Option 3** | Single algorithm execution | ✅ Optional visualization | Runtime metrics |
| **CLI Option 4** | All algorithms comparison | - | ✅ Comparison table + CSV |
| **CLI Option 5** | Experiment runner | - | ✅ Multiple CSV files |
| **CLI Option 6** | Results visualization | ✅ Publication plots | - |
| **CLI Option 7** | VFOA network analysis | ✅ Heatmaps, comparisons | ✅ Batch results CSV |
| **test_visualization.py** | Quick visual demo | ✅ 3 plots (graph/flow/cut) | - |
| **run_all_experiments.py** | Comprehensive benchmarking | - | ✅ 8+ CSV files |
| **plot_results.py** | Results visualization | ✅ Multiple plot types | - |

---

## 📈 Expected Visual Outputs

### From CLI Visualization Options:
1. **Graph Structure Plot**
   - Nodes positioned with spring layout
   - Edges labeled with capacities
   - Source (blue) and Sink (red) highlighted

2. **Flow Plot**
   - Edge thickness proportional to flow
   - Edge labels show "flow/capacity"
   - Max flow value displayed in title

3. **Min-Cut Plot**
   - Source partition colored green
   - Sink partition colored orange
   - Cut edges highlighted in red

### From Plot Generation:
1. **Runtime Bar Charts** - Compare all 6 algorithms
2. **Scaling Plots** - Runtime vs. number of nodes (line plots)
3. **Density Analysis** - Runtime vs. edge density
4. **Algorithm Metrics** - Augmenting paths, relabels, etc.

## ⚡ Quick Command Reference

```bash
# Most important commands for live demo:
python3 code/cli.py                      # Interactive CLI (best for live demo)
python3 code/test_visualization.py        # Quick visual demo
python3 code/run_all_experiments.py       # Full benchmarking
python3 code/plot_results.py             # Generate plots from results
```

## Credits

VFOA Dataset Reference:
- C. Bai, S. Kumar, J. Leskovec, M. Metzger, J.F. Nunamaker, V.S. Subrahmanian
- "Predicting Visual Focus of Attention in Multi-person Discussion Videos"
- International Joint Conference on Artificial Intelligence (IJCAI), 2019
- Dataset: http://snap.stanford.edu/data/comm-f2f-Resistance.html
