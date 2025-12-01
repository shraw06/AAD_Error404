# AAD Final Project - Maximum Flow Algorithms

**Team Name:** Error404  
**Project:** Comprehensive Analysis and Implementation of Maximum Flow Algorithms

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Main Project: MaxFlow Algorithms](#main-project-maxflow-algorithms)
4. [Bonus Part: Image Segmentation](#bonus-part-image-segmentation)
5. [Installation & Setup](#installation--setup)
6. [How to Run](#how-to-run)
7. [Live Demo Instructions](#live-demo-instructions)
8. [Features & Implementation](#features--implementation)
9. [Documentation](#documentation)
10. [Credits](#credits)

---

## Project Overview

This project provides a comprehensive implementation and analysis of **six maximum flow algorithms** with extensive benchmarking, visualization, and real-world applications. The project includes:

1. **Main Project**: MaxFlow algorithm implementations with interactive CLI, benchmarking suite, and VFOA network analysis
2. **Bonus Part**: Image segmentation application using max-flow/min-cut and 3D vizualisation of Edmonds Karp and Cpacoty Scaling algorithms.
---

## Project Structure

```

├── AAD_final_project/
|   ├── README.md
│   ├── maxflow-project/               # Main project
│   │   ├── code/                      # Source code
│   │   │   ├── algos/                 # Algorithm implementations
│   │   │   ├── benchmark/             # Benchmarking suite
│   │   │   ├── utils/                 # Utilities (graph, generators, loaders)
│   │   │   ├── cli.py                 # Interactive CLI
│   │   │   ├── run_all_experiments.py # Batch experiments
│   │   │   └── plot_results.py        # Visualization scripts
│   │   ├── data/                      # Datasets
│   │   │   ├── email-Eu-core.txt      # SNAP network
│   │   │   ├── road_traffic_network.csv
│   │   │   └── comm-f2f-Resistance-network/ # 62 VFOA networks
│   │   ├── plots/                     # Generated visualizations
│   │   ├── results/                   # Experimental results (CSV)
│   │   ├── report/                    # Complexity analysis
│   │   └── README.md                  # Project documentation
│   │
│   └── bonus/                         # Bonus: Image Segmentation + 3D vizualisation + Video animation
│       ├── yellow.ipynb               # Jupyter notebook implementation
│       ├── juku.html                  # HTML export of notebook
│       ├── my_images/                 # Input images
│       ├── output_segmentations/      # Segmentation results
│       └── video_link.txt             # Demo video link
│

```

---

## Main Project: MaxFlow Algorithms

### Implemented Algorithms

1. **Ford-Fulkerson** (DFS-based augmenting paths)
2. **Edmonds-Karp** (BFS shortest augmenting paths)
3. **Dinic's Algorithm** (Level graphs + blocking flows)
4. **Capacity Scaling** (Δ-scaling approach)
5. **Greedy Fattest Path** (Maximum capacity augmenting paths)
6. **Push-Relabel** (Preflow-push with height functions)

### Key Features

#### 1. Interactive CLI (Main Interface)
The CLI provides 7 options for comprehensive max-flow analysis:

**Options 1-6: Core Functionality**
- **Option 1**: Visualize Random Graph
  - Generate various graph types (sparse, dense, bipartite, grid)
  - Run any algorithm with flow visualization
  - See min-cut visualization
  
- **Option 2**: Visualize Real Dataset
  - Load email network (1,005 nodes)
  - Load road traffic network
  - Visualize structure and flow

- **Option 3**: Run Single Algorithm on Chosen Graph
  - Select graph type and parameters
  - Choose algorithm
  - View results with optional visualization

- **Option 4**: Run All Algorithms on Chosen Graph
  - Benchmark all 6 algorithms on same graph
  - Compare runtime and max-flow values
  - Export results to CSV

- **Option 5**: Run Benchmark Experiments
  - Size sweep (varying number of nodes)
  - Density sweep (varying edge density)
  - Bipartite matching experiments
  - Grid networks (image segmentation)
  - Capacity distribution analysis

- **Option 6**: Plot Benchmark Results
  - Runtime comparison bar charts
  - Runtime vs. size line plots
  - Runtime vs. density analysis
  - Generate publication-ready plots

**Option 7: VFOA Network Explorer (NEW)**
- Analyze 24 Visual Focus of Attention networks
- Temporal attention data aggregated into flow networks
- 4 sub-options:
  1. Visualize specific VFOA network with attention heatmap
  2. Compare multiple VFOA networks side-by-side
  3. Run max-flow on specific network
  4. **Batch analyze all 24 networks** (with progress bars)

#### 2. Benchmarking Suite
Comprehensive performance analysis across multiple dimensions:

- **Graph Properties**: Size, density, topology
- **Capacity Distributions**: Uniform, skewed, unit capacities
- **Special Cases**: Bipartite matching, grid networks, layered graphs
- **Metrics Collected**: Runtime, augmenting paths, memory usage, complexity

#### 3. Visualization Capabilities

**Network Visualizations:**
- Graph structure with capacities
- Flow on edges (thickness proportional to flow)
- Min-cut highlighting (source/sink partitions)
- Side-by-side network comparisons

**Statistical Visualizations:**
- Runtime comparison charts
- Scaling analysis plots
- VFOA attention heatmaps
- Algorithm metric comparisons

#### 4. Real Dataset Analysis

**Included Datasets:**
1. **Email Network** (email-Eu-core.txt)
   - 1,005 nodes, ~25,000 edges
   - European research institution communication

2. **Road Traffic Network** (road_traffic_network.csv)
   - 8 nodes, 11 edges with capacities
   - Traffic flow simulation

3. **VFOA Networks** (24 networks)
   - 5-8 participants per network
   - Multi-person discussion attention patterns
   - Temporal data aggregated into static networks
   - Reference: IJCAI 2019 paper by Bai et al.

---

## Bonus Part: Image Segmentation + Video animation + 3D Visualization

**Location**: `AAD_final_project/bonus/`

### Implementation

Interactive Jupyter Notebook (`yellow.ipynb`) demonstrating **foreground/background segmentation** using max-flow/min-cut algorithms.

### Features

- **Multiple Images Tested**
  - Various subjects (person, objects)
  - Different complexity levels
  - Results saved in `output_segmentations/`

### Files

- `yellow.ipynb` - Main implementation (runnable)
- `juku.html` - Static HTML export (for viewing 3D vizualisation of 2 algorithms)
- `video_link.txt` - Demo video showing interactive usage
- `my_images/` - Input images
- `output_segmentations/` - Segmentation results

---

## Installation & Setup

### Prerequisites

- Python 3.7+
- pip package manager

### Step 1: Navigate to Project

```bash
cd maxflow-project
```

### Step 2: Install Dependencies

#### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install required packages
pip install pandas numpy networkx matplotlib
```

#### Option B: System-wide Installation

```bash
pip install pandas numpy networkx matplotlib
```

### Step 3: Verify Installation

```bash
# Test basic functionality
python3 code/test_visualization.py

# Or start the CLI
python3 code/cli.py
```

---

## How to Run

### Method 1: Interactive CLI (Recommended)

```bash
cd maxflow-project
python3 code/cli.py
```

**Navigate the menu:**
- Enter `1-7` to select options
- Follow on-screen prompts
- Results saved automatically to `results/`
- Plots displayed interactively

### Method 2: Individual Scripts

#### Visualize a Random Graph
```bash
python3 code/test_visualization.py
```
Shows: Original graph → Flow graph → Min-cut graph

#### Run All Experiments
```bash
python3 code/run_all_experiments.py
```
Generates CSV files in `results/` for:
- Size sweep analysis
- Density experiments
- Bipartite graphs
- Grid networks
- Capacity distributions

#### Generate Plots
```bash
python3 code/plot_results.py
```
Creates visualizations in `plots/` directory

#### Load and Test Real Datasets
```bash
python3 code/test_dataset_load.py
```

### Method 3: Bonus - Image Segmentation

```bash
cd bonus
jupyter notebook yellow.ipynb
```

Or view the static HTML:
```bash
# Open juku.html in any web browser
```

## Features & Implementation

### Algorithm Implementations

All algorithms in `code/algos/`:

| Algorithm | File | Key Features |
|-----------|------|--------------|
| Ford-Fulkerson | `ford_fulkerson.py` | DFS-based path finding |
| Edmonds-Karp | `edmonds_karp.py` | BFS shortest paths, O(VE²) |
| Dinic | `dinic.py` | Level graphs, blocking flows |
| Capacity Scaling | `capacity_scaling.py` | Δ-scaling optimization |
| Greedy Fattest | `greedy_fattest.py` | Maximum capacity paths |
| Push-Relabel | `push_relabel.py` | Preflow-push with heights |

### Benchmarking Framework

**Location**: `code/benchmark/`

**Components:**
- `runner.py` - Execute algorithms with timing
- `metrics.py` - Collect performance data
- `plots.py` - Generate visualizations
- `aggregate_complexity.py` - Empirical complexity analysis

**Experiments Available:**
1. Size scaling (10 to 200 nodes)
2. Density sweep (0.1 to 0.9)
3. Bipartite matching (varying sizes)
4. Grid networks (image segmentation simulation)
5. Capacity distributions (uniform, skewed, unit)
6. Layered graphs (Dinic optimization)
7. Long-path worst-case (adversarial)

### Utilities

**Location**: `code/utils/`

- `graph.py` - FlowGraph data structure
- `generators.py` - Random graph generation
- `datasets.py` - Load real network data
- `network_datasets.py` - VFOA network loader with temporal aggregation
- `network_analysis.py` - Comparison and statistics
- `visualize.py` - Network and flow visualization
- `mincut.py` - Min-cut extraction

### VFOA Network Analysis

**Key Innovation**: Temporal Aggregation

VFOA networks contain timestamped attention data:
- **Input**: T timesteps × N participants × N targets
- **Process**: Average attention probabilities over time
- **Output**: N×N capacity matrix (scaled to integers)
- **Result**: Static flow network for max-flow analysis

**Formula**: `capacity[i][j] = round(mean(attention[t][i][j]) × 100)`

**Use Cases:**
- Analyze information flow in group discussions
- Identify key participants (high flow)
- Compare communication patterns across groups
- Batch analysis of multiple discussion sessions

---

## Documentation

Comprehensive documentation available in `maxflow-project/`:

1. **README.md** - Quick start and basic usage
5. **report/complexity_summary.md** - Algorithm complexity analysis

### Key Documents Highlights

#### Complexity Analysis (`report/complexity_summary.md`)

Detailed comparison of:
- Time complexity (theoretical and empirical)
- Space complexity
- Practical performance notes

#### VFOA Guide (`NETWORK_DATASETS_GUIDE.md`)

- Dataset format explanation
- Visualization types
- Batch processing instructions

---

---

##  Expected Outputs

### Results Directory (`results/`)

CSV files with benchmarking data:
- `size_sweep_*.csv` - Scaling with graph size
- `density_experiment.csv` - Density analysis
- `bipartite_experiment.csv` - Matching problems
- `vfoa_batch_*.csv` - VFOA network analysis
- Custom experiment results

### Plots Directory (`plots/`)

Visualizations:
- Runtime comparison bar charts
- Scaling analysis line plots
- Algorithm-specific metrics
- VFOA attention heatmaps
- Network structure visualizations

### Sample Output Structure

```
results/
├── size_sweep_dense.csv
├── density_experiment.csv
├── vfoa_batch_dinic_20251202_143055.csv
└── ...

plots/
├── (A)_runtime_vs_n.png
├── (B)_runtime_vs_m_n200.png
├── (L)_memory_usage.png
└── ...
```

---

## Project Highlights

### What Makes This Implementation Special

1. **Comprehensive**: 6 algorithms with full implementations
2. **Interactive**: User-friendly CLI for exploration
3. **Benchmarked**: Extensive performance analysis
4. **Visual**: Rich visualizations of graphs, flows, and min-cuts
5. **Real Data**: Analysis of actual network datasets
6. **Innovative**: VFOA network analysis with temporal aggregation
7. **Documented**: Extensive guides and complexity analysis
8. **Practical**: Bonus image segmentation application

### Algorithms Comparison Summary

| Algorithm | Best For | Time Complexity |
|-----------|----------|-----------------|
| Ford-Fulkerson | Small graphs, simple impl. | O(E·f) |
| Edmonds-Karp | Predictable performance | O(VE²) |
| Dinic | General purpose, fast | O(V²E) |
| Capacity Scaling | Wide capacity ranges | O(E²log C) |
| Greedy Fattest | Skewed capacities | O(E log V·aug) |
| Push-Relabel | Best practical performance | O(V³) |

---


---

## Credits

### VFOA Dataset

- **Authors**: C. Bai, S. Kumar, J. Leskovec, M. Metzger, J.F. Nunamaker, V.S. Subrahmanian
- **Paper**: "Predicting Visual Focus of Attention in Multi-person Discussion Videos"
- **Conference**: International Joint Conference on Artificial Intelligence (IJCAI), 2019
- **Dataset**: http://snap.stanford.edu/data/comm-f2f-Resistance.html

### Other Datasets

- **Email Network**: SNAP Stanford Network Analysis Project
- **Graph Algorithms**: Based on standard textbook implementations (Cormen et al., Kleinberg & Tardos)

---

---

## Academic Context

This project was developed as part of an **Algorithms Analysis and Design** course, demonstrating:

- Algorithm design and analysis
- Performance benchmarking and empirical validation
- Real-world application (image segmentation)
- Network flow theory
- Software engineering best practices

**Course**: Advanced Algorithms and Data Structures  
**Topic**: Maximum Flow Algorithms  
**Team**: Error404

---

## License & Usage

This project is submitted as academic coursework. Code implementations are based on standard algorithms from academic literature and textbooks.

---

## Quick Command Reference

```bash
# Main CLI
cd maxflow-project && python3 code/cli.py

# Run experiments
python3 code/run_all_experiments.py

# Generate plots
python3 code/plot_results.py

# Test visualization
python3 code/test_visualization.py

# Bonus segmentation
cd bonus && jupyter notebook yellow.ipynb
```

---

**Thank you for reviewing our project! 🎉**

For live demonstration, simply run `python3 code/cli.py` and explore the options.
The code is ready to execute and will produce results in real-time.
