# K-Scaling Analysis for AIDE Parallel Execution

## Overview
This directory contains scripts and analysis for understanding how performance scales with different K configurations (K@2 through K@64) in the AIDE parallel optimization framework.

## Directory Structure
```
k_scaling_analysis/
├── data/                    # CSV data from W&B analysis
│   └── k_scaling_raw.csv   # Raw K configuration data
├── plots/                   # Generated visualizations
│   ├── main/               # Primary analysis plots
│   ├── distributions/      # Statistical distributions
│   └── heatmaps/          # Task performance heatmaps
├── reports/                # Analysis reports
│   └── analysis_report.md # Comprehensive findings
└── scripts/                # Analysis scripts
    ├── analyze_with_k64.py # Extended W&B data extraction (includes K@64)
    └── visualize_k_scaling.py # Visualization generation
```

## Scripts

### 1. analyze_with_k64.py
Extended version of the original `analyze_k_configurations.py` that adds K@64 support.
- Extracts performance data from W&B for K@2, K@4, K@8, K@16, K@32, and K@64
- Calculates `mean_best`, percentiles (P10, P25, P50, P75, P90), and best case metrics
- Handles ~111 tasks from the k64_study_log.txt

**Usage:**
```bash
python scripts/analyze_with_k64.py \
    "algorithmic-research-group/kernelbench-level1-k64" \
    --log /home/ubuntu/aide_parallel/k64_study_log.txt \
    --output data/k_scaling_raw.csv \
    --progress \
    --trials 100
```

### 2. visualize_k_scaling.py
Comprehensive visualization script that creates:

#### Main Visualizations
- **Scaling Curve**: Shows how mean_best performance improves with K value
- **Marginal Improvements**: Bar chart showing % gain for each K doubling
- **Efficiency Plot**: Performance per compute unit (mean_best/K)
- **Knee Detection**: Identifies where diminishing returns begin

#### Statistical Visualizations
- **Distribution Boxplots**: Shows variance across tasks at each K
- **Task Heatmap**: Matrix view of all tasks × K configurations
- **Summary Dashboard**: Combined view of key metrics

**Usage:**
```bash
python scripts/visualize_k_scaling.py \
    --input data/k_scaling_raw.csv \
    --output-dir plots
```

## Key Metrics

### Mean Best
The primary metric - average of maximum speedups across sampling trials:
- For K@64/32/16: Deterministic (uses all available data)
- For K@8/4/2: Average across 100 random sampling trials

### Percentiles
- **P10**: 90% of trials achieve at least this (worst case)
- **P50**: Median - typical expected outcome
- **P90**: 10% of trials achieve at least this (good case)

### Efficiency
- Speedup per compute unit (mean_best / K)
- Identifies optimal K value for cost/benefit ratio

## Expected Insights

1. **Performance Plateau**: Where does the scaling curve flatten?
2. **Optimal K**: Best efficiency point for compute budget
3. **Marginal Returns**: How much gain from each K doubling?
4. **Task Variance**: Which tasks benefit most from higher K?

## Safety Notes

- This analysis is READ-ONLY from W&B - no data is modified
- All scripts in this directory are new (not modifying originals)
- Original `analyze_k_configurations.py` remains untouched

## Status

- [x] Directory structure created
- [x] analyze_with_k64.py script created
- [x] visualize_k_scaling.py script created
- [ ] Data collection in progress (~111 tasks)
- [ ] Visualizations pending
- [ ] Final report pending