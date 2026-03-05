# K Configuration Analysis Script

## Overview

`analyze_k_configurations.py` extracts performance metrics for different K configurations (K@32, K@16, K@8, K@4, K@2) from K@64 Weights & Biases (W&B) runs. This helps understand how performance scales with different numbers of parallel optimization attempts.

## What are K Configurations?

In the AIDE optimization framework:
- **K@64** = 16 parallel experiments × 4 sequential iterations = 64 total optimization attempts
- Each experiment runs independently in parallel
- Each iteration within an experiment builds on the previous iteration's best solution

This script analyzes what would happen with fewer attempts by extracting subsets:

| Configuration | Extraction Method | Total Attempts |
|--------------|-------------------|----------------|
| **K@32** | Uses iterations 1-2 from all 16 experiments | 16 × 2 = 32 |
| **K@16** | Uses iteration 1 from all 16 experiments | 16 × 1 = 16 |
| **K@8**  | Uses iteration 1 from 8 random experiments | 8 × 1 = 8 |
| **K@4**  | Uses iteration 1 from 4 random experiments | 4 × 1 = 4 |
| **K@2**  | Uses iteration 1 from 2 random experiments | 2 × 1 = 2 |

### Why This Makes Sense

- **K@32 preserves dependencies**: Each experiment's iteration 2 still sees its own iteration 1's best solution
- **K@16 and below use only iteration 1**: These are all fresh, independent optimization attempts
- **Random sampling for K@8/4/2**: Multiple trials (default 100) show expected performance and variance

## Installation

Requires Python 3.10+ with the following packages:
```bash
pip install wandb pandas numpy
```

## Basic Usage

```bash
# Analyze all tasks in a W&B project
python3.10 analyze_k_configurations.py "entity/project"

# Using a W&B URL directly
python3.10 analyze_k_configurations.py "https://wandb.ai/anthropic/kernelbench-k64"
```

## Advanced Usage

### With Task Names from Log File

If you have a `k64_study_log.txt` file that maps W&B runs to task names:

```bash
python3.10 analyze_k_configurations.py "anthropic/kernelbench" --log k64_study_log.txt
```

### Export Results to CSV

```bash
python3.10 analyze_k_configurations.py "anthropic/kernelbench" --output results.csv
```

The CSV will contain columns: `task_name`, `config`, `mean_best`, `best_case`, `p10`, `p25`, `p50`, `p75`, `p90`, `mean_speedup`, `median_speedup`, `std_dev`, `success_rate`, `num_attempts`, `note`

Note: The CSV also includes `max_speedup` for backward compatibility (same as `best_case`)

### Analyze a Specific Task

```bash
python3.10 analyze_k_configurations.py "anthropic/kernelbench" --task "BatchNorm"
```

### Change Random Sampling Parameters

```bash
# Use a different random seed (default: 42)
python3.10 analyze_k_configurations.py "anthropic/kernelbench" --seed 123

# Change number of sampling trials for K@8/4/2 (default: 100)
python3.10 analyze_k_configurations.py "anthropic/kernelbench" --trials 200
```

## Output Format

### Console Output

```
================================================================================
Task: BatchNorm (2_8)
================================================================================
Config | Mean Best  | P10     | P50     | P90     | Best Case  | Note
------ | ---------- | ------- | ------- | ------- | ---------- | ------------------------------
K@32   | 4.077x     | 3.651x  | 3.737x  | 4.057x  | 4.077x     | iter 1-2, all 15 exp
K@16   | 4.077x     | 3.307x  | 3.737x  | 3.794x  | 4.077x     | iter 1, all 15 exp
K@8    | 3.930x     | 3.753x  | 3.949x  | 4.077x  | 4.077x     | iter 1, 100 trials
K@4    | 3.835x     | 3.753x  | 3.753x  | 4.077x  | 4.077x     | iter 1, 100 trials
K@2    | 3.778x     | 3.720x  | 3.753x  | 4.077x  | 4.077x     | iter 1, 100 trials

Detailed Percentile Distribution (from max values across sampling trials):
  K@8: P10=3.753x | P25=3.753x | P50=3.949x | P75=4.077x | P90=4.077x
  K@4: P10=3.753x | P25=3.753x | P50=3.753x | P75=3.821x | P90=4.077x
  K@2: P10=3.720x | P25=3.737x | P50=3.753x | P75=3.753x | P90=4.077x

Interpretation:
  - Mean Best: Average of maximum speedups across sampling trials
  - P10: 90% of trials achieve at least this (worst case)
  - P50: 50% of trials achieve at least this (typical case)
  - P90: 10% of trials achieve at least this (good case)
  - Best Case: Best outcome across all trials (lucky scenario)
```

### Metrics Explained

- **Mean Best**: For K@8/4/2, this is the average of the maximum speedups across all sampling trials. For K@32/16, this equals the best case since there's no sampling. **This is the primary metric showing expected performance.**
- **P10**: 10th percentile - 90% of trials achieve at least this speedup (worst case scenario)
- **P50**: 50th percentile (median) - typical outcome you can expect
- **P90**: 90th percentile - 10% of trials achieve at least this (good case scenario)
- **Best Case**: The absolute best speedup achieved (what's possible if you're lucky)
- **Note**: Details about how the data was extracted

### Why "Mean Best" Shows Differences While "Best Case" Doesn't

The original version showed misleading results where all K configurations had the same "max" value. This happened because:
- With 100 sampling trials, even K@2 will eventually sample the best-performing experiment
- The best speedup typically comes from one standout experiment in iteration 1

The new **"Mean Best"** metric solves this by showing the **average** maximum speedup across trials, which properly reflects the performance scaling: K@32 ≥ K@16 > K@8 > K@4 > K@2

## Examples

### Example 1: Analyze Multiple Tasks with CSV Export

```bash
python3.10 analyze_k_configurations.py \
    "algorithmic-research-group/kernelbench-level1-k64" \
    --log k64_study_log.txt \
    --output k_analysis_results.csv
```

Output:
- Console: Formatted tables for each task
- CSV file: All results in machine-readable format

### Example 2: Quick Analysis of Recent Runs

```bash
# Analyze the most recent project without task names
python3.10 analyze_k_configurations.py \
    "algorithmic-research-group/kernelbench-latest"
```

The script will automatically group runs by creation timestamp.

### Example 3: Reproducible Analysis with Custom Parameters

```bash
python3.10 analyze_k_configurations.py \
    "algorithmic-research-group/kernelbench" \
    --seed 42 \
    --trials 200 \
    --output reproducible_results.csv
```

Using the same seed ensures identical random sampling across runs.

## Understanding the Results

### Interpreting K Configuration Performance

1. **K@32 vs K@16**: Shows the benefit of running 2 iterations vs 1 iteration
   - If K@32 >> K@16: Iterative refinement is valuable
   - If K@32 ≈ K@16: Most improvement comes from the first iteration

2. **K@16 vs K@8/4/2**: Shows how performance scales with parallelism
   - Small drop from K@16 to K@8: Robust optimization, many good solutions exist
   - Large drop from K@16 to K@2: High variance, need many attempts to find good solutions

3. **Variance in K@8/4/2**: Shown by P25/P75 percentiles
   - Narrow range: Consistent performance regardless of which experiments selected
   - Wide range: Performance heavily depends on which experiments are chosen

### Use Cases

1. **Resource Optimization**: Determine optimal K value for your compute budget
2. **Algorithm Analysis**: Understand if improvements come from parallelism or iteration
3. **Robustness Testing**: Check if good solutions are rare or common
4. **Experiment Planning**: Decide how many parallel experiments to run for future tasks

## Troubleshooting

### "No valid results found for this task"
- Check that the W&B runs have 'iteration' and 'speedup' fields in their history
- Ensure runs completed successfully (not crashed/cancelled)

### Tasks grouped incorrectly
- Use `--log` option with a proper task log file for accurate task names
- By default, runs within 1 minute are grouped as the same task

### Different number of experiments than expected
- K@64 typically has 15-16 experiments per task
- The script adapts to available data (won't attempt K@8 if only 6 experiments exist)

## Technical Details

### Random Sampling Methodology

For K@8, K@4, and K@2:
1. Extract iteration 1 max speedup from all available experiments
2. Filter to only successful experiments (speedup > 0)
3. For each trial (default 100):
   - Randomly sample K experiments without replacement
   - Calculate max speedup from the sample
4. Report statistics across all trials

### Data Extraction from W&B

The script:
1. Fetches run history from W&B API
2. Filters data by iteration number
3. Extracts speedup values where `eval_status == 'success'`
4. Takes maximum speedup per iteration per experiment

## License

This script is part of the AIDE parallel execution framework.