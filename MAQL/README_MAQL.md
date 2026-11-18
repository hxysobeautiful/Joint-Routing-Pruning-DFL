# MAQL Routing and Pruning Experiments

This launcher runs the proposed multi-agent Q-learning (MAQL) algorithm for multiple random-seed settings, supporting the experiments in:

**Joint Routing and Pruning in Decentralized FL: Non-Cooperative Game and Distributed Multi-Agent Reinforcement Learning**

The launcher executes `Proposed.py` sequentially for each selected seed. Each run starts in a separate Python process and uses its corresponding input Q-table.

## Requirements

- Python with NumPy installed.
- `Proposed.py` and its dependencies.
- An input `Q_table.npy` for each selected seed.

The launcher validates that each Q-table is a numeric array with shape:

```python
(10, 10, 10, 10)
```

Additional dependencies and training settings are determined by `Proposed.py`.

## Input and Output Directories

By default, the results root is the `proposed/` directory alongside the launcher.

The current implementation constructs each experiment directory using:

```python
results_root / f"job_1{seed}"
```

The directory mapping is therefore:

| Random seed | Experiment directory |
| --- | --- |
| 42 | `proposed/job_142/` |
| 48 | `proposed/job_148/` |
| 49 | `proposed/job_149/` |
| 50 | `proposed/job_150/` |
| 51 | `proposed/job_151/` |

Place each input Q-table directly inside its experiment directory:

```text
proposed/job_142/Q_table.npy
proposed/job_148/Q_table.npy
proposed/job_149/Q_table.npy
proposed/job_150/Q_table.npy
proposed/job_151/Q_table.npy
```

Method outputs are expected in a separate subdirectory. For example, seed 42 uses:

```text
proposed/job_142/proposed/reward.txt
proposed/job_142/proposed/r_m.txt
proposed/job_142/proposed/one_hop_error
proposed/job_142/proposed/pathbest.npy
```

## Quick Start

Run the following commands from the directory containing the launcher and `Proposed.py`.

### Validate Inputs

```bash
python run_maql.py --dry-run
```

This checks the required Q-tables, verifies algorithm-script syntax, and lists the planned jobs without starting training or creating output directories.

### Run All Configured Seeds

```bash
python run_maql.py
```

The default seed order is:

```text
42, 48, 49, 50, 51
```

Only the `proposed` method is currently enabled, so the default configuration launches five sequential MAQL runs.

### Select Seeds

Run a single seed:

```bash
python run_maql.py --seeds 42
```

Run a subset of seeds:

```bash
python run_maql.py --seeds 42 49 51
```

### Display Training Progress

```bash
python run_maql.py --seeds 42 --show-output
```

By default, the launcher suppresses the algorithm's standard output. Use `--show-output` to display the original per-episode and per-hop messages.

### Specify the Results Root

```bash
python run_maql.py --results-root "/path/to/proposed"
```

The specified directory must contain the corresponding `job_1{seed}/Q_table.npy` files.

### Stop After the First Failure

```bash
python run_maql.py --stop-on-error
```

Otherwise, the launcher records failures and continues with the remaining jobs.

### Select a Python Interpreter

```bash
python run_maql.py --python "/path/to/python"
```

## Generated Files

Each successful run is expected to produce:

| File | Content |
| --- | --- |
| `reward.txt` | Reward results written by the algorithm |
| `r_m.txt` | Model-retention ratios |
| `one_hop_error` | Link-reliability data |
| `pathbest.npy` | Selected routing paths |

A job is reported as successful when the algorithm exits with return code zero and all four expected files exist.

## Algorithm Configuration

The current method configuration is:

```python
METHODS = {
    "proposed": "Proposed.py",
}
```

Other methods are commented out and are not executed. To enable them, restore the corresponding entries in `METHODS` and ensure that their scripts are available and use the expected directory convention.

The launcher does not override the number of training episodes. The intended 50,000-episode setting must be configured in the underlying algorithm script.

## Random Seeds and Q-Table Initialization

For each subprocess, the launcher sets:

```text
RL_SEED
RL_RESULTS_ROOT
PYTHONIOENCODING=utf-8
```

`Proposed.py` must read these variables to select the requested seed, input Q-table, and output location.

The launcher checks that each seed has its own input Q-table; it does not generate or copy Q-tables between seeds. Whether training starts from an initialized or previously trained table depends on the supplied file and the loading logic in `Proposed.py`.

Reproducible random-number initialization must also be implemented in the algorithm script. Setting `RL_SEED` alone does not initialize Python or NumPy random-number generators.

## Repeated Runs

In the current launcher, `--overwrite` defaults to `True`. Existing outputs therefore do not prevent a job from running again, and the underlying algorithm may overwrite them.

To preserve previous results, use a different `--results-root` and prepare the required Q-tables there before execution.

## Execution Summary

After execution, the launcher reports the numbers of completed, skipped, and failed or blocked jobs. It returns a nonzero exit status if any job fails or is blocked.