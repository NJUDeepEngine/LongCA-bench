# Benchmark Experiments for LongCA-Bench


## Kernel-Level Attention Performance and Flexibility

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd longca_bench/attn

bash run_benchmark.sh
```


## Module-Level Distributed Attention Performance and Scalability

basic running command:

```bash
cd exps/dist_attn

export PYTHONPATH="${PYTHONPATH}:/path/to/LongCA-Bench/"

bash run_benchmark.sh --config config_file --profile output_name
```

bench with custom config file:

```bash
cd exps/dist_attn

# 1. Use the default config file (`exps/dist_attn/benchmark_conf.py`)
bash run_benchmark.sh

# 2. Use the specific config file
bash run_benchmark.sh --config config_file

# 3. Use the specific config file
bash run_benchmark.sh --config=config_file
```

bench with nsys profiler command:

```bash
cd exps/dist_attn

# 1. Enable profiling with default output name (cp_benchmark)
bash run_benchmark.sh --profile

# 2. Enable profiling and specify an output name
bash run_benchmark.sh --profile output_name

# 3. Equivalent syntax using '='
bash run_benchmark.sh --profile=output_name

# 4. Disable profiling by default
bash run_benchmark.sh
```

When benchmarking with profiling, user can set env vars `PROFILE_ITER` and `PROFILE_WARMUP` to additionally control the number of iterations and warmups.

custom bench configuration:

The default configuration file `longca_bench/dist_attn/benchmark_conf.py` defines all necessary params for the benchmark, making it easy to adapt the setup to different environments or experiment settings, including:

- SEED
- BENCH_CONFIG (how to bench):
    - bench metrics config (see `longca_bench/utils/benchmark.py` for details):
        - quantiles: quantile points to report results.
        - bench_flops / bench_mem: Whether to evaluate FLOPs or memory.
        - bench_mode: statistic mode (mean, median, min, max).
        - iteration / warmup: number of iterations and warmups for each run.
        - output_path: directory to save bench results.
    - dist_attn_impl: all distributed attn to evaluate, as x-vals.
    - bench sweep config:
        - mask_pattern: all mask patterns to evaluate。 Options: [full, causal, varlen-full, varlen-causal]
        - workload: all pipeline modes to evaluate. Options: [fwd, bwd, 1f1b]
- SAMPLE_CONFIG:
    - defines how to sample datasets to simulate real training scenarios, see `benchmark_conf.py` for details.
- DATA_CONFIG:
    - defines how to generate data to run the bench, see `benchmark_conf.py` for details.
- ATTN_CONFIG:
    - defines how to configure the attention mechanisms, see `benchmark_conf.py` for details.
