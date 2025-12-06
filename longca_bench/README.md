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
cd longca_bench/dist_attn

export PYTHONPATH="${PYTHONPATH}:/path/to/LongCA-Bench/"

bash run_benchmark.sh
```

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
