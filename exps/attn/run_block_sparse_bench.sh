export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=. python run_block_sparse_benchmark.py

PYTHONPATH=. python run_var_block_sparse_benchmark.py
