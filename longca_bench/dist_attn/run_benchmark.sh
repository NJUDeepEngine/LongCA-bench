export NNODES=${NNODES:-1}
export GPUS_PER_NODE=8
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
export NODE_RANK=${RANK:-0}
export MAGI_ATTENTION_HIERARCHICAL_COMM=${MAGI_ATTENTION_HIERARCHICAL_COMM:-0}

if [[ $NNODES -eq 1 ]]; then # single-node
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-16988}
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

export PYTHONPATH=../../

export CUDA_DEVICE_MAX_CONNECTIONS=8
echo "set CUDA_DEVICE_MAX_CONNECTIONS=8"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"
$TORCHRUN_CMD
