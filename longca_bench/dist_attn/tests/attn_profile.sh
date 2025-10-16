export WORLD_SIZE=${WORLD_SIZE:-4}
export GPUS_PER_NODE=${WORLD_SIZE}
export NNODES=1
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-16988}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS attn_profile.py"

# generate a timestamp for the nsys output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

NSYS_CMD="
nsys profile \
    --force-overwrite true \
    -o exp_ring_p2p_fa3_${TIMESTAMP}.nsys-rep \
    --capture-range=cudaProfilerApi \
    $TORCHRUN_CMD
"
$NSYS_CMD
