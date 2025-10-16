# export CUDA_VISIBLE_DEVICES=1,2,3,4

export GPUS_PER_NODE=8
export NNODES=${NNODES:-1}
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
echo $WORLD_SIZE
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-16988}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}


for k in {0..1}; do
    for j in {0..3}; do
        for i in {0..4}; do
            export BASELINE_INDEX=$i
            export MASKTYPE_INDEX=$j
            export WD_INDEX=$k
            export MASTER_PORT=$((MASTER_PORT + 1))

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

            wait

            python -c "import torch; torch.cuda.empty_cache()"
        done
    done
done
