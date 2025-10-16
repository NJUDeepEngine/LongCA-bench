import json
import os
import re
from datetime import datetime
from enum import Enum

import pandas as pd
import torch
import torch.distributed as dist

from exps.attn.baselines.utils import calculate_attn_flops
from exps.dist_attn.baselines.loongtrain import LoongTrain
from exps.dist_attn.baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    get_loongtrain_pg,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from exps.dist_attn.baselines.ulysess import Ulysess
from exps.dist_attn.baselines.usp import USP
from exps.dist_attn.baselines.utils_cp import AttnBackend
from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskIterator
from exps.utils.benchmark import Benchmark, do_bench_flops, perf_report
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType


class AttnImpl(Enum):
    ULYSSESS = "ulysses"
    RING_P2P = "ring_p2p"
    RING_ALLGATHER = "ring_allgather"
    USP = "usp"
    LOONGTRAIN = "loongtrain"


# attention params
SEED = 42
TOTAL_SEQLEN = 8 * 1024 * int(os.environ["WORLD_SIZE"])
Q_HEADS = 64
KV_HEADS = 8
EMBED_DIM = 1024
HIDDEN_SIZE = 128
DTYPE = torch.bfloat16
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
ATTN_BACKEND = AttnBackend.FA3

# mask params
MASK_NUMS = 30
ITERATION = 15
WARMUP = 5

# Optional baseline params (except magi)
DROPOUT = 0.0
SOFTMAX_SCALE = None
DETERMINISTIC = False
CP_PG_META = {
    # ParallelMode.RING: 1,
    # ParallelMode.ULYSESS: 1,
    ParallelMode.RING: 4,
}


quantiles = [0.5, 0.2, 0.8]
already_known_oom_before_run = False


def load_distributed_with_seqlen(seqlen: int):
    if seqlen <= 12 * 1024:
        file_path = "distribution/pile.json"
    elif seqlen < 64 * 1024:
        file_path = "distribution/prolong64k.json"
    elif seqlen <= 1024 * 1024:
        file_path = "distribution/prolong512k.json"
    else:
        raise ValueError("No seqlen distribution can be used!")

    with open(file_path, "r") as f:
        data = json.load(f)

    distribution = {}
    for k, v in data.items():
        nums = tuple(map(int, re.findall(r"\d+", k)))
        distribution[nums] = v

    return distribution


def init_dist_environment(
    attn_impl: AttnImpl,
    world_size: int,
    cp_pg_meta,
):
    rank = int(os.environ.get("RANK", 0))

    # -----    test ring or all-gather   ---- #
    if attn_impl == AttnImpl.RING_ALLGATHER or attn_impl == AttnImpl.RING_P2P:
        # cp_pg_meta = {
        #     ParallelMode.RING: 4,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ring_pg(device_shard)

    # -----    test ulysess   ---- #
    elif attn_impl == AttnImpl.ULYSSESS:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 4,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ulysess_pg(device_shard)

    # -----    test usp   ---- #
    elif attn_impl == AttnImpl.USP:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 2,
        #     ParallelMode.RING: 2,
        # }
        # ulysess [0,1] or ring [0,1]
        # cp_pg_meta = {
        #     ParallelMode.RING: 2,
        #     ParallelMode.ULYSESS: 2,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_usp_pg(device_shard)
    elif attn_impl == AttnImpl.LOONGTRAIN:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 1,
        #     ParallelMode.RING: 4,
        # }
        # cp_pg_meta = {
        #     ParallelMode.RING: 4,
        #     ParallelMode.ULYSESS: 1,
        # }
        # world_size = 4
        # NOTE: param for loongtrain double ring-attention
        ring_num = cp_pg_meta[ParallelMode.RING]
        for i in range(1, ring_num + 1):
            if ring_num % i != 0:
                continue
            if i > ring_num // i:
                break
            window_num = i
        # assert world_size % window_num == 0
        device_shard = init_distributed(world_size=world_size, pg_meta=None)
        cp_group = get_loongtrain_pg(cp_pg_meta, window_num, rank)

    return cp_group


def run_dist_attn(
    seed: int,
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    dropout: float,
    softmax_scale: float,
    deterministic: bool,
    world_size: int,
    attn_mask_type: AttnMaskType,
    cp_pg_meta,
    attn_impl: AttnImpl,
    attn_backend: AttnBackend,
    cp_group,
    iteration: int,
    wd: str,
):
    device = torch.cuda.current_device()

    # -----    init attn module   ---- #

    if attn_impl == AttnImpl.RING_ALLGATHER:
        attn = RingAttnAllGather(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.RING_P2P:
        attn = RingAttnP2P(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.ULYSSESS:
        attn = Ulysess(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [device]
    elif attn_impl == AttnImpl.USP:
        attn = USP(cp_process_group=cp_group, qkv_format="thd", backend=attn_backend)  # type: ignore[assignment]
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.LOONGTRAIN:
        attn = LoongTrain(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]

    # -----    init test data   ---- #

    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)

    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    x.requires_grad_(True)

    # -----    dispatch   ---- #

    # HACK dispatch only support (t, h, d)
    # assert embed_dim % 2 == 0, "only support (t, h, d) in dispatch, so embed_dim must divided 2 in zero"
    x = x.view(total_seqlen, 1, embed_dim)

    x_local = attn.dispatch(x, q_ranges, total_seqlen, "q")
    _ = attn.dispatch(x, k_ranges, total_seqlen, "k")
    _ = attn.dispatch(x, k_ranges, total_seqlen, "v")
    _ = attn.dispatch(x, q_ranges, total_seqlen, "dout")

    x_local = x_local.view(-1, embed_dim)

    # -----   projection ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

    if attn_impl == AttnImpl.ULYSSESS:
        assert world_size % kv_heads == 0 or kv_heads % world_size == 0
        H = world_size // kv_heads
        if H > 1:
            k_local = torch.repeat_interleave(k_local, H, dim=1)
            v_local = torch.repeat_interleave(v_local, H, dim=1)

    # -----   pre_compute ---- #

    attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

    def fn():
        return attn.apply_attn(
            q_local,
            k_local,
            v_local,
            attn_mask_type,
            dropout,
            softmax_scale,
            deterministic,
        )

    if wd == "bwd":
        try:
            out, lse = fn()
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                print(
                    f"Error occured before running {attn_impl} with {attn_mask_type} mask "
                    f"when {total_seqlen=}, {q_heads=} during {wd}: {e=}"
                )
                raise e
            global already_known_oom_before_run
            already_known_oom_before_run = True

        def fn():
            out.backward(dout_local, retain_graph=True)

    elif wd == "fwd+bwd":

        def fn():
            out, lse = attn.apply_attn(
                q_local,
                k_local,
                v_local,
                attn_mask_type,
                dropout,
                softmax_scale,
                deterministic,
            )
            out.backward(dout_local, retain_graph=True)

    return fn


flash_mask_list = [
    FlashMaskType.FULL,
    FlashMaskType.CAUSAL,
    FlashMaskType.FULL_DOCUMENT_LONG,
    FlashMaskType.CAUSAL_DOCUMENT_LONG,
]

attn_impl_list = [
    AttnImpl.ULYSSESS,
    AttnImpl.RING_P2P,
    AttnImpl.RING_ALLGATHER,
    AttnImpl.USP,
    AttnImpl.LOONGTRAIN,
]

cp_pg_mata_list = [
    {
        ParallelMode.ULYSESS: WORLD_SIZE,
    },
    {
        ParallelMode.RING: WORLD_SIZE,
    },
    {
        ParallelMode.RING: WORLD_SIZE,
    },
    {
        ParallelMode.ULYSESS: min(WORLD_SIZE, 8),
        ParallelMode.RING: max(1, WORLD_SIZE // 8),
    },
    {
        ParallelMode.RING: max(1, WORLD_SIZE // 8),
        ParallelMode.ULYSESS: min(WORLD_SIZE, 8),
    },
    {},
]

wd = ["fwd", "bwd", "fwd+bwd"]

attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],  # Argument names to use as an x-axis for the plot.
        x_vals=[TOTAL_SEQLEN],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="attn_impl",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            attn_impl_list[int(os.environ["BASELINE_INDEX"])]
        ],  # Possible values for `line_arg`.
        line_names=[
            attn_impl_list[int(os.environ["BASELINE_INDEX"])].value
        ],  # Label name for the lines.
        styles=[  # Line styles.
            ("green", "--"),
            ("orange", "--"),
            ("steelblue", "--"),
            ("red", "-"),
        ],
        ylabel={  # Label name for the y-axis.
            "flops": "Throughout (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        # Name for the plot. Used also as a file name for saving the plot.
        plot_name=f"{attn_impl_list[int(os.environ['BASELINE_INDEX'])].value} with full mask",
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "mask_nums": MASK_NUMS,
            "mask_type": flash_mask_list[int(os.environ["MASKTYPE_INDEX"])],
            "seed": SEED,
            "cp_pg_meta": cp_pg_mata_list[int(os.environ["BASELINE_INDEX"])],
            "wd": wd[int(os.environ["WD_INDEX"])],
        },
    ),
]


@perf_report(attn_flops_configs)
def run_benchmark(
    mask_nums: int,
    mask_type: FlashMaskType,
    seed: int = 42,
    seqlen: int = 0,
    attn_impl: AttnImpl = AttnImpl.RING_P2P,
    cp_pg_meta: dict = {},
    wd: str = "fwd",
):
    set_seed(seed)
    distribution = load_distributed_with_seqlen(seqlen=seqlen)
    mask_iterator = MaskIterator(
        generate_times=mask_nums,
        generate_mask=mask_type,
        total_seqlen=TOTAL_SEQLEN,
        distribution=distribution,
        to_attn_ranges=True,
        seed=seed,
    )
    cp_group = init_dist_environment(
        attn_impl=attn_impl,
        world_size=WORLD_SIZE,
        cp_pg_meta=cp_pg_meta,
    )

    perf_dict_total = {
        "flops": [0, 0, 0],
        "mem": [0, 0, 0],
    }

    for q_ranges, k_ranges, attn_mask_type, _ in mask_iterator:
        global already_known_oom_before_run
        already_known_oom_before_run = False

        fn = run_dist_attn(
            seed=seed,
            total_seqlen=seqlen,
            embed_dim=EMBED_DIM,
            q_heads=Q_HEADS,
            kv_heads=KV_HEADS,
            hidden_size=HIDDEN_SIZE,
            dtype=DTYPE,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            dropout=DROPOUT,
            softmax_scale=SOFTMAX_SCALE,  # type: ignore
            deterministic=DETERMINISTIC,
            world_size=WORLD_SIZE,
            attn_mask_type=attn_mask_type[0],
            cp_pg_meta=cp_pg_meta,
            attn_impl=attn_impl,
            attn_backend=ATTN_BACKEND,
            cp_group=cp_group,
            iteration=ITERATION,
            wd=wd,
        )

        if already_known_oom_before_run:
            perf_dict = {
                "flops": [-1000, -1000, -1000],
                "mem": [-1000, -1000, -1000],
            }
            perf_dict_total = {
                "flops": [-1000, -1000, -1000],
                "mem": [-1000, -1000, -1000],
            }
            break

        attn_flops_dict = calculate_attn_flops(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=seqlen,
            num_heads_q=Q_HEADS,
            head_dim=HIDDEN_SIZE,
        )
        if wd == "fwd+bwd":
            attn_flops = attn_flops_dict["fwd"] + attn_flops_dict["bwd"]
        else:
            attn_flops = attn_flops_dict[wd]

        try:
            # disable mem test to only test flops for now
            perf_dict = do_bench_flops(
                fn,
                quantiles=quantiles,
                mem_record_mode="peak",
                warmup=WARMUP,
                rep=ITERATION,
                is_distributed=True,
            )

            data = perf_dict["flops"]
            data = torch.tensor(
                data, dtype=torch.float32, device=torch.cuda.current_device()
            )
            dist.all_reduce(data, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            mem = perf_dict["mem"]
            mem = torch.tensor(
                mem, dtype=torch.float32, device=torch.cuda.current_device()
            )
            dist.all_reduce(mem, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            perf_dict["flops"] = data.tolist()  # type: ignore
            perf_dict["mem"] = [m // WORLD_SIZE for m in mem.tolist()]  # type: ignore

            # --------- process report --------- #

            # post process the perf_dict
            def ms_to_tflops(ms: float) -> float:
                return attn_flops / ms * 1e-9

            perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))  # type: ignore
            perf_dict_total["flops"] = [
                perf_dict_total["flops"][i] + perf_dict["flops"][i]
                for i in range(len(perf_dict_total["flops"]))
            ]
            perf_dict_total["mem"] = [
                perf_dict_total["mem"][i] + perf_dict["mem"][i]
                for i in range(len(perf_dict_total["mem"]))
            ]

            # disable mem test
            # def gb(m):
            #     return m / 1024**3

            # perf_dict["mem"] = list(map(gb, perf_dict["mem"]))
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                raise e
            # -1 indicates oom
            perf_dict = {
                "flops": [-1000, -1000, -1000],
                "mem": [-1000, -1000, -1000],
            }
            perf_dict_total = {
                "flops": [-1000, -1000, -1000],
                "mem": [-1000, -1000, -1000],
            }
            break

    perf_dict_total["flops"] = [
        metric / mask_nums for metric in perf_dict_total["flops"]  # type: ignore
    ]
    perf_dict_total["mem"] = [metric / mask_nums for metric in perf_dict_total["mem"]]  # type: ignore
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        new_row = [
            {
                "baseline": attn_impl.value,
                "masktype": mask_type.value,
                "world_size": WORLD_SIZE,
                "wd": wd,
                "ulysses": cp_pg_meta.get(ParallelMode.ULYSESS, -1),
                "ring": cp_pg_meta.get(ParallelMode.RING, -1),
                "mean_tflops": perf_dict_total["flops"][0],
                "mean_peak_mems": perf_dict_total["mem"][0],
            }
        ]

        df_new = pd.DataFrame(new_row)
        output_file = (
            "output/output-" + str(WORLD_SIZE) + "-" + str(mask_type.value) + ".csv"
        )

        if not os.path.exists(output_file):
            df_new.to_csv(output_file, index=False, header=True)
        else:
            df_new.to_csv(output_file, mode="a", index=False, header=False)

    return perf_dict_total


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_{current_time}")
    )

    # torch.cuda.memory._record_memory_history(
    #     max_entries=100000
    # )

    # run_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
    run_benchmark.run(print_data=True, print_value_on_bar=False)
