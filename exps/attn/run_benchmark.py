import json
import os
import re
from datetime import datetime

try:
    import paddle
    import paddle.nn.functional as F
except Exception:
    print("Paddle is not installed")

import torch
from baselines.attn_impl import (
    cudnn_fused_attn_func,
    fa2_func,
    fa2_varlen_func,
    fa3_func,
    fa3_varlen_func,
    flex_attn_func,
    sdpa_func,
    torch_attn_func,
)
from baselines.utils import (
    calculate_attn_flops,
    curanges2document_id,
    generate_flashmask_indices,
    make_block_causal_document_block_mask,
    make_block_causal_document_score_mod,
    make_causal_block_mask,
    make_causal_blockwise_block_mask,
    make_causal_blockwise_mask_score_mod,
    make_causal_mask_score_mod,
    make_global_sliding_window_block_mask,
    make_global_sliding_window_mask_score_mod,
    make_prefix_lm_causal_block_mask,
    make_prefix_lm_causal_mask_score_mod,
    make_prefix_lm_document_block_mask,
    make_prefix_lm_document_mask_score_mod,
    make_share_question_block_mask,
    make_share_question_mask_score_mod,
    make_sliding_window_causal_block_mask,
    make_sliding_window_causal_mask_score_mod,
    make_sliding_window_full_block_mask,
    make_sliding_window_full_mask_score_mod,
    make_varlen_causal_block_mask,
    make_varlen_causal_mask_score_mod,
    make_varlen_full_block_mask,
    make_varlen_full_mask_score_mod,
)
from einops import rearrange

from exps.dist_attn.baselines.shard import set_seed
from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskIterator
from exps.utils.benchmark import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from exps.utils.mask import get_attn_mask_from_ffa_args

impls = [
    "fa3",
    "fa2",
    "cudnn",
    "flex",
    "flash_mask",
    "torch",
    "sdpa",
]  # ignore torch native to avoid OOMzsZasZ·
# impls = ["flash_mask"]
# impls = ["flash_mask", "flex"]
# impls = ["torch"]
# impls = ["fa3", "torch", "sdpa"]


mask_types = ["full"]
# mask_types = ["causal"]
# mask_types = ["full_document"]
# mask_types = ["causal_document"]
# mask_types = ["sliding_window_causal"]
# mask_types = ["share_question"]
# mask_types = ["causal_blockwise"]
# mask_types = ["prefix_lm_causal"]
# mask_types = ["prefix_lm_document"]
# mask_types = ["global_sliding_window"]
# mask_types = ["sliding_window"]
# mask_types = ["block_causal_document"]


varlen_seqlen_distribution = {
    (0, 2 * 1024): 0.16,
    (2 * 1024, 4 * 1024): 0.05,
    (4 * 1024, 8 * 1024): 0.04,
    (8 * 1024, 16 * 1024): 0.06,
    (16 * 1024, 32 * 1024): 0.08,
    (32 * 1024, 64 * 1024): 0.21,
    (64 * 1024, 128 * 1024): 0.4,
    (128 * 1024, 256 * 1024): 0.2,
    (256 * 1024, 512 * 1024): 0.05,
    (512 * 1024, 1024 * 1024): 0.04,
    (1024 * 1024, 2048 * 1024): 0.01,
    (2048 * 1024, 4096 * 1024): 0.01,
}


# ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32, 48, 56, 64]]
# ss = [k * 1024 for k in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
# ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32]]
ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32, 40, 48]]
# ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32, 40]]
ds = [128]
wds = ["fwd", "bwd"]
# wds = ["bwd"]
# wds = ["fwd"]


b = 1
nhq = 64
nhk = 8
dtype = torch.bfloat16

window_size = 1024
block_size = 2048
num_varlen_samples = 16
seed = 42

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]
mask_nums = 1

normal_mask = [
    FlashMaskType.FULL,
    FlashMaskType.CAUSAL,
    FlashMaskType.FULL_DOCUMENT,
    FlashMaskType.CAUSAL_DOCUMENT,
    FlashMaskType.SLIDING_WINDOW_CAUSAL,
    FlashMaskType.SLIDING_WINDOW,
]


attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],  # Argument names to use as an x-axis for the plot.
        x_vals=ss,  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="attn_impl",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=impls,  # Possible values for `line_arg`.
        line_names=impls,  # Label name for the lines.
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
        plot_name=f"attn-{wd} with {mask_type} mask",  # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "hd": hd,
            "wd": wd,
            "mask_type": mask_type,
            "mask_nums": mask_nums,
        },
    )
    for hd in ds
    for wd in wds
    for mask_type in mask_types
]


def load_distributed_with_seqlen(seqlen: int):
    if seqlen <= 12 * 1024:
        file_path = "distribution/pile.json"
    elif seqlen <= 64 * 1024:
        file_path = "distribution/prolong64k.json"
    else:
        raise ValueError("No seqlen distribution can be used!")

    with open(file_path, "r") as f:
        data = json.load(f)

    distribution = {}
    for k, v in data.items():
        nums = tuple(map(int, re.findall(r"\d+", k)))
        distribution[nums] = v

    return distribution


@perf_report(attn_flops_configs)
def attn_benchmark(seqlen, hd, wd, mask_type, attn_impl, mask_nums):
    set_seed(seed)
    perf_dict_total = {
        "flops": [0, 0, 0],
        "mem": [0, 0, 0],
    }

    distribution = load_distributed_with_seqlen(seqlen=seqlen)

    flash_mask_type = FlashMaskType(mask_type)
    mask_iterator = MaskIterator(
        generate_times=mask_nums,
        generate_mask=flash_mask_type,
        total_seqlen=seqlen,
        window_size=(window_size, 0),
        distribution=distribution,
        seed=seed,
    )

    for q_ranges_, k_ranges_, attn_mask_type, mask_factors in mask_iterator:
        assert b == 1, "for now, we only supports b=1 for ffa"
        is_attn_impl_support_this_mask = True
        already_known_oom_before_run = False

        # --------- prepare arguments --------- #

        device = torch.cuda.current_device()
        sq = sk = seqlen  # fi square mask where sq == sk
        sdpa_mask = None

        # calculate attn flops
        if flash_mask_type == FlashMaskType.SLIDING_WINDOW_CAUSAL:
            causal = True
            window_size_tuple = (window_size, 0)
            max_seqlen_q = sq
            max_seqlen_k = sk
            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

        elif flash_mask_type == FlashMaskType.SLIDING_WINDOW:
            causal = False
            window_size_tuple = (window_size, window_size)
            max_seqlen_q = sq
            max_seqlen_k = sk
            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

        elif (
            flash_mask_type == FlashMaskType.FULL_DOCUMENT
            or flash_mask_type == FlashMaskType.CAUSAL_DOCUMENT
        ):
            causal = attn_mask_type[0] == AttnMaskType.CAUSAL
            cu_seqlens = q_ranges_.to_cu_seqlens(seqlen)
            cu_ranges = q_ranges_.to_naive_ranges()
            document_id = curanges2document_id(cu_ranges)

            max_seqlen_q = q_ranges_.max_seqlen
            max_seqlen_k = k_ranges_.max_seqlen
            max_seqlen_kv = max_seqlen_k

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            window_size_tuple = (-1, -1)
        elif (
            flash_mask_type == FlashMaskType.FULL
            or flash_mask_type == FlashMaskType.CAUSAL
        ):
            causal = attn_mask_type[0] == AttnMaskType.CAUSAL
            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            max_seqlen_q = sq
            max_seqlen_k = sk
            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            window_size_tuple = (-1, -1)
        else:
            # other mask logic
            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            if (
                flash_mask_type == FlashMaskType.PREFIX_LM_CAUSAL
                or flash_mask_type == FlashMaskType.PREFIX_LM_DOCUMENT
            ):
                cu_ranges = mask_factors.cu_ranges
                prefix_length = mask_factors.prefix_length
                if mask_factors.cu_seqlens is not None:
                    cu_seqlens_kv = torch.tensor(
                        mask_factors.cu_seqlens, dtype=torch.int32, device=device
                    )
                    document_id = curanges2document_id(cu_ranges)

            if (
                flash_mask_type == FlashMaskType.SHARE_QUESTION
                or flash_mask_type == FlashMaskType.CAUSAL_BLOCKWISE
            ):
                cu_ranges = mask_factors.cu_ranges
                document_id = curanges2document_id(cu_ranges)

            if flash_mask_type == FlashMaskType.GLOBAL_SLIDING_WINDOW:
                global_window_size = mask_factors.window_size

            if flash_mask_type == FlashMaskType.BLOCK_CAUSAL_DOCUMENT:
                cu_seqlens = mask_factors.cu_seqlens
                cu_ranges = mask_factors.cu_ranges
                block_size = mask_factors.block_size

            max_seqlen_q = sq
            max_seqlen_k = sk
            max_seqlen_kv = sk

        attn_flops = attn_flops_dict[wd]

        # --------- prepare data --------- #

        # flash style shape: (b,s,h,d)
        q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype, requires_grad=False)
        k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)
        v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)

        # sdpa style shape: (b,h,s,d)
        if attn_impl in ("sdpa", "torch", "flex"):
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")

            # make block mask
            if attn_impl == "flex":
                if flash_mask_type == FlashMaskType.FULL:
                    score_mod = None
                    block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL:
                    try:
                        block_mask = make_causal_block_mask(sq, sk)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_causal_mask_score_mod()
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SLIDING_WINDOW_CAUSAL:
                    try:
                        block_mask = make_sliding_window_causal_block_mask(
                            sq, sk, window_size=window_size
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_sliding_window_causal_mask_score_mod(
                            window_size=window_size
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL_DOCUMENT:
                    try:
                        block_mask = make_varlen_causal_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_varlen_causal_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.FULL_DOCUMENT:
                    try:
                        block_mask = make_varlen_full_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_varlen_full_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SHARE_QUESTION:
                    try:
                        block_mask = make_share_question_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_share_question_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL_BLOCKWISE:
                    try:
                        block_mask = make_causal_blockwise_block_mask(
                            sq, sk, document_id
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_causal_blockwise_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.PREFIX_LM_CAUSAL:
                    try:
                        block_mask = make_prefix_lm_causal_block_mask(
                            sq, sk, prefix_length
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_prefix_lm_causal_mask_score_mod(prefix_length)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.PREFIX_LM_DOCUMENT:
                    try:
                        block_mask = make_prefix_lm_document_block_mask(
                            sq, sk, prefix_length, document_id, cu_seqlens_kv
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_prefix_lm_document_mask_score_mod(
                            prefix_length, document_id, cu_seqlens_kv
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.GLOBAL_SLIDING_WINDOW:
                    try:
                        block_mask = make_global_sliding_window_block_mask(
                            sq, sk, global_window_size
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_global_sliding_window_mask_score_mod(
                            window_size=global_window_size
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SLIDING_WINDOW:
                    try:
                        block_mask = make_sliding_window_full_block_mask(
                            sq, sk, window_size
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_sliding_window_full_mask_score_mod(
                            window_size=window_size
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.BLOCK_CAUSAL_DOCUMENT:
                    document_id = curanges2document_id(cu_ranges)
                    try:
                        block_mask = make_block_causal_document_block_mask(
                            sq, sk, block_size, document_id
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_block_causal_document_score_mod(
                            block_size, document_id
                        )
                        block_mask = None
                else:
                    raise NotImplementedError(
                        f"mask type {mask_type} not supported for flex attn"
                    )
            elif (
                flash_mask_type != FlashMaskType.FULL
                and flash_mask_type != FlashMaskType.CAUSAL
            ):
                try:
                    attn_type_mapping = [
                        {
                            AttnMaskType.FULL: 0,
                            AttnMaskType.CAUSAL: 1,
                            AttnMaskType.INVCAUSAL: 2,
                            AttnMaskType.BICAUSAL: 3,
                        }[mapping]
                        for mapping in attn_mask_type
                    ]
                    sdpa_mask = get_attn_mask_from_ffa_args(
                        q_ranges=q_ranges_,
                        k_ranges=k_ranges_,
                        attn_type_map=attn_type_mapping,
                        total_seqlen_q=sq,
                        total_seqlen_k=sk,
                        device=torch.cuda.current_device(),
                    )
                    torch.cuda.memory._dump_snapshot()
                except RuntimeError as e:
                    print(f"make varlen causal sdpa mask failed: {e}")

        # ffa style shape: (t,h,d)
        if attn_impl in ("ffa", "cudnn"):
            q = q.view(b * sq, nhq, hd)
            k = k.view(b * sk, nhk, hd)
            v = v.view(b * sk, nhk, hd)

            if attn_impl == "cudnn":
                if flash_mask_type not in normal_mask:
                    is_attn_impl_support_this_mask = False

        # fa style shape:
        #   non-varlen: (b,s,h,d)
        #   varlen: (t,h,d)
        if attn_impl in ("fa2", "fa3"):
            if "document" in mask_type:
                q = q.view(b * sq, nhq, hd)
                k = k.view(b * sk, nhk, hd)
                v = v.view(b * sk, nhk, hd)

            if flash_mask_type not in normal_mask:
                is_attn_impl_support_this_mask = False

        if attn_impl in ("torch") and nhq != nhk:
            assert nhq % nhk == 0
            repeat_times = nhq // nhk
            k = torch.repeat_interleave(k, repeat_times, dim=1)
            v = torch.repeat_interleave(v, repeat_times, dim=1)

        if attn_impl in ("flash_mask"):
            q = paddle.to_tensor(
                q.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )
            k = paddle.to_tensor(
                k.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )
            v = paddle.to_tensor(
                v.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )

        # --------- prepare grads --------- #

        if wd == "bwd":
            if attn_impl not in ("flash_mask"):
                do = torch.randn_like(q)
                # require grads
                [x.requires_grad_(True) for x in [q, k, v, do]]
            else:
                do = paddle.randn_like(q)
                for x in [q, k, v, do]:
                    x.stop_gradient = False

        # --------- prepare func --------- #

        if attn_impl == "torch":

            def fn():
                return torch_attn_func(
                    q,
                    k,
                    v,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=causal if sdpa_mask is None else False,
                    scale=softmax_scale,
                    return_attn_probs=return_attn_probs,
                )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "sdpa":

            def fn():
                return sdpa_func(
                    q,
                    k,
                    v,
                    attn_mask=sdpa_mask,
                    is_causal=causal if sdpa_mask is None else False,
                    scale=softmax_scale,
                    dropout_p=dropout_p,
                    enable_gqa=True,
                )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "fa2":
            if "document" in mask_type:

                def fn():
                    return fa2_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        return_attn_probs=return_attn_probs,
                    )

            else:

                def fn():
                    return fa2_func(
                        q,
                        k,
                        v,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size_tuple,
                        return_attn_probs=return_attn_probs,
                    )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "fa3":
            if "document" in mask_type:

                def fn():
                    return fa3_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size_tuple,
                    )

            else:

                def fn():
                    return fa3_func(
                        q,
                        k,
                        v,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size_tuple,
                    )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "cudnn":

            def fn():
                return cudnn_fused_attn_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    softmax_scale=softmax_scale,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    window_size=window_size_tuple,
                    is_training=wd == "bwd",
                )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "flex":

            def fn():
                return flex_attn_func(
                    q,
                    k,
                    v,
                    scale=softmax_scale,
                    enable_gqa=True,
                    score_mod=score_mod,
                    block_mask=block_mask,
                )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "flash_mask":
            (
                attn_mask_startend_row_indices,
                flashmask_is_causal,
            ) = generate_flashmask_indices(
                sq,
                sk,
                flash_mask_type,
                window_size=mask_factors.window_size,
                cu_ranges=mask_factors.cu_ranges,
                prefix_length=mask_factors.prefix_length,
            )

            def fn():
                return F.flashmask_attention(
                    q,
                    k,
                    v,
                    startend_row_indices=attn_mask_startend_row_indices,
                    causal=flashmask_is_causal,
                )

            if wd == "bwd":
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        # --------- try do the bench --------- #

        if is_attn_impl_support_this_mask:
            if already_known_oom_before_run:
                # -1 indicates oom
                perf_dict = {
                    "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                    "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                }
                perf_dict_total = {
                    "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                    "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                }
                break
            else:
                try:
                    # disable mem test to only test flops for now
                    perf_dict = do_bench_flops(
                        fn,
                        quantiles=quantiles,
                        mem_record_mode="peak",
                        warmup=5,
                        rep=20,
                    )

                    # --------- process report --------- #

                    # post process the perf_dict
                    def ms_to_tflops(ms: float) -> float:
                        return attn_flops / ms * 1e-9

                    perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

                    # disable mem test
                    # def gb(m):
                    #     return m / 1024**3

                    # perf_dict["mem"] = list(map(gb, perf_dict["mem"]))
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured when running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    # -1 indicates oom
                    perf_dict = {
                        "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                        "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                    }
                    perf_dict_total = {
                        "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                        "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                    }
                    print(
                        f"OOM error occured when running for {attn_impl} with {mask_type} mask "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    break
        else:
            # -2 indicates not support
            perf_dict = {
                "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
            }
            perf_dict_total = {
                "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
            }
            break

        print(f"{seqlen=} {perf_dict=}")
        perf_dict_total["flops"] = [
            perf_dict_total["flops"][i] + perf_dict["flops"][i]
            for i in range(len(perf_dict_total["flops"]))
        ]
        perf_dict_total["mem"] = [
            perf_dict_total["mem"][i] + perf_dict["mem"][i]
            for i in range(len(perf_dict_total["mem"]))
        ]

    perf_dict_total["flops"] = [
        metric / mask_nums for metric in perf_dict_total["flops"]
    ]
    perf_dict_total["mem"] = [metric / mask_nums for metric in perf_dict_total["mem"]]

    return perf_dict_total


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_{current_time}")
    )

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
