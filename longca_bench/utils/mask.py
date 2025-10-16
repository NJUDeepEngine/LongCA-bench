import torch

from magi_attention.common.enum import AttnMaskType
from magi_attention.common import AttnRange, AttnRanges
from magi_attention.meta.container import AttnSlice, AttnBucket, AttnChunk


def make_ffa_causal_mask(
    seqlen_q: int,
    seqlen_k: int,
    attn_type_idx: int = 0,
    device: str | int = "cuda",
) -> torch.Tensor:
    max_seqlen = max(seqlen_q, seqlen_k)
    latend_square_full_mask = torch.ones(
        (max_seqlen, max_seqlen),
        dtype=torch.bool,
        device=device,
    )

    match attn_type_idx:
        case 0:  # full
            mask = latend_square_full_mask[:seqlen_q, :seqlen_k]
        case 1:  # causal with bottom-right aligned
            mask = torch.tril(latend_square_full_mask)[-seqlen_q:, -seqlen_k:]
        case 2:  # inv-causal with top-left aligned
            mask = torch.triu(latend_square_full_mask)[:seqlen_q, :seqlen_k]
        case 3:  # bi-causal with bottom-right and top-left (bi-directional) aligned
            mask = (
                torch.tril(latend_square_full_mask)[-seqlen_q:, -seqlen_k:]
                & torch.triu(latend_square_full_mask)[:seqlen_q, :seqlen_k]
            )
        case _:
            raise ValueError(f"Invalid {attn_type_idx=}")

    return mask


def get_attn_mask_from_ffa_args(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_type_map: list[int],
    total_seqlen_q: int,
    total_seqlen_k: int,
    device: str | int = "cuda",
) -> torch.Tensor:
    mask = torch.zeros(
        (total_seqlen_q, total_seqlen_k),
        dtype=torch.bool,
        device=device,
    )

    for q_range, k_range, attn_type_idx in zip(q_ranges, k_ranges, attn_type_map):
        slice_mask = make_ffa_causal_mask(
            seqlen_q=q_range.seqlen,
            seqlen_k=k_range.seqlen,
            attn_type_idx=attn_type_idx,
            device=device,
        )

        mask[
            q_range.start : q_range.end,
            k_range.start : k_range.end,
        ] = slice_mask

    return mask


def _calc_self_attn_areas(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    num_chunks: int,
    chunk_size: int,
) -> AttnBucket:
    """Compute the self-attn areas, with constructing the global bucket,
    which is mainly consists of a list of all the chunks in ascending order, with a length of `cp_size`

    Args:
        q_ranges (AttnRanges): the query ranges
        k_ranges (AttnRanges): the key ranges
        attn_mask_type (list[AttnMaskType]): the attn mask type list
        chunk_size (int | None): the chunk size, which should be divisible by `cp_size`

    Returns:
        global_bucket(AttnBucket): the global bucket
    """

    # -----------    init meta info and global bucket    ----------- #

    global_bucket = AttnBucket()
    range_idx = 0
    n = len(q_ranges)

    # -----------    compute attn areas for self-attn settings    ----------- #

    for chunk_id in range(num_chunks):  # for each chunk
        chunk: AttnChunk = AttnChunk(chunk_id=chunk_id)

        # calculate begin and end of current chunk
        chunk_begin = chunk_id * chunk_size
        chunk_end = (chunk_id + 1) * chunk_size

        # find the first range that intersect with current chunk
        while (
            range_idx < n
            and q_ranges[range_idx].start < chunk_begin
            and q_ranges[range_idx].end <= chunk_begin
        ):
            range_idx += 1

        slice_id = 0
        cur_range_idx = range_idx
        # Iterate from the current range until the start of the range exceeds the current chunk.
        while cur_range_idx < n and q_ranges[cur_range_idx].start < chunk_end:
            mask_type = attn_mask_type[cur_range_idx]
            slice: AttnSlice = AttnSlice(slice_id=slice_id, mask_type=mask_type)

            attn_len = k_ranges[cur_range_idx].seqlen
            attn_q_start, attn_q_end = (
                q_ranges[cur_range_idx].start,
                q_ranges[cur_range_idx].end,
            )
            attn_k_start, attn_k_end = (
                k_ranges[cur_range_idx].start,
                k_ranges[cur_range_idx].end,
            )

            # If the current range has no intersection with the chunk,
            # and the range's start is beyond the end of the chunk, skip it directly.
            if attn_q_start < chunk_begin and attn_q_end <= chunk_begin:
                cur_range_idx += 1
                continue

            q_range_start, q_range_end, k_range_start, k_range_end = (
                None,
                None,
                None,
                None,
            )

            if mask_type == AttnMaskType.CAUSAL:
                q_range_start = max(attn_q_start, chunk_begin, attn_q_end - attn_len)
                q_range_end = min(attn_q_end, chunk_end)
                if q_range_start < q_range_end:
                    # the area of a triangle or a trapezoid
                    diff_slice_end_and_q_end = attn_q_end - q_range_end
                    (k_range_start, k_range_end) = (
                        attn_k_start,
                        attn_k_end - diff_slice_end_and_q_end,
                    )

                    # calculate the base and height of the trapezoid
                    base_of_causal = k_range_end - k_range_start
                    height_of_causal = q_range_end - q_range_start
                    slice.area = (
                        (2 * base_of_causal - height_of_causal + 1)
                        * height_of_causal
                        // 2
                    )
                else:
                    # empty slice
                    (q_range_start, q_range_end) = (q_range_start, q_range_start)
                    (k_range_start, k_range_end) = (attn_k_start, attn_k_start)
                    slice.area = 0
            elif mask_type == AttnMaskType.INVCAUSAL:
                q_range_start = max(attn_q_start, chunk_begin)
                q_range_end = min(attn_q_end, chunk_end, attn_q_start + attn_len)
                if q_range_start < q_range_end:
                    # the area of a triangle or a trapezoid
                    diff_slice_start_and_q_start = q_range_start - attn_q_start
                    (k_range_start, k_range_end) = (
                        attn_k_start + diff_slice_start_and_q_start,
                        attn_k_end,
                    )

                    # calculate the base and height of the trapezoid
                    base_of_causal = k_range_end - k_range_start
                    height_of_causal = q_range_end - q_range_start
                    slice.area = (
                        (2 * base_of_causal - height_of_causal + 1)
                        * height_of_causal
                        // 2
                    )
                else:
                    # empty slice
                    (q_range_start, q_range_end) = (q_range_start, q_range_start)
                    (k_range_start, k_range_end) = (attn_k_start, attn_k_start)
                    slice.area = 0
            elif mask_type == AttnMaskType.BICAUSAL:
                q_range_start = max(attn_q_start, chunk_begin)
                q_range_end = min(attn_q_end, chunk_end)

                diff_slice_start_and_q_start = q_range_start - attn_q_start
                diff_slice_end_and_q_end = attn_q_end - q_range_end

                base_of_parallelogram = attn_len - q_ranges[cur_range_idx].seqlen + 1
                height_of_parallelogram = q_range_end - q_range_start

                if base_of_parallelogram > 0:
                    # the area of a parallelogram
                    slice.area = base_of_parallelogram * height_of_parallelogram
                    k_range_start = attn_k_start + diff_slice_start_and_q_start
                    k_range_end = attn_k_end - diff_slice_end_and_q_end
                else:
                    # empty slice
                    (q_range_start, q_range_end) = (q_range_start, q_range_start)
                    (k_range_start, k_range_end) = (attn_k_start, attn_k_start)
                    slice.area = 0
            elif mask_type == AttnMaskType.FULL:
                # the area of a rectangle
                q_range_start = max(attn_q_start, chunk_begin)
                q_range_end = min(attn_q_end, chunk_end)
                (k_range_start, k_range_end) = (attn_k_start, attn_k_end)
                slice.area = (q_range_end - q_range_start) * attn_len
            else:
                raise ValueError(
                    f"Only support 'FULL', 'CAUSAL', 'BICAUSAL', 'INVCAUSAL', "
                    f"but got {mask_type=}"
                )
            cur_range_idx += 1

            # set q_range, k_range for this slice
            slice.q_range = AttnRange(start=q_range_start, end=q_range_end)
            slice.k_range = AttnRange(start=k_range_start, end=k_range_end)

            if slice.k_range.seqlen > 0 and slice.area > 0:
                # append this q slice to the current chunk except invalid slice
                chunk.q_slices.append(slice)
                chunk.sample_ids.append(cur_range_idx - 1)
                slice_id += 1

        global_bucket.q_chunks.append(chunk)

    return global_bucket
