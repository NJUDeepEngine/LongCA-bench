import random
from dataclasses import dataclass

from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.utils import (
    generate_seqlens,
    seqlens2cu_seqlens,
    varlen_long_seqlen_distribution,
    varlen_short_seqlen_distribution,
)
from magi_attention.api.functools import infer_attn_mask_from_sliding_window
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges


@dataclass
class MaskFactors:
    cu_seqlens: list[int] | None = None
    cu_ranges: list[list[int]] | None = None
    prefix_length: int | None = None
    window_size: int | None = None
    block_size: int | None = None


class MaskGenerator:
    """This is a generator for multiple flash masks, it can be used with
    generate function with flash mask type and data distribution.
    """

    def __init__(self):
        self.generate_function = {
            FlashMaskType.FULL: self.generate_full_mask,
            FlashMaskType.CAUSAL: self.generate_causal_mask,
            FlashMaskType.CAUSAL_DOCUMENT: self.generate_causal_document_mask,
            FlashMaskType.FULL_DOCUMENT: self.generate_full_document_mask,
            FlashMaskType.SHARE_QUESTION: self.generate_share_question_mask,
            FlashMaskType.CAUSAL_BLOCKWISE: self.generate_causal_blockwise_mask,
            FlashMaskType.PREFIX_LM_CAUSAL: self.generate_prefix_lm_causal_mask,
            FlashMaskType.PREFIX_LM_DOCUMENT: self.generate_prefix_lm_document_mask,
            FlashMaskType.SLIDING_WINDOW: self.generate_sliding_window_mask,
            FlashMaskType.SLIDING_WINDOW_CAUSAL: self.generate_sliding_window_causal_mask,
            FlashMaskType.GLOBAL_SLIDING_WINDOW: self.generate_global_sliding_window_mask,
            FlashMaskType.BLOCK_CAUSAL_DOCUMENT: self.generate_block_causal_document_mask,
            FlashMaskType.FULL_DOCUMENT_LONG: self.generate_full_document_long_mask,
            FlashMaskType.CAUSAL_DOCUMENT_LONG: self.generate_causal_document_long_mask,
        }

    def generate(
        self,
        flash_mask_type: FlashMaskType,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        to_attn_ranges: bool = True,
        rng: random.Random | None = None,
    ) -> (
        tuple[list[list[int]], list[list[int]], list[int], MaskFactors]
        | tuple[AttnRanges, AttnRanges, list[AttnMaskType], MaskFactors]
    ):
        """generate a mask with mask type and data distribution

        Returns:
            The returned triple respectively describes the range of the query and key,
            as well as the type of Mask composed of them.
            NOTE: according to to_attn_ranges, it has two return types:
            1. AttnRanges, AttnRanges, list[AttnMaskType]
            2. list[list[int]], list[list[int]], list[bool]
        """
        q_ranges, k_ranges, attn_mask_type, mask_factors = self.generate_function[
            flash_mask_type
        ](
            seqlen_distribute=seqlen_distribute,
            total_seqlen=total_seqlen,
            window_size=window_size,
            rng=rng,
        )

        if to_attn_ranges:
            q_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=q_ranges)
            k_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=k_ranges)
            attn_mask_type_: list[AttnMaskType] = [
                [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.INVCAUSAL,
                    AttnMaskType.BICAUSAL,
                ][mask_type]
                for mask_type in attn_mask_type
            ]
            return (q_ranges_, k_ranges_, attn_mask_type_, mask_factors)

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def generate_causal_document_long_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate document full maks (varlen full mask)"""
        short_seqlen = int(total_seqlen * 0.4)
        short_seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=short_seqlen,
            rng=rng,
        )
        long_seqlens = generate_seqlens(
            distribution={(8 * 1024, 30 * 1024): 1.0},
            total_seqlen=total_seqlen - short_seqlen,
            rng=rng,
        )
        seqlens = short_seqlens + long_seqlens
        rng = rng if rng is not None else random
        rng.shuffle(seqlens)
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [1] * len(seqlens)

        return (ranges, ranges, is_causal_mapping, MaskFactors(cu_ranges=ranges))

    def generate_causal_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate document causal mask (varlen causal mask)"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [1] * len(seqlens)

        return (ranges, ranges, is_causal_mapping, MaskFactors(cu_ranges=ranges))

    def generate_causal_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate causal mask"""
        ranges = [[0, total_seqlen]]
        is_causal_mapping = [1]

        return (ranges, ranges, is_causal_mapping, MaskFactors())

    def generate_full_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate document full maks (varlen full mask)"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [0] * len(seqlens)

        return (ranges, ranges, is_causal_mapping, MaskFactors(cu_ranges=ranges))

    def generate_full_document_long_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate document full maks (varlen full mask)"""
        short_seqlen = int(total_seqlen * 0.4)
        short_seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=short_seqlen,
            rng=rng,
        )
        long_seqlens = generate_seqlens(
            distribution={(8 * 1024, 30 * 1024): 1.0},
            total_seqlen=total_seqlen - short_seqlen,
            rng=rng,
        )
        seqlens = short_seqlens + long_seqlens
        rng = rng if rng is not None else random
        rng.shuffle(seqlens)
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [0] * len(seqlens)

        return (ranges, ranges, is_causal_mapping, MaskFactors(cu_ranges=ranges))

    def generate_full_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate causal mask"""
        ranges = [[0, total_seqlen]]
        is_causal_mapping = [0]

        return (ranges, ranges, is_causal_mapping, MaskFactors())

    def generate_share_question_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate share question mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[int] = []
        for i in range(len(seqlens)):
            if i == 1:
                q_ranges[0] = [0, cu_seqlens[i + 1]]
                k_ranges[0] = [0, cu_seqlens[i + 1]]

                if len(seqlens) > 2:
                    q_ranges.append([cu_seqlens[i + 1], total_seqlen])
                    k_ranges.append([0, cu_seqlens[i]])
                    is_causal_mapping.append(0)
            else:
                q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                is_causal_mapping.append(1)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
        )

        return (q_ranges, k_ranges, is_causal_mapping, mask_factors)

    def generate_causal_blockwise_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate causal blockwise mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        for i in range(len(seqlens)):
            q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
            k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        k_ranges[-1] = [0, total_seqlen]

        is_causal_mapping = [1] * len(seqlens)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
        )

        return (q_ranges, k_ranges, is_causal_mapping, mask_factors)

    def generate_prefix_lm_causal_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate prefix lm causal mask"""
        rng = rng if rng is not None else random
        seqlen = rng.randint(1, total_seqlen)

        if seqlen < total_seqlen:
            q_ranges = [[0, seqlen], [seqlen, total_seqlen]]
            k_ranges = [[0, seqlen], [0, total_seqlen]]
            is_causal_mapping = [0, 1]
        else:
            q_ranges = [[0, total_seqlen]]
            k_ranges = [[0, total_seqlen]]
            is_causal_mapping = [0]

        mask_factors = MaskFactors()
        mask_factors.prefix_length = seqlen

        return (q_ranges, k_ranges, is_causal_mapping, mask_factors)

    def generate_prefix_lm_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate prefix lm document mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        rng = rng if rng is not None else random
        assert len(cu_seqlens) >= 2
        full_seqlen = rng.randint(1, cu_seqlens[1])

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[int] = []
        for i in range(len(seqlens)):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            if full_seqlen < seqlens[i]:
                q_ranges.append([start, start + full_seqlen])
                k_ranges.append([start, start + full_seqlen])
                is_causal_mapping.append(0)

                q_ranges.append([start + full_seqlen, end])
                k_ranges.append([start, end])
                is_causal_mapping.append(1)
            else:
                q_ranges.append([start, end])
                k_ranges.append([start, end])
                is_causal_mapping.append(0)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
            prefix_length=full_seqlen,
        )

        return (q_ranges, k_ranges, is_causal_mapping, mask_factors)

    def generate_sliding_window_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        assert window_size is not None and len(window_size) == 2
        mask_factors = MaskFactors(window_size=window_size[0])
        if window_size[0] >= total_seqlen:
            window_size = (-1, -1)

        q_ranges, k_ranges, attn_mask_type = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(start=0, end=total_seqlen),
            k_range=AttnRange(start=0, end=total_seqlen),
            window_size=window_size,  # type: ignore
        )

        attn_type_map = [
            {
                AttnMaskType.FULL: 0,
                AttnMaskType.CAUSAL: 1,
                AttnMaskType.INVCAUSAL: 2,
                AttnMaskType.BICAUSAL: 3,
            }[mask_type]
            for mask_type in attn_mask_type
        ]

        return (
            q_ranges.to_naive_ranges(),  # type: ignore
            k_ranges.to_naive_ranges(),
            attn_type_map,
            mask_factors,
        )

    def generate_sliding_window_causal_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        assert window_size is not None and len(window_size) == 2 and window_size[1] == 0
        if window_size[0] >= total_seqlen:
            window_size = (-1, 0)

        return self.generate_sliding_window_mask(
            seqlen_distribute=seqlen_distribute,
            total_seqlen=total_seqlen,
            window_size=window_size,
            rng=rng,
        )

    def generate_global_sliding_window_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ):
        rng = rng if rng is not None else random

        window_size_single: int = max(1, rng.randint(1, total_seqlen // 3 - 1))  # type: ignore

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[int] = []

        q_ranges.append([0, total_seqlen])
        k_ranges.append([0, window_size_single])
        is_causal_mapping.append(1)

        q_ranges.append([0, window_size_single])
        k_ranges.append([window_size_single, total_seqlen])
        is_causal_mapping.append(1)

        (
            sw_q_ranges,
            sw_k_ranges,
            sw_attn_mask_type,
        ) = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(start=window_size_single, end=total_seqlen),
            k_range=AttnRange(start=window_size_single, end=total_seqlen),
            window_size=[window_size_single, window_size_single],
        )

        sw_attn_type_map = [
            {
                AttnMaskType.FULL: 0,
                AttnMaskType.CAUSAL: 1,
                AttnMaskType.INVCAUSAL: 2,
                AttnMaskType.BICAUSAL: 3,
            }[mask_type]
            for mask_type in sw_attn_mask_type
        ]

        q_ranges.extend(sw_q_ranges.to_naive_ranges())  # type: ignore
        k_ranges.extend(sw_k_ranges.to_naive_ranges())  # type: ignore
        is_causal_mapping.extend(sw_attn_type_map)

        mask_factors = MaskFactors(window_size=window_size_single)

        return (q_ranges, k_ranges, is_causal_mapping, mask_factors)

    def generate_block_causal_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ):
        block_size = 1024
        assert total_seqlen % block_size == 0
        total_num_of_blocks = total_seqlen // block_size
        remaining_num_of_blocks = total_num_of_blocks
        block_begin = 0

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[int] = []

        cu_seqlens: list[int] = [0]
        cu_ranges: list[list[int]] = []

        rng = rng if rng is not None else random

        while remaining_num_of_blocks > 0:
            num_of_blocks = min(rng.randint(1, 8), remaining_num_of_blocks)
            remaining_num_of_blocks -= num_of_blocks

            for index in range(num_of_blocks):
                q_ranges.append(
                    [
                        block_begin + index * block_size,
                        block_begin + (index + 1) * block_size,
                    ]
                )
                k_ranges.append([block_begin, block_begin + (index + 1) * block_size])
                is_causal_mapping.append(0)

            block_begin += num_of_blocks * block_size

            cu_seqlens.append(block_begin)
            cu_ranges.append([block_begin - num_of_blocks * block_size, block_begin])

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
            block_size=block_size,
        )
        print(f"{q_ranges=} {k_ranges=}")

        return q_ranges, k_ranges, is_causal_mapping, mask_factors


class MaskIterator:
    """This is a iterator for multiple flash masks, it can be used with
    several params in init and get an iterator,
    """

    def __init__(
        self,
        generate_times: int,
        generate_mask: FlashMaskType,
        total_seqlen: int,
        window_size: tuple[int, int] | None = None,
        distribution: dict[tuple[int, int], float] | None = None,
        to_attn_ranges: bool = True,
        seed: int | None = None,
    ):
        # set params in interator
        self.generate_times = generate_times
        self.current_times = 0
        self.generate_mask = generate_mask
        self.total_seqlen = total_seqlen
        self.to_attn_ranges = to_attn_ranges

        if distribution is not None:
            self.seqlen_distribution = distribution
        elif self.total_seqlen > 128 * 1024:  # 128k
            self.seqlen_distribution = varlen_long_seqlen_distribution()
        else:
            self.seqlen_distribution = varlen_short_seqlen_distribution()

        if seed is not None:
            self.random_number_generator = random.Random(seed)  # type: ignore
        else:
            self.random_number_generator = None  # type: ignore

        self.window_size = window_size
        self.mask_generator = MaskGenerator()

    def __iter__(self):
        assert (
            self.generate_times > 0
        ), f"generate times must greater than 0, but got {self.generate_times}"
        return self

    def __next__(self):
        if self.current_times >= self.generate_times:
            raise StopIteration
        value = self.mask_generator.generate(
            flash_mask_type=self.generate_mask,
            seqlen_distribute=self.seqlen_distribution,
            total_seqlen=self.total_seqlen,
            to_attn_ranges=self.to_attn_ranges,
            window_size=self.window_size,
            rng=self.random_number_generator,
        )
        self.current_times += 1
        return value
