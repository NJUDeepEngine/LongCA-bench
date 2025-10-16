from enum import Enum


class FlashMaskType(Enum):
    FULL = "full"
    CAUSAL = "causal"
    CAUSAL_DOCUMENT = "causal_document"
    FULL_DOCUMENT = "full_document"
    SHARE_QUESTION = "share_question"
    CAUSAL_BLOCKWISE = "causal_blockwise"
    PREFIX_LM_DOCUMENT = "prefix_lm_document"
    PREFIX_LM_CAUSAL = "prefix_lm_causal"
    QK_SPARSE = "qk_sparse"
    HASH_SPARSE = "hash_sparse"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_CAUSAL = "sliding_window_causal"
    GLOBAL_SLIDING_WINDOW = "global_sliding_window"
    BLOCK_CAUSAL_DOCUMENT = "block_causal_document"
    FULL_DOCUMENT_LONG = "full_document_long"
    CAUSAL_DOCUMENT_LONG = "causal_document_long"


class MetricsType(Enum):
    RANGES_INFORMATION_ENTROPY = "ranges_information_entropy"
    AREA_INFORMATION_ENTROPY = "areas_information_entropy"
    REMOTE_NORMALIZED_VALUE = "remote_normalized_value"
    MAX_AREA_DIVIDED_BY_TOTAL_AREA = "max_area_divided_by_total_area"
    MAX_AREA_DIVIDED_BY_AVERAGE_AREA = "max_area_divided_by_average_area"
    AREA_GINI_IMPURITY = "area_gini_impurity"
    RANGES_GIMI_IMPURITY = "ranges_gimi_impurity"
    COST_MODEL = "cost_model"
