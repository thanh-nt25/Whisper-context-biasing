import copy
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

# Import các hàm utility từ mã nguồn gốc
from transformers.models.whisper.modeling_whisper import (
    WHISPER_ATTENTION_CLASSES,
    WhisperPositionalEmbedding,
    sinusoids,
    shift_tokens_right,
    _get_unpad_data,
    _compute_mask_indices,
    _median_filter,
    _dynamic_time_warping
)

