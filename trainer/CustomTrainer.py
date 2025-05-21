from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction
from transformers.trainer_utils import EvalPrediction
from typing import Optional, Dict, Any
import torch
from torch.nn import CrossEntropyLoss

class CustomTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if hasattr(self.data_collator, "tmp_bias_spans"):
            inputs["bias_spans"] = self.data_collator.tmp_bias_spans  # gắn lại cho compute_metrics
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
