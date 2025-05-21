from transformers import Seq2SeqTrainer
from typing import Optional, Union, Any, Dict, Tuple
import torch

class CustomTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs

        # Lưu bias_spans lại để truyền về sau
        bias_spans = inputs.get("bias_spans", None)

        # Xóa khỏi inputs để không bị lỗi khi gọi model
        inputs = {k: v for k, v in inputs.items() if k != "bias_spans"}

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs.logits
            else:
                loss = None
                outputs = model(**inputs)
                logits = outputs.logits

        labels = inputs.get("labels")

        return (loss, logits, labels, {"bias_spans": bias_spans})
