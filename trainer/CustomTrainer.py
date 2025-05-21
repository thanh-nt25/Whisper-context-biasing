from transformers import Seq2SeqTrainer
from typing import Optional, Union, Any, Dict, Tuple
import torch

class CustomTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = "labels" in inputs

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

        # ✅ lấy bias_spans (đã được data collator xử lý)
        bias_spans = inputs.get("bias_spans", None)
        extra_inputs = {"bias_spans": bias_spans}  # có thể thêm input_names,... nếu cần

        return (loss, logits, labels, extra_inputs)
