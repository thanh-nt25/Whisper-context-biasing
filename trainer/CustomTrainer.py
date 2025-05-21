from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction
from transformers.trainer_utils import EvalPrediction
from typing import Optional, Dict, Any
import torch
from torch.nn import CrossEntropyLoss

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, bias_spans_dataset: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_spans_dataset = bias_spans_dataset  # Optional external reference
        self._stored_bias_spans = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        bias_spans_batch = inputs.get("bias_spans", None)  # List[List[List[int]]]

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)

        if bias_spans_batch is not None:
            bias_mask = torch.zeros_like(labels, dtype=torch.float)
            for i, spans in enumerate(bias_spans_batch):
                for span in spans:
                    span_tensor = torch.tensor(span, device=labels.device)
                    for j in range(labels.size(1) - len(span) + 1):
                        if torch.equal(labels[i, j:j+len(span)], span_tensor):
                            bias_mask[i, j:j+len(span)] = 1.0
            weights = 1.0 + 4.0 * bias_mask
            loss = loss * weights

        final_loss = loss[labels != -100].mean()
        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if hasattr(self.data_collator, "bias_spans_batch"):
            self._stored_bias_spans = self.data_collator.bias_spans_batch
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        # Đảm bảo bias_spans được gắn vào EvalPrediction
        eval_pred = EvalPrediction(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics if hasattr(output, "metrics") else {}
        )
        if self._stored_bias_spans is not None:
            print("Eval take bias span")
            eval_pred.bias_spans = self._stored_bias_spans

        return eval_pred
