from transformers import Seq2SeqTrainer, EvalPrediction
from typing import Optional, Any
import torch

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_bias_spans = []  # Store bias spans across batches during evaluation

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        bias_spans_batch = inputs.pop("bias_spans", None)  # Remove from model inputs

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)

        if bias_spans_batch is not None:
            # Create a bias weight mask
            bias_mask = torch.zeros_like(labels, dtype=torch.float)
            for i, spans in enumerate(bias_spans_batch):
                for span in spans:
                    for j in range(labels.size(1) - len(span) + 1):
                        if torch.equal(labels[i, j:j+len(span)], torch.tensor(span, device=labels.device)):
                            bias_mask[i, j:j+len(span)] = 1.0

            weights = 1.0 + 4.0 * bias_mask
            loss = loss * weights

        final_loss = loss[labels != -100].mean()
        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        bias_spans = inputs.pop("bias_spans", None)

        # Save bias spans for use in evaluation metrics
        if bias_spans is not None:
            self._all_bias_spans.extend(bias_spans)

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        # Wrap into a full EvalPrediction object with bias info for compute_metrics
        return EvalPrediction(
            predictions=output.predictions,
            label_ids=output.label_ids,
            inputs={"bias_spans": self._all_bias_spans}  # Ensure this is a dict
        )
