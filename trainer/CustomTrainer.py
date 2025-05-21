from transformers import Seq2SeqTrainer, EvalPrediction
import torch
from typing import Optional, List, Dict, Union, Any


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_bias_spans = []  # bộ nhớ tạm để giữ bias spans cho toàn batch

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        has_labels = "labels" in inputs

        # Lưu bias_spans ra ngoài, tránh truyền vào model
        bias_spans = inputs.get("bias_spans", None)

        # Xoá khỏi inputs để không gây lỗi model(**inputs)
        clean_inputs = {k: v for k, v in inputs.items() if k != "bias_spans"}
        clean_inputs = self._prepare_inputs(clean_inputs)

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, clean_inputs, return_outputs=True)
                logits = outputs.logits
            else:
                loss = None
                outputs = model(**clean_inputs)
                logits = outputs.logits

        labels = clean_inputs.get("labels")

        # ✅ Ghi lại bias_spans nếu có
        if bias_spans is not None:
            self._stored_bias_spans.append(bias_spans.detach().cpu())

        return (loss, logits, labels)

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        # gọi hàm evaluation_loop gốc
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # ✅ Gắn bias_spans vào EvalPrediction.inputs
        if self._stored_bias_spans:
            # Ghép các batch lại thành tensor
            bias_spans = torch.cat(self._stored_bias_spans, dim=0).numpy()

            # Tạo EvalPrediction mới (vì nó là immutable)
            output = EvalPrediction(
                predictions=output.predictions,
                label_ids=output.label_ids,
                inputs=(bias_spans,)  # tuple 1 phần tử
            )

            # Reset lại bộ nhớ tạm
            self._stored_bias_spans = []

        return output
