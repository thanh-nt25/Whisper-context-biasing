# Trong file CustomTrainer.py
from transformers.trainer_utils import EvalPrediction
import numpy as np

# Định nghĩa lớp EvalPrediction tùy chỉnh
class CustomEvalPrediction(EvalPrediction):
    def __init__(self, predictions, label_ids, bias_spans=None):
        super().__init__(predictions=predictions, label_ids=label_ids)
        self.bias_spans = bias_spans

# Ghi đè Trainer
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_bias_spans = None
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Lưu bias_spans nếu có
        if "bias_spans" in inputs:
            self.stored_bias_spans = inputs.pop("bias_spans").detach().cpu().numpy()
        
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def prediction_loop(self, *args, **kwargs):
        output = super().prediction_loop(*args, **kwargs)
        # Lưu bias_spans để sử dụng trong evaluate
        if hasattr(self, "stored_bias_spans") and self.stored_bias_spans is not None:
            self.final_bias_spans = self.stored_bias_spans
        
        return output
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # Xóa bias_spans sau khi đã sử dụng
        if hasattr(self, "final_bias_spans"):
            delattr(self, "final_bias_spans")
        
        return metrics
    
    # Ghi đè phương thức _compute_metrics để sử dụng CustomEvalPrediction
    def _compute_metrics(self, preds, eval_dataset=None):
        if self.compute_metrics is not None and preds is not None:
            bias_spans = getattr(self, "final_bias_spans", None)
            
            # Sử dụng CustomEvalPrediction
            if bias_spans is not None:
                eval_preds = CustomEvalPrediction(
                    predictions=preds.predictions,
                    label_ids=preds.label_ids,
                    bias_spans=bias_spans
                )
            else:
                eval_preds = EvalPrediction(
                    predictions=preds.predictions,
                    label_ids=preds.label_ids
                )
            
            return self.compute_metrics(eval_preds)
        return {}