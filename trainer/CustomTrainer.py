# File: /kaggle/working/Whisper-context-biasing/trainer/CustomTrainer.py

from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_spans_queue = []  # Hàng đợi để lưu tất cả bias_spans trong quá trình đánh giá
        
    def training_step(self, model, inputs):
        """
        Perform a training step with bias loss customization if needed
        """
        # Lưu bias_spans nếu cần dùng cho custom loss
        bias_spans = None
        if "bias_spans" in inputs:
            bias_spans = inputs.pop("bias_spans")
        
        # Tiếp tục training step bình thường
        loss = super().training_step(model, inputs)
        
        # Thêm xử lý tùy chỉnh nếu cần (bias weighted loss)
        if bias_spans is not None:
            # Ở đây bạn có thể thêm code để điều chỉnh loss dựa trên bias_spans
            # Ví dụ: loss = loss * custom_weight_based_on_bias(bias_spans)
            pass
            
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step and handle bias_spans specially
        """
        # Lưu bias_spans cho lần dự đoán này
        self.current_bias_spans = None
        if "bias_spans" in inputs:
            self.current_bias_spans = inputs.pop("bias_spans").detach().cpu()
        
        # Thực hiện dự đoán
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # Nếu đang trong quá trình đánh giá, lưu bias_spans
        if not prediction_loss_only and self.current_bias_spans is not None:
            self.bias_spans_queue.append(self.current_bias_spans)
        
        return outputs
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Reset bias_spans_queue mỗi khi bắt đầu đánh giá
        """
        self.bias_spans_queue = []
        return super().get_eval_dataloader(eval_dataset)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Mở rộng evaluation_loop để bao gồm bias_spans
        """
        # Reset hàng đợi bias_spans
        self.bias_spans_queue = []
        
        # Thực hiện đánh giá thông thường
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Lưu bias_spans để sử dụng trong compute_metrics
        if self.bias_spans_queue:
            # Ghép tất cả bias_spans lại 
            # Lưu ý: Cần đảm bảo thứ tự khớp với thứ tự của predictions
            try:
                all_bias_spans = torch.cat(self.bias_spans_queue, dim=0).numpy()
                # Gán vào self để có thể truy cập trong _compute_metrics
                self.all_bias_spans = all_bias_spans
            except Exception as e:
                print(f"Lỗi khi ghép bias_spans: {e}")
                self.all_bias_spans = None
        else:
            self.all_bias_spans = None
        
        return output

    def _compute_metrics(self, preds):
        """
        Mở rộng _compute_metrics để truyền bias_spans vào compute_metrics
        """
        if self.compute_metrics is None:
            return {}

        # Chuẩn bị dữ liệu đầu vào cho compute_metrics
        pred_ids = preds.predictions
        label_ids = preds.label_ids
        
        # Tạo đối tượng metric_args để truyền thêm thông tin vào compute_metrics
        metric_args = {
            "pred_ids": pred_ids,
            "label_ids": label_ids
        }
        
        # Thêm bias_spans nếu có
        if hasattr(self, "all_bias_spans") and self.all_bias_spans is not None:
            metric_args["bias_spans"] = self.all_bias_spans
        
        # Gọi compute_metrics với metric_args
        metrics = self.compute_metrics(EvalPrediction(predictions=pred_ids, label_ids=label_ids), **metric_args)
        
        return metrics