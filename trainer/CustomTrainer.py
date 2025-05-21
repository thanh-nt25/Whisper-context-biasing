# File: /kaggle/working/Whisper-context-biasing/trainer/CustomTrainer.py

from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_spans_batch_info = []  # Lưu cả bias_spans và batch_size
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step and handle bias_spans specially
        """
        # Lưu bias_spans cho lần dự đoán này
        if "bias_spans" in inputs:
            # Lưu bias_spans và kích thước batch
            bias_spans = inputs.pop("bias_spans").detach().cpu()
            batch_size = bias_spans.size(0)
            self.bias_spans_batch_info.append({
                "bias_spans": bias_spans,
                "batch_size": batch_size
            })
        
        # Thực hiện dự đoán
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        return outputs
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Reset bias_spans_batch_info mỗi khi bắt đầu đánh giá
        """
        self.bias_spans_batch_info = []
        return super().get_eval_dataloader(eval_dataset)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Mở rộng evaluation_loop để bao gồm bias_spans
        """
        # Reset thông tin bias_spans
        self.bias_spans_batch_info = []
        
        # Thực hiện đánh giá thông thường
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Xử lý bias_spans để sử dụng trong compute_metrics
        if self.bias_spans_batch_info:
            try:
                # Xử lý bias_spans với các kích thước khác nhau
                self.all_bias_spans = self._process_variable_size_bias_spans()
            except Exception as e:
                print(f"Lỗi khi xử lý bias_spans: {e}")
                self.all_bias_spans = None
        else:
            self.all_bias_spans = None
        
        return output
    
    def _process_variable_size_bias_spans(self):
        """
        Xử lý bias_spans với các kích thước khác nhau
        """
        # In thông tin kích thước của các tensor để debug
        for i, info in enumerate(self.bias_spans_batch_info):
            print(f"Batch {i}: bias_spans shape = {info['bias_spans'].shape}, batch_size = {info['batch_size']}")
        
        # Chuyển tất cả bias_spans thành danh sách 1D
        all_spans = []
        
        for info in self.bias_spans_batch_info:
            bias_spans = info["bias_spans"]
            batch_size = info["batch_size"]
            
            # Xử lý từng mẫu trong batch
            for i in range(batch_size):
                # Lấy bias_spans cho mẫu hiện tại
                sample_spans = bias_spans[i]
                all_spans.append(sample_spans)
        
        # Chuyển thành numpy array để sử dụng trong compute_metrics
        # Không ghép các tensor với torch.cat vì chúng có kích thước khác nhau
        return all_spans
    
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