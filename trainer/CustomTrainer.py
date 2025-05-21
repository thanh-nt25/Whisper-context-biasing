# File: /kaggle/working/Whisper-context-biasing/trainer/CustomTrainer.py

from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from typing import Dict, List, Optional, Union, Any

class CustomTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Lưu bias_spans từ inputs nếu có
        if "bias_spans" in inputs:
            self.bias_spans = inputs.pop("bias_spans")
        
        # Gọi hàm prediction_step của lớp cha
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Gọi phương thức evaluate của lớp cha
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        return metrics
    
    def prediction_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Ghi đè phương thức prediction_loop để truyền bias_spans vào PredictionOutput
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only

        # Tạo và sử dụng progress bar tùy thuộc vào môi trường
        if self.args.do_predict:
            logger.info(f"***** Running {description} *****")

        # Gọi phương thức prediction_loop của lớp cha
        outputs = super().prediction_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Thêm bias_spans vào outputs nếu có
        if hasattr(self, "bias_spans"):
            # Tạo một PredictionOutput mới bao gồm bias_spans
            enhanced_outputs = PredictionOutput(
                predictions=outputs.predictions,
                label_ids=outputs.label_ids,
                metrics=outputs.metrics,
                num_samples=outputs.num_samples,
            )
            # Thêm bias_spans vào đối tượng
            enhanced_outputs.bias_spans = self.bias_spans
            return enhanced_outputs
        
        return outputs