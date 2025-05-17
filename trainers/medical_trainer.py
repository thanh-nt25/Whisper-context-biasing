"""
Trainer tùy chỉnh cho fine-tuning Whisper với y tế
"""

import torch
import random
from transformers import Trainer
import os
import sys
from pathlib import Path
import gc

from utils.evaluation import compute_metrics_whisper_with_prompt

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from models.loss import compute_medical_weighted_loss
from config.config import RANDOM_CONTEXT_PROB

class WhisperMedicalTrainer(Trainer):
    """
    Trainer tùy chỉnh cho fine-tuning Whisper với medical prompts
    """
    
    def __init__(
        self, 
        random_context_prob=0.05, 
        random_contexts=None, 
        medical_terms_mapping=None,
        weight_factors=None,
        processing_class=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.random_context_prob = random_context_prob
        self.random_contexts = random_contexts or []
        self.medical_terms_mapping = medical_terms_mapping or {}
        self.weight_factors = weight_factors or {}
        self.processing_class = processing_class
    
    def _free_memory(self):
        """Giải phóng bộ nhớ GPU"""
        gc.collect()
        torch.cuda.empty_cache()
        
    # def compute_metrics(self, eval_preds):
    #   print("Trgger compute_metrics on WhisperMedicalTrainer method!")
    #   return compute_metrics_whisper_with_prompt(
    #     eval_preds = eval_preds,
    #     tokenizer = self.tokenizer,
    #     prompt_ids_list=getattr(self, "prompt_ids_list", None)
    #   )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Tính loss cho mô hình với context perturbation
        """
        # Áp dụng context perturbation nếu có random_contexts
        # if self.random_contexts and random.random() < self.random_context_prob:
        #     batch_size = inputs["decoder_input_ids"].shape[0]
        #     for i in range(batch_size):
        #         if random.random() < self.random_context_prob:
        #             # Chọn ngẫu nhiên một prompt
        #             random_prompt = random.choice(self.random_contexts)
        #             random_prompt_ids = self.tokenizer(
        #                 random_prompt, 
        #                 return_tensors="pt"
        #             ).input_ids.squeeze(0).to(inputs["decoder_input_ids"].device)
                    
        #             # Đảm bảo kích thước phù hợp
        #             max_len = inputs["decoder_input_ids"].shape[1]
        #             if len(random_prompt_ids) > max_len:
        #                 random_prompt_ids = random_prompt_ids[:max_len]
        #             elif len(random_prompt_ids) < max_len:
        #                 padding = torch.ones(
        #                     max_len - len(random_prompt_ids), 
        #                     dtype=torch.long
        #                 ) * self.tokenizer.pad_token_id
        #                 padding = padding.to(inputs["decoder_input_ids"].device)
        #                 random_prompt_ids = torch.cat([random_prompt_ids, padding])
                    
        #             # Thay thế prompt
        #             inputs["decoder_input_ids"][i] = random_prompt_ids
        
        # Sử dụng forward pass tự động và tính loss mặc định
        outputs = model(
            input_features=inputs.get("input_features"),
            # decoder_input_ids=inputs.get("decoder_input_ids"),
            # decoder_attention_mask=inputs.get("decoder_attention_mask", None),
            labels=inputs.get("labels")
        )
        
        loss = outputs.loss
        
        # Áp dụng trọng số nếu có medical_terms_mapping
        if self.medical_terms_mapping and self.weight_factors and hasattr(outputs, "logits"):
            logits = outputs.logits
            labels = inputs.get("labels")
            
            # Chỉ áp dụng trọng số nếu có token y tế trong labels
            has_medical_term = False
            for batch_idx in range(labels.shape[0]):
                for pos_idx in range(labels.shape[1]):
                    if labels[batch_idx, pos_idx] != -100:  # Bỏ qua token padding
                        token = labels[batch_idx, pos_idx].item()
                        if token in self.medical_terms_mapping:
                            has_medical_term = True
                            break
                if has_medical_term:
                    break
            
            if has_medical_term:
                # Tính custom loss với trọng số
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                custom_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                custom_loss = custom_loss.view(labels.shape)
                
                # Tạo ma trận trọng số
                weights = torch.ones_like(labels, dtype=torch.float)
                for batch_idx in range(labels.shape[0]):
                    for pos_idx in range(labels.shape[1]):
                        if labels[batch_idx, pos_idx] != -100:
                            token = labels[batch_idx, pos_idx].item()
                            if token in self.medical_terms_mapping:
                                term_type = self.medical_terms_mapping[token]
                                weights[batch_idx, pos_idx] = self.weight_factors.get(term_type, 1.1)
                
                # Áp dụng trọng số và tính trung bình
                weighted_loss = custom_loss * weights
                active_loss = labels != -100
                if active_loss.sum() > 0:
                    loss = weighted_loss[active_loss].mean()
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """
        Override training_step để giải phóng bộ nhớ định kỳ
        """
        loss = super().training_step(model, inputs)
        
        # Giải phóng bộ nhớ mỗi 50 bước
        if self.state.global_step % 50 == 0:
            self._free_memory()
            
        return loss
    
    def _evaluate(self, *args, **kwargs):
        """
        Override _evaluate để giải phóng bộ nhớ trước và sau evaluation
        """
        self._free_memory()
        try:
            result = super()._evaluate(*args, **kwargs)
            return result
        finally:
            self._free_memory()
            
class DebugWhisperMedicalTrainer(WhisperMedicalTrainer):
    """
    Phiên bản WhisperMedicalTrainer với debug chi tiết
    """
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Ghi đè phương thức evaluate để debug
        """
        print("\n=== DEBUG: evaluate method called ===")
        
        # Khởi tạo dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        # Gọi phương thức gốc với try/except để bắt lỗi
        try:
            result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            print(f"Evaluate succeeded with result: {result}")
            return result
        except Exception as e:
            print(f"Error in evaluate: {e}")
            import traceback
            traceback.print_exc()
            return {f"{metric_key_prefix}_error": str(e)}
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Ghi đè get_eval_dataloader để debug quá trình tạo dataloader
        """
        print("\n=== DEBUG: get_eval_dataloader called ===")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        print(f"Creating dataloader for {len(eval_dataset)} samples")
        
        try:
            dataloader = super().get_eval_dataloader(eval_dataset)
            print(f"Created dataloader with {len(dataloader)} batches")
            return dataloader
        except Exception as e:
            print(f"Error in get_eval_dataloader: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Ghi đè evaluation_loop để debug quá trình đánh giá
        """
        print("\n=== DEBUG: evaluation_loop called ===")
        print(f"Dataloader size: {len(dataloader)}")
        
        # Đặt model ở chế độ đánh giá
        self.model.eval()
        
        # Khởi tạo các biến
        all_preds = []
        all_labels = []
        processed_batches = 0
        
        print("Starting evaluation loop...")
        for step, inputs in enumerate(dataloader):
            # In thông tin cho các batch đầu tiên
            if step < 3 or step % 20 == 0:
                print(f"Processing batch {step}/{len(dataloader)}")
                print(f"  Batch keys: {inputs.keys()}")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key} shape: {value.shape}")
            
            # Chuyển inputs lên device
            inputs = self._prepare_inputs(inputs)
            
            # Dự đoán cho batch hiện tại
            try:
                loss, logits, labels = self.prediction_step(
                    self.model, inputs, prediction_loss_only, ignore_keys=ignore_keys
                )
                
                if step < 3 or step % 20 == 0:
                    print(f"  Prediction step successful:")
                    print(f"    Loss: {loss}")
                    print(f"    Logits shape: {logits.shape if logits is not None else None}")
                    print(f"    Labels shape: {labels.shape if labels is not None else None}")
                
                # Lưu trữ dự đoán và nhãn
                if logits is not None:
                    all_preds.append(logits.detach().cpu().numpy())
                if labels is not None:
                    all_labels.append(labels.detach().cpu().numpy())
                
                processed_batches += 1
            except Exception as e:
                print(f"Error in prediction_step for batch {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Giải phóng bộ nhớ
            del inputs, loss, logits, labels
            gc.collect()
            if processed_batches % 20 == 0:
                torch.cuda.empty_cache()
        
        print(f"Evaluation loop completed. Processed {processed_batches}/{len(dataloader)} batches")
        
        # Kiểm tra kết quả
        if not all_preds or not all_labels:
            print("WARNING: No predictions or labels collected!")
            return {f"{metric_key_prefix}_wer": 0.0}
        
        try:
            # Ghép các dự đoán và nhãn
            import numpy as np
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            print(f"Concatenated predictions shape: {all_preds.shape}")
            print(f"Concatenated labels shape: {all_labels.shape}")
            
            # Tạo EvalPrediction
            eval_preds = EvalPrediction(predictions=all_preds, label_ids=all_labels)
            
            # Gọi compute_metrics
            metrics = self.compute_metrics(eval_preds)
            print(f"Metrics computed: {metrics}")
            
            return metrics
        except Exception as e:
            print(f"Error in processing evaluation results: {e}")
            import traceback
            traceback.print_exc()
            return {f"{metric_key_prefix}_error": str(e)}
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Ghi đè prediction_step để debug quá trình dự đoán
        """
        # Không in log cho mọi bước dự đoán (sẽ quá nhiều thông tin)
        # Chỉ in cho bước đầu tiên
        if not hasattr(self, "_prediction_step_count"):
            self._prediction_step_count = 0
            print("\n=== DEBUG: First prediction_step called ===")
            print(f"Inputs keys: {inputs.keys()}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key} shape: {value.shape}")
        
        self._prediction_step_count += 1
        
        # Gọi phương thức gốc
        try:
            outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            
            # In thông tin cho bước đầu tiên
            if self._prediction_step_count <= 1:
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    loss, logits, labels = outputs
                    print(f"First prediction step successful:")
                    print(f"  Loss: {loss}")
                    print(f"  Logits shape: {logits.shape if logits is not None else None}")
                    print(f"  Labels shape: {labels.shape if labels is not None else None}")
                else:
                    print(f"Unexpected output format: {outputs}")
            
            return outputs
        except Exception as e:
            print(f"Error in prediction_step (step {self._prediction_step_count}): {e}")
            import traceback
            traceback.print_exc()
            raise e