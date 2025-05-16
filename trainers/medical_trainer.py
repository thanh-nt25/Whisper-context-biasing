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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Tính loss cho mô hình với context perturbation
        """
        # Áp dụng context perturbation nếu có random_contexts
        if self.random_contexts and random.random() < self.random_context_prob:
            batch_size = inputs["decoder_input_ids"].shape[0]
            for i in range(batch_size):
                if random.random() < self.random_context_prob:
                    # Chọn ngẫu nhiên một prompt
                    random_prompt = random.choice(self.random_contexts)
                    random_prompt_ids = self.tokenizer(
                        random_prompt, 
                        return_tensors="pt"
                    ).input_ids.squeeze(0).to(inputs["decoder_input_ids"].device)
                    
                    # Đảm bảo kích thước phù hợp
                    max_len = inputs["decoder_input_ids"].shape[1]
                    if len(random_prompt_ids) > max_len:
                        random_prompt_ids = random_prompt_ids[:max_len]
                    elif len(random_prompt_ids) < max_len:
                        padding = torch.ones(
                            max_len - len(random_prompt_ids), 
                            dtype=torch.long
                        ) * self.tokenizer.pad_token_id
                        padding = padding.to(inputs["decoder_input_ids"].device)
                        random_prompt_ids = torch.cat([random_prompt_ids, padding])
                    
                    # Thay thế prompt
                    inputs["decoder_input_ids"][i] = random_prompt_ids
        
        # Sử dụng forward pass tự động và tính loss mặc định
        outputs = model(
            input_features=inputs.get("input_features"),
            decoder_input_ids=inputs.get("decoder_input_ids"),
            decoder_attention_mask=inputs.get("decoder_attention_mask", None),
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