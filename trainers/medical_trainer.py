"""
Trainer tùy chỉnh cho fine-tuning Whisper với y tế
"""

import torch
import random
from transformers import Trainer
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from models.loss import compute_medical_weighted_loss
from config.config import RANDOM_CONTEXT_PROB

class WhisperMedicalTrainer(Trainer):
    """
    Trainer tùy chỉnh kết hợp context perturbation và weighted loss
    """
    
    def __init__(
        self, 
        random_context_prob=RANDOM_CONTEXT_PROB, 
        random_contexts=None, 
        medical_terms_mapping=None,
        weight_factors=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.random_context_prob = random_context_prob
        self.random_contexts = random_contexts or []
        self.medical_terms_mapping = medical_terms_mapping or {}
        self.weight_factors = weight_factors or {}
    
    def compute_loss(self, model, inputs, return_outputs=False, , num_items_in_batch=None):
        """
        Tính loss với context perturbation và medical weighted loss
        """
        # Áp dụng context perturbation
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
                    
                    # Truncate nếu cần
                    max_len = inputs["decoder_input_ids"].shape[1]
                    if len(random_prompt_ids) > max_len:
                        random_prompt_ids = random_prompt_ids[:max_len]
                    
                    # Pad nếu cần
                    if len(random_prompt_ids) < max_len:
                        padding = torch.ones(
                            max_len - len(random_prompt_ids), 
                            dtype=torch.long
                        ) * self.tokenizer.pad_token_id
                        padding = padding.to(inputs["decoder_input_ids"].device)
                        random_prompt_ids = torch.cat([random_prompt_ids, padding])
                    
                    # Thay thế prompt
                    inputs["decoder_input_ids"][i] = random_prompt_ids
        
        # Tính loss tùy chỉnh nếu có medical_terms_mapping
        if self.medical_terms_mapping and self.weight_factors:
            loss = compute_medical_weighted_loss(
                model, 
                inputs, 
                self.medical_terms_mapping, 
                self.weight_factors
            )
            
            if return_outputs:
                # Forward pass để lấy outputs
                outputs = model(
                    input_features=inputs.get("input_features"),
                    decoder_input_ids=inputs.get("decoder_input_ids"),
                )
                return (loss, outputs)
            return loss
        else:
            # Sử dụng loss mặc định nếu không có medical_terms_mapping
            return super().compute_loss(model, inputs, return_outputs)