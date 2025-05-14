"""
Loss function tùy chỉnh cho Whisper medical fine-tuning
"""

import torch

def compute_medical_weighted_loss(model, inputs, medical_terms_mapping, weight_factors):
    """
    Tính toán Weighted Cross Entropy Loss với trọng số cao hơn cho thuật ngữ y tế
    
    Args:
        model: Mô hình Whisper
        inputs: Input features và labels
        medical_terms_mapping: Dictionary ánh xạ từ token_id sang loại thuật ngữ y tế
        weight_factors: Dictionary chứa trọng số cho từng loại thuật ngữ y tế
    """
    
    input_features = inputs.get("input_features")
    labels = inputs.get("labels")
    decoder_input_ids = inputs.get("decoder_input_ids")
    
    
    outputs = model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        labels=None,  
    )
    
    
    logits = outputs.logits
    
    if logits.size(1) != labels.size(1):
        labels = labels[:, :logits.size(1)]  
    assert logits.shape[:2] == labels.shape, f"Mismatch: logits {logits.shape}, labels {labels.shape}"
    
    
    weights = torch.ones_like(labels, dtype=torch.float)
    
    
    for batch_idx in range(labels.shape[0]):
        for pos_idx in range(labels.shape[1]):
            if labels[batch_idx, pos_idx] != -100:  
                token = labels[batch_idx, pos_idx].item()
                if token in medical_terms_mapping:
                    term_type = medical_terms_mapping[token]
                    weights[batch_idx, pos_idx] = weight_factors.get(term_type, 1.1)
    
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(labels.shape)
    
    
    weighted_loss = loss * weights
    
    
    weighted_loss = weighted_loss[labels != -100].mean()
    
    return weighted_loss