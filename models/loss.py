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
    
    # Forward pass
    outputs = model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        labels=None,  # Không tính loss tự động
    )
    
    # Lấy logits
    logits = outputs.logits
    
    # Tạo ma trận trọng số với giá trị mặc định là 1
    weights = torch.ones_like(labels, dtype=torch.float)
    
    # Gán trọng số cao hơn cho token thuộc thuật ngữ y tế
    for batch_idx in range(labels.shape[0]):
        for pos_idx in range(labels.shape[1]):
            if labels[batch_idx, pos_idx] != -100:  # Bỏ qua token padding
                token = labels[batch_idx, pos_idx].item()
                if token in medical_terms_mapping:
                    term_type = medical_terms_mapping[token]
                    weights[batch_idx, pos_idx] = weight_factors.get(term_type, 1.1)
    
    # Tính Cross Entropy Loss không giảm
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(labels.shape)
    
    # Nhân loss với trọng số
    weighted_loss = loss * weights
    
    # Lấy trung bình loss (bỏ qua các vị trí padding)
    weighted_loss = weighted_loss[labels != -100].mean()
    
    return weighted_loss