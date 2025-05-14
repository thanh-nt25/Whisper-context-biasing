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
    
    # Forward pass với cơ chế tính loss tự động của Whisper
    outputs = model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        labels=labels,  # Sử dụng labels trực tiếp
    )
    
    # Lấy logits và loss chuẩn từ outputs
    logits = outputs.logits
    standard_loss = outputs.loss
    
    # Nếu không có medical_terms để áp dụng trọng số, trả về loss chuẩn
    if not medical_terms_mapping or len(medical_terms_mapping) == 0:
        return standard_loss
    
    # Kiểm tra batch size
    batch_size = labels.shape[0]
    assert logits.shape[0] == batch_size, f"Batch size mismatch: logits {logits.shape[0]}, labels {batch_size}"
    
    # Tạo ma trận trọng số với giá trị mặc định là 1
    weights = torch.ones_like(labels, dtype=torch.float)
    
    # Gán trọng số cao hơn cho token thuộc thuật ngữ y tế
    for batch_idx in range(batch_size):
        for pos_idx in range(labels.shape[1]):
            if labels[batch_idx, pos_idx] != -100:  # Bỏ qua token padding
                token = labels[batch_idx, pos_idx].item()
                if token in medical_terms_mapping:
                    term_type = medical_terms_mapping[token]
                    weights[batch_idx, pos_idx] = weight_factors.get(term_type, 1.1)
    
    # Sử dụng CrossEntropyLoss để tính loss theo từng token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    token_losses = token_losses.view(labels.shape)
    
    # Áp dụng trọng số và tính trung bình
    weighted_token_losses = token_losses * weights
    active_elements = (labels != -100)
    
    if active_elements.any():
        weighted_loss = weighted_token_losses[active_elements].mean()
        return weighted_loss
    else:
        # Trả về loss 0 nếu không có element nào active
        return standard_loss * 0.0