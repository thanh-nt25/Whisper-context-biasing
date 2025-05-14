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
    
    # Kiểm tra kích thước của các tensor
    batch_size = labels.shape[0]
    
    # Forward pass - KHÔNG sử dụng labels trong forward pass
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
    for batch_idx in range(batch_size):
        for pos_idx in range(labels.shape[1]):
            if labels[batch_idx, pos_idx] != -100:  # Bỏ qua token padding
                token = labels[batch_idx, pos_idx].item()
                if token in medical_terms_mapping:
                    term_type = medical_terms_mapping[token]
                    weights[batch_idx, pos_idx] = weight_factors.get(term_type, 1.1)
    
    # Tính loss thủ công (không sử dụng loss tự động của mô hình)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Giải quyết vấn đề kích thước không khớp
    # logits shape: [batch_size, sequence_length, vocab_size]
    # labels shape: [batch_size, target_length]
    
    # Lấy phần cuối cùng của logits có độ dài bằng labels
    if logits.size(1) > labels.size(1):
        # Nếu logits dài hơn, lấy phần cuối cùng
        logits = logits[:, -labels.size(1):, :]
    elif logits.size(1) < labels.size(1):
        # Nếu labels dài hơn, cắt bớt labels
        labels = labels[:, :logits.size(1)]
        weights = weights[:, :logits.size(1)]
    
    # Tính loss
    # Reshape để tính loss: [batch_size * seq_len, vocab_size]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    
    # Chỉ tính loss cho non-padding tokens
    active_loss = labels_flat != -100
    
    if active_loss.sum() > 0:  # Kiểm tra nếu có token nào để tính loss
        active_logits = logits_flat[active_loss]
        active_labels = labels_flat[active_loss]
        
        # Tính loss chỉ trên các active tokens
        token_loss = loss_fct(active_logits, active_labels)
        
        # Reshape trọng số cho active tokens
        weights_flat = weights.reshape(-1)[active_loss]
        
        # Nhân loss với trọng số và lấy trung bình
        weighted_loss = (token_loss * weights_flat).mean()
    else:
        # Nếu không có token nào để tính loss
        weighted_loss = torch.tensor(0.0, device=logits.device)
    
    return weighted_loss