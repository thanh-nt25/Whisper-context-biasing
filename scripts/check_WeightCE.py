import torch
import torch.nn.functional as F
from transformers import WhisperTokenizer

# Khởi tạo tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")

# Danh sách token ID đặc biệt của Whisper
WHISPER_SPECIAL_TOKENS = {50256, 50257, 50258, 50358, 50362}  # <|endoftext|>, <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>

# Hàm kiểm tra token đặc biệt
def is_special_token(token_id):
    return token_id in WHISPER_SPECIAL_TOKENS

# Hàm tính loss weighted cross-entropy
def compute_weighted_ce_loss(lm_logits, labels, bias_spans, bias_weight=1.5):
    """
    Compute weighted cross-entropy loss with bias word weighting, skipping special tokens.
    
    Args:
        lm_logits (torch.Tensor): Logits from model [batch_size, seq_len, vocab_size]
        labels (torch.Tensor): Target labels [batch_size, seq_len] with -100 for ignored tokens
        bias_spans (torch.Tensor): Bias spans [batch_size, max_n_spans, max_span_len]
        bias_weight (float): Weight for bias tokens
    
    Returns:
        tuple: (loss, weights, matches)
    """
    batch_size, seq_len, vocab_size = lm_logits.shape
    weights = torch.ones_like(labels, dtype=torch.float32)  # Shape: [batch_size, seq_len]
    matches = []  # Lưu các vị trí khớp bias_spans: [(sample_idx, start_pos, span_len)]

    # So khớp và gán weights cho bias spans
    for i in range(batch_size):
        sample_matches = []
        for span in bias_spans[i]:  # Mỗi span là danh sách token IDs
            if span.numel() == 0 or torch.all(span == 50256):  # 50256 là pad token
                continue
            span = span.tolist()
            span_len = len([tok for tok in span if tok != 50256])  # Độ dài thực của span
            if span_len == 0:
                continue
            span = span[:span_len]  # Loại bỏ padding trong span
            for j in range(seq_len - span_len + 1):
                if labels[i, j:j + span_len].tolist() == span:
                    # Gán weight chỉ cho các token không phải token đặc biệt
                    for k in range(span_len):
                        token = labels[i, j + k].item()
                        if not is_special_token(token):
                            weights[i, j + k] = bias_weight
                    sample_matches.append((j, span_len, span))
        matches.append(sample_matches)

    # Tính log probabilities
    log_probs = F.log_softmax(lm_logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    log_probs = log_probs.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    labels_flat = labels.view(-1)  # [batch_size * seq_len]
    weights_flat = weights.view(-1)  # [batch_size * seq_len]

    # Tính per-token loss
    per_token_loss = -log_probs[torch.arange(log_probs.size(0)), labels_flat]  # [batch_size * seq_len]
    valid_mask = labels_flat != -100  # Mask cho các token hợp lệ
    per_token_loss = per_token_loss * valid_mask.float()
    weights_flat = weights_flat * valid_mask.float()

    # Tính loss weighted
    weighted_loss = per_token_loss * weights_flat
    loss = weighted_loss.sum() / (valid_mask.sum() + 1e-8)  # Tránh chia cho 0
    return loss, weights, matches

# Dữ liệu mẫu từ input mới
def get_sample_data():
    # Sample 0: Labels và Bias Spans từ text
    text = "Rekool-L tab, which contains rabeprazole, helps alleviate symptoms of acid reflux and heartburn."
    bias_words = ["Rekool-L", "rabeprazole", "acid reflux", "heartburn"]

    # Tokenize text
    tokens = tokenizer(text).input_ids
    labels_0 = [-100] * 10 + tokens + [-999] * 10  # Thêm padding prompt và padding cuối
    labels_0 = labels_0[:76]  # Giới hạn độ dài như sample trước

    # Tokenize bias_words thành spans, bỏ qua token đặc biệt
    bias_spans_0 = []
    for bias_word in bias_words:
        # Tokenize và loại bỏ token đặc biệt
        bias_tokens = tokenizer(bias_word, add_special_tokens=False).input_ids
        bias_spans_0.append(bias_tokens)

    # Padding để đồng bộ độ dài labels
    max_len = 76  # Giữ độ dài cố định như sample trước
    labels_0 = labels_0 + [-100] * (max_len - len(labels_0)) if len(labels_0) < max_len else labels_0[:max_len]

    # Padding để đồng bộ độ dài bias_spans
    max_span_len = max(len(span) for span in bias_spans_0) if bias_spans_0 else 0
    bias_spans_0 = [span + [50256] * (max_span_len - len(span)) for span in bias_spans_0]

    # Kết hợp thành batch (chỉ 1 sample trong batch)
    labels = torch.tensor([labels_0])
    bias_spans = torch.tensor([bias_spans_0])

    # Tạo lm_logits giả lập
    vocab_size = 51865  # Theo Whisper base
    seq_len = labels.shape[1]
    lm_logits = torch.randn(1, seq_len, vocab_size)

    return lm_logits, labels, bias_spans

# Hàm chính để kiểm tra
def main():
    # Lấy dữ liệu mẫu
    lm_logits, labels, bias_spans = get_sample_data()

    # Tính loss và lấy weights, matches
    loss, weights, matches = compute_weighted_ce_loss(lm_logits, labels, bias_spans, bias_weight=1.5)
    print(f"Computed Loss: {loss.item():.6f}\n")

    # In bảng căn chỉnh cho từng sample với khoảng cách cột rộng hơn
    for sample_idx in range(len(matches)):
        print(f"\n=== Sample {sample_idx} ===")
        print(f"Bias Spans: {[tokenizer.decode(bias_spans[sample_idx][i], skip_special_tokens=True) for i in range(len(bias_spans[sample_idx]))]}")
        print(f"{'Position':<15} {'Label Token':<20} {'Decoded Label':<25} {'Weight':<15} {'Match':<30}")
        print("-" * 105)
        
        for pos in range(labels.shape[1]):
            token = labels[sample_idx, pos].item()
            decoded = tokenizer.decode([token], skip_special_tokens=False) if token != -999 else "N/A"
            weight = weights[sample_idx, pos].item()
            
            # Kiểm tra xem vị trí này có nằm trong matches không
            match_info = "No"
            for start_pos, span_len, span in matches[sample_idx]:
                if start_pos <= pos < start_pos + span_len:
                    match_info = f"Yes (Span: {tokenizer.decode(span)})"
                    break
            
            print(f"{pos:<15} {token:<20} {decoded:<25} {weight:<15.2f} {match_info:<30}")

if __name__ == "__main__":
    main()