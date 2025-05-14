import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from config.config import BIAS_WORDS_FILE

def load_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_bias_words(bias_file=BIAS_WORDS_FILE):
    if not os.path.exists(bias_file):
        print(f"Warning: Bias words file not found at {bias_file}")
        return ""
    
    with open(bias_file, 'r', encoding='utf-8') as f:
        bias_words = [line.strip() for line in f if line.strip()]
    
    return ", ".join(bias_words)

def get_audio_path(file_name, base_dir):
    audio_path = os.path.join(base_dir, file_name)
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found at {audio_path}")
    
    return audio_path

def create_prompt(description, bias_words_string):
    return f"<SOP> {description} Medical terms: {bias_words_string}. <SOT>"

def generate_medical_terms_mapping_from_file(tokenizer, bias_file=BIAS_WORDS_FILE):
    """
    Tạo mapping từ token_id sang loại thuật ngữ y tế từ file bias words
    
    Args:
        tokenizer: Tokenizer của Whisper
        bias_file: Đường dẫn đến file chứa bias words
        
    Note: Cấu trúc file bias words định dạng:
        - 15 dòng đầu: thuật ngữ DRUGCHEMICAL
        - 5 dòng tiếp theo: thuật ngữ DIAGNOSTICS
        - 5 dòng tiếp theo: thuật ngữ MEDDEVICETECHNIQUE
        - 5 dòng cuối: distractor (không được đưa vào mapping)
    
    Returns:
        Dictionary ánh xạ từ token_id sang loại của thuật ngữ y tế
    """
    medical_terms_mapping = {}
    
    with open(bias_file, 'r', encoding='utf-8') as f:
        bias_words = [line.strip().lower() for line in f if line.strip()]
    
    if len(bias_words) < 30:
        print(f"Warning: Bias words file has fewer than 30 terms ({len(bias_words)} found)")
    
    drugchemical_terms = bias_words[:15] 
    diagnostics_terms = bias_words[15:20]
    meddevicetechnique_terms = bias_words[20:25]
    # distractor_terms = bias_words[25:30]  # 5 từ cuối (không sử dụng trong mapping)
    
    print(f"Loaded terms - DRUGCHEMICAL: {len(drugchemical_terms)}, "
          f"DIAGNOSTICS: {len(diagnostics_terms)}, "
          f"MEDDEVICETECHNIQUE: {len(meddevicetechnique_terms)}, "
          f"Distractor: {len(bias_words) - len(drugchemical_terms) - len(diagnostics_terms) - len(meddevicetechnique_terms)}")
    
    for term in drugchemical_terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        for token in tokens:
            medical_terms_mapping[token] = "DRUGCHEMICAL"
    
    for term in diagnostics_terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        for token in tokens:
            medical_terms_mapping[token] = "DIAGNOSTICS"
    
    for term in meddevicetechnique_terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        for token in tokens:
            medical_terms_mapping[token] = "MEDDEVICETECHNIQUE"
    
    
    return medical_terms_mapping