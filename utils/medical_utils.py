"""
Các hàm tiện ích cho dữ liệu y tế
"""

import re
import pandas as pd
import os
import sys
from pathlib import Path
import random

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from data_utils.data_processor import load_jsonl, load_bias_words, create_prompt

def extract_medical_terms_from_bias_list(text, bias_words_file):
    """
    Trích xuất các thuật ngữ y tế từ văn bản dựa trên bias list
    
    Args:
        text: Văn bản cần trích xuất
        bias_words_file: Đường dẫn đến file chứa bias words
    
    Returns:
        Danh sách các thuật ngữ được tìm thấy
    """
    results = []
    
    # Chuẩn hóa text
    text = text.lower()
    
    # Đọc bias words
    with open(bias_words_file, 'r', encoding='utf-8') as f:
        bias_words = [line.strip().lower() for line in f if line.strip()]
    
    # Tìm kiếm từng thuật ngữ
    for term in bias_words:
        # Tìm kiếm từng thuật ngữ trong văn bản
        if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            results.append(term)
    
    return results

def generate_random_prompts(jsonl_file, bias_words_file, num_prompts=50):
    """
    Tạo danh sách các prompts ngẫu nhiên cho context perturbation
    
    Args:
        jsonl_file: Đường dẫn đến file JSONL chứa data
        bias_words_file: Đường dẫn đến file bias words
        num_prompts: Số lượng prompts cần tạo
    
    Returns:
        Danh sách các prompt ngẫu nhiên
    """
    # Đọc dữ liệu
    data = load_jsonl(jsonl_file)
    bias_words_string = load_bias_words(bias_words_file)
    
    # Lấy ngẫu nhiên các mẫu
    sampled_data = random.sample(data, min(num_prompts, len(data)))
    
    # Tạo prompts
    random_prompts = [
        create_prompt(item['description'], bias_words_string)
        for item in sampled_data
    ]
    
    return random_prompts

def analyze_medical_errors(references, predictions, bias_words_file):
    """
    Phân tích lỗi nhận dạng các thuật ngữ y tế
    
    Args:
        references: Danh sách các transcripts tham chiếu
        predictions: Danh sách các transcripts dự đoán
        bias_words_file: Đường dẫn đến file bias words
    
    Returns:
        DataFrame phân tích lỗi
    """
    error_analysis = []
    
    for ref, pred in zip(references, predictions):
        # Trích xuất thuật ngữ y tế từ reference và prediction
        ref_terms = extract_medical_terms_from_bias_list(ref, bias_words_file)
        pred_terms = extract_medical_terms_from_bias_list(pred, bias_words_file)
        
        # Tìm thuật ngữ đúng
        correct_terms = set(ref_terms) & set(pred_terms)
        
        # Tìm thuật ngữ bị bỏ sót
        missed_terms = set(ref_terms) - set(pred_terms)
        
        # Tìm thuật ngữ sai (dự đoán nhưng không có trong reference)
        incorrect_terms = set(pred_terms) - set(ref_terms)
        
        # Thêm vào kết quả
        for term in missed_terms:
            error_analysis.append({
                'reference': ref,
                'prediction': pred,
                'term': term,
                'error_type': 'missed'
            })
        
        for term in incorrect_terms:
            error_analysis.append({
                'reference': ref,
                'prediction': pred,
                'term': term,
                'error_type': 'incorrect'
            })
    
    return pd.DataFrame(error_analysis)