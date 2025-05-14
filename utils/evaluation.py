"""
Hàm đánh giá cho Whisper medical fine-tuning
"""

import jiwer
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.medical_utils import extract_medical_terms_from_bias_list
from data_utils.data_processor import get_audio_path, load_bias_words

def calculate_wer(references, predictions):
    """
    Tính Word Error Rate (WER)
    
    Args:
        references: Danh sách các văn bản tham chiếu
        predictions: Danh sách các văn bản dự đoán
    
    Returns:
        WER tổng thể và WER cho từng mẫu
    """
    # Tính WER tổng thể
    overall_wer = jiwer.wer(references, predictions)
    
    # Tính WER cho từng mẫu
    # individual_wers = [
    #     jiwer.wer([ref], [pred]) 
    #     for ref, pred in zip(references, predictions)
    # ]
    
    # return overall_wer, individual_wers
    return overall_wer

def calculate_medical_term_accuracy(references, predictions, bias_words_file):
    """
    Tính độ chính xác trong nhận dạng thuật ngữ y tế
    
    Args:
        references: Danh sách các văn bản tham chiếu
        predictions: Danh sách các văn bản dự đoán
        bias_words_file: Đường dẫn đến file bias words
    
    Returns:
        Dictionary chứa precision, recall, F1
    """
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    for ref, pred in zip(references, predictions):
        # Trích xuất thuật ngữ y tế
        ref_terms = extract_medical_terms_from_bias_list(ref, bias_words_file)
        pred_terms = extract_medical_terms_from_bias_list(pred, bias_words_file)
        
        # True Positive: từ có trong cả ref và pred
        tp += len(set(ref_terms) & set(pred_terms))
        
        # False Positive: từ có trong pred nhưng không có trong ref
        fp += len(set(pred_terms) - set(ref_terms))
        
        # False Negative: từ có trong ref nhưng không có trong pred
        fn += len(set(ref_terms) - set(pred_terms))
    
    # Tính precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Tính recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Tính F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# def evaluate_model(model, jsonl_file, audio_dir, bias_words_file, num_samples=None):
#     """
#     Đánh giá mô hình trên tập dữ liệu
    
#     Args:
#         model: WhisperMedical model
#         jsonl_file: Đường dẫn đến file JSONL chứa data
#         audio_dir: Thư mục chứa audio
#         bias_words_file: Đường dẫn đến file bias words
#         num_samples: Số lượng mẫu cần đánh giá (None = tất cả)
    
#     Returns:
#         Dictionary chứa kết quả đánh giá
#     """
#     from data_utils.data_processor import load_jsonl
    
#     # Đọc dữ liệu
#     data = load_jsonl(jsonl_file)
#     if num_samples:
#         data = data[:num_samples]
    
#     # Đọc bias words
#     bias_words_string = load_bias_words(bias_words_file)
    
#     references = []
#     predictions = []
#     predictions_with_description = []
    
#     for item in tqdm(data, desc="Evaluating"):
#         file_name = item['file']
#         transcript = item['text']
#         description = item['description']
        
#         # Lấy đường dẫn audio
#         audio_path = get_audio_path(file_name, audio_dir)
        
#         # Thêm transcript vào danh sách tham chiếu
#         references.append(transcript)
        
#         # Nhận dạng không có description
#         try:
#             pred_no_desc = model.transcribe(audio_path)
#             predictions.append(pred_no_desc)
#         except Exception as e:
#             print(f"Error in transcribing without description: {e}")
#             predictions.append("ERROR: Could not transcribe")
        
#         # Nhận dạng có description và bias words
#         try:
#             pred_with_desc = model.transcribe(audio_path, description, bias_words_string)
#             predictions_with_description.append(pred_with_desc)
#         except Exception as e:
#             print(f"Error in transcribing with description: {e}")
#             predictions_with_description.append("ERROR: Could not transcribe")
    
#     # Tính WER
#     wer_no_desc, individual_wers_no_desc = calculate_wer(references, predictions)
#     wer_with_desc, individual_wers_with_desc = calculate_wer(references, predictions_with_description)
    
#     # Tính độ chính xác nhận dạng thuật ngữ y tế
#     med_metrics_no_desc = calculate_medical_term_accuracy(references, predictions, bias_words_file)
#     med_metrics_with_desc = calculate_medical_term_accuracy(references, predictions_with_description, bias_words_file)
    
#     # Tạo báo cáo đánh giá chi tiết
#     detailed_results = []
#     for i, (ref, pred_no, pred_with) in enumerate(zip(references, predictions, predictions_with_description)):
#         wer_no = individual_wers_no_desc[i]
#         wer_with = individual_wers_with_desc[i]
        
#         detailed_results.append({
#             "reference": ref,
#             "prediction_no_desc": pred_no,
#             "prediction_with_desc": pred_with,
#             "wer_no_desc": wer_no,
#             "wer_with_desc": wer_with,
#             "wer_improvement": wer_no - wer_with,
#             "file_name": data[i]['file']
#         })
    
#     # Tạo báo cáo đánh giá tổng thể
#     evaluation_results = {
#         "wer": {
#             "no_description": wer_no_desc,
#             "with_description": wer_with_desc,
#             "improvement": wer_no_desc - wer_with_desc,
#             "improvement_percentage": (wer_no_desc - wer_with_desc) / wer_no_desc * 100 if wer_no_desc > 0 else 0
#         },
#         "medical_terms_no_desc": med_metrics_no_desc,
#         "medical_terms_with_desc": med_metrics_with_desc,
#         "detailed_results": detailed_results
#     }
    
#     return evaluation_results
  
def evaluate_model(model, jsonl_file, audio_dir, bias_words_file, num_samples=None, batch_size=16):
    """
    Đánh giá mô hình trên tập dữ liệu

    Args:
        model: WhisperMedical model (đã có transcribe_batch)
        jsonl_file: Đường dẫn đến file JSONL chứa data
        audio_dir: Thư mục chứa audio
        bias_words_file: Đường dẫn đến file bias words
        num_samples: Số lượng mẫu cần đánh giá (None = tất cả)
        batch_size: Kích thước batch để đánh giá

    Returns:
        Dictionary chứa kết quả đánh giá
    """
    from data_utils.data_processor import load_jsonl

    # Đọc dữ liệu
    data = load_jsonl(jsonl_file)
    if num_samples:
        data = data[:num_samples]

    # Đọc bias words
    bias_words_string = load_bias_words(bias_words_file)

    references = []
    audio_paths = []
    descriptions = []

    for item in data:
        file_name = item['file']
        transcript = item['text']
        description = item['description']
        audio_path = get_audio_path(file_name, audio_dir)

        references.append(transcript)
        audio_paths.append(audio_path)
        descriptions.append(description)

    predictions = []
    # Dự đoán theo batch không có description
    print(f"Transcribing {len(audio_paths)} samples without description...")
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Evaluating"):
        batch_audio_paths = audio_paths[i:i+batch_size]
        try:
            preds = model.transcribe_batch(batch_audio_paths)
        except Exception as e:
            preds = ["ERROR"] * len(batch_audio_paths)
            print(f"Error during batch inference: {e}")
        predictions.extend(preds)

    # Tính WER
    print(f"Calculating WER on {jsonl_file} with no description")
    wer_no_desc = calculate_wer(references, predictions)

    evaluation_results = {
        "wer": {
            "no_description": wer_no_desc
        }
    }

    return evaluation_results
