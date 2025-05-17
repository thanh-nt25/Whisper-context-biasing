"""
Hàm đánh giá cho Whisper medical fine-tuning
"""
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

import os
import gc
from jiwer import wer

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.medical_utils import extract_medical_terms_from_bias_list
from data_utils.data_processor import get_audio_path, load_bias_words
from utils.compute_metric import BasicTextNormalizer

def calculate_wer(references, predictions):
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if ref.strip() and pred is not None]
    
    if not valid_pairs:
        return 1.0, [1.0] * len(references)
    
    refs, preds = zip(*valid_pairs)
    
    overall_wer = jiwer.wer(refs, preds)
    
    return overall_wer

def compute_metrics_whisper_with_prompt(eval_preds, tokenizer, prompt_ids_list=None):
    """
    Tính metrics cho model với prompts
    
    Args:
        eval_preds: Tuple (logits, labels) từ evaluation
        tokenizer: Tokenizer để decode predictions
        prompt_ids_list: List các prompt IDs (để loại bỏ phần prompt từ predictions)
    
    Returns:
        Dictionary với metrics
    """
    # Giải phóng bộ nhớ
    import gc
    import torch
    
    print("Trigger compute_metrics_whisper_with_prompt!")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    logits, labels = eval_preds
    
    # Bỏ qua padding token trong labels
    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    
    # Chuyển logits thành predictions
    predictions = logits.argmax(axis=-1)
    
    # Nếu có prompt_ids_list, bỏ đi phần prompt từ predictions
    if prompt_ids_list:
        batch_size = min(len(predictions), len(prompt_ids_list))
        for i in range(batch_size):
            if i < len(prompt_ids_list):
                prompt_len = len(prompt_ids_list[i])
                # Loại bỏ phần prompt từ predictions và labels
                if prompt_len < len(predictions[i]):
                    predictions[i] = predictions[i][prompt_len:]
                if prompt_len < len(labels[i]):
                    labels[i] = labels[i][prompt_len:]
    
    # Xử lý batch để tránh OOM
    batch_size = 16
    total = len(predictions)
    references = []
    decoded_preds = []
    
    for i in range(0, total, batch_size):
        batch_pred = predictions[i:min(i+batch_size, total)]
        batch_label = labels[i:min(i+batch_size, total)]
        
        # Decode predictions và labels
        pred_str = tokenizer.batch_decode(batch_pred, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(batch_label, skip_special_tokens=True)
        
        
        # Normalize
        normalizer = BasicTextNormalizer()
        pred_str = [normalizer(s) for s in pred_str]
        label_str = [normalizer(s) for s in label_str]
        # pred_str = [s.strip().lower() for s in pred_str]
        # label_str = [s.strip().lower() for s in label_str]
        
        references.extend(label_str)
        decoded_preds.extend(pred_str)
        
        # Giải phóng bộ nhớ
        del batch_pred, batch_label
        gc.collect()
    
    # Lưu kết quả ra file
    result_dir = os.path.join("/kaggle/working", "results")
    os.makedirs(result_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, "refs_and_preds.txt"), "w", encoding="utf-8") as f:
        for ref, pred in zip(references, decoded_preds):
            f.write(f"Ref: {ref}\n")
            f.write(f"Pred: {pred}\n\n")
    
    # Tính WER
    wer = calculate_wer(references, decoded_preds)
    print(f"Base WER on utils evaluation: {wer:.4f}")
    
    # Tính thêm CER
    # cer = jiwer.cer(references, decoded_preds)
    
    # Phân tích lỗi cụ thể cho thuật ngữ y tế
    return {
        "wer": wer,
    }


def compute_metrics_whisper_baseline(eval_preds, tokenizer, result_dir="/kaggle/working/results"):
    """
    Tính WER cho baseline Whisper mà không có prompt injection

    Args:
        eval_preds: Tuple (logits, labels)
        tokenizer: Whisper tokenizer
        result_dir: Thư mục lưu kết quả tham chiếu và dự đoán

    Returns:
        Dictionary chứa WER
    """
    print("\n\n Triggered compute_metrics_whisper_baseline()")
    gc.collect()

    # pred_ids = eval_preds.predictions
    # label_ids = eval_preds.label_ids
    # normalizer = BasicTextNormalizer()
    # tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="en", task="transcribe")
    
    # logits, labels = eval_preds
    print(type(eval_preds))
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    
    # pred_ids = logits.argmax(axis=-1)
    
    label_ids = labels.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    normalizer = BasicTextNormalizer()

    batch_size = 4
    total = len(pred_ids)
    results = []

    for i in tqdm(range(0, total, batch_size), desc="Decoding & Computing WER"):
        batch_pred_ids = pred_ids[i:i + batch_size]
        batch_label_ids = label_ids[i:i + batch_size]

        pred_strs = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(batch_label_ids, skip_special_tokens=True)

        for pred, label in zip(pred_strs, label_strs):
            if label.strip() and label != "ignore_time_segment_in_scoring":
                results.append((normalizer(label), normalizer(pred)))

    # Ghi kết quả
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "refs_and_preds.txt"), "w", encoding="utf-8") as f:
        for ref, pred in results:
            f.write(f"Ref: {ref}\n")
            f.write(f"Pred: {pred}\n\n")

    if not results:
        print("Warning: No valid samples for WER calculation.")
        return {"wer": 100.0}

    references = [r for r, _ in results]
    predictions = [p for _, p in results]
    total_wer = 100 * wer(references, predictions)

    print(f"Base WER: {total_wer:.2f}%")
    return {"wer": total_wer}
    
    # new wer
    # print(f"Length of pred_ids: {len(pred_ids)}")
    # cutted_pred_ids = pred_ids
    # cutted_label_ids = label_ids
    
    # for i in tqdm(range(0, len(cutted_pred_ids), batch_size)):
    #     batch_pred_ids = cutted_pred_ids[i : i + batch_size]
    #     batch_label_ids = cutted_label_ids[i : i + batch_size]

    #     pre_strs = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
    #     label_strs = tokenizer.batch_decode(batch_label_ids, skip_special_tokens=True)

    #     filtered_pre_strs = []
    #     filtered_label_strs = []

    #     for pred, label in zip(pre_strs, label_strs):
    #         if label != "ignore_time_segment_in_scoring" and label.strip() != "":
    #             # Skip empty references and 'ignore_time_segment_in_scoring'
    #             filtered_pre_strs.append(normalizer(pred))
    #             filtered_label_strs.append(normalizer(label))

    #     # Only add valid pairs to results
    #     if filtered_pre_strs and filtered_label_strs:
    #         results.extend(zip(filtered_label_strs, filtered_pre_strs))

    # with open(
    #     os.path.join("/kaggle/working", "refs_and_pred.txt"), "w", encoding="utf-8"
    # ) as f:
    #     for ref, pred in results:
    #         f.write(f"Ref:{ref}\n")
    #         f.write(f"Pred:{pred}\n\n")

    # if not results:
    #     print("Warning: No valid samples for WER calculation")
    #     return {"wer": 100.0}  # Worst possible WER

    # pre_strs = [pred for _, pred in results]
    # label_strs = [ref for ref, _ in results]
    # total_wer = 100 * metric.compute(predictions=pre_strs, references=label_strs)

    # return {"wer": total_wer}


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
  
