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


def compute_metrics_whisper_baseline_debug(eval_preds, tokenizer, result_dir="/kaggle/working/results"):
    """
    Phiên bản debug của compute_metrics
    """
    print("\n=== DEBUG: compute_metrics_whisper_baseline called ===")
    
    try:
        print(f"Type of eval_preds: {type(eval_preds)}")
        print(f"Available attributes: {dir(eval_preds)}")
        
        # Kiểm tra predictions và labels
        print(f"Predictions type: {type(eval_preds.predictions)}")
        print(f"Predictions shape/length: {eval_preds.predictions.shape if hasattr(eval_preds.predictions, 'shape') else len(eval_preds.predictions)}")
        
        print(f"Labels type: {type(eval_preds.label_ids)}")
        print(f"Labels shape/length: {eval_preds.label_ids.shape if hasattr(eval_preds.label_ids, 'shape') else len(eval_preds.label_ids)}")
        
        # Kiểm tra một vài mẫu đầu tiên
        if len(eval_preds.predictions) > 0:
            print("\nSample predictions (first 2):")
            for i in range(min(2, len(eval_preds.predictions))):
                print(f"  Prediction {i} type: {type(eval_preds.predictions[i])}")
                print(f"  Shape/length: {eval_preds.predictions[i].shape if hasattr(eval_preds.predictions[i], 'shape') else len(eval_preds.predictions[i])}")
        
        if len(eval_preds.label_ids) > 0:
            print("\nSample labels (first 2):")
            for i in range(min(2, len(eval_preds.label_ids))):
                print(f"  Label {i} type: {type(eval_preds.label_ids[i])}")
                print(f"  Shape/length: {eval_preds.label_ids[i].shape if hasattr(eval_preds.label_ids[i], 'shape') else len(eval_preds.label_ids[i])}")
        
        # Tiếp tục với hàm gốc
        result = {
            "wer": 0.0  # Giá trị mặc định để tránh lỗi
        }
        
        print("Will process predictions and compute WER...")
        
        # Thư mục kết quả
        os.makedirs(result_dir, exist_ok=True)
        
        # Chuẩn bị file đầu ra
        refs_file = os.path.join(result_dir, "refs_debug.txt")
        preds_file = os.path.join(result_dir, "preds_debug.txt")
        
        with open(refs_file, "w", encoding="utf-8") as ref_f, \
             open(preds_file, "w", encoding="utf-8") as pred_f:
            
            normalizer = BasicTextNormalizer()
            valid_samples = 0
            wer_sum = 0
            
            # Xử lý từng mẫu
            for i in range(len(eval_preds.predictions)):
                if i < 5 or i % 50 == 0:
                    print(f"Processing sample {i+1}/{len(eval_preds.predictions)}")
                
                try:
                    # Xử lý nhãn
                    curr_label = eval_preds.label_ids[i].copy()
                    curr_label[curr_label == -100] = tokenizer.pad_token_id
                    
                    # Decode
                    pred_str = tokenizer.decode(eval_preds.predictions[i], skip_special_tokens=True)
                    label_str = tokenizer.decode(curr_label, skip_special_tokens=True)
                    
                    # Kiểm tra tính hợp lệ
                    if label_str.strip() and label_str != "ignore_time_segment_in_scoring":
                        # Chuẩn hóa
                        norm_pred = normalizer(pred_str)
                        norm_label = normalizer(label_str)
                        
                        # Tính WER
                        sample_wer = wer([norm_label], [norm_pred])
                        wer_sum += sample_wer
                        valid_samples += 1
                        
                        # Ghi ra file
                        ref_f.write(f"{norm_label}\n")
                        pred_f.write(f"{norm_pred}\n")
                        
                        # Log vài mẫu đầu tiên
                        if i < 5:
                            print(f"  Sample {i}:")
                            print(f"    Reference: {norm_label}")
                            print(f"    Prediction: {norm_pred}")
                            print(f"    WER: {sample_wer * 100:.2f}%")
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
        
        # Tính WER trung bình
        if valid_samples > 0:
            result["wer"] = (wer_sum / valid_samples) * 100
            print(f"Computed WER: {result['wer']:.2f}% (based on {valid_samples} samples)")
        else:
            result["wer"] = 100.0
            print("No valid samples for WER calculation")
        
        return result
        
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return {"wer": 100.0}
  
