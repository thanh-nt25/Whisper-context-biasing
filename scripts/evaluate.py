"""
Script đánh giá mô hình Whisper medical
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from models.whisper_medical import WhisperMedical
from utils.evaluation import evaluate_model
from config.config import (
    MODEL_SAVE_DIR, DEV_JSONL, DEV_AUDIO_DIR, BIAS_WORDS_FILE
)

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    parser.add_argument("--test_jsonl", type=str, default=DEV_JSONL, help="Đường dẫn đến file JSONL test data")
    parser.add_argument("--test_audio_dir", type=str, default=DEV_AUDIO_DIR, help="Thư mục chứa audio test")
    parser.add_argument("--bias_words_file", type=str, default=BIAS_WORDS_FILE, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--no_bias_words", action="store_true", help="Không sử dụng bias words")
    parser.add_argument("--num_samples", type=int, default=None, help="Số lượng mẫu cần đánh giá")
    
    args = parser.parse_args()
    
    # Thiết lập output file
    
    if args.model_dir.startswith("openai/"):
        whisper_medical = WhisperMedical(model_id=args.model_dir, freeze_encoder=False)
    else:
        whisper_medical = WhisperMedical()
        whisper_medical.load(args.model_dir)

    print(f"Evaluating model from {args.model_dir} on {args.test_jsonl}")
    evaluation_results = evaluate_model(
        whisper_medical, 
        args.test_jsonl,
        args.test_audio_dir,
        None if args.no_bias_words or args.model_dir.startswith("openai/") else args.bias_words_file,
        args.num_samples
    )
    
    
    # Lưu kết quả đánh giá
    # with open(args.output, 'w') as f:
    #     # Đảm bảo detailed_results không được ghi ra file (quá lớn)
    #     results_to_save = {k: v for k, v in evaluation_results.items() if k != 'detailed_results'}
    #     json.dump(results_to_save, f, indent=2)
    
    # # Lưu detailed_results riêng nếu cần
    # detailed_output = args.output.replace('.json', '_detailed.json')
    # with open(detailed_output, 'w') as f:
    #     json.dump(evaluation_results['detailed_results'], f, indent=2)
    
    # Hiển thị kết quả tổng quan
    print("\nKết quả đánh giá:")
    print(f"WER không description: {evaluation_results['wer']['no_description']:.4f}")
    # print(f"WER với description: {evaluation_results['wer']['with_description']:.4f}")
    # print(f"Cải thiện: {evaluation_results['wer']['improvement']:.4f} ({evaluation_results['wer']['improvement_percentage']:.2f}%)")
    
    # print("\nĐộ chính xác nhận dạng thuật ngữ y tế:")
    # print("Không description:")
    # print(f"  Precision: {evaluation_results['medical_terms_no_desc']['precision']:.4f}")
    # print(f"  Recall: {evaluation_results['medical_terms_no_desc']['recall']:.4f}")
    # print(f"  F1: {evaluation_results['medical_terms_no_desc']['f1']:.4f}")
    
    # print("Với description:")
    # print(f"  Precision: {evaluation_results['medical_terms_with_desc']['precision']:.4f}")
    # print(f"  Recall: {evaluation_results['medical_terms_with_desc']['recall']:.4f}")
    # print(f"  F1: {evaluation_results['medical_terms_with_desc']['f1']:.4f}")
    
    # print(f"\nKết quả chi tiết đã được lưu vào {args.output} và {detailed_output}")
    if args.output is None:
        eval_dir = os.path.join(os.path.dirname(args.model_dir), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        model_name = os.path.basename(args.model_dir)
        args.output = os.path.join(eval_dir, f"{model_name}_evaluation.json")
        
    wer_txt_path = args.output.replace('.json', '_wer.txt')
    with open(wer_txt_path, 'w') as f:
        f.write(f"WER (no description): {evaluation_results['wer']['no_description']:.4f}\n")

if __name__ == "__main__":
    main()