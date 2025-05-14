"""
Script chính để chạy toàn bộ quy trình
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ quy trình fine-tuning Whisper cho y tế")
    parser.add_argument("--train_jsonl", type=str, default="data/all_text_train_description.jsonl", 
                        help="Đường dẫn đến file JSONL train data")
    parser.add_argument("--dev_jsonl", type=str, default="data/all_text_dev_description.jsonl",
                        help="Đường dẫn đến file JSONL dev data")
    parser.add_argument("--train_audio_dir", type=str, default="dataset/medical/train",
                        help="Thư mục chứa audio train")
    parser.add_argument("--dev_audio_dir", type=str, default="dataset/medical/dev",
                        help="Thư mục chứa audio dev")
    parser.add_argument("--bias_words_file", type=str, default="data/Blist/bias_list_30.txt",
                        help="Đường dẫn đến file bias words")
    parser.add_argument("--model_name", type=str, default="whisper-medical", help="Tên mô hình")
    parser.add_argument("--no_bias_words", action="store_true", help="Không sử dụng bias words")
    parser.add_argument("--no_weighted_loss", action="store_true", help="Không sử dụng weighted loss")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Số epochs")
    parser.add_argument("--skip_training", action="store_true", help="Bỏ qua bước huấn luyện")
    parser.add_argument("--skip_evaluation", action="store_true", help="Bỏ qua bước đánh giá")
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="Thư mục chứa mô hình đã huấn luyện (cho đánh giá)")
    
    args = parser.parse_args()
    
    # Đường dẫn thư mục
    model_dir = args.model_dir or os.path.join("output/models", args.model_name)
    
    # Huấn luyện mô hình
    if not args.skip_training:
        print("\n==== Huấn luyện mô hình ====")
        
        cmd = [
            "python", "scripts/train.py",
            "--train_jsonl", args.train_jsonl,
            "--dev_jsonl", args.dev_jsonl,
            "--train_audio_dir", args.train_audio_dir,
            "--dev_audio_dir", args.dev_audio_dir,
            "--bias_words_file", args.bias_words_file,
            "--model_name", args.model_name,
            "--learning_rate", str(args.learning_rate),
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs)
        ]
        
        if args.no_bias_words:
            cmd.append("--no_bias_words")
            
        if args.no_weighted_loss:
            cmd.append("--no_weighted_loss")
            
        subprocess.run(cmd)
    
    # Đánh giá mô hình
    if not args.skip_evaluation:
        print("\n==== Đánh giá mô hình ====")
        
        cmd = [
            "python", "scripts/evaluate.py",
            "--model_dir", model_dir,
            "--test_jsonl", args.dev_jsonl,
            "--test_audio_dir", args.dev_audio_dir,
            "--bias_words_file", args.bias_words_file
        ]
        
        if args.no_bias_words:
            cmd.append("--no_bias_words")
            
        subprocess.run(cmd)
    
    print(f"\nQuy trình đã hoàn tất. Mô hình được lưu tại: {model_dir}")

if __name__ == "__main__":
    main()