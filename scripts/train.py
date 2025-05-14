"""
Script huấn luyện mô hình Whisper medical
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from models.whisper_medical import WhisperMedical
from data_utils.dataloader import WhisperMedicalDataset, WhisperDataCollator
from trainers.medical_trainer import WhisperMedicalTrainer
from data_utils.data_processor import load_bias_words, load_jsonl, create_prompt
from utils.medical_utils import generate_random_prompts
from config.config import (
    MODEL_SAVE_DIR, TRAIN_JSONL, DEV_JSONL, TRAIN_AUDIO_DIR, DEV_AUDIO_DIR, 
    BIAS_WORDS_FILE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, FP16,
    RANDOM_CONTEXT_PROB, RANDOM_CONTEXTS_SIZE, WEIGHT_FACTORS
)
from transformers import TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model cho y tế")
    parser.add_argument("--train_jsonl", type=str, default=TRAIN_JSONL, help="Đường dẫn đến file JSONL train data")
    parser.add_argument("--dev_jsonl", type=str, default=DEV_JSONL, help="Đường dẫn đến file JSONL validation data")
    parser.add_argument("--train_audio_dir", type=str, default=TRAIN_AUDIO_DIR, help="Thư mục chứa audio train")
    parser.add_argument("--dev_audio_dir", type=str, default=DEV_AUDIO_DIR, help="Thư mục chứa audio dev")
    parser.add_argument("--bias_words_file", type=str, default=BIAS_WORDS_FILE, help="Đường dẫn đến file bias words")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục để lưu mô hình")
    parser.add_argument("--model_name", type=str, default="whisper-medical", help="Tên mô hình")
    parser.add_argument("--no_bias_words", action="store_true", help="Không sử dụng bias words")
    parser.add_argument("--no_weighted_loss", action="store_true", help="Không sử dụng weighted loss")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Số epochs")
    
    args = parser.parse_args()
    
    # Thiết lập output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(MODEL_SAVE_DIR, args.model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Khởi tạo mô hình
    whisper_medical = WhisperMedical()
    
    # Chuẩn bị bias words
    bias_words_string = None if args.no_bias_words else load_bias_words(args.bias_words_file)
    
    # Tạo datasets
    train_dataset = WhisperMedicalDataset(
        args.train_jsonl, 
        whisper_medical.processor, 
        args.train_audio_dir,
        bias_words_string
    )
    
    val_dataset = WhisperMedicalDataset(
        args.dev_jsonl, 
        whisper_medical.processor, 
        args.dev_audio_dir,
        bias_words_string
    )
    
    # Tạo medical terms mapping nếu sử dụng weighted loss
    medical_terms_mapping = None
    if not args.no_weighted_loss:
        medical_terms_mapping = whisper_medical.create_medical_terms_mapping(args.bias_words_file)
    
    # Chuẩn bị random contexts cho context perturbation
    random_contexts = generate_random_prompts(
        args.train_jsonl, 
        args.bias_words_file, 
        RANDOM_CONTEXTS_SIZE
    )
    
    # Lưu tham số huấn luyện
    training_config = {
        "train_jsonl": args.train_jsonl,
        "dev_jsonl": args.dev_jsonl,
        "train_audio_dir": args.train_audio_dir,
        "dev_audio_dir": args.dev_audio_dir,
        "bias_words_file": args.bias_words_file,
        "no_bias_words": args.no_bias_words,
        "no_weighted_loss": args.no_weighted_loss,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "use_context_perturbation": True,
        "random_context_prob": RANDOM_CONTEXT_PROB,
        "weight_factors": WEIGHT_FACTORS
    }
    
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    
    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=FP16,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Khởi tạo trainer
    trainer = WhisperMedicalTrainer(
        random_context_prob=RANDOM_CONTEXT_PROB,
        random_contexts=random_contexts,
        medical_terms_mapping=medical_terms_mapping,
        weight_factors=WEIGHT_FACTORS,
        model=whisper_medical.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=WhisperDataCollator(whisper_medical.processor),
        tokenizer=whisper_medical.processor.tokenizer,
    )
    
    # Huấn luyện mô hình
    trainer.train()
    
    # Lưu mô hình cuối cùng
    whisper_medical.model = trainer.model
    whisper_medical.save(args.output_dir)
    
    print(f"Đã hoàn thành huấn luyện và lưu mô hình vào {args.output_dir}")

if __name__ == "__main__":
    main()