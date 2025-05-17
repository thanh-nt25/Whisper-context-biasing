"""
Script huấn luyện mô hình Whisper medical
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent.absolute()))
from models.whisper_medical import WhisperMedical
from data_utils.data_loader import WhisperMedicalDataset, WhisperDataCollator
from trainers.medical_trainer import WhisperMedicalTrainer
from data_utils.data_processor import load_bias_words, load_jsonl, create_prompt
from utils.medical_utils import generate_random_prompts
from config.config import (
    MODEL_SAVE_DIR, TRAIN_JSONL, DEV_JSONL, TRAIN_AUDIO_DIR, DEV_AUDIO_DIR, 
    BIAS_WORDS_FILE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, FP16,SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS,
    RANDOM_CONTEXT_PROB, RANDOM_CONTEXTS_SIZE, WEIGHT_FACTORS
)
from transformers import TrainingArguments
from utils.evaluation import compute_metrics_whisper_with_prompt


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
    parser.add_argument("--hf_token", type=str, default="", help="Hugging face token")
    
    args = parser.parse_args()
    
    
    if args.output_dir is None:
        args.output_dir = os.path.join(MODEL_SAVE_DIR, args.model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    
    whisper_medical = WhisperMedical()
    
    
    bias_words_string = None if args.no_bias_words else load_bias_words(args.bias_words_file)
    
    
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
    
    
    medical_terms_mapping = None
    if not args.no_weighted_loss:
        medical_terms_mapping = whisper_medical.create_medical_terms_mapping(args.bias_words_file)
    
    
    random_contexts = generate_random_prompts(
        args.train_jsonl, 
        args.bias_words_file, 
        RANDOM_CONTEXTS_SIZE
    )
    
    
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
    
    # eval_step = int((34358 // 2) // args.batch_size)
    # log_step = int((34358 // 50) // args.batch_size)
    # print(f"eval_step: {eval_step}, log_step: {log_step}")
    
    training_args = TrainingArguments(
        push_to_hub=True,
        hub_model_id="thanh-nt25/whisper-bias-medical-2",
        hub_token=args.hf_token,
        output_dir=args.output_dir,
        save_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="whisper-medical-biasing-2",
        dataloader_num_workers=4,
    )
    
    
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
        processing_class=whisper_medical.processor,
        compute_metrics=lambda eval_preds: compute_metrics_whisper_with_prompt(
            eval_preds,
            tokenizer=whisper_medical.processor.tokenizer,
            prompt_ids_list=[x["prompt_ids"].tolist() for x in val_dataset]
        ),
    )
    
    
    trainer.train()
    
    
    whisper_medical.model = trainer.model
    whisper_medical.save(args.output_dir)
    
    print(f"Đã hoàn thành huấn luyện và lưu mô hình vào {args.output_dir}")

if __name__ == "__main__":
    main()