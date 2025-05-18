"""
Script đánh giá mô hình Whisper medical
"""

import argparse
import os
import librosa
import torch
import gc
import json
from pathlib import Path
import sys
from jiwer import wer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechS2SWhitPadding, compute_wer

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    GenerationConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    # parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    # parser.add_argument("--test_jsonl", type=str, required=True, help="Đường dẫn đến file JSONL test data")
    # parser.add_argument("--test_audio_dir", type=str, required=True, help="Thư mục chứa audio test")
    # parser.add_argument("--bias_words_file", type=str, required=True, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--compare_baseline", action="store_true", help="So sánh với Whisper cơ bản")
    parser.add_argument("--debug_mode", action="store_true", help="Chạy ở chế độ debug chi tiết")
    
    parser.add_argument("--prompt", action="store_true", help="whether to use prompt to decoder")
    parser.add_argument("--random", action="store_true", help="context perturbation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    args.prompt = True
    args.random = False
    args.basic = False
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-base.en')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    
    data_collator = DataCollatorSpeechS2SWhitPadding(processor=processor)
    
    # "/kaggle/input/medical-syn-med-test/medical-united-syn-med-test"
    data_root = "/kaggle/input"
    data_dir = "/medical-syn-med-test/medical-united-syn-med-test"
    
    print("Starting loading data!")
    data_train = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.random)
    data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    data_test = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)    
    
    model = WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-base.en')
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    root_path = "results/"
    os.makedirs(os.path.join(root_path), exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(root_path, "models"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=500,
        # save_total_limit=3,
        # load_best_model_at_end=True,
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_train,
        eval_dataset=data_eval,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_wer,
    )

    if (len(data_test) == 0):
        print("No test data found!")
        exit(0)
    print("length of test data: ", len(data_test))

    print("Starting evaluation!")
    result = trainer.evaluate(data_test)
    print(result)
    
    