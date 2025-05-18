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
    args.random = True # 5% random prompt
    args.basic = False
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-base.en')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    
    data_collator = DataCollatorSpeechS2SWhitPadding(processor=processor)
    
    # "/kaggle/input/medical-syn-med-test/medical-united-syn-med-test"
    data_root = os.path.abspath("data")
    data_dir = "medical-united-syn-med-test"
    
    print("Starting loading data!")
    data_test = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)    

    print(len(data_test))    
    sample = data_test[0]

    print("Input features shape:", sample["input_features"].shape)
    print("Input features dtype:", sample["input_features"].dtype)
    print("Input feature length:", sample["input_features"].shape[1])
    print("Input features:", sample["input_features"])
    print("Labels shape:", sample["labels"].shape)
    print("Labels:", sample["labels"])
    print("Prompt:", sample["prompt"])    
    
    label_ids = sample["labels"].tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(label_ids)
    print(tokens)

    
    # model = WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-base.en')
    
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    
