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

from models.whisper_medical import WhisperMedicalForConditionalGeneration

from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding

from utils.compute_metric import compute_wer

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperConfig
)

from trainer.CustomTrainer import CustomTrainer

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    # parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    # parser.add_argument("--test_jsonl", type=str, required=True, help="Đường dẫn đến file JSONL test data")
    # parser.add_argument("--test_audio_dir", type=str, required=True, help="Thư mục chứa audio test")
    # parser.add_argument("--bias_words_file", type=str, required=True, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    # parser.add_argument("--compare_baseline", action="store_true", help="So sánh với Whisper cơ bản")
    # parser.add_argument("--debug_mode", action="store_true", help="Chạy ở chế độ debug chi tiết")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Đường dẫn đến file kết quả đánh giá")
    
    parser.add_argument("--prompt", action="store_true", help="whether to use prompt to decoder")
    parser.add_argument("--random", action="store_true", help="context perturbation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    args.prompt = True
    args.random = True # 5% random prompt
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-base.en')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        decoder_prev_token_id=tokenizer.convert_tokens_to_ids("<|startofprev|>"),

    )
    
    # "/kaggle/input/medical-syn-med-test/medical-united-syn-med-test"
    
    print("DATA_ROOT:", args.data_root)
    print("DATA_DIR:", args.data_dir)
    print("JSONL_DATA:", args.jsonl_data)
    
    print("Starting loading data!")
    # data_train = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.random)
    # data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,data_dir), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    data_test = PromptWhisperDataset(base_path=os.path.join(args.data_root, args.data_dir), jsonl_data=args.jsonl_data, phase='test', 
                                     feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt)    
    # sample = data_test[0]
    
    # print(sample['input_features'])
    # print(sample['labels'])
    
    # print(sample['input_features'].shape)
    # print(sample['labels'].shape)
    
    # print("Decoded labels:", tokenizer.decode(sample['labels'], skip_special_tokens=False))
    
    # print("pad_token:", tokenizer.pad_token)            # <|endoftext|>
    # print("pad_token_id:", tokenizer.pad_token_id)      # 50256
    # print("eos_token:", tokenizer.eos_token)            # <|endoftext|>
    # print("eos_token_id:", tokenizer.eos_token_id)      # 50256
    # print("tokenizer.decode([50258]): ", tokenizer.decode([50258]))
    # print("tokenizer.decode([50256]): ", tokenizer.decode([50256]))
    # print("tokenizer.decode([50257]): ", tokenizer.decode([50257]))
    # print("tokenizer.decode([50358]): ", tokenizer.decode([50358]))
    # print("tokenizer.decode([50362]): ", tokenizer.decode([50362]))
    # print("tokenizer.decode([0]): ", tokenizer.decode([0]))
    # print("tokenizer.decode([50359]): ", tokenizer.decode([50359]))
    # print("tokenizer.decode([50361]): ", tokenizer.decode([50361]))


    # config = WhisperConfig.from_pretrained("openai/whisper-base.en")
    # model = WhisperMedicalForConditionalGeneration(config)
    
    # here
    model = WhisperMedicalForConditionalGeneration.from_pretrained("openai/whisper-base.en", freeze_encoder=False)
    
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
        include_inputs_for_metrics=True,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=500,
        # save_total_limit=3,
        # load_best_model_at_end=True,
        report_to = []
    )
    
    trainer = CustomTrainer(
        args=training_args,
        model=model,
        # train_dataset=data_train,
        # eval_dataset=data_test,
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
    
    