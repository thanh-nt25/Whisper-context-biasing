import argparse
import os
import librosa
import torch
import gc
import json
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.whisper_medical import WhisperForConditionalGenerationWeightCE
from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.compute_metric import compute_wer
from transformers import (
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperConfig
)
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation whisper medical")
    
    parser.add_argument("--output", type=str, default="/kaggle/working/results", help="Output directory for results")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight for weighted cross-entropy")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Base input data directory")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Subdirectory for audio files")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Path to JSONL metadata")
    parser.add_argument("--random", action="store_true", help="Apply context perturbation (5% random prompt)")
    parser.add_argument("--only_eval_bias_wer", action="store_true", help="param for eval bias_wer only")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face model ID (optional; defaults to openai/whisper-base.en)")
    # parser.add_argument("--refs_pred_file", type=str, required=True, default=None, help="Path to refs and pred")
    
    parser.add_argument("--prompt", action="store_true", help="Use only description prompt in decoder")
    parser.add_argument("--bias_list", action="store_true", help="active all bias list prompt")
    parser.add_argument("--bias_nums", type=int, default=0, help="number of bias words")
    
    args = parser.parse_args()
    
    args.random = True  # 5% random prompt
    
    print("Bool of using prompt:", args.prompt)
    print("Bool of using bias_list:", args.bias_list)
    print("Bool of using random:", args.random)
    print("Bias nums:", args.bias_nums)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-base.en')
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-base.en', language='en', task='transcribe')
    processor = WhisperProcessor.from_pretrained('openai/whisper-base.en', language='en', task='transcribe')
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        decoder_prev_token_id=tokenizer.convert_tokens_to_ids("<|startofprev|>"),
    )
    
    print("DATA_ROOT:", args.data_root)
    print("DATA_DIR:", args.data_dir)
    print("JSONL_DATA:", args.jsonl_data)
    
    print("Starting loading data!")
    data_test = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase='test',
        feature_extractor=feature_extractor,
        audio_type=".mp3",
        tokenizer=tokenizer,
        prompt=args.prompt,
        random=args.random,
        bias_list=args.bias_list,
        bias_nums=args.bias_nums
    )    
    
    # Tạo DataLoader với batch size nhỏ để kiểm tra
    batch_size = 2
    dataloader = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # Lấy batch đầu tiên để kiểm tra
    batch = next(iter(dataloader))

    # In labels trước khi đưa vào DataCollator (từ dataset)
    print("\n=== Labels Before and After DataCollator, and Decoder Input IDs ===")
    for i in range(batch_size):
        # Lấy labels từ dataset (trước DataCollator)
        sample = data_test[i]
        labels_before = sample["labels"].tolist()
        
        # Lấy labels và decoder_input_ids sau DataCollator
        labels_after = batch["labels"][i].tolist()
        decoder_input_ids = batch["decoder_input_ids"][i].tolist()

        # Xác định độ dài tối đa để gióng hàng
        max_len = max(len(labels_before), len(labels_after), len(decoder_input_ids))

        # Padding các list để có cùng độ dài (dùng -999 làm placeholder cho dễ phân biệt)
        labels_before.extend([-999] * (max_len - len(labels_before)))
        labels_after.extend([-999] * (max_len - len(labels_after)))
        decoder_input_ids.extend([-999] * (max_len - len(decoder_input_ids)))

        # In dưới dạng bảng
        print(f"\nSample {i} - Comparison Table:")
        print(f"{'Position':<10} {'Labels Before':<15} {'Labels After':<15} {'Decoder Input IDs':<15}")
        print("-" * 60)
        for pos in range(max_len):
            lb = labels_before[pos]
            la = labels_after[pos]
            di = decoder_input_ids[pos]
            # Decode token nếu không phải -999
            lb_str = tokenizer.decode([lb], skip_special_tokens=False) if lb != -999 else "N/A"
            la_str = tokenizer.decode([la], skip_special_tokens=False) if la != -999 else "N/A"
            di_str = tokenizer.decode([di], skip_special_tokens=False) if di != -999 else "N/A"
            print(f"{pos:<10} {lb:<15} {la:<15} {di:<15} | Decoded: {lb_str:<15} {la_str:<15} {di_str:<15}")
