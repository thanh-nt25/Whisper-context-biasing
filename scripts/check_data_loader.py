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
    
    # In labels từ __getitem__ trước khi đưa vào DataLoader
    print("\n=== Labels from __getitem__ Before DataLoader ===")
    for i in range(min(1, len(data_test))):  # In tối đa 5 sample
        sample = data_test[i]
        labels = sample["labels"].tolist()
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=False)
        print(f"\nSample {i} - Encoded Labels: {labels}")
        print(f"Sample {i} - Decoded Labels: {decoded_labels}")

        # Lấy special token IDs
        sop_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
        sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

        # Tách context và transcript
        sop_pos = labels.index(sop_id) if sop_id in labels else -1
        sot_pos = labels.index(sot_id) if sot_id in labels else -1

        if sop_pos == -1 or sot_pos == -1:
            print("Error: Missing <|startofprev|> or <|startoftranscript|> in labels!")
            continue

        # Tách context (phần trước <|startoftranscript|>) và transcript
        context_tokens = labels[sop_pos + 1:sot_pos]
        transcript_tokens = labels[sot_pos + 1:]

        # Decode transcript
        transcript_text = tokenizer.decode(transcript_tokens, skip_special_tokens=True)
        print(f"Sample {i} - Transcript: '{transcript_text}'")

        # Lấy bias words từ bias_spans
        bias_words = []
        for span in sample["bias_spans"]:
            if span and span[0] != tokenizer.pad_token_id:
                decoded = tokenizer.decode([tok for tok in span if tok != tokenizer.pad_token_id])
                bias_words.append(decoded.lower())
        print(f"Sample {i} - Bias words of this sample: {bias_words}")

        # Kiểm tra chiến lược 1: Chỉ prompt
        if args.prompt and not args.bias_list:
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"Sample {i} - Strategy 1: Description Prompt: '{context_text}'")
            print(f"Sample {i} - Encoded Prompt: {context_tokens}")

        # Kiểm tra chiến lược 2: Chỉ bias list
        elif not args.prompt and args.bias_list and args.bias_nums > 0:
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"\nSample {i} - Strategy 2: Bias List Context: '{context_text}'")

            # Tách các từ trong context
            context_words = context_text.split()
            total_words = len(context_words)
            print(f"Sample {i} - Number of words in context: {total_words} (Expected: {args.bias_nums})")

            # Đếm số từ bias và non-bias
            bias_count = 0
            non_bias_count = 0
            bias_words_in_context = []
            for word in context_words:
                word_lower = word.lower()
                if word_lower in bias_words:
                    bias_count += 1
                    bias_words_in_context.append(word_lower)
                elif word_lower in data_test.bias_pool:
                    bias_count += 1
                else:
                    non_bias_count += 1

            # Tính phần trăm
            bias_percentage = (bias_count / total_words) * 100 if total_words > 0 else 0
            non_bias_percentage = (non_bias_count / total_words) * 100 if total_words > 0 else 0

            print(f"Sample {i} - Number of bias words: {bias_count}")
            print(f"Sample {i} - Number of non-bias words: {non_bias_count}")
            print(f"Sample {i} - Bias percentage: {bias_percentage:.2f}% (Expected: 30%)")
            print(f"Sample {i} - Non-bias percentage: {non_bias_percentage:.2f}% (Expected: 70%)")
            print(f"Sample {i} - Bias words of this sample in context: {bias_words_in_context}")
            print(f"Sample {i} - All bias words present: {set(bias_words).issubset(bias_words_in_context)}")

            # In encode và từ tương ứng
            print(f"\nSample {i} - Encoded Tokens and Corresponding Words:")
            current_word = []
            for token in context_tokens:
                current_word.append(token)
                decoded = tokenizer.decode(current_word, skip_special_tokens=True)
                if decoded and decoded != " ":
                    print(f"Tokens {current_word} => '{decoded}'")
                    current_word = []

        # Kiểm tra chiến lược 3: Description + Bias List
        elif args.prompt and args.bias_list and args.bias_nums > 0:
            # Tìm vị trí "Relate terms:"
            relate_terms_tokens = tokenizer.encode("Relate terms:", add_special_tokens=False)
            relate_pos = -1
            for j in range(len(context_tokens) - len(relate_terms_tokens) + 1):
                if context_tokens[j:j + len(relate_terms_tokens)] == relate_terms_tokens:
                    relate_pos = j
                    break

            if relate_pos == -1:
                print(f"Sample {i} - Error: 'Relate terms:' not found in context!")
                continue

            # Tách description và bias list
            description_tokens = context_tokens[:relate_pos]
            relate_terms_and_bias = context_tokens[relate_pos:]
            bias_list_tokens = context_tokens[relate_pos + len(relate_terms_tokens):]

            # Decode từng phần
            description_text = tokenizer.decode(description_tokens, skip_special_tokens=True)
            relate_text = tokenizer.decode(relate_terms_and_bias, skip_special_tokens=True)
            bias_list_text = tokenizer.decode(bias_list_tokens, skip_special_tokens=True)

            print(f"\nSample {i} - Strategy 3: Description + 'Relate terms:' + Bias List")
            print(f"Sample {i} - Description: '{description_text}'")
            print(f"Sample {i} - Encoded Description: {description_tokens}")
            print(f"Sample {i} - Relate terms: + Bias List: '{relate_text}'")
            print(f"Sample {i} - Bias List: '{bias_list_text}'")
            print(f"Sample {i} - Encoded Bias List: {bias_list_tokens}")

            # Phân tích bias list
            bias_list_words = bias_list_text.split()
            total_words = len(bias_list_words)
            print(f"Sample {i} - Number of words in bias list: {total_words} (Expected: {args.bias_nums})")

            # Đếm số từ bias và non-bias
            bias_count = 0
            non_bias_count = 0
            bias_words_in_context = []
            for word in bias_list_words:
                word_lower = word.lower()
                if word_lower in bias_words:
                    bias_count += 1
                    bias_words_in_context.append(word_lower)
                elif word_lower in data_test.bias_pool:
                    bias_count += 1
                else:
                    non_bias_count += 1

            # Tính phần trăm
            bias_percentage = (bias_count / total_words) * 100 if total_words > 0 else 0
            non_bias_percentage = (non_bias_count / total_words) * 100 if total_words > 0 else 0

            print(f"Sample {i} - Number of bias words: {bias_count}")
            print(f"Sample {i} - Number of non-bias words: {non_bias_count}")
            print(f"Sample {i} - Bias percentage: {bias_percentage:.2f}% (Expected: 30%)")
            print(f"Sample {i} - Non-bias percentage: {non_bias_percentage:.2f}% (Expected: 70%)")
            print(f"Sample {i} - Bias words of this sample in context: {bias_words_in_context}")
            print(f"Sample {i} - All bias words present: {set(bias_words).issubset(bias_words_in_context)}")

            # In encode và từ tương ứng cho bias list
            print(f"\nSample {i} - Encoded Tokens and Corresponding Words in Bias List:")
            current_word = []
            for token in bias_list_tokens:
                current_word.append(token)
                decoded = tokenizer.decode(current_word, skip_special_tokens=True)
                if decoded and decoded != " ":
                    print(f"Tokens {current_word} => '{decoded}'")
                    current_word = []