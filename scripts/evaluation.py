import argparse
import os
import json
import torch
import gc
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.whisper_medical import WhisperForConditionalGenerationWeightCE
from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.compute_metric import compute_wer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper medical model with context biasing")
    parser.add_argument("--output", type=str, default="/kaggle/working/results", help="Output directory for results")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight for weighted cross-entropy")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Base input data directory")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Subdirectory for audio files")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Path to JSONL metadata")
    parser.add_argument("--prompt", action="store_true", help="Use prompt in decoder")
    parser.add_argument("--random", action="store_true", help="Apply context perturbation (5% random prompt)")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face model ID (optional; defaults to openai/whisper-base.en)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using prompt: {args.prompt}")
    print(f"Using random context perturbation: {args.random}")

    # Initialize processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prev_token_id = processor.tokenizer.convert_tokens_to_ids("<|startofprev|>")
    if start_token_id is None or prev_token_id is None:
        raise ValueError("Special tokens <|startoftranscript|> or <|startofprev|> not found in tokenizer vocabulary")

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=start_token_id,
        decoder_prev_token_id=prev_token_id,
    )
    data_collator.training = False  # Disable bias_spans for evaluation

    # Validate data paths
    print(f"DATA_ROOT: {args.data_root}")
    print(f"DATA_DIR: {args.data_dir}")
    print(f"JSONL_DATA: {args.jsonl_data}")
    test_jsonl = os.path.join(args.jsonl_data, "test.jsonl")
    if not os.path.isfile(test_jsonl):
        raise FileNotFoundError(f"Test JSONL file not found: {test_jsonl}")

    # Load test dataset
    print("Loading test dataset...")
    data_test = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase="test",
        feature_extractor=processor.feature_extractor,
        audio_type=".mp3",
        tokenizer=processor.tokenizer,
        prompt=args.prompt,
        random=args.random
    )
    if len(data_test) == 0:
        raise ValueError("Test dataset is empty")
    print(f"Test data length: {len(data_test)}")

    # Initialize model
    model_id = args.hub_model_id if args.hub_model_id else "openai/whisper-base.en"
    print(f"Loading model from: {model_id}")
    try:
        model = WhisperForConditionalGenerationWeightCE.from_pretrained(model_id, bias_weight=args.bias_weight)
        model.config.use_cache = False  # Disable caching for evaluation
        model.freeze_encoder()
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_id}: {str(e)}")

    # Set up output directory
    os.makedirs(args.output, exist_ok=True)

    # Training arguments for evaluation
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_eval_batch_size=args.batch,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=data_test,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_wer,
    )

    # Evaluate
    print("Starting evaluation...")
    result = trainer.evaluate()
    print("Test set evaluation results:", result)

    # Save results to JSON
    results_file = os.path.join(args.output, "test_results.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved evaluation results to {results_file}")

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()