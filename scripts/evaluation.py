import argparse
import os
import json
import torch
import gc
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.whisper_medical import WhisperForConditionalGenerationWeightCE
from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.compute_metric import compute_wer, compute_bias_wer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, WhisperTokenizer
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper medical model with context biasing")
    parser.add_argument("--output", type=str, default="/kaggle/working/results", help="Output directory for results")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight for weighted cross-entropy")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Base input data directory")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Subdirectory for audio files")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Path to JSONL metadata")
    parser.add_argument("--prompt", action="store_true", help="Use prompt in decoder")
    parser.add_argument("--random", action="store_true", help="Apply context perturbation (5% random prompt)")
    parser.add_argument("--only_eval_bias_wer", action="store_true", help="param for eval bias_wer only")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Hugging Face model ID of the trained model")
    parser.add_argument("--refs_pred_file", type=str, required=False, default=None, help="Path to refs and pred (will be overwritten)")
    parser.add_argument("--final_model", action="store_true", default=False, help="eval with final trained model")
    parser.add_argument("--best_checkpoint", action="store_true", default=False, help="eval with best checkpoint")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private repos (optional)")
    return parser.parse_args()

def save_refs_and_preds(trainer, dataset, tokenizer, refs_pred_file):
    """Tạo và lưu file refs_pred_file từ dự đoán của trainer."""
    print(f"Generating predictions for all samples in dataset...")
    predictions = trainer.predict(dataset)
    pred_ids = predictions.predictions  
    label_ids = predictions.label_ids   

    pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    with open(refs_pred_file, "w", encoding="utf-8") as f:
        for ref, pred in zip(label_strs, pred_strs):
            f.write(f"ref: {ref} | pred: {pred}\n")
    print(f"Saved {len(pred_strs)} samples to {refs_pred_file}")

def evaluate_model(trainer, output_dir, model_name, refs_pred_file, bias_spans, tokenizer, only_eval_bias_wer=False):
    if not only_eval_bias_wer:
        print(f"Starting evaluation for WER with {model_name}...")
        result = trainer.evaluate()
        print(f"{model_name} Test set evaluation results:", result)

        results_file = os.path.join(output_dir, f"{model_name}_test_results.json")
        with open(results_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved {model_name} evaluation results to {results_file}")

    print(f"Calculating bias WER with {model_name}...")
    print(f"ref and pred file path: {refs_pred_file}")
    bias_wer_result = compute_bias_wer(refs_pred_file, bias_spans, tokenizer)
    print(f"{model_name} Bias WER result:", bias_wer_result)

    bias_wer_file = os.path.join(output_dir, f"{model_name}_bias_wer_results.json")
    with open(bias_wer_file, "w") as f:
        json.dump(bias_wer_result, f, indent=4)
    print(f"Saved {model_name} bias WER results to {bias_wer_file}")

def find_best_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    best_checkpoint = None
    best_wer = float('inf')
    for checkpoint in checkpoints:
        metric_file = os.path.join(checkpoint_dir, checkpoint, "trainer_state.json")
        if os.path.exists(metric_file):
            with open(metric_file, "r") as f:
                metrics = json.load(f)
                for state in metrics.get("log_history", []):
                    if "eval_wer" in state:
                        wer = state["eval_wer"]
                        if wer < best_wer:
                            best_wer = wer
                            best_checkpoint = os.path.join(checkpoint_dir, checkpoint)
    return best_checkpoint

def download_from_hub(repo_id, local_dir, token=None):
    """Tải toàn bộ repository từ Hugging Face Hub xuống local_dir."""
    print(f"Downloading repository from Hugging Face Hub: {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", token=token)
    print(f"Repository downloaded to {local_dir}")

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using prompt: {args.prompt}")
    print(f"Using random context perturbation: {args.random}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prev_token_id = processor.tokenizer.convert_tokens_to_ids("<|startofprev|>")
    if start_token_id is None or prev_token_id is None:
        raise ValueError("Special tokens <|startoftranscript|> or <|startofprev|> not found in tokenizer vocabulary")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=start_token_id,
        decoder_prev_token_id=prev_token_id,
    )
    data_collator.training = False  

    print(f"DATA_ROOT: {args.data_root}")
    print(f"DATA_DIR: {args.data_dir}")
    print(f"JSONL_DATA: {args.jsonl_data}")
    test_jsonl = os.path.join(args.jsonl_data, "test.jsonl")
    if not os.path.isfile(test_jsonl):
        raise FileNotFoundError(f"Test JSONL file not found: {test_jsonl}")

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

    bias_spans = [data_test[i]["bias_spans"] for i in tqdm(range(len(data_test)), desc="Collecting bias spans", total=len(data_test))]
    print(f"Total bias spans collected: {len(bias_spans)}")

    output_dir = args.output if args.output else os.path.join("/kaggle/working", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Tải toàn bộ repository từ Hugging Face Hub xuống output_dir nếu cần best_checkpoint
    if args.best_checkpoint:
        download_from_hub(repo_id=args.hub_model_id, local_dir=output_dir, token=args.hf_token)

    if not args.final_model and not args.best_checkpoint:
        print("Chọn cờ để test!")
        return

    if args.final_model:
        print(f"Loading final pre-trained model from: {args.hub_model_id}")
        try:
            final_model = WhisperForConditionalGenerationWeightCE.from_pretrained(args.hub_model_id, bias_weight=args.bias_weight)
            # Xóa hoàn toàn forced_decoder_ids khỏi cấu hình
            if hasattr(final_model.config, "forced_decoder_ids"):
                print("Removing forced_decoder_ids from final_model config")
                delattr(final_model.config, "forced_decoder_ids")
            final_model.config.use_cache = False  
            final_model.freeze_encoder()
            final_model.config.suppress_tokens = []
            final_model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load final model from {args.hub_model_id}: {str(e)}")

        training_args_final = Seq2SeqTrainingArguments(
            output_dir=os.path.join(output_dir, "final_model"),
            per_device_eval_batch_size=args.batch,
            predict_with_generate=True,
            generation_max_length=225,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            report_to=[],
            do_train=False,
            do_eval=True,
        )

        trainer_final = Seq2SeqTrainer(
            args=training_args_final,
            model=final_model,
            eval_dataset=data_test,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            compute_metrics=compute_wer if not args.only_eval_bias_wer else None,
        )

        final_refs_pred_file = os.path.join(output_dir, "refs_and_pred.txt")
        save_refs_and_preds(trainer_final, data_test, tokenizer, final_refs_pred_file)
        evaluate_model(trainer_final, output_dir, "refs_and_pred", final_refs_pred_file, bias_spans, tokenizer, args.only_eval_bias_wer)

    if args.best_checkpoint:
        best_checkpoint = find_best_checkpoint(output_dir)
        if best_checkpoint:
            print(f"Loading best checkpoint from: {best_checkpoint}")
            try:
                best_model = WhisperForConditionalGenerationWeightCE.from_pretrained(best_checkpoint, bias_weight=args.bias_weight)
                # Xóa hoàn toàn forced_decoder_ids khỏi cấu hình
                if hasattr(best_model.config, "forced_decoder_ids"):
                    print("Removing forced_decoder_ids from best_model config")
                    delattr(best_model.config, "forced_decoder_ids")
                best_model.config.use_cache = False  
                best_model.freeze_encoder()
                best_model.generation_config.language = "<|en|>"
                best_model.generation_config.task = "transcribe"
                best_model.config.suppress_tokens = []
                best_model.to(device)
            except Exception as e:
                print(f"Failed to load best checkpoint from {best_checkpoint}: {str(e)}")
                raise
            else:
                training_args_best = Seq2SeqTrainingArguments(
                    output_dir=os.path.join(output_dir, "best_checkpoint"),
                    per_device_eval_batch_size=args.batch,
                    predict_with_generate=True,
                    generation_max_length=225,
                    remove_unused_columns=False,
                    fp16=torch.cuda.is_available(),
                    report_to=[],
                    do_train=False,
                    do_eval=True,
                )

                trainer_best = Seq2SeqTrainer(
                    args=training_args_best,
                    model=best_model,
                    eval_dataset=data_test,
                    data_collator=data_collator,
                    tokenizer=processor.tokenizer,
                    compute_metrics=compute_wer if not args.only_eval_bias_wer else None,
                )

                best_refs_pred_file = os.path.join(output_dir, "refs_and_pred.txt")
                save_refs_and_preds(trainer_best, data_test, tokenizer, best_refs_pred_file)
                evaluate_model(trainer_best, output_dir, "refs_and_pred", best_refs_pred_file, bias_spans, tokenizer, args.only_eval_bias_wer)
        else:
            print("No valid checkpoint found in output_dir for evaluation.")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()