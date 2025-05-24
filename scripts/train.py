import argparse
import os
import gc
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import json
import wandb
from huggingface_hub import snapshot_download, HfApi
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, GenerationConfig
from transformers.trainer_callback import TrainerCallback
from transformers import EarlyStoppingCallback

# Đảm bảo PROJECT_ROOT trỏ đúng đến thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"sys.path: {sys.path}")

try:
    from models.whisper_medical import WhisperForConditionalGenerationWeightCE
except ModuleNotFoundError:
    print("Error: Module 'models' not found. Please check if 'models/whisper_medical.py' exists in PROJECT_ROOT.")
    sys.exit(1)

from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.compute_metric import compute_wer, compute_bias_wer
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA

def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper medical model with context biasing")
    parser.add_argument("--output", type=str, default="/kaggle/working/results", help="Output directory for results")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Base input data directory")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Middle input data directory")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Path to JSONL metadata")
    parser.add_argument("--refs_pred_file", type=str, required=False, default=None, help="Path to refs and pred")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight for weighted cross-entropy")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Hugging Face id for saving")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint or Hugging Face Hub")
    parser.add_argument("--prompt", action="store_true", help="Use prompt in decoder")
    parser.add_argument("--random", action="store_true", help="Apply context perturbation")
    parser.add_argument("--bias_list", action="store_true", help="Active all bias list prompt")
    parser.add_argument("--bias_nums", type=int, default=0, help="Number of bias words")
    return parser.parse_args()

def sync_from_hub(repo_id, local_dir, token):
    print(f"Syncing from Hugging Face Hub: {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", token=token)
    print("Download complete.")

def upload_results_to_hub(results_file, repo_id, hub_path, token):
    api = HfApi()
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    try:
        print(f"Uploading {results_file} to {repo_id} at {hub_path}...")
        api.upload_file(
            path_or_fileobj=results_file,
            path_in_repo=hub_path,
            repo_id=repo_id,
            token=token
        )
        print("Upload successful.")
    except Exception as e:
        print(f"Failed to upload {results_file} to Hugging Face Hub: {str(e)}")
        raise

def push_to_hub_if_exists(local_dir, repo_id, token):
    if os.path.exists(local_dir) and any(os.path.isfile(os.path.join(local_dir, f)) for f in os.listdir(local_dir)):
        print(f"⬆Uploading {local_dir} to Hugging Face Hub at {repo_id}...")
        api = HfApi()
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("Upload complete.")
    else:
        print(f"Skipping upload: {local_dir} is empty or does not exist")

class PushToHubOnSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        push_to_hub_if_exists(local_dir=args.output_dir, repo_id=args.hub_model_id, token=args.push_to_hub_token)

def main():
    args = parse_args()
    print(f"Arguments: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using prompt: {args.prompt}")
    print(f"Using random context perturbation: {args.random}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prev_token_id = processor.tokenizer.convert_tokens_to_ids("<|startofprev|>")
    if start_token_id is None or prev_token_id is None:
        raise ValueError("Special tokens <|startoftranscript|> or <|startofprev|> not found in tokenizer vocabulary")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=start_token_id,
        decoder_prev_token_id=prev_token_id,
    )

    print(f"DATA_ROOT: {args.data_root}")
    print(f"DATA_DIR: {args.data_dir}")
    print(f"JSONL_DATA: {args.jsonl_data}")
    for phase in ["train", "dev", "test"]:
        jsonl_path = os.path.join(args.jsonl_data, f"{phase}.jsonl")
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    print("Loading datasets...")
    data_train = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase="train",
        feature_extractor=processor.feature_extractor,
        audio_type=".mp3",
        tokenizer=processor.tokenizer,
        prompt=args.prompt,
        random=args.random,
        bias_list=args.bias_list,
        bias_nums=args.bias_nums
    )
    data_eval = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase="dev",
        feature_extractor=processor.feature_extractor,
        audio_type=".mp3",
        tokenizer=processor.tokenizer,
        prompt=args.prompt,
        random=args.random,
        bias_list=args.bias_list,
        bias_nums=args.bias_nums
    )
    data_test = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase="test",
        feature_extractor=processor.feature_extractor,
        audio_type=".mp3",
        tokenizer=processor.tokenizer,
        prompt=args.prompt,
        random=args.random,
        bias_list=args.bias_list,
        bias_nums=args.bias_nums
    )

    if len(data_train) == 0 or len(data_eval) == 0 or len(data_test) == 0:
        raise ValueError("One or more datasets are empty")
    print(f"Train data length: {len(data_train)}")
    print(f"Eval data length: {len(data_eval)}")
    print(f"Test data length: {len(data_test)}")

    bias_spans = [data_test[i]["bias_spans"] for i in tqdm(range(len(data_test)), desc="Collecting bias spans", total=len(data_test))]

    output_dir = args.output if args.output else os.path.join("/kaggle/working", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Xác định model_source và checkpoint
    checkpoint_dir = None
    model_source = "openai/whisper-base.en"
    if args.resume:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint_dir = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            print(f"Tiếp tục huấn luyện từ checkpoint: {checkpoint_dir}")
        else:
            print(f"Không tìm thấy checkpoint trong output_dir, syncing từ Hugging Face Hub: {args.hub_model_id}")
            sync_from_hub(repo_id=args.hub_model_id, local_dir=output_dir, token=args.hf_token)
            checkpoint_dir = output_dir
    else:
        print("Bắt đầu huấn luyện từ đầu với openai/whisper-base.en")

    # Load mô hình
    try:
        model = WhisperForConditionalGenerationWeightCE.from_pretrained(
            checkpoint_dir if args.resume else model_source,
            bias_weight=args.bias_weight
        )
        model.freeze_encoder()
        if hasattr(model.config, 'forced_decoder_ids'):
            print("Removing forced_decoder_ids from model config")
            model.config.forced_decoder_ids = None
        if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'forced_decoder_ids'):
            print("Removing forced_decoder_ids from generation config")
            model.generation_config.forced_decoder_ids = None
        model.generation_config = GenerationConfig(
            max_length=225,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
            use_cache=False
        )
        model.config.suppress_tokens = []
    except Exception as e:
        raise RuntimeError(f"Không thể load mô hình từ {checkpoint_dir if args.resume else model_source}: {str(e)}")

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters in the model. Check freeze_encoder() or model configuration.")

    wandb_project = args.hub_model_id.split("/")[-1] if args.hub_model_id else "whisper-default"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        max_steps=-1,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=True,
        dataloader_num_workers=1,
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save",
        push_to_hub_token=args.hf_token,
        report_to=["wandb"],
    )

    data_collator.training = True
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_train,
        eval_dataset=data_eval,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_wer,
    )
    trainer.add_callback(PushToHubOnSaveCallback())
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint_dir if args.resume else None)

    data_collator.training = False
    print("Starting final evaluation on test set...")
    result = trainer.evaluate(data_test)
    print("Test set evaluation results:", result)

    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved evaluation results to {results_file}")

    upload_results_to_hub(
        results_file=results_file,
        repo_id=args.hub_model_id,
        hub_path="results/test_results.json",
        token=args.hf_token
    )

    print("Calculating bias WER...")
    refs_pred_file = os.path.join(output_dir, "refs_and_pred.txt")
    bias_wer_result = compute_bias_wer(refs_pred_file, bias_spans, processor.tokenizer)
    print("Bias WER result:", bias_wer_result)

    bias_wer_file = os.path.join(output_dir, "bias_wer_results.json")
    with open(bias_wer_file, "w") as f:
        json.dump(bias_wer_result, f, indent=4)
    print(f"Saved bias WER results to {bias_wer_file}")

    upload_results_to_hub(
        results_file=bias_wer_file,
        repo_id=args.hub_model_id,
        hub_path="results/bias_wer_results.json",
        token=args.hf_token
    )

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()