import argparse
import os
import gc
import torch
from pathlib import Path
from tqdm import tqdm
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.whisper_medical import WhisperForConditionalGenerationWeightCE
from data_utils.data_loader import PromptWhisperDataset
from data_utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.compute_metric import compute_wer, compute_bias_wer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA
from huggingface_hub import snapshot_download, HfApi
from transformers.trainer_callback import TrainerCallback
from transformers import EarlyStoppingCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper medical model with context biasing")
    parser.add_argument("--output", type=str, default="/kaggle/working/results", help="Output directory for results")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight for weighted cross-entropy")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Base input data directory")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Middle input data directory")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="Path to JSONL metadata")
    parser.add_argument("--prompt", action="store_true", help="Use prompt in decoder")
    parser.add_argument("--random", action="store_true", help="Apply context perturbation")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Hugging Face id for saving")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--resume", action="store_true", help="Resume from Hugging Face Hub if output_dir is empty")
    return parser.parse_args()

def sync_from_hub(repo_id, local_dir, token):
    print(f"Syncing from Hugging Face Hub: {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", token=token)
    print("Download complete.")

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

    # Load datasets
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
        random=args.random
    )
    data_eval = PromptWhisperDataset(
        base_path=os.path.join(args.data_root, args.data_dir),
        jsonl_data=args.jsonl_data,
        phase="dev",
        feature_extractor=processor.feature_extractor,
        audio_type=".mp3",
        tokenizer=processor.tokenizer,
        prompt=args.prompt,
        random=args.random
    )
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

    if len(data_train) == 0 or len(data_eval) == 0 or len(data_test) == 0:
        raise ValueError("One or more datasets are empty")
    print(f"Train data length: {len(data_train)}")
    print(f"Eval data length: {len(data_eval)}")
    print(f"Test data length: {len(data_test)}")
    
    bias_spans = [data_test[i]["bias_spans"] for i in tqdm(range(len(data_test)), desc="Collecting bias spans", total=len(data_test))]

    # Calculate steps
    # iteration_steps = int(len(data_train) * args.epoch // (args.batch * 2))  # Account for gradient_accumulation_steps=8
    # eval_step = int((len(data_train) // 2) // (args.batch * 2))
    # log_step = int((len(data_train) // 50) // (args.batch * 2))
    # print(f"Max steps: {iteration_steps}")
    # print(f"Eval step: {eval_step}")
    # print(f"Log step: {log_step}")

    # Set output directory
    output_dir = args.output if args.output else os.path.join("/kaggle/working", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model_source = args.hub_model_id if args.resume else "openai/whisper-base.en"
    print(f"Loading model from: {model_source}")

    # Tải checkpoint từ Hugging Face Hub nếu resume=True và output_dir trống
    if args.resume and not os.listdir(output_dir):
        sync_from_hub(repo_id=model_source, local_dir=output_dir, token=args.hf_token)

    # Tìm checkpoint mới nhất trong output_dir
    checkpoint_dir = None
    if args.resume:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint_dir = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            print(f"Tiếp tục huấn luyện từ checkpoint: {checkpoint_dir}")
        else:
            print("Không tìm thấy checkpoint trong output_dir, load mô hình từ model_source")

    # Khởi tạo mô hình
    try:
        # Nếu có checkpoint, load từ checkpoint; nếu không, load từ model_source
        model = WhisperForConditionalGenerationWeightCE.from_pretrained(
            checkpoint_dir if checkpoint_dir else model_source,
            bias_weight=args.bias_weight
        )
        model.freeze_encoder()
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
      
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters in the model. Check freeze_encoder() or model configuration.")

    wandb_project = "whisper-default"  # Fallback project name
    if args.hub_model_id:
        wandb_project = args.hub_model_id.split("/")[-1]

    # Training arguments
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
        load_best_model_at_end=True, # load best eval wer result
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
        # wandb_project=wandb_project,  
        # run_name=f"run-{output_dir.split('/')[-1]}"
    )

    # Initialize trainer
    data_collator.training = True  # Enable bias_spans for training
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

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None)

    # Evaluate on test set
    data_collator.training = False  # Disable bias_spans for evaluation
    print("Starting final evaluation on test set...")
    result = trainer.evaluate(data_test)
    print("Test set evaluation results:", result)
    
    # Save evaluation results to JSON
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved evaluation results to {results_file}")
    
    # Tính bias_wer
    print("Calculating bias WER...")
    refs_pred_file = os.path.join(output_dir, "refs_and_pred.txt")
    bias_wer_result = compute_bias_wer(refs_pred_file, bias_spans, tokenizer)
    print("Bias WER result:", bias_wer_result)

    # Lưu bias_wer vào JSON
    bias_wer_file = os.path.join(output_dir, "bias_wer_results.json")
    with open(bias_wer_file, "w") as f:
        json.dump(bias_wer_result, f, indent=4)
    print(f"Saved bias WER results to {bias_wer_file}")

    # Upload results to Hugging Face Hub
    upload_results_to_hub(
        results_file=results_file,
        repo_id=training_args.hub_model_id,
        hub_path="results/test_results.json",
        token=args.hf_token
    )
    
    upload_results_to_hub(
        results_file=bias_wer_file,
        repo_id=training_args.hub_model_id,
        hub_path="results/bias_wer_results.json",
        token=args.hf_token
    )
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()