"""
Script đánh giá mô hình Whisper medical
"""

import argparse
import os
import torch
import gc
import json
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.whisper_medical import WhisperMedical
from data_utils.dataloader import WhisperMedicalDataset, WhisperDataCollator
from trainers.medical_trainer import WhisperMedicalTrainer
# from data_utils.data_collator import WhisperDataCollator
from utils.evaluation import compute_metrics_whisper_with_prompt, compute_metrics_whisper_baseline

from transformers import TrainingArguments

# lambda eval_preds: compute_metrics_whisper_baseline(
#           eval_preds=eval_preds,
#           tokenizer=whisper_medical.processor.tokenizer
#         )

whisper_medical = WhisperMedical(model_id="openai/whisper-base.en", freeze_encoder=False)

def my_compute_metrics(eval_preds):
  return compute_metrics_whisper_baseline(
    eval_preds=eval_preds,
    tokenizer=whisper_medical.processor.tokenizer,
    # prompt_ids_list=None
  )
  
def test_model_basic_functionality(whisper_medical, audio_path):
    """Kiểm tra chức năng cơ bản của model với một file âm thanh"""
    print(f"Testing model with audio: {audio_path}")
    
    # Tải âm thanh
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Xử lý đầu vào
    input_features = whisper_medical.processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(whisper_medical.device)
    
    print(f"Input features shape: {input_features.shape}")
    
    # Chạy inference với model
    print("Running model.generate()...")
    with torch.no_grad():
        generated_ids = whisper_medical.model.generate(
            input_features,
            max_length=256
        )
    
    # Decode kết quả
    transcription = whisper_medical.processor.tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0].strip()
    
    print(f"Generated transcription: {transcription}")
    return transcription
# def calculate_wer(references, predictions):
#     valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
#                    if ref.strip() and pred is not None]
    
#     if not valid_pairs:
#         return 1.0, [1.0] * len(references)
    
#     refs, preds = zip(*valid_pairs)
    
#     overall_wer = jiwer.wer(refs, preds)
    
#     return overall_wer


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Đường dẫn đến file JSONL test data")
    parser.add_argument("--test_audio_dir", type=str, required=True, help="Thư mục chứa audio test")
    parser.add_argument("--bias_words_file", type=str, required=True, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--compare_baseline", action="store_true", help="So sánh với Whisper cơ bản")
    
    args = parser.parse_args()
    
    # Thiết lập output file
    if args.output is None:
        eval_dir = os.path.join(os.path.dirname(args.model_dir), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        model_name = os.path.basename(args.model_dir)
        args.output = os.path.join(eval_dir, f"{model_name}_evaluation.json")
    
    # Đọc bias words từ file
    with open(args.bias_words_file, 'r', encoding='utf-8') as f:
        bias_words = [line.strip() for line in f if line.strip()]
    
    bias_words_string = ", ".join(bias_words)
    
    print(f"Loaded {len(bias_words)} bias words.")
    
    if args.model_dir.startswith("openai/"):
        whisper_medical = WhisperMedical(model_id=args.model_dir, freeze_encoder=False)
    else:
        whisper_medical = WhisperMedical()
        whisper_medical.load(args.model_dir)    
        
    # Tạo test dataset
    test_dataset = WhisperMedicalDataset(
        args.test_jsonl, 
        whisper_medical.processor, 
        audio_dir=args.test_audio_dir,
        bias_words_string=None,
        max_prompt_length=190,
        random_prob=0  # Không sử dụng perturbation trong test
    )
    
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    sample_audio_path = os.path.join(args.test_audio_dir, test_dataset.data[0]['file'])
    
    print("\n=== Testing basic model functionality ===")
    transcription = test_model_basic_functionality(whisper_medical, sample_audio_path)
    
    reference = test_dataset.data[0]['text']
    print(f"Reference transcription: {reference}")
    
    # for i in range(min(3, len(test_dataset))):
    #     sample = test_dataset[i]
    #     print(f"Sample {i}:")
    #     print(f"  Keys: {sample.keys()}")
    #     print(f"  Input shape: {sample['input_features'].shape if 'input_features' in sample else 'N/A'}")
    #     print(f"  Label shape: {sample['labels'].shape if 'labels' in sample else 'N/A'}")
    
    # training_args = TrainingArguments(
    #     output_dir="/kaggle/working",
    #     per_device_eval_batch_size=1,
    #     eval_accumulation_steps=2,
    #     remove_unused_columns=False,
    #     do_eval=True,
    #     report_to="none",  # Không cần push log
    # )
    
    # trainer = WhisperMedicalTrainer(
    #     model=whisper_medical.model,
    #     args=training_args,
    #     tokenizer=whisper_medical.processor.tokenizer,
    #     # data_collator=DebugWhisperDataCollator(whisper_medical.processor),
    #     # prompt_ids_list=None,  # if exists
    #     data_collator=WhisperDataCollator(whisper_medical.processor),
    #     compute_metrics=my_compute_metrics
    # )

    # print(f"Before evaluation, test_dataset has {len(test_dataset)} samples")
    
    # results = trainer.evaluate(eval_dataset=test_dataset)
    # print(results)
    
    # with open("test_results.json", "w", encoding="utf-8") as f:
    #   json.dump(results, f, indent=2)
      
if __name__ == "__main__":
    # Giải phóng bộ nhớ trước khi bắt đầu
    gc.collect()
    torch.cuda.empty_cache()
    
    main()

    # data_collator = WhisperDataCollator(    
    
    # results = {
    #     "with_description": {},
    #     "without_description": {}
    # }
    
    # # # Đánh giá với description
    # total_samples = len(test_dataset)
    # successes = 0
    # error_count = 0
    
    # with_description_refs = []
    # with_description_preds = []
    
    # print(f"Evaluating {total_samples} samples with description...")
    
    # for i, item in enumerate(test_dataset):
    #     if i % 10 == 0:
    #         print(f"Processing {i}/{total_samples}...")
        
    #     try:
    #         audio_path = os.path.join(args.test_audio_dir, item["file_name"])
    #         transcript = item["transcript"]
    #         description = item["description"]
            
    #         # Transcribe với description
    #         prediction = whisper_medical.transcribe(audio_path, description, bias_words_string)
            
    #         with_description_refs.append(transcript)
    #         with_description_preds.append(prediction)
            
    #         successes += 1
    #     except Exception as e:
    #         print(f"Error processing sample {i}: {e}")
    #         error_count += 1
    
    # # Tính WER cho with_description
    # from utils.evaluation import calculate_wer
    # wer_with_desc, _ = calculate_wer(with_description_refs, with_description_preds)
    
    # results["with_description"]["wer"] = wer_with_desc
    # results["with_description"]["successful_samples"] = successes
    # results["with_description"]["error_count"] = error_count
    
    # Đánh giá không có description
    # if args.compare_baseline:
    # without_description_refs = []
    # without_description_preds = []
    # successes = 0
    # error_count = 0
    
    # print(f"Evaluating {total_samples} samples without description...")
    
    # for i, item in enumerate(test_dataset):
    #     if i % 10 == 0:
    #         print(f"Processing {i}/{total_samples}...")
        
    #     try:
    #         audio_path = os.path.join(args.test_audio_dir, item["file_name"])
    #         transcript = item["transcript"]
            
    #         # Transcribe không có description
    #         prediction = whisper_medical.transcribe(audio_path)
            
    #         without_description_refs.append(transcript)
    #         without_description_preds.append(prediction)
            
    #         successes += 1
    #     except Exception as e:
    #         print(f"Error processing sample {i}: {e}")
    #         error_count += 1
    
    # results = compute_metrics_whisper_with_prompt(
    #     # whisper_medical, 
    #     # test_dataset, 
    #     # bias_words_string
    #     eval_preds=...,
    #     tokenizer=whisper_medical.processor.tokenizer,
    # )
    
    # Tính WER cho without_description
    # wer_without_desc, _ = calculate_wer(without_description_refs, without_description_preds)
    # print(f"WER without description: {wer_without_desc:.4f}")
    # results["without_description"]["wer"] = wer_without_desc
        # results["without_description"]["successful_samples"] = successes
        # results["without_description"]["error_count"] = error_count
        
        # Tính cải thiện
        # improvement = wer_without_desc - wer_with_desc
        # improvement_percent = (improvement / wer_without_desc) * 100 if wer_without_desc > 0 else 0
        
        # results["improvement"] = {
        #     "absolute": improvement,
        #     "percent": improvement_percent
        # }
    
    # Lưu kết quả đánh giá
    # with open(args.output, 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # Lưu các dự đoán cụ thể
    # predictions_dir = os.path.join(os.path.dirname(args.output), "predictions")
    # os.makedirs(predictions_dir, exist_ok=True)
    
    # with open(os.path.join(predictions_dir, "with_description_predictions.txt"), 'w') as f:
    #     for ref, pred in zip(with_description_refs, with_description_preds):
    #         f.write(f"Ref: {ref}\n")
    #         f.write(f"Pred: {pred}\n\n")
    
    # if args.compare_baseline:
    #     with open(os.path.join(predictions_dir, "without_description_predictions.txt"), 'w') as f:
    #         for ref, pred in zip(without_description_refs, without_description_preds):
    #             f.write(f"Ref: {ref}\n")
    #             f.write(f"Pred: {pred}\n\n")
    
    # Hiển thị kết quả
    # print("\nEvaluation Results:")
    # print(f"WER with description: {wer_without_desc:.4f}")
    
    # if args.compare_baseline:
    #     print(f"WER without description: {results['without_description']['wer']:.4f}")
    #     print(f"Improvement: {results['improvement']['absolute']:.4f} ({results['improvement']['percent']:.2f}%)")
    
    # print(f"\nResults saved to {args.output}")

