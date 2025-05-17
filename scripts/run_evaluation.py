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
from utils.compute_metric import BasicTextNormalizer

from models.whisper_medical import WhisperMedical
from data_utils.dataloader import WhisperMedicalDataset, WhisperDataCollator
from trainers.medical_trainer import WhisperMedicalTrainer, DebugWhisperMedicalTrainer
# from data_utils.data_collator import WhisperDataCollator
from utils.evaluation import compute_metrics_whisper_with_prompt, compute_metrics_whisper_baseline_debug

from transformers import TrainingArguments

# lambda eval_preds: compute_metrics_whisper_baseline(
#           eval_preds=eval_preds,
#           tokenizer=whisper_medical.processor.tokenizer
#         )

whisper_medical = WhisperMedical(model_id="openai/whisper-base.en", freeze_encoder=False)


  
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
  
def simple_evaluation(model, tokenizer, test_dataset, result_file="simple_eval_results.txt"):
    """
    Hàm đánh giá đơn giản không sử dụng trainer
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_references = []
    all_predictions = []
    
    # Mở file kết quả
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("file_name\treference\tprediction\n")
        
        # Xử lý từng mẫu
        for i in range(len(test_dataset)):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(test_dataset)}")
            
            # Lấy mẫu và đưa lên device
            sample = test_dataset[i]
            input_features = sample["input_features"].unsqueeze(0).to(device)
            
            # Chạy inference
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features,
                    max_length=256
                )
            
            # Decode kết quả
            transcription = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Lấy transcription tham chiếu
            reference = sample["transcript"]
            
            # Lưu vào danh sách để tính WER
            all_references.append(reference)
            all_predictions.append(transcription)
            
            # Ghi ra file
            f.write(f"{sample['file_name']}\t{reference}\t{transcription}\n")
            
            # Giải phóng bộ nhớ
            del input_features, generated_ids
            if i % 20 == 19:
                gc.collect()
                torch.cuda.empty_cache()
    
    # Tính WER
    normalizer = BasicTextNormalizer()
    norm_refs = [normalizer(ref) for ref in all_references]
    norm_preds = [normalizer(pred) for pred in all_predictions]
    
    wer_score = wer(norm_refs, norm_preds) * 100
    print(f"Evaluation complete. WER: {wer_score:.2f}%")
    
    return {"wer": wer_score}  

# def calculate_wer(references, predictions):
#     valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
#                    if ref.strip() and pred is not None]
    
#     if not valid_pairs:
#         return 1.0, [1.0] * len(references)
    
#     refs, preds = zip(*valid_pairs)
    
#     overall_wer = jiwer.wer(refs, preds)
    
#     return overall_wer


def debug_full_evaluation_process(model, tokenizer, test_dataset, output_dir):
    """
    Phân tích đầy đủ quá trình đánh giá
    """
    print("\n=== STARTING FULL EVALUATION DEBUG PROCESS ===")
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Kiểm tra mô hình
    print("\n1. Model check:")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Đảm bảo model được cấu hình đúng
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 2. Kiểm tra dataset
    print("\n2. Dataset check:")
    print(f"Dataset size: {len(test_dataset)}")
    sample = test_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input features shape: {sample['input_features'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # 3. Kiểm tra data collator
    print("\n3. Testing data collators:")
    
    # Collator 1: Không có decoder_input_ids
    collator1 = ConsistentWhisperDataCollator(tokenizer, include_decoder_input_ids=False)
    batch1 = collator1([test_dataset[0], test_dataset[1]])
    print("Collator without decoder_input_ids:")
    print(f"  Batch keys: {batch1.keys()}")
    
    # Collator 2: Có decoder_input_ids
    collator2 = ConsistentWhisperDataCollator(tokenizer, include_decoder_input_ids=True)
    batch2 = collator2([test_dataset[0], test_dataset[1]])
    print("Collator with decoder_input_ids:")
    print(f"  Batch keys: {batch2.keys()}")
    
    # 4. Kiểm tra model.forward
    print("\n4. Testing model.forward:")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Kiểm tra batch thứ nhất (không có decoder_input_ids)
    batch1_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch1.items()}
    
    try:
        with torch.no_grad():
            outputs1 = model(
                input_features=batch1_device["input_features"],
                labels=batch1_device["labels"]
            )
        print("model.forward without decoder_input_ids succeeded!")
        print(f"  Loss: {outputs1.loss}")
        print(f"  Logits shape: {outputs1.logits.shape}")
    except Exception as e:
        print(f"Error in model.forward without decoder_input_ids: {e}")
    
    # Kiểm tra batch thứ hai (có decoder_input_ids)
    if "decoder_input_ids" in batch2:
        batch2_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch2.items()}
        
        try:
            with torch.no_grad():
                outputs2 = model(
                    input_features=batch2_device["input_features"],
                    decoder_input_ids=batch2_device["decoder_input_ids"],
                    labels=batch2_device["labels"]
                )
            print("model.forward with decoder_input_ids succeeded!")
            print(f"  Loss: {outputs2.loss}")
            print(f"  Logits shape: {outputs2.logits.shape}")
        except Exception as e:
            print(f"Error in model.forward with decoder_input_ids: {e}")
    
    # 5. Kiểm tra model.generate
    print("\n5. Testing model.generate:")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                batch1_device["input_features"],
                max_length=256
            )
        print("model.generate without decoder_input_ids succeeded!")
        print(f"  Generated ids shape: {generated_ids.shape}")
        
        # Decode kết quả
        transcriptions = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        print(f"  First transcription: {transcriptions[0]}")
    except Exception as e:
        print(f"Error in model.generate without decoder_input_ids: {e}")
    
    if "decoder_input_ids" in batch2:
        try:
            with torch.no_grad():
                generated_ids_with_prompt = model.generate(
                    batch2_device["input_features"],
                    decoder_input_ids=batch2_device["decoder_input_ids"],
                    max_length=256
                )
            print("model.generate with decoder_input_ids succeeded!")
            print(f"  Generated ids shape: {generated_ids_with_prompt.shape}")
            
            # Decode kết quả
            transcriptions = tokenizer.batch_decode(
                generated_ids_with_prompt, 
                skip_special_tokens=True
            )
            print(f"  First transcription: {transcriptions[0]}")
        except Exception as e:
            print(f"Error in model.generate with decoder_input_ids: {e}")
    
    # 6. Kiểm tra trainer.prediction_step với các collator khác nhau
    print("\n6. Testing trainer.prediction_step:")
    
    # 6.1 Với collator không có decoder_input_ids
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=2,
        remove_unused_columns=False,
    )
    
    trainer1 = DebugWhisperMedicalTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator1,
        compute_metrics=lambda eval_preds: compute_metrics_whisper_baseline_debug(
            eval_preds=eval_preds,
            tokenizer=tokenizer
        )
    )
    
    print("Testing prediction_step with collator without decoder_input_ids:")
    try:
        loss, logits, labels = trainer1.prediction_step(
            model, batch1_device, prediction_loss_only=False
        )
        print("prediction_step succeeded!")
        print(f"  Loss: {loss}")
        print(f"  Logits shape: {logits.shape if logits is not None else None}")
        print(f"  Labels shape: {labels.shape if labels is not None else None}")
    except Exception as e:
        print(f"Error in prediction_step: {e}")
        import traceback
        traceback.print_exc()
    
    # 6.2 Với collator có decoder_input_ids
    if "decoder_input_ids" in batch2:
        trainer2 = DebugWhisperMedicalTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=collator2,
            compute_metrics=lambda eval_preds: compute_metrics_whisper_baseline_debug(
                eval_preds=eval_preds,
                tokenizer=tokenizer
            )
        )
        
        print("Testing prediction_step with collator with decoder_input_ids:")
        try:
            loss, logits, labels = trainer2.prediction_step(
                model, batch2_device, prediction_loss_only=False
            )
            print("prediction_step succeeded!")
            print(f"  Loss: {loss}")
            print(f"  Logits shape: {logits.shape if logits is not None else None}")
            print(f"  Labels shape: {labels.shape if labels is not None else None}")
        except Exception as e:
            print(f"Error in prediction_step: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. Kiểm tra trainer.evaluate với một tập nhỏ
    print("\n7. Testing trainer.evaluate with a small subset:")
    
    small_dataset = torch.utils.data.Subset(test_dataset, range(10))
    
    # 7.1 Với collator không có decoder_input_ids
    print("Testing evaluate with collator without decoder_input_ids:")
    
    try:
        results1 = trainer1.evaluate(eval_dataset=small_dataset)
        print(f"evaluate succeeded with results: {results1}")
    except Exception as e:
        print(f"Error in evaluate: {e}")
        import traceback
        traceback.print_exc()
    
    # 7.2 Với collator có decoder_input_ids (nếu chưa gặp lỗi trước đó)
    if "decoder_input_ids" in batch2:
        print("Testing evaluate with collator with decoder_input_ids:")
        
        try:
            results2 = trainer2.evaluate(eval_dataset=small_dataset)
            print(f"evaluate succeeded with results: {results2}")
        except Exception as e:
            print(f"Error in evaluate: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== EVALUATION DEBUG PROCESS COMPLETED ===")
    
    return {"debug_completed": True}
"""
def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Đường dẫn đến file JSONL test data")
    parser.add_argument("--test_audio_dir", type=str, required=True, help="Thư mục chứa audio test")
    parser.add_argument("--bias_words_file", type=str, required=True, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--compare_baseline", action="store_true", help="So sánh với Whisper cơ bản")
    parser.add_argument("--debug_mode", action="store_true", help="Chạy ở chế độ debug chi tiết")
    parser.add_argument("--use_decoder_input_ids", action="store_true", help="Sử dụng decoder_input_ids")
    
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
    
    if args.debug_mode:
        debug_results = debug_full_evaluation_process(
            model=whisper_medical.model,
            tokenizer=whisper_medical.processor.tokenizer,
            test_dataset=test_dataset,
            output_dir=output_dir
        )
        
        # Lưu kết quả debug
        debug_output = os.path.join(output_dir, "debug_results.json")
        with open(debug_output, "w", encoding="utf-8") as f:
            json.dump(debug_results, f, indent=2)
        
        print(f"Debug results saved to {debug_output}")
        return
    
    # Nếu không ở chế độ debug, tiếp tục với đánh giá bình thường
    
    # Tạo trainer với cài đặt phù hợp
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,  # Batch size nhỏ để giảm thiểu lỗi
        eval_accumulation_steps=4,     # Tăng giá trị này nếu cần thiết
        remove_unused_columns=False,
        do_eval=True,
        report_to="none",
    )
    
    # Tạo data collator phù hợp
    data_collator = ConsistentWhisperDataCollator(
        whisper_medical.processor, 
        include_decoder_input_ids=args.use_decoder_input_ids
    )
    
    # Tạo trainer
    trainer = DebugWhisperMedicalTrainer(
       model=whisper_medical.model,
       args=training_args,
       tokenizer=whisper_medical.processor.tokenizer,
       data_collator=data_collator,
       compute_metrics=lambda eval_preds: compute_metrics_whisper_baseline_debug(
           eval_preds=eval_preds,
           tokenizer=whisper_medical.processor.tokenizer,
           result_dir=output_dir
       )
   )
    
    # Đánh giá mô hình - thử với một tập nhỏ trước
    print("\nTesting evaluation with a small subset first:")
    small_dataset = torch.utils.data.Subset(test_dataset, range(5))
    
    try:
        small_results = trainer.evaluate(eval_dataset=small_dataset)
        print(f"Small evaluation succeeded with results: {small_results}")
        
        # Nếu đánh giá tập nhỏ thành công, tiếp tục với tập đầy đủ
        print("\nRunning full evaluation:")
        full_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Full evaluation results: {full_results}")
        
        # Lưu kết quả
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error in trainer.evaluate: {e}")
        import traceback
        traceback.print_exc()
        
        # Nếu trainer.evaluate() gặp lỗi, sử dụng simple_evaluation
        print("\nFalling back to simple_evaluation:")
        
        def simple_evaluation(model, tokenizer, test_dataset, result_file):
            model.eval()
            device = next(model.parameters()).device
            
            all_references = []
            all_predictions = []
            
            # Mở file kết quả
            with open(result_file, "w", encoding="utf-8") as f:
                f.write("file_name\treference\tprediction\n")
                
                # Xử lý từng mẫu
                for i in range(len(test_dataset)):
                    if i % 10 == 0:
                        print(f"Processing sample {i+1}/{len(test_dataset)}")
                    
                    # Lấy mẫu và đưa lên device
                    sample = test_dataset[i]
                    input_features = sample["input_features"].unsqueeze(0).to(device)
                    
                    # Xử lý với decoder_input_ids nếu cần
                    if args.use_decoder_input_ids and "decoder_input_ids" in sample:
                        decoder_input_ids = sample["decoder_input_ids"].unsqueeze(0).to(device)
                        
                        # Chạy inference với decoder_input_ids
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_features,
                                decoder_input_ids=decoder_input_ids,
                                max_length=256
                            )
                    else:
                        # Chạy inference không có decoder_input_ids
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_features,
                                max_length=256
                            )
                    
                    # Decode kết quả
                    transcription = tokenizer.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0].strip()
                    
                    # Lấy transcription tham chiếu
                    reference = sample["transcript"]
                    
                    # Lưu vào danh sách để tính WER
                    all_references.append(reference)
                    all_predictions.append(transcription)
                    
                    # Ghi ra file
                    file_name = sample.get("file_name", f"sample_{i}")
                    f.write(f"{file_name}\t{reference}\t{transcription}\n")
                    
                    # Giải phóng bộ nhớ
                    del input_features, generated_ids
                    if i % 20 == 19:
                        gc.collect()
                        torch.cuda.empty_cache()
            
            # Tính WER
            normalizer = BasicTextNormalizer()
            norm_refs = [normalizer(ref) for ref in all_references]
            norm_preds = [normalizer(pred) for pred in all_predictions]
            
            wer_score = wer(norm_refs, norm_preds) * 100
            print(f"Evaluation complete. WER: {wer_score:.2f}%")
            
            return {"wer": wer_score}
        
        # Chạy simple_evaluation
        simple_results = simple_evaluation(
            model=whisper_medical.model,
            tokenizer=whisper_medical.processor.tokenizer,
            test_dataset=test_dataset,
            result_file=os.path.join(output_dir, "simple_results.txt")
        )
        
        # Lưu kết quả
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(simple_results, f, indent=2)
        
        print(f"Simple evaluation results saved to {args.output}")
   
    # So sánh với Whisper cơ bản nếu được yêu cầu
    if args.compare_baseline:
        print("\nEvaluating baseline Whisper model:")
        
        baseline_whisper = WhisperMedical(model_id="openai/whisper-base", freeze_encoder=False)
        
        # Sử dụng simple_evaluation để đánh giá baseline - đáng tin cậy hơn
        baseline_results = simple_evaluation(
            model=baseline_whisper.model,
            tokenizer=baseline_whisper.processor.tokenizer,
            test_dataset=test_dataset,
            result_file=os.path.join(output_dir, "baseline_results.txt")
        )
        
        # So sánh kết quả
        main_wer = full_results.get("wer", simple_results.get("wer", 0))
        baseline_wer = baseline_results.get("wer", 0)
        
        print("\n=== Comparison ===")
        print(f"Fine-tuned model WER: {main_wer:.2f}%")
        print(f"Baseline model WER: {baseline_wer:.2f}%")
        print(f"Improvement: {baseline_wer - main_wer:.2f}%")
        
        # Lưu kết quả so sánh
        comparison_results = {
            "fine_tuned_model": {"wer": main_wer},
            "baseline_model": {"wer": baseline_wer},
            "improvement": baseline_wer - main_wer
        }
        
        comparison_output = os.path.join(output_dir, "comparison_results.json")
        with open(comparison_output, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to {comparison_output}")
"""

def debug_prediction_process(model, tokenizer, test_dataset, output_dir):
    """
    Thực hiện debug chi tiết quá trình dự đoán
    """
    print("\n=== STARTING DETAILED PREDICTION DEBUG ===")
    
    # Đặt model ở chế độ đánh giá
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Kiểm tra dataset
    print("\n1. Dataset inspection:")
    print(f"Dataset size: {len(test_dataset)}")
    sample = test_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key} shape: {value.shape}")
    
    # 2. Kiểm tra data collator
    print("\n2. Testing WhisperDataCollator:")
    collator = WhisperDataCollator(tokenizer)
    batch = collator([test_dataset[0], test_dataset[1]])
    print(f"Collated batch keys: {batch.keys()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key} shape: {value.shape}")
    
    # 3. Kiểm tra model.forward
    print("\n3. Testing model.forward:")
    
    # Chuyển batch lên device
    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    try:
        with torch.no_grad():
            # Chỉ sử dụng các keys cần thiết
            model_inputs = {
                "input_features": batch_device["input_features"],
                "labels": batch_device["labels"]
            }
            
            outputs = model(**model_inputs)
        
        print("model.forward succeeded!")
        print(f"  Loss: {outputs.loss}")
        print(f"  Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"Error in model.forward: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Kiểm tra model.generate
    print("\n4. Testing model.generate:")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                batch_device["input_features"],
                max_length=256
            )
        
        print("model.generate succeeded!")
        print(f"  Generated ids shape: {generated_ids.shape}")
        
        # Decode kết quả
        transcriptions = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        for i, trans in enumerate(transcriptions):
            print(f"  Transcription {i}: {trans}")
    except Exception as e:
        print(f"Error in model.generate: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Kiểm tra prediction_step
    print("\n5. Testing prediction_step:")
    
    # Tạo training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=2,
        remove_unused_columns=False,
    )
    
    # Tạo trainer để kiểm tra prediction_step
    debug_trainer = DebugWhisperMedicalTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator
    )
    
    try:
        loss, logits, labels = debug_trainer.prediction_step(
            model, batch_device, prediction_loss_only=False
        )
        
        print("prediction_step succeeded!")
        print(f"  Loss: {loss}")
        print(f"  Logits shape: {logits.shape if logits is not None else None}")
        print(f"  Labels shape: {labels.shape if labels is not None else None}")
        
        # Nếu prediction_step thành công, kiểm tra quá trình đánh giá với một tập nhỏ
        small_dataset = torch.utils.data.Subset(test_dataset, range(3))
        print("\nTesting evaluate with a very small subset (3 samples):")
        
        try:
            results = debug_trainer.evaluate(eval_dataset=small_dataset)
            print(f"evaluate succeeded with results: {results}")
        except Exception as e:
            print(f"Error in evaluate: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error in prediction_step: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Mô phỏng quá trình evaluation_loop
    print("\n6. Simulating evaluation_loop:")
    
    # Tạo dataloader
    dataloader = DataLoader(
        small_dataset,
        batch_size=1,
        collate_fn=collator,
        drop_last=False
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Mô phỏng evaluation_loop
    all_preds = []
    all_labels = []
    
    print("Processing batches:")
    for step, inputs in enumerate(dataloader):
        print(f"  Batch {step}:")
        print(f"    Keys: {inputs.keys()}")
        
        # Chuyển inputs lên device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Dự đoán
        try:
            with torch.no_grad():
                outputs = model(
                    input_features=inputs["input_features"],
                    labels=inputs["labels"]
                )
                
                # Lấy logits và labels
                logits = outputs.logits
                labels = inputs["labels"]
                
                print(f"    Forward pass successful")
                print(f"    Loss: {outputs.loss}")
                print(f"    Logits shape: {logits.shape}")
                
                # Lưu dự đoán và nhãn
                predictions = logits.argmax(dim=-1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                all_preds.append(predictions)
                all_labels.append(labels)
        except Exception as e:
            print(f"    Error in batch {step}: {e}")
            continue
    
    # Kiểm tra dự đoán và nhãn
    if all_preds and all_labels:
        print("\nGot predictions and labels:")
        print(f"  Number of prediction batches: {len(all_preds)}")
        print(f"  Number of label batches: {len(all_labels)}")
        
        # Concatenate
        try:
            import numpy as np
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            print(f"  Concatenated predictions shape: {all_preds.shape}")
            print(f"  Concatenated labels shape: {all_labels.shape}")
            
            # Thử compute_metrics
            eval_preds = EvalPrediction(predictions=all_preds, label_ids=all_labels)
            
            metrics = compute_metrics_whisper_baseline_debug(
                eval_preds=eval_preds,
                tokenizer=tokenizer,
                result_dir=output_dir
            )
            
            print(f"  Computed metrics: {metrics}")
        except Exception as e:
            print(f"  Error in concatenation or metrics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No predictions or labels collected")
    
    print("\n=== PREDICTION DEBUG COMPLETED ===")
    
    return {
        "debug_complete": True,
        "dataset_size": len(test_dataset)
    }
    
    
def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Whisper medical")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa mô hình đã huấn luyện")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Đường dẫn đến file JSONL test data")
    parser.add_argument("--test_audio_dir", type=str, required=True, help="Thư mục chứa audio test")
    parser.add_argument("--bias_words_file", type=str, required=True, help="Đường dẫn đến file bias words")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn đến file kết quả đánh giá")
    parser.add_argument("--compare_baseline", action="store_true", help="So sánh với Whisper cơ bản")
    parser.add_argument("--debug_mode", action="store_true", help="Chạy ở chế độ debug chi tiết")
    
    args = parser.parse_args()
    
    # Thiết lập output file và thư mục
    if args.output is None:
        eval_dir = os.path.join(os.path.dirname(args.model_dir), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        model_name = os.path.basename(args.model_dir)
        args.output = os.path.join(eval_dir, f"{model_name}_evaluation.json")
    
    # Tạo thư mục output
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc bias words từ file
    with open(args.bias_words_file, 'r', encoding='utf-8') as f:
        bias_words = [line.strip() for line in f if line.strip()]
    
    bias_words_string = ", ".join(bias_words)
    
    print(f"Loaded {len(bias_words)} bias words.")
    
    # Tạo hoặc tải mô hình
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
    
    # Nếu ở chế độ debug, thực hiện debug quá trình predict
    if args.debug_mode:
        debug_prediction_process(
            model=whisper_medical.model,
            tokenizer=whisper_medical.processor.tokenizer,
            test_dataset=test_dataset,
            output_dir=output_dir
        )
        return
    
    # Cài đặt training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,  # Batch size nhỏ để giảm thiểu lỗi
        eval_accumulation_steps=4,     # Tăng giá trị này nếu cần thiết
        remove_unused_columns=False,
        do_eval=True,
        report_to="none",
    )
    
    # Sử dụng data collator từ code gốc của bạn
    data_collator = WhisperDataCollator(whisper_medical.processor)
    
    # Tạo debug trainer
    trainer = DebugWhisperMedicalTrainer(
        model=whisper_medical.model,
        args=training_args,
        tokenizer=whisper_medical.processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics_whisper_baseline_debug(
            eval_preds=eval_preds,
            tokenizer=whisper_medical.processor.tokenizer,
            result_dir=output_dir
        )
    )
    
    # Đánh giá mô hình - thử với một tập nhỏ trước
    print("\nTesting evaluation with a small subset first:")
    small_dataset = torch.utils.data.Subset(test_dataset, range(5))
    
    try:
        small_results = trainer.evaluate(eval_dataset=small_dataset)
        print(f"Small evaluation succeeded with results: {small_results}")
        
        # Nếu đánh giá tập nhỏ thành công, tiếp tục với tập đầy đủ
        print("\nRunning full evaluation:")
        full_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Full evaluation results: {full_results}")
        
        # Lưu kết quả
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error in trainer.evaluate: {e}")
        import traceback
        traceback.print_exc()
        
        # Nếu trainer.evaluate() gặp lỗi, sử dụng simple_evaluation
        print("\nFalling back to simple_evaluation:")
        
        def simple_evaluation(model, tokenizer, test_dataset, result_file):
            """
            Hàm đánh giá đơn giản không sử dụng trainer
            """
            model.eval()
            device = next(model.parameters()).device
            
            all_references = []
            all_predictions = []
            
            # Mở file kết quả
            with open(result_file, "w", encoding="utf-8") as f:
                f.write("file_name\treference\tprediction\n")
                
                # Xử lý từng mẫu
                for i in range(len(test_dataset)):
                    if i % 10 == 0:
                        print(f"Processing sample {i+1}/{len(test_dataset)}")
                    
                    # Lấy mẫu và đưa lên device
                    sample = test_dataset[i]
                    input_features = sample["input_features"].unsqueeze(0).to(device)
                    
                    # Chạy inference không có decoder_input_ids
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_features,
                            max_length=256
                        )
                    
                    # Decode kết quả
                    transcription = tokenizer.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0].strip()
                    
                    # Lấy transcription tham chiếu
                    reference = sample["transcript"]
                    
                    # Lưu vào danh sách để tính WER
                    all_references.append(reference)
                    all_predictions.append(transcription)
                    
                    # Ghi ra file
                    file_name = sample.get("file_name", f"sample_{i}")
                    f.write(f"{file_name}\t{reference}\t{transcription}\n")
                    
                    # Giải phóng bộ nhớ
                    del input_features, generated_ids
                    if i % 20 == 19:
                        gc.collect()
                        torch.cuda.empty_cache()
            
            # Tính WER
            normalizer = BasicTextNormalizer()
            norm_refs = [normalizer(ref) for ref in all_references]
            norm_preds = [normalizer(pred) for pred in all_predictions]
            
            wer_score = wer(norm_refs, norm_preds) * 100
            print(f"Evaluation complete. WER: {wer_score:.2f}%")
            
            return {"wer": wer_score}
        
        # Chạy simple_evaluation
        simple_results = simple_evaluation(
            model=whisper_medical.model,
            tokenizer=whisper_medical.processor.tokenizer,
            test_dataset=test_dataset,
            result_file=os.path.join(output_dir, "simple_results.txt")
        )
        
        # Lưu kết quả
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(simple_results, f, indent=2)
        
        print(f"Simple evaluation results saved to {args.output}")
    
    # So sánh với Whisper cơ bản nếu được yêu cầu
    if args.compare_baseline:
        print("\nEvaluating baseline Whisper model:")
        
        baseline_whisper = WhisperMedical(model_id="openai/whisper-base", freeze_encoder=False)
        
        # Sử dụng simple_evaluation để đánh giá baseline - đáng tin cậy hơn
        baseline_results = simple_evaluation(
            model=baseline_whisper.model,
            tokenizer=baseline_whisper.processor.tokenizer,
            test_dataset=test_dataset,
            result_file=os.path.join(output_dir, "baseline_results.txt")
        )
        
        # So sánh kết quả
        main_wer = full_results.get("wer", simple_results.get("wer", 0))
        baseline_wer = baseline_results.get("wer", 0)
        
        print("\n=== Comparison ===")
        print(f"Fine-tuned model WER: {main_wer:.2f}%")
        print(f"Baseline model WER: {baseline_wer:.2f}%")
        print(f"Improvement: {baseline_wer - main_wer:.2f}%")
        
        # Lưu kết quả so sánh
        comparison_results = {
            "fine_tuned_model": {"wer": main_wer},
            "baseline_model": {"wer": baseline_wer},
            "improvement": baseline_wer - main_wer
        }
        
        comparison_output = os.path.join(output_dir, "comparison_results.json")
        with open(comparison_output, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to {comparison_output}")

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

