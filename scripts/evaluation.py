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

# from trainer.CustomTrainer import CustomTrainer

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
from config.config import DATA_ROOT, DATA_DIR, JSONL_DATA



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation whisper medical")
    parser.add_argument("--output", type=str, default=None, help="output path")
    parser.add_argument("--bias_weight", type=float, default=1.5, help="Bias weight of Weight Cross Entropy")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="base input data dir")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="middle input data dir")
    parser.add_argument("--jsonl_data", type=str, default=JSONL_DATA, help="path to jsonl metadata")
    parser.add_argument("--prompt", action="store_true", help="whether to use prompt to decoder")
    parser.add_argument("--random", action="store_true", help="context perturbation")
    parser.add_argument('--batch', type=int, default=8, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    
    args = parser.parse_args()
    
    args.random = True # 5% random prompt
    
    print("Bool of using prompt: ", args.prompt)
    print("Bool of using random: ", args.random)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # args.prompt = True
    
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
                                     feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.prompt)    
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

    dataloader = DataLoader(
        dataset=data_test,
        batch_size=25,  # hoặc 1, tùy bạn muốn kiểm tra bao nhiêu sample
        collate_fn=data_collator,
        shuffle=False
    )
    
    batch = next(iter(dataloader))

    # In thông tin batch
    print("Keys in batch:", batch.keys())
    print("Shape of input_features:", batch["input_features"].shape)
    print("Shape of labels:", batch["labels"].shape)

    # In bias_spans
    if "bias_spans" in batch:
        print("Shape of bias_spans:", batch["bias_spans"].shape)
        print("bias_spans:")
        print(batch["bias_spans"])

        # Optional: decode từng span
        for i in range(batch["bias_spans"].shape[0]):
            print(f"\nSample {i}:")
            for j in range(batch["bias_spans"].shape[1]):
                span = batch["bias_spans"][i, j].tolist()
                if all(tok == 50256 for tok in span):
                    continue
                decoded = tokenizer.decode([tok for tok in span if tok != 50256])
                print(f"  Span {j}: {span} => '{decoded}'")
    else:
        print("No bias_spans found in batch!")

    # config = WhisperConfig.from_pretrained("openai/whisper-base.en")
    # model = WhisperMedicalForConditionalGeneration(config)
    
    # here
    # model = WhisperForConditionalGenerationWeightCE.from_pretrained("openai/whisper-base.en")
    # model.freeze_encoder()
    
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    
    # root_path = "results/"
    # os.makedirs(os.path.join(root_path), exist_ok=True)
    
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=os.path.join(root_path, "models"),
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     predict_with_generate=True,
    #     generation_max_length=225,
    #     remove_unused_columns=False,
    #     gradient_accumulation_steps=8,
    #     # evaluation_strategy="epoch",
    #     # include_inputs_for_metrics=True,
    #     include_for_metrics=["inputs"],
    #     save_strategy="epoch",
    #     logging_strategy="epoch",
    #     learning_rate=1e-5,
    #     num_train_epochs=10,
    #     weight_decay=0.01,
    #     warmup_steps=500,
    #     # save_total_limit=3,
    #     # load_best_model_at_end=True,
    #     report_to = []
    # )
    
    # trainer = Seq2SeqTrainer(
    #     args=training_args,
    #     model=model,
    #     # train_dataset=data_train,
    #     # eval_dataset=data_eval,
    #     data_collator=data_collator,
    #     tokenizer=processor.feature_extractor,
    #     compute_metrics=compute_wer,
    # )

    # if (len(data_test) == 0):
    #     print("No test data found!")
    #     exit(0)
    # print("length of test data: ", len(data_test))

    # print("Starting evaluation!")
    # result = trainer.evaluate(data_test)
    # print(result)
    
    
    