"""
Dataset và DataCollator cho Whisper medical fine-tuning dựa trên JSONL
"""

import torch
from torch.utils.data import Dataset
import librosa
import os
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import WhisperProcessor
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from data_utils.data_processor import load_jsonl, get_audio_path, load_bias_words, create_prompt

class WhisperMedicalDataset(Dataset):
    """
    Dataset cho fine-tuning Whisper với medical prompts từ JSONL
    """
    def __init__(self, jsonl_file, processor, audio_dir, bias_words_string=None):
        """
        Khởi tạo dataset
        
        Args:
            jsonl_file: Đường dẫn đến file JSONL chứa metadata
            processor: WhisperProcessor để xử lý dữ liệu
            audio_dir: Thư mục chứa file audio
            bias_words_string: Chuỗi các bias words cố định (có thể None)
        """
        self.data = load_jsonl(jsonl_file)
        self.processor = processor
        self.audio_dir = audio_dir
        self.bias_words_string = bias_words_string
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Lấy thông tin từ JSONL
        item = self.data[idx]
        file_name = item['file']
        transcript = item['text']
        description = item['description']
        
        # Lấy đường dẫn âm thanh
        audio_path = get_audio_path(file_name, self.audio_dir)
        
        # Tạo prompt
        prompt = create_prompt(description, self.bias_words_string)
        
        # Xử lý âm thanh
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.squeeze(0)
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            # Trả về tensor rỗng trong trường hợp lỗi
            input_features = torch.zeros((1, 80, 3000))
        
        # Tokenize prompt
        prompt_ids = self.processor.tokenizer(
            prompt, 
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Tokenize transcript (labels)
        # with self.processor.as_target_processor():
        #     labels = self.processor(transcript, return_tensors="pt").input_ids.squeeze(0)
        # Tokenize transcript (labels)
        labels = self.processor.tokenizer(
            transcript, 
            return_tensors="pt"
        ).input_ids.squeeze(0)

        
        return {
            "input_features": input_features,
            "prompt_ids": prompt_ids,
            "labels": labels,
            "transcript": transcript,
            "file_name": file_name,
            "description": description
        }

@dataclass
class WhisperDataCollator:
    """
    Collator cho batching dữ liệu Whisper
    """
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [feature["input_features"] for feature in features]
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        
        # Pad prompt_ids
        prompt_ids = [feature["prompt_ids"] for feature in features]
        prompt_ids = torch.nn.utils.rnn.pad_sequence(prompt_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        
        # Pad labels
        labels = [feature["labels"] for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Lưu thông tin thêm (không được pad)
        transcripts = [feature["transcript"] for feature in features]
        file_names = [feature["file_name"] for feature in features]
        descriptions = [feature["description"] for feature in features]
        
        return {
            "input_features": input_features,
            "decoder_input_ids": prompt_ids,
            "labels": labels,
            "transcripts": transcripts,
            "file_names": file_names,
            "descriptions": descriptions
        }