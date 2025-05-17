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
import random
import json

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from data_utils.data_processor import load_jsonl, get_audio_path, load_bias_words, create_prompt

class WhisperMedicalDataset(Dataset):
    """
    Dataset cho fine-tuning Whisper với medical prompts từ JSONL
    """
    def __init__(self, jsonl_file, processor, audio_dir, bias_words_string=None, max_prompt_length=190, random_prob=0.05):
        """
        Khởi tạo dataset
        
        Args:
            jsonl_file: Đường dẫn đến file JSONL chứa metadata
            processor: WhisperProcessor để xử lý dữ liệu
            audio_dir: Thư mục chứa file audio
            bias_words_string: Chuỗi các bias words cố định (có thể None)
            max_prompt_length: Độ dài tối đa cho prompt
            random_prob: Xác suất cho context perturbation
        """
        self.processor = processor
        self.audio_dir = audio_dir
        self.bias_words_string = bias_words_string
        self.max_prompt_length = max_prompt_length
        self.random_prob = random_prob
        self.data = self._load_jsonl(jsonl_file)
        self.prompt_pool = [item["description"] for item in self.data]  # Tạo pool các description
        
    def _load_jsonl(self, jsonl_file):
        """Load data từ file JSONL"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Lấy thông tin từ JSONL
        item = self.data[idx]
        file_name = item['file']
        transcript = item['text']
        description = item['description']
        
        # Lấy đường dẫn âm thanh
        audio_path = os.path.join(self.audio_dir, file_name)
        
        # Tạo prompt với bias words
        if self.bias_words_string:
            prompt = f"{description} Medical terms: {self.bias_words_string}"
        else:
            prompt = description
            
        # Áp dụng context perturbation trong quá trình huấn luyện
        if 'train' in self.audio_dir and random.random() < self.random_prob:
            prompt = random.choice(self.prompt_pool)
            
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
        
        # Tokenize prompt với token đặc biệt
        prompt_with_tokens = f"<|startofprev|>{prompt}<|endofprev|><|startoftranscript|>"
        
        # Tokenize và cắt bỏ nếu prompt quá dài
        prompt_ids = self.processor.tokenizer(
            prompt_with_tokens, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.squeeze(0)
        
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[:self.max_prompt_length]
        
        # Tokenize transcript
        # with self.processor.tokenizer.as_target_processor():
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "input_features": input_features,
            "decoder_input_ids": prompt_ids,
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
    processor: any

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [feature["input_features"] for feature in features]
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True) # make four dimension input features
        
        if input_features.dim() == 4 and input_features.shape[1] == 1:
          input_features = input_features.squeeze(1)
        
        # Pad decoder_input_ids (prompt)
        # decoder_input_ids = [feature["decoder_input_ids"] for feature in features]
        # decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
        #     decoder_input_ids, 
        #     batch_first=True, 
        #     padding_value=self.processor.tokenizer.pad_token_id
        # )
        
        # Tạo decoder attention mask
        # decoder_attention_mask = torch.ones_like(decoder_input_ids)
        # decoder_attention_mask[decoder_input_ids == self.processor.tokenizer.pad_token_id] = 0
        
        # Pad labels (transcript)
        labels = [feature["labels"] for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100  # -100 là giá trị đặc biệt để bỏ qua khi tính loss
        )
        
        # Lưu metadata
        transcripts = [feature.get("transcript", "") for feature in features]
        file_names = [feature.get("file_name", "") for feature in features]
        descriptions = [feature.get("description", "") for feature in features]
        
        batch = {
            "input_features": input_features,
            # "decoder_input_ids": decoder_input_ids,
            # "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "transcripts": transcripts,
            "file_names": file_names,
            "descriptions": descriptions
        }
        
        return batch

# """
# Dataset và DataCollator cho Whisper medical fine-tuning dựa trên JSONL
# """

# import torch
# from torch.utils.data import Dataset
# import librosa
# import os
# from dataclasses import dataclass
# from typing import Dict, List, Union
# from transformers import WhisperProcessor
# import sys
# from pathlib import Path

# # Thêm thư mục gốc vào path
# sys.path.append(str(Path(__file__).parent.parent.absolute()))
# from data_utils.data_processor import load_jsonl, get_audio_path, load_bias_words, create_prompt

# class WhisperMedicalDataset(Dataset):
#     """
#     Dataset cho fine-tuning Whisper với medical prompts từ JSONL
#     """
#     def __init__(self, jsonl_file, processor, audio_dir, bias_words_string=None):
#         self.data = load_jsonl(jsonl_file)
#         self.processor = processor
#         self.audio_dir = audio_dir
#         self.bias_words_string = bias_words_string
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         file_name = item['file']
#         transcript = item['text']
#         description = item['description']

#         # Đường dẫn file audio
#         audio_path = get_audio_path(file_name, self.audio_dir)

#         # Tạo prompt
#         prompt = create_prompt(description, self.bias_words_string)  # Ex: "<|startoftranscript|> Desc... Terms: ..."

#         # Ghép prompt + transcript → đây sẽ là cả input và label
#         full_text = prompt + " " + transcript

#         # Tokenize toàn bộ decoder sequence
#         tokenized = self.processor.tokenizer(full_text, return_tensors="pt")
#         decoder_input_ids = tokenized.input_ids.squeeze(0)
#         labels = decoder_input_ids.clone()

#         # Mask phần prompt trong labels để không tính loss
#         prompt_len = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.size(1)
#         labels[:prompt_len] = -100

#         # Xử lý input audio
#         try:
#             audio, sr = librosa.load(audio_path, sr=16000)
#             input_features = self.processor(
#                 audio,
#                 sampling_rate=16000,
#                 return_tensors="pt"
#             ).input_features.squeeze(0)
#         except Exception as e:
#             print(f"Error processing audio file {audio_path}: {e}")
#             input_features = torch.zeros((1, 80, 3000))

#         return {
#             "input_features": input_features,
#             "decoder_input_ids": decoder_input_ids,
#             "labels": labels,
#             "transcript": transcript,
#             "file_name": file_name,
#             "description": description
#         }

# @dataclass
# class WhisperDataCollator:
#     """
#     Collator cho batching dữ liệu Whisper
#     """
#     processor: WhisperProcessor

#     def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
#         input_features = [f["input_features"] for f in features]
#         input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)

#         decoder_input_ids = [f["decoder_input_ids"] for f in features]
#         decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
#             decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
#         )

#         labels = [f["labels"] for f in features]
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=-100
#         )

#         transcripts = [f["transcript"] for f in features]
#         file_names = [f["file_name"] for f in features]
#         descriptions = [f["description"] for f in features]

#         return {
#             "input_features": input_features,
#             "decoder_input_ids": decoder_input_ids,
#             "labels": labels,
#             "transcripts": transcripts,
#             "file_names": file_names,
#             "descriptions": descriptions
#         }
