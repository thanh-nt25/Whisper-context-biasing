from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
import sys
from pathlib import Path
import librosa

class WhisperMedicalForConditionalGeneration:
    def __init__(self, model_id="openai/whisper-base", freeze_encoder=True):
        self.processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if freeze_encoder:
            print("Freezing encoder")
            self._freeze_encoder()
        else:
            print("Load full base model without freezing encoder")
    
    def _freeze_encoder(self):
        """Đóng băng encoder để chỉ fine-tune decoder"""
        # Kiểm tra cấu trúc của mô hình Whisper
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Frozen encoder (model.encoder)")
        elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            print("Frozen encoder (model.model.encoder)")
        else:
            print("WARNING: Không thể xác định encoder trong mô hình Whisper. Bỏ qua việc đóng băng encoder.")
        
    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Đã lưu mô hình và processor vào {output_dir}")
    
    def load(self, model_dir):
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Đã tải mô hình và processor từ {model_dir}")
    
