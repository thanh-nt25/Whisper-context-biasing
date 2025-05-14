"""
Mô hình Whisper với các tùy chỉnh cho y tế
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
import sys
from pathlib import Path
import librosa

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from config.config import BASE_MODEL, FREEZE_ENCODER
from data_utils.data_processor import create_prompt, generate_medical_terms_mapping_from_file

class WhisperMedical:
    """
    Lớp bao bọc Whisper với các tùy chỉnh cho y tế
    """
    
    def __init__(self, model_id=BASE_MODEL, freeze_encoder=FREEZE_ENCODER):
        """
        Khởi tạo mô hình Whisper
        
        Args:
            model_id: ID của pretrained model
            freeze_encoder: Có đóng băng encoder hay không
        """
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Đóng băng encoder để chỉ fine-tune decoder"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def create_medical_terms_mapping(self, bias_file=None):
        """
        Tạo mapping từ token_id sang loại thuật ngữ y tế từ file bias words
        
        Args:
            bias_file: Đường dẫn đến file chứa bias words
        
        Returns:
            Dictionary ánh xạ từ token_id sang loại thuật ngữ y tế
        """
        return generate_medical_terms_mapping_from_file(self.processor.tokenizer, bias_file)
    
    def save(self, output_dir):
        """Lưu cả mô hình và processor"""
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Đã lưu mô hình và processor vào {output_dir}")
    
    def load(self, model_dir):
        """Tải mô hình và processor từ thư mục đã lưu"""
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Đã tải mô hình và processor từ {model_dir}")
    
    def transcribe(self, audio_path, description=None, bias_words=None):
        """
        Nhận dạng âm thanh với description và bias words tùy chọn
        
        Args:
            audio_path: Đường dẫn đến file âm thanh
            description: Mô tả về nội dung âm thanh (có thể None)
            bias_words: Chuỗi bias words (có thể None)
        
        Returns:
            Văn bản được nhận dạng
        """
        # Tạo prompt
        prompt = "<SOT>"
        if description:
            if bias_words:
                prompt = create_prompt(description, bias_words)
            else:
                prompt = f"<SOP> {description} <SOT>"
        
        # Xử lý âm thanh
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Tokenize prompt
        prompt_ids = self.processor.tokenizer(
            prompt, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Generate transcript
        predicted_ids = self.model.generate(
            input_features,
            decoder_input_ids=prompt_ids
        )
        
        # Decode và trả về kết quả
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription