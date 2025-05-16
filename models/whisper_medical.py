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
    
    def __init__(self, model_id="openai/whisper-base", freeze_encoder=True):
        """
        Khởi tạo mô hình Whisper
        
        Args:
            model_id: ID của pretrained model
            freeze_encoder: Có đóng băng encoder hay không
        """
        self.processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        # Cấu hình để không sử dụng forced_decoder_ids 
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if freeze_encoder:
            self._freeze_encoder()
    
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
    
    def create_medical_terms_mapping(self, bias_words_list):
        """
        Tạo mapping từ token_id sang loại thuật ngữ y tế từ danh sách bias words
        
        Args:
            bias_words_list: Danh sách các bias words
        
        Returns:
            Dictionary ánh xạ từ token_id sang loại "DRUGCHEMICAL"
        """
        medical_terms_mapping = {}
        
        for term in bias_words_list:
            tokens = self.processor.tokenizer.encode(term.lower(), add_special_tokens=False)
            for token in tokens:
                medical_terms_mapping[token] = "DRUGCHEMICAL"
        
        return medical_terms_mapping
    
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
        Nhận dạng âm thanh với description và bias words
        
        Args:
            audio_path: Đường dẫn đến file âm thanh
            description: Mô tả về nội dung âm thanh (có thể None)
            bias_words: Chuỗi bias words (có thể None)
        
        Returns:
            Văn bản được nhận dạng
        """
        # Chuẩn bị prompt
        prompt = "<|startoftranscript|>"
        if description:
            if bias_words:
                prompt = f"<|startofprev|>{description} Medical terms: {bias_words}<|endofprev|><|startoftranscript|>"
            else:
                prompt = f"<|startofprev|>{description}<|endofprev|><|startoftranscript|>"
        
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
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        # Generate transcript
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                decoder_input_ids=prompt_ids,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
        
        # Decode kết quả
        transcription = self.processor.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()
        
        return transcription

# class WhisperMedical:
#     """
#     Lớp bao bọc Whisper với các tùy chỉnh cho y tế
#     """
    
#     def __init__(self, model_id=BASE_MODEL, freeze_encoder=FREEZE_ENCODER):
#         """
#         Khởi tạo mô hình Whisper
        
#         Args:
#             model_id: ID của pretrained model
#             freeze_encoder: Có đóng băng encoder hay không
#         """
#         self.processor = WhisperProcessor.from_pretrained(model_id)
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         if freeze_encoder:
#             self._freeze_encoder()
    
#     def _freeze_encoder(self):
#         """Đóng băng encoder để chỉ fine-tune decoder"""
#         # Kiểm tra cấu trúc của mô hình Whisper
#         if hasattr(self.model, "encoder"):
#             for param in self.model.encoder.parameters():
#                 param.requires_grad = False
#             print("Frozen encoder (model.encoder)")
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
#             for param in self.model.model.encoder.parameters():
#                 param.requires_grad = False
#             print("Frozen encoder (model.model.encoder)")
#         else:
#             print("WARNING: Không thể xác định encoder trong mô hình Whisper. Bỏ qua việc đóng băng encoder.")
#             # Kiểm tra tất cả thuộc tính của mô hình để debug
#             print("Available attributes:", [attr for attr in dir(self.model) if not attr.startswith('_')])
    
#     def create_medical_terms_mapping(self, bias_file=None):
#         """
#         Tạo mapping từ token_id sang loại thuật ngữ y tế từ file bias words
        
#         Args:
#             bias_file: Đường dẫn đến file chứa bias words
        
#         Returns:
#             Dictionary ánh xạ từ token_id sang loại thuật ngữ y tế
#         """
#         return generate_medical_terms_mapping_from_file(self.processor.tokenizer, bias_file)
    
#     def save(self, output_dir):
#         """Lưu cả mô hình và processor"""
#         self.model.save_pretrained(output_dir)
#         self.processor.save_pretrained(output_dir)
#         print(f"Đã lưu mô hình và processor vào {output_dir}")
    
#     def load(self, model_dir):
#         """Tải mô hình và processor từ thư mục đã lưu"""
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
#         self.processor = WhisperProcessor.from_pretrained(model_dir)
#         self.model.to(self.device)
#         print(f"Đã tải mô hình và processor từ {model_dir}")
    
#     def transcribe(self, audio_path, description=None, bias_words=None):
#         """
#         Nhận dạng âm thanh với description và bias words tùy chọn
        
#         Args:
#             audio_path: Đường dẫn đến file âm thanh
#             description: Mô tả về nội dung âm thanh (có thể None)
#             bias_words: Chuỗi bias words (có thể None)
        
#         Returns:
#             Văn bản được nhận dạng
#         """
#         # Tạo prompt
#         prompt = "<SOT>"
#         if description:
#             if bias_words:
#                 prompt = create_prompt(description, bias_words)
#             else:
#                 prompt = f"<SOP> {description} <SOT>"
        
#         # Xử lý âm thanh
#         audio, sr = librosa.load(audio_path, sr=16000)
#         input_features = self.processor(
#             audio, 
#             sampling_rate=16000, 
#             return_tensors="pt"
#         ).input_features.to(self.device)
        
#         # Tokenize prompt
#         prompt_ids = self.processor.tokenizer(
#             prompt, 
#             return_tensors="pt"
#         ).input_ids.to(self.device)
        
#         # Generate transcript - với phiên bản API mới
#         try:
#             # Phương pháp mới (không sử dụng forced_decoder_ids)
#             predicted_ids = self.model.generate(
#                 input_features,
#                 decoder_input_ids=prompt_ids,
#                 language="en"  # Chỉ định ngôn ngữ là tiếng Anh
#             )
#         except Exception as e:
#             print(f"Error with new API: {e}")
#             try:
#                 # Thử phương pháp cũ hơn
#                 generation_config = self.model.generation_config
#                 generation_config.forced_decoder_ids = None  # Tắt forced_decoder_ids
                
#                 predicted_ids = self.model.generate(
#                     input_features,
#                     decoder_input_ids=prompt_ids,
#                     language="en",
#                     generation_config=generation_config
#                 )
#             except Exception as e2:
#                 print(f"Error with fallback method: {e2}")
#                 # Phương pháp dự phòng cuối cùng - chỉ sử dụng tham số tối thiểu
#                 predicted_ids = self.model.generate(
#                     input_features,
#                     decoder_input_ids=prompt_ids
#                 )
        
#         # Decode và trả về kết quả
#         transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         return transcription
      
#     def transcribe_batch(self, audio_paths, descriptions=None, bias_words_string=None, batch_size=8):
#         results = []
#         for i in range(0, len(audio_paths), batch_size):
#             batch_paths = audio_paths[i:i+batch_size]
#             batch_descs = descriptions[i:i+batch_size] if descriptions else [None] * len(batch_paths)

#             inputs = []
#             prompts = []

#             for path, desc in zip(batch_paths, batch_descs):
#                 audio, _ = librosa.load(path, sr=16000)
#                 input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
#                 inputs.append(input_features)

#                 if desc:
#                     if bias_words_string:
#                         prompt = create_prompt(desc, bias_words_string)
#                     else:
#                         prompt = f"<SOP> {desc} <SOT>"
#                 else:
#                     prompt = "<SOT>"
#                 prompt_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids[0]
#                 prompts.append(prompt_ids)

#             inputs = torch.stack(inputs).to(self.device)
#             prompt_input = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).to(self.device)

#             try:
#                 predicted_ids = self.model.generate(
#                     inputs,
#                     decoder_input_ids=prompt_input
#                 )
#             except:
#                 generation_config = self.model.generation_config
#                 generation_config.forced_decoder_ids = None
#                 predicted_ids = self.model.generate(
#                     inputs,
#                     decoder_input_ids=prompt_input,
#                     generation_config=generation_config
#                 )

#             decoded = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
#             results.extend(decoded)

#         return results
