import numpy as np
import os

import torch
import torchaudio.transforms as at
import torchaudio
import editdistance
import av
import librosa

import json
import random

class calc_metrics:
    def __init__(self):
        pass

    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        distance = 0
        tokens = 0
        wer_list = []
        processed_preds = []
        processed_refs = []
        exclude = [",", "?", ".", "!", ";"]
        for ref, pred in zip(refs, preds):
            pred = pred.lower()
            pred = "".join(ch for ch in pred if ch not in exclude)
            processed_preds.append(pred)
            processed_refs.append(ref)  # do not process ref
            cur_dist = editdistance.distance(pred.split(" "), ref.split(" "))
            cur_tokens = len(ref.split(" "))
            wer_list.append(cur_dist / cur_tokens)
            distance += cur_dist
            tokens += cur_tokens

        return {"wer": distance / tokens}, (wer_list, processed_preds, processed_refs)

def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    with av.open(wave_path, metadata_errors="ignore") as container:
        decode = container.decode(audio=0)
        aframes_list = [frame.to_ndarray() for frame in decode]
        aframes = np.concatenate(aframes_list, 1)
        # Convert to float32 for processing. We normalize by dividing by 32768.0 (2^15) to get range [-1, 1]
        wav = torch.from_numpy(aframes).float() / 32768.0
        wav = wav.mean(dim=0)  # Taking the mean to convert from stereo to mono
        cur_sample_rate = container.streams.audio[0].rate
        if cur_sample_rate != sample_rate:
            resampler = at.Resample(orig_freq=cur_sample_rate, new_freq=sample_rate)
            wav = resampler(wav)
        if wav.mean() == 0:
            print(wave_path, "empty!")
    return wav

class PromptWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, jsonl_data, phase, feature_extractor, tokenizer, prompt=False, audio_type=".wav", sample_rate=16000, random=False):
        super().__init__()
        self.phase = phase
        self.base_path = base_path
        self.jsonl_data = jsonl_data
        self.sample_rate = sample_rate
        self.prompt = prompt
        self.random_prompt = random
        self.data = []
        self.prompt_pool = []
        self.audio_type = audio_type
        self._load_data()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def _initialize_prompt_pool(self):
        # Initialize the prompt pool with a list of prompts
        jsonl_path = os.path.join("data", self.jsonl_data, f"{self.phase}.jsonl")

        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Jsonl file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    json_data = json.loads(line)
                    prompt = json_data.get("description", "")
                    if prompt:
                        self.prompt_pool.append(prompt)
                except json.JSONDecodeError:
                    print(f"[WARNING] Ignore JSON line: {line.strip()}")
        
    # def _load_data(self):
    #     # Walk through the directory structure
    #     # print("Base path:", self.base_path)
    #     for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
    #         wav_files = [f for f in files if f.endswith(f'{self.audio_type}')]
    #         json_files = [f for f in files if f.endswith('.json')]
    #         for wav_file in wav_files:
    #             base_name = os.path.splitext(wav_file)[0] # Remove the file extension
    #             json_file_name = f"{base_name}.json"
    #             if json_file_name in json_files:
    #                 json_file_path = os.path.join(root, json_file_name)
    #                 # Open the json file and extract "text" and "prompt"
    #                 with open(json_file_path, 'r', encoding='utf-8') as json_file:
    #                     json_data = json.load(json_file)
    #                     text = json_data.get("text", "")
    #                     prompt = json_data.get("prompt", "")
    #                     #random_prompt = random.choice(self.prompt_pool) if self.prompt_pool else ""
    #                     if self.prompt_pool: # truyen vao tu args
    #                         random_prompt = random.choice(self.prompt_pool)
    #                     else:
    #                         random_prompt = ""
    #                 self.data.append([os.path.join(root, wav_file),
    #                     prompt,
    #                     random_prompt,
    #                     text
    #                 ])
    
    def _load_data(self):
        jsonl_path = os.path.join("data", self.jsonl_data, f"{self.phase}.jsonl")

        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Jsonl file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    json_data = json.loads(line)
                    audio_filename = json_data.get("file", "")
                    text = json_data.get("text", "")
                    prompt = json_data.get("description", "")

                    if not audio_filename:
                        continue

                    random_prompt = random.choice(self.prompt_pool) if self.prompt_pool else ""

                    self.data.append([
                        audio_filename,
                        prompt,
                        random_prompt,
                        text
                    ])
                except json.JSONDecodeError:
                    print(f"[WARNING] Ignore JSON line: {line.strip()}")

    def __len__(self):
        return len(self.data)
    
    # input feature
    # labels = prompt + labels
    def __getitem__(self, id):
        audio_filename, prompt, random_prompt, raw_text = self.data[id]
        audio_path = os.path.join(self.base_path, self.phase, audio_filename)

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
            processed_audio = torch.tensor(processed_audio[0])

            # Encode label (transcript)
            encoded_label = self.tokenizer.encode(raw_text.lower(), add_special_tokens=False)

            if self.prompt:
                if self.random_prompt and 'train' in self.phase:
                    if torch.rand([]) < 0.05:
                        prompt_text = random_prompt
                    else:
                        prompt_text = prompt
                else:
                    prompt_text = prompt

                encoded_prompt = self.tokenizer.encode(prompt_text.lower(), add_special_tokens=False)

                if len(encoded_prompt) > 190:
                    encoded_prompt = encoded_prompt[:190]

                # Add special tokens between prompt and label
                start_of_prev = self.tokenizer.convert_tokens_to_ids("<|startofprev|>")
                start_of_transcript = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

                full_sequence = [start_of_prev] + encoded_prompt + [start_of_transcript] + encoded_label
                full_sequence_tensor = torch.tensor(full_sequence)

                return {
                    "input_features": processed_audio,
                    "labels": full_sequence_tensor
                }
            else:
                raise ValueError("prompt must be used.")

        except Exception as e:
            print(f"Error processing sample {id}, file: {audio_path}, error: {str(e)}")
            raise e
