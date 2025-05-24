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
    def __init__(self, base_path, jsonl_data, phase, feature_extractor, tokenizer, prompt=False, 
                 bias_list=False, audio_type=".wav", sample_rate=16000, random=False, bias_nums=0, bias_desc = False):
        super().__init__()
        self.phase = phase
        self.base_path = base_path
        self.jsonl_data = jsonl_data
        self.sample_rate = sample_rate
        self.prompt = prompt  # Kích hoạt chiến lược 1 hoặc 3
        self.bias_list = bias_list  # Kích hoạt chiến lược 2 hoặc 3
        self.random_prompt = random
        self.bias_nums = bias_nums  # Số lượng từ trong dãy bias, mặc định 0
        self.data = []
        self.prompt_pool = []
        self.bias_pool = set()  # Pool cho bias words
        self.non_bias_pool = set()  # Pool cho từ không phải bias
        self.audio_type = audio_type
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self._initialize_prompt_pool()
        self._load_data()
        self._initialize_pools()  # Khởi tạo pool bias và non-bias
        self.bias_desc = bias_desc

    def _initialize_prompt_pool(self):
        # Initialize the prompt pool with a list of full prompts
        jsonl_path = os.path.join(self.jsonl_data, f"{self.phase}.jsonl")

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

    def _initialize_pools(self):
        # init bias pool and non bias pool
        jsonl_path = os.path.join(self.jsonl_data, f"{self.phase}.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    json_data = json.loads(line)
                    bias_words = json_data.get("bias_words", [])
                    text = json_data.get("text", "").lower()
                    for word in bias_words:
                        self.bias_pool.add(word.lower())
                    words = text.split()
                    non_bias_words = []
                    for w in words:
                        cleaned_word = ''.join(char for char in w if char not in [",", "?", ".", "!", ";"])
                        if cleaned_word and cleaned_word not in self.bias_pool:
                            non_bias_words.append(cleaned_word)
                    self.non_bias_pool.update(non_bias_words)
                except json.JSONDecodeError:
                    print(f"[WARNING] Ignore JSON line: {line.strip()}")

    def _load_data(self):
        jsonl_path = os.path.join(self.jsonl_data, f"{self.phase}.jsonl")

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
                    bias_per_sample = json_data.get("bias_words", [])

                    if not audio_filename:
                        continue

                    random_prompt = random.choice(self.prompt_pool) if self.prompt_pool else ""

                    self.data.append([
                        audio_filename,
                        prompt,
                        random_prompt,
                        text,
                        bias_per_sample
                    ])
                except json.JSONDecodeError:
                    print(f"[WARNING] Ignore JSON line: {line.strip()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        audio_filename, prompt, random_prompt, raw_text, bias_words = self.data[id]
        audio_path = os.path.join(self.base_path, self.phase, audio_filename)

        bias_spans = []  # list per sample
        for word in bias_words:
            ids = self.tokenizer.encode(word.lower(), add_special_tokens=False)
            if ids:
                bias_spans.append(ids)

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
            processed_audio = torch.tensor(processed_audio[0])

            # Encode label (transcript)
            # encoded_label = self.tokenizer.encode(raw_text.lower(), add_special_tokens=False)
            encoded_label = self.tokenizer.encode(raw_text.lower())

            # default full sequence
            start_of_transcript = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            # full_sequence = [start_of_transcript] + encoded_label
            full_sequence = encoded_label
            full_sequence_tensor = torch.tensor(full_sequence)

            if self.prompt or self.bias_list:
                start_of_prev = self.tokenizer.convert_tokens_to_ids("<|startofprev|>")

                # 1. Only description
                if self.prompt and not self.bias_list:
                    if not self.random_prompt or 'train' not in self.phase:
                        prompt_text = prompt
                    else:
                        if torch.rand([]) < 0.05:
                            # print(f"Truong hop random prompt cho id {id}")
                            prompt_text = random_prompt
                        else:
                            prompt_text = prompt

                    if prompt_text:
                        encoded_prompt = self.tokenizer.encode(prompt_text.lower(), add_special_tokens=False)
                        if len(encoded_prompt) > 190:
                            encoded_prompt = encoded_prompt[:190]
                    else:
                        print(f"Error for extracting prompt of {id}: prompt_text is {'None' if prompt_text is None else 'empty'}")
                        encoded_prompt = []

                    # full_sequence = [start_of_prev] + encoded_prompt + [start_of_transcript] + encoded_label
                    full_sequence = [start_of_prev] + encoded_prompt + encoded_label
                    full_sequence_tensor = torch.tensor(full_sequence)

                # 2. only bias list, 30% bias, 70% non bias
                elif not self.prompt and self.bias_list and self.bias_nums > 0:
                    bias_words_list = []
                    if self.bias_pool and self.non_bias_pool:
                        if torch.rand([]) < 0.05:
                            bias_words_list = random.sample(list(self.non_bias_pool), self.bias_nums)
                        else:
                            # add bias of current sample
                            if bias_words:
                                bias_words_list.extend([word.lower() for word in bias_words])

                            total_bias_needed = max(1, int(self.bias_nums * 0.3))
                            num_bias_to_add = total_bias_needed - len(bias_words_list)
                            if num_bias_to_add > 0 and self.bias_pool:
                                available_bias = list(self.bias_pool - set(bias_words_list))
                                if available_bias:
                                    bias_words_list.extend(random.sample(available_bias, min(num_bias_to_add, len(available_bias))))

                            num_remaining = self.bias_nums - len(bias_words_list)
                            if num_remaining > 0 and self.non_bias_pool:
                                available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                                if available_non_bias:
                                    bias_words_list.extend(random.sample(available_non_bias, min(num_remaining, len(available_non_bias))))

                        if len(bias_words_list) > self.bias_nums:
                            bias_words_list = bias_words_list[:self.bias_nums]
                        while len(bias_words_list) < self.bias_nums and self.non_bias_pool:
                            available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                            if available_non_bias:
                                bias_words_list.append(random.choice(available_non_bias))

                        space_token = self.tokenizer.encode(" ", add_special_tokens=False)
                        encoded_bias = []
                        for i, word in enumerate(bias_words_list):
                            encoded_word = self.tokenizer.encode(word, add_special_tokens=False)
                            encoded_bias.extend(encoded_word)
                            if i < len(bias_words_list) - 1: 
                                encoded_bias.extend(space_token)
                        if not encoded_bias:
                          print(f"Warning: encoded_bias is empty for sample {id}. bias_words_list: {bias_words_list}")
                        

                        # full_sequence = [start_of_prev] + encoded_bias + [start_of_transcript] + encoded_label
                        full_sequence = [start_of_prev] + encoded_bias + encoded_label
                        full_sequence_tensor = torch.tensor(full_sequence)
                    else:
                      raise ValueError(f"bias_pool or non_bias_pool is empty for sample {id}")
                #3. description + bias list
                elif self.prompt and self.bias_list and self.bias_nums > 0 and not self.bias_desc:
                    if not self.random_prompt or 'train' not in self.phase:
                        prompt_text = prompt
                    else:
                        if torch.rand([]) < 0.05:
                            prompt_text = random_prompt
                        else:
                            prompt_text = prompt

                    if prompt_text:
                        encoded_prompt = self.tokenizer.encode(prompt_text.lower(), add_special_tokens=False)
                        if len(encoded_prompt) > 150:
                            encoded_prompt = encoded_prompt[:150]
                    else:
                        print(f"Error for extracting prompt of {id}: prompt_text is {'None' if prompt_text is None else 'empty'}")
                        encoded_prompt = []


                    relate_terms = self.tokenizer.encode("Relate terms: ", add_special_tokens=False)

                    bias_words_list = []
                    if self.bias_pool and self.non_bias_pool:
                        if torch.rand([]) < 0.05:
                            bias_words_list = random.sample(list(self.non_bias_pool), self.bias_nums)
                        else:
                            if bias_words:
                                bias_words_list.extend([word.lower() for word in bias_words])

                            total_bias_needed = max(1, int(self.bias_nums * 0.3))
                            num_bias_to_add = total_bias_needed - len(bias_words_list)
                            if num_bias_to_add > 0 and self.bias_pool:
                                available_bias = list(self.bias_pool - set(bias_words_list))
                                if available_bias:
                                    bias_words_list.extend(random.sample(available_bias, min(num_bias_to_add, len(available_bias))))

                            num_remaining = self.bias_nums - len(bias_words_list)
                            if num_remaining > 0 and self.non_bias_pool:
                                available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                                if available_non_bias:
                                    bias_words_list.extend(random.sample(available_non_bias, min(num_remaining, len(available_non_bias))))

                        if len(bias_words_list) > self.bias_nums:
                            bias_words_list = bias_words_list[:self.bias_nums]
                        while len(bias_words_list) < self.bias_nums and self.non_bias_pool:
                            available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                            if available_non_bias:
                                bias_words_list.append(random.choice(available_non_bias))

                        space_token = self.tokenizer.encode(" ", add_special_tokens=False)
                        encoded_bias = []
                        for i, word in enumerate(bias_words_list):
                            encoded_word = self.tokenizer.encode(word, add_special_tokens=False)
                            encoded_bias.extend(encoded_word)
                            if i < len(bias_words_list) - 1: 
                                encoded_bias.extend(space_token)
                        if not encoded_bias:
                          print(f"Warning: encoded_bias is empty for sample {id}. bias_words_list: {bias_words_list}")
                                


                    # description + "Relate terms:" + bias list + label
                    # full_sequence = [start_of_prev] + encoded_prompt + relate_terms + encoded_bias + [start_of_transcript] + encoded_label
                    full_sequence = [start_of_prev] + encoded_prompt + relate_terms + encoded_bias + encoded_label
                    full_sequence_tensor = torch.tensor(full_sequence)     
                    
                #4. reverse bias list + description
                elif self.prompt and self.bias_list and self.bias_nums > 0 and self.bias_desc:
                    if not self.random_prompt or 'train' not in self.phase:
                        prompt_text = prompt
                    else:
                        if torch.rand([]) < 0.05:
                            prompt_text = random_prompt
                        else:
                            prompt_text = prompt

                    if prompt_text:
                        encoded_prompt = self.tokenizer.encode(prompt_text.lower(), add_special_tokens=False)
                        if len(encoded_prompt) > 150:
                            encoded_prompt = encoded_prompt[:150]
                    else:
                        print(f"Error for extracting prompt of {id}: prompt_text is {'None' if prompt_text is None else 'empty'}")
                        encoded_prompt = []


                    relate_terms = self.tokenizer.encode("Relate terms: ", add_special_tokens=False)

                    bias_words_list = []
                    if self.bias_pool and self.non_bias_pool:
                        if torch.rand([]) < 0.05:
                            bias_words_list = random.sample(list(self.non_bias_pool), self.bias_nums)
                        else:
                            if bias_words:
                                bias_words_list.extend([word.lower() for word in bias_words])

                            total_bias_needed = max(1, int(self.bias_nums * 0.3))
                            num_bias_to_add = total_bias_needed - len(bias_words_list)
                            if num_bias_to_add > 0 and self.bias_pool:
                                available_bias = list(self.bias_pool - set(bias_words_list))
                                if available_bias:
                                    bias_words_list.extend(random.sample(available_bias, min(num_bias_to_add, len(available_bias))))

                            num_remaining = self.bias_nums - len(bias_words_list)
                            if num_remaining > 0 and self.non_bias_pool:
                                available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                                if available_non_bias:
                                    bias_words_list.extend(random.sample(available_non_bias, min(num_remaining, len(available_non_bias))))

                        if len(bias_words_list) > self.bias_nums:
                            bias_words_list = bias_words_list[:self.bias_nums]
                        while len(bias_words_list) < self.bias_nums and self.non_bias_pool:
                            available_non_bias = list(self.non_bias_pool - set(bias_words_list))
                            if available_non_bias:
                                bias_words_list.append(random.choice(available_non_bias))

                        space_token = self.tokenizer.encode(" ", add_special_tokens=False)
                        encoded_bias = []
                        for i, word in enumerate(bias_words_list):
                            encoded_word = self.tokenizer.encode(word, add_special_tokens=False)
                            encoded_bias.extend(encoded_word)
                            if i < len(bias_words_list) - 1: 
                                encoded_bias.extend(space_token)
                                
                        if not encoded_bias:
                          print(f"Warning: encoded_bias is empty for sample {id}. bias_words_list: {bias_words_list}")
                                
                    
                    full_sequence = [start_of_prev] + relate_terms + encoded_bias + encoded_prompt + encoded_label
                    full_sequence_tensor = torch.tensor(full_sequence)                    
               
            return {
                "input_features": processed_audio,
                "labels": full_sequence_tensor,
                "bias_spans": bias_spans
            }

        except Exception as e:
            print(f"Error processing sample {id}, file: {audio_path}, error: {str(e)}")
            raise e