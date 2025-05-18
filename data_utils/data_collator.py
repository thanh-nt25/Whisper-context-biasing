from transformers import WhisperFeatureExtractor, WhisperTokenizer
from typing import Any, Dict, List, Union
import torch
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare feature extractor, tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-base", language="Hindi", task="transcribe"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(batch):
    # audio를 16kHZ로 load
    audio = batch["audio"]
    # padding & trucation 적용,log-mel spectrogram으로 변환
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# define a data collator
@dataclass
class DataCollatorSpeechS2SWhitPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if features[0]["prompt"].numel() > 0:
            prompt_features = [{"input_ids": feature["prompt"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            combined_feature = []
            for prompt, label in zip(prompt_features, label_features):
                prompt_ids = (
                    prompt["input_ids"].tolist()
                    if isinstance(prompt["input_ids"], torch.Tensor)
                    else prompt["input_ids"]
                )
                label_ids = (
                    label["input_ids"].tolist()
                    if isinstance(label["input_ids"], torch.Tensor)
                    else label["input_ids"]
                )

                combined_ids = prompt_ids + [50257] + label_ids  # noi token
                combined_feature.append({"input_ids": combined_ids})

            labels_batch = self.processor.tokenizer.pad(
                combined_feature, return_tensors="pt"
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            # Ensure 'prompts' is a list of tensors and pad them to the same length
            prompts = [
                prompt["input_ids"].clone().detach() for prompt in prompt_features
            ]
            max_len = max([prompt.size(0) for prompt in prompts])
            padded_prompts = [
                torch.nn.functional.pad(
                    prompt,
                    (0, max_len - prompt.size(0)),
                    value=self.processor.tokenizer.pad_token_id,
                )
                for prompt in prompts
            ]

            # Stack the padded prompts
            # batch["prompts"] = torch.stack(padded_prompts)

            batch["labels"] = labels

        else:
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels
            # batch["prompts"] = None

        return batch

