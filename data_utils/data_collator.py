from transformers import WhisperFeatureExtractor, WhisperTokenizer
from typing import Any, Dict, List, Optional, Union
import torch
from dataclasses import dataclass
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare feature extractor, tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-base", language="en", task="transcribe"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# # define a data collator
# @dataclass
# class DataCollatorSpeechS2SWhitPadding:
#     processor: Any

#     def __call__(
#         self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
#     ) -> Dict[str, torch.Tensor]:
#         input_features = [
#             {"input_features": feature["input_features"]} for feature in features
#         ]
#         batch = self.processor.feature_extractor.pad(
#             input_features, return_tensors="pt"
#         )

#         if features[0]["prompt"].numel() > 0:
#             prompt_features = [{"input_ids": feature["prompt"]} for feature in features]
#             label_features = [{"input_ids": feature["labels"]} for feature in features]

#             combined_feature = []
#             for prompt, label in zip(prompt_features, label_features):
#                 prompt_ids = (
#                     prompt["input_ids"].tolist()
#                     if isinstance(prompt["input_ids"], torch.Tensor)
#                     else prompt["input_ids"]
#                 )
#                 label_ids = (
#                     label["input_ids"].tolist()
#                     if isinstance(label["input_ids"], torch.Tensor)
#                     else label["input_ids"]
#                 )

#                 combined_ids = prompt_ids + [50257] + label_ids  # noi token
#                 combined_feature.append({"input_ids": combined_ids})

#             labels_batch = self.processor.tokenizer.pad(
#                 combined_feature, return_tensors="pt"
#             )

#             labels = labels_batch["input_ids"].masked_fill(
#                 labels_batch.attention_mask.ne(1), -100
#             )

#             if (
#                 (labels[:, 0] == self.processor.tokenizer.bos_token_id)
#                 .all()
#                 .cpu()
#                 .item()
#             ):
#                 labels = labels[:, 1:]

#             # Ensure 'prompts' is a list of tensors and pad them to the same length
#             prompts = [
#                 prompt["input_ids"].clone().detach() for prompt in prompt_features
#             ]
#             max_len = max([prompt.size(0) for prompt in prompts])
#             padded_prompts = [
#                 torch.nn.functional.pad(
#                     prompt,
#                     (0, max_len - prompt.size(0)),
#                     value=self.processor.tokenizer.pad_token_id,
#                 )
#                 for prompt in prompts
#             ]

#             # Stack the padded prompts
#             # batch["prompts"] = torch.stack(padded_prompts) #?? khong lam the nay duoc, phai gop ca vao trong labels

#             batch["labels"] = labels

#         else:
#             label_features = [{"input_ids": feature["labels"]} for feature in features]
#             labels_batch = self.processor.tokenizer.pad(
#                 label_features, return_tensors="pt"
#             )

#             labels = labels_batch["input_ids"].masked_fill(
#                 labels_batch.attention_mask.ne(1), -100
#             )

#             if (
#                 (labels[:, 0] == self.processor.tokenizer.bos_token_id)
#                 .all()
#                 .cpu()
#                 .item()
#             ):
#                 labels = labels[:, 1:]

#             batch["labels"] = labels
#             # batch["prompts"] = None

#         return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`)
            The start-of-prompt token id of the decoder
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            # max_length=self.max_target_length,
            # padding=self.target_padding,
            padding = "longest",
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        # decoder_input_ids are the labels shifted to the right (for teacher forcing)
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch