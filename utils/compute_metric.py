from transformers import WhisperTokenizer
import evaluate
from tqdm import tqdm
import torch

import os

import re
import unicodedata
import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s


# metric
metric = evaluate.load("wer")

def compute_wer(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    normalizer = BasicTextNormalizer()
    
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')
    sot_token_id = 20257
    pad_token_id = 20256

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    total_wer = 0
    batch_results = []
    results = []
    batch_size = 4
    
    print("\n\nDone inference!")
    print("Start decoding and calculating WER...")

    cutted_label_ids = []
    cutted_pred_ids = []

    # cut prompt (dang truoc <SOT>) => luon luon cat duoc ke ca khi ko co prompt
    for i in tqdm(range(len(pred_ids))):
        label_tensor = torch.tensor(label_ids[i])
        pred_tensor = torch.tensor(pred_ids[i])

        label_pos = (label_tensor == sot_token_id).nonzero(as_tuple=False).flatten()
        pred_pos = (pred_tensor == sot_token_id).nonzero(as_tuple=False).flatten()

        label_start = label_pos[0].item() + 1 if len(label_pos) > 0 else 0
        pred_start = pred_pos[0].item() + 1 if len(pred_pos) > 0 else 0

        cutted_label_ids.append(label_tensor[label_start:])
        cutted_pred_ids.append(pred_tensor[pred_start:])
      

    for i in tqdm(range(0, len(cutted_pred_ids), batch_size)):
        batch_pred_ids = cutted_pred_ids[i:i + batch_size]
        batch_label_ids = cutted_label_ids[i:i + batch_size]

        pre_strs = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(batch_label_ids, skip_special_tokens=True)

        filtered_pre_strs = []
        filtered_label_strs = []

        for pred, label in zip(pre_strs, label_strs):
            if label != 'ignore_time_segment_in_scoring':
                # 'ignore_time_segment_in_scoring'
                filtered_pre_strs.append(normalizer(pred))
                filtered_label_strs.append(normalizer(label))

        if filtered_pre_strs and filtered_label_strs:
                pre_strs, label_strs = zip(*zip(filtered_pre_strs, filtered_label_strs))
        else:
            pre_strs, label_strs = (), ()
        results.extend(zip(label_strs, pre_strs))

    with open(os.path.join("/kaggle/working", 'refs_and_pred.txt'), 'w') as f:
        for ref, pred in results:
            f.write(f'Ref : {ref}\n')
            f.write(f'Pred:{pred}\n\n')

          
    pre_strs = [pred for _, pred in results]
    label_strs = [ref for ref, _ in results]
    
    total_wer = 100 * metric.compute(predictions=pre_strs, references=label_strs)

    return {
        'wer': total_wer
    }
