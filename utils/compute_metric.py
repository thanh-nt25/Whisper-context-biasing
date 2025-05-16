# import evaluate
from transformers import WhisperTokenizer
from jiwer import wer
from tqdm import tqdm

import numpy as np
import os

import re
import unicodedata
import regex

from rapidfuzz.distance import Opcodes

# wer_metric = evaluate.load("wer")

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

def compute_metrics_whisper_with_prompt(eval_preds, tokenizer, prompt_ids_list=None, save_path="/kaggle/working"):
    """
    eval_preds: tuple(predictions, labels)
    prompt_ids_list: List[List[int]], chứa các prompt_ids tương ứng từng sample
    """
    from jiwer import wer
    normalizer = BasicTextNormalizer()
    predictions, labels = eval_preds

    tokenizer.pad_token_id = tokenizer.pad_token_id or 0
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds, decoded_labels = [], []

    for i in range(len(predictions)):
        pred_ids = predictions[i]
        label_ids = labels[i]
        prompt_ids = prompt_ids_list[i] if prompt_ids_list is not None else []

        # Cat prompt
        cut_len = len(prompt_ids) + 1 if len(prompt_ids) > 0 else 0
        pred_text = tokenizer.decode(pred_ids[cut_len:], skip_special_tokens=True)
        label_text = tokenizer.decode(label_ids[cut_len:], skip_special_tokens=True)

        pred_text = normalizer(pred_text)
        label_text = normalizer(label_text)

        if label_text.strip() and label_text != "ignore_time_segment_in_scoring":
            decoded_preds.append(pred_text)
            decoded_labels.append(label_text)

    output_file = os.path.join(save_path, "refs_and_pred.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for ref, pred in zip(decoded_labels, decoded_preds):
            f.write(f"Ref: {ref}\n")
            f.write(f"Pred: {pred}\n\n")

    if not decoded_labels:
        return {"wer": 100.0}

    return {"wer": 100 * wer(decoded_labels, decoded_preds)}