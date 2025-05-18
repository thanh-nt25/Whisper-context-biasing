# import evaluate
from transformers import WhisperTokenizer
from jiwer import wer
from tqdm import tqdm

import numpy as np
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
# def compute_wer(pred, prompts):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    normalizer = BasicTextNormalizer()
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    total_wer = 0
    results = []
    batch_size = 4
    print("\n\nDone inference!")
    print("Start decoding and calculating WER...")

    # cutted_label_ids = []
    # cutted_pred_ids = []

    # if len(prompts) != 0:
    #     i la sample, cutted se lay toan bo doan ve sau, prompts o dang truoc
    #     for i in tqdm(range(0, len(pred_ids))):
    #         cutted_pred_ids.append(pred_ids[i][len(prompts[i][0])+1:])
    #         cutted_label_ids.append(label_ids[i][len(prompts[i][0])+1:])

    for i in tqdm(range(0, len(pred_ids), batch_size)):
        batch_pred_ids = pred_ids[i:i + batch_size]
        batch_label_ids = label_ids[i:i + batch_size]

        pre_strs = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(batch_label_ids, skip_special_tokens=True)
        # pre_strs, label_strs = zip(*[(normalizer(pred), normalizer(label)) for pred, label in zip(pre_strs, label_strs) if label != 'ignore_time_segment_in_scoring'])

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

    return {'wer': total_wer}
