# import evaluate
from transformers import WhisperTokenizer
from jiwer import wer
from tqdm import tqdm

import numpy as np
import os

# wer_metric = evaluate.load("wer")

class BasicTextNormalizer:
    def __call__(self, s: str):
        import re
        import unicodedata
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove parentheses
        s = unicodedata.normalize("NFKD", s)
        s = ''.join([c for c in s if not unicodedata.category(c).startswith("M")])
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
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