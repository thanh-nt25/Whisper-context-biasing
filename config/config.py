
"""
Cấu hình và tham số cho quá trình huấn luyện
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
OUTPUT_DIR = "/kaggle/working/output"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


DATA_DIR = "/kaggle/working/Whisper-context-biasing/data"
DATASET_DIR = "/kaggle/input/medical-syn-med-75/medical-united-syn-med-75"
TRAIN_JSONL = os.path.join(DATA_DIR, "all_texts_train_description.jsonl")
DEV_JSONL = os.path.join(DATA_DIR, "all_texts_dev_description.jsonl")
BIAS_WORDS_FILE = os.path.join(DATA_DIR, "Blist", "bias_list_30.txt")
TRAIN_AUDIO_DIR = os.path.join(DATASET_DIR, "train")
DEV_AUDIO_DIR = os.path.join(DATASET_DIR, "dev")


BASE_MODEL = "openai/whisper-base"
FREEZE_ENCODER = True


BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3  
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50
FP16 = True
RANDOM_CONTEXT_PROB = 0.05
RANDOM_CONTEXTS_SIZE = 50


WEIGHT_FACTORS = {
    "DRUGCHEMICAL": 2.0,
    "DIAGNOSTICS": 1.8,
    "MEDDEVICETECHNIQUE": 1.8,
    
}


PROMPT_TEMPLATE = "<SOP> {description} Medical terms: {bias_words}. <SOT>"