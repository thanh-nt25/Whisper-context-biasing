import os
from pathlib import Path

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_OUTPUT = "/kaggle/working/"

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
OUTPUT_DIR = KAGGLE_OUTPUT
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# DATASET_DIR = "medical-syn-med-75/medical-united-syn-med-75"
DATASET_DIR = "data/medical-united-syn-med-75"

DATA_DIR = "data"
TRAIN_JSONL = os.path.join(DATA_DIR, "all_text_train_description.jsonl")
DEV_JSONL = os.path.join(DATA_DIR, "all_text_dev_description.jsonl")
BIAS_WORDS_FILE = os.path.join(DATA_DIR, "Blist", "bias_list_30.txt")
DEV_AUDIO_DIR = os.path.join(DATASET_DIR, "medical", "dev")
TRAIN_AUDIO_DIR = os.path.join(DATASET_DIR, "medical", "train")
TEST_AUDIO_DIR = os.path.join(DATASET_DIR, "medical", "test")


BASE_MODEL = "openai/whisper-base"
FREEZE_ENCODER = True


BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
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
    "SURGERY": 1.1 # weak           
}


PROMPT_TEMPLATE = "<SOP> {description} Medical terms: {bias_words}. <SOT>"