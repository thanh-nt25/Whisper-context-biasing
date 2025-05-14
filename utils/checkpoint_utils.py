# %%writefile utils/checkpoint_utils.py
import os
import time
import torch

CHECKPOINT_INTERVAL = 15  
last_checkpoint_time = time.time()

def save_checkpoint_if_needed(model, output_dir, step):
    global last_checkpoint_time
    current_time = time.time()
    if current_time - last_checkpoint_time > CHECKPOINT_INTERVAL * 60:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{output_dir}/checkpoint_{step}.pt")
        last_checkpoint_time = current_time
        print(f"Saved checkpoint at step {step}")
        return True
    return False