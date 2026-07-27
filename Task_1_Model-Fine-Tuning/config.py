"""
Configuration file for LoRA Fine-Tuning
"""

# ----------------------------
# Base Model
# ----------------------------

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ----------------------------
# Dataset
# ----------------------------

DATASET_PATH = "datasets"

# ----------------------------
# LoRA
# ----------------------------

LORA_OUTPUT = "model/lora_output"
LORA_PATH = LORA_OUTPUT

# ----------------------------
# Inference
# ----------------------------

PROMPT = "A beautiful sunflower in a garden"

GENERATED_IMAGES = "outputs/generated_images"

# ----------------------------
# Training Hyperparameters
# ----------------------------

IMAGE_SIZE = 512

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

NUM_STEPS = 1000

SEED = 42