"""
Launch LoRA Fine-Tuning
"""

import os

from config import (
    MODEL_ID,
    DATASET_PATH,
    LORA_OUTPUT,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_STEPS,
)


def main():

    command = f"""
accelerate launch train_text_to_image_lora.py \
--pretrained_model_name_or_path {MODEL_ID} \
--train_data_dir {DATASET_PATH} \
--resolution {IMAGE_SIZE} \
--train_batch_size {BATCH_SIZE} \
--learning_rate {LEARNING_RATE} \
--max_train_steps {NUM_STEPS} \
--output_dir {LORA_OUTPUT}
"""

    os.system(command)


if __name__ == "__main__":
    main()