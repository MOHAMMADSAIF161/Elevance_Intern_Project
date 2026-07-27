import os
import torch
from diffusers import StableDiffusionPipeline

from config import (
    MODEL_ID,
    LORA_PATH,
    GENERATED_IMAGES,
    PROMPT,
)


def main():

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_lora_weights(LORA_PATH)

    image = pipe(
        PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    os.makedirs(GENERATED_IMAGES, exist_ok=True)

    output_path = os.path.join(
        GENERATED_IMAGES,
        "sunflower.png"
    )

    image.save(output_path)

    print(f"Image saved at {output_path}")


if __name__ == "__main__":
    main()