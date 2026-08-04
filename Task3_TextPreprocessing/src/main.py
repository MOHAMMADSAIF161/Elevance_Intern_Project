"""
main.py

Complete Text-to-Image pipeline using
Hugging Face Transformers and
Stable Diffusion v1.5
"""

from diffusers import StableDiffusionPipeline
import torch

from preprocess import TextPreprocessor
from visualize import EmbeddingVisualizer
from generate import ImageGenerator


def main():

    # -------------------------------------------------
    # Load Stable Diffusion Pipeline
    # -------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    pipe = pipe.to(device)

    # -------------------------------------------------
    # User Prompt
    # -------------------------------------------------

    prompt = (
        "A beautiful red rose blooming "
        "in a green garden under bright sunlight."
    )

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------

    preprocessor = TextPreprocessor(pipe)

    inputs, prompt_embeds = preprocessor.preprocess(prompt)

    preprocessor.save_outputs(
        inputs,
        prompt_embeds
    )

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    visualizer = EmbeddingVisualizer()

    embeddings = visualizer.load_embeddings(
        "outputs/prompt_embeddings.npy"
    )

    visualizer.plot(embeddings)

    # -------------------------------------------------
    # Image Generation
    # -------------------------------------------------

    generator = ImageGenerator(pipe)

    image = generator.generate(prompt_embeds)

    generator.save(image)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()