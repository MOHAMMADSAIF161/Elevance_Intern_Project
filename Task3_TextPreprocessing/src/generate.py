"""
generate.py

Generates an image using Stable Diffusion v1.5 from
precomputed CLIP prompt embeddings.
"""

import torch


class ImageGenerator:
    def __init__(self, pipe):
        """
        Parameters
        ----------
        pipe : StableDiffusionPipeline
            Loaded Stable Diffusion v1.5 pipeline.
        """
        self.pipe = pipe
        self.device = pipe.device
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    def create_negative_embeddings(self):
        """
        Create embeddings for an empty negative prompt.
        """

        negative_prompt = ""

        negative_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        negative_inputs = {
            k: v.to(self.device)
            for k, v in negative_inputs.items()
        }

        with torch.no_grad():
            negative_prompt_embeds = (
                self.text_encoder(**negative_inputs)
                .last_hidden_state
            )

        return negative_prompt_embeds

    def generate(
        self,
        prompt_embeds,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42,
    ):
        """
        Generate image from prompt embeddings.
        """

        negative_prompt_embeds = self.create_negative_embeddings()

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def save(self, image, output_path="outputs/generated_image.png"):
        """
        Save generated image.
        """
        image.save(output_path)
        print(f"Image saved to: {output_path}")