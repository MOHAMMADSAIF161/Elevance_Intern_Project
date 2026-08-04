"""
preprocess.py

Preprocess text using the Stable Diffusion v1.5 CLIP tokenizer and
text encoder. Generates token IDs and prompt embeddings.
"""

import os
import json
import numpy as np
import torch


class TextPreprocessor:
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

    def preprocess(self, prompt):
        """
        Tokenize the prompt and generate embeddings.

        Parameters
        ----------
        prompt : str
            Input text prompt.

        Returns
        -------
        inputs : BatchEncoding
            Tokenized prompt.
        prompt_embeds : torch.Tensor
            CLIP text embeddings.
        """

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                **inputs
            ).last_hidden_state

        return inputs, prompt_embeds

    def save_outputs(
        self,
        inputs,
        prompt_embeds,
        output_dir="outputs"
    ):
        """
        Save token IDs and embeddings.
        """

        os.makedirs(output_dir, exist_ok=True)

        # Save token IDs
        token_ids = inputs["input_ids"][0].cpu().tolist()

        token_file = os.path.join(
            output_dir,
            "tokens.json"
        )

        with open(token_file, "w") as f:
            json.dump(token_ids, f, indent=4)

        # Save embeddings
        embedding_file = os.path.join(
            output_dir,
            "prompt_embeddings.npy"
        )

        np.save(
            embedding_file,
            prompt_embeds[0].cpu().numpy()
        )

        print(f"Saved Token IDs : {token_file}")
        print(f"Saved Embeddings: {embedding_file}")