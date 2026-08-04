"""
visualize.py

Visualize CLIP text embeddings as a heatmap.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class EmbeddingVisualizer:
    def __init__(self):
        pass

    def load_embeddings(self, embedding_path):
        """
        Load saved embeddings.
        """
        embeddings = np.load(embedding_path)
        return embeddings

    def plot(
        self,
        embeddings,
        output_path="outputs/embedding_heatmap.png"
    ):
        """
        Create and save an embedding heatmap.
        """

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(14, 6))

        plt.imshow(
            embeddings,
            aspect="auto",
            interpolation="nearest"
        )

        plt.title("CLIP Text Embedding Heatmap")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Token Position")

        plt.colorbar()

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        print(f"Heatmap saved to: {output_path}")