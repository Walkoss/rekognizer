import os
from typing import List

import requests

import numpy as np

FACENET_HOST = os.environ.get("FACENET_HOST")
FACENET_PORT = os.environ.get("FACENET_PORT")
MODEL_NAME = "facenet"
SIGNATURE_NAME = "calculate_embeddings"
THRESHOLD = 0.8


class Facenet:
    @staticmethod
    def get_embeddings(images: np.array):
        json = {
            "signature_name": SIGNATURE_NAME,
            "inputs": {"images": images.tolist(), "phase": False},
        }

        response = requests.post(
            f"http://{FACENET_HOST}:{FACENET_PORT}/v1/models/{MODEL_NAME}:predict",
            json=json,
        )

        return response.json()["outputs"]

    @staticmethod
    def get_similarities(embeddings: np.array) -> List[bool]:
        result = [True]

        if len(embeddings) > 1:
            reference_embedding = embeddings[0]
            for embedding in embeddings[1:]:
                dist = np.sqrt(
                    np.sum(np.square(np.subtract(reference_embedding, embedding)))
                )
                result.append(bool(dist < THRESHOLD))

        return result
