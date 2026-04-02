import numpy as np

EPS = 1e-6

class Similarity:
    def cosine(self, v1, v2):

        v1 = np.asarray(v1)
        v2 = np.asarray(v2)

        if v1.shape != v2.shape:
            raise ValueError(f"Embedding mismatch: {v1.shape} vs {v2.shape}")

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS)

    def relevance(self, v_c, v_e):
        return (1 + self.cosine(v_c, v_e)) / 2