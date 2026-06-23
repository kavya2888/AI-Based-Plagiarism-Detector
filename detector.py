"""
detector.py
------------
Core plagiarism detection logic.

Uses sentence embeddings (via SentenceTransformers) and cosine similarity
(via scikit-learn) to measure semantic similarity between texts — allowing
detection of paraphrased plagiarism, not just exact word matches.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.config import MODEL_NAME, PLAGIARISM_THRESHOLD, HIGH_RISK_THRESHOLD
from app.utils import clean_text, split_into_sentences, format_percentage


class PlagiarismDetector:
    """
    Detects plagiarism between two texts, or between a text and a corpus
    of reference documents, using sentence embeddings and cosine similarity.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the detector and loads the embedding model.

        Args:
            model_name (str): Name of the pretrained SentenceTransformer model.
        """
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.\n")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a vector embedding for a given text.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Embedding vector.
        """
        return self.model.encode([text])[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes overall cosine similarity between two full texts.

        Args:
            text1 (str): First text.
            text2 (str): Second text.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        cleaned1 = clean_text(text1)
        cleaned2 = clean_text(text2)

        emb1 = self.get_embedding(cleaned1).reshape(1, -1)
        emb2 = self.get_embedding(cleaned2).reshape(1, -1)

        score = cosine_similarity(emb1, emb2)[0][0]
        return float(score)

    def compare_texts(self, text1: str, text2: str) -> dict:
        """
        Compares two texts and returns a structured result including
        the similarity score and a plagiarism verdict.

        Args:
            text1 (str): First text (e.g., submitted document).
            text2 (str): Second text (e.g., reference document).

        Returns:
            dict: Result containing score, percentage, and verdict.
        """
        score = self.compute_similarity(text1, text2)
        verdict = self._get_verdict(score)

        return {
            "similarity_score": round(score, 4),
            "similarity_percentage": format_percentage(score),
            "verdict": verdict,
        }

    def compare_sentence_level(self, text1: str, text2: str) -> list:
        """
        Performs a finer-grained, sentence-by-sentence comparison between
        two texts. Useful for pinpointing exactly which sentences were
        likely copied or paraphrased.

        Args:
            text1 (str): First text.
            text2 (str): Second text.

        Returns:
            list[dict]: List of per-sentence match results, sorted by
                        similarity score (highest first).
        """
        sentences1 = split_into_sentences(text1)
        sentences2 = split_into_sentences(text2)

        if not sentences1 or not sentences2:
            return []

        embeddings1 = self.model.encode(sentences1)
        embeddings2 = self.model.encode(sentences2)

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        results = []
        for i, sentence1 in enumerate(sentences1):
            best_match_idx = int(np.argmax(similarity_matrix[i]))
            best_score = float(similarity_matrix[i][best_match_idx])

            results.append({
                "source_sentence": sentence1,
                "matched_sentence": sentences2[best_match_idx],
                "similarity_score": round(best_score, 4),
                "similarity_percentage": format_percentage(best_score),
                "flagged": best_score >= PLAGIARISM_THRESHOLD,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    def check_against_corpus(self, text: str, corpus: dict) -> list:
        """
        Compares a single text against an entire corpus of reference
        documents and returns similarity scores for each.

        Args:
            text (str): The text to check (e.g., a submitted document).
            corpus (dict): Mapping of {filename: file_content}.

        Returns:
            list[dict]: Sorted list of results (highest similarity first),
                        one entry per corpus document.
        """
        results = []

        for filename, ref_text in corpus.items():
            score = self.compute_similarity(text, ref_text)
            results.append({
                "document": filename,
                "similarity_score": round(score, 4),
                "similarity_percentage": format_percentage(score),
                "verdict": self._get_verdict(score),
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    @staticmethod
    def _get_verdict(score: float) -> str:
        """
        Converts a raw similarity score into a human-readable verdict.

        Args:
            score (float): Cosine similarity score (0 to 1).

        Returns:
            str: One of "Original", "Possible Plagiarism", or
                "High Risk of Plagiarism".
        """
        if score >= HIGH_RISK_THRESHOLD:
            return "High Risk of Plagiarism"
        elif score >= PLAGIARISM_THRESHOLD:
            return "Possible Plagiarism"
        else:
            return "Original"
