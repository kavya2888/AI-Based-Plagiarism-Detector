"""
utils.py
---------
Utility functions for text cleaning, preprocessing, and file handling
used throughout the plagiarism detection pipeline.
"""

import re
import string
import os

from app.config import MIN_CHUNK_LENGTH, LOWERCASE, REMOVE_PUNCTUATION


def read_text_file(file_path: str) -> str:
    """
    Reads and returns the content of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Raw text content of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, optionally lowercasing,
    and optionally stripping punctuation.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines

    if LOWERCASE:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        text = text.translate(str.maketrans("", "", string.punctuation))

    return text.strip()


def split_into_sentences(text: str) -> list:
    """
    Splits a block of text into sentences using basic punctuation-based
    splitting. Filters out very short/noisy fragments.

    Args:
        text (str): Input text (can be raw or pre-cleaned).

    Returns:
        list[str]: List of sentence-level chunks.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = [
        s.strip() for s in raw_sentences
        if len(s.strip()) >= MIN_CHUNK_LENGTH
    ]

    return sentences


def load_corpus(corpus_dir: str) -> dict:
    """
    Loads all .txt files from a corpus directory into a dictionary.

    Args:
        corpus_dir (str): Path to the corpus directory.

    Returns:
        dict: Mapping of {filename: file_content}.
    """
    corpus = {}

    if not os.path.isdir(corpus_dir):
        raise NotADirectoryError(f"Corpus directory not found: {corpus_dir}")

    for filename in sorted(os.listdir(corpus_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(corpus_dir, filename)
            corpus[filename] = read_text_file(file_path)

    return corpus


def format_percentage(score: float) -> str:
    """
    Converts a similarity score (0.0 - 1.0) into a readable percentage string.

    Args:
        score (float): Similarity score between 0 and 1.

    Returns:
        str: Formatted percentage string, e.g. "87.45%".
    """
    return f"{score * 100:.2f}%"
