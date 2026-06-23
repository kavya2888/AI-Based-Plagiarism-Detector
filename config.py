"""
config.py
----------
Centralized configuration for the AI-Based Plagiarism Detector.
Holds model settings, similarity thresholds, and file paths.
"""

import os

# ----------------------------
# Model Settings
# ----------------------------
# Pretrained SentenceTransformer model used to generate text embeddings.
# 'all-MiniLM-L6-v2' is lightweight and fast, ideal for similarity tasks.
MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------
# Similarity Thresholds
# ----------------------------
# Cosine similarity score above which two texts are flagged as plagiarized.
# Range: 0.0 (no similarity) to 1.0 (identical meaning).
PLAGIARISM_THRESHOLD = 0.75

# Similarity score above which a "high risk" warning is shown.
HIGH_RISK_THRESHOLD = 0.90

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")

SAMPLE_FILE_1 = os.path.join(DATA_DIR, "sample1.txt")
SAMPLE_FILE_2 = os.path.join(DATA_DIR, "sample2.txt")

# ----------------------------
# Text Preprocessing Settings
# ----------------------------
# Minimum number of characters required in a sentence/chunk to be considered
# for comparison (filters out noise like empty lines or stray punctuation).
MIN_CHUNK_LENGTH = 10

# Whether to lowercase text before comparison.
LOWERCASE = True

# Whether to remove punctuation before comparison.
REMOVE_PUNCTUATION = True
