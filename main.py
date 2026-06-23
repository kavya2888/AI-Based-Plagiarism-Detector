"""
main.py
--------
Entry point for the AI-Based Plagiarism Detector.

Loads sample text files, runs plagiarism detection between them,
performs a sentence-level breakdown, and checks one sample against
a corpus of reference documents — printing all results to the console.
"""

from app.detector import PlagiarismDetector
from app.utils import read_text_file, load_corpus
from app.config import SAMPLE_FILE_1, SAMPLE_FILE_2, CORPUS_DIR


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_pairwise_comparison(detector: PlagiarismDetector):
    """Compares sample1.txt and sample2.txt directly."""
    print_header("PAIRWISE COMPARISON: sample1.txt vs sample2.txt")

    text1 = read_text_file(SAMPLE_FILE_1)
    text2 = read_text_file(SAMPLE_FILE_2)

    result = detector.compare_texts(text1, text2)

    print(f"Similarity Score : {result['similarity_score']}")
    print(f"Similarity %     : {result['similarity_percentage']}")
    print(f"Verdict          : {result['verdict']}")

    return text1, text2


def run_sentence_level_comparison(detector: PlagiarismDetector, text1: str, text2: str):
    """Breaks down the comparison sentence-by-sentence for more detail."""
    print_header("SENTENCE-LEVEL BREAKDOWN")

    results = detector.compare_sentence_level(text1, text2)

    if not results:
        print("Not enough sentence-level content to compare.")
        return

    for i, match in enumerate(results, start=1):
        flag = "⚠️  FLAGGED" if match["flagged"] else "OK"
        print(f"\n[{i}] {flag} (Similarity: {match['similarity_percentage']})")
        print(f"    Source  : {match['source_sentence']}")
        print(f"    Matched : {match['matched_sentence']}")


def run_corpus_check(detector: PlagiarismDetector):
    """Checks sample1.txt against all documents in the corpus directory."""
    print_header("CORPUS CHECK: sample1.txt vs data/corpus/*.txt")

    text1 = read_text_file(SAMPLE_FILE_1)
    corpus = load_corpus(CORPUS_DIR)

    print(f"Loaded {len(corpus)} reference document(s) from corpus.\n")

    results = detector.check_against_corpus(text1, corpus)

    for result in results:
        print(f"Document   : {result['document']}")
        print(f"Similarity : {result['similarity_percentage']}")
        print(f"Verdict    : {result['verdict']}")
        print("-" * 40)


def main():
    print_header("AI-BASED PLAGIARISM DETECTOR")

    detector = PlagiarismDetector()

    # 1. Direct comparison between two sample files
    text1, text2 = run_pairwise_comparison(detector)

    # 2. Sentence-level breakdown of that comparison
    run_sentence_level_comparison(detector, text1, text2)

    # 3. Check sample1 against the full reference corpus
    run_corpus_check(detector)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
