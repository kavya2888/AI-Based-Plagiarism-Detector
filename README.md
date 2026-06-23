# AI-Based Plagiarism Detector

An AI-powered plagiarism detection system that uses **sentence embeddings** and **cosine similarity** to identify both exact and paraphrased plagiarism between text documents — going beyond simple keyword or string matching.

---

## How It Works

1. Input texts are cleaned and preprocessed (lowercased, punctuation stripped, whitespace normalized).
2. Each text (or sentence) is converted into a numerical vector ("embedding") using a pretrained Sentence-Transformers model.
3. Cosine similarity is calculated between embeddings to measure how semantically close two pieces of text are — even if the wording is different.
4. Based on configurable similarity thresholds, the system classifies the result as:
   * **Original**
   * **Possible Plagiarism**
   * **High Risk of Plagiarism**

Because this approach compares *meaning* rather than exact words, it can catch paraphrased plagiarism that traditional string-matching tools would miss.

---

## Project Structure

```
ai-plagiarism-detector/
│
├── app/
│   ├── __init__.py
│   ├── detector.py        # Core plagiarism detection logic (embeddings + similarity)
│   ├── utils.py            # Text cleaning, preprocessing helpers
│   └── config.py           # Threshold values, model name, settings
│
├── data/
│   ├── sample1.txt          # Input test file
│   ├── sample2.txt          # Input test file (paraphrased version of sample1)
│   └── corpus/
│       ├── doc1.txt          # Reference document (climate change)
│       ├── doc2.txt          # Reference document (solar system)
│       └── doc3.txt          # Reference document (AI — overlaps with sample1/sample2)
│
├── tests/
│   ├── __init__.py
│   └── test_detector.py     # Unit tests for plagiarism detection logic
│
├── main.py                  # Entry point of the project
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

* **Python 3.9+**
* [`sentence-transformers`](https://www.sbert.net/) — for generating semantic text embeddings
* [`scikit-learn`](https://scikit-learn.org/) — for cosine similarity calculations
* `numpy` — for numerical operations
* `pytest` — for unit testing

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/ai-plagiarism-detector.git
   cd ai-plagiarism-detector
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

Run the main script to compare the sample files and check them against the reference corpus:

```bash
python main.py
```

This will:
* Compare `data/sample1.txt` against `data/sample2.txt`
* Print an overall similarity score and verdict
* Show a sentence-by-sentence breakdown of matches
* Check `sample1.txt` against every document in `data/corpus/`

---

## Running Tests

Unit tests are written using `pytest`:

```bash
pytest tests/ -v
```

---

## Example Output

```text
============================================================
AI-BASED PLAGIARISM DETECTOR
============================================================
Loading embedding model: all-MiniLM-L6-v2 ...
Model loaded successfully.


============================================================
PAIRWISE COMPARISON: sample1.txt vs sample2.txt
============================================================
Similarity Score : 0.9123
Similarity %     : 91.23%
Verdict          : High Risk of Plagiarism

============================================================
SENTENCE-LEVEL BREAKDOWN
============================================================

[1] ⚠️  FLAGGED (Similarity: 96.42%)
    Source  : Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.
    Matched : Artificial intelligence refers to the simulation of human thinking by computer systems.

[2] ⚠️  FLAGGED (Similarity: 93.81%)
    Source  : Machine learning, a subset of AI, enables systems to learn from data and improve their performance over time without being explicitly programmed.
    Matched : Machine learning is a branch of AI that allows systems to learn from data and get better over time without explicit programming.

============================================================
CORPUS CHECK: sample1.txt vs data/corpus/*.txt
============================================================
Loaded 3 reference document(s) from corpus.

Document   : doc3.txt
Similarity : 88.57%
Verdict    : Possible Plagiarism
----------------------------------------
Document   : doc1.txt
Similarity : 12.04%
Verdict    : Original
----------------------------------------
Document   : doc2.txt
Similarity : 9.76%
Verdict    : Original
----------------------------------------

Done.
```

*(Exact similarity percentages may vary slightly depending on model version.)*

---

## Configuration

You can adjust detection sensitivity in `app/config.py`:

| Setting                | Description                                      | Default            |
|-------------------------|---------------------------------------------------|---------------------|
| `MODEL_NAME`            | Pretrained SentenceTransformer model to use       | `all-MiniLM-L6-v2`  |
| `PLAGIARISM_THRESHOLD`  | Score above which text is flagged as plagiarized  | `0.75`              |
| `HIGH_RISK_THRESHOLD`   | Score above which text is "High Risk"             | `0.90`              |
| `MIN_CHUNK_LENGTH`      | Minimum sentence length to be considered          | `10`                |

---

## Future Enhancements

* Web-based UI for uploading and checking documents
* REST API for integration with other systems (e.g., LMS platforms)
* Support for PDF/DOCX file uploads
* Multi-language plagiarism detection
* Visual similarity heatmaps between documents
* Database-backed corpus for large-scale reference checking

---

## License

This project is open-source and available under the [MIT License](LICENSE).
