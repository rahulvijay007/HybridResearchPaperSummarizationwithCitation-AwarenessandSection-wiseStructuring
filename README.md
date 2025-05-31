# 🧠 Research Paper Summarization: A Hybrid Approach with Citation Awareness & Section-wise Structuring

> A lightweight yet powerful system for summarizing research papers using a combination of extractive (LexRank), abstractive (T5-small/DistilBART), citation-aware graph modeling, and section-wise structured outputs — optimized for low-resource environments with optional Text-to-Speech (TTS) support.

---

## 🚀 Features

- 📄 **PDF Parsing**: Reads academic papers directly from PDF.
- 🧩 **Section-wise Segmentation**: Extracts structured content (abstract, intro, methods, results, etc.).
- 🧠 **Hybrid Summarization**:
  - 🔍 **LexRank** for extractive summaries.
  - 📝 **T5-small / DistilBART** for abstractive generation.
- 🔗 **Citation Graph Modeling**:
  - Graph-based scoring to emphasize cited or cited-by sentences.
  - Fusion of citations into section-aware summaries.
- 🔊 **Optional TTS Output**: Converts the summary into natural speech using LE2E.
- ⚙️ **Lightweight Deployment**: Runs on CPU with minimal memory usage (~8GB RAM).
- 📊 **Evaluation**: Includes ROUGE, BLEU, and BERTScore metrics.

---

## 🗂️ Project Structure

```
research-paper-summarization/
├── data/
│   └── sample_papers/         # PDFs and extracted sections
├── models/
│   └── summarizers/           # LexRank + T5/DistilBART logic
├── utils/
│   └── citation_graph.py      # Graph modeling utilities
├── evaluation/
│   └── metrics.py             # ROUGE, BLEU, BERTScore
├── tts/
│   └── le2e_tts.py            # Optional TTS module
├── main.py                    # Entrypoint to run everything
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/research-paper-summarization.git
cd research-paper-summarization
pip install -r requirements.txt
```

---

## 📥 Usage

### 1. Add your paper
Put a `.pdf` file in the `data/sample_papers/` directory.

### 2. Run the pipeline
```bash
python main.py --pdf data/sample_papers/your_paper.pdf --tts False
```

You can enable TTS with `--tts True` if needed.

### 3. Output
- 📑 Summaries saved as text files (section-wise).
- 🔉 Audio file generated (if TTS enabled).
- 📊 Evaluation scores printed in terminal.

---

## 📚 Models Used

- **LexRank** (TextRank variant for extractive summarization)
- **T5-small / DistilBART** (via HuggingFace Transformers)
- **SciBERT / Sentence-BERT** for citation graph embeddings (optional)

---

## 🧪 Evaluation

Evaluated using:
- ROUGE-1 / ROUGE-L
- BLEU-4
- BERTScore

> All metrics are calculated against gold summaries (if available) or between LexRank and abstractive outputs for consistency analysis.

---

## 🧠 Citation Graph-Based Enhancement

Each section’s sentences are scored using a citation-aware graph:
- **Nodes**: Sentences
- **Edges**: Citation/contextual links
- **Scores**: TF-IDF + PageRank centrality

---

## 🔊 Optional TTS with LE2E

Enable end-to-end neural TTS to generate an audio version of the final summary.

```bash
python main.py --pdf your_paper.pdf --tts True
```

---

## 📌 Dependencies

- `PyPDF2`
- `transformers`
- `scikit-learn`
- `networkx`
- `numpy`
- `nltk`
- `rouge-score`
- `bert-score`
- `torch`
- `TTS` (LE2E)

Install via:
```bash
pip install -r requirements.txt
```

---

## 🤖 Author

**Rahul**  
Backend + AI/ML Developer  
> Specializing in NLP, summarization, deep learning, and real-time intelligent systems.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
