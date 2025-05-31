# ===================== SETUP & IMPORTS =====================
import os
import re
import json
import string
import numpy as np
import pandas as pd
from pathlib import Path
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import networkx as nx
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import torch
from gtts import gTTS
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ===================== PDF PARSING & SEGMENTATION =====================
os.environ['NLTK_DATA'] = './nltk_data'
nltk.data.path.append('./nltk_data')

pdf_path = "Literature Review/Citation-aware Graph Contrastive Learning.pdf"
print(f"Selected file: {pdf_path}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

SECTION_TITLES = [
    "abstract", "introduction", "background", "related work",
    "methodology", "methods", "approach", "experiments",
    "results", "discussion", "conclusion", "references"
]

SECTION_REGEX = re.compile(
    r"^\s*(" + "|".join([re.escape(s) for s in SECTION_TITLES]) + r")\s*$",
    re.IGNORECASE
)

def segment_sections(raw_text):
    sections = {}
    lines = raw_text.split('\n')
    current_section = "unknown"
    sections[current_section] = []

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        header_match = SECTION_REGEX.match(clean_line.lower())
        if header_match:
            current_section = header_match.group(0).lower()
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(clean_line)

    for key in sections:
        sections[key] = ' '.join(sections[key])

    return sections

raw_text = extract_text_from_pdf(pdf_path)
segmented_text = segment_sections(raw_text)

with open("segmented_output.json", "w") as f:
    json.dump(segmented_text, f, indent=2)

# ===================== PREPROCESSING FUNCTIONS =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\[\]\(\),.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    return sent_tokenize(text)

def remove_stopwords(sentence):
    tokens = sentence.split()
    return ' '.join([w for w in tokens if w not in stop_words])

def detect_citations(text):
    return re.findall(r'\[[0-9, ]+\]|\([A-Za-z., ]+\d{4}\)', text)

# ===================== EXTRACTIVE SUMMARIZATION =====================
from sklearn.metrics.pairwise import cosine_similarity

def extractive_summary_lexrank(sentences, top_n=5):
    if len(sentences) <= top_n:
        return sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [sent for _, sent in ranked_sentences[:top_n]]

extractive_summaries = {}
for section, content in segmented_text.items():
    cleaned = clean_text(content)
    sentences = tokenize_sentences(cleaned)
    extractive_summaries[section] = extractive_summary_lexrank(sentences, top_n=5)

with open("extractive_summaries.json", "w") as f:
    json.dump(extractive_summaries, f, indent=2)

# ===================== ABSTRACTIVE SUMMARIZATION (T5) =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def paraphrase_with_t5(text, max_len=100):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = t5_model.generate(inputs, max_length=max_len, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

final_abstractive_summary = {}
for section, sentences in extractive_summaries.items():
    combined_text = " ".join(sentences)
    try:
        summary = paraphrase_with_t5(combined_text)
    except Exception as e:
        summary = "Abstractive summarization failed."
    final_abstractive_summary[section] = summary

with open("final_abstractive_summary.json", "w") as f:
    json.dump(final_abstractive_summary, f, indent=2)

# ===================== CITATION-AWARE GRAPH SUMMARIZATION =====================
def extract_citations(text):
    bracketed = re.findall(r'\[(\d+)\]', text)
    named = re.findall(r'\(([^()]+?,\s*\d{4})\)', text)
    return bracketed + named

def build_citation_graph(sentences):
    graph = nx.Graph()
    for i, sent_i in enumerate(sentences):
        graph.add_node(i, text=sent_i)
        citations_i = set(extract_citations(sent_i))
        for j in range(i + 1, len(sentences)):
            sent_j = sentences[j]
            citations_j = set(extract_citations(sent_j))
            overlap = len(citations_i.intersection(citations_j))
            if overlap > 0:
                graph.add_edge(i, j, weight=overlap)
    return graph

def citation_aware_lexrank(sentences, top_n=5):
    if len(sentences) <= top_n:
        return sentences
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf_matrix)
    citation_graph = build_citation_graph(sentences)
    base_graph = nx.from_numpy_array(cosine_sim)
    for i, j, data in citation_graph.edges(data=True):
        if base_graph.has_edge(i, j):
            base_graph[i][j]['weight'] += data['weight']
        else:
            base_graph.add_edge(i, j, weight=data['weight'])
    scores = nx.pagerank(base_graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [s for _, s in ranked[:top_n]]

citation_aware_summaries = {}
for section, content in segmented_text.items():
    cleaned = clean_text(content)
    sents = tokenize_sentences(cleaned)
    citation_aware_summaries[section] = citation_aware_lexrank(sents, top_n=5)

# ===================== EVALUATION =====================
def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}

def evaluate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    return round(bleu, 4)

def evaluate_bert(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    return round(F1[0].item(), 4)

reference_summary = {}

results = {}
for section in final_abstractive_summary:
    ref = reference_summary.get(section, "")
    gen = final_abstractive_summary[section]
    if not ref.strip():
        continue
    rouge = evaluate_rouge(ref, gen)
    bleu = evaluate_bleu(ref, gen)
    bert = evaluate_bert(ref, gen)
    results[section] = {
        "ROUGE": rouge,
        "BLEU": bleu,
        "BERTScore": bert
    }

with open("summary_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ===================== TEXT-TO-SPEECH (TTS) =====================
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
os.makedirs("tts_outputs", exist_ok=True)
for section, summary in final_abstractive_summary.items():
    filename = f"tts_outputs/{section}.wav"
    print(f"🔉 Generating TTS for {section} → {filename}")
    tts.tts_to_file(text=summary, file_path=filename)
