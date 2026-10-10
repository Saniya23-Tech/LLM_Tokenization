
# Byte-Level BPE Tokenizer
### From Raw Text to Tokens — Built from Scratch.

An end-to-end implementation of a Byte-Level Byte Pair Encoding (BPE) tokenizer, exploring how text is transformed into tokens and token IDs for Large Language Models (LLMs).

<p align="center">
  <a href="https://llmtokenization.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_App-FF4B4B?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/NLP-Tokenization-8A2BE2?style=flat-square" alt="NLP"/>
  <img src="https://img.shields.io/badge/Algorithm-Byte--Level_BPE-0A7B83?style=flat-square" alt="BPE"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

## Overview

Tokenization is a fundamental stage in the LLM pipeline. Before a language model can process text, it must be represented as tokens and mapped to numerical IDs.

This project implements Byte-Level BPE from scratch, covering the journey from raw text preprocessing and vocabulary learning to encoding, decoding, performance benchmarking, and interactive exploration.

**The engineering focus:** optimizing BPE training by replacing repeated full-dataset scans with indexed lookups and incremental pair-frequency updates.

## Key Highlights

- **Custom BPE implementation** — Learn subword merges from training data instead of relying entirely on a pretrained tokenizer.
- **Byte-level text representation** — Represent all 256 possible byte values through a reversible byte-to-Unicode mapping.
- **Optimized training algorithm** — Maintain pair-frequency counts and a pair-to-words index to update only affected word sequences.
- **End-to-end tokenization** — Convert text into tokens, map tokens to IDs, and reconstruct the original text.
- **Correctness checks** — Validate encoding-decoding round trips.
- **Interactive deployment** — Experiment with your own input through a live Streamlit application.

## Performance Optimization

### From repeated scans to indexed updates

The original implementation repeatedly scanned word sequences to recalculate adjacent-pair frequencies after merges.

The optimized implementation maintains two structures:

| Data structure | Purpose |
|---|---|
| `pair_counts` | Tracks the frequency of each adjacent token pair |
| `pair_to_words` | Identifies word sequences containing each pair |

After a merge, the implementation retrieves affected words through the index, removes their previous pair contributions, updates their sequences, and adds their new pair contributions.

This avoids rebuilding pair counts from scratch after every merge.

### Benchmark Results

| Implementation | Training time |
|---|---:|
| Original BPE | 67.461 seconds |
| Optimized BPE | 2.673 seconds |

**~25.2× observed speedup** in the recorded experiment.

*Benchmark timings are specific to the tested workload and environment; results may vary across datasets and hardware.*

## Pipeline

```text
Raw Text / Dataset
        ↓
Data Preprocessing
        ↓
Byte-Level Representation
        ↓
Regex Pre-tokenization
        ↓
Word & Pair Frequency Counting
        ↓
Optimized BPE Training
        ↓
Learned Merges + Vocabulary
        ↓
Tokenization → Token IDs
        ↓
Decoding & Round-Trip Checks
        ↓
Interactive Streamlit Demo
```

## Tech Stack

- **Python** — Core implementation
- **Regex** — Text pre-tokenization
- **Collections** — Frequency counting and indexed updates
- **Hugging Face Datasets** — Training data loading
- **Pickle** — Saving and loading tokenizer artifacts
- **Jupyter Notebook** — Experiments and benchmarking
- **Streamlit** — Interactive web application

## Project Structure

```text
LLM_Tokenization/
├── Tokenization.ipynb       # Training and experiments
├── tokenization.py          # Tokenization, encoding and decoding
├── app.py                   # Streamlit application
├── requirements.txt         # Dependencies
└── artifacts/
    └── bpe_tokenizer.pkl    # Saved tokenizer
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Saniya23-Tech/LLM_Tokenization.git
cd LLM_Tokenization
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, typically `http://localhost:8501`.

Ensure that `artifacts/bpe_tokenizer.pkl` is available before launching the app.

## What I Learned

- How byte-level tokenization and BPE merge rules work.
- How vocabulary construction and token-ID mapping support language-model inputs.
- How data structures can reduce redundant computation in iterative algorithms.
- How to benchmark algorithmic improvements using execution time.
- How to separate training code from inference and deployment.
- How to package an NLP implementation into an interactive web application.

## Scope

This project focuses on the **tokenizer**, not on training or deploying a complete LLM. Text generation requires a language model in addition to tokenization.

## Live Application

**Explore the tokenizer:**  
https://llmtokenization.streamlit.app/

Enter text, inspect its tokens and token IDs, and check whether decoding reconstructs the original input.

---

<p align="center">
  <b>Built to understand the internals of LLM tokenization — one merge at a time.</b>
</p>
