# AI Research Assistant with RAG

An AI-powered research assistant that combines **Retrieval-Augmented Generation (RAG)**, semantic vector search, LLM-based document question answering, academic research retrieval, and web search.

Built using **Python, Streamlit, LangChain, Groq, Google Gemini, Pinecone, arXiv, Wikipedia, DuckDuckGo, BeautifulSoup, Selenium, and pdfplumber**.

---

## Overview

This project provides a unified interface for working with research documents and external information through three primary modes:

- **Document Q&A** — Upload PDF documents and ask questions grounded in their contents using a RAG pipeline.
- **Research Assistant** — Search academic sources such as arXiv and Wikipedia and use the retrieved information to answer research questions.
- **Web Search** — Determine when external web information is required, retrieve relevant search results, and summarize the retrieved content.

The project also includes a dedicated **RAG evaluation framework** with a 25-question benchmark covering factual questions, reasoning questions, unanswerable questions, and adversarial cases.

---

## Key Features

- Retrieval-Augmented Generation (RAG) for document question answering
- PDF text and table extraction using `pdfplumber`
- Recursive document chunking with overlap
- Semantic embeddings using Google's `gemini-embedding-001`
- 3072-dimensional document embeddings
- Vector storage and similarity search using Pinecone
- Top-k semantic retrieval
- LLM-based answer generation using Groq
- Academic research retrieval from arXiv
- Wikipedia-based research retrieval
- Web search using DuckDuckGo
- Static web scraping using Requests and BeautifulSoup
- Dynamic web scraping using Selenium
- LLM-based routing for determining whether web search is required
- Dedicated RAG evaluation pipeline
- LLM-as-a-judge answer evaluation
- Separate Pinecone index for evaluation experiments
- Streamlit-based interactive interface

---

# System Architecture

The application consists of three major information-retrieval workflows.

## 1. Document Q&A — RAG Pipeline

The document question-answering workflow follows a standard Retrieval-Augmented Generation architecture:

```text
                  PDF Upload
                      |
                      v
             PDF Text / Table
                 Extraction
                      |
                      v
              Document Chunks
                      |
                      v
          Gemini Embedding Model
          gemini-embedding-001
                      |
                      v
                 Pinecone
                Vector DB
                      |
                User Question
                      |
                      v
              Query Embedding
                      |
                      v
          Similarity Retrieval
                 Top-K = 4
                      |
                      v
            Retrieved Context
                      |
                      v
                  Groq LLM
                      |
                      v
                   Answer
```

### RAG Pipeline

The current document RAG implementation performs the following steps:

1. Upload a PDF document.
2. Extract text and tables from the PDF.
3. Split the document into overlapping chunks.
4. Generate embeddings for each chunk using `gemini-embedding-001`.
5. Store the embeddings in Pinecone.
6. Embed the user's question using the same embedding model.
7. Retrieve the top 4 semantically similar chunks using cosine similarity.
8. Pass the retrieved context and question to the Groq LLM.
9. Generate the final answer.

---

# 2. Research Assistant

The research assistant retrieves information from external research sources including:

- arXiv
- Wikipedia

For arXiv papers, the application can download the associated PDF, extract its text, generate a concise summary, and provide the resulting research context to the LLM.

```text
                  User Query
                      |
          +-----------+-----------+
          |                       |
          v                       v
      arXiv Search          Wikipedia Search
          |                       |
          v                       v
    Paper Metadata          Article Content
          |                       |
          v                       |
     PDF Download                 |
          |                       |
          v                       |
    PDF Extraction               |
          |                       |
          v                       |
     LLM Summary                 |
          |                       |
          +-----------+-----------+
                      |
                      v
                Context Assembly
                      |
                      v
                   Groq LLM
                      |
                      v
                    Answer
```

### Implementation Note

The current arXiv/Wikipedia workflow is separate from the Pinecone document RAG pipeline.

The retrieved research sources are currently fetched and summarized before being provided to the LLM. They are not automatically embedded into the application's Pinecone index.

A future extension could introduce a unified research RAG pipeline:

```text
Fetch
  |
Extract
  |
Chunk
  |
Embed
  |
Pinecone
  |
Retrieve
  |
Generate
```

---

# 3. Web Search

The application includes an LLM-based routing mechanism that determines whether a query requires external web information.

The router checks whether:

- external information is required,
- the available context is insufficient,
- relevant context is missing,
- or the query requires up-to-date information.

The current implementation uses a special `<SEARCH>` token to indicate that web search should be performed.

```text
                    User Query
                        |
                        v
               LLM Search Router
                        |
              +---------+---------+
              |                   |
              v                   v
       Search Required       No Search
              |                   |
              v                   v
       Search Keywords       Existing Context
              |                   |
              v                   |
       DuckDuckGo Search           |
              |                   |
              v                   |
        Web Content               |
              |                   |
              v                   |
         Summarization            |
              |                   |
              +---------+---------+
                        |
                        v
                      Result
```

The current routing mechanism can be improved using structured output or tool/function calling for more reliable decision-making.

---

# Document Processing

## PDF Extraction

PDF documents are processed using `pdfplumber`.

The application extracts:

- Page text
- Tables
- Basic image presence information

Each page is represented as a LangChain `Document` with page-level metadata.

### Image Handling

The current implementation detects and counts images in PDF pages but does not perform image understanding or OCR.

Therefore, information contained exclusively inside:

- charts
- diagrams
- scanned pages
- screenshots
- other visual elements

may not be available to the current text-based RAG pipeline.

Potential improvements include OCR and vision-language models.

---

# Chunking

Documents are split using LangChain's:

```text
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk size:    1000
Chunk overlap: 200
```

The overlap helps preserve context when relevant information crosses chunk boundaries.

---

# Embeddings

The document RAG pipeline uses Google's:

```text
gemini-embedding-001
```

The current Pinecone configuration uses:

```text
Dimension: 3072
Metric:    Cosine Similarity
```

The same embedding model is used for both:

- document embeddings
- query embeddings

This ensures that documents and queries exist in the same embedding space.

Changing the embedding model generally requires re-embedding the existing document corpus because embeddings generated by different models are not directly interchangeable.

---

# Vector Database

The project uses **Pinecone** for vector storage and semantic retrieval.

The application uses:

```text
research-rag-3072
```

for the document RAG workflow.

The evaluation workflow uses a separate index:

```text
research-rag-eval
```

Keeping evaluation data separate prevents evaluation experiments from contaminating the application's document index.

---

# Retrieval

The current document QA pipeline performs similarity-based retrieval from Pinecone.

The current configuration retrieves:

```text
Top K = 4
```

The retrieval process is:

```text
Question
    |
    v
Query Embedding
    |
    v
Pinecone
    |
    v
Cosine Similarity
    |
    v
Top 4 Relevant Chunks
    |
    v
Groq LLM
    |
    v
Answer
```

## Retrieval Improvements

The current implementation does not include a dedicated reranking stage.

A potential improved architecture would be:

```text
Query
  |
  v
Vector Search
Top 10-20 Candidates
  |
  v
Reranker
  |
  v
Top 3-5 Chunks
  |
  v
LLM
```

Vector retrieval can focus on recall while a reranking stage can improve the precision and ordering of the final context.

---

# LLM Generation

The document Q&A pipeline uses Groq-hosted language models.

The current evaluation configuration uses:

```text
openai/gpt-oss-120b
```

The LLM receives the user's question together with the retrieved document context and generates the final response.

---

# Evaluation

A dedicated evaluation framework is included under:

```text
evaluation/
```

The benchmark contains **25 questions** divided into four categories:

| Category | Questions |
|----------|-----------|
| Factual | 10 |
| Reasoning / Multi-hop | 7 |
| Unanswerable | 5 |
| Adversarial | 3 |
| **Total** | **25** |

The evaluation workflow is:

```text
Evaluation Questions
        |
        v
     RAG Pipeline
        |
        v
  Generated Answers
        |
        v
      LLM Judge
        |
        v
 Correct / Incorrect
        |
        v
 Evaluation Results
```

## Baseline Result

The current baseline achieved:

```text
Overall Answer Accuracy: 88%
Correct Answers:         22 / 25
```

Category-level results:

| Category | Accuracy |
|----------|----------|
| Factual | 90.0% |
| Reasoning | 85.7% |
| Unanswerable | 80.0% |
| Adversarial | 100% |
| **Overall** | **88.0%** |

The evaluation uses an LLM judge to compare generated answers against expected answers.

### Important Interpretation

The 88% result represents **answer accuracy on this 25-question benchmark**.

It should not be interpreted as a complete measure of RAG quality.

The current evaluation does not independently measure:

- Retrieval Recall@K
- Retrieval Precision@K
- MRR
- Context faithfulness
- Hallucination rate
- End-to-end latency
- Token usage
- Cost

These are potential extensions of the evaluation framework.

---

# Evaluation Files

```text
evaluation/
├── __init__.py
├── questions.json
├── questions.py
├── run_evaluation.py
├── evaluate_answers.py
├── evaluate_results.py
├── baseline_results.json
└── answer_accuracy_results.json
```

### `questions.json`

Contains the evaluation questions and expected answers/evidence.

### `run_evaluation.py`

Runs the RAG pipeline against the evaluation questions and stores:

- questions
- expected answers
- generated answers
- retrieved documents
- document metadata

### `evaluate_answers.py`

Uses an LLM judge to classify generated answers as:

```text
CORRECT
INCORRECT
```

and calculates overall and category-level answer accuracy.

### `baseline_results.json`

Stores baseline RAG outputs and retrieved context.

### `answer_accuracy_results.json`

Stores the answer evaluation results.

---

# RAG Evaluation Metrics

A more comprehensive RAG evaluation setup can include three layers.

## Retrieval Metrics

```text
Recall@K
Precision@K
Hit Rate@K
MRR
nDCG
```

These evaluate whether the retrieval system finds the relevant information.

## Generation Metrics

```text
Exact Match
Token F1
ROUGE
Semantic Similarity
```

These evaluate the generated answer against reference answers.

## Grounding Metrics

```text
Faithfulness / Groundedness
Context Precision
Context Recall
Citation Quality
```

These evaluate whether the generated answer is supported by the retrieved context.

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| UI | Streamlit |
| LLM | Groq |
| Embeddings | Google Gemini |
| Vector Database | Pinecone |
| RAG Framework | LangChain |
| PDF Processing | pdfplumber |
| Academic Search | arXiv |
| Knowledge Retrieval | Wikipedia |
| Web Search | DuckDuckGo |
| Static Scraping | Requests + BeautifulSoup |
| Dynamic Scraping | Selenium |
| Evaluation | Custom Evaluation Framework + LLM Judge |

---

# Project Structure

```text
.
├── app.py
├── groq_chain.py
├── research_chain.py
├── web_scrape_chain.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── evaluation/
│   ├── __init__.py
│   ├── questions.json
│   ├── questions.py
│   ├── run_evaluation.py
│   ├── evaluate_answers.py
│   ├── evaluate_results.py
│   ├── baseline_results.json
│   └── answer_accuracy_results.json
│
├── Notebooks/
│   ├── Gemini_scrape_chain.ipynb
│   ├── Groq Model RAG Chains.ipynb
│   ├── OpenAI RAG chain.ipynb
│   └── Research_Retreival_rag_chain.ipynb
│
├── Python Scripts/
│   ├── gemini_scraper_chain.py
│   ├── groq_model_chains.py
│   ├── openAI_scrape_chain.py
│   ├── openai_chain.py
│   └── reserach_scraper_chains.py
│
└── Text Corpus examples/
    └── Corpus.pdf
```

---

# Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Ayush-Sri589/RAG-AI-Research-Assistant.git
cd RAG-AI-Research-Assistant
```

## 2. Create a Python Environment

Using Conda:

```bash
conda create -n rag-project python=3.10
conda activate rag-project
```

Or use another Python virtual environment.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure API Keys

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

If using components that require OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key
```

**Never commit API keys or other secrets to Git.**

The repository's `.gitignore` is configured to exclude local environment files and temporary files.

---

# Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the Streamlit URL displayed in the terminal.

---

# Usage

## Document Q&A

1. Upload a PDF.
2. The document is extracted and divided into chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in Pinecone.
5. Ask a question about the document.
6. Relevant chunks are retrieved.
7. The LLM generates an answer using the retrieved context.

## Research Assistant

Ask a research-oriented question.

The application can retrieve relevant information from:

- arXiv
- Wikipedia

The retrieved information is processed and passed to the LLM for answer generation.

## Web Search

Ask a question that requires current or external information.

The system determines whether web search is required and, when necessary:

1. Generates search terms.
2. Searches using DuckDuckGo.
3. Retrieves web content.
4. Summarizes the retrieved information.
5. Uses the resulting context to generate a response.

---

# Limitations

The current implementation has several areas that can be improved.

### Retrieval

The document pipeline currently relies on top-k vector similarity search without reranking.

### PDF Images

Images are detected but their contents are not currently interpreted.

OCR or vision-language models could improve support for scanned documents, charts, and diagrams.

### Retrieval Evaluation

The current benchmark primarily measures final answer accuracy. Dedicated retrieval metrics such as Recall@K, Hit Rate@K, and MRR can be added.

### LLM-as-a-Judge

LLM-based evaluation can introduce evaluator bias or inconsistency.

Using an independent judge model, structured outputs, multiple evaluators, and human spot checks could improve evaluation reliability.

### Evaluation Dataset

The benchmark currently contains 25 questions. A larger and more diverse evaluation dataset would provide stronger statistical confidence.

### Search Routing

The current web-search router relies on an LLM-generated `<SEARCH>` token. Structured output or tool/function calling would provide a more robust routing mechanism.

### Research Source RAG

arXiv and Wikipedia retrieval currently operate separately from the Pinecone document RAG pipeline. A unified indexing and retrieval architecture could provide more consistent retrieval and source attribution.

---

# Future Improvements

Potential improvements include:

- Cross-encoder reranking
- Hybrid keyword + vector retrieval
- Query rewriting
- Multi-query retrieval
- Metadata-aware filtering
- OCR for scanned PDFs
- Vision-language processing for charts and diagrams
- Unified research-source indexing
- Citation generation and source attribution
- Retrieval Recall@K / MRR evaluation
- Faithfulness and groundedness evaluation
- Larger evaluation datasets
- Independent or multi-model LLM judging
- Latency and token-cost tracking
- Persistent document/user namespaces in Pinecone
- Duplicate document detection using document IDs or file hashes
- Structured tool routing
- Streaming LLM responses

---

