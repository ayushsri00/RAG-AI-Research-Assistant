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