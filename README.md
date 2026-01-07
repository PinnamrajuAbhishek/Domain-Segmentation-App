

 Domain Segmentation App

Hybrid SBERT–LLM Domain Classification System

This project implements a hybrid semantic domain segmentation system that classifies a user query into the top 3 most relevant business domains using a combination of:

SBERT (Sentence Transformers) for fast semantic similarity search
Local LLM (Qwen2.5-0.5B-Instruct) for contextual reranking
Flask for a lightweight backend API
CSS for a clean, responsive web UI

The system is designed for enterprise and procurement-style query classification, where fine-grained domain understanding is critical.

---

 Key Features:

Semantic Retrieval with SBERT:
  Uses cosine similarity between query embeddings and domain descriptions.

Clean vs Excluded Text Scoring
  Final SBERT score =
  similarity(clean_text) − similarity(excluded_text)
  → reduces false positives across overlapping domains.

LLM-based Reranking (Local)
  A small local LLM (Qwen2.5-0.5B-Instruct) selects the best 3 domain labels from SBERT candidates.

Fail-Safe Design
  Automatically falls back to SBERT Top-3 if the LLM output is invalid.

Web UI + REST API
  Clean Tailwind-based UI and JSON API endpoint.

Fully Local Execution
  No external APIs or cloud dependencies.



System Architecture


User Query
   │
   ▼
SBERT Embedding
   │
   ▼
Cosine Similarity (Clean − Excluded)
   │
   ▼
Top-K Domain Candidates (SBERT)
   │
   ▼
LLM Reranking (Qwen 2.5)
   │
   ▼
Top-3 Domain Labels


📁 Project Structure

```
Domain-Segmentation-App/
├── app.py                     # Flask backend (SBERT + LLM logic)
├── templates/
│   └── index.html              # Tailwind CSS UI
├── static/                     # (optional) static assets
├── Presentation How to Buy.xlsx # Domain dataset (Excel)
└── README.md
```

---

Models Used

| Component           | Model                        |
| ------------------- | ---------------------------- |
| Sentence Embeddings | `all-MiniLM-L6-v2`           |
| LLM Reranking       | `Qwen/Qwen2.5-0.5B-Instruct` |

---

Dataset Format (Excel)

The Excel sheet must contain the following columns:

| Column Name     | Description                                     |
| --------------- | ----------------------------------------------- |
| `Domain`        | Domain label                                    |
| `clean_text`    | Positive description of the domain              |
| `excluded_text` | Concepts explicitly NOT belonging to the domain |

---



 Web Interface

* Enter a natural language query
* Click Segment Domain
* View the Top-3 predicted domains
* Includes loading indicators and error handling

---
 Why SBERT + LLM?

* SBERT → fast, scalable semantic retrieval
* LLM → deeper reasoning between closely related domains
* Hybrid approach balances performance + accuracy, making it suitable for enterprise-scale systems.



Future Enhancements

* Confidence scores per domain
* Multi-language query support
* Domain explanations per prediction
* Docker deployment




