# RAG-Eval Observability & Evaluation Dashboard

A production-ready AI observability and evaluation platform for RAG (Retrieval-Augmented Generation) systems. This application allows you to ingest documents, chat with an interactive RAG chatbot, and evaluate generation and retrieval quality in real-time or side-by-side using **RAGAS** metrics. Performance stats (latency, token count, cost) are automatically logged to a local SQLite database and visualized with **Plotly**.

---

## 🚀 Key Features

1. **💬 Interactive Chat & Real-Time Eval**: 
   - Upload PDF, TXT, or MD documents.
   - Query the model and retrieve source chunks.
   - Run Ragas evaluation metrics on-demand (Faithfulness, Answer Relevancy, Context Precision, and Context Recall when a ground truth is supplied).
   - Track total latency, generation token counts, and input/output API cost in real-time.
2. **📊 Side-by-Side Model Comparison**:
   - Run the same query simultaneously against `mistral-large-latest`, `mistral-medium-latest`, and `mistral-small-latest`.
   - Side-by-side response rendering.
   - Plotly bar charts comparing latency, token pricing, and quality scores.
3. **🔍 Embedding Engine Benchmarking**:
   - Compare `mistral-embed`, `bge-small-en-v1.5`, and local `sentence-transformers`.
   - Side-by-side inspection of top retrieved text passages.
   - Retrieval latency and Context Precision/Recall analysis.
4. **📝 Query Observability Ledger**:
   - Full history of all queries, responses, and evaluation runs stored in a structured SQLite database.
   - Interactive search, filtering by model, and deep details inspection.
5. **📈 System Analytics**:
   - Financial and performance overview.
   - Charts showing total cost trends, query distribution, and average metric performance.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.11+, LangChain, langchain-mistralai
- **Vector DB**: ChromaDB
- **Database**: SQLite (with clean base abstraction for PostgreSQL porting)
- **Evaluation**: RAGAS (LLM-as-a-judge run through Mistral Large)
- **Visualization**: Plotly

---

## 📁 Project Directory Structure

```
├── app.py                  # Main Streamlit Dashboard Application
├── requirements.txt        # Project Dependencies
├── test_backend.py         # Local backend integration test suite
├── run_benchmark.py        # Benchmarking script and metrics compiler
├── .env                    # System Environment Variables (API Keys)
├── src/
│   ├── __init__.py
│   ├── config.py           # Configs, pricing profiles, and directory setup
│   ├── db/
│   │   ├── __init__.py
│   │   ├── interface.py    # Database abstract interface (BaseDatabase)
│   │   ├── models.py       # Data Transfer Objects (DTOs)
│   │   └── sqlite.py       # SQLite concrete implementation of BaseDatabase
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py   # Swappable embedding models factory
│   │   ├── llm.py          # Configurable ChatMistralAI LLM factory
│   │   ├── vectorstore.py  # ChromaDB indexing and retrieval manager
│   │   └── pipeline.py     # End-to-end RAG orchestrator (with cost/latency logs)
│   └── eval/
│       ├── __init__.py
│       └── evaluator.py    # Ragas evaluation logic using Mistral judges
└── data/                   # Dynamic directory created at runtime
    ├── chroma_db/          # Persistent ChromaDB collections
    ├── uploads/            # Ingested PDF/TXT source files
    └── rag_eval.db         # SQLite observability database
```

---

## ⚙️ Setup and Installation

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Ensure a `.env` file exists in the root directory:
```env
MISTRAL_API_KEY="your-mistral-api-key"
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 🧪 Running the Backend Test Suite
You can execute `test_backend.py` to verify the SQLite database connection, DTO structures, and imports correctness:
```bash
python test_backend.py
```
