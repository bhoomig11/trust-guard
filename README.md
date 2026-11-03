# TrustGuard AI

TrustGuard AI is a secure retrieval-augmented generation (RAG) service for financial compliance teams. It ingests policy documents (PDF and TXT), indexes them in a local vector store, and answers questions through a Claude agent that chooses between two tools: a policy retriever and a calculator for fees and limits. Every answer is grounded in the indexed documents and returned with citations to the source document and page.

Privacy and auditability are built into the request path. Each query passes through Microsoft Presidio, which replaces names, SSNs, and account and card numbers with placeholders before anything reaches the vector store or the model. Every transaction is written to a structured JSON audit log with the original and masked query, the retrieved chunks and similarity scores, the model prompt and response, and latency.

## Tech stack

- **API:** Python, FastAPI, Pydantic
- **LLM:** Anthropic Claude (Haiku 5.5 by default) with tool use
- **Retrieval:** SentenceTransformers (local embeddings), ChromaDB, LangChain text splitters
- **Privacy:** Microsoft Presidio, spaCy
- **Data:** pypdf, Pandas
- **Quality:** Pytest, Ruff, mypy
- **DevOps:** Docker, Docker Compose, GitHub Actions, GitHub Container Registry

## Run it

```bash
cp .env.example .env          # add your ANTHROPIC_API_KEY
docker compose up --build
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What is the fee on a $2,000 foreign card purchase?"}'
```

API docs are served at `http://localhost:8000/docs`. The sample corpus describes a fictional bank, Cobalt Ridge Financial, alongside public FinCEN guidance.
