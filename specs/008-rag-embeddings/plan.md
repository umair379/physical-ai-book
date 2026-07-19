# Implementation Plan: RAG Embeddings Ingestion Pipeline

**Branch**: `008-rag-embeddings` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-rag-embeddings/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a Python-based ingestion pipeline to crawl Docusaurus documentation from Vercel deployment, extract and chunk text content, generate vector embeddings using Cohere API, and store them in Qdrant Cloud for semantic search. Implementation uses a single-file approach (`backend/main.py`) with modular functions for crawling, chunking, embedding, and storage, orchestrated by a main() entry point.

## Technical Context

**Language/Version**: Python 3.9+ (Python 3.11 recommended for async improvements)
**Primary Dependencies**:
- `requests` or `httpx` - HTTP client for URL fetching
- `beautifulsoup4` - HTML parsing and content extraction
- `lxml` - Fast XML/HTML parser for BeautifulSoup
- `tiktoken` - OpenAI's tokenizer for accurate token counting
- `cohere` - Cohere Python SDK for embeddings API
- `qdrant-client` - Qdrant Python client for vector storage
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation and settings management

**Package Manager**: `uv` (modern Python package manager, faster than pip)
**Storage**: Qdrant Cloud Free Tier (vector database, 1M vectors max, cosine similarity)
**Testing**: Manual validation with test queries (automated testing in future iterations)
**Target Platform**: Local development environment or cloud execution (Vercel, Railway, Docker)
**Project Type**: Single-file CLI script (`backend/main.py`)
**Performance Goals**:
- Full ingestion completes in <30 minutes for 100-page documentation site
- Crawl rate: 2-5 pages/second (respecting rate limits)
- Embedding batch size: 100 texts per API request
- >99% embedding generation success rate (with retries)

**Constraints**:
- Cohere API rate limits (free/trial tier: varies by account, handle with exponential backoff)
- Qdrant Cloud Free Tier: 1M vectors max, 1 cluster
- Memory: Process chunks in batches to avoid OOM on large sites
- Network: Requires stable internet connection for API calls

**Scale/Scope**:
- Target corpus: 100-200 documentation pages (~500KB-2MB total text)
- Expected chunks: 1,000-5,000 chunks (512 tokens each)
- Expected vectors: 1,000-5,000 embeddings (1024 dimensions)
- Expected ingestion time: 10-30 minutes (depends on site size and API rate limits)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Compliance | Evidence |
|-----------|------------|----------|
| **I. Specification-First Development** | ✅ PASS | Formal spec created via `/sp.specify` (specs/008-rag-embeddings/spec.md). All requirements traced to user stories. Implementation follows spec workflow. |
| **II. Accuracy and Non-Hallucination** | ✅ PASS | Pipeline ingests ONLY from specified Vercel URL. No external data sources. Crawling limited to discovered sitemap/links. No invented content. |
| **III. Reproducibility and Developer Clarity** | ✅ PASS | Single-file design (`backend/main.py`) simplifies setup. Environment variables documented in `.env.example`. Step-by-step quickstart guide planned (Phase 1). |
| **IV. AI-Native Authoring** | ✅ PASS | Spec created via `/sp.specify`, plan via `/sp.plan`, tasks via `/sp.tasks`. PHR created for specification work (001-create-rag-embeddings-spec.spec.prompt.md). |
| **V. Modular and Clean Architecture** | ✅ PASS | Single-file design with modular functions (crawl, chunk, embed, store). Each function independently testable. No cross-component dependencies. |
| **VI. Security and Secrets Management** | ✅ PASS | Cohere API key and Qdrant credentials loaded from environment variables (`.env`). `.env.example` documents required vars. No hardcoded secrets in code. |
| **VII. Testability and Verification** | ✅ PASS | Spec includes 12 Given/When/Then acceptance scenarios. Success criteria measurable (crawl coverage, chunk accuracy, search quality). Test script planned for semantic search validation. |

**Overall Status**: ✅ **PASS** - All constitutional principles satisfied.

**Notes**:
- Single-file design (`backend/main.py`) aligns with user's request for simplicity and "complete within 3-5 tasks" timeline
- Modular functions within single file balance simplicity with testability
- No complexity violations requiring justification (Complexity Tracking table remains empty)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Single-file ingestion pipeline with modular functions
├── .env.example         # Template for required environment variables
├── pyproject.toml       # uv project configuration and dependencies
└── README.md            # Setup and usage instructions

# Generated at runtime (gitignored)
backend/.env             # Actual secrets (COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY)
backend/.venv/           # Virtual environment created by uv
backend/output/          # Optional: intermediate outputs (crawled pages, chunks JSON)
```

**Structure Decision**: Single-file approach (`backend/main.py`) chosen for simplicity and rapid development. All functionality (crawling, chunking, embedding, storage) implemented as modular functions within one file:

- `fetch_sitemap(base_url)` → List[str]: Discover URLs from sitemap.xml
- `crawl_pages(urls)` → List[DocumentPage]: Fetch and parse HTML pages
- `extract_content(html)` → str: Extract main content from HTML
- `chunk_text(text, metadata)` → List[TextChunk]: Split text into semantic chunks
- `generate_embeddings(chunks)` → List[VectorEmbedding]: Batch embed via Cohere API
- `store_in_qdrant(embeddings)` → None: Upsert vectors to Qdrant collection
- `validate_search(queries)` → Dict: Test semantic search quality
- `main()` → None: Orchestrate full pipeline end-to-end

This structure delivers on user's constraint "complete within 3-5 tasks" by minimizing boilerplate and enabling focused implementation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
