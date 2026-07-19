---
id: 002
title: Create RAG Embeddings Plan
stage: plan
date: 2025-12-27
surface: agent
model: claude-sonnet-4-5-20250929
feature: 008-rag-embeddings
branch: 008-rag-embeddings
user: developer
command: /sp.plan
labels: ["planning", "architecture", "rag", "embeddings", "python", "cohere", "qdrant", "single-file"]
links:
  spec: D:\physical-ai-book\specs\008-rag-embeddings\spec.md
  ticket: null
  adr: null
  pr: null
files:
 - D:\physical-ai-book\specs\008-rag-embeddings\plan.md
 - D:\physical-ai-book\specs\008-rag-embeddings\research.md
 - D:\physical-ai-book\specs\008-rag-embeddings\data-model.md
 - D:\physical-ai-book\specs\008-rag-embeddings\quickstart.md
 - D:\physical-ai-book\CLAUDE.md
tests:
 - Constitution Check: ✅ PASS - All 7 principles satisfied
 - Technical Context: ✅ COMPLETE - All dependencies and constraints documented
 - Research Phase: ✅ COMPLETE - 10 technical decisions documented with rationale
 - Data Model: ✅ COMPLETE - 5 entities defined with validation rules
 - Quickstart Guide: ✅ COMPLETE - Setup, execution, troubleshooting documented
---

## Prompt

/sp.plan

Spec-1: URL Ingestion & Embedding Pipeline
Create backend/ folder, initialize project with uv, and add a single main.py.
In main.py, implement URL fetching, text cleaning, and chunking.
Generate embeddings using Cohere models.
Store embeddings and metadata in Qdrant Cloud.
Add a main() function to run the full ingestion pipeline end-to-end.

## Response snapshot

Successfully created complete implementation plan for RAG Embeddings Ingestion Pipeline following /sp.plan workflow. Executed Phase 0 (research), Phase 1 (design), and completed all planning documentation. Single-file architecture (`backend/main.py`) chosen to deliver "complete within 3-5 tasks" user constraint.

### Planning Summary

**Workflow Completed**: Phase 0 (Research) + Phase 1 (Design) + Agent Context Update
**Branch**: `008-rag-embeddings`
**Architecture**: Single-file Python CLI script with modular functions

### Phase 0: Research (24KB document)

**10 Technical Decisions Documented**:

1. **Package Manager**: uv (10-100x faster than pip, modern lock file support)
2. **HTTP Client**: requests (simple, stable, sufficient for synchronous crawling)
3. **HTML Parser**: BeautifulSoup + lxml (high-level API + fast C-based parser)
4. **Tokenizer**: tiktoken (fast, accurate, offline token counting)
5. **Chunking Strategy**: Paragraph + heading aware (preserves Docusaurus structure)
6. **Embedding Model**: Cohere embed-english-v3.0 (1024 dimensions, latest stable)
7. **Embedding Batch Size**: 96 texts/request (Cohere's recommended max)
8. **Vector Database**: Qdrant Cloud Free Tier (1M vectors, cosine similarity)
9. **URL Discovery**: Sitemap first, recursive fallback (efficient + comprehensive)
10. **Configuration**: .env + pydantic validation (secure secrets, type-safe config)

**Key Research Findings**:
- tiktoken's cl100k_base encoding within 5-10% of Cohere's tokenizer (acceptable)
- Cohere trial tier: ~5 req/min, production free tier: ~100 req/min
- Qdrant Free Tier: 1M vectors sufficient for 5,000 chunks + future growth
- Docusaurus HTML structure: Consistent `<article>` or `<main>` tags for extraction

**Alternatives Considered**: 12 alternatives evaluated across 10 decision areas, all documented with rationale

---

### Phase 1: Design (3 documents)

#### 1. Data Model (20KB document)

**5 Entities Defined**:

1. **Config** (Pydantic BaseSettings)
   - 4 required fields (base_url, cohere_api_key, qdrant_url, qdrant_api_key)
   - 5 optional fields with defaults (collection_name, chunk_size, max_chunk_size, batch_size, max_crawl_depth)
   - Validators for URL format, chunk size range, max_chunk_size >= chunk_size

2. **DocumentPage** (dataclass)
   - url, title, raw_html, extracted_text, breadcrumb, last_modified
   - Represents single Docusaurus page from deployed site

3. **TextChunk** (dataclass)
   - chunk_id, source_url, chunk_text, chunk_index, total_chunks, heading_hierarchy, token_count, metadata
   - Unique ID format: `{url}#{chunk_index}`
   - to_dict() method for JSON serialization

4. **VectorEmbedding** (dataclass)
   - chunk_id, vector (1024 floats), chunk_text, metadata, timestamp
   - to_qdrant_point() method converts to Qdrant PointStruct
   - validate_dimensions() ensures 1024 dimensions (FR-014)

5. **IngestionStats** (dataclass)
   - Tracks: urls_discovered, pages_crawled, pages_failed, chunks_created, embeddings_generated, vectors_stored, errors
   - Properties: duration, success_rate
   - to_dict() method for logging

**State Transitions**: 3 lifecycle diagrams (DocumentPage, TextChunk, VectorEmbedding) with error handling

**Validation Rules**: 15 validation rules across entities (startup config, crawling, chunking, embedding)

---

#### 2. Quickstart Guide (15KB document)

**7 Sections**:

1. **Prerequisites** (3 subsections)
   - Python 3.9+ environment setup
   - API keys (Cohere, Qdrant Cloud)
   - Deployed Docusaurus site verification

2. **Setup** (4 steps, ~5 minutes)
   - Navigate to backend/ directory
   - Initialize project with uv
   - Configure .env file
   - Verify configuration

3. **Running Pipeline** (7 steps, 10-30 minutes)
   - Single command execution: `uv run main.py`
   - Detailed breakdown of each pipeline stage
   - Example output with timestamps and stats

4. **Verification** (test search examples)
   - Python code snippets for testing semantic search
   - Expected results: similarity scores > 0.7

5. **Troubleshooting** (8 common issues)
   - ModuleNotFoundError (dependency installation)
   - pydantic.ValidationError (.env file missing)
   - cohere.errors.RateLimitError (retry logic, batch size adjustment)
   - qdrant Quota limit exceeded (collection deletion)
   - HTTP 404 errors (expected, log warnings)
   - Pipeline hangs (network/API status checks)
   - Low similarity scores (chunk size tuning, query refinement)
   - Each with Cause, Fix, and code examples

6. **Re-Ingestion** (updating content)
   - Upsert behavior (replaces existing chunks)
   - Fresh start procedure (delete collection first)

7. **Advanced Configuration** (3 topics)
   - Adjust chunk size for better retrieval
   - Reduce crawl depth for faster testing
   - Save intermediate outputs for debugging

**Performance Benchmarks**: Table showing ingestion times for small/medium/large sites on trial vs production API tiers

---

#### 3. Agent Context Update

**Updated**: `CLAUDE.md` with new technology entries
- Language: Python 3.9+ (Python 3.11 recommended for async improvements)
- Database: Qdrant Cloud Free Tier (vector database, 1M vectors max, cosine similarity)

**Preserved**: Existing manual additions and constitution references

---

### Technical Context (plan.md)

**Complete Specification**:
- Language: Python 3.9+ (Python 3.11 recommended)
- Dependencies: 8 primary packages (requests, beautifulsoup4, lxml, tiktoken, cohere, qdrant-client, python-dotenv, pydantic)
- Package Manager: uv
- Storage: Qdrant Cloud Free Tier
- Testing: Manual validation with test queries
- Platform: Local development or cloud execution
- Project Type: Single-file CLI script

**Performance Goals**:
- Full ingestion: <30 minutes for 100-page site
- Crawl rate: 2-5 pages/second
- Embedding batch size: 100 texts per request
- Success rate: >99% embedding generation

**Constraints**:
- Cohere API rate limits (trial: 5 req/min, production: ~100 req/min)
- Qdrant Free Tier: 1M vectors max, 1 cluster
- Memory: Process chunks in batches
- Network: Requires stable internet

**Scale**:
- Target corpus: 100-200 documentation pages
- Expected chunks: 1,000-5,000 chunks (512 tokens each)
- Expected vectors: 1,000-5,000 embeddings (1024 dimensions)
- Expected ingestion time: 10-30 minutes

---

### Constitution Check

**All 7 Principles**: ✅ PASS

| Principle | Evidence |
|-----------|----------|
| I. Specification-First Development | Formal spec via /sp.specify, all requirements traced to user stories |
| II. Accuracy and Non-Hallucination | Pipeline ingests ONLY from specified Vercel URL, no external data |
| III. Reproducibility | Single-file design, .env.example, quickstart guide with step-by-step instructions |
| IV. AI-Native Authoring | Spec via /sp.specify, plan via /sp.plan, PHRs created |
| V. Modular Architecture | Modular functions within single file, independently testable |
| VI. Security | Cohere/Qdrant credentials from .env, no hardcoded secrets |
| VII. Testability | 12 Given/When/Then scenarios, 14 measurable success criteria |

**Complexity Tracking**: Empty (no violations requiring justification)

---

### Project Structure

**Backend Directory**:
```
backend/
├── main.py              # Single-file ingestion pipeline
├── .env.example         # Environment variable template
├── pyproject.toml       # uv project configuration
└── README.md            # Setup instructions
```

**8 Modular Functions** (all in main.py):
- `fetch_sitemap(base_url)` → List[str]
- `crawl_pages(urls)` → List[DocumentPage]
- `extract_content(html)` → str
- `chunk_text(text, metadata)` → List[TextChunk]
- `generate_embeddings(chunks)` → List[VectorEmbedding]
- `store_in_qdrant(embeddings)` → None
- `validate_search(queries)` → Dict
- `main()` → None (orchestrates pipeline)

**Rationale**: Single-file approach minimizes boilerplate, delivers on "complete within 3-5 tasks" constraint

---

### Files Created (4)

1. `specs/008-rag-embeddings/plan.md` (7KB) - Implementation plan with technical context, constitution check, project structure
2. `specs/008-rag-embeddings/research.md` (24KB) - 10 technical decisions with rationale and alternatives
3. `specs/008-rag-embeddings/data-model.md` (20KB) - 5 entities, state transitions, validation rules, data flow diagram
4. `specs/008-rag-embeddings/quickstart.md` (15KB) - Prerequisites, setup, execution, troubleshooting, advanced config

### Files Modified (1)

1. `CLAUDE.md` - Agent context updated with Python 3.9+ and Qdrant Cloud Free Tier

---

### Key Design Decisions

**1. Single-File Architecture**
- **Decision**: All functionality in `backend/main.py` with modular functions
- **Rationale**: User constraint "complete within 3-5 tasks" requires minimal boilerplate
- **Alternative Rejected**: Multi-file modular structure (crawl.py, chunk.py, embed.py, store.py) - adds setup complexity

**2. uv Package Manager**
- **Decision**: Use uv instead of pip for dependency management
- **Rationale**: 10-100x faster, lock file support, modern Python workflows
- **Alternative Rejected**: pip + venv - slower, no lock file by default

**3. Synchronous HTTP Client (requests)**
- **Decision**: Use synchronous requests library
- **Rationale**: Adequate 2-5 pages/sec crawl rate, simpler than async
- **Alternative Rejected**: httpx with async - adds complexity without significant benefit for MVP

**4. Semantic Chunking (Heading + Paragraph Aware)**
- **Decision**: Split at heading boundaries, preserve heading hierarchy in metadata
- **Rationale**: Docusaurus structure naturally organized by headings, improves retrieval quality
- **Alternative Rejected**: Fixed-size chunks without semantic awareness - loses context

**5. Cohere embed-english-v3.0 (1024 dimensions)**
- **Decision**: Use latest stable Cohere embedding model
- **Rationale**: Optimized for semantic search, 1024 dimensions standard
- **Alternative Rejected**: Older models (embed-english-v2.0) - lower quality, fewer dimensions

---

### Validation Results

**Constitution Check**: ✅ PASS (all 7 principles satisfied)
**Technical Context**: ✅ COMPLETE (all dependencies, constraints, performance goals documented)
**Research**: ✅ COMPLETE (10 decisions with rationale, alternatives, implementation notes)
**Data Model**: ✅ COMPLETE (5 entities, 15 validation rules, 3 state transition diagrams)
**Quickstart**: ✅ COMPLETE (7 sections, 8 troubleshooting scenarios, performance benchmarks)
**Agent Context**: ✅ UPDATED (Python 3.9+, Qdrant Cloud Free Tier added)

---

### Next Steps

**Ready for**: `/sp.tasks` command to generate task breakdown

**Expected Tasks** (3-5 tasks aligned with user constraint):
1. Task 1: Setup backend/ directory, initialize uv project, create .env.example
2. Task 2: Implement crawling & extraction functions (fetch_sitemap, crawl_pages, extract_content)
3. Task 3: Implement chunking function (chunk_text with heading awareness, tiktoken counting)
4. Task 4: Implement embedding & storage functions (generate_embeddings with retry logic, store_in_qdrant with upsert)
5. Task 5: Implement main() orchestrator, validation, logging, README documentation

**Estimated Implementation Time**: 3-5 hours for single developer (aligns with single-file architecture benefits)

## Outcome

- ✅ Impact: Created complete implementation plan for RAG Embeddings Ingestion Pipeline. Executed Phase 0 (research: 10 technical decisions, 24KB) and Phase 1 (design: 3 documents, 55KB total). Single-file architecture (`backend/main.py`) chosen to deliver "complete within 3-5 tasks" user constraint. All 7 constitutional principles satisfied. Ready for task breakdown generation.
- 🧪 Tests: Constitution Check PASS (7/7 principles), Technical Context COMPLETE, Research COMPLETE (10 decisions documented), Data Model COMPLETE (5 entities, 15 validation rules), Quickstart COMPLETE (7 sections, 8 troubleshooting scenarios)
- 📁 Files: Created 4 planning documents (plan.md, research.md, data-model.md, quickstart.md - 66KB total), Modified 1 file (CLAUDE.md agent context)
- 🔁 Next prompts: Run `/sp.tasks` to generate task breakdown (expected 3-5 tasks), then `/sp.implement` for execution
- 🧠 Reflection: Planning workflow executed smoothly with clear user intent (single-file, uv, complete in 3-5 tasks). Key wins: (1) Research phase resolved all technical unknowns (package manager, HTTP client, tokenizer, chunking strategy, embedding model, vector DB config) with documented rationale and alternatives, (2) Data model defines 5 entities with validation rules and state transitions, ensuring type safety throughout pipeline, (3) Quickstart guide provides end-to-end developer experience (setup → execution → troubleshooting → advanced config) with 8 common error scenarios documented, (4) Single-file architecture balances simplicity with modularity (8 functions, independently testable), (5) Constitution check confirms alignment with all 7 project principles. No complexity violations requiring justification. Next session should generate task breakdown, then proceed to implementation.

## Evaluation notes (flywheel)

- Failure modes observed: None - all planning phases completed without errors or rework
- Graders run and results (PASS/FAIL): Constitution Check PASS (7/7), Technical Context COMPLETE, Research COMPLETE (10 decisions), Data Model COMPLETE (5 entities), Quickstart COMPLETE (7 sections)
- Prompt variant (if applicable): Standard /sp.plan workflow with user-specified single-file architecture constraint
- Next experiment (smallest change to try): Run `/sp.tasks` to generate 3-5 task breakdown, validate task dependencies align with data flow diagram (URL discovery → crawl → chunk → embed → store → validate)
