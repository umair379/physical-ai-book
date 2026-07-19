# Feature Specification: RAG Embeddings Ingestion Pipeline

**Feature Branch**: `008-rag-embeddings`
**Created**: 2025-12-26
**Status**: Draft
**Input**: User description: "Deploy book URLs, generate embeddings, and store them in a vector database. Target audience: Developers integrating RAG with documentation websites. Focus: Reliable ingestion, embedding, and storage of book content for retrieval. Success criteria: All public Docusaurus URLs are crawled and cleaned, Text is chunked and embedded using Cohere models, Embeddings are stored and indexed in Qdrant successfully, Vector search returns relevant chunks for test queries. Constraints: Tech stack: Python, Cohere Embeddings, Qdrant (Cloud Free Tier), Data source: Deployed Vercel URLs only, Format: Modular scripts with clear config/env handling, Timeline: Complete within 3-5 tasks. Not building: Retrieval or ranking logic, Agent or chatbot logic, Frontend or FastAPI integration, User authentication or analytics"

## User Scenarios & Testing

### User Story 1 - Crawl and Extract Documentation Content (Priority: P1) 🎯 MVP

As a developer setting up RAG for documentation, I want to crawl all public Docusaurus URLs from the deployed Vercel site and extract clean text content so that I have structured data ready for embedding.

**Why this priority**: Crawling and extraction is the foundation - without reliably getting clean text from all documentation pages, embedding and storage are impossible. This is the first critical step that enables all subsequent work.

**Independent Test**: Can be fully tested by running the crawler script against the Vercel deployment URL and verifying that all expected pages are discovered, fetched, and cleaned (HTML stripped, markdown preserved, metadata extracted). Delivers standalone value by providing a reusable content extraction pipeline.

**Acceptance Scenarios**:

1. **Given** a deployed Docusaurus site URL on Vercel, **When** the crawler runs, **Then** it discovers all public HTML pages via sitemap or recursive link following
2. **Given** a list of discovered URLs, **When** the crawler fetches each page, **Then** it extracts the main content area (article/docs container) excluding navigation/footer/sidebar
3. **Given** raw HTML content, **When** the cleaning process runs, **Then** it produces plain text or markdown with whitespace normalized and code blocks preserved
4. **Given** extracted content, **When** metadata is captured, **Then** it includes URL, page title, breadcrumb path, and last-modified timestamp

---

### User Story 2 - Chunk Text for Optimal Retrieval (Priority: P2)

As a developer preparing documentation for RAG, I want to split extracted text into semantic chunks of appropriate size so that embedding and retrieval are accurate and context-aware.

**Why this priority**: Chunking strategy directly impacts retrieval quality - chunks that are too large lose specificity, chunks that are too small lose context. This must be done before embedding but is independent of crawling.

**Independent Test**: Can be tested by providing sample documentation text and verifying that the chunking algorithm produces chunks with target size (e.g., 512-1024 tokens), respects semantic boundaries (paragraphs, headings, code blocks), and includes overlap or context preservation where appropriate.

**Acceptance Scenarios**:

1. **Given** extracted documentation text, **When** the chunking algorithm runs, **Then** it splits text into chunks of configurable target size (default: 512 tokens)
2. **Given** text with headings and sections, **When** chunking occurs, **Then** it prefers splitting at heading boundaries rather than mid-paragraph
3. **Given** code blocks in documentation, **When** chunking occurs, **Then** it keeps code blocks intact within a single chunk when size permits
4. **Given** split chunks, **When** chunking completes, **Then** each chunk retains metadata (source URL, heading hierarchy, chunk index) for traceability

---

### User Story 3 - Generate and Store Vector Embeddings (Priority: P3)

As a developer building a RAG system, I want to generate vector embeddings for text chunks using Cohere models and store them in Qdrant so that I can perform semantic search over documentation content.

**Why this priority**: Embedding and storage enable the actual search capability, but they depend on having crawled and chunked content first. This is the final step to make content searchable.

**Independent Test**: Can be tested by providing sample text chunks, generating embeddings via Cohere API, storing them in a Qdrant collection, and verifying that vector search returns semantically similar chunks for test queries.

**Acceptance Scenarios**:

1. **Given** a batch of text chunks, **When** embeddings are generated, **Then** the Cohere API is called with the correct model (e.g., embed-english-v3.0) and returns vectors of expected dimensionality (e.g., 1024 dimensions)
2. **Given** generated embeddings and metadata, **When** they are stored in Qdrant, **Then** a collection is created (if not exists) with the correct vector configuration (dimension, distance metric)
3. **Given** embeddings stored in Qdrant, **When** a test semantic search query is executed, **Then** it returns the top-k most similar chunks with similarity scores above a threshold (e.g., > 0.7)
4. **Given** stored vectors, **When** the ingestion process runs again with updated content, **Then** it updates or replaces existing embeddings for changed chunks without duplicating

---

### Edge Cases

- What happens when a Docusaurus page is very large (> 50KB of text)? The chunker should split it into multiple chunks with preserved context.
- How does the crawler handle pages with dynamic JavaScript content? It should use a headless browser or ensure Docusaurus SSG provides full static HTML.
- What if the Cohere API rate limit is exceeded? The system should implement retry logic with exponential backoff and batch requests appropriately.
- How does the system handle Qdrant Cloud Free Tier limits (e.g., 1M vectors, 1 cluster)? It should validate collection size before ingestion and fail gracefully if limits are approached.
- What if a page returns 404 or 500 during crawling? The system should log the error, skip that URL, and continue with remaining pages without crashing.
- How does the chunking handle non-English content or special characters? It should preserve UTF-8 encoding and handle multilingual text gracefully.
- What happens if two chunks have identical embeddings? Qdrant should store both with unique IDs (URL + chunk index) to preserve all content.

## Requirements

### Functional Requirements

**Crawling & Extraction**

- **FR-001**: System MUST crawl all publicly accessible URLs from a deployed Docusaurus site on Vercel
- **FR-002**: System MUST extract the main documentation content (excluding navigation, sidebar, footer) from each HTML page
- **FR-003**: System MUST preserve markdown formatting, code blocks, and special characters during text cleaning
- **FR-004**: System MUST capture metadata for each page (URL, title, breadcrumb path, last-modified timestamp if available)
- **FR-005**: System MUST handle crawl errors gracefully (404, 500, timeouts) by logging and continuing without crashing

**Chunking**

- **FR-006**: System MUST split extracted text into chunks with a configurable target token size (default: 512 tokens, range: 256-1024)
- **FR-007**: System MUST respect semantic boundaries when chunking (prefer splitting at headings, paragraphs, code block boundaries)
- **FR-008**: System MUST keep code blocks intact within a single chunk when their size is below the target chunk size
- **FR-009**: System MUST attach metadata to each chunk (source URL, heading hierarchy, chunk index, total chunks for that URL)
- **FR-010**: System SHOULD implement overlapping chunks or context preservation (e.g., include previous heading in chunk metadata) to improve retrieval accuracy

**Embedding Generation**

- **FR-011**: System MUST generate vector embeddings using Cohere's embedding API (model: embed-english-v3.0 or latest stable version)
- **FR-012**: System MUST batch embedding requests to optimize API calls and respect rate limits (e.g., 100 texts per request)
- **FR-013**: System MUST implement retry logic with exponential backoff for transient API failures (rate limits, network errors)
- **FR-014**: System MUST validate embedding dimensions match expected model output (e.g., 1024 dimensions for embed-english-v3.0)

**Vector Storage**

- **FR-015**: System MUST store embeddings in Qdrant Cloud (Free Tier) with appropriate collection configuration (dimension, distance metric: cosine)
- **FR-016**: System MUST create a Qdrant collection if it doesn't exist, or update existing collection for re-ingestion
- **FR-017**: System MUST store chunk text, metadata (URL, title, heading, chunk index), and vector embedding for each chunk as a Qdrant point
- **FR-018**: System MUST implement upsert logic to update embeddings for changed content without creating duplicates (use URL + chunk index as unique ID)
- **FR-019**: System MUST validate Qdrant Cloud Free Tier limits (1M vectors max) before ingestion and fail gracefully if limit is exceeded

**Configuration & Environment**

- **FR-020**: System MUST support configuration via environment variables or config file (Vercel URL, Cohere API key, Qdrant URL/API key, chunking parameters)
- **FR-021**: System MUST validate all required environment variables/config at startup and fail fast with clear error messages if missing
- **FR-022**: System MUST use modular script design (separate scripts for crawl, chunk, embed, store) to enable independent testing and re-runs

**Validation & Testing**

- **FR-023**: System MUST provide a test script that executes semantic search queries against the populated Qdrant collection and validates results
- **FR-024**: System MUST log ingestion progress (pages crawled, chunks created, embeddings generated, vectors stored) with timestamps for monitoring

### Key Entities

**DocumentPage**
- **Purpose**: Represents a single Docusaurus page from the deployed site
- **Attributes**: URL (string, unique), title (string), breadcrumb path (list of strings), raw HTML content (string), extracted text (string), last-modified timestamp (datetime, optional)

**TextChunk**
- **Purpose**: Represents a semantic chunk of text ready for embedding
- **Attributes**: chunk ID (string, derived from URL + chunk index), source URL (string), chunk text (string), chunk index (int), total chunks for source (int), heading hierarchy (list of strings), token count (int)

**VectorEmbedding**
- **Purpose**: Represents a vector embedding with metadata stored in Qdrant
- **Attributes**: chunk ID (string, unique), vector (list of floats, 1024 dimensions), chunk text (string), metadata (dict: URL, title, heading, chunk index), timestamp (datetime)

**CrawlConfig**
- **Purpose**: Configuration for crawling and ingestion pipeline
- **Attributes**: base URL (string, Vercel deployment URL), Cohere API key (string), Qdrant URL (string), Qdrant API key (string), target chunk size (int, default 512), max crawl depth (int, default 10), batch size for embeddings (int, default 100)

## Success Criteria

### Measurable Outcomes

**Coverage & Quality**

- **SC-001**: 100% of publicly accessible Docusaurus pages (as listed in sitemap or discoverable via links) are crawled without errors
- **SC-002**: Extracted text from pages has >95% accuracy (no missing paragraphs, code blocks are preserved, minimal HTML artifacts)
- **SC-003**: Chunks have average token count within 10% of target size (e.g., 460-560 tokens for target 512)
- **SC-004**: <5% of chunks violate semantic boundaries (split mid-sentence or mid-code block)

**Performance & Reliability**

- **SC-005**: Full ingestion pipeline (crawl → chunk → embed → store) completes for a 100-page documentation site in under 30 minutes
- **SC-006**: Embedding generation succeeds for >99% of chunks (handles transient API failures with retries)
- **SC-007**: Vector storage in Qdrant succeeds for 100% of generated embeddings without duplicates or data loss

**Search Quality**

- **SC-008**: Semantic search for test queries (e.g., "How to set up authentication?") returns relevant chunks in top-5 results with similarity score >0.7
- **SC-009**: 90% of test queries (covering different topics from documentation) return at least one highly relevant chunk (similarity >0.8) in top-3 results
- **SC-010**: Re-ingestion of updated content (e.g., modified page) correctly updates embeddings without creating duplicate vectors

**Developer Experience**

- **SC-011**: Pipeline can be executed with a single command (e.g., `python main.py --config config.yaml`) after environment variables are set
- **SC-012**: Clear error messages are displayed for all failure modes (missing config, API errors, network issues, quota limits) with actionable suggestions
- **SC-013**: Logs provide sufficient detail to diagnose issues (page URLs processed, chunk counts, API call statistics, error stack traces)
- **SC-014**: Modular scripts allow re-running individual stages (e.g., re-embed without re-crawling) to save time during development

## Scope & Constraints

### In Scope

1. Crawling public Docusaurus URLs from deployed Vercel site
2. Extracting and cleaning HTML content to plain text or markdown
3. Chunking text with semantic boundary awareness
4. Generating vector embeddings via Cohere API
5. Storing embeddings in Qdrant Cloud (Free Tier)
6. Validating search quality with test queries
7. Modular Python scripts with environment variable configuration
8. Error handling and retry logic for API calls
9. Logging and progress tracking

### Out of Scope

1. Retrieval or ranking logic for RAG (this is ingestion only)
2. Agent or chatbot implementation
3. Frontend interface or FastAPI backend
4. User authentication or role-based access control
5. Analytics or usage tracking
6. Real-time updates or incremental ingestion (initial ingestion pipeline only)
7. Multi-language support beyond English (English-only for MVP)
8. Custom embedding models (Cohere only)
9. Alternative vector databases (Qdrant only)

### Technical Constraints

1. **Tech Stack**: Python 3.9+, Cohere Python SDK, Qdrant Python client
2. **Data Source**: Deployed Vercel URLs only (no local file ingestion)
3. **Vector Database**: Qdrant Cloud Free Tier (1M vectors max, 1 cluster)
4. **Embedding Model**: Cohere embed-english-v3.0 (or latest stable version)
5. **Rate Limits**: Cohere API trial/free tier rate limits (handled with batching and retries)
6. **Format**: Modular scripts (crawl.py, chunk.py, embed.py, store.py, main.py orchestrator)

### Dependencies

1. Deployed Docusaurus site on Vercel with public URLs
2. Cohere API account with API key (free tier or paid)
3. Qdrant Cloud account with cluster URL and API key (Free Tier)
4. Python environment with requests, beautifulsoup4, tiktoken (or Cohere tokenizer), cohere, qdrant-client libraries
5. Internet connectivity for API calls

### Assumptions

1. Docusaurus site is fully static (SSG) with no client-side routing that requires JavaScript execution (or crawler can handle JS if needed)
2. Sitemap is available at `/sitemap.xml` or pages are discoverable via recursive link crawling
3. Page HTML structure is consistent (Docusaurus default theme or similar) to enable reliable content extraction
4. Target chunk size of 512 tokens is appropriate for documentation retrieval (configurable if needed)
5. Cohere embed-english-v3.0 model is sufficient for English documentation (1024 dimensions)
6. Qdrant Free Tier limits (1M vectors) are sufficient for the documentation corpus size (estimate: <100K chunks for typical documentation site)
7. Cosine similarity is the appropriate distance metric for semantic search
8. No authentication is required to access Vercel-deployed URLs (public documentation)
9. Re-ingestion is infrequent enough that full re-crawl is acceptable (no delta/incremental ingestion in MVP)
10. Test queries can be manually defined based on documentation topics (no automated query generation)
