# Physical AI Book - Backend

This directory contains the backend services for the Physical AI book project, including data ingestion, retrieval pipeline, and RAG agent.

## Features

### Data Ingestion Pipeline (`main.py`)
- **Sitemap Discovery**: Automatically discovers URLs from sitemap.xml with fallback to recursive crawling
- **Smart HTML Extraction**: Extracts main content from Docusaurus pages, removing navigation and sidebars
- **Semantic Chunking**: Respects heading boundaries while maintaining target chunk size (512 tokens)
- **Cohere Embeddings**: Uses embed-english-v3.0 model (1024 dimensions)
- **Qdrant Storage**: Stores vectors with metadata in Qdrant Cloud
- **Search Validation**: Tests semantic search with configurable test queries

### Retrieval Pipeline (`retrieve.py`)
- Semantic search over book content
- Query embedding generation
- Top-k retrieval from Qdrant

### RAG Agent (`agent.py`) ⭐ NEW - Feature 010
- AI agent powered by OpenAI Assistants API
- Answers questions using only book content
- Source citations in responses
- Conversation history for follow-up queries
- Interactive and single-query modes

## Prerequisites

- Python 3.9+ (recommended: 3.11+)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Cohere API key (free tier available at https://cohere.com)
- Qdrant Cloud cluster (free tier available at https://cloud.qdrant.io)

## Installation

### 1. Install uv (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

```bash
cd backend
uv sync
```

This will:
- Create a virtual environment at `.venv/`
- Install all dependencies from `pyproject.toml`
- Lock versions in `uv.lock`

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```bash
# OpenAI API (for RAG Agent)
OPENAI_API_KEY="sk-proj-your-openai-key-here"

# Vercel Deployment
DEPLOY_VERCEL_URL="https://your-site.vercel.app"

# Cohere API
COHERE_API_KEY="your_cohere_api_key_here"

# Qdrant Cloud
QDRANT_URL="https://your-cluster-id.qdrant.io:6333"
QDRANT_API_KEY="your_qdrant_api_key_here"

# Optional Configuration (defaults shown)
COLLECTION_NAME=docusaurus_docs
CHUNK_SIZE=512
MAX_CHUNK_SIZE=1024
BATCH_SIZE=96
MAX_CRAWL_DEPTH=3
```

## Usage

### RAG Agent (Feature 010) - Recommended Starting Point

**Single Query Mode:**
```bash
uv run python agent.py "What is physical AI?"
```

**Interactive Mode:**
```bash
uv run python agent.py
# Then type your questions
# Type 'exit' or 'quit' to stop
```

**Example Conversation:**
```
You: What is physical AI?
Agent: Physical AI refers to AI systems that interact with the physical world...
[Source: Introduction - https://physical-ai-book...]

You: What are its key components?
Agent: Based on the previous context, the key components include...
```

### Retrieval Validation (Feature 009)

Test the retrieval pipeline:

```bash
uv run python retrieve.py --query "What is ROS 2?" --top-k 3
```

### Data Ingestion Pipeline (Feature 008)

Run the full ingestion pipeline using configuration from `.env`:

```bash
uv run python main.py
```

**Command-Line Options:**

```bash
# Override base URL
uv run python main.py --base-url https://example.com

# Override collection name
uv run python main.py --collection my_docs

# Specify custom test queries
uv run python main.py --test-queries "What is AI?" "Explain transformers"
```

**Help:**

```bash
uv run python main.py --help
```

## Pipeline Stages

The pipeline executes in the following phases:

### Phase 1: Setup
- Load and validate configuration
- Initialize logging (console + timestamped file)

### Phase 2: URL Discovery
- Fetch sitemap.xml from base URL
- Fallback to recursive crawling if sitemap unavailable
- Log discovered URLs

### Phase 3: Crawling & Extraction
- Fetch each page with timeout (10s)
- Extract main content using BeautifulSoup
- Remove navigation, sidebars, footers
- Extract metadata (title, breadcrumb)
- Track success/failure statistics

### Phase 4: Text Chunking
- Split text into semantic chunks (target: 512 tokens)
- Detect markdown headings and preserve hierarchy
- Respect heading boundaries when splitting
- Validate chunk sizes (min: 50, max: 1024 tokens)

### Phase 5: Embedding Generation
- Generate embeddings via Cohere API (embed-english-v3.0)
- Process in batches (default: 96 chunks)
- Validate embedding dimensions (1024)
- Retry with exponential backoff on failures

### Phase 6: Vector Storage
- Create Qdrant collection (cosine distance)
- Upsert vectors with metadata
- Track storage statistics

### Phase 7: Search Validation
- Run test queries against stored vectors
- Validate similarity scores (target: >0.7)
- Report average and minimum scores

## Output

### Console Logging

The pipeline logs progress to both console and a timestamped log file:

```
======================================================================
RAG Embeddings Ingestion Pipeline
======================================================================
2025-12-27 10:30:00 - INFO - Configuration loaded successfully
2025-12-27 10:30:00 - INFO - Base URL: https://example.com
...
======================================================================
Pipeline Complete - Final Statistics
======================================================================
URLs discovered:        150
Pages crawled:          148
Pages failed:           2
Success rate:           98.7%
Chunks created:         450
Embeddings generated:   450
Vectors stored:         450
Duration:               180.5s (3.0 minutes)
Collection:             docusaurus_docs
======================================================================
✓ RAG ingestion pipeline completed successfully!
```

### Log File

Timestamped log file: `ingestion_YYYYMMDD_HHMMSS.log`

Contains detailed logs including:
- Configuration validation
- URL discovery
- Per-page crawling results
- Chunking statistics
- Embedding batch processing
- Vector storage operations
- Search validation results
- Error details (if any)

## Configuration Reference

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DEPLOY_VERCEL_URL` | Base URL of Docusaurus site | `https://example.vercel.app` |
| `COHERE_API_KEY` | Cohere API key | `abc123...` |
| `QDRANT_URL` | Qdrant cluster URL | `https://xyz.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant API key | `eyJ...` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | `docusaurus_docs` | Qdrant collection name |
| `CHUNK_SIZE` | `512` | Target chunk size (tokens) |
| `MAX_CHUNK_SIZE` | `1024` | Maximum chunk size (tokens) |
| `BATCH_SIZE` | `96` | Embedding batch size |
| `MAX_CRAWL_DEPTH` | `3` | Max depth for recursive crawl |

## Troubleshooting

### Configuration Errors

**Problem**: `Configuration error: field required`

**Solution**: Ensure all required variables are set in `.env`:
- DEPLOY_VERCEL_URL
- COHERE_API_KEY
- QDRANT_URL
- QDRANT_API_KEY

### Sitemap Not Found

**Problem**: `Sitemap fetch failed: 404. Falling back to recursive crawl...`

**Solution**: This is expected if the site doesn't have sitemap.xml. The pipeline will automatically use recursive crawling.

### HTTP Timeout Errors

**Problem**: `Timeout fetching https://...`

**Solution**:
- Check your internet connection
- Verify the base URL is accessible
- Some sites may be slow - the pipeline will retry and continue

### Cohere API Errors

**Problem**: `Cohere API rate limit exceeded`

**Solution**:
- Free tier has rate limits
- Reduce `BATCH_SIZE` in `.env`
- Wait and retry (the pipeline has automatic retry with backoff)

### Qdrant Connection Errors

**Problem**: `Failed to connect to Qdrant`

**Solution**:
- Verify `QDRANT_URL` includes `:6333` port
- Check API key is correct
- Ensure cluster is active in Qdrant Cloud console

### Low Similarity Scores

**Problem**: `Some queries below 0.7 similarity threshold`

**Solution**:
- This may indicate content quality or chunking issues
- Try adjusting `CHUNK_SIZE` (e.g., 768 instead of 512)
- Check that the site content matches your test queries
- Review the log file to see actual search results

## Architecture

### Single-File Design

All code is in `main.py` (~880 lines) for simplicity:

```
main.py
├── Imports (26 lines)
├── Config & Dataclasses (154 lines)
│   ├── Config (pydantic BaseSettings)
│   ├── DocumentPage
│   ├── TextChunk
│   ├── VectorEmbedding
│   └── IngestionStats
├── Logging Setup (14 lines)
├── Phase 3: Crawling (164 lines)
│   ├── fetch_sitemap()
│   ├── recursive_crawl()
│   ├── fetch_page()
│   ├── extract_content()
│   ├── extract_metadata()
│   └── crawl_pages()
├── Phase 4: Chunking (120 lines)
│   ├── count_tokens()
│   ├── detect_headings()
│   ├── chunk_text()
│   └── chunk_all_pages()
├── Phase 5: Embedding & Storage (198 lines)
│   ├── retry_with_backoff() [decorator]
│   ├── generate_embeddings()
│   ├── create_qdrant_collection()
│   ├── store_in_qdrant()
│   └── validate_search()
└── Phase 6: Integration (190 lines)
    └── main() [orchestration]
```

### Dependencies

**Core**:
- `requests` - HTTP client
- `beautifulsoup4` + `lxml` - HTML parsing
- `tiktoken` - Token counting (OpenAI's tokenizer)
- `cohere` - Embedding API client
- `qdrant-client` - Vector database client
- `pydantic` - Configuration validation
- `python-dotenv` - Environment variables

**Total**: 44 packages (8 direct + 36 transitive)

## Development

### Project Structure

```
backend/
├── main.py              # All implementation
├── .env                 # Secrets (gitignored)
├── .env.example         # Template
├── pyproject.toml       # uv dependencies
├── uv.lock              # Locked versions
├── README.md            # This file
├── .venv/               # Virtual environment (gitignored)
└── *.log                # Log files (gitignored)
```

### Running Tests

The pipeline includes built-in validation via `validate_search()`. For manual testing:

```python
# Test chunking with sample text
from main import chunk_text

text = """
# Introduction

This is a test document.

## Section 1

Content here...
"""

chunks = chunk_text(text, {'url': 'test', 'title': 'Test'})
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.token_count} tokens")
```

### Adding Custom Test Queries

Specify test queries via CLI:

```bash
uv run python main.py --test-queries \
  "What is reinforcement learning?" \
  "How do transformers work?" \
  "Explain gradient descent"
```

## Success Criteria

The pipeline validates against these criteria:

- **SC-001**: 100% of pages crawled (check `success_rate` in final stats)
- **SC-003**: Chunk token counts within 10% of target (check `Average tokens` in output)
- **SC-008**: Test queries return similarity >0.7 (check validation results)

## License

MIT

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the log file for detailed error messages
3. Open an issue in the project repository
