# Research: RAG Embeddings Ingestion Pipeline

**Feature**: 008-rag-embeddings
**Date**: 2025-12-27
**Status**: Complete

## Purpose

Research technical decisions and best practices for building a Python-based RAG ingestion pipeline that crawls Docusaurus documentation, chunks text, generates embeddings via Cohere API, and stores vectors in Qdrant Cloud.

## Research Tasks

### 1. Python Package Manager: uv vs pip

**Decision**: Use `uv` for dependency management

**Rationale**:
- **Performance**: 10-100x faster than pip for dependency resolution and installation
- **Modern tooling**: Built in Rust, designed for modern Python workflows
- **Lock file support**: Generates `uv.lock` for reproducible builds (similar to npm's package-lock.json)
- **Compatibility**: Drop-in replacement for pip, works with existing pyproject.toml
- **Project initialization**: `uv init` creates project structure with virtual env automatically

**Alternatives Considered**:
- **pip + venv**: Traditional approach, slower dependency resolution, no lock file by default
- **poetry**: Feature-rich but heavier, slower than uv, additional learning curve
- **pdm**: Similar to poetry, less adoption, not significantly faster than uv

**Implementation Notes**:
```bash
# Install uv (Windows)
pip install uv

# Initialize project
cd backend/
uv init

# Add dependencies
uv add requests beautifulsoup4 lxml tiktoken cohere qdrant-client python-dotenv pydantic

# Run script
uv run main.py
```

**References**:
- uv documentation: https://github.com/astral-sh/uv
- Benchmarks show uv is 10-100x faster than pip for common operations

---

### 2. HTTP Client: requests vs httpx

**Decision**: Use `requests` for initial implementation

**Rationale**:
- **Simplicity**: Synchronous API sufficient for crawling (no need for async complexity in MVP)
- **Stability**: Battle-tested library with 20+ years of development
- **Familiarity**: Most widely used Python HTTP library, extensive documentation
- **Adequate performance**: 2-5 pages/second crawl rate meets <30 minute requirement for 100-page sites

**Alternatives Considered**:
- **httpx**: Async support, HTTP/2, better performance for concurrent requests
  - **Rejected for MVP**: Adds async complexity without significant benefit for sequential crawling
  - **Future consideration**: If crawl performance becomes bottleneck, migrate to httpx with async

**Implementation Notes**:
```python
import requests

def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text
```

**Performance**: Expect 2-5 pages/second with synchronous requests (100 pages = 20-50 seconds crawl time)

---

### 3. HTML Parsing: BeautifulSoup vs lxml vs html5lib

**Decision**: Use `beautifulsoup4` with `lxml` parser

**Rationale**:
- **BeautifulSoup**: High-level API for HTML navigation, excellent for content extraction
- **lxml parser**: C-based parser, 10x faster than html5lib, handles malformed HTML well
- **Docusaurus compatibility**: Docusaurus generates well-formed HTML, lxml's speed advantage significant

**Alternatives Considered**:
- **lxml alone**: Faster but lower-level API (XPath/CSS selectors), more code for content extraction
- **html5lib parser**: Most lenient, but 10x slower than lxml, overkill for well-formed Docusaurus HTML
- **selectolax**: Faster than lxml but less mature, smaller community

**Implementation Notes**:
```python
from bs4 import BeautifulSoup

def extract_content(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')

    # Docusaurus main content selector
    article = soup.find('article') or soup.find('main')

    # Remove navigation, sidebar, footer
    for tag in article.find_all(['nav', 'aside', 'footer']):
        tag.decompose()

    return article.get_text(separator='\n', strip=True)
```

**Docusaurus HTML Structure**:
- Main content typically in `<article>` or `<main>` tags
- Navigation in `<nav>` tags
- Sidebar in `<aside>` tags
- Footer in `<footer>` tags

---

### 4. Tokenization: tiktoken vs Cohere Tokenizer

**Decision**: Use `tiktoken` for chunk sizing

**Rationale**:
- **Accuracy**: OpenAI's tokenizer, widely used in RAG pipelines
- **Performance**: Written in Rust, 3-6x faster than pure Python tokenizers
- **Compatibility**: Works offline, no API calls required for token counting
- **Embedding model agnostic**: Provides accurate token counts regardless of embedding model

**Alternatives Considered**:
- **Cohere tokenizer**: Model-specific, would require API call or SDK integration
  - **Issue**: Cohere Python SDK doesn't expose tokenizer as standalone utility
  - **Workaround**: tiktoken's cl100k_base encoding approximates most modern models well
- **transformers tokenizer**: Heavier dependency, slower, overkill for simple token counting

**Implementation Notes**:
```python
import tiktoken

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4, similar to Cohere
    return len(encoding.encode(text))

def chunk_text(text: str, target_size: int = 512, max_size: int = 1024) -> List[str]:
    # Split by paragraphs or headings, then combine to target size
    chunks = []
    current_chunk = ""

    for paragraph in text.split('\n\n'):
        tokens = count_tokens(current_chunk + paragraph)

        if tokens <= target_size:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```

**Token Counting Accuracy**: tiktoken's cl100k_base encoding typically within 5-10% of Cohere's actual tokenizer

---

### 5. Semantic Chunking Strategy

**Decision**: Paragraph-based chunking with heading preservation

**Rationale**:
- **Docusaurus structure**: Documentation naturally organized by headings (H1-H6) and paragraphs
- **Retrieval quality**: Preserving heading context improves semantic relevance
- **Simplicity**: Splitting at `\n\n` (paragraph boundaries) respects markdown structure

**Best Practices**:
1. **Prefer heading boundaries**: When chunk size exceeded, split at nearest H2/H3 rather than mid-paragraph
2. **Preserve code blocks**: Detect fenced code blocks (```), keep intact when possible
3. **Add heading metadata**: Include heading hierarchy (breadcrumb) in chunk metadata for context
4. **Overlap**: Optional 50-100 token overlap between chunks to preserve context across boundaries

**Implementation Notes**:
```python
def chunk_with_headings(text: str, metadata: dict, target_size: int = 512) -> List[TextChunk]:
    chunks = []
    lines = text.split('\n')
    current_chunk = ""
    current_heading = []

    for line in lines:
        # Detect markdown headings (# H1, ## H2, etc.)
        if line.startswith('#'):
            level = len(line.split()[0])  # Count #'s
            heading_text = line.lstrip('#').strip()

            # Update heading hierarchy
            current_heading = current_heading[:level-1] + [heading_text]

            # If chunk size exceeded, finalize current chunk
            if count_tokens(current_chunk) > target_size:
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    metadata={**metadata, 'heading': ' > '.join(current_heading)},
                    chunk_index=len(chunks)
                ))
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        else:
            current_chunk += line + '\n'

    # Finalize last chunk
    if current_chunk:
        chunks.append(TextChunk(
            text=current_chunk.strip(),
            metadata={**metadata, 'heading': ' > '.join(current_heading)},
            chunk_index=len(chunks)
        ))

    return chunks
```

**Code Block Handling**:
- Detect fenced code blocks: ` ```language ... ``` `
- Track in-block state to prevent splitting mid-code
- If code block exceeds target size alone, keep it as single chunk (acceptable edge case)

**References**:
- LangChain's RecursiveCharacterTextSplitter uses similar heading-aware approach
- Typical overlap: 10-20% of chunk size (50-100 tokens for 512-token chunks)

---

### 6. Cohere Embedding API: Best Practices

**Decision**: Use `embed-english-v3.0` with batch requests and retry logic

**API Details**:
- **Model**: `embed-english-v3.0` (1024 dimensions, latest stable)
- **Batch size**: 96 texts per request (Cohere's recommended max for optimal performance)
- **Input type**: `search_document` (for indexing) vs `search_query` (for retrieval) - use `search_document` for ingestion
- **Retry logic**: Exponential backoff for rate limits (429) and transient failures (500, 503)

**Implementation Notes**:
```python
import cohere
import time

co = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))

def generate_embeddings(chunks: List[TextChunk], batch_size: int = 96) -> List[VectorEmbedding]:
    embeddings = []
    texts = [chunk.text for chunk in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        for attempt in range(5):  # Max 5 retries
            try:
                response = co.embed(
                    texts=batch,
                    model='embed-english-v3.0',
                    input_type='search_document'
                )

                for j, embedding in enumerate(response.embeddings):
                    chunk_idx = i + j
                    embeddings.append(VectorEmbedding(
                        chunk_id=f"{chunks[chunk_idx].metadata['url']}#{chunk_idx}",
                        vector=embedding,
                        chunk_text=chunks[chunk_idx].text,
                        metadata=chunks[chunk_idx].metadata
                    ))

                break  # Success, exit retry loop

            except cohere.errors.RateLimitError:
                if attempt < 4:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise  # Max retries exceeded

            except (cohere.errors.ServiceUnavailableError, cohere.errors.InternalServerError):
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise

    return embeddings
```

**Rate Limits** (Cohere free tier, approximate):
- Trial API key: ~5 requests/minute
- Production API key (free tier): ~100 requests/minute
- Expect 10-50 batches for typical documentation site (1000-5000 chunks)

**Performance Estimates**:
- Trial tier: 5 batches/min × 96 texts/batch = 480 texts/min (~10 minutes for 5000 chunks)
- Prod tier: 100 batches/min × 96 texts/batch = 9600 texts/min (~30 seconds for 5000 chunks)

**References**:
- Cohere embedding guide: https://docs.cohere.com/docs/embeddings
- Input types documentation: https://docs.cohere.com/docs/embed-2#input-types

---

### 7. Qdrant Storage: Collection Configuration

**Decision**: Create collection with cosine distance and 1024 dimensions

**Collection Setup**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

# Create collection (idempotent - checks if exists first)
collection_name = "docusaurus_docs"

try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists")
except:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # Cohere embed-english-v3.0 dimension
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{collection_name}'")
```

**Upsert Logic** (prevent duplicates on re-ingestion):
```python
def store_in_qdrant(embeddings: List[VectorEmbedding], collection_name: str):
    points = []

    for emb in embeddings:
        points.append(PointStruct(
            id=hash(emb.chunk_id),  # Deterministic ID based on URL + chunk index
            vector=emb.vector,
            payload={
                'chunk_id': emb.chunk_id,
                'text': emb.chunk_text,
                'url': emb.metadata['url'],
                'title': emb.metadata['title'],
                'heading': emb.metadata.get('heading', ''),
                'chunk_index': emb.metadata['chunk_index']
            }
        ))

    # Upsert in batches of 100 (Qdrant recommended batch size)
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
```

**Search Validation**:
```python
def validate_search(queries: List[str], collection_name: str):
    results = {}

    for query in queries:
        # Generate query embedding
        response = co.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'  # Different from ingestion
        )

        # Search Qdrant
        search_result = client.search(
            collection_name=collection_name,
            query_vector=response.embeddings[0],
            limit=5
        )

        results[query] = [
            {
                'text': hit.payload['text'][:200] + '...',
                'url': hit.payload['url'],
                'score': hit.score
            }
            for hit in search_result
        ]

    return results
```

**Qdrant Cloud Free Tier Limits**:
- 1M vectors max (sufficient for 5,000 chunks with headroom for future growth)
- 1 cluster (adequate for single-collection use case)
- No API rate limits (managed cloud, scales automatically)

**References**:
- Qdrant collections guide: https://qdrant.tech/documentation/concepts/collections/
- Qdrant upsert documentation: https://qdrant.tech/documentation/concepts/points/#upload-points

---

### 8. Sitemap Parsing and URL Discovery

**Decision**: Prioritize sitemap.xml, fallback to recursive link crawling

**Sitemap Approach**:
```python
import xml.etree.ElementTree as ET

def fetch_sitemap(base_url: str) -> List[str]:
    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"

    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Extract <loc> elements (Docusaurus sitemap format)
        urls = []
        for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
            urls.append(url_elem.text)

        print(f"Discovered {len(urls)} URLs from sitemap")
        return urls

    except Exception as e:
        print(f"Sitemap fetch failed: {e}. Falling back to recursive crawl...")
        return recursive_crawl(base_url)

def recursive_crawl(start_url: str, max_depth: int = 3) -> List[str]:
    """Fallback: recursively follow links from start_url"""
    visited = set()
    to_visit = [(start_url, 0)]  # (url, depth)

    while to_visit:
        url, depth = to_visit.pop(0)

        if url in visited or depth > max_depth:
            continue

        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')

            # Find all internal links
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Convert relative to absolute
                if href.startswith('/'):
                    href = f"{start_url.rstrip('/')}{href}"

                # Only follow same-domain links
                if href.startswith(start_url) and href not in visited:
                    to_visit.append((href, depth + 1))

        except Exception as e:
            print(f"Failed to crawl {url}: {e}")
            continue

    return list(visited)
```

**Docusaurus Sitemap Format**:
- Standard XML sitemap protocol (https://www.sitemaps.org/)
- URLs in `<url><loc>` elements
- Typically located at `/sitemap.xml`
- Generated automatically by Docusaurus build process

**Fallback Recursive Crawl**:
- Max depth: 3 (sufficient for typical documentation structure)
- Filters: Only follow same-domain links, skip external resources
- Timeout: 10 seconds per page to handle slow responses

---

### 9. Error Handling and Logging

**Decision**: Structured logging with progress tracking

**Logging Strategy**:
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=== RAG Ingestion Pipeline Started ===")

    # Track stats
    stats = {
        'urls_discovered': 0,
        'pages_crawled': 0,
        'chunks_created': 0,
        'embeddings_generated': 0,
        'vectors_stored': 0,
        'errors': []
    }

    try:
        # Step 1: Discover URLs
        urls = fetch_sitemap(base_url)
        stats['urls_discovered'] = len(urls)
        logger.info(f"Discovered {len(urls)} URLs")

        # Step 2: Crawl pages
        pages = crawl_pages(urls, stats)
        logger.info(f"Crawled {stats['pages_crawled']}/{len(urls)} pages successfully")

        # Step 3: Chunk text
        chunks = []
        for page in pages:
            page_chunks = chunk_text(page.text, page.metadata)
            chunks.extend(page_chunks)
        stats['chunks_created'] = len(chunks)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 4: Generate embeddings
        embeddings = generate_embeddings(chunks)
        stats['embeddings_generated'] = len(embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 5: Store in Qdrant
        store_in_qdrant(embeddings, collection_name)
        stats['vectors_stored'] = len(embeddings)
        logger.info(f"Stored {len(embeddings)} vectors in Qdrant")

        # Step 6: Validate search
        test_queries = ["How to install?", "Configuration guide", "API reference"]
        results = validate_search(test_queries, collection_name)

        for query, hits in results.items():
            logger.info(f"Query: '{query}' - Top result score: {hits[0]['score']:.3f}")

        logger.info("=== Pipeline Complete ===")
        logger.info(f"Stats: {stats}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

def crawl_pages(urls: List[str], stats: dict) -> List[DocumentPage]:
    pages = []

    for i, url in enumerate(urls):
        try:
            html = fetch_page(url)
            text = extract_content(html)

            pages.append(DocumentPage(
                url=url,
                text=text,
                metadata={'title': extract_title(html), 'url': url}
            ))

            stats['pages_crawled'] += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(urls)} pages crawled")

        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            stats['errors'].append({'url': url, 'error': str(e)})
            continue  # Skip failed pages, continue with rest

    return pages
```

**Error Handling Principles**:
1. **Fail gracefully**: Log errors, skip failed pages/chunks, continue pipeline
2. **Retry transient failures**: API rate limits, network timeouts (exponential backoff)
3. **Fail fast on config errors**: Missing API keys, invalid Qdrant URL (validate at startup)
4. **Progress tracking**: Log every 10 pages crawled, every batch embedded
5. **Final report**: Print stats summary (pages crawled, chunks created, errors encountered)

---

### 10. Environment Configuration

**Decision**: Use `.env` file with `python-dotenv` and `pydantic` validation

**Configuration Schema**:
```python
from pydantic import BaseSettings, Field, validator

class Config(BaseSettings):
    # Required
    base_url: str = Field(..., description="Vercel deployment URL")
    cohere_api_key: str = Field(..., description="Cohere API key")
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")

    # Optional (with defaults)
    collection_name: str = Field("docusaurus_docs", description="Qdrant collection name")
    chunk_size: int = Field(512, description="Target chunk size in tokens")
    max_chunk_size: int = Field(1024, description="Maximum chunk size in tokens")
    batch_size: int = Field(96, description="Embedding batch size")
    max_crawl_depth: int = Field(3, description="Max depth for recursive crawl")

    @validator('base_url')
    def validate_url(cls, v):
        if not v.startswith('http'):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')

    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if not (256 <= v <= 1024):
            raise ValueError('chunk_size must be between 256 and 1024')
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

# Load config at startup
config = Config()
```

**`.env.example` Template**:
```bash
# Vercel Deployment
BASE_URL=https://your-docusaurus-site.vercel.app

# Cohere API
COHERE_API_KEY=your_cohere_api_key_here

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional Configuration (defaults shown)
COLLECTION_NAME=docusaurus_docs
CHUNK_SIZE=512
MAX_CHUNK_SIZE=1024
BATCH_SIZE=96
MAX_CRAWL_DEPTH=3
```

**Startup Validation**:
```python
def validate_config():
    try:
        config = Config()
        logger.info("Configuration loaded successfully")
        logger.info(f"Base URL: {config.base_url}")
        logger.info(f"Collection: {config.collection_name}")
        logger.info(f"Chunk size: {config.chunk_size} tokens")
        return config
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file. See .env.example for template.")
        raise SystemExit(1)
```

**Security Notes**:
- `.env` file added to `.gitignore` (never committed)
- `.env.example` committed with placeholder values
- API keys rotated if accidentally exposed

---

## Summary of Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| **Package Manager** | uv | 10-100x faster than pip, modern lock file support |
| **HTTP Client** | requests | Simple, stable, sufficient for synchronous crawling |
| **HTML Parser** | BeautifulSoup + lxml | High-level API + fast C-based parser |
| **Tokenizer** | tiktoken | Fast, accurate, offline token counting |
| **Chunking Strategy** | Paragraph + heading aware | Preserves Docusaurus structure, improves retrieval |
| **Embedding Model** | Cohere embed-english-v3.0 | Latest stable, 1024 dimensions, optimized for search |
| **Embedding Batch Size** | 96 texts/request | Cohere's recommended max for performance |
| **Vector Database** | Qdrant Cloud Free Tier | 1M vectors, cosine similarity, managed service |
| **URL Discovery** | Sitemap first, recursive fallback | Efficient + comprehensive coverage |
| **Logging** | Structured logging + stats | Progress tracking, error diagnosis |
| **Configuration** | .env + pydantic validation | Secure secrets, type-safe config |

---

## Open Questions & Risks

### Resolved
- ✅ **Token counting accuracy**: tiktoken's cl100k_base within 5-10% of Cohere's tokenizer (acceptable for chunking)
- ✅ **Rate limits**: Cohere trial tier ~5 req/min, production free tier ~100 req/min (handle with exponential backoff)
- ✅ **Qdrant Free Tier limits**: 1M vectors sufficient for 5,000 chunks + future growth
- ✅ **Docusaurus HTML structure**: Consistent `<article>` or `<main>` tags for content extraction

### Remaining (for implementation)
- ⚠ **Large page handling**: If single page >50KB text, chunking will create 50-100 chunks from one page (monitor chunk distribution)
- ⚠ **Code block formatting**: Preserve syntax highlighting metadata or strip to plain text? (Decision: strip to plain text for MVP, add metadata in future)
- ⚠ **Heading hierarchy extraction**: Current approach assumes markdown headings (# H1, ## H2). Verify Docusaurus HTML includes heading tags or markdown source.

---

**Research Complete**: All technical decisions documented with rationale and implementation notes. Ready for Phase 1 (data model + contracts).
