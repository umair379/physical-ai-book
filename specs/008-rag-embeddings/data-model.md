# Data Model: RAG Embeddings Ingestion Pipeline

**Feature**: 008-rag-embeddings
**Date**: 2025-12-27
**Status**: Complete

## Purpose

Define data structures for the RAG ingestion pipeline, extracted from functional requirements in spec.md. These models ensure type safety and validation throughout the pipeline (crawling, chunking, embedding, storage).

## Entities

### 1. Config

**Purpose**: Application configuration loaded from environment variables

**Source**: FR-020 (configuration via environment variables)

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `base_url` | `str` | ✅ | - | Must start with `http://` or `https://` | Vercel deployment URL |
| `cohere_api_key` | `str` | ✅ | - | Non-empty string | Cohere API key |
| `qdrant_url` | `str` | ✅ | - | Must start with `http://` or `https://` | Qdrant Cloud cluster URL |
| `qdrant_api_key` | `str` | ✅ | - | Non-empty string | Qdrant API key |
| `collection_name` | `str` | ❌ | `"docusaurus_docs"` | Non-empty string | Qdrant collection name |
| `chunk_size` | `int` | ❌ | `512` | 256 ≤ value ≤ 1024 | Target chunk size in tokens |
| `max_chunk_size` | `int` | ❌ | `1024` | ≥ `chunk_size` | Maximum chunk size in tokens |
| `batch_size` | `int` | ❌ | `96` | 1 ≤ value ≤ 96 | Embedding batch size |
| `max_crawl_depth` | `int` | ❌ | `3` | 1 ≤ value ≤ 10 | Max depth for recursive crawl |

**Pydantic Implementation**:
```python
from pydantic import BaseSettings, Field, validator

class Config(BaseSettings):
    # Required
    base_url: str = Field(..., description="Vercel deployment URL")
    cohere_api_key: str = Field(..., description="Cohere API key")
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")

    # Optional
    collection_name: str = Field("docusaurus_docs")
    chunk_size: int = Field(512)
    max_chunk_size: int = Field(1024)
    batch_size: int = Field(96)
    max_crawl_depth: int = Field(3)

    @validator('base_url', 'qdrant_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v.rstrip('/')

    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if not (256 <= v <= 1024):
            raise ValueError('chunk_size must be between 256 and 1024')
        return v

    @validator('max_chunk_size')
    def validate_max_chunk_size(cls, v, values):
        if 'chunk_size' in values and v < values['chunk_size']:
            raise ValueError('max_chunk_size must be >= chunk_size')
        return v

    class Config:
        env_file = '.env'
```

---

### 2. DocumentPage

**Purpose**: Represents a single Docusaurus page fetched from the deployed site

**Source**: FR-001 to FR-005 (crawling & extraction), Key Entities in spec.md

**Attributes**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | `str` | ✅ | Unique page URL (primary identifier) |
| `title` | `str` | ✅ | Page title extracted from `<title>` tag |
| `raw_html` | `str` | ✅ | Original HTML content (for debugging) |
| `extracted_text` | `str` | ✅ | Clean text after removing nav/sidebar/footer |
| `breadcrumb` | `List[str]` | ❌ | Navigation breadcrumb (e.g., ["Docs", "Guide", "Installation"]) |
| `last_modified` | `Optional[datetime]` | ❌ | Last-modified timestamp from HTTP headers or sitemap |

**Dataclass Implementation**:
```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class DocumentPage:
    url: str
    title: str
    raw_html: str
    extracted_text: str
    breadcrumb: List[str] = None
    last_modified: Optional[datetime] = None

    def __post_init__(self):
        if self.breadcrumb is None:
            self.breadcrumb = []
```

**Usage Example**:
```python
page = DocumentPage(
    url="https://example.vercel.app/docs/intro",
    title="Introduction",
    raw_html="<html>...</html>",
    extracted_text="This is the intro page...",
    breadcrumb=["Docs", "Introduction"],
    last_modified=datetime(2025, 12, 27, 10, 30)
)
```

---

### 3. TextChunk

**Purpose**: Represents a semantic chunk of text ready for embedding

**Source**: FR-006 to FR-010 (chunking), Key Entities in spec.md

**Attributes**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chunk_id` | `str` | ✅ | Unique identifier (format: `{url}#{chunk_index}`) |
| `source_url` | `str` | ✅ | URL of source DocumentPage |
| `chunk_text` | `str` | ✅ | The actual text content of the chunk |
| `chunk_index` | `int` | ✅ | Zero-based index of chunk within source page |
| `total_chunks` | `int` | ✅ | Total number of chunks from source page |
| `heading_hierarchy` | `List[str]` | ✅ | Breadcrumb of headings (e.g., ["Installation", "Prerequisites"]) |
| `token_count` | `int` | ✅ | Actual token count (measured with tiktoken) |
| `metadata` | `dict` | ✅ | Additional metadata (title, breadcrumb, etc.) |

**Dataclass Implementation**:
```python
@dataclass
class TextChunk:
    chunk_id: str
    source_url: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    heading_hierarchy: List[str]
    token_count: int
    metadata: dict

    @staticmethod
    def create_id(url: str, index: int) -> str:
        """Generate deterministic chunk ID"""
        return f"{url}#{index}"

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'source_url': self.source_url,
            'chunk_text': self.chunk_text,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'heading_hierarchy': self.heading_hierarchy,
            'token_count': self.token_count,
            'metadata': self.metadata
        }
```

**Usage Example**:
```python
chunk = TextChunk(
    chunk_id="https://example.vercel.app/docs/intro#0",
    source_url="https://example.vercel.app/docs/intro",
    chunk_text="This is the introduction to our documentation...",
    chunk_index=0,
    total_chunks=3,
    heading_hierarchy=["Introduction"],
    token_count=487,
    metadata={
        'title': 'Introduction',
        'breadcrumb': ['Docs', 'Introduction'],
        'url': 'https://example.vercel.app/docs/intro'
    }
)
```

---

### 4. VectorEmbedding

**Purpose**: Represents a vector embedding with metadata for storage in Qdrant

**Source**: FR-011 to FR-019 (embedding generation & storage), Key Entities in spec.md

**Attributes**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chunk_id` | `str` | ✅ | Unique identifier (same as TextChunk.chunk_id) |
| `vector` | `List[float]` | ✅ | 1024-dimensional embedding from Cohere |
| `chunk_text` | `str` | ✅ | Original chunk text (stored for display in search results) |
| `metadata` | `dict` | ✅ | Metadata from TextChunk (url, title, heading, chunk_index) |
| `timestamp` | `datetime` | ✅ | When embedding was generated |

**Dataclass Implementation**:
```python
@dataclass
class VectorEmbedding:
    chunk_id: str
    vector: List[float]
    chunk_text: str
    metadata: dict
    timestamp: datetime

    def to_qdrant_point(self) -> dict:
        """Convert to Qdrant PointStruct format"""
        return {
            'id': hash(self.chunk_id),  # Deterministic numeric ID
            'vector': self.vector,
            'payload': {
                'chunk_id': self.chunk_id,
                'text': self.chunk_text,
                'url': self.metadata.get('url', ''),
                'title': self.metadata.get('title', ''),
                'heading': ' > '.join(self.metadata.get('heading_hierarchy', [])),
                'chunk_index': self.metadata.get('chunk_index', 0),
                'timestamp': self.timestamp.isoformat()
            }
        }

    @staticmethod
    def validate_dimensions(vector: List[float], expected: int = 1024):
        """Validate embedding dimensions (FR-014)"""
        if len(vector) != expected:
            raise ValueError(f"Expected {expected} dimensions, got {len(vector)}")
```

**Usage Example**:
```python
embedding = VectorEmbedding(
    chunk_id="https://example.vercel.app/docs/intro#0",
    vector=[0.123, -0.456, ..., 0.789],  # 1024 floats
    chunk_text="This is the introduction to our documentation...",
    metadata={
        'url': 'https://example.vercel.app/docs/intro',
        'title': 'Introduction',
        'heading_hierarchy': ['Introduction'],
        'chunk_index': 0
    },
    timestamp=datetime.now()
)

# Validate dimensions before storage
VectorEmbedding.validate_dimensions(embedding.vector)

# Convert to Qdrant format
point = embedding.to_qdrant_point()
```

---

### 5. IngestionStats

**Purpose**: Track pipeline execution statistics for logging and monitoring

**Source**: FR-024 (logging ingestion progress)

**Attributes**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `urls_discovered` | `int` | `0` | Number of URLs found in sitemap/crawl |
| `pages_crawled` | `int` | `0` | Number of pages successfully fetched |
| `pages_failed` | `int` | `0` | Number of pages that failed to crawl |
| `chunks_created` | `int` | `0` | Total number of chunks generated |
| `embeddings_generated` | `int` | `0` | Number of embeddings successfully created |
| `vectors_stored` | `int` | `0` | Number of vectors uploaded to Qdrant |
| `errors` | `List[dict]` | `[]` | List of errors encountered (url, error message) |
| `start_time` | `datetime` | `now()` | Pipeline start timestamp |
| `end_time` | `Optional[datetime]` | `None` | Pipeline end timestamp |

**Dataclass Implementation**:
```python
@dataclass
class IngestionStats:
    urls_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    errors: List[dict] = None
    start_time: datetime = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now()

    def finalize(self):
        """Mark pipeline complete"""
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Pipeline duration in seconds"""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Percentage of pages successfully crawled"""
        total = self.pages_crawled + self.pages_failed
        return (self.pages_crawled / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return {
            'urls_discovered': self.urls_discovered,
            'pages_crawled': self.pages_crawled,
            'pages_failed': self.pages_failed,
            'chunks_created': self.chunks_created,
            'embeddings_generated': self.embeddings_generated,
            'vectors_stored': self.vectors_stored,
            'success_rate': f"{self.success_rate:.1f}%",
            'duration_seconds': self.duration,
            'errors_count': len(self.errors),
            'errors': self.errors[:5]  # Only first 5 errors
        }
```

**Usage Example**:
```python
stats = IngestionStats()

# During pipeline execution
stats.urls_discovered = 150
stats.pages_crawled = 148
stats.pages_failed = 2
stats.chunks_created = 3420
stats.embeddings_generated = 3420
stats.vectors_stored = 3420
stats.errors.append({'url': 'https://example.com/404', 'error': 'HTTP 404'})

# At pipeline completion
stats.finalize()

print(f"Pipeline completed in {stats.duration:.1f}s with {stats.success_rate:.1f}% success rate")
# Output: Pipeline completed in 847.3s with 98.7% success rate
```

---

## State Transitions

### DocumentPage Lifecycle

```
[URL Discovered] → [HTML Fetched] → [Content Extracted] → [Ready for Chunking]
```

**States**:
1. **URL Discovered**: URL found in sitemap or recursive crawl
2. **HTML Fetched**: `raw_html` populated from HTTP request
3. **Content Extracted**: `extracted_text` populated after HTML parsing
4. **Ready for Chunking**: DocumentPage complete, can be chunked

**Error Transitions**:
- HTTP error (404, 500) → Skip page, log error in `IngestionStats.errors`
- Timeout → Retry up to 3 times, then skip
- Parse error → Log warning, attempt chunking with raw text

---

### TextChunk Lifecycle

```
[DocumentPage Text] → [Split by Headings/Paragraphs] → [Token Count Validated] → [Ready for Embedding]
```

**States**:
1. **DocumentPage Text**: Input from `DocumentPage.extracted_text`
2. **Split by Headings/Paragraphs**: Apply semantic chunking algorithm
3. **Token Count Validated**: Measure tokens with tiktoken, ensure within `chunk_size` to `max_chunk_size` range
4. **Ready for Embedding**: TextChunk complete with metadata

**Validation Rules**:
- Minimum tokens: 50 (discard chunks smaller than 50 tokens - likely noise)
- Maximum tokens: `max_chunk_size` (1024 default)
- Target tokens: `chunk_size` (512 default)

---

### VectorEmbedding Lifecycle

```
[TextChunk] → [Batched for API] → [Embedded via Cohere] → [Dimensions Validated] → [Stored in Qdrant]
```

**States**:
1. **TextChunk**: Input from chunking stage
2. **Batched for API**: Grouped with up to 96 other chunks
3. **Embedded via Cohere**: API call to `co.embed()`
4. **Dimensions Validated**: Verify 1024 dimensions (FR-014)
5. **Stored in Qdrant**: Upserted to collection

**Error Handling**:
- Rate limit (429) → Exponential backoff, retry up to 5 times
- Service error (500, 503) → Exponential backoff, retry up to 5 times
- Dimension mismatch → Raise error, halt pipeline (indicates model change)
- Qdrant quota exceeded → Fail gracefully with clear error message (FR-019)

---

## Data Flow Diagram

```
┌─────────────┐
│ Config      │ (Loaded from .env at startup)
│ (pydantic)  │
└─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: URL Discovery                                    │
│ fetch_sitemap(base_url) → List[str]                      │
│ OR recursive_crawl(base_url) → List[str]                 │
└─────────────────────────────────────────────────────────┘
       │
       ▼ (List of URLs)
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Crawling & Extraction                            │
│ crawl_pages(urls) → List[DocumentPage]                   │
│ - fetch_page(url) → raw_html                             │
│ - extract_content(raw_html) → extracted_text             │
│ - extract_metadata(raw_html) → title, breadcrumb         │
└─────────────────────────────────────────────────────────┘
       │
       ▼ (List of DocumentPage)
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Chunking                                         │
│ chunk_all_pages(pages) → List[TextChunk]                 │
│ - chunk_text(page.extracted_text, metadata)              │
│ - count_tokens(chunk_text) → token_count                 │
│ - create_chunk_id(url, index)                            │
└─────────────────────────────────────────────────────────┘
       │
       ▼ (List of TextChunk)
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Embedding Generation                             │
│ generate_embeddings(chunks) → List[VectorEmbedding]      │
│ - Batch chunks (96 per request)                          │
│ - co.embed(batch) → embeddings                           │
│ - Retry on rate limits (exponential backoff)             │
│ - Validate dimensions (1024)                             │
└─────────────────────────────────────────────────────────┘
       │
       ▼ (List of VectorEmbedding)
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Vector Storage                                   │
│ store_in_qdrant(embeddings, collection_name)             │
│ - Create collection if not exists                        │
│ - Convert to PointStruct format                          │
│ - Upsert in batches of 100                               │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Search Validation                                │
│ validate_search(test_queries, collection_name)           │
│ - Embed queries with input_type='search_query'           │
│ - Search Qdrant collection                               │
│ - Verify similarity scores > 0.7                         │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ IngestionStats (logged throughout)                       │
└─────────────────────────────────────────────────────────┘
```

---

## Validation Rules

### Config Validation (Startup)
- ✅ All required fields present (FR-021)
- ✅ URLs start with http:// or https://
- ✅ `chunk_size` between 256 and 1024
- ✅ `max_chunk_size` ≥ `chunk_size`
- ✅ `batch_size` between 1 and 96

### DocumentPage Validation (After Crawling)
- ✅ URL is unique (no duplicates)
- ✅ `extracted_text` is not empty
- ⚠ `title` may be empty (log warning, use URL as fallback)

### TextChunk Validation (After Chunking)
- ✅ `token_count` ≥ 50 (discard smaller chunks)
- ✅ `token_count` ≤ `max_chunk_size`
- ✅ `chunk_id` is unique (URL + index combination)
- ✅ `heading_hierarchy` is not empty (at least page title)

### VectorEmbedding Validation (Before Storage)
- ✅ Vector has exactly 1024 dimensions (FR-014)
- ✅ All vector elements are floats
- ✅ `chunk_id` matches source TextChunk

---

**Data Model Complete**: All entities defined with validation rules and state transitions. Ready for implementation.
