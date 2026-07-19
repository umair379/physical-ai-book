# Data Model: RAG Retrieval Validation

**Feature**: 009-rag-retrieval
**Created**: 2025-12-28
**Purpose**: Document data structures used in validation script

## Overview

This document defines the 4 key entities used in retrieve.py for validating RAG retrieval pipeline. All entities are implemented as dataclasses (not Pydantic models) for simplicity.

---

## Entity 1: Query

**Purpose**: Represents a user's text input for semantic search

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| query_text | str | Yes | Original query text from user/test suite |
| embedding_vector | List[float] | Yes | 1024-dimensional embedding from Cohere embed-english-v3.0 |
| category | str | No | Test category (common/edge_cases/adversarial/topic_coverage) |
| expected_min_score | float | No | Minimum similarity threshold for validation (e.g., 0.8) |
| expected_url_pattern | str | No | Expected URL substring for topic coverage validation |

**Python Implementation**:
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Query:
    """User query for semantic search"""
    query_text: str
    embedding_vector: List[float]
    category: Optional[str] = None
    expected_min_score: Optional[float] = None
    expected_url_pattern: Optional[str] = None

    def __post_init__(self):
        """Validate embedding dimension"""
        if len(self.embedding_vector) != 1024:
            raise ValueError(f"Expected 1024-dim embedding, got {len(self.embedding_vector)}")
```

**Example**:
```python
query = Query(
    query_text="What is physical AI?",
    embedding_vector=[0.123, -0.456, ...],  # 1024 floats
    category="common",
    expected_min_score=0.8,
    expected_url_pattern="intro"
)
```

**Lifecycle**:
1. Created from test_queries.json or CLI argument
2. Embedding generated via Cohere API (input_type='search_query')
3. Passed to search_qdrant() for vector search
4. Used in validation to check results against expected thresholds

**Validation Rules** (FR-008):
- query_text must be non-empty (edge case: empty string handled)
- embedding_vector must be exactly 1024 dimensions
- expected_min_score range: 0.0-1.0 (cosine similarity)

---

## Entity 2: SearchResult

**Purpose**: Single matching chunk returned from Qdrant semantic search

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| score | float | Yes | Cosine similarity score (0.0-1.0) |
| chunk_id | str | Yes | Unique chunk identifier (format: "url#chunk_index") |
| text | str | Yes | Chunk text content (may be truncated preview) |
| url | str | Yes | Source documentation page URL |
| title | str | Yes | Page title |
| heading | str | Yes | Heading hierarchy (e.g., "Module 1 > Introduction") |
| chunk_index | int | Yes | Position within document (0-based) |
| total_chunks | int | No | Total chunks in source document |
| timestamp | str | Yes | ISO 8601 ingestion timestamp |

**Python Implementation**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    """Single search result from Qdrant"""
    score: float
    chunk_id: str
    text: str
    url: str
    title: str
    heading: str
    chunk_index: int
    timestamp: str
    total_chunks: Optional[int] = None

    @classmethod
    def from_scored_point(cls, point: ScoredPoint) -> 'SearchResult':
        """Create SearchResult from Qdrant ScoredPoint"""
        payload = point.payload
        return cls(
            score=point.score,
            chunk_id=payload['chunk_id'],
            text=payload['text'],
            url=payload['url'],
            title=payload['title'],
            heading=payload['heading'],
            chunk_index=payload['chunk_index'],
            timestamp=payload['timestamp'],
            total_chunks=payload.get('total_chunks')
        )

    def validate_metadata(self) -> tuple[bool, List[str]]:
        """Validate all required fields present and non-null (FR-009)"""
        missing = []
        required_fields = [
            'chunk_id', 'text', 'url', 'title',
            'heading', 'chunk_index', 'timestamp'
        ]

        for field in required_fields:
            value = getattr(self, field)
            if value is None or (isinstance(value, str) and value.strip() == ''):
                missing.append(field)

        return (len(missing) == 0, missing)
```

**Example**:
```python
result = SearchResult(
    score=0.583,
    chunk_id="https://physical-ai-book.vercel.app/docs/intro#0",
    text="Welcome to the Physical AI Book. Build intelligent robotic systems...",
    url="https://physical-ai-book-lake-three.vercel.app/docs/intro",
    title="Introduction",
    heading="Getting Started > Introduction",
    chunk_index=0,
    total_chunks=3,
    timestamp="2025-12-28T08:27:00Z"
)
```

**Lifecycle**:
1. Returned from Qdrant query_points() as ScoredPoint
2. Converted to SearchResult via from_scored_point()
3. Metadata validated with validate_metadata() (SC-004)
4. Displayed to user with display_results() (FR-005)
5. Used in test case validation for score/URL pattern checks

**Validation Rules** (SC-004):
- score: 0.0 ≤ score ≤ 1.0
- chunk_id: Format "url#chunk_index"
- url: Valid HTTPS URL (from Vercel deployment)
- chunk_index: Non-negative integer
- timestamp: ISO 8601 format (from Feature 008 ingestion)

---

## Entity 3: CollectionMetadata

**Purpose**: Information about Qdrant vector collection

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| collection_name | str | Yes | Name of Qdrant collection (e.g., "docusaurus_docs") |
| points_count | int | Yes | Total vectors stored (expected: 192) |
| vector_dimension | int | Yes | Embedding dimension (1024 for Cohere embed-english-v3.0) |
| distance_metric | str | Yes | Distance function (Cosine for semantic search) |
| status | str | Yes | Collection status (GREEN/YELLOW/RED) |

**Python Implementation**:
```python
from dataclasses import dataclass
from qdrant_client.models import CollectionInfo

@dataclass
class CollectionMetadata:
    """Qdrant collection metadata"""
    collection_name: str
    points_count: int
    vector_dimension: int
    distance_metric: str
    status: str

    @classmethod
    def from_collection_info(cls, name: str, info: CollectionInfo) -> 'CollectionMetadata':
        """Create CollectionMetadata from Qdrant CollectionInfo"""
        return cls(
            collection_name=name,
            points_count=info.points_count,
            vector_dimension=info.config.params.vectors.size,
            distance_metric=info.config.params.vectors.distance.name,
            status=info.status.name
        )

    def validate_against_spec(self) -> tuple[bool, List[str]]:
        """Validate collection meets spec requirements (SC-001)"""
        issues = []

        # SC-001: Expected 192 vectors from Feature 008
        if self.points_count != 192:
            issues.append(f"Expected 192 points, found {self.points_count}")

        # Cohere embed-english-v3.0 uses 1024 dimensions
        if self.vector_dimension != 1024:
            issues.append(f"Expected 1024 dimensions, found {self.vector_dimension}")

        # Semantic search requires cosine similarity
        if self.distance_metric.upper() != "COSINE":
            issues.append(f"Expected COSINE distance, found {self.distance_metric}")

        # Collection should be healthy
        if self.status != "GREEN":
            issues.append(f"Collection status {self.status} (expected GREEN)")

        return (len(issues) == 0, issues)
```

**Example**:
```python
metadata = CollectionMetadata(
    collection_name="docusaurus_docs",
    points_count=192,
    vector_dimension=1024,
    distance_metric="COSINE",
    status="GREEN"
)

is_valid, issues = metadata.validate_against_spec()
if not is_valid:
    print(f"Validation failed: {issues}")
```

**Lifecycle**:
1. Retrieved via client.get_collection() in verify_connection() (FR-001)
2. Converted from Qdrant CollectionInfo to CollectionMetadata
3. Validated against spec expectations (SC-001)
4. Displayed to user for verification (FR-002)

**Validation Rules** (SC-001):
- points_count: Must be 192 (from Feature 008 ingestion)
- vector_dimension: Must be 1024 (Cohere model constraint)
- distance_metric: Must be "COSINE" (semantic search requirement)
- status: Must be "GREEN" (healthy collection)

---

## Entity 4: QueryMetrics

**Purpose**: Performance measurements for validation

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| query_text | str | Yes | Original query text for traceability |
| embedding_time_ms | float | Yes | Time to generate embedding (Cohere API call) |
| search_time_ms | float | Yes | Time to search Qdrant (vector search) |
| total_latency_ms | float | Yes | Total query latency (embedding + search) |
| result_count | int | Yes | Number of results returned |
| best_score | float | Yes | Highest similarity score in results |
| timestamp | str | Yes | ISO 8601 timestamp when query executed |

**Python Implementation**:
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import statistics

@dataclass
class QueryMetrics:
    """Performance metrics for a single query"""
    query_text: str
    embedding_time_ms: float
    search_time_ms: float
    total_latency_ms: float
    result_count: int
    best_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics across multiple queries"""
    query_metrics: List[QueryMetrics] = field(default_factory=list)

    def add_query(self, metrics: QueryMetrics):
        """Add metrics for a single query"""
        self.query_metrics.append(metrics)

    def _percentile(self, data: List[float], p: float) -> float:
        """Compute percentile (e.g., p=0.95 for p95)"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_summary(self) -> dict:
        """Compute aggregate statistics (SC-005)"""
        if not self.query_metrics:
            return {}

        embed_times = [m.embedding_time_ms for m in self.query_metrics]
        search_times = [m.search_time_ms for m in self.query_metrics]
        total_times = [m.total_latency_ms for m in self.query_metrics]

        return {
            'queries_count': len(self.query_metrics),
            'embed_avg_ms': statistics.mean(embed_times),
            'embed_p50_ms': self._percentile(embed_times, 0.50),
            'embed_p95_ms': self._percentile(embed_times, 0.95),
            'embed_p99_ms': self._percentile(embed_times, 0.99),
            'search_avg_ms': statistics.mean(search_times),
            'search_p50_ms': self._percentile(search_times, 0.50),
            'search_p95_ms': self._percentile(search_times, 0.95),
            'search_p99_ms': self._percentile(search_times, 0.99),
            'total_avg_ms': statistics.mean(total_times),  # SC-005: <3000ms
            'total_p95_ms': self._percentile(total_times, 0.95),
            'total_p99_ms': self._percentile(total_times, 0.99),
        }

    def validate_performance(self) -> tuple[bool, List[str]]:
        """Validate performance against SC-005 thresholds"""
        summary = self.get_summary()
        issues = []

        # SC-005: Average latency under 3 seconds
        if summary.get('total_avg_ms', 0) > 3000:
            issues.append(f"Average latency {summary['total_avg_ms']:.0f}ms exceeds 3000ms threshold")

        # Best practice: p95 < 5s (2x buffer on average)
        if summary.get('total_p95_ms', 0) > 5000:
            issues.append(f"P95 latency {summary['total_p95_ms']:.0f}ms exceeds 5000ms threshold")

        # Cohere embedding should be fast (<1s p95)
        if summary.get('embed_p95_ms', 0) > 1000:
            issues.append(f"Embedding p95 {summary['embed_p95_ms']:.0f}ms exceeds 1000ms threshold")

        return (len(issues) == 0, issues)
```

**Example**:
```python
# Single query metrics
query_metrics = QueryMetrics(
    query_text="What is physical AI?",
    embedding_time_ms=487.3,
    search_time_ms=142.8,
    total_latency_ms=630.1,
    result_count=3,
    best_score=0.583,
    timestamp="2025-12-28T12:34:56Z"
)

# Aggregate metrics
perf_metrics = PerformanceMetrics()
perf_metrics.add_query(query_metrics)
# ... add more queries ...

summary = perf_metrics.get_summary()
print(f"Average latency: {summary['total_avg_ms']:.0f}ms")
print(f"P95 latency: {summary['total_p95_ms']:.0f}ms")

is_valid, issues = perf_metrics.validate_performance()
if not is_valid:
    print(f"Performance validation failed: {issues}")
```

**Lifecycle**:
1. Created during execute_query_with_metrics() (FR-010)
2. Timing captured with time.perf_counter() before/after each phase
3. Added to PerformanceMetrics aggregate
4. Logged with query results (FR-007)
5. Summarized at end of test suite run
6. Validated against SC-005 thresholds

**Validation Rules** (SC-005):
- total_avg_ms: Must be <3000ms (spec requirement)
- total_p95_ms: Should be <5000ms (best practice, 2x buffer)
- embed_p95_ms: Should be <1000ms (Cohere API baseline)
- search_p95_ms: Should be <500ms (Qdrant baseline for 192 vectors)

---

## Entity Relationships

```
┌─────────────────┐
│     Query       │
│  - query_text   │
│  - embedding    │
└────────┬────────┘
         │
         │ 1:N (top-k results)
         ▼
┌─────────────────┐      ┌──────────────────┐
│  SearchResult   │      │  QueryMetrics    │
│  - score        │      │  - query_text    │
│  - chunk_id     │      │  - embed_time_ms │
│  - text         │      │  - search_time_ms│
│  - url          │      │  - best_score    │
│  - metadata     │      └──────────────────┘
└─────────────────┘               │
         │                        │
         │                        │ N:1 (aggregate)
         │                        ▼
         │              ┌──────────────────────┐
         │              │ PerformanceMetrics   │
         │              │  - query_metrics[]   │
         │              │  - get_summary()     │
         │              └──────────────────────┘
         │
         │ N:1 (stored in)
         ▼
┌──────────────────────┐
│ CollectionMetadata   │
│  - collection_name   │
│  - points_count      │
│  - vector_dimension  │
│  - distance_metric   │
└──────────────────────┘
```

**Flow**:
1. **Query** → generate embedding → search Qdrant → **SearchResult** list
2. **SearchResult** → validate metadata (FR-009) → display to user (FR-005)
3. **QueryMetrics** → record timings → aggregate in **PerformanceMetrics** (FR-010)
4. **CollectionMetadata** → verify infrastructure ready before queries (SC-001)

---

## Implementation Notes

### Why Dataclasses (Not Pydantic)?

**Decision**: Use Python dataclasses for all entities

**Rationale**:
- **Simplicity**: Validation script doesn't need Pydantic's advanced features (validators, serialization)
- **Performance**: Dataclasses have zero runtime overhead vs Pydantic's validation cost
- **Dependency Reduction**: pydantic-settings only needed for Config (env vars), not data entities
- **Clarity**: @dataclass syntax cleaner for simple structs without complex validation

**When to Use Pydantic**:
- Configuration (ValidationConfig): Needs env var loading, type coercion
- Future API endpoints: Would need JSON serialization/validation

### Type Hints

All entities use full type hints for IDE support and static analysis:
```python
from typing import List, Optional, Dict
from dataclasses import dataclass

@dataclass
class Example:
    required_field: str
    optional_field: Optional[int] = None
    list_field: List[float] = field(default_factory=list)
```

### Validation Methods

Each entity includes a validation method following pattern:
```python
def validate_X(self) -> tuple[bool, List[str]]:
    """Validate entity against requirements.

    Returns:
        (is_valid, issues_list)
    """
    issues = []
    # ... validation logic ...
    return (len(issues) == 0, issues)
```

This pattern enables:
- Clear pass/fail status (boolean)
- Detailed error messages (list of strings)
- Consistent validation interface across all entities

---

## Mapping to Spec Requirements

| Entity | Spec Section | Requirements |
|--------|-------------|--------------|
| Query | FR-003 | Generate embeddings with input_type='search_query' |
| SearchResult | FR-005, FR-009 | Display with metadata, validate completeness |
| CollectionMetadata | FR-001, FR-002, SC-001 | Connect to Qdrant, verify 192 points |
| QueryMetrics | FR-007, FR-010, SC-005 | Log metrics, measure latency <3s |

---

## Summary

Four entities provide complete data model for validation script:

1. **Query**: User input + embedding (supports test suite categories)
2. **SearchResult**: Qdrant match + full metadata (enables validation)
3. **CollectionMetadata**: Infrastructure status (pre-flight check)
4. **QueryMetrics**: Performance tracking (compliance with SC-005)

All entities use dataclasses for simplicity, include validation methods, and map directly to spec requirements (FR-001 through FR-010, SC-001 through SC-007).
