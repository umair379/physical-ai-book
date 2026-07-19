# Research: RAG Retrieval Validation Implementation

**Feature**: 009-rag-retrieval
**Created**: 2025-12-28
**Phase**: Phase 0 - Technical Research

## Overview

This document captures research findings for implementing a single-file Python CLI script (`retrieve.py`) to validate the RAG retrieval pipeline. Research focused on battle-tested patterns for validation tools, reusing proven approaches from Feature 008 ingestion pipeline.

---

## 1. Single-File Script Architecture

### Decision
Implement retrieve.py as a single-file CLI script (~200-300 lines) in the backend folder, following functional composition pattern (not OOP).

### Rationale
- **Simplicity**: Validation scripts benefit from linear execution flow without abstraction overhead
- **Maintainability**: Developers can understand entire validation logic in one file
- **Reusability**: Proven pattern from Feature 008's main.py (730 lines, similar complexity)
- **Timeline**: Single file meets 1-2 task constraint from spec

### Alternatives Considered
1. **Multi-file package** (retrieve/, __init__.py, models.py, validators.py)
   - Rejected: Over-engineering for 5-10 test queries
   - Cost: Additional complexity without proportional benefit
2. **OOP class hierarchy** (ValidatorBase, QdrantValidator, CohereValidator)
   - Rejected: Unnecessary abstraction for sequential validation steps
   - Cost: More code to maintain without reusability gains

### Implementation Guidance

**File Structure** (estimated 250 lines):
```python
#!/usr/bin/env python3
"""RAG Retrieval Validation Script

Validates that stored embeddings from Feature 008 can be successfully retrieved
and that semantic search returns relevant results.

Usage:
    python retrieve.py --query "What is physical AI?"
    python retrieve.py --test-suite queries.json --verbose
"""

# Imports (lines 1-20)
import os, sys, logging, argparse, time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
import cohere
from dotenv import load_dotenv

# Configuration (lines 21-40)
class ValidationConfig(BaseSettings):
    """Configuration loaded from .env file"""
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    cohere_api_key: str = Field(..., alias='COHERE_API_KEY')
    qdrant_url: str = Field(..., alias='QDRANT_URL')
    qdrant_api_key: str = Field(..., alias='QDRANT_API_KEY')
    collection_name: str = Field("docusaurus_docs", alias='COLLECTION_NAME')

# Logging Setup (lines 41-70)
def setup_logging(verbose: bool = False):
    """Configure dual-format logging (console human-readable + file structured)"""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # File handler
    file_handler = logging.FileHandler(f'validation_{datetime.now():%Y%m%d_%H%M%S}.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console)
    logger.addHandler(file_handler)

# Core Functions (lines 71-200)
def verify_connection(config: ValidationConfig) -> Dict:
    """FR-001, FR-002: Connect to Qdrant and retrieve collection metadata"""
    pass

def generate_query_embedding(query: str, config: ValidationConfig) -> List[float]:
    """FR-003: Generate query embedding using Cohere"""
    pass

def search_qdrant(query_vector: List[float], client: QdrantClient,
                  config: ValidationConfig, top_k: int = 3) -> List[ScoredPoint]:
    """FR-004: Execute semantic search and return top-k results"""
    pass

def validate_result_metadata(result: ScoredPoint) -> bool:
    """FR-009: Validate required metadata fields present"""
    pass

def display_results(results: List[ScoredPoint], query: str):
    """FR-005: Display search results with metadata"""
    pass

def run_test_suite(test_queries: Dict, config: ValidationConfig) -> Dict:
    """FR-006: Execute batch test queries"""
    pass

# Main Orchestration (lines 201-250)
def main():
    parser = argparse.ArgumentParser(description='RAG Retrieval Validation')
    parser.add_argument('--query', help='Single test query')
    parser.add_argument('--test-suite', help='JSON file with test queries')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config, verify connection, run queries, log metrics
    # Exit with appropriate status code

if __name__ == "__main__":
    sys.exit(main())
```

**Key Design Decisions**:
- **Functional composition**: Functions accept config/clients as parameters, no global state
- **Dataclasses for metrics**: `@dataclass` for PerformanceMetrics, QueryResult (not Pydantic models)
- **Early validation**: verify_connection() runs first, fail-fast if infrastructure broken
- **Explicit resource management**: Create Qdrant/Cohere clients once in main(), pass to functions

---

## 2. Error Handling Strategy

### Decision
Implement fail-fast error handling with retry-with-backoff for API calls. Validation scripts should surface errors immediately (not suppress).

### Rationale
- **Validation vs Production**: Unlike ingestion pipeline (Feature 008) which must complete despite transient errors, validation tools should fail loudly to surface issues
- **Developer Experience**: Clear error messages guide developers to fix root causes (missing credentials, network issues, API quotas)
- **Reuse Proven Pattern**: Feature 008's retry decorator (main.py:498-517) proven stable

### Alternatives Considered
1. **Aggressive retry with circuit breaker** (like ingestion pipeline)
   - Rejected: Validation script runtime <1 minute, no need for complex retry logic
   - Cost: Masks intermittent failures developers should investigate
2. **No retry, fail immediately**
   - Rejected: Too brittle for common transient API errors (503, network blips)
   - Cost: False negatives from temporary API unavailability

### Implementation Guidance

**Reuse from main.py:498-517** (retry decorator):
```python
from functools import wraps
import time

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds, doubles each retry (default 1.0)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. Retry in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator
```

**Apply to API functions**:
```python
@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_query_embedding(query: str, config: ValidationConfig) -> List[float]:
    """FR-003: Generate query embedding with retry on transient failures"""
    co = cohere.Client(config.cohere_api_key)
    response = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query',  # CRITICAL: Different from ingestion!
        embedding_types=['float']
    )
    return response.embeddings.float[0]

@retry_with_backoff(max_retries=3, base_delay=1.0)
def search_qdrant(query_vector: List[float], client: QdrantClient,
                  config: ValidationConfig, top_k: int = 3) -> List[ScoredPoint]:
    """FR-004: Execute semantic search with retry on network errors"""
    return client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=top_k,
        timeout=120  # From Feature 008 fix for Qdrant Free Tier
    ).points
```

**Error Taxonomy** (FR-008):
```python
class ValidationError(Exception):
    """Base exception for validation failures"""
    pass

class ConnectionError(ValidationError):
    """Qdrant or Cohere connection failed after retries"""
    pass

class ConfigurationError(ValidationError):
    """Missing or invalid credentials in .env"""
    pass

class DataQualityError(ValidationError):
    """Retrieved results missing required metadata"""
    pass
```

**Configuration Validation** (reuse pattern from main.py:33-72):
```python
def validate_config() -> ValidationConfig:
    """Validate configuration before running validation"""
    load_dotenv()

    try:
        config = ValidationConfig()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error("Required .env variables: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY")
        raise ConfigurationError(f"Invalid configuration: {e}")

    return config
```

---

## 3. Logging Strategy

### Decision
Implement dual-format logging: console (human-readable) + file (structured). Use Python's standard logging module with two handlers.

### Rationale
- **Developer UX**: Console output clean and scannable during interactive validation
- **Debugging**: File logs preserve full context (timestamps, stack traces, metrics) for troubleshooting
- **Production Readiness**: Structured logs compatible with log aggregation tools (e.g., CloudWatch)
- **Proven Pattern**: Reuses approach from Feature 008 main.py

### Alternatives Considered
1. **Console-only logging**
   - Rejected: Loses context when script completes, hard to share results
2. **JSON-only logging (structlog)**
   - Rejected: Poor interactive UX (hard to read during development)
   - Cost: Additional dependency for minimal benefit in validation script
3. **Database logging** (store results in SQLite)
   - Rejected: Over-engineering for manual validation tool
   - Cost: Adds persistence complexity without clear requirement

### Implementation Guidance

**Setup Function** (lines 41-70):
```python
import logging
from datetime import datetime

def setup_logging(verbose: bool = False):
    """Configure dual-format logging.

    Args:
        verbose: If True, set console to DEBUG level, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler - human-readable format
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(levelname)-8s %(message)s'  # Left-aligned level for clean columns
    ))

    # File handler - structured format
    log_filename = f'validation_{datetime.now():%Y%m%d_%H%M%S}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG in file
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    ))

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_filename}")
```

**Logging Levels by Function** (FR-007):
```python
# verify_connection()
logger.info(f"Connecting to Qdrant at {config.qdrant_url}")
logger.info(f"Collection '{config.collection_name}' status: {status}, points: {count}")

# generate_query_embedding()
logger.debug(f"Generating embedding for query: {query[:50]}...")
logger.debug(f"Embedding dimension: {len(embedding)}")

# search_qdrant()
logger.info(f"Searching for top-{top_k} results")
logger.info(f"Retrieved {len(results)} results")

# display_results()
logger.info(f"\nQuery: {query}")
for i, result in enumerate(results, 1):
    logger.info(f"  {i}. Score: {result.score:.3f} | {result.payload.get('title', 'N/A')}")
    logger.debug(f"     URL: {result.payload.get('url')}")
    logger.debug(f"     Text: {result.payload.get('text', '')[:100]}...")

# run_test_suite()
logger.info(f"Running test suite with {len(queries)} queries")
logger.info(f"Completed {passed}/{total} queries successfully")
```

**Performance Logging** (FR-010):
```python
logger.info(f"Query latency: {total_ms:.0f}ms (embed: {embed_ms:.0f}ms, search: {search_ms:.0f}ms)")
logger.info(f"Performance summary: embed_p95={metrics.embed_p95:.0f}ms, search_p95={metrics.search_p95:.0f}ms")
```

---

## 4. Performance Measurement

### Decision
Use `time.perf_counter()` for high-resolution latency measurement. Track phase-separated metrics (embedding vs search) and compute percentiles (p50, p95, p99) not just averages.

### Rationale
- **Accuracy**: `perf_counter()` provides nanosecond resolution, immune to system clock adjustments (unlike `time.time()`)
- **Insight**: Separate embedding/search phases identifies bottlenecks (is Cohere or Qdrant slow?)
- **Production Relevance**: Percentiles (p95, p99) reveal tail latency issues masked by averages
- **SC-005 Compliance**: Spec requires <3s average latency, but p95 more important for user experience

### Alternatives Considered
1. **time.time()** (wall-clock time)
   - Rejected: Subject to system clock adjustments (NTP sync), lower resolution
2. **Average-only metrics**
   - Rejected: Hides tail latency (1 slow query out of 10 masked by 9 fast queries)
3. **External profiling** (cProfile, py-spy)
   - Rejected: Overkill for network I/O bound validation script

### Implementation Guidance

**Metrics Data Structure**:
```python
from dataclasses import dataclass, field
from typing import List
import statistics

@dataclass
class PerformanceMetrics:
    """Track query performance metrics (FR-010)"""
    embedding_times_ms: List[float] = field(default_factory=list)
    search_times_ms: List[float] = field(default_factory=list)
    total_times_ms: List[float] = field(default_factory=list)

    def add_query(self, embed_ms: float, search_ms: float):
        """Record timings for a single query"""
        self.embedding_times_ms.append(embed_ms)
        self.search_times_ms.append(search_ms)
        self.total_times_ms.append(embed_ms + search_ms)

    def _percentile(self, data: List[float], p: float) -> float:
        """Compute percentile (e.g., p=0.95 for p95)"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_summary(self) -> dict:
        """Compute percentile summary (SC-005)"""
        if not self.total_times_ms:
            return {}

        return {
            'queries_count': len(self.total_times_ms),
            'embed_p50_ms': self._percentile(self.embedding_times_ms, 0.50),
            'embed_p95_ms': self._percentile(self.embedding_times_ms, 0.95),
            'embed_p99_ms': self._percentile(self.embedding_times_ms, 0.99),
            'search_p50_ms': self._percentile(self.search_times_ms, 0.50),
            'search_p95_ms': self._percentile(self.search_times_ms, 0.95),
            'search_p99_ms': self._percentile(self.search_times_ms, 0.99),
            'total_avg_ms': statistics.mean(self.total_times_ms),
            'total_p95_ms': self._percentile(self.total_times_ms, 0.95),
        }
```

**Measurement Pattern**:
```python
import time

def execute_query_with_metrics(query: str, config: ValidationConfig,
                                top_k: int, metrics: PerformanceMetrics) -> List[ScoredPoint]:
    """Execute query and record performance metrics (FR-010)"""

    # Phase 1: Embedding generation
    embed_start = time.perf_counter()
    query_vector = generate_query_embedding(query, config)
    embed_duration_ms = (time.perf_counter() - embed_start) * 1000

    # Phase 2: Vector search
    search_start = time.perf_counter()
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120)
    results = search_qdrant(query_vector, client, config, top_k)
    search_duration_ms = (time.perf_counter() - search_start) * 1000

    # Record metrics
    metrics.add_query(embed_duration_ms, search_duration_ms)

    logger.debug(f"Latency: {embed_duration_ms + search_duration_ms:.0f}ms "
                 f"(embed: {embed_duration_ms:.0f}ms, search: {search_duration_ms:.0f}ms)")

    return results
```

**Performance Thresholds** (from Feature 008 baseline):
```python
PERFORMANCE_THRESHOLDS = {
    'embed_p95_ms': 1000,  # Cohere API typically <500ms, allow 2x buffer
    'search_p95_ms': 500,  # Qdrant search typically <200ms for 192 vectors
    'total_avg_ms': 3000,  # SC-005 requirement
}

def validate_performance(metrics: PerformanceMetrics) -> bool:
    """Check if performance meets SC-005 thresholds"""
    summary = metrics.get_summary()

    passed = True
    if summary['total_avg_ms'] > PERFORMANCE_THRESHOLDS['total_avg_ms']:
        logger.warning(f"Average latency {summary['total_avg_ms']:.0f}ms exceeds {PERFORMANCE_THRESHOLDS['total_avg_ms']}ms threshold")
        passed = False

    if summary['embed_p95_ms'] > PERFORMANCE_THRESHOLDS['embed_p95_ms']:
        logger.warning(f"Embedding p95 {summary['embed_p95_ms']:.0f}ms exceeds threshold")
        passed = False

    return passed
```

---

## 5. Test Query Design

### Decision
Implement stratified test query design with 4 categories: common queries (>0.8 similarity), edge cases (>0.7), adversarial queries (<0.5), and topic coverage. Store queries in JSON file for reusability.

### Rationale
- **SC-003 Validation**: Need multiple queries to verify "100% return at least 1 result >0.4 similarity"
- **SC-007 Validation**: Module-specific queries verify correct documentation section retrieval
- **Regression Detection**: Adversarial queries catch false positives (irrelevant results scoring high)
- **Repeatability**: JSON file enables consistent validation across runs, shareable with team

### Alternatives Considered
1. **Hardcoded queries in script**
   - Rejected: Hard to update, no separation of test data from code
2. **Random query generation**
   - Rejected: Non-deterministic, can't verify specific documentation coverage
3. **Query from user via stdin**
   - Rejected: Not suitable for automated validation, requires manual input each run

### Implementation Guidance

**Test Suite JSON Format** (`backend/test_queries.json`):
```json
{
  "common": {
    "description": "General queries expected to return highly relevant results (>0.8 similarity)",
    "queries": [
      {
        "query": "What is physical AI?",
        "expected_min_score": 0.8,
        "expected_url_pattern": "intro"
      },
      {
        "query": "How do I set up ROS 2?",
        "expected_min_score": 0.8,
        "expected_url_pattern": "module-1"
      },
      {
        "query": "Explain computer vision basics",
        "expected_min_score": 0.8,
        "expected_url_pattern": "module-2"
      }
    ]
  },
  "edge_cases": {
    "description": "Complex queries with multiple concepts (>0.7 similarity)",
    "queries": [
      {
        "query": "Compare transformers and RNNs for sequence modeling in physical AI",
        "expected_min_score": 0.7,
        "expected_url_pattern": null
      },
      {
        "query": "What are deployment options for robotic systems?",
        "expected_min_score": 0.7,
        "expected_url_pattern": null
      }
    ]
  },
  "adversarial": {
    "description": "Off-topic or gibberish queries (should score <0.5)",
    "queries": [
      {
        "query": "How to cook pasta?",
        "expected_max_score": 0.5,
        "reason": "Off-topic (cooking, not robotics)"
      },
      {
        "query": "asdfghjkl random noise query",
        "expected_max_score": 0.5,
        "reason": "Gibberish input"
      }
    ]
  },
  "topic_coverage": {
    "description": "Verify all major documentation sections are indexed (>0.75 similarity)",
    "queries": [
      {
        "query": "ROS 2 basics",
        "expected_min_score": 0.75,
        "expected_url_pattern": "module-1",
        "module": "Module 1"
      },
      {
        "query": "Computer vision techniques",
        "expected_min_score": 0.75,
        "expected_url_pattern": "module-2",
        "module": "Module 2"
      },
      {
        "query": "Neural network fundamentals",
        "expected_min_score": 0.75,
        "expected_url_pattern": "module-3",
        "module": "Module 3"
      }
    ]
  }
}
```

**Query Execution Logic** (FR-006):
```python
import json
from pathlib import Path

def load_test_suite(filepath: str) -> Dict:
    """Load test queries from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def run_test_suite(test_suite: Dict, config: ValidationConfig,
                   top_k: int, metrics: PerformanceMetrics) -> Dict:
    """Execute all test queries and validate results (SC-003, SC-007)"""
    results = {
        'total_queries': 0,
        'passed': 0,
        'failed': 0,
        'failures': []
    }

    for category, category_data in test_suite.items():
        logger.info(f"\n=== Category: {category} ===")
        logger.info(f"Description: {category_data['description']}")

        for test_case in category_data['queries']:
            query_text = test_case['query']
            results['total_queries'] += 1

            # Execute query
            search_results = execute_query_with_metrics(query_text, config, top_k, metrics)

            # Validate results
            passed = validate_test_case(test_case, search_results, category)
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    'category': category,
                    'query': query_text,
                    'actual_score': search_results[0].score if search_results else 0.0
                })

            display_results(search_results, query_text)

    logger.info(f"\n=== Test Suite Summary ===")
    logger.info(f"Total queries: {results['total_queries']}")
    logger.info(f"Passed: {results['passed']} ({results['passed']/results['total_queries']*100:.1f}%)")
    logger.info(f"Failed: {results['failed']}")

    return results

def validate_test_case(test_case: Dict, results: List[ScoredPoint], category: str) -> bool:
    """Validate search results against test case expectations"""
    if not results:
        logger.error(f"  ❌ No results returned")
        return False

    best_score = results[0].score

    # Common/Edge/Coverage: check minimum score
    if 'expected_min_score' in test_case:
        if best_score < test_case['expected_min_score']:
            logger.error(f"  ❌ Score {best_score:.3f} below threshold {test_case['expected_min_score']}")
            return False

    # Adversarial: check maximum score (should be low)
    if 'expected_max_score' in test_case:
        if best_score > test_case['expected_max_score']:
            logger.error(f"  ❌ Score {best_score:.3f} above threshold {test_case['expected_max_score']}")
            return False

    # Topic coverage: check URL pattern (SC-007)
    if test_case.get('expected_url_pattern'):
        url = results[0].payload.get('url', '')
        if test_case['expected_url_pattern'] not in url:
            logger.error(f"  ❌ URL pattern '{test_case['expected_url_pattern']}' not in {url}")
            return False

    logger.info(f"  ✅ Score: {best_score:.3f}")
    return True
```

**Initial Test Queries** (minimal set for SC-003):
```python
# If no JSON file provided, use minimal hardcoded queries
DEFAULT_QUERIES = [
    "What is physical AI?",
    "How do I set up ROS 2?",
    "Explain computer vision basics",
    "What are neural network fundamentals?",
    "Describe robotic system deployment"
]
```

---

## 6. Key Implementation Differences from Feature 008

### Cohere input_type Parameter

**CRITICAL**: Query embedding uses `input_type='search_query'`, different from ingestion's `input_type='search_document'`.

**From Cohere Documentation**:
- `search_document`: Optimized for document embeddings (stored in vector DB)
- `search_query`: Optimized for query embeddings (user questions)

**Feature 008 Ingestion** (main.py:~550):
```python
response = co.embed(
    texts=[chunk['text'] for chunk in batch],
    model='embed-english-v3.0',
    input_type='search_document',  # Embedding documents for storage
    embedding_types=['float']
)
```

**Feature 009 Retrieval** (retrieve.py):
```python
response = co.embed(
    texts=[query],
    model='embed-english-v3.0',
    input_type='search_query',  # Embedding user query for search
    embedding_types=['float']
)
```

**Why This Matters**:
- Cohere's embed-english-v3.0 model applies different transformations based on input_type
- Using wrong type degrades search relevance (wrong semantic space)
- Must match ingestion (search_document) with retrieval (search_query)

### Qdrant Timeout Configuration

**Reuse from Feature 008 fix** (main.py:585, 622, 654):
```python
client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120)
```

**Reason**: Qdrant Free Tier has strict default timeouts. Feature 008 encountered `httpx.ReadTimeout` during upsert, fixed by adding `timeout=120` seconds.

### Qdrant Search API (Updated Method)

**Feature 008 initially used deprecated API**, fixed in main.py:673-677:
```python
# Deprecated (AttributeError in qdrant-client v1.16.2)
search_results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
)

# Current API (use in retrieve.py)
search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=top_k
).points  # Returns ScoredPoint list
```

### Error Handling Philosophy

**Ingestion (Feature 008)**: Aggressive retry, log errors but continue
- Goal: Ingest all 192 chunks even if some fail temporarily
- Strategy: Retry with exponential backoff, skip failed chunks, report at end

**Retrieval (Feature 009)**: Fail-fast, surface errors immediately
- Goal: Validate infrastructure is working correctly
- Strategy: Retry 3 times for transient errors, then fail with clear message

**Code Comparison**:
```python
# Feature 008: Continue on error
try:
    embeddings = generate_embeddings(batch)
except Exception as e:
    logger.error(f"Batch {i} failed: {e}")
    failed_batches.append(i)
    continue  # Process next batch

# Feature 009: Fail on error
try:
    embedding = generate_query_embedding(query, config)
except Exception as e:
    logger.error(f"Query embedding failed: {e}")
    sys.exit(1)  # Stop validation, surface issue
```

---

## 7. Metadata Validation

### Decision
Validate that all retrieved results contain required metadata fields (FR-009, SC-004) with non-null values.

### Required Fields (from Feature 008 ingestion)
```python
REQUIRED_METADATA_FIELDS = [
    'chunk_id',      # Format: "url#chunk_index"
    'text',          # Chunk text (may be truncated in payload)
    'url',           # Source page URL
    'title',         # Page title
    'heading',       # Heading hierarchy (e.g., "Module 1 > Introduction")
    'chunk_index',   # Integer position in document
    'timestamp'      # Ingestion timestamp
]
```

### Implementation Guidance

**Validation Function** (FR-009):
```python
def validate_result_metadata(result: ScoredPoint) -> tuple[bool, List[str]]:
    """Validate that result contains all required metadata fields.

    Returns:
        (is_valid, missing_fields)
    """
    missing_fields = []

    for field in REQUIRED_METADATA_FIELDS:
        if field not in result.payload:
            missing_fields.append(field)
        elif result.payload[field] is None:
            missing_fields.append(f"{field} (null)")

    return (len(missing_fields) == 0, missing_fields)

def validate_all_results(results: List[ScoredPoint]) -> bool:
    """Validate metadata for all search results (SC-004)"""
    all_valid = True

    for i, result in enumerate(results, 1):
        is_valid, missing = validate_result_metadata(result)
        if not is_valid:
            logger.error(f"Result {i} missing fields: {', '.join(missing)}")
            all_valid = False

    if all_valid:
        logger.info(f"✅ All {len(results)} results have complete metadata")

    return all_valid
```

**Display Function with Metadata** (FR-005):
```python
def display_results(results: List[ScoredPoint], query: str):
    """Display search results with metadata (FR-005)"""
    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Retrieved {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        payload = result.payload

        logger.info(f"{i}. Score: {result.score:.3f}")
        logger.info(f"   Title: {payload.get('title', 'N/A')}")
        logger.info(f"   Heading: {payload.get('heading', 'N/A')}")
        logger.info(f"   URL: {payload.get('url', 'N/A')}")
        logger.info(f"   Chunk: {payload.get('chunk_index', '?')}/{payload.get('total_chunks', '?')}")

        # Text preview (first 150 chars)
        text = payload.get('text', '')
        preview = text[:150] + '...' if len(text) > 150 else text
        logger.info(f"   Text: {preview}")
        logger.info("")
```

---

## 8. Configuration Reuse

### Decision
Reuse ValidationConfig class pattern from Feature 008's Config class (main.py:33-72), leveraging pydantic-settings for type-safe .env loading.

### Implementation Guidance

**Config Class** (lines 21-40):
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class ValidationConfig(BaseSettings):
    """Configuration loaded from .env file.

    Required environment variables:
        COHERE_API_KEY: Cohere API key for embedding generation
        QDRANT_URL: Qdrant Cloud instance URL
        QDRANT_API_KEY: Qdrant API key for authentication
        COLLECTION_NAME: Name of Qdrant collection (default: docusaurus_docs)
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra .env variables
    )

    cohere_api_key: str = Field(..., alias='COHERE_API_KEY')
    qdrant_url: str = Field(..., alias='QDRANT_URL')
    qdrant_api_key: str = Field(..., alias='QDRANT_API_KEY')
    collection_name: str = Field("docusaurus_docs", alias='COLLECTION_NAME')

def validate_config() -> ValidationConfig:
    """Load and validate configuration from .env file.

    Raises:
        ConfigurationError: If required variables missing or invalid
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        config = ValidationConfig()
        logger.info("Configuration loaded successfully")
        logger.debug(f"Collection: {config.collection_name}")
        logger.debug(f"Qdrant URL: {config.qdrant_url}")
        return config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error("Required .env variables: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY")
        raise ConfigurationError(f"Invalid configuration: {e}")
```

**Expected .env** (reuse from Feature 008):
```bash
# Cohere API
COHERE_API_KEY=BUq6Z6ewir2YTV7ghQAReujSx7lc8VKh8zsx46iP

# Qdrant Cloud
QDRANT_URL=https://dbe06b27-f4e3-4c82-a911-09160423ee6c.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Collection
COLLECTION_NAME=docusaurus_docs
```

---

## 9. Checklist for retrieve.py Implementation

**Before Writing Code**:
- [ ] Read main.py (Feature 008) to understand proven patterns
- [ ] Review spec.md FR-001 through FR-010 and SC-001 through SC-007
- [ ] Identify reusable code sections (retry decorator, Config class, logging setup)

**During Implementation**:
- [ ] Reuse retry_with_backoff decorator from main.py:498-517
- [ ] Reuse Config class pattern from main.py:33-72 (adapt to ValidationConfig)
- [ ] Use input_type='search_query' for Cohere embeddings (NOT 'search_document')
- [ ] Add timeout=120 to QdrantClient initialization
- [ ] Use client.query_points() (NOT deprecated client.search())
- [ ] Implement dual-format logging (console + file)
- [ ] Use time.perf_counter() for latency measurement
- [ ] Validate metadata fields for all results (FR-009)
- [ ] Create test_queries.json with stratified test cases

**After Implementation**:
- [ ] Test with default .env credentials (Feature 008 infrastructure)
- [ ] Verify SC-001: Connection succeeds, reports 192 points
- [ ] Verify SC-003: All test queries return results >0.4 similarity
- [ ] Verify SC-004: All results have complete metadata (no null fields)
- [ ] Verify SC-005: Average latency <3 seconds for 10 queries
- [ ] Verify SC-006: Error handling works (test with invalid credentials)
- [ ] Verify SC-007: Module-specific queries return correct sections
- [ ] Create PHR documenting implementation (history/prompts/009-rag-retrieval/)

---

## Summary

This research identified battle-tested patterns for implementing retrieve.py as a single-file Python CLI validation script (~250 lines). Key decisions:

1. **Architecture**: Functional composition, reuse patterns from Feature 008 main.py
2. **Error Handling**: Fail-fast with retry-with-backoff for API calls
3. **Logging**: Dual-format (console + file) for developer UX and debugging
4. **Performance**: time.perf_counter() with percentile tracking (p50, p95, p99)
5. **Test Design**: Stratified queries (common/edge/adversarial/coverage) in JSON file
6. **Critical Details**: input_type='search_query', timeout=120, query_points() API

**Estimated Implementation Effort**: 1-2 tasks as specified (1 task for core script, 1 task for test suite + validation)

**Next Phase**: Proceed to Phase 1 - Generate data-model.md (4 key entities) and quickstart.md (usage examples)
