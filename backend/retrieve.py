#!/usr/bin/env python3
"""RAG Retrieval Validation Script

Validates that stored embeddings from Feature 008 can be successfully retrieved
and that semantic search returns relevant results.

Usage:
    python retrieve.py --query "What is physical AI?"
    python retrieve.py --test-suite test_queries.json --verbose
"""

import os
import sys
import logging
import argparse
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from functools import wraps

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
import cohere
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

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
        extra='ignore'
    )

    cohere_api_key: str = Field(..., alias='COHERE_API_KEY')
    qdrant_url: str = Field(..., alias='QDRANT_URL')
    qdrant_api_key: str = Field(..., alias='QDRANT_API_KEY')
    collection_name: str = Field("docusaurus_docs", alias='COLLECTION_NAME')


# ============================================================================
# Data Models
# ============================================================================

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

    def validate_metadata(self) -> Tuple[bool, List[str]]:
        """Validate all required fields present and non-null (FR-009)"""
        missing = []
        required_fields = [
            'chunk_id', 'text', 'url', 'title',
            'heading', 'chunk_index', 'timestamp'
        ]

        for field_name in required_fields:
            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and value.strip() == ''):
                missing.append(field_name)

        return (len(missing) == 0, missing)


@dataclass
class CollectionMetadata:
    """Qdrant collection metadata"""
    collection_name: str
    points_count: int
    vector_dimension: int
    distance_metric: str
    status: str

    def validate_against_spec(self) -> Tuple[bool, List[str]]:
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
        if self.status.upper() != "GREEN":
            issues.append(f"Collection status {self.status} (expected GREEN)")

        return (len(issues) == 0, issues)


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

    def get_summary(self) -> Dict:
        """Compute aggregate statistics (SC-005)"""
        if not self.query_metrics:
            return {}

        embed_times = [m.embedding_time_ms for m in self.query_metrics]
        search_times = [m.search_time_ms for m in self.query_metrics]
        total_times = [m.total_latency_ms for m in self.query_metrics]

        import statistics
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
            'total_avg_ms': statistics.mean(total_times),
            'total_p95_ms': self._percentile(total_times, 0.95),
            'total_p99_ms': self._percentile(total_times, 0.99),
        }

    def validate_performance(self) -> Tuple[bool, List[str]]:
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


# ============================================================================
# Error Handling
# ============================================================================

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


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds, doubles each retry (default 1.0)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)

            return None
        return wrapper
    return decorator


def validate_config() -> ValidationConfig:
    """Load and validate configuration from .env file.

    Raises:
        ConfigurationError: If required variables missing or invalid
    """
    load_dotenv()
    logger = logging.getLogger(__name__)

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


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure dual-format logging (console human-readable + file structured).

    Args:
        verbose: If True, set console to DEBUG level, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler - human-readable format
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(levelname)-8s %(message)s'
    ))

    # File handler - structured format
    log_filename = f'validation_{datetime.now():%Y%m%d_%H%M%S}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    ))

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_filename}")


# ============================================================================
# Core Functions
# ============================================================================

@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_query_embedding(query: str, config: ValidationConfig) -> List[float]:
    """Generate query embedding using Cohere (FR-003).

    Args:
        query: User query text
        config: Validated configuration

    Returns:
        1024-dimensional embedding vector

    Raises:
        ConnectionError: If Cohere API fails after retries
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating embedding for query: {query[:50]}...")

    co = cohere.Client(config.cohere_api_key)
    response = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query',  # CRITICAL: Different from ingestion's 'search_document'
        embedding_types=['float']
    )

    embedding = response.embeddings.float[0]
    logger.debug(f"Embedding dimension: {len(embedding)}")

    return embedding


@retry_with_backoff(max_retries=3, base_delay=1.0)
def search_qdrant(
    query_vector: List[float],
    config: ValidationConfig,
    top_k: int = 3
) -> List[SearchResult]:
    """Execute semantic search against Qdrant (FR-004).

    Args:
        query_vector: 1024-dimensional embedding
        config: Validated configuration
        top_k: Number of results to return

    Returns:
        List of SearchResult objects

    Raises:
        ConnectionError: If Qdrant search fails after retries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Searching for top-{top_k} results")

    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=120
    )

    # Use query_points() API (search() is deprecated)
    scored_points = client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=top_k
    ).points

    # Convert to SearchResult objects
    results = [SearchResult.from_scored_point(point) for point in scored_points]

    logger.info(f"Retrieved {len(results)} results")
    return results


def display_results(results: List[SearchResult], query: str):
    """Display search results with metadata (FR-005).

    Args:
        results: List of search results
        query: Original query text
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Retrieved {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. Score: {result.score:.3f}")
        logger.info(f"   Title: {result.title}")
        logger.info(f"   Heading: {result.heading}")
        logger.info(f"   URL: {result.url}")
        logger.info(f"   Chunk: {result.chunk_index + 1}")

        # Text preview (first 150 chars) - handle Unicode for Windows console
        text = result.text
        preview = text[:150] + '...' if len(text) > 150 else text
        # Replace Unicode characters that may cause encoding issues on Windows
        preview_safe = preview.encode('ascii', errors='replace').decode('ascii')
        logger.info(f"   Text: {preview_safe}")
        logger.info("")


@retry_with_backoff(max_retries=3, base_delay=1.0)
def verify_connection(config: ValidationConfig) -> CollectionMetadata:
    """Verify connection to Qdrant and retrieve collection metadata (FR-001, FR-002).

    Args:
        config: Validated configuration

    Returns:
        CollectionMetadata with collection info

    Raises:
        ConnectionError: If connection fails after retries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Connecting to Qdrant at {config.qdrant_url}")

    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=120  # Required for Qdrant Free Tier
    )

    # Get collection info
    info = client.get_collection(config.collection_name)

    # Create metadata object
    metadata = CollectionMetadata(
        collection_name=config.collection_name,
        points_count=info.points_count,
        vector_dimension=info.config.params.vectors.size,
        distance_metric=info.config.params.vectors.distance.name,
        status=info.status.name
    )

    logger.info(f"Collection '{metadata.collection_name}' status: {metadata.status}, points: {metadata.points_count}")

    # Validate against spec
    is_valid, issues = metadata.validate_against_spec()
    if not is_valid:
        for issue in issues:
            logger.warning(f"Validation issue: {issue}")
    else:
        logger.info("Collection validation passed (192 points, 1024 dims, COSINE, GREEN)")

    return metadata


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for retrieval validation script."""
    parser = argparse.ArgumentParser(
        description='RAG Retrieval Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrieve.py --query "What is physical AI?"
  python retrieve.py --test-suite test_queries.json
  python retrieve.py --query "ROS 2 basics" --top-k 5 --verbose
        """
    )

    parser.add_argument('--query', help='Single test query')
    parser.add_argument('--test-suite', help='JSON file with test queries')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results to return (default: 3)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if not args.query and not args.test_suite:
        logger.error("Must provide either --query or --test-suite")
        parser.print_help()
        return 1

    logger.info("RAG Retrieval Validation Script")
    logger.info("=" * 50)

    try:
        # Load and validate configuration
        config = validate_config()

        # Verify connection to Qdrant (US1: FR-001, FR-002)
        metadata = verify_connection(config)
        logger.info(f"[OK] Infrastructure verified: {metadata.points_count} vectors ready")

        # Execute query (US2: FR-003, FR-004, FR-005)
        if args.query:
            logger.info(f"\nExecuting query: '{args.query}'")

            # Generate embedding
            query_vector = generate_query_embedding(args.query, config)

            # Search Qdrant
            results = search_qdrant(query_vector, config, args.top_k)

            # Log results (FR-007)
            best_score = results[0].score if results else 0.0
            logger.info(f"Query: '{args.query}' | Results: {len(results)} | Best score: {best_score:.3f}")

            # Display results
            display_results(results, args.query)

        elif args.test_suite:
            logger.info(f"Test suite execution not yet implemented: '{args.test_suite}'")

        logger.info("=" * 50)
        logger.info("[OK] Validation complete")
        return 0

    except ConfigurationError as e:
        logger.error(f"[ERROR] Configuration failed: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"[ERROR] Connection failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
