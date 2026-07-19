#!/usr/bin/env python3
"""
RAG Embeddings Ingestion Pipeline

Single-file implementation for crawling Docusaurus documentation,
chunking text, generating embeddings with Cohere, and storing in Qdrant.
"""

import os
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from functools import wraps

import requests
from bs4 import BeautifulSoup
import tiktoken
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# ============================================================================
# Configuration and Data Structures (Phase 2)
# ============================================================================

class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Required - map to actual env var names
    base_url: str = Field(..., description="Vercel deployment URL", alias='DEPLOY_VERCEL_URL')
    cohere_api_key: str = Field(..., description="Cohere API key")
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")

    # Optional
    collection_name: str = Field("docusaurus_docs", description="Qdrant collection name")
    chunk_size: int = Field(512, description="Target chunk size in tokens")
    max_chunk_size: int = Field(1024, description="Maximum chunk size in tokens")
    batch_size: int = Field(96, description="Embedding batch size")
    max_crawl_depth: int = Field(3, description="Max depth for recursive crawl")

    @field_validator('base_url', 'qdrant_url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v.rstrip('/')

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if not (256 <= v <= 1024):
            raise ValueError('chunk_size must be between 256 and 1024')
        return v

    @field_validator('max_chunk_size')
    @classmethod
    def validate_max_chunk_size(cls, v, info: ValidationInfo):
        if info.data.get('chunk_size') and v < info.data['chunk_size']:
            raise ValueError('max_chunk_size must be >= chunk_size')
        return v

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
        'populate_by_name': True,
        'extra': 'ignore'  # Ignore extra env vars (e.g., OPENAI_API_KEY used by agent.py)
    }


@dataclass
class DocumentPage:
    """Represents a single Docusaurus page from the deployed site."""
    url: str
    title: str
    raw_html: str
    extracted_text: str
    breadcrumb: List[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None


@dataclass
class TextChunk:
    """Represents a semantic chunk of text ready for embedding."""
    chunk_id: str
    source_url: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    heading_hierarchy: List[str]
    token_count: int
    metadata: Dict

    @staticmethod
    def create_id(url: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        return f"{url}#{index}"


@dataclass
class VectorEmbedding:
    """Represents a vector embedding with metadata for storage in Qdrant."""
    chunk_id: str
    vector: List[float]
    chunk_text: str
    metadata: Dict
    timestamp: datetime

    def to_qdrant_point(self) -> PointStruct:
        """Convert to Qdrant PointStruct format."""
        return PointStruct(
            id=hash(self.chunk_id) & ((1 << 63) - 1),  # Ensure positive 64-bit int
            vector=self.vector,
            payload={
                'chunk_id': self.chunk_id,
                'text': self.chunk_text,
                'url': self.metadata.get('url', ''),
                'title': self.metadata.get('title', ''),
                'heading': ' > '.join(self.metadata.get('heading_hierarchy', [])),
                'chunk_index': self.metadata.get('chunk_index', 0),
                'timestamp': self.timestamp.isoformat()
            }
        )


@dataclass
class IngestionStats:
    """Track pipeline execution statistics."""
    urls_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    errors: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def finalize(self):
        """Mark pipeline complete."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Pipeline duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Percentage of pages successfully crawled."""
        total = self.pages_crawled + self.pages_failed
        return (self.pages_crawled / total * 100) if total > 0 else 0.0


# ============================================================================
# Logging Configuration (Phase 2)
# ============================================================================

def setup_logging():
    """Configure logging with file and console output."""
    log_filename = f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ============================================================================
# Configuration Validation and Startup (Phase 2)
# ============================================================================

def validate_config() -> Config:
    """Load and validate configuration at startup."""
    load_dotenv()

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


# Initialize logger
logger = setup_logging()


# ============================================================================
# Phase 3: Crawling & Extraction (User Story 1)
# ============================================================================

def fetch_sitemap(base_url: str) -> List[str]:
    """Discover URLs from sitemap.xml."""
    sitemap_url = f"{base_url}/sitemap.xml"

    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        urls = []

        # Handle sitemap namespace
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for url_elem in root.findall('.//ns:loc', namespaces):
            urls.append(url_elem.text)

        logger.info(f"Discovered {len(urls)} URLs from sitemap")
        return urls

    except Exception as e:
        logger.warning(f"Sitemap fetch failed: {e}. Falling back to recursive crawl...")
        return []


def recursive_crawl(start_url: str, max_depth: int = 3) -> List[str]:
    """Fallback: recursively follow links from start_url."""
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
            logger.warning(f"Failed to crawl {url}: {e}")
            continue

    logger.info(f"Recursive crawl discovered {len(visited)} URLs")
    return list(visited)


def fetch_page(url: str) -> str:
    """Fetch page with timeout and error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {url}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} fetching {url}")
        raise
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        raise


def extract_content(html: str) -> str:
    """Extract main content using BeautifulSoup."""
    soup = BeautifulSoup(html, 'lxml')

    # Try to find main content (Docusaurus typically uses <article> or <main>)
    article = soup.find('article') or soup.find('main')

    if not article:
        # Fallback: use body but remove common nav/sidebar/footer elements
        article = soup.find('body')
        if article:
            for tag in article.find_all(['nav', 'aside', 'footer', 'header']):
                tag.decompose()

    if article:
        # Remove navigation, sidebar, footer elements
        for tag in article.find_all(['nav', 'aside', 'footer']):
            tag.decompose()

        return article.get_text(separator='\n', strip=True)

    return ""


def extract_metadata(html: str) -> Dict[str, any]:
    """Extract title and breadcrumb from HTML."""
    soup = BeautifulSoup(html, 'lxml')

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Extract breadcrumb (common patterns for Docusaurus)
    breadcrumb = []
    breadcrumb_nav = soup.find('nav', class_='breadcrumbs') or soup.find('nav', {'aria-label': 'breadcrumb'})

    if breadcrumb_nav:
        for link in breadcrumb_nav.find_all('a'):
            breadcrumb.append(link.get_text(strip=True))

    return {
        'title': title,
        'breadcrumb': breadcrumb
    }


def crawl_pages(urls: List[str], stats: IngestionStats) -> List[DocumentPage]:
    """Orchestrate fetch + extract for all URLs."""
    pages = []

    for i, url in enumerate(urls):
        try:
            # Fetch page
            html = fetch_page(url)

            # Extract content and metadata
            text = extract_content(html)
            metadata = extract_metadata(html)

            # Create DocumentPage
            pages.append(DocumentPage(
                url=url,
                title=metadata['title'],
                raw_html=html,
                extracted_text=text,
                breadcrumb=metadata['breadcrumb']
            ))

            stats.pages_crawled += 1

            # Progress logging every 10 pages
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(urls)} pages crawled")

        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            stats.pages_failed += 1
            stats.errors.append({'url': url, 'error': str(e)})
            continue

    logger.info(f"Crawled {stats.pages_crawled}/{len(urls)} pages successfully")
    return pages


# ============================================================================
# Phase 4: Chunking (User Story 2)
# ============================================================================

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4, similar to Cohere
    return len(encoding.encode(text))


def detect_headings(lines: List[str]) -> List[tuple]:
    """Extract markdown headings with their line numbers."""
    headings = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            # Count heading level
            level = len(stripped.split()[0]) if stripped.split() else 0
            heading_text = stripped.lstrip('#').strip()

            if heading_text:
                headings.append((i, level, heading_text))

    return headings


def chunk_text(text: str, metadata: Dict, target_size: int = 512, max_size: int = 1024) -> List[TextChunk]:
    """Split text into semantic chunks with heading preservation."""
    chunks = []
    lines = text.split('\n')

    # Detect headings
    headings = detect_headings(lines)
    heading_hierarchy = []

    current_chunk = ""
    current_line_idx = 0

    for i, line in enumerate(lines):
        # Update heading hierarchy
        for heading_idx, (h_line, h_level, h_text) in enumerate(headings):
            if i == h_line:
                # Update hierarchy (keep only headings at or above this level)
                heading_hierarchy = heading_hierarchy[:h_level-1] + [h_text]

        # Add line to current chunk
        current_chunk += line + '\n'
        token_count = count_tokens(current_chunk)

        # Check if we should finalize this chunk
        should_finalize = False

        # Finalize if we exceed target size and we're at a heading boundary
        if token_count >= target_size:
            # Check if next line is a heading (good split point)
            if i + 1 < len(lines):
                next_line_is_heading = any(h[0] == i + 1 for h in headings)
                if next_line_is_heading or token_count >= max_size:
                    should_finalize = True
            else:
                should_finalize = True  # End of document

        if should_finalize and current_chunk.strip():
            # Validate chunk size (min 50 tokens)
            if token_count >= 50:
                chunk_id = TextChunk.create_id(metadata.get('url', ''), len(chunks))
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    source_url=metadata.get('url', ''),
                    chunk_text=current_chunk.strip(),
                    chunk_index=len(chunks),
                    total_chunks=0,  # Will update after all chunks created
                    heading_hierarchy=heading_hierarchy.copy(),
                    token_count=token_count,
                    metadata=metadata
                ))

            current_chunk = ""

    # Finalize last chunk
    if current_chunk.strip():
        token_count = count_tokens(current_chunk)
        if token_count >= 50:
            chunk_id = TextChunk.create_id(metadata.get('url', ''), len(chunks))
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                source_url=metadata.get('url', ''),
                chunk_text=current_chunk.strip(),
                chunk_index=len(chunks),
                total_chunks=0,
                heading_hierarchy=heading_hierarchy.copy(),
                token_count=token_count,
                metadata=metadata
            ))

    # Update total_chunks for all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    return chunks


def chunk_all_pages(pages: List[DocumentPage]) -> List[TextChunk]:
    """Process all DocumentPages into TextChunks."""
    all_chunks = []

    for page in pages:
        metadata = {
            'url': page.url,
            'title': page.title,
            'breadcrumb': page.breadcrumb
        }

        page_chunks = chunk_text(page.extracted_text, metadata)
        all_chunks.extend(page_chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


# ============================================================================
# Phase 5: Embedding & Storage (User Story 3)
# ============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying API calls with exponential backoff."""
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
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)

            return None
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=2.0)
def generate_embeddings(chunks: List[TextChunk], batch_size: int = 96, cohere_api_key: str = "") -> List[VectorEmbedding]:
    """Generate embeddings via Cohere API with batching and retry logic."""
    if not cohere_api_key:
        raise ValueError("Cohere API key is required")

    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)

    embeddings = []
    total_chunks = len(chunks)

    logger.info(f"Generating embeddings for {total_chunks} chunks (batch_size={batch_size})")

    # Process in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [chunk.chunk_text for chunk in batch]

        logger.info(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch)} chunks)")

        # Call Cohere API
        response = co.embed(
            texts=batch_texts,
            model='embed-english-v3.0',
            input_type='search_document',
            embedding_types=['float']
        )

        # Validate dimension (should be 1024 for embed-english-v3.0)
        if response.embeddings.float:
            first_embedding = response.embeddings.float[0]
            if len(first_embedding) != 1024:
                raise ValueError(f"Expected embedding dimension 1024, got {len(first_embedding)}")

        # Create VectorEmbedding objects
        for chunk, vector in zip(batch, response.embeddings.float):
            embeddings.append(VectorEmbedding(
                chunk_id=chunk.chunk_id,
                vector=vector,
                chunk_text=chunk.chunk_text,
                metadata={
                    'url': chunk.source_url,
                    'title': chunk.metadata.get('title', ''),
                    'heading_hierarchy': chunk.heading_hierarchy,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'token_count': chunk.token_count
                },
                timestamp=datetime.now()
            ))

        # Progress logging
        logger.info(f"Generated {len(embeddings)}/{total_chunks} embeddings")

    logger.info(f"Successfully generated {len(embeddings)} embeddings")
    return embeddings


def create_qdrant_collection(collection_name: str, dimension: int = 1024, qdrant_url: str = "", qdrant_api_key: str = ""):
    """Force recreate Qdrant collection from .env config, ignoring existing state."""
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Qdrant URL and API key are required")

    # Initialize Qdrant client with increased timeout
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)

    # Always delete existing collection if present (ignore existing state)
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)

    if collection_exists:
        logger.info(f"Deleting existing collection '{collection_name}' (force recreate)...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection '{collection_name}'")

    # Create fresh collection with cosine distance
    logger.info(f"Creating collection '{collection_name}' with dimension {dimension}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

    # Verify collection was created successfully
    collections_after = client.get_collections().collections
    collection_created = any(c.name == collection_name for c in collections_after)

    if not collection_created:
        error_msg = f"FATAL: Collection '{collection_name}' does not exist after creation attempt"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Successfully created collection '{collection_name}' with dimension {dimension}")
    return client


def store_in_qdrant(embeddings: List[VectorEmbedding], collection_name: str, qdrant_url: str = "", qdrant_api_key: str = "") -> int:
    """Store embeddings in Qdrant with upsert logic."""
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Qdrant URL and API key are required")

    # Initialize Qdrant client with increased timeout
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)

    # Convert to PointStruct objects
    points = [embedding.to_qdrant_point() for embedding in embeddings]

    logger.info(f"Upserting {len(points)} vectors to collection '{collection_name}'")

    # Upsert in batches of 100 (Qdrant best practice)
    batch_size = 100
    total_stored = 0

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        total_stored += len(batch)

        if (i + batch_size) % 500 == 0:
            logger.info(f"Upserted {total_stored}/{len(points)} vectors")

    logger.info(f"Successfully stored {total_stored} vectors in Qdrant")
    return total_stored


def validate_search(test_queries: List[str], collection_name: str, qdrant_url: str = "", qdrant_api_key: str = "", cohere_api_key: str = "") -> Dict:
    """Validate semantic search by running test queries."""
    if not qdrant_url or not qdrant_api_key or not cohere_api_key:
        raise ValueError("Qdrant and Cohere credentials are required")

    # Initialize clients with increased timeout
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
    co = cohere.Client(cohere_api_key)

    results = {}

    logger.info(f"Running {len(test_queries)} validation queries")

    for query in test_queries:
        # Generate query embedding
        response = co.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query',
            embedding_types=['float']
        )

        query_vector = response.embeddings.float[0]

        # Search Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        ).points

        # Extract top results with scores
        top_results = [
            {
                'text': result.payload.get('text', '')[:100] + '...',
                'url': result.payload.get('url', ''),
                'score': result.score
            }
            for result in search_results
        ]

        results[query] = {
            'top_results': top_results,
            'best_score': search_results[0].score if search_results else 0.0
        }

        logger.info(f"Query: '{query}' | Best score: {results[query]['best_score']:.3f}")

    # Calculate average best score
    avg_score = sum(r['best_score'] for r in results.values()) / len(results) if results else 0.0
    logger.info(f"Average best score: {avg_score:.3f}")

    return results


# ============================================================================
# Phase 6: Integration & Orchestration
# ============================================================================

def main():
    """Main orchestration function for the RAG ingestion pipeline."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RAG Embeddings Ingestion Pipeline for Docusaurus documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py                              # Use .env configuration
  python main.py --base-url https://...      # Override base URL

Environment variables (via .env):
  BASE_URL, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY
  COLLECTION_NAME, CHUNK_SIZE, MAX_CHUNK_SIZE, BATCH_SIZE
        """
    )

    parser.add_argument('--base-url', help='Override BASE_URL from .env')
    parser.add_argument('--collection', help='Override COLLECTION_NAME from .env')
    parser.add_argument('--test-queries', nargs='+', help='Test queries for search validation')

    args = parser.parse_args()

    # Initialize
    logger.info("=" * 70)
    logger.info("RAG Embeddings Ingestion Pipeline")
    logger.info("=" * 70)

    # Load configuration
    config = validate_config()

    # Override from CLI arguments if provided
    if args.base_url:
        config.base_url = args.base_url
        logger.info(f"Base URL overridden from CLI: {config.base_url}")

    if args.collection:
        config.collection_name = args.collection
        logger.info(f"Collection name overridden from CLI: {config.collection_name}")

    # Initialize stats
    stats = IngestionStats()

    try:
        # Phase 3: Crawl & Extract
        logger.info("-" * 70)
        logger.info("Phase 3: Crawling & Extraction")
        logger.info("-" * 70)

        # Discover URLs
        urls = fetch_sitemap(config.base_url)

        if not urls:
            logger.warning("Sitemap empty or unavailable, using recursive crawl...")
            urls = recursive_crawl(config.base_url, max_depth=config.max_crawl_depth)

        stats.urls_discovered = len(urls)
        logger.info(f"Discovered {stats.urls_discovered} URLs")

        # Crawl pages
        pages = crawl_pages(urls, stats)

        if not pages:
            logger.error("No pages were successfully crawled. Exiting.")
            return 1

        logger.info(f"Crawled {stats.pages_crawled} pages (success rate: {stats.success_rate:.1f}%)")

        # Phase 4: Chunking
        logger.info("-" * 70)
        logger.info("Phase 4: Text Chunking")
        logger.info("-" * 70)

        chunks = chunk_all_pages(pages)
        stats.chunks_created = len(chunks)

        if not chunks:
            logger.error("No chunks were created. Exiting.")
            return 1

        # Calculate chunk statistics
        token_counts = [chunk.token_count for chunk in chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

        logger.info(f"Created {stats.chunks_created} chunks")
        logger.info(f"  Average tokens: {avg_tokens:.0f} (target: {config.chunk_size})")
        logger.info(f"  Token range: {min_tokens} - {max_tokens}")

        # Phase 5: Embedding & Storage
        logger.info("-" * 70)
        logger.info("Phase 5: Embedding & Storage")
        logger.info("-" * 70)

        # Create Qdrant collection
        create_qdrant_collection(
            collection_name=config.collection_name,
            dimension=1024,
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key
        )

        # Generate embeddings
        embeddings = generate_embeddings(
            chunks=chunks,
            batch_size=config.batch_size,
            cohere_api_key=config.cohere_api_key
        )
        stats.embeddings_generated = len(embeddings)

        logger.info(f"Generated {stats.embeddings_generated} embeddings")

        # Store in Qdrant
        vectors_stored = store_in_qdrant(
            embeddings=embeddings,
            collection_name=config.collection_name,
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key
        )
        stats.vectors_stored = vectors_stored

        logger.info(f"Stored {stats.vectors_stored} vectors in Qdrant")

        # Phase 7: Validation
        logger.info("-" * 70)
        logger.info("Phase 7: Search Validation")
        logger.info("-" * 70)

        # Default test queries if not provided
        test_queries = args.test_queries or [
            "What is a transformer?",
            "How do I train a neural network?",
            "Explain reinforcement learning"
        ]

        search_results = validate_search(
            test_queries=test_queries,
            collection_name=config.collection_name,
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key,
            cohere_api_key=config.cohere_api_key
        )

        # Calculate validation metrics
        avg_score = sum(r['best_score'] for r in search_results.values()) / len(search_results)
        min_score = min(r['best_score'] for r in search_results.values())

        logger.info(f"Search validation complete")
        logger.info(f"  Average best score: {avg_score:.3f}")
        logger.info(f"  Minimum best score: {min_score:.3f}")

        if min_score >= 0.7:
            logger.info(f"  All queries meet >0.7 similarity threshold")
        else:
            logger.warning(f"  WARNING: Some queries below 0.7 similarity threshold")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        stats.finalize()
        return 1

    # Finalize and report
    stats.finalize()

    logger.info("=" * 70)
    logger.info("Pipeline Complete - Final Statistics")
    logger.info("=" * 70)
    logger.info(f"URLs discovered:        {stats.urls_discovered}")
    logger.info(f"Pages crawled:          {stats.pages_crawled}")
    logger.info(f"Pages failed:           {stats.pages_failed}")
    logger.info(f"Success rate:           {stats.success_rate:.1f}%")
    logger.info(f"Chunks created:         {stats.chunks_created}")
    logger.info(f"Embeddings generated:   {stats.embeddings_generated}")
    logger.info(f"Vectors stored:         {stats.vectors_stored}")
    logger.info(f"Duration:               {stats.duration:.1f}s ({stats.duration / 60:.1f} minutes)")
    logger.info(f"Collection:             {config.collection_name}")
    logger.info("=" * 70)

    if stats.errors:
        logger.warning(f"Errors encountered: {len(stats.errors)}")
        logger.warning("See log file for details")

    logger.info("RAG ingestion pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
