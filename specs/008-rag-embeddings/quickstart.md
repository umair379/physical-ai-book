# Quickstart Guide: RAG Embeddings Ingestion Pipeline

**Feature**: 008-rag-embeddings
**Date**: 2025-12-27
**Audience**: Developers setting up RAG ingestion for Docusaurus documentation

## Overview

This guide walks you through setting up and running the RAG embeddings ingestion pipeline to crawl Docusaurus documentation, generate vector embeddings, and store them in Qdrant for semantic search.

**Estimated Time**: 15-30 minutes (setup) + 10-30 minutes (ingestion runtime)

---

## Prerequisites

### 1. Python Environment
- **Python 3.9+** (Python 3.11 recommended)
- **uv package manager** installed

```bash
# Check Python version
python --version  # Should be >= 3.9

# Install uv (if not already installed)
pip install uv

# Verify uv installation
uv --version
```

### 2. API Keys & Accounts

#### Cohere API Key
1. Sign up at https://cohere.com/
2. Navigate to API Keys section
3. Create new API key (free tier available)
4. Copy the API key (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

#### Qdrant Cloud Account
1. Sign up at https://qdrant.tech/cloud/
2. Create a new cluster (Free Tier: 1GB storage, 1M vectors)
3. Note the cluster URL (format: `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.qdrant.io`)
4. Create API key from cluster dashboard
5. Copy API key

### 3. Deployed Docusaurus Site
- Vercel deployment URL (e.g., `https://your-site.vercel.app`)
- Site must be publicly accessible (no authentication)
- Verify sitemap exists: `https://your-site.vercel.app/sitemap.xml`

---

## Setup (5 minutes)

### Step 1: Navigate to Backend Directory

```bash
cd backend/
```

### Step 2: Initialize Project with uv

```bash
# Initialize uv project (creates pyproject.toml and .venv)
uv init

# Install dependencies
uv add requests beautifulsoup4 lxml tiktoken cohere qdrant-client python-dotenv pydantic
```

**Expected Output**:
```
Resolved 15 packages in 1.2s
Installed 15 packages in 850ms
 + beautifulsoup4==4.12.2
 + cohere==4.37.0
 + lxml==5.0.1
 + pydantic==2.5.3
 + python-dotenv==1.0.0
 + qdrant-client==1.7.0
 + requests==2.31.0
 + tiktoken==0.5.2
 + ...
```

### Step 3: Configure Environment Variables

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   # Open in your text editor
   notepad .env  # Windows
   # OR
   nano .env     # Linux/Mac
   ```

3. **Fill in required values**:
   ```bash
   # Vercel Deployment
   BASE_URL=https://your-docusaurus-site.vercel.app

   # Cohere API
   COHERE_API_KEY=your_cohere_api_key_here

   # Qdrant Cloud
   QDRANT_URL=https://your-cluster-id.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key_here

   # Optional Configuration (defaults shown)
   COLLECTION_NAME=docusaurus_docs
   CHUNK_SIZE=512
   MAX_CHUNK_SIZE=1024
   BATCH_SIZE=96
   MAX_CRAWL_DEPTH=3
   ```

4. **Save and close the file**

### Step 4: Verify Configuration

```bash
# Quick config test (will be built into main.py)
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('BASE_URL:', os.getenv('BASE_URL')); print('Config loaded successfully!')"
```

**Expected Output**:
```
BASE_URL: https://your-docusaurus-site.vercel.app
Config loaded successfully!
```

---

## Running the Ingestion Pipeline (10-30 minutes)

### Single Command Execution

```bash
uv run main.py
```

**What Happens**:
1. **Config Validation** (5 seconds)
   - Loads environment variables
   - Validates required fields
   - Connects to Qdrant (tests connection)

2. **URL Discovery** (5-30 seconds)
   - Fetches `sitemap.xml` from `BASE_URL`
   - Parses URLs (typically 100-200 pages for documentation sites)
   - Falls back to recursive crawl if sitemap not found

3. **Crawling & Extraction** (1-5 minutes)
   - Fetches each page via HTTP
   - Extracts main content (removes nav/sidebar/footer)
   - Logs progress every 10 pages

4. **Chunking** (30-60 seconds)
   - Splits text into semantic chunks (default: 512 tokens)
   - Preserves heading hierarchy
   - Counts tokens with tiktoken

5. **Embedding Generation** (5-20 minutes)
   - Batches chunks (96 per request)
   - Calls Cohere API for embeddings
   - Retries on rate limits (exponential backoff)
   - **Note**: Duration depends on API key tier (trial: ~5 req/min, production free: ~100 req/min)

6. **Vector Storage** (30-60 seconds)
   - Creates Qdrant collection if not exists
   - Upserts vectors in batches of 100
   - Validates storage success

7. **Search Validation** (10-30 seconds)
   - Runs test queries
   - Verifies similarity scores > 0.7
   - Displays top results

---

## Example Output

```
2025-12-27 10:15:32 - INFO - === RAG Ingestion Pipeline Started ===
2025-12-27 10:15:32 - INFO - Configuration loaded successfully
2025-12-27 10:15:32 - INFO - Base URL: https://your-site.vercel.app
2025-12-27 10:15:32 - INFO - Collection: docusaurus_docs
2025-12-27 10:15:32 - INFO - Chunk size: 512 tokens

2025-12-27 10:15:37 - INFO - Discovered 147 URLs from sitemap

2025-12-27 10:15:45 - INFO - Progress: 10/147 pages crawled
2025-12-27 10:15:53 - INFO - Progress: 20/147 pages crawled
...
2025-12-27 10:17:12 - INFO - Crawled 145/147 pages successfully
2025-12-27 10:17:12 - WARNING - Failed to crawl https://your-site.vercel.app/404: HTTP 404
2025-12-27 10:17:12 - WARNING - Failed to crawl https://your-site.vercel.app/old-page: HTTP 404

2025-12-27 10:17:15 - INFO - Created 3,427 chunks

2025-12-27 10:17:20 - INFO - Embedding batch 1/36 (96 chunks)
2025-12-27 10:17:28 - INFO - Embedding batch 2/36 (96 chunks)
...
2025-12-27 10:22:45 - INFO - Generated 3,427 embeddings

2025-12-27 10:22:48 - INFO - Created Qdrant collection 'docusaurus_docs'
2025-12-27 10:23:15 - INFO - Stored 3,427 vectors in Qdrant

2025-12-27 10:23:18 - INFO - Query: 'How to install?' - Top result score: 0.847
2025-12-27 10:23:21 - INFO - Query: 'Configuration guide' - Top result score: 0.823
2025-12-27 10:23:24 - INFO - Query: 'API reference' - Top result score: 0.891

2025-12-27 10:23:24 - INFO - === Pipeline Complete ===
2025-12-27 10:23:24 - INFO - Stats: {
  'urls_discovered': 147,
  'pages_crawled': 145,
  'pages_failed': 2,
  'chunks_created': 3427,
  'embeddings_generated': 3427,
  'vectors_stored': 3427,
  'success_rate': '98.6%',
  'duration_seconds': 472.1,
  'errors_count': 2
}
```

---

## Verification

### Test Semantic Search

After ingestion completes, you can test search queries:

```python
# Add to bottom of main.py or create test_search.py
from qdrant_client import QdrantClient
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

def search(query: str, limit: int = 5):
    # Embed query
    response = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query'
    )

    # Search Qdrant
    results = client.search(
        collection_name=os.getenv('COLLECTION_NAME', 'docusaurus_docs'),
        query_vector=response.embeddings[0],
        limit=limit
    )

    print(f"\nQuery: '{query}'\n")
    for i, hit in enumerate(results, 1):
        print(f"{i}. Score: {hit.score:.3f}")
        print(f"   URL: {hit.payload['url']}")
        print(f"   Text: {hit.payload['text'][:150]}...\n")

# Test queries
search("How do I get started?")
search("What are the system requirements?")
search("How to configure authentication?")
```

**Expected Results**:
- Similarity scores > 0.7 for relevant chunks
- Top result matches query intent
- Chunks from appropriate documentation sections

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cohere'"

**Cause**: Dependencies not installed or wrong Python environment

**Fix**:
```bash
# Ensure you're in backend/ directory
cd backend/

# Reinstall dependencies with uv
uv sync

# Run with uv (ensures correct venv)
uv run main.py
```

---

### Issue: "pydantic.ValidationError: base_url field required"

**Cause**: `.env` file missing or not loaded

**Fix**:
```bash
# Check .env file exists
ls .env  # Should exist

# Verify BASE_URL is set
cat .env | grep BASE_URL  # Should show: BASE_URL=https://...

# If .env missing, copy from template
cp .env.example .env

# Edit .env with your values
nano .env
```

---

### Issue: "cohere.errors.RateLimitError: You have exceeded your rate limit"

**Cause**: Cohere API rate limit reached (trial tier: ~5 req/min)

**Fix**: Pipeline already has retry logic with exponential backoff. Just wait - it will automatically retry.

**To speed up** (if you have production API key):
```bash
# Update .env with production API key
COHERE_API_KEY=your_production_key_here

# Increase batch size (optional)
BATCH_SIZE=96  # Already at max
```

**Alternative**: Reduce batch size to slow down requests:
```bash
# In .env
BATCH_SIZE=50  # Slower but fewer rate limits
```

---

### Issue: "qdrant_client.exceptions.UnexpectedResponse: Quota limit exceeded"

**Cause**: Qdrant Free Tier limit reached (1M vectors max)

**Fix**:
```bash
# Check collection size
uv run python -c "from qdrant_client import QdrantClient; import os; from dotenv import load_dotenv; load_dotenv(); client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY')); print(client.get_collection(os.getenv('COLLECTION_NAME', 'docusaurus_docs')))"

# If approaching 1M vectors, delete old collection:
uv run python -c "from qdrant_client import QdrantClient; import os; from dotenv import load_dotenv; load_dotenv(); client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY')); client.delete_collection(os.getenv('COLLECTION_NAME', 'docusaurus_docs')); print('Collection deleted')"

# Re-run ingestion
uv run main.py
```

---

### Issue: "requests.exceptions.HTTPError: 404 Client Error"

**Cause**: URL not found (page was deleted or moved)

**Effect**: Pipeline logs warning and continues with remaining pages

**Fix**: No action needed - this is expected for some pages. Check final stats for success rate:
```
'success_rate': '98.6%'  # 2 out of 147 pages failed - acceptable
```

If success rate < 90%, check `BASE_URL` is correct:
```bash
# In .env
BASE_URL=https://correct-site.vercel.app  # No trailing slash
```

---

### Issue: Pipeline hangs during embedding generation

**Cause**: Network issue or Cohere API unresponsive

**Fix**:
```bash
# Check Cohere API status: https://status.cohere.com/

# If API is down, wait and retry

# If network issue, check internet connection:
curl https://api.cohere.ai/  # Should return response
```

**Logs to check**:
```
2025-12-27 10:17:20 - INFO - Embedding batch 1/36 (96 chunks)
[HANGS HERE]
```

If hanging for >5 minutes, Ctrl+C and retry. Exponential backoff may be waiting for rate limit to clear.

---

### Issue: Low similarity scores (< 0.5) for test queries

**Cause**: Chunking produced poor semantic boundaries, or query doesn't match content

**Fix**:

1. **Check chunk size**:
   ```bash
   # In .env, reduce chunk size for more granular chunks
   CHUNK_SIZE=384  # Smaller chunks, more specific
   ```

2. **Re-run ingestion**:
   ```bash
   uv run main.py
   ```

3. **Try different queries**:
   ```python
   # Match actual documentation topics
   search("installation steps")  # Instead of generic "how to install"
   search("API authentication")  # Instead of "auth"
   ```

---

## Re-Ingestion (Updating Content)

If documentation content changes, re-run the pipeline to update embeddings:

```bash
uv run main.py
```

**What Happens**:
- Existing Qdrant collection is **upserted** (not deleted)
- Chunks with same `chunk_id` (URL + index) are replaced
- New pages create new chunks
- Deleted pages remain in collection (manual cleanup required)

**To start fresh** (delete old embeddings):
```bash
# Delete collection first
uv run python -c "from qdrant_client import QdrantClient; import os; from dotenv import load_dotenv; load_dotenv(); client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY')); client.delete_collection(os.getenv('COLLECTION_NAME', 'docusaurus_docs')); print('Collection deleted')"

# Then re-run ingestion
uv run main.py
```

---

## Advanced Configuration

### Adjust Chunk Size for Better Retrieval

**Problem**: Chunks too large (losing specificity) or too small (losing context)

**Solution**:
```bash
# In .env
CHUNK_SIZE=384   # Smaller chunks (more specific, better for precise queries)
# OR
CHUNK_SIZE=768   # Larger chunks (more context, better for broad queries)
```

**Guideline**:
- Technical docs (code examples, API reference): 256-512 tokens
- Conceptual docs (guides, tutorials): 512-1024 tokens

---

### Reduce Crawl Depth for Faster Testing

**Problem**: Full site crawl takes too long during testing

**Solution**:
```bash
# In .env
MAX_CRAWL_DEPTH=1  # Only crawl homepage and direct links (faster for testing)
```

**Restore for production**:
```bash
MAX_CRAWL_DEPTH=3  # Default (captures most documentation sites)
```

---

### Save Intermediate Outputs for Debugging

**Add to `main.py`** (optional):
```python
import json

# After crawling
with open('output/pages.json', 'w') as f:
    json.dump([page.to_dict() for page in pages], f, indent=2)

# After chunking
with open('output/chunks.json', 'w') as f:
    json.dump([chunk.to_dict() for chunk in chunks], f, indent=2)

print("Intermediate outputs saved to output/")
```

**Create output directory**:
```bash
mkdir output/
```

---

## Performance Benchmarks

### Typical Ingestion Times

| Documentation Size | URLs | Chunks | Cohere Tier | Time |
|--------------------|------|--------|-------------|------|
| Small (20-50 pages) | 30 | 500 | Trial | 3-5 min |
| Medium (50-150 pages) | 100 | 2,000 | Trial | 10-20 min |
| Large (150-300 pages) | 200 | 5,000 | Trial | 25-45 min |
| Medium (50-150 pages) | 100 | 2,000 | Production Free | 2-4 min |
| Large (150-300 pages) | 200 | 5,000 | Production Free | 5-10 min |

**Bottleneck**: Cohere API rate limits (trial: 5 req/min, production free: ~100 req/min)

---

## Next Steps

After successful ingestion:

1. **Integrate with RAG Application**: Use `qdrant_client` to query the collection in your chatbot/search application

2. **Monitor Collection Size**: Check Qdrant dashboard for vector count, storage usage

3. **Set Up Periodic Re-Ingestion**: Schedule cron job or GitHub Action to re-run pipeline when documentation updates

4. **Optimize Chunking**: Experiment with `CHUNK_SIZE` values for your specific use case

5. **Add Retrieval Layer**: Build ranking logic, multi-query retrieval, or hybrid search

---

## Summary

You've successfully:
- ✅ Set up Python environment with uv
- ✅ Configured API keys for Cohere and Qdrant
- ✅ Run RAG ingestion pipeline end-to-end
- ✅ Verified semantic search quality
- ✅ Learned troubleshooting and re-ingestion procedures

**Documentation**: See `research.md` for technical decisions and `data-model.md` for data structures.

**Questions?** Check troubleshooting section above or review ingestion logs for specific errors.
