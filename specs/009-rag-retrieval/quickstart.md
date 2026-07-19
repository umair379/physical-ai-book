# Quickstart: RAG Retrieval Validation

**Feature**: 009-rag-retrieval
**Created**: 2025-12-28
**Purpose**: Usage guide for retrieve.py validation script

## Overview

This guide shows how to use `retrieve.py` to validate the RAG retrieval pipeline. The script connects to Qdrant, runs test queries, and validates that semantic search returns relevant results.

**Prerequisites**:
- Feature 008 completed (192 vectors stored in Qdrant)
- Valid .env credentials (Cohere API key, Qdrant URL/API key)
- Python 3.11+ with dependencies installed

---

## Installation

### 1. Navigate to Backend Folder
```bash
cd backend
```

### 2. Verify .env Configuration
Ensure `.env` contains credentials from Feature 008:
```bash
# .env file
COHERE_API_KEY=BUq6Z6ewir2YTV7ghQAReujSx7lc8VKh8zsx46iP
QDRANT_URL=https://dbe06b27-f4e3-4c82-a911-09160423ee6c.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
COLLECTION_NAME=docusaurus_docs
```

### 3. Install Dependencies (if not already installed)
```bash
uv sync
```

Dependencies used by retrieve.py:
- `qdrant-client` - Qdrant vector database client
- `cohere` - Cohere embedding API
- `pydantic-settings` - Configuration management
- `python-dotenv` - .env file loading

---

## Basic Usage

### Single Query Validation

Run a single test query to verify retrieval works:

```bash
python retrieve.py --query "What is physical AI?"
```

**Expected Output**:
```
INFO     Logging to validation_20251228_123456.log
INFO     Configuration loaded successfully
INFO     Connecting to Qdrant at https://dbe06b27-f4e3-4c82-a911-09160423ee6c...
INFO     Collection 'docusaurus_docs' status: GREEN, points: 192

Query: 'What is physical AI?'
Retrieved 3 results:

1. Score: 0.583
   Title: Welcome to the Physical AI Blog
   Heading: Blog > Welcome
   URL: https://physical-ai-book-lake-three.vercel.app/blog/2025/12/26/welcome
   Chunk: 0/1
   Text: Welcome to the Physical AI Blog. This blog will cover the latest developments in physical AI, robotics, and embodied intelligence...

2. Score: 0.577
   Title: Introduction
   Heading: Getting Started > Introduction
   URL: https://physical-ai-book-lake-three.vercel.app/docs/intro
   Chunk: 0/3
   Text: Welcome to the Physical AI Book. Build intelligent robotic systems that perceive, reason, and act in the physical world...

3. Score: 0.432
   Title: Module 1: The Robotic Nervous System (ROS 2)
   Heading: Modules > Module 1
   URL: https://physical-ai-book-lake-three.vercel.app/docs/module-1/
   Chunk: 0/5
   Text: Module 1: The Robotic Nervous System (ROS 2). Learn the fundamentals of ROS 2, the Robot Operating System...

INFO     Query latency: 630ms (embed: 487ms, search: 143ms)
INFO     ✅ All 3 results have complete metadata
```

### Test Suite Validation

Run predefined test queries from JSON file:

```bash
python retrieve.py --test-suite test_queries.json
```

**test_queries.json** (create this file in backend folder):
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
        "query": "Compare transformers and RNNs for sequence modeling",
        "expected_min_score": 0.7
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
      }
    ]
  }
}
```

**Expected Output**:
```
INFO     Running test suite with 5 queries

=== Category: common ===
Description: General queries expected to return highly relevant results (>0.8 similarity)

Query: 'What is physical AI?'
  ✅ Score: 0.583

Query: 'How do I set up ROS 2?'
  ✅ Score: 0.812

Query: 'Explain computer vision basics'
  ✅ Score: 0.791

=== Category: edge_cases ===
Description: Complex queries with multiple concepts (>0.7 similarity)

Query: 'Compare transformers and RNNs for sequence modeling'
  ✅ Score: 0.723

=== Category: adversarial ===
Description: Off-topic or gibberish queries (should score <0.5)

Query: 'How to cook pasta?'
  ✅ Score: 0.187

=== Test Suite Summary ===
Total queries: 5
Passed: 5 (100.0%)
Failed: 0

=== Performance Summary ===
Queries: 5
Embedding p95: 523ms
Search p95: 178ms
Total avg: 712ms
Total p95: 731ms

✅ All performance thresholds met (avg <3000ms)
```

---

## Command-Line Options

### --query
Execute a single test query.

```bash
python retrieve.py --query "What is ROS 2?"
```

### --test-suite
Load test queries from JSON file.

```bash
python retrieve.py --test-suite test_queries.json
```

### --top-k
Number of results to return per query (default: 3).

```bash
python retrieve.py --query "What is physical AI?" --top-k 5
```

**Output**: Returns top-5 most similar chunks instead of top-3.

### --verbose
Enable debug logging for detailed output.

```bash
python retrieve.py --query "What is physical AI?" --verbose
```

**Additional Debug Output**:
```
DEBUG    Collection: docusaurus_docs
DEBUG    Qdrant URL: https://dbe06b27-f4e3-4c82-a911-09160423ee6c...
DEBUG    Generating embedding for query: What is physical AI?...
DEBUG    Embedding dimension: 1024
DEBUG    Latency: 630ms (embed: 487ms, search: 143ms)
DEBUG    URL: https://physical-ai-book-lake-three.vercel.app/blog/2025/12/26/welcome
DEBUG    Text: Welcome to the Physical AI Blog. This blog will cover...
```

---

## Common Validation Scenarios

### Scenario 1: Verify Infrastructure (SC-001)

**Goal**: Confirm Qdrant connection and 192 stored vectors.

```bash
python retrieve.py --query "test"
```

**Check Output For**:
```
INFO     Collection 'docusaurus_docs' status: GREEN, points: 192
```

**Success Criteria**:
- Connection succeeds (no errors)
- Status: GREEN
- Points count: 192

### Scenario 2: Test Query Relevance (SC-002, SC-003)

**Goal**: Verify queries return relevant results with >0.4 similarity.

```bash
python retrieve.py --test-suite test_queries.json
```

**Check Output For**:
```
Total queries: 5
Passed: 5 (100.0%)
```

**Success Criteria** (SC-003):
- 100% of queries return at least 1 result with score >0.4
- Manual inspection: top results match query intent (SC-002)

### Scenario 3: Validate Metadata Completeness (SC-004)

**Goal**: Confirm all results have required fields.

```bash
python retrieve.py --query "What is physical AI?"
```

**Check Output For**:
```
INFO     ✅ All 3 results have complete metadata
```

**Success Criteria** (SC-004):
- All results have non-null: chunk_id, text, url, title, heading, chunk_index, timestamp
- No "missing fields" errors logged

### Scenario 4: Performance Baseline (SC-005)

**Goal**: Verify average query latency <3 seconds.

```bash
python retrieve.py --test-suite test_queries.json
```

**Check Output For**:
```
=== Performance Summary ===
Total avg: 712ms

✅ All performance thresholds met (avg <3000ms)
```

**Success Criteria** (SC-005):
- Average latency <3000ms for batch of 10 queries
- No performance warnings in logs

### Scenario 5: Error Handling (SC-006)

**Goal**: Verify graceful handling of errors.

**Test 1: Missing credentials**
```bash
# Temporarily rename .env
mv .env .env.backup
python retrieve.py --query "test"
```

**Expected Output**:
```
ERROR    Configuration validation failed: Field required [type=missing, input_value=...]
ERROR    Required .env variables: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

**Test 2: Invalid Qdrant URL**
```bash
# Edit .env with fake URL
QDRANT_URL=https://invalid-url.fake
python retrieve.py --query "test"
```

**Expected Output** (after 3 retries):
```
ERROR    verify_connection failed after 3 attempts: [Connection error details]
```

**Success Criteria** (SC-006):
- Script fails gracefully (no crashes)
- Clear error messages logged
- Exit code 1 (failure status)

### Scenario 6: Topic Coverage (SC-007)

**Goal**: Verify module-specific queries return correct sections.

**test_queries_coverage.json**:
```json
{
  "topic_coverage": {
    "description": "Verify all major documentation sections are indexed",
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

```bash
python retrieve.py --test-suite test_queries_coverage.json
```

**Expected Output**:
```
Query: 'ROS 2 basics'
  ✅ Score: 0.812
  ✅ URL contains 'module-1'

Query: 'Computer vision techniques'
  ✅ Score: 0.791
  ✅ URL contains 'module-2'

Query: 'Neural network fundamentals'
  ✅ Score: 0.763
  ✅ URL contains 'module-3'
```

**Success Criteria** (SC-007):
- 80% of module-specific queries return results from correct documentation section
- Verified by URL path matching expected pattern

---

## Interpreting Results

### Similarity Scores

**Score Range**: 0.0 (no similarity) to 1.0 (identical)

**Interpretation**:
- **>0.8**: Highly relevant (query matches document topic closely)
- **0.6-0.8**: Relevant (query and document share concepts)
- **0.4-0.6**: Somewhat relevant (may contain useful information)
- **<0.4**: Not relevant (likely off-topic)

**Spec Threshold** (SC-003): >0.4 (minimum acceptable similarity)

**Best Practice**: Common queries should achieve >0.7 for good user experience.

### Latency Breakdown

**Example Output**:
```
Query latency: 630ms (embed: 487ms, search: 143ms)
```

**Components**:
- **embed**: Cohere API call to generate 1024-dim embedding
  - Baseline: 200-500ms (depends on API latency)
  - Warning: >1000ms (check network or Cohere status)
- **search**: Qdrant vector search (cosine similarity)
  - Baseline: 50-200ms (for 192 vectors)
  - Warning: >500ms (check Qdrant connection)
- **total**: embed + search
  - Threshold: <3000ms average (SC-005)

**Performance Tips**:
- Run from same region as Qdrant instance (US East 4)
- Use stable network connection
- Batch queries to amortize overhead

### Metadata Fields

**Required Fields** (SC-004):
```
chunk_id: "https://physical-ai-book.vercel.app/docs/intro#0"
text: "Welcome to the Physical AI Book..."
url: "https://physical-ai-book-lake-three.vercel.app/docs/intro"
title: "Introduction"
heading: "Getting Started > Introduction"
chunk_index: 0
timestamp: "2025-12-28T08:27:00Z"
```

**Validation**:
- Script automatically checks all fields present and non-null
- ✅ = All fields valid
- ❌ = Missing or null fields (logs specific field names)

---

## Troubleshooting

### Error: "Configuration validation failed"

**Cause**: Missing or invalid .env credentials.

**Fix**:
1. Verify `.env` file exists in `backend/` folder
2. Check required variables present:
   ```bash
   grep -E "COHERE_API_KEY|QDRANT_URL|QDRANT_API_KEY|COLLECTION_NAME" .env
   ```
3. Verify no extra quotes or spaces:
   ```bash
   # Correct
   COHERE_API_KEY=BUq6Z6ewir2YTV7ghQAReujSx7lc8VKh8zsx46iP

   # Incorrect (extra quotes)
   COHERE_API_KEY="BUq6Z6ewir2YTV7ghQAReujSx7lc8VKh8zsx46iP"
   ```

### Error: "The read operation timed out"

**Cause**: Qdrant client timeout too short (default 60s insufficient for Free Tier).

**Fix**:
- Verify `retrieve.py` uses `timeout=120` in QdrantClient initialization:
  ```python
  client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120)
  ```
- This fix was applied in Feature 008 and should be reused

### Warning: "Score X.XXX below threshold"

**Cause**: Query not relevant to indexed documentation.

**Expected Behavior**:
- Adversarial queries (off-topic) should score low (<0.5)
- Common queries scoring <0.4 indicates problem

**Debug Steps**:
1. Check query text for typos
2. Verify Feature 008 ingested correct documentation
3. Try broader query:
   ```bash
   # Too specific (may not match)
   python retrieve.py --query "How to configure ROS 2 parameters for Gazebo simulation?"

   # Broader (more likely to match)
   python retrieve.py --query "ROS 2 basics"
   ```

### Error: "Expected 192 points, found X"

**Cause**: Qdrant collection not fully ingested.

**Fix**:
1. Re-run Feature 008 ingestion pipeline:
   ```bash
   cd backend
   uv run main.py
   ```
2. Verify ingestion completed successfully:
   ```bash
   # Should show: Points count: 192
   python -c "from main import validate_config; from qdrant_client import QdrantClient; config = validate_config(); client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120); info = client.get_collection(config.collection_name); print(f'Points: {info.points_count}')"
   ```

### No Results Returned

**Cause**: Query embedding not matching any stored vectors.

**Debug Steps**:
1. Verify collection status GREEN:
   ```bash
   python retrieve.py --query "test" --verbose
   ```
2. Check embedding dimension (should be 1024):
   ```
   DEBUG    Embedding dimension: 1024
   ```
3. Try generic query to verify search works:
   ```bash
   python retrieve.py --query "robot" --top-k 5
   ```
4. If still no results, check Qdrant collection recreated with vectors:
   ```bash
   # Should show >0 points
   python -c "from main import validate_config; from qdrant_client import QdrantClient; config = validate_config(); client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120); info = client.get_collection(config.collection_name); print(f'Status: {info.status}, Points: {info.points_count}')"
   ```

---

## Log Files

Each run creates a timestamped log file:

**File Location**:
```
backend/validation_20251228_123456.log
```

**Content**: Structured logs with timestamps, function names, line numbers.

**Example**:
```
2025-12-28 12:34:56,123 - __main__ - INFO - verify_connection:45 - Connecting to Qdrant...
2025-12-28 12:34:56,234 - __main__ - INFO - verify_connection:52 - Collection 'docusaurus_docs' status: GREEN, points: 192
2025-12-28 12:34:56,345 - __main__ - DEBUG - generate_query_embedding:78 - Generating embedding for query: What is physical AI?...
2025-12-28 12:34:56,832 - __main__ - DEBUG - generate_query_embedding:85 - Embedding dimension: 1024
2025-12-28 12:34:56,987 - __main__ - INFO - search_qdrant:102 - Retrieved 3 results
```

**When to Review Logs**:
- Debugging failures (check ERROR and WARNING lines)
- Performance analysis (search for "latency" lines)
- Verifying API calls (HTTP Request lines)
- Sharing results with team (attach log file)

---

## Next Steps

After successful validation:

1. **Document Results**: Save test suite output for team review
2. **Integrate with CI/CD**: Add validation script to pre-deployment checks
3. **Expand Test Coverage**: Add more queries to test_queries.json
4. **Proceed to Feature 010**: Build chatbot/agent using validated retrieval

**Optional Enhancements**:
- Add performance monitoring (track metrics over time)
- Create automated regression tests
- Integrate with alerting (Slack/email on failures)

---

## Summary

**Key Commands**:
```bash
# Single query
python retrieve.py --query "What is physical AI?"

# Test suite
python retrieve.py --test-suite test_queries.json

# Verbose mode
python retrieve.py --query "test" --verbose

# Custom top-k
python retrieve.py --query "ROS 2" --top-k 5
```

**Success Criteria Checklist**:
- [ ] SC-001: Connection succeeds, 192 points reported
- [ ] SC-002: Manual inspection confirms result relevance (5 samples)
- [ ] SC-003: 100% of test queries return results >0.4 similarity
- [ ] SC-004: All results have complete metadata (no null fields)
- [ ] SC-005: Average latency <3 seconds for 10 queries
- [ ] SC-006: Error handling works (test with invalid credentials)
- [ ] SC-007: 80% of module queries return correct sections

**Files Referenced**:
- `backend/retrieve.py` - Main validation script
- `backend/test_queries.json` - Test suite definition
- `backend/.env` - Configuration (credentials)
- `backend/validation_*.log` - Execution logs
