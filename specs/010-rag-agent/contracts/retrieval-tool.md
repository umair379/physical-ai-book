# Retrieval Tool Contract

**Feature**: 010-rag-agent | **Type**: Function Tool | **Date**: 2025-12-28

## Overview

The Retrieval Tool is a function-based tool that integrates with the existing Qdrant retrieval pipeline (Feature 009) to fetch relevant book content chunks. The agent automatically invokes this tool when it needs to answer questions about book content.

## Tool Schema

### Function Signature

```python
def retrieve_book_content(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves relevant book content chunks from Qdrant vector database.

    Args:
        query: User question or search query (required)
        top_k: Number of chunks to retrieve, default 3, range 1-10 (optional)

    Returns:
        List of dictionaries containing:
        - text (str): Chunk content
        - score (float): Similarity score (0.0-1.0)
        - title (str): Page title
        - url (str): Source URL
        - heading (str): Section hierarchy

    Raises:
        ValueError: If query is empty or top_k is out of range
        RuntimeError: If retrieval service is unavailable
    """
```

### OpenAI Tool Schema (JSON)

```json
{
  "type": "function",
  "function": {
    "name": "retrieve_book_content",
    "description": "Retrieves relevant book content chunks from the Physical AI book using semantic search. Use this tool when the user asks questions about book topics like ROS 2, computer vision, ML frameworks, or any technical content in the book. Returns top-k most similar chunks with source metadata.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The user's question or search query to find relevant content"
        },
        "top_k": {
          "type": "integer",
          "description": "Number of most relevant chunks to retrieve (default 3, max 10)",
          "default": 3,
          "minimum": 1,
          "maximum": 10
        }
      },
      "required": ["query"]
    }
  }
}
```

## Input Validation

### Query Parameter
- **Type**: string
- **Required**: Yes
- **Constraints**:
  - MUST NOT be empty string
  - SHOULD be at least 3 characters for meaningful retrieval
  - Typical length: 5-200 characters
- **Examples**:
  - ✅ "What is physical AI?"
  - ✅ "How do I install ROS 2?"
  - ✅ "Explain computer vision basics"
  - ❌ "" (empty)
  - ❌ "a" (too short, likely not meaningful)

### top_k Parameter
- **Type**: integer
- **Required**: No (defaults to 3)
- **Constraints**:
  - MUST be between 1 and 10 inclusive
  - Default: 3 (balanced between context and performance)
- **Rationale**:
  - 1-2: Minimal context, faster
  - 3-5: Good balance (recommended)
  - 6-10: Maximum context, slower retrieval

## Output Format

### Success Response

Returns a list of dictionaries (JSON-serializable), each representing a retrieved chunk:

```python
[
    {
        "text": "Physical AI refers to artificial intelligence systems that interact with and manipulate the physical world...",
        "score": 0.87,
        "title": "Introduction to Physical AI",
        "url": "https://physical-ai-book.pages.dev/docs/intro",
        "heading": "Introduction > What is Physical AI?"
    },
    {
        "text": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software...",
        "score": 0.82,
        "title": "Module 1: ROS 2 Basics",
        "url": "https://physical-ai-book.pages.dev/docs/module-1/ros2-basics",
        "heading": "Module 1 > ROS 2 Basics > Overview"
    },
    # ... up to top_k results
]
```

### Empty Results Response

When no relevant chunks found (similarity scores below threshold):

```python
[]  # Empty list
```

**Agent Handling**: When empty list received, agent MUST respond "I don't have information about that in my knowledge base" (per FR-007).

### Error Response

When retrieval fails (network error, API unavailable, etc.):

```python
# Raises RuntimeError with descriptive message
raise RuntimeError("Retrieval service temporarily unavailable. Please try again later.")
```

**Agent Handling**: Agent should catch error and inform user gracefully (per SC-007).

## Implementation Details

### Integration with Feature 009

The tool implementation calls existing retrieval pipeline functions:

```python
# Pseudocode (actual implementation in agent.py)
def retrieve_book_content(query: str, top_k: int = 3) -> List[Dict]:
    # Import from backend/retrieve.py
    from backend.retrieve import generate_query_embedding, search_qdrant, config

    # Validate inputs
    if not query or len(query.strip()) < 3:
        raise ValueError("Query must be at least 3 characters")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    # Generate embedding for query
    query_vector = generate_query_embedding(query)  # Cohere API call

    # Search Qdrant
    results = search_qdrant(query_vector, top_k)  # Qdrant API call

    # Convert SearchResult dataclass instances to dicts
    return [
        {
            "text": r.text,
            "score": r.score,
            "title": r.title,
            "url": r.url,
            "heading": r.heading
        }
        for r in results
    ]
```

### Performance Constraints

- **Execution Time**: MUST complete within 3 seconds (inherited from Feature 009 SC-005)
- **Breakdown**:
  - Embedding generation (Cohere): ~1 second
  - Qdrant search: ~1 second
  - Serialization: <0.1 second
  - Total: ~2-3 seconds typical

### Error Scenarios

| Scenario | Error Type | Agent Behavior |
|----------|-----------|----------------|
| Empty query | ValueError | Request clarification from user |
| top_k out of range | ValueError | Use default (3) or request correction |
| Cohere API down | RuntimeError | Inform user "retrieval temporarily unavailable" |
| Qdrant API down | RuntimeError | Inform user "retrieval temporarily unavailable" |
| Network timeout | RuntimeError | Inform user "retrieval temporarily unavailable" |
| Zero results | (no error) | Return empty list, agent responds "not available" |

## Usage Example

### Agent Invocation Flow

1. **User**: "What is physical AI?"
2. **Agent**: (decides to use retrieval tool)
3. **Tool Call**: `retrieve_book_content(query="What is physical AI?", top_k=3)`
4. **Tool Returns**: 3 chunks about physical AI with scores 0.87, 0.82, 0.78
5. **Agent**: Reads chunks, generates response using only retrieved context
6. **Agent**: Appends citation: "Source: Introduction to Physical AI - https://..."
7. **User**: Sees grounded response with source reference

### Follow-up Query Flow

1. **User**: "How do I get started with it?"
2. **Agent**: (understands "it" = physical AI from conversation history)
3. **Tool Call**: `retrieve_book_content(query="getting started with physical AI", top_k=3)`
4. **Tool Returns**: 3 chunks about prerequisites, installation, tutorials
5. **Agent**: Generates response in context of previous conversation
6. **User**: Sees contextual response with citations

## Testing Criteria

### Unit Tests (Future)
- ✅ Valid query with default top_k returns results
- ✅ Valid query with custom top_k returns correct number of results
- ✅ Empty query raises ValueError
- ✅ top_k = 0 raises ValueError
- ✅ top_k = 11 raises ValueError
- ✅ Results contain all required fields (text, score, title, url, heading)

### Integration Tests (Manual for MVP)
- ✅ Query "What is physical AI?" returns relevant chunks with score >0.4
- ✅ Query about non-existent topic returns empty list or low-score results
- ✅ Tool execution completes within 3 seconds

### Acceptance Criteria (from spec)
- ✅ US1-AS2: Tool successfully queries Qdrant and returns top-k chunks (FR-004)
- ✅ US2-AS1: Agent calls retrieval tool and receives chunks (FR-002, FR-005)
- ✅ SC-005: Tool execution time <3 seconds (inherited from Feature 009)

## Dependencies

- **backend/retrieve.py**: generate_query_embedding(), search_qdrant(), SearchResult, config
- **Cohere API**: Embedding generation (embed-english-v3.0)
- **Qdrant API**: Vector search (collection "docusaurus_docs", 192 vectors)
- **.env**: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

## Notes

- **No Modifications to Feature 009**: This tool is a thin wrapper around existing functions, no changes to retrieve.py required
- **In-Memory Only**: No database storage, all results returned directly to agent
- **Read-Only**: Tool only queries Qdrant, does not modify vectors or metadata
- **Stateless**: Each tool call is independent, no state maintained between calls
