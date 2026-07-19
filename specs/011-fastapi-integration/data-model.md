# Data Model: FastAPI Backend Integration

**Feature**: 011-fastapi-integration
**Date**: 2025-12-28

## Overview

This document defines the data entities for the FastAPI backend that bridges the frontend chat UI with the existing RAG agent from Feature 010.

---

## API Request/Response Entities

###  1. QueryRequest

**Purpose**: Represents an incoming user query from the frontend chat interface.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `query` | `string` | Yes | Min length: 1<br/>Max length: 10,000<br/>Must not be whitespace-only | The user's question about Physical AI book content |
| `thread_id` | `string` | No | UUID format (if provided) | OpenAI thread ID for conversation history (omit for new conversation) |
| `top_k` | `integer` | No | Min: 1<br/>Max: 10<br/>Default: 3 | Number of relevant chunks to retrieve from vector database |

**Example**:
```json
{
  "query": "What is ROS 2 and how does it differ from ROS 1?",
  "thread_id": "thread_abc123xyz",
  "top_k": 3
}
```

**Validation Rules**:
- `query` must contain at least one non-whitespace character
- If `thread_id` is provided, it must be a valid UUID format
- `top_k` defaults to 3 if omitted

---

### 2. QueryResponse

**Purpose**: Represents the agent's response to a user query, including the answer and source citations.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `answer` | `string` | Yes | The agent's text response grounded in book content |
| `sources` | `array[Source]` | Yes | List of source citations (empty array if no sources) |
| `thread_id` | `string` | Yes | OpenAI thread ID for this conversation (for follow-up queries) |
| `response_time_ms` | `integer` | Yes | Time taken to generate response (milliseconds) |
| `request_id` | `string` | Yes | Unique request identifier for tracing/debugging |

**Example**:
```json
{
  "answer": "ROS 2 is the next generation of the Robot Operating System...",
  "sources": [
    {
      "title": "ROS 2 Overview",
      "url": "https://physical-ai-book.vercel.app/docs/ros2/overview",
      "score": 0.92,
      "chunk_index": 2
    },
    {
      "title": "Migration from ROS 1",
      "url": "https://physical-ai-book.vercel.app/docs/ros2/migration",
      "score": 0.87,
      "chunk_index": 5
    }
  ],
  "thread_id": "thread_abc123xyz",
  "response_time_ms": 4523,
  "request_id": "req_20251228_143022_xyz"
}
```

---

### 3. Source

**Purpose**: Represents a single source citation from the RAG retrieval process.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | `string` | Yes | Title of the source document/page |
| `url` | `string` | Yes | Full URL to the source page in the book |
| `score` | `float` | Yes | Semantic similarity score (0.0 - 1.0) |
| `chunk_index` | `integer` | Yes | Index of the retrieved chunk within the source document |

**Example**:
```json
{
  "title": "Computer Vision Fundamentals",
  "url": "https://physical-ai-book.vercel.app/docs/cv/fundamentals",
  "score": 0.94,
  "chunk_index": 3
}
```

**Validation Rules**:
- `score` must be between 0.0 and 1.0 (inclusive)
- `url` must be a valid HTTP/HTTPS URL
- `chunk_index` must be non-negative

---

### 4. ErrorResponse

**Purpose**: Represents error information returned when a request fails.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error` | `string` | Yes | Machine-readable error code (e.g., "invalid_query") |
| `message` | `string` | Yes | Human-readable error message |
| `request_id` | `string` | Yes | Unique request identifier for tracing |
| `details` | `object` | No | Additional debugging information (omitted in production) |

**Example**:
```json
{
  "error": "invalid_query",
  "message": "Query cannot be empty",
  "request_id": "req_20251228_143022_abc"
}
```

**Error Codes**:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_query` | 400 | Query is empty or contains only whitespace |
| `query_too_long` | 400 | Query exceeds maximum length (10,000 characters) |
| `validation_error` | 422 | Request body doesn't match expected schema |
| `assistant_execution_error` | 500 | OpenAI Assistant failed to generate response |
| `retrieval_service_error` | 500 | Qdrant vector database unavailable |
| `embedding_generation_error` | 502 | Cohere API failed to generate embeddings |
| `internal_server_error` | 500 | Unexpected server error |

---

### 5. HealthResponse

**Purpose**: Represents the health check response for service monitoring.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | `string` | Yes | Service status ("healthy" or "degraded") |
| `agent_ready` | `boolean` | Yes | Whether RAG agent is initialized |
| `timestamp` | `string` | Yes | ISO 8601 timestamp of health check |
| `version` | `string` | Yes | API version string |

**Example**:
```json
{
  "status": "healthy",
  "agent_ready": true,
  "timestamp": "2025-12-28T14:30:22.123Z",
  "version": "1.0.0"
}
```

---

## Internal Entities (Not Exposed via API)

### 6. AgentMessage

**Purpose**: Internal representation of messages exchanged with the OpenAI Assistant.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `"user" \| "assistant"` | Message sender |
| `content` | `string` | Message text |
| `thread_id` | `string` | OpenAI thread identifier |
| `timestamp` | `datetime` | When message was created |

**Usage**: Tracked internally for debugging and logging; not exposed to frontend.

---

### 7. RetrievalResult

**Purpose**: Internal result from the RAG retrieval pipeline (Feature 010).

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_text` | `string` | Retrieved text chunk |
| `score` | `float` | Similarity score (0.0 - 1.0) |
| `title` | `string` | Source document title |
| `url` | `string` | Source document URL |
| `heading` | `string` | Section heading within document |
| `chunk_index` | `integer` | Chunk position in document |

**Usage**: Transformed into `Source` objects for API response.

---

## Entity Relationships

```
QueryRequest (from frontend)
    ↓
AgentMessage (internal: user message)
    ↓
RetrievalResult[] (internal: from Feature 010 pipeline)
    ↓
AgentMessage (internal: assistant response)
    ↓
QueryResponse (to frontend)
    ├── answer: string
    ├── sources: Source[]
    ├── thread_id: string
    └── response_time_ms: integer
```

---

## State Transitions

### Query Processing Flow

```
1. Receive QueryRequest
   ├─ Validate query field (non-empty, length <= 10,000)
   ├─ Validate top_k (1 <= top_k <= 10)
   └─ Validate thread_id (UUID format if provided)

2. Execute RAG Agent
   ├─ Call agent.ask(query, thread_id)
   ├─ Agent retrieves chunks via Feature 010
   ├─ Agent generates response with OpenAI
   └─ Agent returns answer + sources

3. Build QueryResponse
   ├─ Extract answer text
   ├─ Map RetrievalResult[] → Source[]
   ├─ Include thread_id for conversation history
   └─ Calculate response_time_ms

4. Return QueryResponse (HTTP 200)
```

### Error Handling Flow

```
1. Validation Error
   └─ Return ErrorResponse (HTTP 422)

2. Agent Execution Error
   ├─ Log full error with stack trace
   ├─ Build sanitized ErrorResponse
   └─ Return ErrorResponse (HTTP 500)

3. External Service Error (Cohere/Qdrant)
   ├─ Log service name + error details
   ├─ Build ErrorResponse with suggestion
   └─ Return ErrorResponse (HTTP 500 or 502)
```

---

## Validation Rules Summary

### QueryRequest Validation
- `query`:
  - MUST NOT be empty or whitespace-only
  - MUST be <= 10,000 characters
  - MAY contain Unicode characters
- `thread_id`:
  - MUST be valid UUID format (if provided)
  - MAY be omitted for new conversations
- `top_k`:
  - MUST be between 1 and 10 (inclusive)
  - Defaults to 3 if omitted

### Source Validation
- `score`:
  - MUST be between 0.0 and 1.0 (inclusive)
  - SHOULD be >= 0.5 for high-quality retrieval (configurable threshold)
- `url`:
  - MUST be valid HTTP/HTTPS URL
  - SHOULD be from allowed domain (e.g., physical-ai-book.vercel.app)

### ErrorResponse Validation
- `error`:
  - MUST be snake_case string
  - MUST match predefined error code list
- `message`:
  - MUST be user-friendly (no stack traces)
  - SHOULD suggest remediation action

---

## Pydantic Models (Implementation Reference)

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    thread_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be whitespace-only')
        return v

class Source(BaseModel):
    title: str
    url: str
    score: float = Field(..., ge=0.0, le=1.0)
    chunk_index: int = Field(..., ge=0)

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    thread_id: str
    response_time_ms: int
    request_id: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    timestamp: str
    version: str
```

---

## Notes

- All timestamps use ISO 8601 format (UTC)
- All field names use snake_case for consistency with Python conventions
- Request IDs use format: `req_{YYYYMMDD}_{HHMMSS}_{random}` for traceability
- Thread IDs are managed by OpenAI Assistants API (format: `thread_{uuid}`)
- Response times measured in milliseconds for precision
- Similarity scores are floats in range [0.0, 1.0] where 1.0 = perfect match

---

**Status**: Data model complete - ready for API contract generation
