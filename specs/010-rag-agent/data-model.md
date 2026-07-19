# Data Model: AI Agent with Retrieval-Augmented Capabilities

**Feature**: 010-rag-agent | **Date**: 2025-12-28

## Overview

This feature uses existing data models from Feature 009 (retrieval pipeline) and adds minimal agent-specific structures for conversation management. No new database entities required - all conversation state is in-memory.

## Entities

### 1. Agent (Orchestration Layer)

**Purpose**: Manages conversation flow and tool invocation using OpenAI Agents SDK

**Attributes**:
- `conversation_history`: List[Message] - Accumulated user/assistant message pairs
- `registered_tools`: List[Tool] - Available tools (retrieval tool in this MVP)
- `system_prompt`: str - Instructions for agent behavior (use only retrieved content, cite sources)

**Lifecycle**:
- Created: On agent initialization (once per session)
- Updated: After each user query (history appended)
- Destroyed: On script termination (in-memory only, not persisted)

**Validation Rules**:
- System prompt MUST instruct agent to use only retrieved chunks (FR-006)
- System prompt MUST instruct agent to include source citations (FR-009)
- System prompt MUST instruct agent to inform user when information not available (FR-007)

**Relationships**:
- Has-many: Message (conversation history)
- Has-many: Tool (registered tools)

---

### 2. RetrievalTool (Function-based Tool)

**Purpose**: Interfaces with existing Qdrant retrieval pipeline to fetch relevant book chunks

**Attributes** (Tool Schema):
- `name`: str = "retrieve_book_content"
- `description`: str = "Retrieves relevant book content chunks from Qdrant vector database based on query"
- `parameters`:
  - `query` (str, required): User question or search query
  - `top_k` (int, optional, default=3): Number of chunks to retrieve

**Implementation**:
- Calls `generate_query_embedding(query)` from backend/retrieve.py
- Calls `search_qdrant(query_vector, top_k)` from backend/retrieve.py
- Returns: List[SearchResult] (existing dataclass from Feature 009)

**Validation Rules**:
- Query MUST not be empty string (FR-003)
- top_k MUST be integer between 1 and 10 (reasonable range)
- Tool execution MUST complete within 3 seconds (inherited constraint from Feature 009)

**Error Handling**:
- Network failures → Return error message to agent "Retrieval temporarily unavailable"
- Zero results → Return empty list (agent handles per FR-007)
- Cohere API errors → Return error message to agent

**Relationships**:
- Uses: SearchResult (from Feature 009)
- Calls: generate_query_embedding(), search_qdrant() (from Feature 009)

---

### 3. Message (Conversation Turn)

**Purpose**: Represents a single user or assistant message in conversation history

**Attributes**:
- `role`: Literal["user", "assistant"] - Message sender
- `content`: str - Message text
- `timestamp`: datetime - When message was created (optional, for logging)

**Lifecycle**:
- Created: On user input (role="user") or agent response (role="assistant")
- Updated: Never (immutable)
- Destroyed: On agent session end (in-memory only)

**Validation Rules**:
- Content MUST not be empty string
- Role MUST be either "user" or "assistant"

**Storage**: In-memory list, not persisted to database (Assumption 5 from spec)

---

### 4. SearchResult (Reused from Feature 009)

**Purpose**: Represents a retrieved chunk from Qdrant with metadata

**Attributes** (existing dataclass from backend/retrieve.py):
- `text`: str - Chunk content
- `score`: float - Similarity score (0.0-1.0)
- `title`: str - Page title
- `url`: str - Source URL
- `heading`: str - Section hierarchy

**Source**: `backend/retrieve.py` (Feature 009)

**Usage in Agent**:
- Agent receives List[SearchResult] from RetrievalTool
- Agent formats results into context for response generation
- Agent extracts title and URL for source citations (FR-009)

**No modifications needed** - reused as-is

---

## Data Flow

```
User Query
    ↓
Agent (receives query)
    ↓
Agent invokes RetrievalTool(query, top_k=3)
    ↓
RetrievalTool → generate_query_embedding(query)  [Cohere API]
    ↓
RetrievalTool → search_qdrant(query_vector, 3)   [Qdrant API]
    ↓
RetrievalTool returns List[SearchResult]
    ↓
Agent formats chunks into context
    ↓
Agent generates response using only retrieved context
    ↓
Agent appends citation (title, URL) to response
    ↓
Agent returns response to user
    ↓
Conversation history updated (user message + assistant response)
```

## State Management

**In-Memory State** (not persisted):
- Conversation history: List[Message]
- Current session only
- Cleared on agent restart

**External State** (persistent):
- Qdrant vectors: 192 embeddings (from Feature 008, read-only)
- Configuration: .env file (OPENAI_API_KEY, COHERE_API_KEY, QDRANT_URL, etc.)

**No Database Required**: All conversation state is transient, aligned with Assumption 5 (in-memory storage)

## Validation Summary

| Entity | Key Validations |
|--------|----------------|
| Agent | System prompt enforces grounding, citations, zero-result handling |
| RetrievalTool | Query non-empty, top_k range 1-10, execution timeout 3s |
| Message | Content non-empty, role in ["user", "assistant"] |
| SearchResult | (Inherited from Feature 009, no changes) |

## Dependencies

- **Feature 009**: SearchResult dataclass, generate_query_embedding(), search_qdrant()
- **Feature 008**: Qdrant collection "docusaurus_docs" with 192 vectors
- **OpenAI SDK**: Agent class, Tool registration, conversation management
- **Existing libraries**: cohere, qdrant-client, pydantic-settings (already installed)
