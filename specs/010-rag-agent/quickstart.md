# Quick Start: RAG Agent with OpenAI Assistants API

**Feature**: 010-rag-agent | **Date**: 2025-12-28

## Overview

This guide walks through setting up and running the RAG agent that answers questions about Physical AI book content using retrieval-augmented generation.

**Time to First Query**: ~5 minutes (assuming OpenAI API key ready)

---

## Prerequisites

### 1. Environment Setup

Ensure existing retrieval pipeline (Feature 009) is functional:

```bash
# Navigate to backend
cd backend

# Verify existing .env has Cohere and Qdrant credentials
cat .env
# Should contain:
# COHERE_API_KEY=...
# QDRANT_URL=...
# QDRANT_API_KEY=...
# COLLECTION_NAME=docusaurus_docs
```

### 2. Get OpenAI API Key

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Navigate to API Keys section
3. Create new key (starts with `sk-proj-...`)
4. Add to `backend/.env`:

```bash
echo "OPENAI_API_KEY=sk-proj-your-key-here" >> backend/.env
```

### 3. Install OpenAI SDK

```bash
# Add to dependencies
uv add openai

# Verify installation
uv run python -c "import openai; print(openai.__version__)"
# Should print: 1.58.1 or higher
```

**Expected output**: Version number (e.g., `1.58.1`)

---

## Quick Test (Single Query)

### Step 1: Verify Agent Script

The agent script is located at `backend/agent.py` with the following implementation:

```python
#!/usr/bin/env python3
"""Minimal RAG Agent - Quick Test Version"""

import os, sys, json, time
from openai import OpenAI
from dotenv import load_dotenv
from backend.retrieve import generate_query_embedding, search_qdrant, ValidationConfig
from qdrant_client import QdrantClient

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_book_content(query: str, top_k: int = 3) -> str:
    """Retrieval tool - bridges to Feature 009 pipeline"""
    config = ValidationConfig()
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=120)
    query_vector = generate_query_embedding(query, config)
    results = search_qdrant(query_vector, qdrant, config, top_k)

    if not results:
        return "No relevant information found."

    chunks = [
        f"[Chunk {i}] (Score: {r.score:.3f})\\n"
        f"Title: {r.payload.get('title')}\\nURL: {r.payload.get('url')}\\n"
        f"Content: {r.payload.get('text')}"
        for i, r in enumerate(results, 1)
    ]
    return "\\n---\\n".join(chunks)

# Tool schema
tool = {
    "type": "function",
    "function": {
        "name": "retrieve_book_content",
        "description": "Search Physical AI book documentation",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }
    }
}

# Create assistant and thread
assistant = client.beta.assistants.create(
    name="RAG Book Assistant",
    instructions="Answer using ONLY retrieved book content. Include source citations. If information not available, say so explicitly.",
    model="gpt-4o-mini",
    tools=[tool]
)
thread = client.beta.threads.create()

# Query execution
def ask(query: str) -> str:
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=query)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

    while run.status in ["queued", "in_progress", "requires_action"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action":
            outputs = []
            for tc in run.required_action.submit_tool_outputs.tool_calls:
                if tc.function.name == "retrieve_book_content":
                    args = json.loads(tc.function.arguments)
                    result = retrieve_book_content(**args)
                    outputs.append({"tool_call_id": tc.id, "output": result})

            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id, run_id=run.id, tool_outputs=outputs
            )
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id, limit=1)
    return messages.data[0].content[0].text.value

# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'Your question here'")
        sys.exit(1)

    print(f"\\nQuerying agent: {sys.argv[1]}")
    response = ask(sys.argv[1])
    print(f"\\nAgent: {response}\\n")
```

### Step 2: Run First Query

```bash
# From backend directory
cd backend
uv run python agent.py "What is physical AI?"
```

**Expected output** (3-5 seconds):

```
Querying agent: What is physical AI?

Agent: Physical AI refers to artificial intelligence systems that interact with and manipulate the physical world. According to the book content, it involves...

[Agent response with citation to source URL]
```

### Step 3: Verify Results

✅ **Success criteria**:
1. Response time <10 seconds
2. Answer references book content
3. Source citation included (title and/or URL mentioned)
4. No error messages

❌ **Common errors**:
- `AuthenticationError`: Check `OPENAI_API_KEY` in `.env`
- `ModuleNotFoundError`: Run with `uv run python` not plain `python`
- `Connection failed`: Verify Cohere and Qdrant credentials (from Feature 009)

---

## Interactive Conversation (Follow-ups)

### Single-Session Conversation

```bash
# Modify agent.py CLI section for interactive mode:
# Replace the if __name__ == "__main__": block with:

if __name__ == "__main__":
    print("RAG Agent (type 'exit' to quit)\\n")

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        if not query.strip():
            continue

        response = ask(query)
        print(f"Agent: {response}\\n")
```

### Example Conversation

```
RAG Agent (type 'exit' to quit)

You: What is physical AI?
Agent: Physical AI refers to AI systems that interact with the physical world through sensors and actuators. It combines perception, reasoning, and action to solve real-world problems. [Source: Introduction - https://...]

You: What are its key components?
Agent: Based on the previous context about physical AI, the key components include:
1. Perception systems (computer vision, sensors)
2. Decision-making algorithms (ML/RL)
3. Actuation mechanisms (robotics, control systems)
[Source: Module 1 - https://...]

You: exit
```

**Note**: Conversation history maintained automatically via OpenAI Thread object.

---

## Testing Scenarios

### Test 1: Basic Q&A (User Story 1)

**Query**: "What is ROS 2?"

**Expected**:
- Retrieves 3 chunks about ROS 2
- Response explains ROS 2 using only retrieved content
- Source citation included

**Validation**: `SC-002` - Answer based on book content only

---

### Test 2: Unknown Topic (User Story 2 - Edge Case)

**Query**: "How do I train a GAN model?"

**Expected**:
- Agent attempts retrieval (may return 0-1 results with low scores)
- Response: "I don't have information about that in my knowledge base" or similar
- No hallucinated content

**Validation**: `SC-003` - 100% correct "not available" responses for non-book topics

---

### Test 3: Follow-up Query (User Story 3)

**Initial Query**: "What is computer vision?"

**Follow-up**: "What are its applications in robotics?"

**Expected**:
- Agent retrieves chunks about computer vision applications
- Response contextual to previous question
- Mentions robotics use cases from book

**Validation**: `SC-004` - Maintains context for at least 3-message exchange

---

## Performance Benchmarking

### Measure Response Time

```python
# Add timing to ask() function:
import time

def ask(query: str) -> str:
    start_time = time.time()

    # ... existing code ...

    elapsed = time.time() - start_time
    print(f"[Response time: {elapsed:.2f}s]")

    return messages.data[0].content[0].text.value
```

**Target**: <10 seconds (SC-005)

**Breakdown**:
- Embedding generation: ~0.5s
- Qdrant search: ~0.2s
- OpenAI API call: ~2-4s
- Total: ~3-5s typical

---

## Cost Monitoring

### Estimate Token Usage

```python
# After running query, check usage:
run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
if hasattr(run, 'usage'):
    print(f"Tokens used: {run.usage.total_tokens}")
    print(f"Input: {run.usage.prompt_tokens}, Output: {run.usage.completion_tokens}")
```

**Expected per query** (with 3 chunks):
- Input: ~1,000 tokens (instructions + retrieved chunks + query)
- Output: ~200 tokens (agent response)
- Total: ~1,200 tokens

**Cost** (gpt-4o-mini):
- Input: $0.15/1M × 1,000 = $0.00015
- Output: $0.60/1M × 200 = $0.00012
- **Total: ~$0.00027 per query**

**For 100 test queries**: ~$0.03 total

---

## Troubleshooting

### Error: `Incorrect API key provided`

**Cause**: Invalid or missing `OPENAI_API_KEY` in `.env`

**Fix**:
```bash
# Verify key exists
grep OPENAI_API_KEY backend/.env

# If missing, add it:
echo "OPENAI_API_KEY=sk-proj-your-key" >> backend/.env

# Restart script (reload .env)
```

---

### Error: `ModuleNotFoundError: No module named 'openai'`

**Cause**: Running with plain `python` instead of `uv run`

**Fix**:
```bash
# Always use uv run
uv run python agent.py "Your question"

# Or activate virtual environment:
cd backend
uv sync
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows
cd ..
python agent.py "Your question"
```

---

### Error: `Retrieval failed: Connection timeout`

**Cause**: Qdrant or Cohere API credentials invalid/expired

**Fix**:
```bash
# Test retrieval pipeline directly (Feature 009)
cd backend
uv run python retrieve.py --query "test" --top-k 1

# If this works, issue is with agent.py
# If this fails, check Feature 009 setup
```

---

### Error: `Rate limit exceeded`

**Cause**: Too many OpenAI API requests in short time

**Fix**:
```python
# Add exponential backoff retry (see research.md section 7)
# Or wait 60 seconds and retry
# Or upgrade OpenAI API tier for higher limits
```

---

### Agent Returns Empty Responses

**Cause**: Thread or run status not handled correctly

**Debug**:
```python
# Add debug logging in ask() function:
print(f"Run status: {run.status}")
if run.status == "failed":
    print(f"Error: {run.last_error}")

# Check if tool was called:
if run.status == "requires_action":
    print(f"Tool calls: {run.required_action.submit_tool_outputs.tool_calls}")
```

---

### Slow Response (>10 seconds)

**Possible causes**:
1. Large number of chunks (try reducing `top_k` to 1-2)
2. Network latency to OpenAI API
3. Qdrant/Cohere API slowness

**Diagnosis**:
```python
# Add timing breakdowns:
import time

def ask(query: str) -> str:
    t0 = time.time()

    # Add message
    client.beta.threads.messages.create(...)
    t1 = time.time()
    print(f"Message creation: {t1-t0:.2f}s")

    # Run assistant
    run = client.beta.threads.runs.create(...)
    t2 = time.time()
    print(f"Run creation: {t2-t1:.2f}s")

    # Poll loop
    while run.status in [...]:
        # ... poll logic ...
        pass
    t3 = time.time()
    print(f"Polling + tool execution: {t3-t2:.2f}s")

    # Retrieve messages
    messages = client.beta.threads.messages.list(...)
    t4 = time.time()
    print(f"Message retrieval: {t4-t3:.2f}s")
```

---

## Next Steps

After verifying basic functionality:

1. **Run full test suite** (see `spec.md` User Stories for test scenarios)
2. **Measure success criteria**:
   - SC-002: Test 5 questions from book (100% accuracy?)
   - SC-003: Test 3 adversarial questions (100% "not available"?)
   - SC-004: Test 3-message conversation (context maintained?)
   - SC-005: Response time <10s? (measure with timing)
   - SC-006: Citations present in 80%+ responses?
3. **Refine system prompt** if agent hallucinates or doesn't cite sources
4. **Implement error handling** (see research.md section 7)
5. **Add logging** for debugging and monitoring

---

## Summary

**Minimal setup** (5 steps):
1. Add `OPENAI_API_KEY` to `backend/.env`
2. Run `uv add openai`
3. Create `agent.py` (copy code from above)
4. Run `uv run python agent.py "What is physical AI?"`
5. Verify response includes content from book and source citation

**Expected time**: 5 minutes

**Expected cost**: ~$0.0003 per query ($0.03 for 100 test queries)

**Expected performance**: 3-5 seconds per query

**Success markers**:
- ✅ Agent responds using retrieved book content only
- ✅ Source citations included in responses
- ✅ Follow-up queries maintain conversation context
- ✅ Unknown topics handled gracefully ("not available")
- ✅ Response time <10 seconds
