# Quickstart: FastAPI Backend Integration

**Feature**: 011-fastapi-integration
**Estimated Setup Time**: 15 minutes
**Prerequisites**: Python 3.13+, Node.js 18+, Feature 010 (RAG Agent) working

## Architecture Overview

```
┌─────────────────────────────────┐
│   Docusaurus Frontend (React)  │
│   - Chatbot UI components       │
│   - Fixed-position chat window  │
│   - React Context state mgmt    │
└──────────────┬──────────────────┘
               │ HTTP POST /api/query
               │ (JSON)
┌──────────────▼──────────────────┐
│   FastAPI Backend (Python)      │
│   - CORS middleware             │
│   - Query endpoint              │
│   - Error handlers              │
└──────────────┬──────────────────┘
               │ import agent.py
               │
┌──────────────▼──────────────────┐
│   RAG Agent (Feature 010)       │
│   - OpenAI Assistants API       │
│   - Retrieval tool              │
│   - Thread management           │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
  ┌─────────┐   ┌────────┐
  │ Qdrant  │   │ Cohere │
  │ Vector  │   │ Embed  │
  │   DB    │   │  API   │
  └─────────┘   └────────┘
```

---

## Backend Setup (api.py)

### 1. Create API Server File

**File**: `D:\physical-ai-book\backend\api.py`

```python
#!/usr/bin/env python3
"""FastAPI server for RAG agent frontend integration."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import os
import logging

# Import existing RAG agent from Feature 010
from agent import ask, thread

#============================================================================
# Configuration
#============================================================================

app = FastAPI(
    title="Physical AI RAG Backend",
    version="1.0.0",
    description="FastAPI backend for Physical AI book chatbot"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

#============================================================================
# Request/Response Models
#============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    thread_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)

class Source(BaseModel):
    title: str
    url: str
    score: float
    chunk_index: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    thread_id: str
    response_time_ms: int
    request_id: str

#============================================================================
# Endpoints
#============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a query to the RAG agent."""
    start_time = time.time()

    try:
        # Call RAG agent from Feature 010
        answer = ask(request.query)

        # Extract sources (placeholder - implement based on agent.py output)
        sources = []

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            answer=answer,
            sources=sources,
            thread_id=thread.id if thread else "thread_new",
            response_time_ms=response_time_ms,
            request_id=f"req_{int(time.time())}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_ready": True,
        "timestamp": time.time(),
        "version": "1.0.0"
    }

#============================================================================
# Run Server
#============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Install Dependencies

```bash
cd backend
uv add fastapi uvicorn[standard]
```

### 3. Run Server

```bash
cd backend
uv run python api.py
```

**Expected Output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Test with curl

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is physical AI?"}'
```

---

## Frontend Setup (Docusaurus)

### 1. Swizzle Root Component

```bash
cd frontend-book
npm run swizzle @docusaurus/theme-classic Root -- --wrap
```

### 2. Create Chatbot Component

**File**: `frontend-book/src/components/Chatbot/index.tsx`

```typescript
import React, { useState } from 'react';
import styles from './styles.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function Chatbot(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();
      const assistantMessage: Message = { role: 'assistant', content: data.answer };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  if (!isExpanded) {
    return (
      <button
        className={styles.toggleButton}
        onClick={() => setIsExpanded(true)}
      >
        💬 Ask AI
      </button>
    );
  }

  return (
    <div className={styles.chatbot}>
      <div className={styles.header}>
        <span>Physical AI Assistant</span>
        <button onClick={() => setIsExpanded(false)}>✕</button>
      </div>
      <div className={styles.messages}>
        {messages.map((msg, idx) => (
          <div key={idx} className={styles[msg.role]}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className={styles.input}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about Physical AI..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
```

### 3. Add Styles

**File**: `frontend-book/src/components/Chatbot/styles.module.css`

```css
.toggleButton {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 12px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 14px;
  z-index: 9999;
}

.chatbot {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 400px;
  height: 600px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  z-index: 9999;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.header {
  padding: 16px;
  background: #007bff;
  color: white;
  border-radius: 12px 12px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.user {
  background: #e3f2fd;
  padding: 8px 12px;
  border-radius: 8px;
  margin-bottom: 8px;
  max-width: 80%;
  margin-left: auto;
}

.assistant {
  background: #f5f5f5;
  padding: 8px 12px;
  border-radius: 8px;
  margin-bottom: 8px;
  max-width: 80%;
}

.input {
  display: flex;
  padding: 16px;
  border-top: 1px solid #ddd;
}

.input input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 8px;
}

.input button {
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
```

### 4. Integrate in Root.tsx

**File**: `frontend-book/src/theme/Root.tsx`

```typescript
import React from 'react';
import Chatbot from '@site/src/components/Chatbot';

export default function Root({children}) {
  return (
    <>
      {children}
      <Chatbot />
    </>
  );
}
```

### 5. Run Frontend

```bash
cd frontend-book
npm start
```

**Expected**: Chatbot toggle button appears in bottom-right corner

---

## Testing

### 1. Backend Health Check

```bash
curl http://localhost:8000/health
```

**Expected**:
```json
{
  "status": "healthy",
  "agent_ready": true,
  "timestamp": 1735401022.123,
  "version": "1.0.0"
}
```

### 2. Query Endpoint

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ROS 2?"}'
```

### 3. Frontend Integration

1. Open `http://localhost:3000`
2. Click "💬 Ask AI" button
3. Type "What is physical AI?"
4. Click "Send"
5. Verify response appears in chat window

---

## Troubleshooting

### CORS Error in Browser
**Symptom**: Console shows "No 'Access-Control-Allow-Origin' header"
**Fix**: Verify CORS middleware includes `http://localhost:3000` in `allow_origins`

### 500 Error from API
**Symptom**: API returns 500 Internal Server Error
**Fix**: Check `backend/api.py` logs for Python exceptions, verify agent.py works independently

### Chatbot Not Appearing
**Symptom**: No chat button on frontend
**Fix**: Verify Root.tsx was created and imports Chatbot correctly

---

## Next Steps

1. Implement source citations in response
2. Add conversation history (thread_id management)
3. Improve error handling with custom exceptions
4. Add loading states and animations
5. Deploy to production (Vercel + Railway/Render)

---

**Status**: Quickstart complete - MVP setup in 15 minutes
