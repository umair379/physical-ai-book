# Research: FastAPI Backend Integration for RAG System

**Feature**: 011-fastapi-integration
**Date**: 2025-12-28
**Status**: Complete

## Research Overview

This document consolidates research findings for integrating the existing RAG agent (Feature 010) with a web frontend using FastAPI. Three main areas were investigated:

1. **CORS Configuration** - Enabling cross-origin requests from Docusaurus frontend
2. **Error Handling** - FastAPI exception handling and structured error responses
3. **Docusaurus Chatbot Integration** - Adding a persistent chat UI to the documentation site

---

## Decision 1: CORS Middleware Configuration

### Decision
Use FastAPI's `CORSMiddleware` with environment-based configuration for development (localhost) and production (deployed URLs).

### Rationale
- **Security**: Explicit origin allowlist prevents unauthorized cross-origin access
- **Flexibility**: Environment variables enable different settings for dev/prod without code changes
- **FastAPI Native**: Built-in middleware requires no additional dependencies
- **Standards Compliant**: Follows CORS specification and browser security model

### Configuration Pattern

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

def get_cors_origins():
    """Get CORS allowed origins based on environment."""
    env = os.getenv("ENVIRONMENT", "development")

    if env == "development":
        return [
            "http://localhost:3000",      # Docusaurus dev server
            "http://127.0.0.1:3000",
        ]
    elif env == "production":
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
        return [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
    else:
        return []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)
```

### Key Parameters

| Parameter | Development | Production | Notes |
|-----------|------------|------------|-------|
| `allow_origins` | `["http://localhost:3000"]` | From `ALLOWED_ORIGINS` env var | NEVER use `["*"]` with credentials |
| `allow_credentials` | `True` | `True` | Required for auth headers/cookies |
| `allow_methods` | `["GET", "POST", "OPTIONS"]` | Same | Explicit whitelist only |
| `allow_headers` | `["Content-Type", "Authorization"]` | Same | Minimal required headers |
| `max_age` | `600` (10 min) | `600` | Preflight cache duration |

### Security Rules
1. **Never use wildcards with credentials**: `allow_origins=["*"]` + `allow_credentials=True` is a critical vulnerability
2. **HTTPS in production**: Always use `https://` URLs for production origins
3. **Principle of least privilege**: Only allow necessary methods and headers
4. **Validate origins dynamically**: Log and block requests from unauthorized origins

### Alternatives Considered
- **Custom CORS middleware**: Unnecessary complexity, FastAPI's built-in solution is sufficient
- **Nginx/reverse proxy CORS**: Adds deployment dependency, prefer application-level control
- **Wildcard origins**: Rejected for security reasons

---

## Decision 2: Error Handling Strategy

### Decision
Implement custom exception classes with global exception handlers that return structured JSON error responses aligned with HTTP status codes.

### Rationale
- **Type Safety**: Custom exceptions provide type-safe error handling with IDE support
- **Separation of Concerns**: Exception handlers centralize error formatting logic
- **Consistent API Contract**: All errors follow same JSON schema with `error`, `message`, `request_id` fields
- **Debugging Support**: Request IDs enable tracing errors across logs and API calls
- **User-Friendly**: Internal errors logged with details, clients receive sanitized messages

### Exception Hierarchy

```python
# Custom exception base class
class APIError(Exception):
    """Base class for API errors."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

# Domain-specific exceptions
class InvalidQueryError(APIError):
    """Raised when query validation fails (HTTP 400)."""
    pass

class RetrievalServiceError(APIError):
    """Raised when Qdrant/vector search fails (HTTP 500)."""
    pass

class EmbeddingGenerationError(APIError):
    """Raised when Cohere embedding fails (HTTP 502)."""
    pass

class AssistantExecutionError(APIError):
    """Raised when OpenAI Assistant fails (HTTP 500)."""
    pass
```

### HTTP Status Code Mapping

| Status Code | Exception Type | When to Use | Client Action |
|-------------|----------------|-------------|---------------|
| 400 Bad Request | `InvalidQueryError` | Empty query, invalid format | Fix query and retry |
| 422 Unprocessable Entity | Pydantic `ValidationError` | Missing required fields, type mismatch | Fix request body schema |
| 500 Internal Server Error | `RetrievalServiceError`<br/>`AssistantExecutionError` | Qdrant down, OpenAI timeout, agent crash | Retry later or contact support |
| 502 Bad Gateway | `EmbeddingGenerationError` | Cohere API failure | Retry later |

### Error Response Schema

```python
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str           # Machine-readable error code (e.g., "invalid_query")
    message: str         # Human-readable error message
    request_id: str      # Unique request identifier for tracing
    details: dict = {}   # Optional debugging information (omit in production)
```

### Exception Handler Pattern

```python
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(AssistantExecutionError)
async def assistant_error_handler(request: Request, exc: AssistantExecutionError):
    """Handle OpenAI Assistant failures (500)."""
    logger.error(
        f"Assistant execution failed: {exc.message}",
        extra={
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", None)
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "assistant_execution_error",
            "message": "Failed to generate response",
            "request_id": getattr(request.state, "request_id", None)
        }
    )
```

### Logging Strategy
- **Internal logs**: Include stack traces, request IDs, full error details
- **Client responses**: Sanitized messages, no sensitive data or stack traces
- **Log levels**: INFO for requests, ERROR for server errors, CRITICAL for unexpected exceptions
- **Structured logging**: JSON format with request_id for correlation

### Alternatives Considered
- **Generic HTTPException everywhere**: Less type-safe, harder to track error sources
- **Status code-based error handling**: Doesn't capture domain semantics
- **No custom exceptions**: Loses context about error origin (Qdrant vs OpenAI vs validation)

---

## Decision 3: Docusaurus Chatbot Integration

### Decision
Implement chatbot UI as a fixed-position React component wrapped by a custom `Root.tsx` theme component, using React Context for state management and Fetch API for backend communication.

### Rationale
- **Persistent State**: Root component wrapper ensures chat history survives page navigation
- **Non-Invasive**: Doesn't modify Docusaurus core components (Layout, Navbar)
- **Zero Dependencies**: Uses built-in React Context and Fetch API (no Redux, Axios, or external state libs)
- **Docusaurus Best Practice**: Swizzling Root is the recommended approach for global UI elements
- **Mobile Responsive**: Fixed positioning adapts to different screen sizes

### Component Architecture

```
src/theme/Root.tsx (swizzled wrapper)
  └── ChatProvider (React Context)
        ├── {children} (Docusaurus app)
        └── Chatbot (fixed-position container)
              ├── ChatToggle (floating button)
              └── ChatWindow (when expanded)
                    ├── ChatMessage[] (conversation history)
                    │     └── ChatCitation[] (source links)
                    └── ChatInput (text area + send button)
```

### File Structure

```
frontend-book/
└── src/
    ├── theme/
    │   └── Root.tsx                           # Swizzled wrapper component
    └── components/
        └── Chatbot/
            ├── index.tsx                      # Main Chatbot component
            ├── ChatContext.tsx                # React Context provider
            ├── ChatWindow.tsx                 # Chat UI container
            ├── ChatMessage.tsx                # Individual message display
            ├── ChatCitation.tsx               # Source citation badge
            ├── ChatInput.tsx                  # User input field
            ├── ChatToggle.tsx                 # Open/close button
            ├── useChatAPI.ts                  # Custom hook for API calls
            ├── useChatHistory.ts              # Custom hook for history management
            ├── types.ts                       # TypeScript interfaces
            └── styles.module.css              # Component styles
```

### State Management Pattern

```typescript
// ChatContext.tsx
interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  isExpanded: boolean;
}

type ChatAction =
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'TOGGLE_EXPANDED' };

const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    // ... other cases
  }
};

export const ChatProvider: React.FC = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  return (
    <ChatContext.Provider value={{ state, dispatch }}>
      {children}
    </ChatContext.Provider>
  );
};
```

### API Integration Pattern

```typescript
// useChatAPI.ts
export const useChatAPI = () => {
  const sendQuery = async (query: string): Promise<ChatResponse> => {
    const apiUrl = process.env.API_URL || 'http://localhost:8000';

    const response = await fetch(`${apiUrl}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Request failed');
    }

    return response.json();
  };

  return { sendQuery };
};
```

### Styling Strategy

```css
/* styles.module.css */
.chatbotContainer {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9999; /* Above Docusaurus navbar (z-index: 100) */
  max-width: 400px;
  max-height: 600px;
}

@media (max-width: 768px) {
  .chatbotContainer {
    bottom: 10px;
    right: 10px;
    left: 10px;
    max-width: 100%;
  }
}
```

### Citation Display Pattern

**Inline badges with expandable details:**

```typescript
// ChatCitation.tsx
const ChatCitation: React.FC<{citation: Citation}> = ({citation}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={styles.citation}>
      <button
        className={styles.citationBadge}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-label={`Source ${citation.index}`}
      >
        [{citation.index}]
      </button>
      {isExpanded && (
        <div className={styles.citationDetails}>
          <a href={citation.url} target="_blank" rel="noopener noreferrer">
            {citation.title}
          </a>
          <span>Relevance: {(citation.score * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
};
```

### Configuration in docusaurus.config.ts

```typescript
module.exports = {
  // ... other config
  customFields: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
  },
};
```

### Alternatives Considered
- **Layout component modification**: Too invasive, breaks on Docusaurus updates
- **Plugin approach**: Overkill for this use case, Root wrapper is simpler
- **Redux for state**: Unnecessary complexity, React Context sufficient
- **Third-party chat components**: NIH (not invented here), prefer custom implementation for full control
- **Footer-based citation display**: Less discoverable than inline badges

---

## Additional Considerations

### Performance
- **Debounce input**: Prevent rapid-fire API calls while user is typing
- **Request cancellation**: Use AbortController to cancel in-flight requests when new query submitted
- **Optimistic UI**: Show user message immediately before API response
- **Skeleton loaders**: Display loading state instead of blank screen

### Accessibility
- **Keyboard navigation**: Tab through chat window, Enter to send, Escape to close
- **ARIA labels**: Screen reader announcements for messages, citations
- **Focus management**: Auto-focus input when chat opens
- **Color contrast**: Ensure citation badges meet WCAG AA standards

### Security
- **XSS prevention**: Sanitize user input before rendering (React does this by default)
- **Rate limiting**: Prevent abuse by limiting requests per user/IP (future enhancement)
- **Input validation**: Frontend validation before API call (length limits, character whitelist)

---

## Technology Stack Summary

| Component | Technology Choice | Rationale |
|-----------|------------------|-----------|
| Backend Framework | FastAPI | Async support, auto-generated OpenAPI docs, Pydantic validation |
| CORS Middleware | FastAPI CORSMiddleware | Native, standards-compliant, environment-aware |
| Error Handling | Custom exceptions + global handlers | Type-safe, consistent API contract, structured logging |
| Frontend Framework | Docusaurus v3 + React | Static site generator for docs, React components for chat UI |
| State Management | React Context + useReducer | Zero dependencies, sufficient for chat history |
| API Client | Fetch API | Native browser API, no external dependencies |
| Styling | CSS Modules | Scoped styles, no conflicts with Docusaurus themes |
| Component Integration | Root.tsx swizzling | Docusaurus best practice for global UI elements |

---

## Implementation Sequence

### Backend (api.py)
1. Create FastAPI app with CORS middleware
2. Define Pydantic request/response models
3. Implement custom exception classes
4. Register global exception handlers
5. Create POST /api/query endpoint that calls agent.py
6. Create GET /health endpoint for monitoring
7. Add request ID middleware for tracing
8. Configure logging with structured output

### Frontend (Docusaurus)
1. Swizzle Root component: `npm run swizzle @docusaurus/theme-classic Root -- --wrap`
2. Create Chatbot component directory structure
3. Implement ChatContext provider with useReducer
4. Build ChatToggle (floating button)
5. Build ChatWindow (container)
6. Build ChatMessage (user/assistant display)
7. Build ChatCitation (source badges)
8. Build ChatInput (text area + send button)
9. Implement useChatAPI hook for backend communication
10. Add CSS styles with fixed positioning and responsive design
11. Test integration with mock data
12. Connect to real FastAPI backend and test end-to-end

---

## References

- [FastAPI CORS Documentation](https://fastapi.tiangolo.com/tutorial/cors/)
- [FastAPI Error Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [Docusaurus Swizzling Guide](https://docusaurus.io/docs/swizzling)
- [React Context API](https://react.dev/reference/react/useContext)
- [MDN CORS Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

---

**Status**: All research complete - ready for implementation planning phase
