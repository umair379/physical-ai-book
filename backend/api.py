#!/usr/bin/env python3
"""
FastAPI Backend for Physical AI RAG System

Provides REST API endpoints for the RAG chatbot, integrating with the
OpenAI Assistants-based agent from Feature 010.
"""

import logging
import time
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Logging Configuration (T004)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


# Custom log adapter to include request_id
class RequestIDAdapter(logging.LoggerAdapter):
    """Log adapter that adds request_id to log messages."""
    def process(self, msg: str, kwargs: Any):
        request_id = self.extra.get('request_id', 'unknown') if self.extra else 'unknown'
        return f'[{request_id}] {msg}', kwargs


# ============================================================================
# FastAPI App Initialization (T002)
# ============================================================================

app = FastAPI(
    title="Physical AI RAG Backend",
    description="FastAPI backend for Physical AI book chatbot with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# CORS Middleware Configuration (T003) - UPDATED FOR LIVE DEPLOYMENT
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type"],
    max_age=600,
)


# ============================================================================
# Pydantic Request/Response Models (T005)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for POST /api/query endpoint."""
    query: str = Field(..., min_length=1, max_length=10000, description="User's question about Physical AI book content")
    thread_id: Optional[str] = Field(None, description="OpenAI thread ID for conversation history")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not whitespace-only."""
        if not v.strip():
            raise ValueError('Query cannot be whitespace-only')
        return v


class Source(BaseModel):
    """Source citation from RAG retrieval."""
    title: str = Field(..., description="Title of the source document/page")
    url: str = Field(..., description="Full URL to the source page")
    score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    chunk_index: int = Field(..., ge=0, description="Index of the retrieved chunk")


class QueryResponse(BaseModel):
    """Response model for POST /api/query endpoint."""
    answer: str = Field(..., description="AI-generated answer grounded in book content")
    sources: List[Source] = Field(..., description="List of source citations")
    thread_id: str = Field(..., description="OpenAI thread ID for this conversation")
    response_time_ms: int = Field(..., description="Time taken to generate response in milliseconds")
    request_id: str = Field(..., description="Unique request identifier for tracing")


class ErrorResponse(BaseModel):
    """Error response model for failed requests."""
    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional debugging information")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status (healthy or degraded)")
    agent_ready: bool = Field(..., description="Whether RAG agent is initialized")
    timestamp: str = Field(..., description="ISO 8601 timestamp of health check")
    version: str = Field(..., description="API version string")


# ============================================================================
# Custom Exception Classes (T006)
# ============================================================================

class APIError(Exception):
    """Base class for API errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details if details is not None else {}
        super().__init__(self.message)


class InvalidQueryError(APIError):
    """Raised when query validation fails (HTTP 400)."""
    pass


class AssistantExecutionError(APIError):
    """Raised when OpenAI Assistant fails (HTTP 500)."""
    pass


class RetrievalServiceError(APIError):
    """Raised when Qdrant/vector search fails (HTTP 500)."""
    pass


class EmbeddingGenerationError(APIError):
    """Raised when Cohere embedding fails (HTTP 502)."""
    pass


# ============================================================================
# Global Exception Handlers (T007)
# ============================================================================

@app.exception_handler(InvalidQueryError)
async def invalid_query_handler(request: Request, exc: InvalidQueryError):
    """Handle invalid query errors (HTTP 400)."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"[{request_id}] Invalid query: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_query",
            "message": exc.message,
            "request_id": request_id
        }
    )


@app.exception_handler(AssistantExecutionError)
async def assistant_error_handler(request: Request, exc: AssistantExecutionError):
    """Handle OpenAI Assistant failures (HTTP 500)."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Assistant execution failed: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "assistant_execution_error",
            "message": "Failed to generate response",
            "request_id": request_id
        }
    )


@app.exception_handler(RetrievalServiceError)
async def retrieval_error_handler(request: Request, exc: RetrievalServiceError):
    """Handle vector database failures (HTTP 500)."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Retrieval service error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "retrieval_service_error",
            "message": "Failed to retrieve content from knowledge base",
            "request_id": request_id
        }
    )


@app.exception_handler(EmbeddingGenerationError)
async def embedding_error_handler(request: Request, exc: EmbeddingGenerationError):
    """Handle Cohere API failures (HTTP 502)."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Embedding generation failed: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=502,
        content={
            "error": "embedding_generation_error",
            "message": "Failed to generate query embedding",
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors (HTTP 500)."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.critical(f"[{request_id}] Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    )


# ============================================================================
# Request ID Middleware (T008)
# ============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    request.state.request_id = request_id

    # LOG INCOMING REQUEST
    origin_header = request.headers.get('origin')
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Origin: {origin_header if origin_header else 'none'}")

    response = await call_next(request)

    # LOG RESPONSE STATUS
    logger.info(f"[{request_id}] Response: {response.status_code}")

    return response


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def submit_query(request: Request, query_request: QueryRequest):
    """
    Submit a query to the RAG agent (T010-T017).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    if not query_request.query.strip():
        raise InvalidQueryError("Query cannot be empty or whitespace-only")

    if len(query_request.query) > 10000:
        raise InvalidQueryError("Query exceeds maximum length of 10,000 characters")

    logger.info(
        f"[{request_id}] Processing query (length={len(query_request.query)}, thread_id={query_request.thread_id})"
    )

    try:
        from agent import ask, thread

        answer = ask(query_request.query)
        
        # Pylance Safety Fix: Check if thread object exists and has an attribute 'id'
        current_thread_id = "thread_new"
        if thread and hasattr(thread, 'id') and thread.id:
            current_thread_id = str(thread.id)

        sources: List[Source] = []
        response_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[{request_id}] Query processed successfully (response_time_ms={response_time_ms}, thread_id={current_thread_id}, answer_length={len(answer)})"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            thread_id=current_thread_id,
            response_time_ms=response_time_ms,
            request_id=request_id
        )

    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"[{request_id}] Agent execution failed (response_time_ms={response_time_ms}): {str(e)}",
            exc_info=True
        )
        raise AssistantExecutionError(
            f"Failed to generate response: {str(e)}",
            details={"response_time_ms": response_time_ms}
        )


@app.get("/")
async def root():
    return {"message": "Physical AI Backend Running!"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# ============================================================================
# Main Entry Point (Updated for dynamic port mapping on HF Spaces)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Hugging Face Spaces dynamically assigns a port via environment variable. 
    # Fallback to 8000 for local runs.
    port = int(os.environ.get("PORT", 8000))
    
    # Live deployment works best with reload=False to avoid container crashing loops
    is_development = os.environ.get("ENVIRONMENT", "production") == "development"
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        log_level="info"
    )