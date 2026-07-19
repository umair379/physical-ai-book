#!/usr/bin/env python3
"""Minimal RAG Agent with OpenAI Assistants API

This agent uses retrieval-augmented generation to answer questions about Physical AI
book content by querying a Qdrant vector database with semantic search.

Usage:
    # From backend directory
    uv run python agent.py "What is physical AI?"

    # Or from project root
    cd backend && uv run python agent.py "What is physical AI?"

    # Interactive mode
    uv run python agent.py

Features:
    - Semantic search over book documentation using Cohere embeddings
    - OpenAI Assistant with retrieval tool integration
    - Source citation in responses
    - Conversation history for follow-up queries
"""

import os
import sys
import json
import time
from typing import List, Optional
from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run, Message, TextContentBlock
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from dotenv import load_dotenv

# Import existing retrieval pipeline from Feature 009 (same directory)
from retrieve import generate_query_embedding, search_qdrant, ValidationConfig, SearchResult


# ============================================================================
# Configuration
# ============================================================================

# Load .env from current directory (backend/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not found in .env")
    print("Please add your OpenAI API key to backend/.env:")
    print('OPENAI_API_KEY="sk-proj-your-key-here"')
    sys.exit(1)

client = OpenAI(api_key=openai_api_key)


# ============================================================================
# Retrieval Tool Implementation
# ============================================================================

def retrieve_book_content(query: str, top_k: int = 3) -> str:
    """Retrieval tool that bridges to Feature 009 pipeline.

    This function is called by the OpenAI Assistant when it needs to search
    book content to answer a user's question.

    Args:
        query: User question or search query
        top_k: Number of most relevant chunks to retrieve (default 3)

    Returns:
        Formatted string containing retrieved chunks with metadata

    Contract:
        - Input validation: query must be non-empty
        - Output format: structured text with chunk text, score, title, URL
        - Error handling: returns error message if retrieval fails
        - Performance: completes within 3 seconds (inherited from Feature 009)
    """
    try:
        # Validate input
        if not query or len(query.strip()) == 0:
            return "Error: Query cannot be empty"

        # Load configuration from environment variables
        # ValidationConfig is a Pydantic BaseSettings that auto-loads from .env
        # The env_file setting in ValidationConfig.model_config handles this
        config = ValidationConfig(_env_file='.env')  # Explicit env_file for Pylance

        # Generate embedding for query
        query_vector = generate_query_embedding(query, config)

        # Search Qdrant
        results: List[SearchResult] = search_qdrant(query_vector, config, top_k)

        # Handle zero results
        if not results:
            return "No relevant information found in the knowledge base."

        # Format results for agent context
        formatted_chunks: List[str] = []
        for i, result in enumerate(results, 1):
            chunk = (
                f"[Chunk {i}] (Similarity Score: {result.score:.3f})\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Section: {result.heading}\n"
                f"Content: {result.text}\n"
            )
            formatted_chunks.append(chunk)

        return "\n---\n".join(formatted_chunks)

    except Exception as e:
        # Graceful error handling
        return f"Retrieval service temporarily unavailable: {str(e)}"


# ============================================================================
# Tool Schema Definition
# ============================================================================

# OpenAI function tool schema (per contracts/retrieval-tool.md)
RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_book_content",
        "description": (
            "Retrieves relevant book content chunks from the Physical AI book using semantic search. "
            "Use this tool when the user asks questions about book topics like ROS 2, computer vision, "
            "ML frameworks, or any technical content in the book. Returns top-k most similar chunks "
            "with source metadata."
        ),
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


# ============================================================================
# Agent Initialization
# ============================================================================

# System instructions (FR-006, FR-007, FR-009)
SYSTEM_INSTRUCTIONS = """You are a helpful AI assistant that answers questions about Physical AI using a knowledge base.

IMPORTANT RULES:
1. Answer using ONLY the retrieved book content provided by the retrieve_book_content tool
2. If the tool returns "No relevant information found", inform the user that the information is not available in your knowledge base
3. ALWAYS include source citations in your responses (title and URL) to indicate which chunks informed your answer
4. Do NOT make up information or answer from general knowledge - use only the retrieved content
5. If the user asks a follow-up question, use conversation context to understand references like "it", "that", etc.
"""

# Create OpenAI Assistant with retrieval tool
assistant: Assistant = client.beta.assistants.create(
    name="Physical AI RAG Assistant",
    instructions=SYSTEM_INSTRUCTIONS,
    model="gpt-4o-mini",  # Fast and cost-effective model
    tools=[RETRIEVAL_TOOL]  # type: ignore[list-item]
)

# Create conversation thread for history management (FR-008)
thread = client.beta.threads.create()

print(f"[OK] Agent initialized (Assistant ID: {assistant.id})")
print(f"[OK] Conversation thread created (Thread ID: {thread.id})")


# ============================================================================
# Helper Functions for Type-Safe Content Extraction
# ============================================================================

def extract_text_from_message(message: Message) -> Optional[str]:
    """Safely extract text content from a message, handling all content block types.

    Args:
        message: OpenAI message object

    Returns:
        Extracted text string if available, None otherwise
    """
    if not message.content:
        return None

    for content_block in message.content:
        # Only extract from TextContentBlock - ignore images, refusals, etc.
        if isinstance(content_block, TextContentBlock):
            # TextContentBlock has .text.value - safe to access
            return content_block.text.value
        # Skip non-text content blocks (ImageFileContentBlock, ImageURLContentBlock, RefusalContentBlock)
        # We don't access .text on these types to avoid Pylance errors

    return None


# ============================================================================
# Query Execution
# ============================================================================

def ask(query: str) -> str:
    """Execute a query against the RAG agent.

    Args:
        query: User question

    Returns:
        Agent response text

    Process:
        1. Add user message to thread
        2. Create run to execute assistant
        3. Poll run status and handle tool calls
        4. Submit tool outputs when required
        5. Extract and return final response
    """
    # Add user message to thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )

    # Create run
    run: Run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Poll until completion
    while run.status in ["queued", "in_progress", "requires_action"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        # Handle tool calls with proper type guards
        if run.status == "requires_action":
            required_action = run.required_action

            # Type guard: ensure required_action is not None
            if required_action is None:
                break

            submit_tool_outputs_action = required_action.submit_tool_outputs

            # Type guard: ensure submit_tool_outputs is not None
            if submit_tool_outputs_action is None:
                break

            tool_calls = submit_tool_outputs_action.tool_calls

            # Type guard: ensure tool_calls exists
            if not tool_calls:
                break

            # Build properly typed tool outputs using SDK's ToolOutput class
            tool_outputs: List[ToolOutput] = []

            for tool_call in tool_calls:
                # Type guard: ensure tool_call has function attribute
                if not hasattr(tool_call, 'function'):
                    continue

                if tool_call.function.name == "retrieve_book_content":
                    # Parse arguments
                    args = json.loads(tool_call.function.arguments)

                    # Execute retrieval tool
                    result = retrieve_book_content(**args)

                    # Create properly typed ToolOutput object
                    tool_output = ToolOutput(
                        tool_call_id=tool_call.id,
                        output=result
                    )
                    tool_outputs.append(tool_output)

            # Submit tool outputs if any were collected
            if tool_outputs:
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )

        # Avoid overwhelming API with rapid polls
        time.sleep(1)

    # Check for errors
    if run.status == "failed":
        error_message = "Unknown error"
        if run.last_error is not None:
            error_message = run.last_error.message
        return f"Error: {error_message}"

    # Retrieve final assistant message
    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        limit=1,
        order="desc"
    )

    # Type-safe message content extraction
    if messages.data and len(messages.data) > 0:
        message = messages.data[0]
        text_content = extract_text_from_message(message)

        if text_content is not None:
            return text_content
        else:
            return "Error: No text content in response"

    return "Error: No response from assistant"


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}\n")
        response = ask(query)
        print(f"Agent: {response}\n")
    else:
        # Interactive mode (User Story 3)
        print("\n=== Physical AI RAG Agent ===")
        print("Ask questions about Physical AI book content.")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                query_input = input("You: ").strip()

                if not query_input:
                    continue

                if query_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                response = ask(query_input)
                print(f"\nAgent: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
