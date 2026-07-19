/**
 * TypeScript Type Definitions for Physical AI RAG Backend API
 *
 * Generated from OpenAPI specification for Feature 011: FastAPI Backend Integration
 * Use these types in Docusaurus chatbot components for type-safe API communication
 */

/**
 * Request body for POST /api/query
 */
export interface QueryRequest {
  /**
   * User's question about Physical AI book content
   * @minLength 1
   * @maxLength 10000
   */
  query: string;

  /**
   * OpenAI thread ID for conversation history (omit for new conversation)
   * @format uuid
   */
  thread_id?: string;

  /**
   * Number of relevant chunks to retrieve
   * @minimum 1
   * @maximum 10
   * @default 3
   */
  top_k?: number;
}

/**
 * Response body for successful POST /api/query
 */
export interface QueryResponse {
  /**
   * AI-generated answer grounded in book content
   */
  answer: string;

  /**
   * List of source citations (empty array if no relevant content found)
   */
  sources: Source[];

  /**
   * OpenAI thread ID for this conversation (use for follow-up queries)
   */
  thread_id: string;

  /**
   * Time taken to generate response (milliseconds)
   */
  response_time_ms: number;

  /**
   * Unique request identifier for tracing
   */
  request_id: string;
}

/**
 * Source citation from RAG retrieval
 */
export interface Source {
  /**
   * Title of the source document/page
   */
  title: string;

  /**
   * Full URL to the source page
   * @format uri
   */
  url: string;

  /**
   * Semantic similarity score (0.0 - 1.0)
   * @minimum 0.0
   * @maximum 1.0
   */
  score: number;

  /**
   * Index of the retrieved chunk within the source document
   * @minimum 0
   */
  chunk_index: number;
}

/**
 * Error response for failed requests
 */
export interface ErrorResponse {
  /**
   * Machine-readable error code
   */
  error:
    | 'invalid_query'
    | 'query_too_long'
    | 'validation_error'
    | 'assistant_execution_error'
    | 'retrieval_service_error'
    | 'embedding_generation_error'
    | 'internal_server_error';

  /**
   * Human-readable error message
   */
  message: string;

  /**
   * Unique request identifier for tracing
   */
  request_id: string;

  /**
   * Additional debugging information (omitted in production)
   */
  details?: Record<string, unknown>;
}

/**
 * Health check response from GET /health
 */
export interface HealthResponse {
  /**
   * Overall service status
   */
  status: 'healthy' | 'degraded';

  /**
   * Whether RAG agent is initialized and ready
   */
  agent_ready: boolean;

  /**
   * ISO 8601 timestamp of health check
   * @format date-time
   */
  timestamp: string;

  /**
   * API version string
   */
  version: string;
}

/**
 * HTTP status codes returned by the API
 */
export enum HttpStatus {
  OK = 200,
  BAD_REQUEST = 400,
  UNPROCESSABLE_ENTITY = 422,
  INTERNAL_SERVER_ERROR = 500,
  BAD_GATEWAY = 502,
}

/**
 * API client configuration
 */
export interface APIConfig {
  /**
   * Base URL for the API
   * @default "http://localhost:8000"
   */
  baseUrl: string;

  /**
   * Request timeout in milliseconds
   * @default 30000
   */
  timeout?: number;
}

/**
 * Chat message for frontend state management
 */
export interface ChatMessage {
  /**
   * Message ID (generated client-side)
   */
  id: string;

  /**
   * Message role
   */
  role: 'user' | 'assistant';

  /**
   * Message content
   */
  content: string;

  /**
   * Source citations (only for assistant messages)
   */
  sources?: Source[];

  /**
   * Timestamp when message was created
   */
  timestamp: Date;

  /**
   * Whether this message is currently loading
   */
  isLoading?: boolean;

  /**
   * Error message if request failed
   */
  error?: string;
}

/**
 * Chat state for React Context
 */
export interface ChatState {
  /**
   * Conversation messages
   */
  messages: ChatMessage[];

  /**
   * OpenAI thread ID for conversation history
   */
  threadId: string | null;

  /**
   * Whether a request is in progress
   */
  isLoading: boolean;

  /**
   * Current error message
   */
  error: string | null;

  /**
   * Whether chat window is expanded
   */
  isExpanded: boolean;
}

/**
 * Actions for chat state reducer
 */
export type ChatAction =
  | { type: 'ADD_USER_MESSAGE'; payload: string }
  | { type: 'ADD_ASSISTANT_MESSAGE'; payload: { answer: string; sources: Source[]; threadId: string } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'CLEAR_ERROR' }
  | { type: 'TOGGLE_EXPANDED' }
  | { type: 'RESET_CONVERSATION' };

/**
 * API client methods
 */
export interface APIClient {
  /**
   * Submit a query to the RAG agent
   */
  submitQuery(request: QueryRequest): Promise<QueryResponse>;

  /**
   * Check service health
   */
  healthCheck(): Promise<HealthResponse>;
}

/**
 * Hook return type for useChatAPI
 */
export interface UseChatAPI {
  /**
   * Submit a query and update chat state
   */
  sendMessage(query: string): Promise<void>;

  /**
   * Check if backend is healthy
   */
  checkHealth(): Promise<boolean>;

  /**
   * Current loading state
   */
  isLoading: boolean;

  /**
   * Current error state
   */
  error: string | null;
}
