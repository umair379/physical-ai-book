# Tasks: FastAPI Backend Integration for RAG System

**Feature**: 011-fastapi-integration
**Input**: Design documents from `/specs/011-fastapi-integration/`
**Prerequisites**: spec.md (user stories), research.md (architectural decisions), data-model.md (API entities), contracts/ (OpenAPI spec), quickstart.md (MVP reference)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This feature uses **web app** structure:
- **Backend**: `backend/` (Python/FastAPI)
- **Frontend**: `frontend-book/src/` (Docusaurus/React)
- **Specs**: `specs/011-fastapi-integration/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency installation

- [X] T001 Install FastAPI and Uvicorn in backend/pyproject.toml using `uv add fastapi uvicorn[standard]`
- [X] T002 [P] Create backend/api.py with FastAPI app initialization and basic health endpoint
- [X] T003 [P] Configure CORS middleware in backend/api.py per research.md recommendations (localhost:3000 origin)
- [X] T004 [P] Set up logging configuration in backend/api.py with request ID tracking

**Checkpoint**: Basic FastAPI server runs on localhost:8000 with CORS enabled

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create Pydantic request/response models in backend/api.py: QueryRequest, QueryResponse, Source, ErrorResponse, HealthResponse (based on data-model.md)
- [X] T006 [P] Define custom exception classes in backend/api.py: InvalidQueryError, AssistantExecutionError, RetrievalServiceError (based on research.md error handling strategy)
- [X] T007 [P] Implement global exception handlers in backend/api.py for each custom exception type (400, 422, 500, 502 status codes)
- [X] T008 [P] Add request ID middleware in backend/api.py to generate unique request_id for each incoming request
- [X] T009 Verify backend/agent.py from Feature 010 is importable and functional (run `uv run python backend/agent.py "test"`)

**Checkpoint**: Foundation ready - FastAPI server has all models, error handling, and agent integration verified

---

## Phase 3: User Story 1 - Query Submission via API (Priority: P1) 🎯 MVP

**Goal**: Enable frontend to send queries and receive AI-generated answers with source citations

**Independent Test**: Send POST request to /api/query with `{"query": "What is physical AI?"}` and verify response contains answer + sources + thread_id

### Implementation for User Story 1

- [X] T010 [US1] Implement POST /api/query endpoint handler in backend/api.py that accepts QueryRequest
- [X] T011 [US1] Add query validation in POST /api/query: check non-empty, max length 10,000, whitespace-only rejection
- [X] T012 [US1] Integrate backend/agent.py ask() function call within POST /api/query endpoint
- [X] T013 [US1] Parse agent response and extract answer text, thread_id from backend/agent.py output
- [X] T014 [US1] Extract source citations from retrieval results and map to Source[] schema (title, url, score, chunk_index)
- [X] T015 [US1] Calculate response_time_ms and build QueryResponse with all required fields
- [X] T016 [US1] Add error handling for agent failures: wrap agent.ask() in try/except, raise AssistantExecutionError on failure
- [X] T017 [US1] Add logging for query processing: log query length, response time, success/failure with request_id

**Verification**:
- [ ] V001 [US1] Test with curl: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query": "What is physical AI?"}'`
- [ ] V002 [US1] Verify response has answer, sources array, thread_id, response_time_ms, request_id fields
- [ ] V003 [US1] Test query with no results: `{"query": "How do I bake cookies?"}` returns empty sources array

**Checkpoint**: POST /api/query endpoint works end-to-end with RAG agent from Feature 010

---

## Phase 4: User Story 2 - Error Handling and Status Reporting (Priority: P2)

**Goal**: Return structured error responses with appropriate HTTP status codes for validation and server errors

**Independent Test**: Send malformed requests and verify HTTP status codes (400, 422, 500) and error message structure

### Implementation for User Story 2

- [ ] T018 [US2] Test Pydantic validation error handling: send request with missing 'query' field, verify HTTP 422 response
- [ ] T019 [US2] Implement InvalidQueryError handler for empty/whitespace queries: return HTTP 400 with error code "invalid_query"
- [ ] T020 [US2] Implement InvalidQueryError handler for oversized queries (>10,000 chars): return HTTP 400 with error code "query_too_long"
- [ ] T021 [US2] Test AssistantExecutionError handling: simulate OpenAI API failure, verify HTTP 500 response with sanitized error message
- [ ] T022 [US2] Add detailed internal logging for all errors: log full exception stack trace with request_id, but return user-friendly message to client
- [ ] T023 [US2] Implement catch-all exception handler for unexpected errors: return HTTP 500 with error code "internal_server_error"

**Verification**:
- [ ] V004 [US2] Test validation error: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{}'` returns HTTP 422
- [ ] V005 [US2] Test empty query: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query": ""}'` returns HTTP 400
- [ ] V006 [US2] Verify error response schema: check error, message, request_id fields are present

**Checkpoint**: All error scenarios return structured ErrorResponse with correct HTTP status codes

---

## Phase 5: User Story 3 - Health Check and Service Monitoring (Priority: P3)

**Goal**: Provide health check endpoint for frontend circuit breakers and monitoring tools

**Independent Test**: Send GET request to /health and verify HTTP 200 with service status information

### Implementation for User Story 3

- [ ] T024 [US3] Implement GET /health endpoint in backend/api.py returning HealthResponse schema
- [ ] T025 [US3] Add agent readiness check in /health: verify backend/agent.py module is importable and agent initialized
- [ ] T026 [US3] Add timestamp field to /health response: return current ISO 8601 timestamp
- [ ] T027 [US3] Add version field to /health response: hardcode "1.0.0" or read from environment variable
- [ ] T028 [US3] Set status field logic: "healthy" if agent ready, "degraded" if agent initialization failed

**Verification**:
- [ ] V007 [US3] Test health endpoint: `curl http://localhost:8000/health` returns HTTP 200
- [ ] V008 [US3] Verify health response schema: check status, agent_ready, timestamp, version fields
- [ ] V009 [US3] Verify response time: health check should respond within 1 second

**Checkpoint**: Health endpoint provides accurate service status for monitoring

---

## Phase 6: Frontend Chatbot Integration

**Goal**: Add persistent chatbot UI to Docusaurus site that calls the FastAPI backend

**Independent Test**: Open http://localhost:3000, click chat button, send query, verify response appears

### Frontend Setup

- [ ] T029 Swizzle Docusaurus Root component: run `cd frontend-book && npm run swizzle @docusaurus/theme-classic Root -- --wrap`
- [ ] T030 [P] Create frontend-book/src/components/Chatbot/index.tsx with ChatBot component skeleton
- [ ] T031 [P] Create frontend-book/src/components/Chatbot/styles.module.css with fixed-position container styles (z-index: 9999, bottom-right placement)
- [ ] T032 [P] Create frontend-book/src/components/Chatbot/types.ts with TypeScript interfaces from contracts/frontend-types.ts

### Chatbot State Management

- [ ] T033 Create frontend-book/src/components/Chatbot/ChatContext.tsx with React Context provider and useReducer for chat state
- [ ] T034 Define ChatState interface in ChatContext: messages[], thread_id, isLoading, error, isExpanded
- [ ] T035 Define ChatAction types in ChatContext: ADD_USER_MESSAGE, ADD_ASSISTANT_MESSAGE, SET_LOADING, SET_ERROR, TOGGLE_EXPANDED, RESET_CONVERSATION
- [ ] T036 Implement chatReducer function in ChatContext handling all action types

### Chatbot UI Components

- [ ] T037 [P] Create frontend-book/src/components/Chatbot/ChatToggle.tsx: floating button to open/close chat
- [ ] T038 [P] Create frontend-book/src/components/Chatbot/ChatWindow.tsx: container with header, messages area, input area
- [ ] T039 [P] Create frontend-book/src/components/Chatbot/ChatMessage.tsx: individual message display (user vs assistant styling)
- [ ] T040 [P] Create frontend-book/src/components/Chatbot/ChatCitation.tsx: inline citation badge `[1]` with expandable details
- [ ] T041 [P] Create frontend-book/src/components/Chatbot/ChatInput.tsx: text input with send button and Enter key handler

### API Integration

- [ ] T042 Create frontend-book/src/components/Chatbot/useChatAPI.ts custom hook with sendMessage function
- [ ] T043 Implement fetch POST request to http://localhost:8000/api/query in useChatAPI
- [ ] T044 Add error handling in useChatAPI: catch network errors, parse ErrorResponse, update chat state with error message
- [ ] T045 Add loading state management in useChatAPI: set isLoading before request, clear after response

### Integration and Polish

- [ ] T046 Update frontend-book/src/theme/Root.tsx to wrap {children} with ChatProvider and render Chatbot component
- [ ] T047 Add CSS for mobile responsiveness in styles.module.css: adjust width/height for screens < 768px
- [ ] T048 Add keyboard accessibility: Tab navigation, Enter to send, Escape to close
- [ ] T049 Add ARIA labels for screen readers on all interactive elements

**Verification**:
- [ ] V010 Frontend visual test: Chat toggle button appears in bottom-right corner on all pages
- [ ] V011 Frontend interaction test: Click toggle, type query, press Enter, verify assistant response appears
- [ ] V012 Frontend citation test: Verify citation badges `[1] [2]` appear inline with answer, click to expand details
- [ ] V013 Frontend error test: Stop backend server, send query, verify error message displays in chat
- [ ] V014 Mobile test: Open on mobile viewport (375px width), verify chat window resizes appropriately

**Checkpoint**: Full end-to-end integration: frontend chatbot → FastAPI → RAG agent → response display with citations

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Production readiness improvements and documentation

- [ ] T050 [P] Add API endpoint documentation: ensure OpenAPI docs are auto-generated at http://localhost:8000/docs
- [ ] T051 [P] Add request/response examples to FastAPI endpoint docstrings
- [ ] T052 [P] Configure logging format: structured JSON logs with timestamp, level, message, request_id
- [ ] T053 [P] Add environment variable documentation in backend/.env.example for ENVIRONMENT, ALLOWED_ORIGINS
- [ ] T054 Create integration test script: backend/test_integration.sh that starts server, runs curl tests, verifies responses
- [ ] T055 Document deployment instructions in specs/011-fastapi-integration/deployment.md (Vercel frontend + Railway/Render backend)
- [ ] T056 Add conversation history persistence: save thread_id in frontend localStorage, restore on page reload
- [ ] T057 Add loading animations: skeleton loader while waiting for agent response
- [ ] T058 Add rate limiting documentation: note future enhancement for production (not implemented in MVP)

**Checkpoint**: Feature ready for production deployment with documentation

---

## Dependencies & Execution Order

### User Story Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundation)
                     ↓
    ┌────────────────┼────────────────┐
    ↓                ↓                ↓
Phase 3 (US1)   Phase 4 (US2)   Phase 5 (US3)
    ↓                ↓                ↓
    └────────────────┼────────────────┘
                     ↓
              Phase 6 (Frontend)
                     ↓
              Phase 7 (Polish)
```

**Critical Path**: Phase 1 → Phase 2 → Phase 3 (US1) → Phase 6 (Frontend) = MVP

**Independent Stories**:
- User Story 2 (Error Handling) can be implemented in parallel with US1 after Phase 2
- User Story 3 (Health Check) can be implemented in parallel with US1/US2 after Phase 2
- Frontend integration (Phase 6) depends on US1 being complete

### Parallel Execution Opportunities

**After Phase 2 completes, these can run in parallel**:
- T010-T017 (US1 implementation)
- T018-T023 (US2 implementation)
- T024-T028 (US3 implementation)

**Frontend components can be built in parallel**:
- T029-T032 (Setup)
- T037-T041 (UI Components) can all be built simultaneously
- T042-T045 (API integration) depends on T010-T017 (US1) being complete

---

## Implementation Strategy

### MVP (Minimum Viable Product)

**Scope**: Phase 1 + Phase 2 + Phase 3 (US1) + Phase 6 (Frontend basic integration)
**Estimated tasks**: 29 tasks (T001-T049, excluding US2, US3, and polish)
**Deliverable**: Working chatbot UI that can send queries and display answers with citations

**MVP Test**:
1. Start backend: `cd backend && uv run python api.py`
2. Start frontend: `cd frontend-book && npm start`
3. Open http://localhost:3000
4. Click chat toggle, type "What is physical AI?", press Enter
5. Verify answer appears with citation links

### Incremental Delivery

**Release 1 (MVP)**: US1 + Basic Frontend (Phases 1-3, 6 subset)
- Core query/response functionality
- Basic chat UI
- No error handling sophistication
- No health monitoring

**Release 2**: + US2 (Error Handling) (Phase 4)
- Structured error responses
- Better debugging experience
- User-friendly error messages

**Release 3**: + US3 (Health Check) + Polish (Phases 5, 7)
- Service monitoring
- Production-ready logging
- Documentation complete
- Deployment ready

---

## Task Validation

### Format Compliance
✅ All tasks follow `- [ ] [ID] [P?] [Story?] Description with file path` format
✅ Sequential task IDs (T001-T058)
✅ Story labels present for US1, US2, US3 phases
✅ [P] markers for parallelizable tasks
✅ File paths included in all implementation tasks

### Completeness Check
✅ Setup phase: Dependencies and basic structure
✅ Foundational phase: Models, exceptions, middleware
✅ User Story 1: Full query/response pipeline
✅ User Story 2: Comprehensive error handling
✅ User Story 3: Health check endpoint
✅ Frontend: Complete chatbot UI with API integration
✅ Polish: Documentation, testing, production readiness

### Independent Test Criteria
✅ US1: curl test for POST /api/query returns answer + sources
✅ US2: curl tests for error scenarios return correct status codes
✅ US3: curl test for GET /health returns service status
✅ Frontend: Browser test shows working chat interface

---

## Summary

**Total Tasks**: 58 implementation tasks + 14 verification tasks
**MVP Tasks**: 29 (50% of total)
**Parallel Tasks**: 24 tasks marked with [P]
**User Stories**: 3 (P1: Query API, P2: Error Handling, P3: Health Check)
**Phases**: 7 (Setup → Foundation → 3 User Stories → Frontend → Polish)

**Critical Path to MVP**:
Phase 1 (4 tasks) → Phase 2 (5 tasks) → Phase 3/US1 (8 tasks) → Phase 6/Frontend (21 tasks) = **38 tasks**

**Estimated Implementation Time**:
- MVP: 1-2 days (with all infrastructure from Feature 010 working)
- Full feature: 2-3 days (including error handling, health check, polish)

**Next Step**: Run `/sp.implement` to execute tasks in dependency order
