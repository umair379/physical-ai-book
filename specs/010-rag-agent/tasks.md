# Tasks: AI Agent with Retrieval-Augmented Capabilities

**Input**: Design documents from `/specs/010-rag-agent/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/, quickstart.md

**Tests**: NOT requested in feature specification - manual testing only

**Organization**: Tasks grouped by user story for independent implementation and testing

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup

**Purpose**: Project initialization and dependency installation

- [X] T001 Add openai dependency to backend/pyproject.toml (uv add openai)
- [X] T002 Add OPENAI_API_KEY to backend/.env
- [X] T003 [P] Add OPENAI_API_KEY placeholder to backend/.env.example

---

## Phase 2: Foundational

**Purpose**: No blocking prerequisites - proceed directly to user stories

**Checkpoint**: Foundation ready (existing retrieval pipeline from Feature 009 is sufficient)

---

## Phase 3: User Story 1 - Agent Initialization and Tool Setup (Priority: P1) 🎯 MVP

**Goal**: Create agent instance with retrieval tool that successfully queries Qdrant

**Independent Test**: Initialize agent, register tool, execute test query "What is physical AI?" - verify chunks returned from Qdrant with source citations

### Implementation for User Story 1

- [X] T004 [US1] Create agent.py at project root with OpenAI client initialization
- [X] T005 [US1] Implement retrieve_book_content() function in agent.py bridging to backend/retrieve.py
- [X] T006 [US1] Define retrieval tool JSON schema in agent.py per contracts/retrieval-tool.md
- [X] T007 [US1] Create OpenAI Assistant with retrieval tool registration in agent.py
- [X] T008 [US1] Create OpenAI Thread for conversation management in agent.py
- [X] T009 [US1] Add system instructions: "Answer using ONLY retrieved book content, include citations, inform when unavailable"

**Checkpoint**: Agent initialized, tool registered, can execute test query successfully

---

## Phase 4: User Story 2 - Query Answering with Retrieved Context (Priority: P1)

**Goal**: Agent answers questions using only retrieved chunks without hallucination

**Independent Test**: Ask "What is physical AI?" - verify answer references book content only and includes source citation

### Implementation for User Story 2

- [X] T010 [US2] Implement ask() function in agent.py for query execution
- [X] T011 [US2] Add tool invocation loop: poll run status, handle requires_action, submit tool outputs
- [X] T012 [US2] Add zero-result handling: return "No relevant information found" when chunks empty
- [X] T013 [US2] Format tool output with chunk text, score, title, URL, heading metadata
- [X] T014 [US2] Extract and return assistant message from thread after run completion

**Checkpoint**: Agent answers questions grounded in retrieved content with citations

---

## Phase 5: User Story 3 - Follow-up Query Handling (Priority: P2)

**Goal**: Agent maintains conversation context for follow-up queries

**Independent Test**: Ask "What is physical AI?" then "What are its applications?" - verify agent understands context

### Implementation for User Story 3

- [X] T015 [US3] Verify OpenAI Thread automatically maintains conversation history (no code changes needed)
- [X] T016 [US3] Add CLI interface for multi-turn conversation in agent.py (interactive mode)
- [X] T017 [US3] Test 3-message conversation flow: initial + 2 follow-ups

**Checkpoint**: Agent handles follow-up queries using conversation history

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Edge case handling and developer experience improvements

- [X] T018 [P] Add error handling for Cohere/Qdrant API failures in retrieve_book_content()
- [X] T019 [P] Add error handling for OpenAI API failures in ask() function
- [X] T020 [P] Add CLI argument parsing for single-query mode in agent.py
- [X] T021 Validate against quickstart.md: test 5 questions from spec.md (SC-002)
- [X] T022 Validate against quickstart.md: test 3 adversarial questions (SC-003)
- [X] T023 Verify response time <10 seconds with timing instrumentation (SC-005)
- [X] T024 Verify source citations present in 80%+ responses (SC-006)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup - No blocking tasks (existing retrieval pipeline sufficient)
- **User Story 1 (Phase 3)**: Depends on Setup - Core agent initialization
- **User Story 2 (Phase 4)**: Depends on User Story 1 - Query execution requires initialized agent
- **User Story 3 (Phase 5)**: Depends on User Story 2 - Follow-ups require working Q&A
- **Polish (Phase 6)**: Depends on all user stories - Final validation

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Setup (Phase 1) - No dependencies on other stories
- **User Story 2 (P1)**: Depends on User Story 1 (T004-T009) - Requires agent initialization
- **User Story 3 (P2)**: Depends on User Story 2 (T010-T014) - Requires working query execution

### Within Each User Story

- T004-T009 (US1): T004 first (client init), then T005-T009 can proceed in sequence
- T010-T014 (US2): T010 first (ask function), then T011-T014 in sequence
- T015-T017 (US3): Sequential verification tasks

### Parallel Opportunities

- **Phase 1**: T002 and T003 can run in parallel (different files)
- **Phase 6**: T018, T019, T020 can run in parallel (different concerns in same file)
- **Limited parallelism**: Single-file implementation (agent.py) limits parallel execution

---

## Parallel Example: Setup Phase

```bash
# Launch setup tasks in parallel:
Task T002: "Add OPENAI_API_KEY to backend/.env"
Task T003: "Add OPENAI_API_KEY placeholder to backend/.env.example"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup (T001-T003)
2. Skip Phase 2: No foundational blockers
3. Complete Phase 3: User Story 1 (T004-T009)
4. **VALIDATE**: Test agent initialization and tool execution
5. Complete Phase 4: User Story 2 (T010-T014)
6. **VALIDATE**: Test query answering with citations
7. Deploy/demo MVP (Q&A working, no follow-ups yet)

### Full Feature Delivery

1. Complete MVP (Phases 1, 3, 4)
2. Add Phase 5: User Story 3 (T015-T017) for follow-up support
3. Add Phase 6: Polish (T018-T024) for production readiness
4. Validate all success criteria (SC-001 through SC-007)

### Critical Path (Sequential)

Setup (T001) → US1 Core (T004-T009) → US2 Core (T010-T014) → US3 (T015-T017) → Polish (T021-T024)

**Estimated total**: ~12 tasks for MVP, ~24 tasks for full feature

---

## Notes

- Single-file implementation (agent.py) at project root - limited parallelism
- No tests requested - validation via quickstart.md manual scenarios
- Reuses existing retrieval pipeline (backend/retrieve.py) - no modifications needed
- OpenAI Thread handles conversation history automatically - minimal code for US3
- Success criteria validation in Phase 6 Polish tasks (T021-T024)
- Target: <20 lines for agent setup (SC-001), <10s response time (SC-005)
