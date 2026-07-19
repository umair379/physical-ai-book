# Tasks: RAG Retrieval Validation

**Input**: Design documents from `/specs/009-rag-retrieval/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Organization**: Tasks organized by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3, US4)
- File paths: `backend/` (single project)

---

## Phase 1: Setup

**Purpose**: Project initialization

- [x] T001 Verify Feature 008 infrastructure (192 vectors in Qdrant, valid .env credentials)
- [x] T002 Create backend/test_queries.json with initial test queries (common, edge_cases, adversarial, topic_coverage)

---

## Phase 2: Foundational

**Purpose**: Core script structure (blocks all user stories)

- [x] T003 Create backend/retrieve.py with imports, ValidationConfig class, setup_logging() function
- [x] T004 [P] Add retry_with_backoff decorator (reuse pattern from main.py:498-517)
- [x] T005 [P] Add validate_config() function for .env validation
- [x] T006 Create dataclasses: Query, SearchResult, CollectionMetadata, QueryMetrics, PerformanceMetrics in backend/retrieve.py
- [x] T007 Add argparse CLI interface (--query, --test-suite, --top-k, --verbose) in main() function

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - Verify Vector Retrieval (Priority: P1) 🎯 MVP

**Goal**: Connect to Qdrant, retrieve collection metadata, verify 192 vectors stored

**Independent Test**: Run `python retrieve.py --query "test"`, verify output shows "status: GREEN, points: 192"

### Implementation for User Story 1

- [x] T008 [US1] Implement verify_connection() function in backend/retrieve.py (FR-001, FR-002)
  - Connect with QdrantClient(timeout=120)
  - Call get_collection() for metadata
  - Create CollectionMetadata from result
  - Validate 192 points, 1024 dims, COSINE, GREEN status
  - Log collection info
- [x] T009 [US1] Add verify_connection() call to main() function for connection test
- [x] T010 [US1] Test US1 acceptance scenarios (valid credentials → success, invalid credentials → error)

**Checkpoint**: US1 complete - can connect to Qdrant and verify infrastructure

---

## Phase 4: User Story 2 - Execute Test Queries (Priority: P1)

**Goal**: Run predefined test queries, retrieve top-k results with similarity scores

**Independent Test**: Run `python retrieve.py --query "What is physical AI?"`, verify returns 3 results with scores

### Implementation for User Story 2

- [x] T011 [P] [US2] Implement generate_query_embedding() function in backend/retrieve.py (FR-003)
  - Use Cohere client with input_type='search_query' (CRITICAL)
  - Return 1024-dim embedding
  - Add retry_with_backoff decorator
- [x] T012 [P] [US2] Implement search_qdrant() function in backend/retrieve.py (FR-004)
  - Use client.query_points() API (not deprecated search())
  - Return top-k ScoredPoint list
  - Add retry_with_backoff decorator
- [x] T013 [US2] Implement display_results() function in backend/retrieve.py (FR-005)
  - Show score, title, heading, URL, chunk index, text preview
  - Format output for console readability
- [x] T014 [US2] Add single query execution flow to main() function
  - Call generate_query_embedding() → search_qdrant() → display_results()
  - Log query text, result count, best score (FR-007)
- [x] T015 [US2] Test US2 acceptance scenarios (query returns results, logs written, module-specific queries)

**Checkpoint**: US1 + US2 complete - can execute queries and see results

---

## Phase 5: User Story 3 - Validate Result Quality (Priority: P2)

**Goal**: Inspect metadata fields, validate completeness

**Independent Test**: Run query, verify output shows "✅ All results have complete metadata"

### Implementation for User Story 3

- [ ] T016 [US3] Implement validate_result_metadata() function in backend/retrieve.py (FR-009)
  - Check 7 required fields: chunk_id, text, url, title, heading, chunk_index, timestamp
  - Return (is_valid, missing_fields)
- [ ] T017 [US3] Add metadata validation to display_results() function (SC-004)
  - Call validate_result_metadata() for each result
  - Log "✅ All results have complete metadata" or errors
- [ ] T018 [US3] Test US3 acceptance scenarios (all fields present, heading hierarchy correct)

**Checkpoint**: US1 + US2 + US3 complete - metadata validation working

---

## Phase 6: User Story 4 - Performance Validation (Priority: P3)

**Goal**: Measure query latency, establish performance baselines

**Independent Test**: Run test suite, verify output shows "avg latency <3000ms"

### Implementation for User Story 4

- [ ] T019 [US4] Implement execute_query_with_metrics() function in backend/retrieve.py (FR-010)
  - Use time.perf_counter() for embedding phase
  - Use time.perf_counter() for search phase
  - Create QueryMetrics with timings
  - Add to PerformanceMetrics aggregate
- [ ] T020 [US4] Add PerformanceMetrics.get_summary() method (calculate p50, p95, p99)
- [ ] T021 [US4] Add PerformanceMetrics.validate_performance() method (check SC-005 thresholds)
- [ ] T022 [US4] Implement load_test_suite() function in backend/retrieve.py (FR-006)
  - Load test_queries.json
  - Parse categories and query metadata
- [ ] T023 [US4] Implement run_test_suite() function in backend/retrieve.py (FR-006)
  - Iterate through test categories
  - Execute queries with execute_query_with_metrics()
  - Track passed/failed count
  - Display performance summary
- [ ] T024 [US4] Add test suite execution flow to main() function (--test-suite arg)
- [ ] T025 [US4] Test US4 acceptance scenarios (batch completes <30s, latency breakdown logged)

**Checkpoint**: All user stories complete - full validation script functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, edge cases, final validation

- [ ] T026 [P] Add error handling for edge cases (empty query, empty collection, rate limits, network failures)
- [ ] T027 [P] Expand backend/test_queries.json with comprehensive test cases (5+ queries per category)
- [ ] T028 Run all quickstart.md validation scenarios (SC-001 through SC-007)
- [ ] T029 Create validation log examples in backend/ for documentation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational completion
  - US1 (P1): Infrastructure verification - foundation for all queries
  - US2 (P1): Query execution - depends on US1 connection
  - US3 (P2): Metadata validation - enhances US2 display
  - US4 (P3): Performance tracking - wraps US2 query execution
- **Polish (Phase 7)**: Depends on all user stories

### Sequential Order (Recommended)

1. Phase 1 (Setup) → Phase 2 (Foundational) → US1 → US2 → US3 → US4 → Polish
2. Reason: Each story builds on previous (US2 needs US1 connection, US3 enhances US2 display, US4 wraps US2 queries)

### Parallel Opportunities

**Within Phase 2 (Foundational)**:
- T004 (retry decorator) + T005 (config validation) can run in parallel

**Within Phase 4 (US2)**:
- T011 (generate_query_embedding) + T012 (search_qdrant) can run in parallel

**Within Phase 7 (Polish)**:
- T026 (error handling) + T027 (test expansion) can run in parallel

---

## Parallel Example: User Story 2

```bash
# Launch core functions together:
Task T011: "Implement generate_query_embedding() in backend/retrieve.py"
Task T012: "Implement search_qdrant() in backend/retrieve.py"

# Then integrate sequentially:
Task T013: "Implement display_results()" (needs T012 results)
Task T014: "Add query flow to main()" (needs T011+T012+T013)
```

---

## Implementation Strategy

### MVP First (US1 + US2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (verify connection)
4. Complete Phase 4: User Story 2 (execute queries)
5. **STOP and VALIDATE**: Test with `python retrieve.py --query "What is physical AI?"`
6. Delivers core retrieval validation

### Incremental Delivery

1. Setup + Foundational → Script skeleton ready
2. + US1 → Can verify Qdrant connection (infrastructure check)
3. + US2 → Can run queries and see results (core validation)
4. + US3 → Can validate metadata completeness (quality check)
5. + US4 → Can measure performance baselines (optimization ready)
6. Each story adds value without breaking previous stories

---

## Task Summary

- **Total Tasks**: 29
- **Setup**: 2 tasks
- **Foundational**: 5 tasks
- **User Story 1** (P1): 3 tasks
- **User Story 2** (P1): 5 tasks
- **User Story 3** (P2): 3 tasks
- **User Story 4** (P3): 7 tasks
- **Polish**: 4 tasks

**Parallel Opportunities**: 5 task pairs can run in parallel (marked with [P])

**MVP Scope**: Phase 1 + Phase 2 + US1 + US2 (15 tasks total)

**Estimated Lines**: ~250 lines backend/retrieve.py + ~100 lines backend/test_queries.json

---

## Notes

- Single file implementation: All code in backend/retrieve.py (except test_queries.json)
- Reuse patterns from backend/main.py (Feature 008): retry decorator, Config class, logging
- Critical: Use input_type='search_query' for Cohere (not 'search_document')
- Critical: Use timeout=120 for QdrantClient (Qdrant Free Tier requirement)
- Critical: Use client.query_points() API (search() is deprecated)
- Each user story independently testable via quickstart.md scenarios
- Commit after each task or logical group
