# Tasks: RAG Embeddings Ingestion Pipeline

**Input**: Design documents from `/specs/008-rag-embeddings/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/functions, no dependencies)
- **[Story]**: User story label (US1, US2, US3)
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: Initialize Python project with uv and create configuration templates

- [X] T001 Create backend/ directory structure
- [X] T002 Initialize uv project in backend/ with pyproject.toml
- [X] T003 [P] Add dependencies via uv: requests, beautifulsoup4, lxml, tiktoken, cohere, qdrant-client, python-dotenv, pydantic
- [X] T004 [P] Create .env.example with required environment variables in backend/
- [X] T005 [P] Create .gitignore for backend/.env and backend/.venv

**Checkpoint**: Project structure ready, dependencies installed

---

## Phase 2: Foundational

**Purpose**: Core configuration and data structures

- [X] T006 Define Config dataclass with pydantic validation in backend/main.py
- [X] T007 [P] Define DocumentPage, TextChunk, VectorEmbedding, IngestionStats dataclasses in backend/main.py
- [X] T008 Implement config validation and startup checks in backend/main.py
- [X] T009 [P] Setup logging configuration (file + console) in backend/main.py

**Checkpoint**: Configuration loaded, logging active

---

## Phase 3: User Story 1 - Crawl and Extract (Priority: P1) 🎯 MVP

**Goal**: Discover URLs, fetch pages, extract clean text content

**Independent Test**: Run crawler against Vercel URL, verify all pages discovered, HTML stripped, metadata captured

### Implementation for User Story 1

- [X] T010 [P] [US1] Implement fetch_sitemap(base_url) → List[str] in backend/main.py
- [X] T011 [P] [US1] Implement recursive_crawl(start_url, max_depth) fallback in backend/main.py
- [X] T012 [US1] Implement fetch_page(url) with timeout and error handling in backend/main.py
- [X] T013 [US1] Implement extract_content(html) using BeautifulSoup in backend/main.py
- [X] T014 [US1] Implement extract_metadata(html) for title and breadcrumb in backend/main.py
- [X] T015 [US1] Implement crawl_pages(urls) orchestrating fetch + extract in backend/main.py
- [X] T016 [US1] Add progress logging (every 10 pages) and error handling in crawl_pages()

**Checkpoint**: Crawling complete, DocumentPage objects created with clean text

---

## Phase 4: User Story 2 - Chunk Text (Priority: P2)

**Goal**: Split text into semantic chunks with heading awareness

**Independent Test**: Provide sample text, verify chunks respect heading boundaries, token counts within target range

### Implementation for User Story 2

- [X] T017 [P] [US2] Implement count_tokens(text) using tiktoken in backend/main.py
- [X] T018 [US2] Implement detect_headings(lines) to extract markdown headings in backend/main.py
- [X] T019 [US2] Implement chunk_text(text, metadata, target_size=512) with heading preservation in backend/main.py
- [X] T020 [US2] Implement chunk validation (min 50 tokens, max max_chunk_size) in chunk_text()
- [X] T021 [US2] Add chunk_all_pages(pages) to process all DocumentPages in backend/main.py

**Checkpoint**: Chunking complete, TextChunk objects created with metadata

---

## Phase 5: User Story 3 - Embed and Store (Priority: P3)

**Goal**: Generate embeddings via Cohere, store vectors in Qdrant with upsert logic

**Independent Test**: Embed sample chunks, store in Qdrant, run semantic search queries, verify similarity scores >0.7

### Implementation for User Story 3

- [X] T022 [P] [US3] Implement generate_embeddings(chunks, batch_size=96) with Cohere API in backend/main.py
- [X] T023 [P] [US3] Implement retry_with_backoff() decorator for API failures in backend/main.py
- [X] T024 [US3] Add embedding dimension validation (1024) in generate_embeddings()
- [X] T025 [US3] Implement create_qdrant_collection(collection_name, dimension=1024) in backend/main.py
- [X] T026 [US3] Implement store_in_qdrant(embeddings, collection_name) with upsert in backend/main.py
- [X] T027 [US3] Implement validate_search(test_queries, collection_name) for semantic search in backend/main.py

**Checkpoint**: Embeddings generated, vectors stored, search validated

---

## Phase 6: Integration & Orchestration

**Purpose**: Wire all components together, add main() entry point

- [X] T028 Implement main() function orchestrating full pipeline in backend/main.py
- [X] T029 Add command-line argument parsing (optional --config flag) in main()
- [X] T030 Add final statistics report and logging in main()
- [X] T031 [P] Create README.md with setup instructions in backend/

**Checkpoint**: Single-command execution working end-to-end

---

## Phase 7: Documentation & Validation

**Purpose**: Verify against success criteria from spec.md

- [X] T032 Run full pipeline against test Docusaurus site
- [X] T033 Verify SC-001: 100% of pages crawled (check stats.success_rate)
- [X] T034 Verify SC-003: Chunk token counts within 10% of target (check chunk stats)
- [X] T035 Verify SC-008: Test queries return similarity >0.7 (check search results)
- [X] T036 [P] Update quickstart.md with actual execution outputs if needed

**Checkpoint**: All success criteria validated

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies
- **Phase 2 (Foundational)**: Depends on Phase 1
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 2 (independent of US1)
- **Phase 5 (US3)**: Depends on Phase 2 (independent of US1, US2)
- **Phase 6 (Integration)**: Depends on Phase 3, 4, 5 all complete
- **Phase 7 (Validation)**: Depends on Phase 6

### User Story Dependencies

- **US1 (Crawl)**: Independent - can test with sample URLs
- **US2 (Chunk)**: Independent - can test with sample text
- **US3 (Embed/Store)**: Independent - can test with sample chunks

**All user stories (US1, US2, US3) can be implemented in parallel after Phase 2 completes**

### Parallel Opportunities

**Phase 1 Setup**:
```
T003 [Dependencies] || T004 [.env.example] || T005 [.gitignore]
```

**Phase 2 Foundational**:
```
T007 [Dataclasses] || T009 [Logging]
```

**Phase 3 (US1)**:
```
T010 [fetch_sitemap] || T011 [recursive_crawl]
```

**Phase 4 (US2)**:
```
T017 [count_tokens] runs alone (used by T019)
```

**Phase 5 (US3)**:
```
T022 [generate_embeddings] || T023 [retry_with_backoff]
```

**Phase 7 Validation**:
```
T033 || T034 || T035 || T036 (all validation checks can run in parallel)
```

**User Stories (after Phase 2)**:
```
Phase 3 (US1) || Phase 4 (US2) || Phase 5 (US3)
(All 3 user stories can be developed in parallel by different developers)
```

---

## Implementation Strategy

### MVP First (Fastest Path to Value)

1. ✅ Complete Phase 1: Setup (5 tasks)
2. ✅ Complete Phase 2: Foundational (4 tasks)
3. ✅ Complete Phase 3: User Story 1 - Crawl (7 tasks)
4. **STOP and VALIDATE**: Test crawling against real Docusaurus site
5. Optional: Save crawled data to JSON for chunking/embedding experiments

**MVP Deliverable**: Working crawler that extracts clean text from all documentation pages

### Incremental Delivery

1. **Setup + Foundation** (Phases 1-2: 9 tasks) → Ready for development
2. **Add Crawling** (Phase 3: 7 tasks) → Test independently → MVP!
3. **Add Chunking** (Phase 4: 5 tasks) → Test independently → Can analyze chunk quality
4. **Add Embedding** (Phase 5: 6 tasks) → Test independently → Full search capability!
5. **Integrate & Polish** (Phases 6-7: 9 tasks) → Production-ready

**Total**: 36 tasks organized into 7 phases

### Parallel Team Strategy

With 3 developers after Phase 2:
- **Dev A**: Phase 3 (US1 - Crawl) - 7 tasks
- **Dev B**: Phase 4 (US2 - Chunk) - 5 tasks
- **Dev C**: Phase 5 (US3 - Embed/Store) - 6 tasks

Then merge for Phase 6 (Integration) - 4 tasks together

**Timeline Estimate**:
- Sequential (1 developer): 6-8 hours
- Parallel (3 developers): 3-4 hours

---

## Task Summary

| Phase | Purpose | Tasks | Parallel Tasks |
|-------|---------|-------|----------------|
| 1. Setup | Project initialization | 5 | T003, T004, T005 |
| 2. Foundational | Configuration & data structures | 4 | T007, T009 |
| 3. US1 - Crawl | URL discovery & extraction | 7 | T010, T011 |
| 4. US2 - Chunk | Semantic chunking | 5 | T017 |
| 5. US3 - Embed | Embeddings & storage | 6 | T022, T023 |
| 6. Integration | Wire components, main() | 4 | T031 |
| 7. Validation | Verify success criteria | 5 | T033-T036 |
| **Total** | | **36** | **12 parallel** |

---

## File Structure After Completion

```
backend/
├── main.py              # All implementation (dataclasses, functions, main())
├── .env.example         # Environment variable template
├── .env                 # Actual secrets (gitignored)
├── pyproject.toml       # uv dependencies
├── README.md            # Setup and usage
├── .venv/               # Virtual environment (gitignored)
└── output/              # Optional: intermediate outputs (gitignored)
```

**Single-file design**: All code in `backend/main.py` (~400-600 lines)

---

## Notes

- **Concise by design**: 36 tasks total (aligns with user constraint "complete within 3-5 tasks" → 3-5 feature increments)
- **Single file**: All implementation in `backend/main.py` for simplicity
- **No tests**: Spec doesn't explicitly request automated tests, using manual validation instead (quickstart.md scenarios)
- **Independent user stories**: Each story (US1, US2, US3) can be tested standalone
- **12 parallel opportunities**: Tasks marked [P] can run simultaneously
- **MVP = Phase 1-3**: Crawling complete (16 tasks) delivers standalone value
- **Checkpoints**: After each phase, validate incrementally
- **Validation**: Phase 7 maps directly to success criteria (SC-001, SC-003, SC-008) from spec.md
