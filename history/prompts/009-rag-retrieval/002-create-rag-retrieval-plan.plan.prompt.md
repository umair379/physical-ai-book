---
id: 002
title: Create RAG Retrieval Plan
stage: plan
date: 2025-12-28
surface: agent
model: claude-sonnet-4-5-20250929
feature: 009-rag-retrieval
branch: 009-rag-retrieval
user: developer
command: /sp.plan
labels: ["plan", "rag", "retrieval", "validation", "architecture", "research", "data-model"]
links:
  spec: D:\physical-ai-book\specs\009-rag-retrieval\spec.md
  ticket: null
  adr: null
  pr: null
files:
 - D:\physical-ai-book\specs\009-rag-retrieval\research.md
 - D:\physical-ai-book\specs\009-rag-retrieval\data-model.md
 - D:\physical-ai-book\specs\009-rag-retrieval\quickstart.md
 - D:\physical-ai-book\CLAUDE.md
tests:
 - Research completion: ✅ PASS - All 9 research areas documented with implementation guidance
 - Data model definition: ✅ PASS - 4 entities defined (Query, SearchResult, CollectionMetadata, QueryMetrics)
 - Quickstart creation: ✅ PASS - 6 validation scenarios with command examples
 - Agent context update: ✅ PASS - CLAUDE.md updated with feature context
---

## Prompt

/sp.plan Spec-2: Retrieval & Pipeline Validation

Create a single file retrieve.py in the backend folder
Connect to Qdrant and load existing vector collections
Accept a test query and perform top-k similarity search
Validate results using returned text, metadata, and source URLs

## Response snapshot

Successfully completed implementation planning for RAG Retrieval Validation feature (009-rag-retrieval). Generated comprehensive research document covering 9 critical implementation areas: single-file script architecture (~250 lines, functional composition), error handling strategy (fail-fast with retry-with-backoff), dual-format logging (console + file), performance measurement (time.perf_counter with percentiles), stratified test query design (common/edge/adversarial/coverage), key implementation differences from Feature 008 (input_type='search_query', timeout=120, query_points() API), metadata validation (7 required fields), configuration reuse (ValidationConfig from main.py), and implementation checklist. Defined complete data model with 4 entities (Query, SearchResult, CollectionMetadata, QueryMetrics) using dataclasses with validation methods. Created quickstart guide with 6 validation scenarios covering all success criteria (SC-001 through SC-007). Ready for task generation phase.

### Planning Summary

**Feature**: RAG Retrieval Validation (Feature 009)
**Branch**: 009-rag-retrieval
**Implementation**: Single-file Python CLI script (backend/retrieve.py, ~250 lines)

---

### Architecture Decisions

**1. Single-File Script Design** (research.md section 1)
- **Decision**: Functional composition pattern, ~250 lines in retrieve.py
- **Rationale**: Simplicity for validation tool, proven in Feature 008 main.py (730 lines)
- **Alternatives Rejected**: Multi-file package (over-engineering), OOP class hierarchy (unnecessary abstraction)
- **Patterns to Reuse**:
  - retry_with_backoff decorator (main.py:498-517)
  - ValidationConfig class (main.py:33-72 pattern)
  - Dual-format logging (console + file)

**2. Error Handling Philosophy** (research.md section 2)
- **Decision**: Fail-fast with retry-with-backoff for API calls
- **Rationale**: Validation scripts should surface errors immediately (not suppress like ingestion pipeline)
- **Key Difference from Feature 008**: Ingestion retries aggressively to complete 192 chunks, validation fails after 3 retries to surface infrastructure issues
- **Implementation**: Reuse retry decorator with max_retries=3, base_delay=2.0s

**3. Performance Measurement Strategy** (research.md section 4)
- **Decision**: time.perf_counter() with phase-separated metrics and percentile tracking
- **Rationale**: Percentiles (p95, p99) reveal tail latency masked by averages, phase separation identifies bottlenecks
- **Thresholds** (from Feature 008 baseline):
  - Embedding p95 <1000ms (Cohere API baseline)
  - Search p95 <500ms (Qdrant baseline for 192 vectors)
  - Total avg <3000ms (SC-005 requirement)

**4. Test Query Design** (research.md section 5)
- **Decision**: Stratified queries in JSON file (common/edge/adversarial/coverage)
- **Rationale**: Different query types validate different aspects (relevance, edge cases, false positives, module coverage)
- **Storage**: backend/test_queries.json for reusability and team collaboration

---

### Critical Implementation Details

**From Feature 008 Learnings** (research.md section 6):

1. **Cohere input_type Parameter** (CRITICAL):
   - Retrieval: `input_type='search_query'`
   - Ingestion: `input_type='search_document'`
   - Reason: Different semantic spaces optimized for queries vs documents

2. **Qdrant Timeout Configuration**:
   - Must use `timeout=120` in QdrantClient initialization
   - Reason: Free Tier has strict timeouts, Feature 008 encountered ReadTimeout errors

3. **Qdrant Search API**:
   - Current: `client.query_points().points`
   - Deprecated: `client.search()` (raises AttributeError in v1.16.2)

4. **Error Tolerance**:
   - Ingestion: Continue on error, report at end
   - Retrieval: Fail immediately with clear message

---

### Data Model (data-model.md)

**Entity 1: Query**
- Attributes: query_text (str), embedding_vector (List[float]), category (str), expected thresholds
- Implementation: Dataclass with __post_init__ validation (1024-dim check)
- Purpose: Represents user input for semantic search

**Entity 2: SearchResult**
- Attributes: score (float), chunk_id (str), text (str), url (str), title (str), heading (str), chunk_index (int), timestamp (str)
- Implementation: Dataclass with from_scored_point() factory and validate_metadata() method
- Purpose: Single matching chunk from Qdrant with complete metadata

**Entity 3: CollectionMetadata**
- Attributes: collection_name (str), points_count (int), vector_dimension (int), distance_metric (str), status (str)
- Implementation: Dataclass with from_collection_info() factory and validate_against_spec() method (checks 192 points, 1024 dims, COSINE, GREEN)
- Purpose: Infrastructure verification before queries

**Entity 4: QueryMetrics**
- Attributes: query_text (str), embedding_time_ms (float), search_time_ms (float), total_latency_ms (float), result_count (int), best_score (float)
- Parent: PerformanceMetrics dataclass with query_metrics[] list and get_summary() percentile calculation
- Purpose: Performance tracking and SC-005 validation

**Why Dataclasses**: Simplicity for validation tool, zero runtime overhead, cleaner syntax than Pydantic for simple structs

---

### File Structure

**New File**:
```
backend/retrieve.py (~250 lines)
├── Imports (lines 1-20)
├── Configuration (lines 21-40): ValidationConfig class
├── Logging Setup (lines 41-70): setup_logging() with dual handlers
├── Core Functions (lines 71-200):
│   ├── verify_connection() - FR-001, FR-002
│   ├── generate_query_embedding() - FR-003
│   ├── search_qdrant() - FR-004
│   ├── validate_result_metadata() - FR-009
│   ├── display_results() - FR-005
│   └── run_test_suite() - FR-006
└── Main Orchestration (lines 201-250): argparse + main()
```

**New Test Data**:
```
backend/test_queries.json
├── common: queries with expected_min_score >0.8
├── edge_cases: complex queries >0.7
├── adversarial: off-topic queries <0.5
└── topic_coverage: module-specific with expected_url_pattern
```

**Planning Artifacts**:
- `specs/009-rag-retrieval/research.md` - 9 implementation areas documented
- `specs/009-rag-retrieval/data-model.md` - 4 entities with validation methods
- `specs/009-rag-retrieval/quickstart.md` - 6 validation scenarios

---

### Validation Scenarios (from quickstart.md)

**Scenario 1: Verify Infrastructure (SC-001)**
```bash
python retrieve.py --query "test"
```
Expected: Connection succeeds, reports 192 points, status GREEN

**Scenario 2: Test Query Relevance (SC-002, SC-003)**
```bash
python retrieve.py --test-suite test_queries.json
```
Expected: 100% queries return results >0.4 similarity

**Scenario 3: Validate Metadata Completeness (SC-004)**
```bash
python retrieve.py --query "What is physical AI?"
```
Expected: All results have 7 required fields (chunk_id, text, url, title, heading, chunk_index, timestamp)

**Scenario 4: Performance Baseline (SC-005)**
```bash
python retrieve.py --test-suite test_queries.json
```
Expected: Average latency <3000ms for 10 queries

**Scenario 5: Error Handling (SC-006)**
Test with missing credentials and invalid URL
Expected: Graceful failure with clear error messages

**Scenario 6: Topic Coverage (SC-007)**
Test module-specific queries (ROS 2, computer vision, neural networks)
Expected: 80% return results from correct documentation section

---

### Dependencies

**From Feature 008** (already installed):
- `qdrant-client` - Qdrant vector database client
- `cohere` - Cohere embedding API
- `pydantic-settings` - Configuration management
- `python-dotenv` - .env file loading

**No New Dependencies**: Validation script reuses Feature 008 infrastructure

---

### Configuration Reuse

**Expected .env** (from Feature 008):
```bash
COHERE_API_KEY=BUq6Z6ewir2YTV7ghQAReujSx7lc8VKh8zsx46iP
QDRANT_URL=https://dbe06b27-f4e3-4c82-a911-09160423ee6c.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
COLLECTION_NAME=docusaurus_docs
```

---

### Implementation Checklist (from research.md section 9)

**Before Writing Code**:
- [ ] Read main.py (Feature 008) to understand proven patterns
- [ ] Review spec.md FR-001 through FR-010 and SC-001 through SC-007
- [ ] Identify reusable code sections (retry decorator, Config class, logging setup)

**During Implementation**:
- [ ] Reuse retry_with_backoff decorator from main.py:498-517
- [ ] Reuse Config class pattern from main.py:33-72 (adapt to ValidationConfig)
- [ ] Use input_type='search_query' for Cohere embeddings (NOT 'search_document')
- [ ] Add timeout=120 to QdrantClient initialization
- [ ] Use client.query_points() (NOT deprecated client.search())
- [ ] Implement dual-format logging (console + file)
- [ ] Use time.perf_counter() for latency measurement
- [ ] Validate metadata fields for all results (FR-009)
- [ ] Create test_queries.json with stratified test cases

**After Implementation**:
- [ ] Test with default .env credentials (Feature 008 infrastructure)
- [ ] Verify SC-001 through SC-007 with validation scenarios
- [ ] Create PHR documenting implementation

---

### Estimated Implementation Effort

**From Spec**: 1-2 tasks
**Breakdown**:
- Task 1: Core script implementation (retrieve.py + basic test queries) - 1 task
- Task 2: Test suite expansion + comprehensive validation - 1 task (optional)

**Estimated Lines of Code**: ~250 lines retrieve.py + ~100 lines test_queries.json = 350 lines total

---

### Research Findings Summary

**9 Research Areas Documented**:
1. Single-file script architecture (~250 lines, functional composition)
2. Error handling strategy (fail-fast with retry-with-backoff)
3. Logging strategy (dual-format: console human-readable + file structured)
4. Performance measurement (time.perf_counter with percentiles p50/p95/p99)
5. Test query design (stratified: common/edge/adversarial/coverage)
6. Key differences from Feature 008 (input_type, timeout, API, error tolerance)
7. Metadata validation (7 required fields, 100% non-null)
8. Configuration reuse (ValidationConfig pattern from main.py)
9. Implementation checklist (before/during/after code writing)

**Key Patterns to Reuse from main.py**:
- Lines 498-517: retry_with_backoff decorator
- Lines 33-72: Config class pattern (adapt to ValidationConfig)
- Logging setup: Dual handlers (console + file)
- Qdrant client: timeout=120, query_points() API

---

### Next Steps

**Ready for Task Generation**: /sp.tasks to break down implementation into actionable tasks

**Recommended Task Breakdown**:
1. Create retrieve.py core functions (verify_connection, generate_query_embedding, search_qdrant)
2. Implement metadata validation and result display (validate_result_metadata, display_results)
3. Add test suite support (run_test_suite, load_test_suite, validate_test_case)
4. Create test_queries.json with stratified queries
5. Implement performance tracking (QueryMetrics, PerformanceMetrics)
6. Add CLI interface (argparse, main orchestration)
7. Validate all SC-001 through SC-007 with quickstart scenarios

## Outcome

- ✅ Impact: Successfully completed implementation planning for RAG Retrieval Validation feature (009-rag-retrieval). Generated comprehensive research document (research.md) covering 9 critical implementation areas with code patterns, thresholds, and best practices. Defined complete data model (data-model.md) with 4 entities (Query, SearchResult, CollectionMetadata, QueryMetrics) using dataclasses with validation methods, type hints, and factory patterns. Created detailed quickstart guide (quickstart.md) with 6 validation scenarios, command-line examples, troubleshooting section, and success criteria checklist covering SC-001 through SC-007. Updated agent context (CLAUDE.md) with feature-specific patterns. Planning artifacts ready for task generation phase.

- 🧪 Tests: Research completion PASS (9 sections documented with implementation guidance), Data model definition PASS (4 entities with validation methods), Quickstart creation PASS (6 scenarios covering all success criteria), Agent context update PASS (CLAUDE.md updated)

- 📁 Files: Created 3 planning artifacts - specs/009-rag-retrieval/research.md (9 sections, ~800 lines, implementation patterns from Feature 008), specs/009-rag-retrieval/data-model.md (4 entities, entity relationships diagram, validation rules), specs/009-rag-retrieval/quickstart.md (6 validation scenarios, CLI examples, troubleshooting guide), updated CLAUDE.md with feature context

- 🔁 Next prompts: Ready for `/sp.tasks` to generate actionable task breakdown for retrieve.py implementation (estimated 1-2 tasks: core script + test suite). Recommended task structure: (1) Core functions (verify_connection, generate_query_embedding, search_qdrant), (2) Metadata validation and display, (3) Test suite support, (4) Performance tracking, (5) CLI interface, (6) Validation with quickstart scenarios

- 🧠 Reflection: Planning phase highly effective due to comprehensive research from agent task and clear spec from Feature 001. Key architectural decisions made: (1) Single-file design (~250 lines) balances simplicity with maintainability for validation tool, (2) Fail-fast error handling appropriate for validation (vs ingestion's aggressive retry), (3) Dataclasses chosen over Pydantic for zero overhead and cleaner syntax, (4) Stratified test query design enables systematic validation of relevance/edge cases/false positives/coverage, (5) Reusing proven patterns from Feature 008 (retry decorator, config class, timeout=120) reduces implementation risk. Critical implementation details documented: input_type='search_query' (not 'search_document'), query_points() API (not deprecated search()), performance thresholds (embed p95 <1000ms, search p95 <500ms, total avg <3000ms). Research findings translate directly to implementation checklist (before/during/after code writing) with specific line references to main.py for pattern reuse. Quickstart guide comprehensive with 6 scenarios covering all SC-001 through SC-007, command examples, troubleshooting, and log interpretation. Estimated 350 total lines (250 retrieve.py + 100 test_queries.json) aligns with 1-2 task timeline from spec. Ready for task generation with high confidence in implementation approach.

## Evaluation notes (flywheel)

- Failure modes observed: None - planning completed successfully without blockers or rework
- Graders run and results (PASS/FAIL): Research completion PASS (9 areas documented), Data model definition PASS (4 entities with validation), Quickstart creation PASS (6 scenarios), Agent context update PASS
- Prompt variant (if applicable): Standard /sp.plan workflow with additional guidance "Spec-2: Retrieval & Pipeline Validation, single file retrieve.py in backend folder"
- Next experiment (smallest change to try): Proceed to /sp.tasks to generate actionable task breakdown. Potential optimization: Consider combining metadata validation + display into single task to reduce task overhead (currently split as tasks 2-3). Evaluate during task generation if 1-task approach (implement entire retrieve.py) feasible given clear architecture and reusable patterns from Feature 008.
