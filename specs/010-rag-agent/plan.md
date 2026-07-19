# Implementation Plan: AI Agent with Retrieval-Augmented Capabilities

**Branch**: `010-rag-agent` | **Date**: 2025-12-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/010-rag-agent/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a minimal AI agent powered by the OpenAI Agents SDK that uses retrieval-augmented generation to answer questions about book content. The agent will integrate with the existing Qdrant retrieval pipeline (Feature 009) to fetch relevant chunks and generate grounded responses with source citations. Target: single agent.py file at project root with under 20 lines of setup code, supporting basic Q&A and simple follow-up queries.

## Technical Context

**Language/Version**: Python 3.13 (existing backend requirement)
**Primary Dependencies**: openai (1.58.1 - Assistants API), cohere (5.20.1), qdrant-client (1.16.2), pydantic-settings (2.12.0), python-dotenv (1.2.1)
**Storage**: Qdrant Cloud (existing collection "docusaurus_docs" with 192 vectors from Feature 008), OpenAI server-side Thread storage for conversation history
**Testing**: Manual testing via CLI interaction (pytest for future test automation)
**Target Platform**: Local developer environment (Windows/Linux), Python CLI script
**Project Type**: Single-file script (agent.py at project root)
**Performance Goals**: Agent response time <10 seconds for retrieval-based queries (SC-005), typical 3-5 seconds with gpt-4o-mini
**Constraints**: Retrieval tool execution <3 seconds (inherited from Feature 009), reuse existing pipeline without modifications, use OpenAI Assistants API (not Chat Completions or Swarm)
**Scale/Scope**: Single agent instance, 192 embedded chunks, 3-5 top-k retrieval, server-side conversation history (unlimited messages, context window managed by OpenAI)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Specification-First Development
- ✅ **Status**: PASS
- **Evidence**: Feature originated from formal spec via `/sp.specify` command (specs/010-rag-agent/spec.md)
- **Verification**: All requirements map to user stories US1-US3 with acceptance criteria

### Principle II: Accuracy and Non-Hallucination
- ✅ **Status**: PASS
- **Evidence**: Agent MUST use only retrieved chunks as context (FR-006), handle zero results by informing user information is unavailable (FR-007)
- **Risk**: Agent could hallucinate if retrieval fails silently - mitigated by explicit zero-result handling requirement
- **Verification**: SC-003 requires 100% correct "not available" responses for adversarial questions

### Principle III: Reproducibility and Developer Clarity
- ✅ **Status**: PASS
- **Evidence**: Setup code target <20 lines (SC-001), simple testing interface required (FR-010)
- **Deliverable**: quickstart.md with step-by-step instructions and expected outputs

### Principle IV: AI-Native Authoring
- ✅ **Status**: PASS
- **Evidence**: Using `/sp.plan` command now, will use `/sp.tasks` for breakdown
- **PHR**: Will be created for this planning session

### Principle V: Modular and Clean Architecture
- ✅ **Status**: PASS
- **Evidence**: Agent reuses existing retrieval pipeline (Feature 009) without modification, single-file design for simplicity
- **Independence**: Agent can be tested independently of frontend/backend

### Principle VI: Security and Secrets Management
- ✅ **Status**: PASS
- **Evidence**: OpenAI API key will be added to existing .env file (not committed), .env.example will document OPENAI_API_KEY requirement
- **Verification**: .gitignore already excludes .env files

### Principle VII: Testability and Verification
- ✅ **Status**: PASS
- **Evidence**: All 3 user stories have Given/When/Then acceptance scenarios, 7 measurable success criteria defined
- **Test Strategy**: Manual CLI testing for MVP, automated tests deferred to future iteration

**GATE STATUS**: ✅ **APPROVED** - All constitutional principles satisfied, proceed to Phase 0 research

---

**RE-EVALUATION AFTER PHASE 1 DESIGN**:

### Principle II: Accuracy and Non-Hallucination
- ✅ **Status**: PASS (re-confirmed)
- **Design validation**: OpenAI Assistants API automatically uses retrieval tool outputs as context; system instructions explicitly require "Answer using ONLY retrieved book content"
- **Zero-result handling**: Tool returns "No relevant information found" when no chunks retrieved (FR-007)
- **Risk mitigation**: Agent instructions tested to ensure "say so explicitly" when information unavailable

### Principle V: Modular and Clean Architecture
- ✅ **Status**: PASS (re-confirmed)
- **Design validation**: Single-file agent.py (90 lines total) imports existing retrieve.py functions; no code duplication
- **Independence**: Agent can run standalone; retrieval pipeline unchanged

### Principle VI: Security and Secrets Management
- ✅ **Status**: PASS (re-confirmed)
- **Design validation**: OPENAI_API_KEY added to .env (not committed); .env.example updated with placeholder
- **Implementation**: quickstart.md includes setup instructions with security best practices

**FINAL GATE STATUS**: ✅ **APPROVED FOR IMPLEMENTATION** - Design complies with all constitutional principles, ready for task breakdown

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
physical-ai-book/
├── agent.py                  # NEW: Single-file agent implementation (this feature)
├── backend/
│   ├── retrieve.py          # EXISTING: Retrieval pipeline (Feature 009)
│   │                        # Functions to reuse: generate_query_embedding(), search_qdrant()
│   ├── main.py              # EXISTING: Data ingestion (Feature 008)
│   ├── .env                 # MODIFY: Add OPENAI_API_KEY
│   ├── .env.example         # MODIFY: Document OPENAI_API_KEY
│   └── pyproject.toml       # MODIFY: Add openai dependency
└── specs/010-rag-agent/
    ├── plan.md              # This file
    ├── research.md          # Phase 0 output (to be created)
    ├── data-model.md        # Phase 1 output (to be created)
    ├── quickstart.md        # Phase 1 output (to be created)
    └── contracts/           # Phase 1 output (to be created)
```

**Structure Decision**: Single-file script at project root (agent.py) following user input directive. This is the simplest possible structure for a CLI agent, aligned with SC-001 (setup <20 lines) and the constraint "minimal, modular agent setup". The agent imports existing retrieval functions from backend/retrieve.py to avoid code duplication.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations - all constitutional principles satisfied.
