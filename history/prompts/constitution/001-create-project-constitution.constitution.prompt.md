---
id: 001
title: Create project constitution
stage: constitution
date: 2025-12-23
surface: agent
model: claude-sonnet-4-5
feature: none
branch: master
user: developer
command: /sp.constitution
labels: ["constitution", "initialization", "governance", "principles"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A (documentation creation)
---

## Prompt

```
/sp.constitution

Project: Spec-Driven Book with Embedded RAG Chatbot

Core principles:
- Specification-first development (Spec-Kit Plus)
- Accuracy and non-hallucination
- Reproducibility and clarity for developers
- AI-native authoring using Claude Code

Key standards:
- Book written in Docusaurus and deployed to GitHub Pages
- All content generated and structured via Spec-Kit Plus
- Chapters must map directly to specifications
- Code examples must be runnable and repository-aligned
- Chatbot answers must be grounded only in book content
- Selected-text Q&A must use only the highlighted text

Technical stack:
- Frontend: Docusaurus
- Backend: FastAPI
- AI: OpenAI Agents / ChatKit SDKs
- Vector DB: Qdrant Cloud (Free Tier)
- Relational DB: Neon Serverless Postgres

Constraints:
- No external data access for RAG
- No hallucinated responses
- Secrets documented, not committed
- Clean, modular architecture

Success criteria:
- Public GitHub Pages deployment
- Accurate RAG chatbot embedded in the book
- Verified spec-to-implementation consistency
```

## Response snapshot

Created initial constitution (v1.0.0) for Physical AI Book project with seven core principles:

1. **Specification-First Development**: All content/features originate from formal Spec-Kit Plus specifications
2. **Accuracy and Non-Hallucination**: AI content grounded in verified sources, RAG answers only from book
3. **Reproducibility and Developer Clarity**: All examples/workflows must be reproducible by developers
4. **AI-Native Authoring**: Claude Code and Spec-Kit Plus as primary authoring tools
5. **Modular and Clean Architecture**: Independent components (Docusaurus, FastAPI, RAG service)
6. **Security and Secrets Management**: No credentials in version control, documented but not exposed
7. **Testability and Verification**: Clear acceptance criteria and verification steps for all features

Technical stack documented: Docusaurus (frontend), FastAPI (backend), OpenAI Agents/ChatKit (AI), Qdrant Cloud (vector DB), Neon Postgres (relational DB).

Development workflow established with 4 quality gates: Specification Complete, Plan Approved, Implementation Complete, Deployment Ready.

PHR and ADR processes defined with automatic routing and significance testing.

## Outcome

- ✅ Impact: Constitution v1.0.0 created, establishing governance foundation for Physical AI Book project
- 🧪 Tests: N/A (documentation creation, no code tests required)
- 📁 Files: Created `.specify/memory/constitution.md` with complete principle definitions
- 🔁 Next prompts: `/sp.specify` to create first feature specification, or `/sp.plan` to begin architectural planning
- 🧠 Reflection: Constitution provides clear governance model aligned with Spec-Kit Plus workflow, emphasizing accuracy and reproducibility critical for educational content

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Manual validation - PASS (all placeholders filled, principles clear and testable)
- Prompt variant (if applicable): N/A (initial constitution creation)
- Next experiment (smallest change to try): Monitor template compliance during first feature cycle; adjust quality gates if overly restrictive
