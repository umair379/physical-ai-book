---
id: 002
title: Create Module 4 VLA Plan
stage: plan
date: 2025-12-25
surface: agent
model: claude-sonnet-4-5
feature: 004-vla
branch: 004-vla
user: user
command: /sp.plan
labels: ["planning", "vla", "voice-recognition", "llm-planning", "whisper", "gpt4", "docusaurus"]
links:
  spec: specs/004-vla/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/004-vla/plan.md
tests:
  - Constitution check (7/7 principles PASS)
---

## Prompt

```
- Add Module 4 to Docusaurus with 3 chapters as .md files (Voice-to-Action, Cognitive Planning, Capstone Project).
- Include examples, code snippets, and workflows for voice commands, LLM-based planning, and autonomous humanoid tasks.
```

## Response snapshot

Successfully created comprehensive implementation plan for Module 4: Vision-Language-Action (VLA) with detailed technical context, research decisions, and phase-by-phase breakdown.

**Plan Structure:**

**Technical Context:**
- **Stack**: Docusaurus 3.9.2 (Markdown/MDX content), Python 3.10+ (Whisper/LLM code examples)
- **Scale**: 3 chapters, 4 content files, 25-30 code examples, 4-6 Mermaid diagrams
- **Performance**: Docusaurus build less than 60s, site 60 FPS, +5-10s build time vs Module 3
- **Constraints**: Educational content with OpenAI API access OR free LLaMA 3 alternative, microphone required

**Phase 0 Research Decisions:**

1. **Whisper Model Selection**:
   - Primary: Base model (74M params, 0.5-1s latency, 90%+ accuracy)
   - Alternative: Tiny (39M params, 0.2-0.5s latency, 85% accuracy)
   - Advanced: Large (1550M params, 0.1-0.3s latency, best accuracy, GPU required)
   - Rationale: Base is sweet spot for most students (accuracy + latency balance)

2. **LLM API Choice**:
   - Primary: GPT-4 Turbo ($0.01/1K input, $0.03/1K output, best quality)
   - Free Alternative: Local LLaMA 3 8B via Ollama (no cost, slower, acceptable quality)
   - Budget: GPT-3.5-turbo ($0.0005/1K, faster, slightly lower quality)
   - Rationale: GPT-4 most reliable for JSON robot plans, $5-10 per student acceptable

3. **ROS 2 Action Client Pattern**:
   - Primary: Asynchronous with feedback callbacks
   - Rationale: Shows real-time status (PENDING → ACTIVE → SUCCEEDED), models real-world patterns

**Phase 1 Design Artifacts:**

- **data-model.md**: Module 4 instance (7-9 hours, $5-10 API costs OR free), Chapter 1 (Whisper voice-to-action, 100 min), Chapter 2 (LLM planning, 110 min), Chapter 3 (Capstone, 120 min)
- **companion-repo-spec.md**: chapter-1-voice-to-action/ (Whisper scripts, audio samples), chapter-2-llm-planning/ (OpenAI/LLaMA 3 examples, prompts, validators), chapter-3-capstone/ (full pipeline integration)

**Implementation Strategy (5 Phases, ~130 tasks estimated):**

- **Phase 1: Setup** (5 tasks): module-4/ directory, sidebars.ts update, index.md, intro.md teaser
- **Phase 2: Chapter 1 - Voice-to-Action** (~35 tasks): Whisper installation, microphone config, real-time transcription, intent parsing (regex/spaCy), ROS 2 action client, confidence filtering, hands-on exercises
- **Phase 3: Chapter 2 - LLM Planning** (~40 tasks): LLM comparison (GPT-4/Claude/LLaMA 3), OpenAI API integration, Ollama setup, prompt engineering (JSON schemas), plan validation, execution, replanning, safety constraints
- **Phase 4: Chapter 3 - Capstone** (~35 tasks): Architecture overview, voice handler, LLM planner, Nav2 controller, YOLO perception, manipulation retry, full pipeline, dynamic obstacles, ambiguity resolution, success metrics
- **Phase 5: Polish** (~15 tasks): Cross-references, Mermaid diagrams (voice-to-action flow, LLM pipeline, capstone, safety), external links, Docusaurus build validation

**Acceptance Criteria:**

- Content: All 15 FRs mapped, complete code examples with imports/dependencies/API key handling
- Technical: Whisper base verified, GPT-4 API tested, LLaMA 3 Ollama tested, ROS 2 Humble actions tested
- Educational: Theory before code, commented examples, step-by-step exercises, troubleshooting, API cost transparency
- Build: No errors, no MDX syntax issues, 60 FPS, less than 15s build time increase

## Outcome

- ✅ Impact: Complete implementation plan for Module 4 with research decisions, design artifacts, and 5-phase breakdown (~130 tasks)
- 🧪 Tests: Constitution check validation (7/7 principles PASS)
- 📁 Files: Created plan.md in specs/004-vla/
- 🔁 Next prompts: /sp.tasks to generate detailed task breakdown from plan.md
- 🧠 Reflection: Plan balances paid (GPT-4) and free (LLaMA 3) LLM options for accessibility. Whisper base model recommended as sweet spot (accuracy + latency). Async ROS 2 action pattern chosen for educational value (shows real-time feedback). All code examples will include API key management via environment variables. Estimated $5-10 per student for OpenAI API usage is acceptable with free alternatives documented.

## Evaluation notes (flywheel)

- Failure modes observed: None - plan created successfully with informed technical decisions
- Graders run and results (PASS/FAIL): Constitution check PASS (7/7 principles)
- Prompt variant (if applicable): N/A - standard /sp.plan workflow
- Next experiment (smallest change to try): Consider adding specific LLM prompt templates (system prompts for robot planning) directly in plan.md Phase 1 artifacts to reduce ambiguity during implementation
