# Implementation Plan: Module 4 - Vision-Language-Action (VLA)

**Branch**: `004-vla` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: User request: "Add Module 4 to Docusaurus with 3 chapters as .md files (Voice-to-Action, Cognitive Planning, Capstone Project). Include examples, code snippets, and workflows for voice commands, LLM-based planning, and autonomous humanoid tasks."

## Summary

Create Module 4 educational content for the Physical AI Book teaching Vision-Language-Action (VLA) paradigm with OpenAI Whisper for voice recognition, LLM-based cognitive planning (GPT-4/Claude/LLaMA 3), and autonomous humanoid integration. Content follows Module 1-3 patterns with Docusaurus markdown files, code examples, and hands-on exercises. Students learn to convert voice commands into robot actions, use LLMs to generate multi-step plans, and build complete autonomous systems.

## Technical Context

**Language/Version**: JavaScript/Node.js 18+ (Docusaurus build), Markdown/MDX (content authoring), Python 3.10+ (code examples)
**Primary Dependencies**: Docusaurus 3.9.2 (already installed), @docusaurus/theme-mermaid 3.9.2 (already installed for diagrams)
**Project Type**: Documentation website (Docusaurus-based static site) - extending existing frontend-book/ project
**Scale/Scope**: 3 chapters for Module 4, ~15-20 pages total content, 25-30 code examples (Whisper Python scripts, LLM API calls, ROS 2 action clients, capstone integration)
**Performance Goals**: Docusaurus build completes in under 60 seconds, site renders at 60 FPS
**Constraints**: Educational content must be beginner-friendly yet technically accurate, all code examples must be runnable with specified dependencies (OpenAI API, microphone, ROS 2 Humble)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Specification-First Development ✅
- Module 4 content maps directly to spec.md user stories (US1: Voice-to-Action P1, US2: LLM Planning P2, US3: Capstone P3)
- Each FR-001 to FR-015 requirement translates to specific content sections

### Principle II: Accuracy and Non-Hallucination ✅
- Will reference real tools (OpenAI Whisper, GPT-4 API, Anthropic Claude API, LLaMA 3 via Ollama)
- Will cite actual API documentation (OpenAI platform docs, Anthropic docs, ROS 2 action tutorials)
- Will link to official resources (GitHub openai/whisper, platform.openai.com/docs)
- No invented APIs or fictitious LLM capabilities

### Principle III: Reproducibility and Developer Clarity ✅
- Will provide exact environment setup (Ubuntu 22.04, Python 3.10+, PyTorch 2.0+, microphone config)
- All Whisper Python scripts will include complete imports (whisper, sounddevice, numpy, json)
- LLM API examples will specify exact endpoints and request formats (OpenAI Chat Completions, Claude Messages API)
- ROS 2 action client code will document action types and goal message structures
- API cost estimates and rate limit handling documented

### Principle IV: AI-Native Authoring ✅
- Plan created via /sp.plan command (this document)
- Tasks will be generated via /sp.tasks command
- PHRs will capture AI interactions during content creation

### Principle V: Modular and Clean Architecture ✅
- 3 independently testable user stories (chapters can be completed in sequence)
- Each chapter focuses on single VLA component (voice → LLM planning → capstone integration)
- Code examples are self-contained with minimal cross-dependencies

### Principle VI: Security and Privacy ✅
- API keys managed via environment variables (.env files, never committed)
- Voice data processed in real-time (no persistent storage)
- LLM API calls use secure HTTPS connections
- Educational context with user consent for voice recording

### Principle VII: Testability and Continuous Validation ✅
- Each code example will include expected output documentation
- Hands-on exercises will have verification checklists
- Success criteria measurable via performance metrics (95% transcription accuracy, 90% task completion, 85% plan feasibility)

**Gate Result**: ✅ PASS - All 7 principles satisfied

---

## Project Structure

### Documentation (this feature)

```text
specs/004-vla/
├── plan.md              # This file
├── spec.md              # Feature specification (already exists)
├── research.md          # Phase 0 output (to be created)
├── data-model.md        # Phase 1 output (to be created)
├── companion-repo-spec.md  # Companion code repository structure
├── checklists/
│   └── requirements.md  # Spec validation checklist (already exists)
└── tasks.md             # Phase 2 output (created by /sp.tasks)
```

### Source Code (repository root)

```text
frontend-book/docs/module-4/
├── index.md          # Module 4 overview page
├── chapter-1-voice-to-action.md
├── chapter-2-llm-planning.md
└── chapter-3-capstone.md

frontend-book/static/img/module-4/
└── (diagrams, screenshots if needed)

frontend-book/sidebars.ts  # Update with Module 4 navigation

frontend-book/docs/intro.md  # Update with Module 4 teaser
```

## Complexity Tracking

| Metric | Value | Notes |
|--------|-------|-------|
| **Content Pages** | 4 files | index.md + 3 chapters |
| **Code Examples** | ~25-30 examples | Python (Whisper, LLM APIs, ROS 2 actions), JSON (LLM prompts, plans) |
| **Mermaid Diagrams** | 4-6 diagrams | Voice-to-action flow, LLM planning pipeline, capstone integration, safety validation |
| **External Links** | 15+ links | OpenAI docs, Anthropic docs, ROS 2 tutorials, research papers on VLA |
| **Estimated LOC** | ~2200 lines | Markdown content across all chapters |
| **Build Time Impact** | +5-10 seconds | Additional pages and code blocks |

---

## Phase 0: Research & Technology Decisions

**Purpose**: Resolve technical unknowns before design. Determine which Whisper model to recommend, LLM API integration patterns, and ROS 2 action client best practices for voice-triggered execution.

### Research Questions

1. **Whisper Model Selection**: Which Whisper model size (tiny, base, small, medium, large) provides the best balance of accuracy and latency for real-time voice commands on typical student hardware (laptop CPU vs GPU)?

2. **LLM API Choice**: Should we recommend OpenAI GPT-4 (best performance, highest cost), Anthropic Claude (good performance, moderate cost), or local LLaMA 3 (free, slower, lower quality) as the primary LLM for robot planning exercises?

3. **ROS 2 Action Client Pattern**: What is the most beginner-friendly pattern for triggering ROS 2 actions from Python voice command scripts - synchronous blocking calls, asynchronous futures, or action feedback callbacks?

### Research Outputs

Create `specs/004-vla/research.md` with:

**Decision 1**: Whisper model recommendation for voice-to-action
- **Primary**: Whisper "base" model (74M parameters, ~1GB disk, 0.5-1s latency on CPU)
- **Alternative**: Whisper "tiny" for resource-constrained systems (39M parameters, 0.2-0.5s latency)
- **Advanced**: Whisper "large" for students with GPUs (1550M parameters, best accuracy, 0.1-0.3s on RTX 3060)
- **Rationale**: Base model achieves 90%+ accuracy on clean speech with acceptable latency for interactive robot control. Tiny sacrifices accuracy (85%), large requires GPU. Base is the sweet spot for most students.

**Decision 2**: LLM API recommendation for cognitive planning
- **Primary**: OpenAI GPT-4 Turbo via API (best plan quality, $0.01/1K tokens input, $0.03/1K tokens output)
- **Free Alternative**: Local LLaMA 3 8B via Ollama (no API costs, slower, acceptable quality for educational use)
- **Budget Alternative**: OpenAI GPT-3.5-turbo ($0.0005/1K tokens, faster, slightly lower quality)
- **Rationale**: GPT-4 generates most reliable JSON-formatted robot plans with safety constraints. Estimated $5-10 per student for module completion is acceptable. Free alternatives documented for accessibility.

**Decision 3**: ROS 2 action client pattern
- **Primary**: Asynchronous action client with feedback callbacks (best for educational demonstration)
- **Rationale**: Shows students real-time feedback (PENDING → ACTIVE → SUCCEEDED), allows cancellation, teaches async programming. More complex than blocking calls but models real-world robot control patterns.

---

## Phase 1: Design Artifacts

**Purpose**: Create technical design documents before task breakdown. Define content structure, code example formats, and validation criteria.

### 1. Data Model (`specs/004-vla/data-model.md`)

Define content entities following Module 1-3 pattern:

#### Module 4 Instance

```yaml
module_id: "module-4"
module_number: 4
title: "Vision-Language-Action (VLA)"
description: "Master voice-controlled humanoid robots using OpenAI Whisper for speech recognition, large language models for cognitive planning, and end-to-end autonomous task execution."
learning_objectives:
  - "Convert natural language voice commands into structured robot actions using OpenAI Whisper"
  - "Parse voice transcriptions to extract intent (action type, object parameters, spatial constraints)"
  - "Use LLMs (GPT-4, Claude, LLaMA 3) to generate multi-step robot action plans from high-level tasks"
  - "Validate and execute LLM-generated plans with safety constraints (velocity limits, workspace boundaries)"
  - "Integrate voice recognition, LLM planning, Nav2 navigation, and YOLOv8 perception into autonomous systems"
prerequisites:
  - "Module 1 completion (ROS 2 actions, action clients, action servers)"
  - "Module 2 completion (Gazebo/Unity simulation for robot testing)"
  - "Module 3 completion (Isaac ROS perception, Nav2 navigation, YOLOv8 object detection)"
  - "Python 3.10+ with PyTorch 2.0+ installed"
  - "USB microphone or laptop mic with Ubuntu 22.04 ALSA/PulseAudio support"
  - "OpenAI API key OR local LLaMA 3 installation via Ollama"
estimated_duration: "7-9 hours"
api_costs: "$5-10 per student (OpenAI API usage) OR $0 (local LLaMA 3)"
```

#### Chapter 1: Voice-to-Action with OpenAI Whisper

```yaml
chapter_id: "chapter-1-voice-to-action"
chapter_number: 1
title: "Voice-to-Action with OpenAI Whisper"
description: "Learn to convert natural language voice commands into executable robot actions using OpenAI Whisper for speech-to-text, intent parsing, and ROS 2 action mapping."
learning_outcomes:
  - "Install OpenAI Whisper (base model) and configure microphone for real-time audio streaming"
  - "Transcribe voice commands with 95%+ word accuracy in quiet environments"
  - "Parse transcribed text to extract action intent and parameters (action type, object, location, quantity)"
  - "Map parsed intents to ROS 2 action goals (cmd_vel, NavigateToPose, custom manipulation)"
  - "Handle low-confidence transcriptions (less than 0.7) and ambiguous commands with clarification prompts"
code_examples:
  - "whisper_installation_verification"
  - "microphone_audio_streaming"
  - "realtime_voice_transcription"
  - "intent_parsing_regex_patterns"
  - "ros2_action_client_voice_triggered"
estimated_reading_time: 100
hands_on_exercises:
  - "Exercise 1: Install Whisper and transcribe pre-recorded voice commands"
  - "Exercise 2: Build intent parser for 10 common robot commands"
  - "Exercise 3: Trigger ROS 2 cmd_vel action with voice command 'Move forward 2 meters'"
```

#### Chapter 2: Cognitive Planning with LLMs

```yaml
chapter_id: "chapter-2-llm-planning"
chapter_number: 2
title: "Cognitive Planning with LLMs"
description: "Master LLM-based robot planning where GPT-4 or LLaMA 3 decomposes high-level tasks into multi-step action sequences with validation and error handling."
learning_outcomes:
  - "Design LLM prompts for robot planning with JSON-formatted output schemas"
  - "Call OpenAI GPT-4 API or local LLaMA 3 to generate 5+ step plans from tasks like 'Clean the room'"
  - "Validate LLM-generated plans for feasibility (action types exist, parameters in valid ranges, no safety violations)"
  - "Execute multi-step plans using ROS 2 action sequences from Module 3 (Nav2, manipulation)"
  - "Implement replanning when actions fail (send context to LLM, request alternative plan within 5 seconds)"
code_examples:
  - "openai_gpt4_api_call"
  - "llm_prompt_template_robot_planning"
  - "json_schema_validation_action_plan"
  - "plan_executor_ros2_actions"
  - "error_handling_replanning"
  - "local_llama3_ollama_integration"
estimated_reading_time: 110
hands_on_exercises:
  - "Exercise 1: Design LLM prompt for 'Set the table' task and generate plan"
  - "Exercise 2: Validate plan for infeasible actions (unreachable locations, missing objects)"
  - "Exercise 3: Execute 7-step 'Prepare coffee' plan in Isaac Sim with failure recovery"
```

#### Chapter 3: Capstone Project - Autonomous Humanoid

```yaml
chapter_id: "chapter-3-capstone"
chapter_number: 3
title: "Capstone Project: Autonomous Humanoid"
description: "Integrate all modules (ROS 2, simulation, Isaac perception, Nav2, voice, LLM) into a complete autonomous humanoid system executing voice-commanded multi-step tasks."
learning_outcomes:
  - "Build end-to-end pipeline: voice command → Whisper transcription → intent parsing → LLM planning → Nav2 navigation → YOLO detection → manipulation"
  - "Handle dynamic obstacles during navigation (Nav2 local planner re-routing)"
  - "Resolve ambiguities with LLM reasoning (multiple object candidates, unclear spatial references)"
  - "Implement grasp failure recovery (3 retries with adjusted positions, then replan)"
  - "Achieve 90%+ task completion rate on 10 sequential voice-commanded tasks"
code_examples:
  - "capstone_main_pipeline"
  - "voice_llm_nav_perception_integration"
  - "yolov8_object_query_llm_decision"
  - "manipulation_retry_logic"
  - "task_completion_metrics"
estimated_reading_time: 120
hands_on_exercises:
  - "Exercise 1: Complete capstone setup (Whisper + GPT-4/LLaMA 3 + Nav2 + YOLO + manipulation)"
  - "Exercise 2: Execute 'Bring me the blue bottle from the shelf' with full pipeline"
  - "Exercise 3: Test 10 different voice commands and measure success rate (target: 90%+)"
```

### 2. Companion Repository Spec (`specs/004-vla/companion-repo-spec.md`)

Structure for example code repository (similar to Module 2-3):

```text
physical-ai-book-examples/module-4-vla/
├── README.md
├── chapter-1-voice-to-action/
│   ├── scripts/
│   │   ├── install_whisper.sh
│   │   ├── verify_microphone.py
│   │   ├── realtime_transcription.py
│   │   ├── intent_parser.py
│   │   └── voice_action_client.py
│   ├── audio_samples/
│   │   └── test_commands.wav
│   ├── configs/
│   │   └── action_primitives.json
│   └── expected_output.txt
├── chapter-2-llm-planning/
│   ├── scripts/
│   │   ├── openai_api_example.py
│   │   ├── llama3_ollama_example.py
│   │   ├── prompt_templates.py
│   │   ├── plan_validator.py
│   │   └── plan_executor.py
│   ├── prompts/
│   │   ├── system_prompt_robot_planner.txt
│   │   └── example_tasks.json
│   ├── configs/
│   │   ├── action_schema.json
│   │   └── safety_constraints.yaml
│   └── expected_output.txt
└── chapter-3-capstone/
    ├── scripts/
    │   ├── capstone_main.py
    │   ├── voice_handler.py
    │   ├── llm_planner.py
    │   ├── nav_controller.py
    │   ├── perception_handler.py
    │   └── manipulation_controller.py
    ├── launch/
    │   └── capstone_full_stack.launch.py
    ├── configs/
    │   ├── capstone_params.yaml
    │   └── test_scenarios.json
    └── expected_output.txt
```

### 3. Post-Design Re-Evaluation (Constitution Check)

After creating data-model.md and companion-repo-spec.md, re-validate:

**Principle II (Accuracy)**:
- ✅ research.md documents Whisper model selection (base vs tiny vs large with latency benchmarks)
- ✅ data-model.md specifies exact versions (Whisper base 74M params, GPT-4 Turbo, LLaMA 3 8B, ROS 2 Humble)
- ✅ companion-repo-spec.md provides runnable code examples with expected outputs

**Principle III (Reproducibility)**:
- ✅ research.md provides Whisper installation via pip (pip install openai-whisper)
- ✅ data-model.md includes microphone verification scripts (sounddevice, PyAudio)
- ✅ companion-repo-spec.md has expected_output.txt for each chapter

**Gate Result**: ✅ PASS - Design artifacts support reproducible implementation

---

## Constraints & Invariants

### Hard Constraints

1. **OpenAI API Key Required (or Local LLM Alternative)**: All LLM planning content assumes GPT-4 API access OR local LLaMA 3
   - Mitigation: Document both paid (OpenAI) and free (LLaMA 3 via Ollama) options in prerequisites
   - Risk: Students without API access may struggle with LLM quality differences

2. **Microphone Hardware Required**: Voice recognition requires working USB microphone or laptop mic
   - Mitigation: Provide text-based command fallback for students without microphone
   - Invariant: Real-time voice is core to VLA paradigm - no way to fully simulate

3. **Python 3.10+ with PyTorch**: Whisper requires PyTorch 2.0+ for model inference
   - Mitigation: Document pip installation (pip install torch torchvision torchaudio)
   - Invariant: Cannot use Whisper without PyTorch dependency

### Soft Constraints

1. **Quiet Environment for Best Results**: Whisper achieves 95%+ accuracy with SNR greater than 20 dB
   - Rationale: Students in noisy environments may see 85% accuracy instead
   - Flexibility: Include noise filtering examples (spectral subtraction)

2. **Internet Connection for LLM APIs**: OpenAI and Anthropic require network access
   - Rationale: Local LLaMA 3 works offline but with lower quality
   - Flexibility: Provide cached LLM plan examples for offline testing

### Non-Goals

- **Custom Whisper Model Training**: Out of scope (use pre-trained models only)
- **Fine-Tuning LLMs for Robotics**: Out of scope (students use pre-trained models via API)
- **Real Hardware Voice Interface**: Focus on simulation; far-field microphone arrays out of scope
- **Multi-Modal VLA Models**: Sequential pipeline only (vision → LLM → action, not unified model)
- **Reinforcement Learning for LLM Improvement**: Out of scope (fixed LLM capabilities)

---

## Implementation Strategy

### Phase-by-Phase Breakdown

**Phase 1: Setup & Infrastructure** (Similar to Module 2-3 T001-T008)
1. Create `frontend-book/docs/module-4/` directory
2. Create `frontend-book/static/img/module-4/` for diagrams
3. Update `frontend-book/sidebars.ts` with Module 4 navigation
4. Create `module-4/index.md` overview page
5. Update `frontend-book/docs/intro.md` with Module 4 teaser

**Phase 2: Chapter 1 - Voice-to-Action** (FR-001 to FR-005, ~35 tasks)
1. Whisper installation and verification section
2. Microphone configuration (ALSA, PulseAudio, sounddevice library)
3. Real-time audio streaming and transcription
4. Intent parsing patterns (regex, NLP libraries like spaCy)
5. ROS 2 action client integration (cmd_vel, NavigateToPose)
6. Confidence score filtering and error handling
7. Hands-on exercises with verification checklists

**Phase 3: Chapter 2 - LLM Planning** (FR-006 to FR-011, ~40 tasks)
1. LLM selection comparison (GPT-4 vs Claude vs LLaMA 3)
2. OpenAI API integration (Chat Completions endpoint)
3. LLaMA 3 local integration (Ollama setup)
4. Prompt engineering for robot planning (system prompts, JSON schemas)
5. Plan validation (action types, parameter ranges, dependencies)
6. Plan execution with ROS 2 action sequences
7. Error handling and replanning (send failure context to LLM)
8. Safety constraint checking (velocity limits, workspace boundaries)
9. Hands-on exercises with multi-step plan execution

**Phase 4: Chapter 3 - Capstone Project** (FR-012 to FR-015, ~35 tasks)
1. Capstone architecture overview (voice → LLM → Nav2 → YOLO → manipulation)
2. Voice handler module integration
3. LLM planner module integration
4. Navigation controller (Nav2 goal sending)
5. Perception handler (YOLOv8 object detection queries)
6. Manipulation controller (grasp/place with retry logic)
7. Full pipeline orchestration
8. Dynamic obstacle handling (Nav2 re-routing)
9. Ambiguity resolution (LLM reasoning for multiple candidates)
10. Task completion metrics (success rate calculation)
11. Hands-on exercises with 10 test scenarios

**Phase 5: Polish & Validation** (~15 tasks)
1. Cross-reference links between chapters
2. Mermaid diagrams (voice-to-action flow, LLM planning pipeline, capstone architecture, safety validation)
3. External resource links (OpenAI docs, Anthropic docs, ROS 2 tutorials, research papers)
4. Docusaurus build test and syntax validation
5. Performance verification (build time, render FPS)

---

## Acceptance Criteria

### Content Completeness

- [ ] All 15 functional requirements (FR-001 to FR-015) mapped to content sections
- [ ] Each chapter has introduction, conceptual explanation, code examples, and hands-on exercise
- [ ] Module 4 index page includes learning objectives, prerequisites, estimated duration, API costs
- [ ] All code examples include complete context (imports, dependencies, expected output, API key handling)

### Technical Accuracy

- [ ] Whisper installation verified with base model on Ubuntu 22.04 + Python 3.10+
- [ ] OpenAI GPT-4 API integration tested with robot planning prompts
- [ ] LLaMA 3 local integration tested with Ollama on Ubuntu 22.04
- [ ] ROS 2 action client examples tested with Humble + Isaac Sim
- [ ] All Python scripts are syntactically correct and executable
- [ ] All JSON schemas validated against OpenAI API response format

### Educational Quality

- [ ] Concepts explained before code (theory → practice)
- [ ] Each code example has commented explanations
- [ ] Hands-on exercises have step-by-step instructions
- [ ] Expected outputs documented for verification
- [ ] Troubleshooting sections for common errors (API rate limits, transcription failures, plan validation errors)
- [ ] API cost transparency (per-request pricing, total estimated cost, free alternatives)

### Build & Performance

- [ ] Docusaurus build completes without errors
- [ ] No MDX syntax errors (escaped less than, greater than, & symbols)
- [ ] Site renders at 60 FPS on modern browsers
- [ ] Build time increase less than 15 seconds vs Module 3

---

## Follow-Up Tasks

After `/sp.tasks` generates task breakdown:

1. **Content Creation**: Implement all tasks in dependency order
2. **Code Testing**: Validate all examples in clean environment (Ubuntu 22.04 + Python 3.10 + microphone)
3. **Companion Repo**: Create physical-ai-book-examples/module-4-vla/ with tested code
4. **API Cost Verification**: Test with OpenAI API to confirm $5-10 estimate per student
5. **User Testing**: 5 students test exercises and report completion rate, API cost, transcription accuracy

---

**Next Step**: Run `/sp.tasks` to generate detailed task breakdown from this plan.
