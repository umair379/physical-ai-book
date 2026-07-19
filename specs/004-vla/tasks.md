---
description: "Task list for Module 4: Vision-Language-Action (VLA) implementation"
---

# Tasks: Module 4 - Vision-Language-Action (VLA)

**Input**: Design documents from `/specs/004-vla/`
**Prerequisites**: plan.md (implementation strategy), spec.md (3 user stories: P1 Voice-to-Action, P2 LLM Planning, P3 Capstone)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Documentation: `frontend-book/docs/module-4/`
- Static assets: `frontend-book/static/img/module-4/`
- Configuration: `frontend-book/sidebars.ts`, `frontend-book/docs/intro.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure for Module 4

- [X] T001 Create module-4 directory in frontend-book/docs/module-4/
- [X] T002 [P] Create static assets directory in frontend-book/static/img/module-4/
- [X] T003 Update sidebars.ts with Module 4 navigation entry
- [X] T004 Create module-4/index.md overview page with learning objectives, prerequisites, and API cost transparency
- [X] T005 Update frontend-book/docs/intro.md with Module 4 teaser section

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Design artifacts and companion repository specification that inform all user story implementations

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create specs/004-vla/data-model.md with Module 4 instance, Chapter 1-3 entities, learning outcomes, code examples, and estimated reading times
- [X] T007 Create specs/004-vla/companion-repo-spec.md with physical-ai-book-examples/module-4-vla/ directory structure (chapter-1-voice-to-action/, chapter-2-llm-planning/, chapter-3-capstone/)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Voice-to-Action with Speech Recognition (Priority: P1) 🎯 MVP

**Goal**: Students learn to convert natural language voice commands into structured robot actions using OpenAI Whisper for speech-to-text, intent parsing, and ROS 2 action mapping.

**Independent Test**: Student speaks "Move forward two meters" → Whisper transcribes → extracts intent (move_forward, distance: 2.0) → ROS 2 action executes → robot moves 2 meters

### Implementation for User Story 1

#### Introduction & Prerequisites (FR-001)

- [X] T008 [P] [US1] Create chapter-1-voice-to-action.md introduction section explaining Voice-to-Action paradigm, VLA foundations, and chapter scope
- [X] T009 [P] [US1] Add prerequisites section in chapter-1-voice-to-action.md (Module 1 ROS 2 actions, Python 3.10+, Ubuntu 22.04, USB microphone)

#### Section 1: OpenAI Whisper Installation (FR-001, FR-002)

- [X] T010 [P] [US1] Write "1.1 What is OpenAI Whisper?" section explaining speech-to-text, model architecture, and use cases in chapter-1-voice-to-action.md
- [X] T011 [P] [US1] Write "1.2 Whisper Model Comparison" section with table comparing tiny/base/small/medium/large (parameters, latency, accuracy, GPU requirement) in chapter-1-voice-to-action.md
- [X] T012 [US1] Write "1.3 Installing Whisper" section with pip installation commands (pip install openai-whisper torch torchvision torchaudio) in chapter-1-voice-to-action.md
- [X] T013 [US1] Add Whisper installation verification code example (Python script loading base model, transcribing test audio) in chapter-1-voice-to-action.md

#### Section 2: Microphone Configuration (FR-001, FR-005)

- [X] T014 [P] [US1] Write "2.1 Microphone Setup on Ubuntu" section with ALSA/PulseAudio configuration in chapter-1-voice-to-action.md
- [X] T015 [P] [US1] Add microphone verification code example using sounddevice library (list devices, test recording) in chapter-1-voice-to-action.md
- [X] T016 [US1] Write "2.2 Real-Time Audio Streaming" section explaining audio buffers, sample rates (16 kHz for Whisper), and streaming patterns in chapter-1-voice-to-action.md
- [X] T017 [US1] Add real-time audio streaming code example (sounddevice callback, numpy arrays, Whisper inference) in chapter-1-voice-to-action.md

#### Section 3: Voice Transcription (FR-002, FR-005)

- [X] T018 [P] [US1] Write "3.1 Whisper Transcription API" section explaining model.transcribe() method, audio parameter, language detection in chapter-1-voice-to-action.md
- [X] T019 [P] [US1] Add basic transcription code example (load audio file, transcribe with base model, print text) in chapter-1-voice-to-action.md
- [X] T020 [US1] Write "3.2 Confidence Scores and Filtering" section explaining no_speech_prob, logprob, 0.7 confidence threshold in chapter-1-voice-to-action.md
- [X] T021 [US1] Add confidence filtering code example (check confidence score, reject low-confidence transcriptions, prompt retry) in chapter-1-voice-to-action.md

#### Section 4: Intent Parsing (FR-003)

- [X] T022 [P] [US1] Write "4.1 Natural Language Intent Extraction" section explaining action type, object parameters, spatial constraints, quantity parsing in chapter-1-voice-to-action.md
- [X] T023 [P] [US1] Add regex-based intent parser code example (patterns for "move forward X meters", "navigate to location", "pick up object") in chapter-1-voice-to-action.md
- [X] T024 [US1] Write "4.2 Advanced Intent Parsing with NLP" section introducing spaCy for entity recognition (optional advanced technique) in chapter-1-voice-to-action.md
- [X] T025 [US1] Add spaCy intent parser code example (dependency parsing, entity extraction) in chapter-1-voice-to-action.md

#### Section 5: ROS 2 Action Mapping (FR-004, FR-005)

- [X] T026 [P] [US1] Write "5.1 Voice Commands to ROS 2 Actions" section explaining action primitives (cmd_vel, NavigateToPose, custom manipulation) in chapter-1-voice-to-action.md
- [X] T027 [P] [US1] Add action mapping code example (intent dictionary to ROS 2 action goal message) in chapter-1-voice-to-action.md
- [X] T028 [US1] Write "5.2 Asynchronous Action Client Pattern" section explaining async action client, feedback callbacks, PENDING → ACTIVE → SUCCEEDED states in chapter-1-voice-to-action.md
- [X] T029 [US1] Add ROS 2 action client code example (create action client, send goal, monitor feedback, handle result) in chapter-1-voice-to-action.md

#### Section 6: Error Handling (FR-005)

- [X] T030 [P] [US1] Write "6.1 Handling Ambiguous Commands" section explaining missing parameters, clarification prompts via TTS in chapter-1-voice-to-action.md
- [X] T031 [P] [US1] Add ambiguity detection code example (check required parameters, prompt user for missing info) in chapter-1-voice-to-action.md
- [X] T032 [US1] Write "6.2 Noise and Transcription Errors" section explaining SNR less than 10 dB, retry strategies, text fallback in chapter-1-voice-to-action.md
- [X] T033 [US1] Add noise handling code example (detect low SNR, ask user to repeat command) in chapter-1-voice-to-action.md

#### Hands-On Exercises (FR-013)

- [X] T034 [P] [US1] Write Exercise 1 in chapter-1-voice-to-action.md: Install Whisper, transcribe pre-recorded commands, measure accuracy
- [X] T035 [P] [US1] Write Exercise 2 in chapter-1-voice-to-action.md: Build intent parser for 10 robot commands, test with voice input
- [X] T036 [US1] Write Exercise 3 in chapter-1-voice-to-action.md: Trigger ROS 2 cmd_vel action with "Move forward 2 meters" voice command, verify robot moves 2 meters in simulation

#### Diagrams & Visual Aids

- [X] T037 [P] [US1] Create Mermaid diagram for voice-to-action pipeline (microphone → Whisper → intent parser → ROS 2 action → robot) in chapter-1-voice-to-action.md
- [X] T038 [P] [US1] Create Mermaid diagram for confidence filtering decision tree (confidence greater than 0.7 → execute, less than 0.7 → retry) in chapter-1-voice-to-action.md

#### Troubleshooting & Tips

- [X] T039 [US1] Add troubleshooting section in chapter-1-voice-to-action.md (microphone not detected, Whisper CUDA errors, low transcription accuracy, ROS 2 action timeout)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - students can speak voice commands and trigger robot actions with 90%+ success rate

---

## Phase 4: User Story 2 - Cognitive Planning with LLMs (Priority: P2)

**Goal**: Students leverage large language models (GPT-4, Claude, LLaMA 3) to translate high-level tasks into multi-step ROS 2 action sequences with validation, execution, and replanning.

**Independent Test**: Student sends "Set the table for dinner" → LLM generates 5+ step plan → robot executes using Nav2 and manipulation actions → task completes autonomously

### Implementation for User Story 2

#### Introduction (FR-006)

- [X] T040 [P] [US2] Create chapter-2-llm-planning.md introduction section explaining cognitive planning, LLM reasoning for robotics, and chapter scope
- [X] T041 [P] [US2] Add prerequisites section in chapter-2-llm-planning.md (US1 completion, OpenAI API key OR Ollama installed, Module 3 Nav2 knowledge)

#### Section 1: LLM Selection for Robot Planning (FR-006)

- [X] T042 [P] [US2] Write "1.1 LLM Options for Robot Planning" section comparing GPT-4 Turbo, Anthropic Claude, LLaMA 3, Gemini (quality, cost, latency, context window) in chapter-2-llm-planning.md
- [X] T043 [P] [US2] Add LLM comparison table in chapter-2-llm-planning.md (model, cost per 1K tokens, latency p95, plan quality rating, context window)
- [X] T044 [US2] Write "1.2 API Cost Estimation" section with cost breakdown ($0.01/1K input $0.03/1K output for GPT-4, estimated $5-10 per student) in chapter-2-llm-planning.md
- [X] T045 [US2] Write "1.3 Free Alternatives" section explaining local LLaMA 3 via Ollama (no API costs, slower, acceptable quality) in chapter-2-llm-planning.md

#### Section 2: OpenAI GPT-4 API Integration (FR-006, FR-007)

- [X] T046 [P] [US2] Write "2.1 OpenAI API Setup" section with API key configuration (.env file, environment variables, never commit keys) in chapter-2-llm-planning.md
- [X] T047 [P] [US2] Add OpenAI API installation code example (pip install openai, import openai, set API key) in chapter-2-llm-planning.md
- [X] T048 [US2] Write "2.2 Chat Completions API" section explaining messages format, system/user roles, response parsing in chapter-2-llm-planning.md
- [X] T049 [US2] Add basic GPT-4 API call code example (send "Clean the room" task, receive text response) in chapter-2-llm-planning.md

#### Section 3: Local LLaMA 3 Integration (FR-006)

- [X] T050 [P] [US2] Write "3.1 Installing Ollama for Local LLMs" section with Ollama installation on Ubuntu 22.04 in chapter-2-llm-planning.md
- [X] T051 [P] [US2] Add Ollama setup code example (curl install, ollama pull llama3, ollama run llama3) in chapter-2-llm-planning.md
- [X] T052 [US2] Write "3.2 LLaMA 3 API Integration" section explaining Ollama REST API, localhost:11434, JSON request format in chapter-2-llm-planning.md
- [X] T053 [US2] Add LLaMA 3 API call code example (requests library, send task, parse response) in chapter-2-llm-planning.md

#### Section 4: Prompt Engineering for Robot Planning (FR-007, FR-008)

- [X] T054 [P] [US2] Write "4.1 System Prompts for Robot Planning" section explaining role definition, action vocabulary, JSON schema enforcement in chapter-2-llm-planning.md
- [X] T055 [P] [US2] Add system prompt template code example (robot planner role, available actions: navigate_to, grasp_object, place_object, open_door, close_door, JSON output format) in chapter-2-llm-planning.md
- [X] T056 [US2] Write "4.2 JSON Schema Validation" section explaining jsonschema library, required fields (step_id, action_type, parameters), valid action types in chapter-2-llm-planning.md
- [X] T057 [US2] Add JSON schema validation code example (define schema, validate LLM response, reject invalid plans) in chapter-2-llm-planning.md

#### Section 5: Plan Validation (FR-008, FR-011)

- [X] T058 [P] [US2] Write "5.1 Feasibility Checking" section explaining action type exists, parameters in valid ranges, required objects present, dependencies satisfied in chapter-2-llm-planning.md
- [X] T059 [P] [US2] Add plan validation code example (check each step's action type, validate parameters, verify object existence in environment) in chapter-2-llm-planning.md
- [X] T060 [US2] Write "5.2 Safety Constraints" section explaining velocity limits (max 1.0 m/s), workspace boundaries, object weight limits (max 50 N), collision avoidance in chapter-2-llm-planning.md
- [X] T061 [US2] Add safety validation code example (clamp velocities, check workspace bounds, reject unsafe plans) in chapter-2-llm-planning.md

#### Section 6: Plan Execution (FR-008, FR-010)

- [X] T062 [P] [US2] Write "6.1 Converting Plans to ROS 2 Actions" section explaining step-by-step execution, action goal creation, sequential vs parallel execution in chapter-2-llm-planning.md
- [X] T063 [P] [US2] Add plan executor code example (iterate steps, create ROS 2 action goals from JSON, send to action servers, wait for completion) in chapter-2-llm-planning.md
- [X] T064 [US2] Write "6.2 Integrating with Nav2 and Perception" section explaining navigate_to uses Nav2, grasp_object uses YOLOv8 detection, object locations from Isaac ROS in chapter-2-llm-planning.md
- [X] T065 [US2] Add Nav2 integration code example (LLM plan step → NavigateToPose goal → send to Nav2 action server) in chapter-2-llm-planning.md

#### Section 7: Error Handling and Replanning (FR-009)

- [X] T066 [P] [US2] Write "7.1 Detecting Action Failures" section explaining ABORTED state, timeout detection, force sensor thresholds in chapter-2-llm-planning.md
- [X] T067 [P] [US2] Add failure detection code example (monitor action result, detect ABORTED, capture failure context) in chapter-2-llm-planning.md
- [X] T068 [US2] Write "7.2 Sending Failure Context to LLM" section explaining context serialization (current step, failure reason, environment state), replanning request in chapter-2-llm-planning.md
- [X] T069 [US2] Add replanning code example (send failure context to LLM, request alternative plan, validate new plan, resume execution within 5 seconds) in chapter-2-llm-planning.md

#### Hands-On Exercises (FR-014)

- [X] T070 [P] [US2] Write Exercise 1 in chapter-2-llm-planning.md: Design LLM prompt for "Set the table", send to GPT-4 or LLaMA 3, generate 5+ step plan with valid JSON
- [X] T071 [P] [US2] Write Exercise 2 in chapter-2-llm-planning.md: Validate plan for infeasible actions (unreachable locations, missing objects, unsafe velocities)
- [X] T072 [US2] Write Exercise 3 in chapter-2-llm-planning.md: Execute "Prepare coffee" 7-step plan in Isaac Sim, test failure recovery with obstacle blocking navigation

#### Diagrams & Visual Aids

- [X] T073 [P] [US2] Create Mermaid diagram for LLM planning pipeline (task input → system prompt → LLM → JSON plan → validation → ROS 2 execution) in chapter-2-llm-planning.md
- [X] T074 [P] [US2] Create Mermaid diagram for replanning flow (action executes → fails → capture context → send to LLM → new plan → retry) in chapter-2-llm-planning.md
- [X] T075 [P] [US2] Create Mermaid diagram for safety validation (plan → check velocity → check workspace → check force → PASS/FAIL) in chapter-2-llm-planning.md

#### Troubleshooting & Tips

- [X] T076 [US2] Add troubleshooting section in chapter-2-llm-planning.md (OpenAI API rate limit 429, invalid JSON from LLM, timeout greater than 10s, plan validation failures)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - students can voice-trigger single actions (US1) OR generate multi-step LLM plans and execute them (US2)

---

## Phase 5: User Story 3 - Capstone Project: Autonomous Humanoid (Priority: P3)

**Goal**: Students integrate all modules (ROS 2, simulation, Isaac perception, Nav2, voice, LLM) into a complete autonomous humanoid system executing voice-commanded multi-step tasks with 90%+ success rate.

**Independent Test**: Student speaks "Bring me the blue bottle from the shelf" → full pipeline executes (Whisper → intent → LLM plan → Nav2 nav → YOLO detection → grasp → return) → robot delivers bottle with 90%+ success

### Implementation for User Story 3

#### Introduction (FR-012)

- [X] T077 [P] [US3] Create chapter-3-capstone.md introduction section explaining capstone as culmination of Modules 1-4, end-to-end VLA pipeline, and chapter scope
- [X] T078 [P] [US3] Add prerequisites section in chapter-3-capstone.md (US1+US2 completion, Module 3 Isaac/Nav2, microphone, LLM API, Isaac Sim environment)

#### Section 1: Capstone Architecture (FR-012)

- [X] T079 [P] [US3] Write "1.1 System Architecture Overview" section explaining full pipeline components (voice handler, LLM planner, Nav2 controller, perception handler, manipulation controller) in chapter-3-capstone.md
- [X] T080 [P] [US3] Create Mermaid diagram for capstone architecture (voice input → Whisper → intent parser → LLM planner → action executor → Nav2/YOLO/manipulation → robot) in chapter-3-capstone.md
- [X] T081 [US3] Write "1.2 Data Flow and State Management" section explaining execution context (robot position, battery, gripper state, detected objects, task progress) in chapter-3-capstone.md

#### Section 2: Voice Handler Module (FR-012)

- [X] T082 [P] [US3] Write "2.1 Voice Handler Implementation" section explaining microphone streaming, Whisper transcription, intent extraction as integrated module in chapter-3-capstone.md
- [X] T083 [P] [US3] Add voice handler code example (VoiceHandler class, listen() method, transcribe() method, extract_intent() method) in chapter-3-capstone.md

#### Section 3: LLM Planner Module (FR-012)

- [X] T084 [P] [US3] Write "3.1 LLM Planner Implementation" section explaining task decomposition, plan generation, validation, JSON output in chapter-3-capstone.md
- [X] T085 [P] [US3] Add LLM planner code example (LLMPlanner class, generate_plan() method, validate_plan() method, GPT-4 or LLaMA 3 integration) in chapter-3-capstone.md

#### Section 4: Navigation Controller Module (FR-012)

- [X] T086 [P] [US3] Write "4.1 Nav2 Controller Implementation" section explaining NavigateToPose action client, waypoint following, obstacle avoidance in chapter-3-capstone.md
- [X] T087 [P] [US3] Add Nav2 controller code example (NavController class, navigate_to() method, send NavigateToPose goal, monitor feedback) in chapter-3-capstone.md

#### Section 5: Perception Handler Module (FR-012)

- [X] T088 [P] [US3] Write "5.1 YOLOv8 Perception Integration" section explaining object detection queries, filtering by color/class, spatial reasoning with LLM in chapter-3-capstone.md
- [X] T089 [P] [US3] Add perception handler code example (PerceptionHandler class, detect_objects() method, YOLOv8 inference, query results) in chapter-3-capstone.md

#### Section 6: Manipulation Controller Module (FR-012)

- [X] T090 [P] [US3] Write "6.1 Manipulation with Retry Logic" section explaining grasp action, force sensor feedback, 3 retry attempts with adjusted positions in chapter-3-capstone.md
- [X] T091 [P] [US3] Add manipulation controller code example (ManipulationController class, grasp_object() method, retry logic, force threshold detection) in chapter-3-capstone.md

#### Section 7: Full Pipeline Integration (FR-012)

- [X] T092 [US3] Write "7.1 Orchestrating the Pipeline" section explaining main execution loop, module coordination, state transitions (LISTENING → PLANNING → NAVIGATING → GRASPING → RETURNING) in chapter-3-capstone.md
- [X] T093 [US3] Add capstone main pipeline code example (CapstoneMain class, execute_task() method, integrates VoiceHandler + LLMPlanner + NavController + PerceptionHandler + ManipulationController) in chapter-3-capstone.md

#### Section 8: Dynamic Obstacle Handling (FR-012)

- [X] T094 [P] [US3] Write "8.1 Handling Dynamic Obstacles" section explaining Nav2 local planner re-routing, DWB controller, real-time obstacle detection in chapter-3-capstone.md
- [X] T095 [P] [US3] Add dynamic obstacle code example (Nav2 detects obstacle → local planner re-routes → task continues without failure) in chapter-3-capstone.md

#### Section 9: Ambiguity Resolution (FR-012)

- [X] T096 [P] [US3] Write "9.1 Resolving Ambiguous Commands" section explaining multiple object candidates, LLM reasoning (closest, most accessible), user clarification prompts in chapter-3-capstone.md
- [X] T097 [P] [US3] Add ambiguity resolution code example (YOLO detects 2 red cups → LLM decides based on proximity → or asks user "Which red cup?") in chapter-3-capstone.md

#### Section 10: Success Metrics and Evaluation (FR-012, FR-015)

- [X] T098 [P] [US3] Write "10.1 Task Completion Metrics" section explaining success rate calculation (completed tasks / total tasks), 90%+ target, failure taxonomy in chapter-3-capstone.md
- [X] T099 [P] [US3] Add metrics tracking code example (TaskMetrics class, log task start/end, calculate success rate, report failures) in chapter-3-capstone.md

#### Hands-On Exercises (FR-015)

- [X] T100 [P] [US3] Write Exercise 1 in chapter-3-capstone.md: Complete capstone setup (install all dependencies, configure Whisper + LLM + Nav2 + YOLO + manipulation)
- [X] T101 [P] [US3] Write Exercise 2 in chapter-3-capstone.md: Execute "Bring me the blue bottle from the shelf" with full pipeline, verify 90%+ success rate
- [X] T102 [US3] Write Exercise 3 in chapter-3-capstone.md: Test 10 different voice commands sequentially (bring, find, clean, set table), measure task completion rate (target: 9/10 = 90%+)

#### Diagrams & Visual Aids

- [X] T103 [P] [US3] Create Mermaid sequence diagram for full capstone execution (user speaks → Whisper → intent → LLM → Nav2 → YOLO → grasp → return → success) in chapter-3-capstone.md
- [X] T104 [P] [US3] Create Mermaid state diagram for task execution states (IDLE → LISTENING → PLANNING → NAVIGATING → DETECTING → GRASPING → RETURNING → SUCCESS/FAILURE) in chapter-3-capstone.md

#### Troubleshooting & Tips

- [X] T105 [US3] Add troubleshooting section in chapter-3-capstone.md (integration errors, topic mismatches, action server timeouts, YOLO detection failures, grasp failures, LLM format errors)

**Checkpoint**: All user stories should now be independently functional - students can execute voice-commanded multi-step autonomous tasks with 90%+ completion rate

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [X] T106 [P] Add cross-reference links in module-4/index.md to all 3 chapters
- [X] T107 [P] Add cross-reference links in chapter-1-voice-to-action.md to Chapter 2 (LLM planning), Chapter 3 (capstone), Module 1 (ROS 2 actions)
- [X] T108 [P] Add cross-reference links in chapter-2-llm-planning.md to Chapter 1 (voice), Chapter 3 (capstone), Module 3 (Nav2, YOLOv8)
- [X] T109 [P] Add cross-reference links in chapter-3-capstone.md to all previous chapters and modules
- [X] T110 [P] Add external resource links in module-4/index.md (OpenAI Whisper GitHub, OpenAI API docs, Anthropic Claude docs, ROS 2 action tutorials)
- [X] T111 [P] Add external resource links in chapter-1-voice-to-action.md (Whisper model card, sounddevice docs, spaCy docs)
- [X] T112 [P] Add external resource links in chapter-2-llm-planning.md (OpenAI Chat Completions API, Ollama docs, jsonschema library)
- [X] T113 [P] Add external resource links in chapter-3-capstone.md (Nav2 docs, Isaac ROS perception, YOLOv8 docs)
- [X] T114 Add callout boxes in chapter-1-voice-to-action.md (info boxes for tips, warning boxes for API costs, danger boxes for common errors)
- [X] T115 Add callout boxes in chapter-2-llm-planning.md (cost warnings, safety validation warnings, API rate limit warnings)
- [X] T116 Add callout boxes in chapter-3-capstone.md (integration complexity warnings, success rate expectations)
- [X] T117 Run Docusaurus build test and validate no errors (npm run build in frontend-book/)
- [X] T118 Validate no MDX syntax errors (check all less than, greater than, & symbols are escaped or written as text)
- [X] T119 Verify all code blocks have language tags (```python, ```bash, ```json, ```yaml)
- [X] T120 Check build time increase is less than 15 seconds compared to Module 3 baseline

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User Story 1 (P1 Voice-to-Action): Can start after Foundational - No dependencies on other stories
  - User Story 2 (P2 LLM Planning): Can start after Foundational - Independent but builds on voice concepts from US1
  - User Story 3 (P3 Capstone): Requires US1 and US2 completion for integration examples
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - Fully independent
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent but references US1 concepts
- **User Story 3 (P3)**: Requires US1 and US2 completion - Integrates all previous work

### Within Each User Story

- Introduction and prerequisites MUST be written first
- Sections can be written in parallel if they cover different topics
- Code examples should follow their corresponding concept sections
- Hands-on exercises written after all concept sections are complete
- Diagrams can be created in parallel with content
- Troubleshooting sections written last after all content is complete

### Parallel Opportunities

- **Setup (Phase 1)**: T002 can run in parallel with T001
- **Foundational (Phase 2)**: T006 and T007 can run in parallel
- **User Story 1**: Within each section, concept writing and code examples can be parallelized (e.g., T010 and T011, T014 and T015)
- **User Story 2**: Similar parallelization within sections
- **User Story 3**: Similar parallelization within sections
- **Polish (Phase 6)**: Most tasks (T106-T116) can run in parallel, except build tests (T117-T120) must run sequentially after content is complete

---

## Parallel Example: User Story 1 - Voice-to-Action

```bash
# After Foundational phase completes, launch all User Story 1 introduction tasks in parallel:
Task: "Create chapter-1-voice-to-action.md introduction section" (T008)
Task: "Add prerequisites section in chapter-1-voice-to-action.md" (T009)

# Launch all Section 1 tasks in parallel:
Task: "Write What is OpenAI Whisper section" (T010)
Task: "Write Whisper Model Comparison section" (T011)

# After concepts are written, launch Section 2 tasks in parallel:
Task: "Write Microphone Setup section" (T014)
Task: "Add microphone verification code example" (T015)

# Continue this pattern for all sections
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Voice-to-Action)
4. **STOP and VALIDATE**: Test voice commands trigger robot actions with 90%+ success
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (Voice-to-Action) → Test independently with voice commands → Deploy/Demo (MVP!)
3. Add User Story 2 (LLM Planning) → Test independently with multi-step plans → Deploy/Demo
4. Add User Story 3 (Capstone) → Test full pipeline with 90%+ success rate → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Voice-to-Action) - T008-T039
   - Developer B: User Story 2 (LLM Planning) - T040-T076
   - Developer C: User Story 3 (Capstone) - T077-T105 (waits for US1+US2 concepts to reference)
3. Stories complete and integrate independently
4. Team collaborates on Polish phase (T106-T120)

---

## Notes

- [P] tasks = different files/sections, no dependencies within phase
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each section or logical group of tasks
- Stop at any checkpoint to validate story independently
- All code examples MUST include complete imports, dependencies, API key handling via environment variables
- API cost transparency is critical - document estimated costs and free alternatives in every relevant section
- Safety constraints (velocity limits, workspace boundaries, force limits) must be validated before plan execution
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
