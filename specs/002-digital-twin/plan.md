# Implementation Plan: Module 2 - The Digital Twin (Gazebo & Unity)

**Branch**: `002-digital-twin` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-digital-twin/spec.md`

**User Intent**: Add Module 2 to the Docusaurus project and create 3 chapters (Gazebo Physics Simulation, Unity Environments, Sensor Simulation) as .md files. Write concise explanations, diagrams, and configuration examples for Gazebo physics, Unity rendering, and simulated sensors.

## Summary

Create Module 2 of the Physical AI Book as Docusaurus Markdown chapters teaching physics-based simulation with Gazebo, high-fidelity environment design with Unity, and sensor simulation for robotics. This module builds on Module 1 (ROS 2 fundamentals and URDF modeling) to enable students to create complete digital twins of humanoid robots in virtual environments.

**Technical Approach**: Add three new Markdown chapter files to the existing Docusaurus project at `frontend-book/docs/module-2/`, create module overview page, embed Gazebo world file examples and Unity configuration snippets as fenced code blocks, include Mermaid diagrams for physics concepts and sensor pipelines, and link to companion repository for complete simulation environments.

## Technical Context

**Language/Version**: JavaScript/Node.js 18+ (Docusaurus build), Markdown/MDX (content authoring)
**Primary Dependencies**: Docusaurus 3.9.2 (already installed), @docusaurus/theme-mermaid 3.9.2 (already installed), React 19.0 (Docusaurus framework)
**Storage**: Git repository (GitHub) for version control, Companion repository for Gazebo worlds, Unity scenes, and sensor configurations
**Testing**: Manual content validation, Link validation for external resources (Gazebo tutorials, Unity Robotics Hub docs), Visual verification of Mermaid diagrams, Code example validation in companion repo CI/CD
**Target Platform**: Web browsers (static HTML/CSS/JS served via GitHub Pages)
**Project Type**: Documentation website (Docusaurus-based static site) - extending existing frontend-book/ project
**Performance Goals**: Fast page loads (<2s initial, <500ms navigation), Mermaid diagram rendering <1s, Good Lighthouse scores (>90 performance)
**Constraints**: Static site only (no backend for Module 2), All Gazebo/Unity examples must be reproducible on Ubuntu 22.04 + ROS 2 Humble, Markdown-only content (no custom React components), Must integrate seamlessly with existing Module 1 structure
**Scale/Scope**: 3 chapters for Module 2, ~12-18 pages total content (including module overview, 3 chapters with multiple sections), 15-25 code examples (Gazebo world files, Unity C# scripts, sensor configs), Companion repository with tested simulation environments

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Specification-First Development ✅
- **Status**: PASS
- **Evidence**: Module 2 content maps directly to spec.md user stories (US1: Gazebo Physics P1, US2: Unity Environments P2, US3: Sensor Simulation P3)
- **Verification**: Each chapter corresponds to a prioritized user story with acceptance criteria (15 acceptance scenarios total)

### Principle II: Accuracy and Non-Hallucination ✅
- **Status**: PASS
- **Evidence**: All Gazebo world files and Unity configurations will be stored in companion repository and tested before inclusion. External links reference official documentation (Gazebo: http://gazebosim.org/tutorials, Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- **Verification**: FR-001 to FR-019 require complete, runnable examples. Spec mandates exact versions (Gazebo Garden/11, Unity 2022.3 LTS, Unity Robotics Hub 0.7.0)
- **Risk Mitigation**: Companion repository uses CI/CD to validate all simulation examples build and run correctly

### Principle III: Reproducibility and Developer Clarity ✅
- **Status**: PASS
- **Evidence**: Spec Assumptions section documents exact environment (Ubuntu 22.04, ROS 2 Humble, Gazebo Garden/11, Unity 2022 LTS, GPU: GTX 1060 equivalent). FR-006, FR-012, FR-019 specify hands-on exercises with expected outputs
- **Verification**: SC-001 requires students can create Gazebo simulation in under 20 minutes. SC-007 targets 90% reader completion rate
- **Implementation**: Quickstart.md will document installation steps for Gazebo, Unity Hub, and Unity Robotics Hub with troubleshooting for common Linux issues

### Principle IV: AI-Native Authoring ✅
- **Status**: PASS
- **Evidence**: Module created via /sp.specify, planned via /sp.plan, will generate tasks via /sp.tasks
- **Verification**: This plan.md generated through Spec-Kit Plus workflow. PHR 006 documents specification phase

### Principle V: Modular and Clean Architecture ✅
- **Status**: PASS
- **Evidence**: Module 2 is self-contained in `frontend-book/docs/module-2/` directory. Three chapters are independently readable (Chapter 1 Gazebo-only, Chapter 2 Unity-only, Chapter 3 integrates both)
- **Verification**: No cross-module content dependencies beyond Module 1 prerequisite knowledge. Companion repository organized by chapter
- **Structure**: Follows Module 1 pattern (module-N/index.md + chapter-N-name.md files)

### Principle VI: Security and Secrets Management ✅
- **Status**: PASS (N/A for Module 2)
- **Evidence**: Module 2 contains only educational content about local simulation tools. No secrets, API keys, or credentials required
- **Verification**: Gazebo and Unity run locally without cloud authentication. If future cloud alternatives added (Risk 4 mitigation), will use .env pattern
- **Future**: When RAG chatbot added (later module), will use .env for API keys per constitution

### Principle VII: Testability and Verification ✅
- **Status**: PASS
- **Evidence**: Spec includes 7 measurable success criteria (SC-001 to SC-007) with quantitative metrics (time: "under 20 minutes", performance: "60+ FPS", accuracy: "< 1cm error at 5m range")
- **Verification**: 15 acceptance scenarios (5 per user story) define testable outcomes. FR-006, FR-012, FR-019 specify hands-on exercises with verification steps
- **Testing Strategy**: Companion repository CI/CD validates Gazebo worlds load, Unity scenes build, sensor data publishes to ROS 2 topics

**Overall Constitution Compliance**: ✅ **PASS** - All principles satisfied

**Complexity Justification**: None required - implementation extends existing Docusaurus project (approved stack) with educational Markdown content

---

**Post-Design Re-Evaluation** (After Phase 1):

All principles remain satisfied after detailed design:

- ✅ **Principle I**: Content structure in data-model.md maps all chapters to user stories (P1: Gazebo, P2: Unity, P3: Sensors)
- ✅ **Principle II**: research.md documents physics engine selection (Bullet primary), Unity Robotics Hub version (0.7.0+), sensor noise models (Gaussian 1-3%)
- ✅ **Principle III**: research.md provides exact configurations (timestep 1ms, damping 3-8, friction 0.8-1.0), quickstart guidance for Ubuntu 22.04 setup
- ✅ **Principle IV**: All artifacts generated via Spec-Kit Plus workflow (plan.md, research.md, data-model.md)
- ✅ **Principle V**: Module 2 self-contained in frontend-book/docs/module-2/, companion repo organized by chapter
- ✅ **Principle VI**: N/A for Module 2 - local simulation tools require no secrets
- ✅ **Principle VII**: data-model.md defines validation checklist (Gazebo loads, Unity compiles, sensors publish, physics >= 0.9x real-time)

**No new risks or violations identified**. Design ready for task generation via `/sp.tasks`.

## Project Structure

### Documentation (this feature)

```text
specs/002-digital-twin/
├── spec.md              # Feature specification (created via /sp.specify)
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output - Gazebo/Unity/sensor decisions
├── data-model.md        # Phase 1 output - Content structure model
├── quickstart.md        # Phase 1 output - Gazebo/Unity installation guide
├── contracts/           # Phase 1 output - Content organization schema
│   └── content-structure.yaml  # Chapter/section organization
├── checklists/
│   └── requirements.md  # Specification quality checklist (PASS)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

**Structure Decision**: Documentation website using Docusaurus. This module extends the existing `frontend-book/` Docusaurus project initialized for Module 1. Content lives in `docs/module-2/` as Markdown files, following the established pattern.

```text
physical-ai-book/                 # Repository root
├── frontend-book/                # Existing Docusaurus project
│   ├── docs/                     # Docusaurus content directory
│   │   ├── intro.md              # Landing page (existing)
│   │   ├── module-1/             # Module 1: ROS 2 (existing, completed)
│   │   │   ├── index.md
│   │   │   ├── chapter-1-fundamentals.md
│   │   │   ├── chapter-2-python-integration.md
│   │   │   └── chapter-3-urdf-modeling.md
│   │   └── module-2/             # Module 2: Digital Twin (NEW - this feature)
│   │       ├── index.md          # Module 2 overview page
│   │       ├── chapter-1-gazebo-physics.md        # Chapter 1: Gazebo Physics Simulation
│   │       ├── chapter-2-unity-environments.md    # Chapter 2: Unity Environments
│   │       └── chapter-3-sensor-simulation.md     # Chapter 3: Sensor Simulation
│   ├── static/                   # Static assets
│   │   └── img/
│   │       ├── module-1/         # Module 1 assets (existing)
│   │       └── module-2/         # Module 2 assets (NEW)
│   │           ├── gazebo-physics-engines.png      # Placeholder for diagram
│   │           ├── unity-ros-pipeline.png          # Placeholder for diagram
│   │           └── sensor-noise-comparison.png     # Placeholder for diagram
│   ├── sidebars.ts               # Navigation configuration (UPDATE - add Module 2)
│   ├── docusaurus.config.ts      # Docusaurus config (existing, no changes needed)
│   ├── package.json              # Dependencies (existing, Mermaid already added)
│   └── .gitignore                # Git ignore patterns (existing)
├── specs/                        # Spec-Kit Plus documentation
│   ├── 001-ros2-module/          # Module 1 spec (existing)
│   └── 002-digital-twin/         # Module 2 spec (this feature)
└── history/
    └── prompts/
        └── 002-digital-twin/     # PHRs for this feature
```

**Companion Repository** (separate GitHub repo - to be created):

```text
physical-ai-book-examples/        # Companion code repository (existing from Module 1)
├── module-1-ros2/                # Module 1 examples (existing)
└── module-2-digital-twin/        # Module 2 examples (NEW)
    ├── chapter-1-gazebo-physics/
    │   ├── README.md             # Chapter 1 examples overview
    │   ├── simple_world.world    # Basic Gazebo world with gravity
    │   ├── humanoid_physics.world # Humanoid robot with physics config
    │   ├── collision_demo.world  # Collision detection example
    │   ├── urdf/
    │   │   └── simple_humanoid.urdf  # Humanoid URDF for physics demo
    │   └── expected_output.txt   # What to see in Gazebo GUI
    ├── chapter-2-unity-environments/
    │   ├── README.md             # Chapter 2 examples overview
    │   ├── UnityProject/         # Unity 2022.3 LTS project
    │   │   ├── Assets/
    │   │   │   ├── Scenes/
    │   │   │   │   └── IndoorEnvironment.unity  # Example scene
    │   │   │   ├── Scripts/
    │   │   │   │   ├── ROSConnection.cs         # ROS 2 integration script
    │   │   │   │   └── CameraPublisher.cs       # Camera sensor publisher
    │   │   │   └── RobotModels/
    │   │   │       └── HumanoidRobot.fbx        # Imported robot model
    │   │   └── Packages/
    │   │       └── manifest.json # Unity Robotics Hub dependency
    │   └── expected_output.txt   # What to see in Unity + ROS 2 topics
    └── chapter-3-sensor-simulation/
        ├── README.md             # Chapter 3 examples overview
        ├── gazebo_sensors/
        │   ├── lidar_world.world # Gazebo world with LiDAR-equipped robot
        │   ├── depth_camera.world # Gazebo depth camera example
        │   └── imu_robot.world   # IMU sensor configuration
        ├── unity_sensors/
        │   ├── SensorScene.unity # Unity scene with multi-sensor robot
        │   └── Scripts/
        │       ├── LidarSimulator.cs    # LiDAR simulation script
        │       ├── DepthCameraROS.cs    # Depth camera ROS publisher
        │       └── IMUPublisher.cs      # IMU data publisher
        ├── ros2_subscribers/
        │   ├── lidar_subscriber.py      # ROS 2 LiDAR data processor
        │   ├── depth_subscriber.py      # Depth camera data processor
        │   └── sensor_fusion_basic.py   # Simple sensor fusion example
        └── expected_output.txt   # Sensor data examples and verification
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No violations - table not needed*

## Phase 0: Research & Technology Decisions

*Research agents currently running to resolve key decisions. Results will be consolidated here.*

### Research Tasks

1. **Gazebo Physics Engine Selection** (ODE vs Bullet vs DART)
   - Agent researching: When to recommend each engine for educational content
   - Target: Concrete recommendation for Chapter 1 examples

2. **Unity-ROS 2 Integration Approach** (Unity Robotics Hub vs ROS-TCP-Connector)
   - Agent researching: Best tool for beginners, version compatibility, setup process
   - Target: Step-by-step integration strategy for Chapter 2

3. **Sensor Simulation Best Practices** (Gazebo vs Unity, noise models, synchronization)
   - Agent researching: Realistic parameters, performance trade-offs, tool selection
   - Target: Practical guidance for Chapter 3 multi-sensor examples

*Research findings will be documented in `research.md` after agent completion*

## Phase 1: Design & Contracts

*Will be completed after Phase 0 research is consolidated*

### Planned Artifacts

1. **data-model.md**: Content entities (Module, Chapter, Section, CodeExample, Diagram, ExternalLink) - following Module 1 pattern
2. **contracts/content-structure.yaml**: Complete chapter organization with all sections, code examples, diagrams, and external links for Module 2
3. **quickstart.md**: Installation guide for Gazebo Garden/11, Unity 2022 LTS, Unity Robotics Hub on Ubuntu 22.04 with troubleshooting
4. **Agent context update**: Add Module 2 technology stack to CLAUDE.md

---

**Status**: Phase 0 in progress (research agents running). Plan will be completed after research consolidation.
