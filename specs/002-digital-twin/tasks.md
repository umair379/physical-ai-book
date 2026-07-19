# Tasks: Module 2 - The Digital Twin (Gazebo & Unity)

**Input**: Design documents from `/specs/002-digital-twin/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Note**: Docusaurus is already initialized in `frontend-book/` directory with Module 1 complete. Tasks will create Module 2 content within that existing project.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus content**: `frontend-book/docs/` (existing directory)
- **Module 2 assets**: `frontend-book/static/img/module-2/`
- **Sidebar config**: `frontend-book/sidebars.ts`
- **Companion repo**: `specs/002-digital-twin/companion-repo-spec.md` (to be created)
- Paths shown below assume repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Configure existing Docusaurus project for Module 2 content

- [x] T001 Verify Docusaurus and Mermaid are still working in frontend-book/ (npm run start test)
- [x] T002 [P] Create module-2 content directory at frontend-book/docs/module-2/
- [x] T003 [P] Create assets directory at frontend-book/static/img/module-2/
- [x] T004 Update frontend-book/sidebars.ts to add Module 2 navigation structure (after Module 1)
- [x] T005 [P] Verify .gitignore still covers node_modules/, .docusaurus/, build/ in frontend-book/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create module overview page at frontend-book/docs/module-2/index.md with learning objectives, prerequisites (Module 1), and chapter structure
- [x] T007 [P] Create companion repository structure specification at specs/002-digital-twin/companion-repo-spec.md (similar to Module 1 pattern, covering Gazebo worlds, Unity projects, sensor configs)
- [x] T008 [P] Add Module 2 intro/teaser to frontend-book/docs/intro.md linking to module-2/index.md

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Gazebo Physics Simulation (Priority: P1) 🎯 MVP

**Goal**: Teach Gazebo physics engine architecture and simulate realistic humanoid robot dynamics

**Independent Test**: Reader can create Gazebo world with humanoid robot URDF, configure Bullet physics, observe realistic falling/collision behavior

### Content Creation for User Story 1

- [x] T009 [P] [US1] Create Chapter 1 file at frontend-book/docs/module-2/chapter-1-gazebo-physics.md with frontmatter (sidebar_position: 2, title, description, keywords)
- [x] T010 [US1] Write "Introduction: Physics Simulation in Robotics" section explaining why physics engines matter for digital twins
- [x] T011 [US1] Write "Gazebo Physics Engine Architecture" section (FR-001) explaining ODE, Bullet, DART with decision tree from research.md
- [x] T012 [US1] Create physics engine comparison Mermaid table showing ODE vs Bullet vs DART (ease, performance, use cases) per research.md
- [x] T013 [US1] Write "Bullet Physics Configuration" section (FR-002) with default world file example: timestep 1ms, real-time factor 1.0, gravity -9.81
- [x] T014 [US1] Add Bullet world file code example (XML) with commented parameters: `<physics type="bullet">`, `<max_step_size>0.001</max_step_size>`, ground plane friction 0.8
- [x] T015 [US1] Write "Gravity and World Configuration" section explaining gravity vector, ground plane setup, physics timestep trade-offs
- [x] T016 [US1] Write "Collision Detection Configuration" section (FR-003) explaining collision geometries, contact properties, surface friction
- [x] T017 [US1] Add collision geometry code example (XML in URDF) showing visual vs collision elements, box/sphere/mesh geometries
- [x] T018 [US1] Create callout box (warning) about common pitfall: missing collision geometry → limbs pass through each other
- [x] T019 [US1] Write "Understanding Inertia Tensors" section (FR-005) explaining mass, center of mass, inertia calculation formula I = (1/12)*m*(h² + d²)
- [x] T020 [US1] Add inertia tensor code example (XML in URDF) with correct vs incorrect inertia values and their effects
- [x] T021 [US1] Create callout box (tip) with inertia calculator tool reference from companion repo
- [x] T022 [US1] Write "Complete Humanoid URDF Example" section (FR-004) integrating physics, collision, inertia for realistic falling robot
- [x] T023 [US1] Add complete humanoid URDF code example with Bullet-tuned parameters: damping 5.0, friction 1.0, proper inertia
- [x] T024 [US1] Write "Common Physics Pitfalls" section covering: timestep too large, unrealistic inertia, friction confusion, damping issues (from research.md)
- [x] T025 [US1] Create troubleshooting table showing pitfall → symptom → solution (e.g., "Timestep >5ms" → "Collision penetration" → "Use 1-2ms")
- [x] T026 [US1] Write "Hands-On Exercise: Falling Humanoid Robot" section (FR-006) with step-by-step instructions
- [x] T027 [US1] Add hands-on exercise steps: 1) Create world file, 2) Add humanoid URDF, 3) Launch Gazebo, 4) Observe falling behavior, 5) Measure FPS
- [x] T028 [US1] Add expected output description: robot falls to ground, limbs collide realistically, real-time factor ~1.0 on GTX 1060
- [x] T029 [US1] Add external link: Gazebo tutorials (http://gazebosim.org/tutorials) with description
- [x] T030 [US1] Add external link: Bullet physics documentation (https://pybullet.org/) with description
- [x] T031 [US1] Add external link: URDF inertia specification (http://wiki.ros.org/urdf/XML/inertia) with description
- [x] T032 [US1] Add callout boxes (info, tip, success) for: physics engine selection, timestep defaults, expected FPS benchmarks

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Unity Environments (Priority: P2)

**Goal**: Teach Unity environment design with photorealistic rendering and ROS 2 integration

**Independent Test**: Reader can create Unity scene with humanoid robot, design indoor environment with interactive objects, export camera feed to ROS 2 topic

### Content Creation for User Story 2

- [x] T0 [P] [US2] Create Chapter 2 file at frontend-book/docs/module-2/chapter-2-unity-environments.md with frontmatter
- [x] T0 [US2] Write "Introduction: Unity for Robot Simulation" section explaining Unity's strengths (rendering, environments) vs Gazebo
- [x] T0 [US2] Write "Unity Robotics Hub Setup" section (FR-010) with installation instructions for Unity 2022.3 LTS on Ubuntu 22.04
- [x] T0 [US2] Add Unity Robotics Hub installation steps: 1) Install Unity Hub, 2) Install Unity 2022.3 LTS, 3) Add ROS2-For-Unity package via Package Manager
- [x] T0 [US2] Create callout box (warning) about common Ubuntu 22.04 issue: Unity Hub AppImage not launching → use .deb package
- [x] T0 [US2] Write "Importing Robot Models" section (FR-008) covering URDF and FBX import workflows
- [x] T0 [US2] Add robot import code example showing Unity Robotics Hub URDF importer usage and FBX manual import steps
- [x] T0 [US2] Write "Environment Design Basics" section (FR-009) explaining GameObjects, terrain, lighting, physics settings
- [x] T0 [US2] Add environment design example: create indoor scene with walls, floor, furniture (5+ interactive objects)
- [x] T0 [US2] Write "Photorealistic Lighting and Materials" section (FR-007) covering Unity's rendering pipeline, HDRP/URP, shadows, reflections
- [x] T0 [US2] Add lighting configuration example showing directional light, area lights, baked vs real-time lighting for 60+ FPS
- [x] T0 [US2] Create performance callout box (tip): GTX 1060 can handle 5 interactive objects + robot at 60 FPS with standard rendering
- [x] T0 [US2] Write "Unity-ROS 2 Integration Workflow" section (FR-010) explaining ROSConnection, topic publishing, message serialization
- [x] T0 [US2] Add Unity ROS connection C# code example: ROSConnection.cs script establishing ws://localhost:10000 connection
- [x] T0 [US2] Create ROS 2 integration Mermaid diagram: Unity Camera → CameraPublisher.cs → ROS2-For-Unity → sensor_msgs/Image → /camera/image_raw
- [x] T0 [US2] Write "Camera Sensor Configuration" section (FR-011) explaining RGB cameras, depth cameras, segmentation cameras in Unity
- [x] T0 [US2] Add camera publisher C# code example: CameraPublisher.cs capturing Unity camera frame and converting to sensor_msgs/Image
- [x] T0 [US2] Add depth camera export C# code example: converting Unity depth buffer to sensor_msgs/Image (16UC1 format, millimeters)
- [x] T0 [US2] Write "Interactive Objects and Physics" section explaining Rigidbody components, colliders, joint constraints for doors/drawers
- [x] T0 [US2] Add interactive object C# script example: door hinge joint triggered by robot end-effector collision
- [x] T0 [US2] Write "Hands-On Exercise: Indoor Environment with HRI" section (FR-012) with step-by-step Unity project creation
- [x] T0 [US2] Add hands-on exercise steps: 1) Create Unity project, 2) Import robot, 3) Design room, 4) Add 5 interactive objects, 5) Configure ROS 2 publisher, 6) Test in Unity Editor
- [x] T0 [US2] Add expected output description: Unity scene renders at 60+ FPS, ROS 2 terminal shows /camera/image_raw at 20-30 Hz, objects interact with physics
- [x] T0 [US2] Write "Troubleshooting Unity on Linux" section covering: Python version conflicts, LD_LIBRARY_PATH, ROS_DOMAIN_ID=0 configuration
- [x] T0 [US2] Add external link: Unity Robotics Hub GitHub (https://github.com/Unity-Technologies/Unity-Robotics-Hub) with description
- [x] T0 [US2] Add external link: Unity Learn tutorials (https://learn.unity.com) with description for environment design
- [x] T0 [US2] Add external link: ROS2-For-Unity package documentation with version compatibility notes (Unity 2022.3 + ROS 2 Humble)
- [x] T0 [US2] Add callout boxes (info, tip, success) for: Unity Hub installation, ROS 2 connection verification, expected FPS on GTX 1060

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Sensor Simulation (Priority: P3)

**Goal**: Teach realistic sensor simulation (LiDAR, depth cameras, IMUs) with noise models and ROS 2 integration

**Independent Test**: Reader can configure multi-sensor robot in Gazebo, add realistic noise, subscribe to sensor topics in ROS 2, implement basic sensor fusion

### Content Creation for User Story 3

- [x] T0 [P] [US3] Create Chapter 3 file at frontend-book/docs/module-2/chapter-3-sensor-simulation.md with frontmatter
- [x] T0 [US3] Write "Introduction: Sensor Simulation for Perception" section explaining why simulated sensors enable perception algorithm development
- [x] T0 [US3] Write "LiDAR Simulation in Gazebo" section (FR-013) explaining ray sensor plugin, range, resolution, FOV, update rate
- [x] T0 [US3] Add Gazebo LiDAR plugin code example (XML/SDF): `<sensor type="ray">`, 360 samples, 30m range, 20 Hz, Gaussian noise σ=0.01
- [x] T0 [US3] Create LiDAR configuration Mermaid diagram showing: Gazebo ray sensor → ros_gz_bridge → sensor_msgs/LaserScan → /lidar/scan topic
- [x] T0 [US3] Write "Depth Camera Simulation" section (FR-014) covering Gazebo and Unity depth camera configuration
- [x] T0 [US3] Add Gazebo depth camera code example (XML/SDF): `<sensor type="depth_camera">`, 640x480 resolution, 20 Hz, noise σ=0.02
- [x] T0 [US3] Add Unity depth camera C# script example: capturing depth buffer, converting to sensor_msgs/Image with 16UC1 encoding
- [x] T0 [US3] Write "IMU Sensor Simulation" section (FR-015) explaining accelerometer, gyroscope, magnetometer data generation
- [x] T0 [US3] Add Gazebo IMU plugin code example (XML/SDF): `<sensor type="imu">`, 100 Hz update rate, noise on linear_acceleration and angular_velocity
- [x] T0 [US3] Write "Realistic Noise Models" section (FR-017) covering Gaussian noise (1-3%), motion blur (frame averaging), occlusion filtering
- [x] T0 [US3] Add noise configuration code examples showing: Gaussian noise in Gazebo (`<noise type="gaussian">`, `<stddev>0.01</stddev>`)
- [x] T0 [US3] Create sensor noise comparison table: LiDAR (1-3% Gaussian), Depth camera (2% Gaussian + motion blur), IMU (0.01 rad/s gyro noise)
- [x] T0 [US3] Write "Gazebo vs Unity for Sensor Simulation" section (FR-018) with decision guidance table from research.md
- [x] T0 [US3] Create Gazebo vs Unity comparison table: Physics integration, Sensor plugins, Noise models, ROS 2 support, Learning curve, Best for
- [x] T0 [US3] Add recommendation callout (tip): Use Gazebo for sensor fundamentals (native plugins), Unity for photorealistic camera data
- [x] T0 [US3] Write "Sensor Synchronization with ROS 2" section explaining ApproximateTimeSynchronizer, ±100ms slop, queue size
- [x] T0 [US3] Add ROS 2 sensor synchronization Python code example: message_filters.ApproximateTimeSynchronizer for LiDAR + depth + IMU
- [x] T0 [US3] Create multi-sensor fusion Mermaid diagram: /lidar/scan + /camera/depth/image + /imu/data → ApproximateTimeSynchronizer → sensor_fusion_callback
- [x] T0 [US3] Write "Performance Optimization for Student Hardware" section covering: ray count limits, depth resolution, GPU budget on GTX 1060
- [x] T0 [US3] Add performance guidance table: GTX 1060 3GB → 360 rays LiDAR (20 Hz) + 640x480 depth (20 Hz) + IMU (100 Hz) = 60% GPU utilization
- [x] T0 [US3] Write "Subscribing to Sensor Topics in ROS 2" section (FR-016) with Python subscriber examples for each sensor type
- [x] T0 [US3] Add ROS 2 LiDAR subscriber code example: `rclpy.create_subscription(LaserScan, '/lidar/scan', callback)`
- [x] T0 [US3] Add ROS 2 depth camera subscriber code example: processing sensor_msgs/Image (16UC1) to numpy array
- [x] T0 [US3] Add ROS 2 IMU subscriber code example: extracting linear_acceleration.x, y, z and angular_velocity.x, y, z
- [x] T0 [US3] Write "Hands-On Exercise: Multi-Sensor Fusion" section (FR-019) with step-by-step robot configuration and fusion algorithm
- [x] T0 [US3] Add hands-on exercise steps: 1) Create Gazebo world with multi-sensor robot, 2) Configure LiDAR + depth + IMU, 3) Launch simulation, 4) Run ROS 2 subscribers, 5) Implement basic fusion
- [x] T0 [US3] Add basic sensor fusion code example: combining LiDAR range data with IMU orientation for obstacle detection
- [x] T0 [US3] Add expected output description: All sensors publishing to topics, ApproximateTimeSynchronizer synchronizing within 100ms, fusion algorithm running at 10 Hz
- [x] T0 [US3] Add external link: Gazebo sensor plugins documentation (http://gazebosim.org/tutorials?cat=sensors)
- [x] T0 [US3] Add external link: ROS 2 message_filters package (https://github.com/ros-perception/message_filters) with synchronization examples
- [x] T0 [US3] Add external link: ROS 2 sensor_msgs documentation for LaserScan, Image, Imu message types
- [x] T0 [US3] Add callout boxes (info, tip, success) for: sensor noise parameters, GTX 1060 performance limits, expected synchronization latency

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [x] T [P] Review all chapters for consistent tone (conversational, second person "you") and technical accuracy
- [x] T [P] Verify all code examples have proper syntax highlighting (```xml for Gazebo world files, ```csharp for Unity scripts, ```python for ROS 2)
- [x] T [P] Check all external links use stable URLs (http://gazebosim.org not latest/, Unity Robotics Hub specific commit/tag)
- [x] T [P] Validate all Mermaid diagrams render correctly with npm run start in frontend-book/
- [x] T [P] Add alt text to all placeholder images (if any diagrams exported as images instead of Mermaid)
- [x] T Test full Docusaurus build with npm run build in frontend-book/ (ensure Module 2 builds without errors)
- [x] T [P] Update frontend-book/docs/intro.md to mention Module 2 availability and link to module-2/index.md
- [x] T [P] Add "Next Steps" section to Chapter 3 linking to companion repository setup and Module 3 (if planned)
- [x] T Verify sidebar navigation order: Intro → Module 1 → Module 2 (Chapter 1 → Chapter 2 → Chapter 3)
- [x] T [P] Add meta tags for SEO in each chapter frontmatter (description, keywords: gazebo, unity, lidar, sensors, physics simulation)
- [x] T Validate all acceptance criteria from spec.md are addressed: SC-001 (Gazebo simulation <20 min), SC-002 (physics engine explanation), SC-003 (Unity 60+ FPS), SC-004 (LiDAR <1cm error), SC-005 (Unity-ROS 2 integration), SC-006 (sensor fusion <100ms), SC-007 (90% completion rate content clarity)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1 - Gazebo)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2 - Unity)**: Can start after Foundational (Phase 2) - Independently testable (different tool from Gazebo)
- **User Story 3 (P3 - Sensors)**: Can start after Foundational (Phase 2) - Integrates both Gazebo and Unity but independently testable

### Within Each User Story

- Content creation tasks can run in sequence (T009 → T010 → T011...)
- Tasks marked [P] within a story can run in parallel if resources available
- External links can be added in parallel with content writing
- Callouts can be added in parallel with section writing
- Code examples should be created after section text is drafted (for context)
- Diagrams (Mermaid) should be created after explanatory text is written

### Parallel Opportunities

- **Phase 1 Setup**: T002, T003, T005 can run in parallel
- **Phase 2 Foundational**: T007, T008 can run in parallel
- **Within User Stories**: Content sections for different chapters can be written in parallel
  - Chapter 1 sections (T010-T032) can be written by one person
  - Chapter 2 sections (T034-T060) can be written by another person in parallel
  - Chapter 3 sections (T062-T093) can be written by a third person in parallel
- **Phase 6 Polish**: Most polish tasks (T094-T104) can run in parallel except T099 (build test) should run after content validation

---

## Parallel Example: All User Stories

```bash
# After Foundational phase completes, launch all user stories in parallel:

# Team Member A: User Story 1 (Chapter 1 - Gazebo)
Task: "Create Chapter 1 file at frontend-book/docs/module-2/chapter-1-gazebo-physics.md"
Task: "Write Gazebo Physics Engine Architecture section..."
Task: "Create physics engine comparison Mermaid table..."
# Continue with T009-T032

# Team Member B: User Story 2 (Chapter 2 - Unity) - SIMULTANEOUSLY
Task: "Create Chapter 2 file at frontend-book/docs/module-2/chapter-2-unity-environments.md"
Task: "Write Unity Robotics Hub Setup section..."
Task: "Create ROS 2 integration Mermaid diagram..."
# Continue with T033-T060

# Team Member C: User Story 3 (Chapter 3 - Sensors) - SIMULTANEOUSLY
Task: "Create Chapter 3 file at frontend-book/docs/module-2/chapter-3-sensor-simulation.md"
Task: "Write LiDAR Simulation in Gazebo section..."
Task: "Create multi-sensor fusion Mermaid diagram..."
# Continue with T061-T093
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T008) - **CRITICAL GATE**
3. Complete Phase 3: User Story 1 (T009-T032)
4. **STOP and VALIDATE**: Test Chapter 1 independently
   - Verify all Mermaid diagrams render
   - Check all code examples have correct syntax highlighting (XML for Gazebo)
   - Validate external links work (Gazebo tutorials, Bullet docs, URDF spec)
   - Read through for clarity and flow
   - Verify research.md decisions (Bullet primary) are reflected correctly
5. Deploy/preview if ready (npm run build && npm run serve)

**MVP Deliverable**: Chapter 1 (Gazebo Physics Simulation) complete and validated

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Preview (MVP!)
3. Add User Story 2 → Test independently → Deploy/Preview
4. Add User Story 3 → Test independently → Deploy/Preview
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T008)
2. Once Foundational is done:
   - Developer A: User Story 1 (T009-T032) - Gazebo chapter
   - Developer B: User Story 2 (T033-T060) - Unity chapter
   - Developer C: User Story 3 (T061-T093) - Sensors chapter
3. Stories complete and integrate independently
4. All developers: Polish phase (T094-T104) in parallel

---

## Notes

- **[P] tasks**: Different files, no dependencies - can run in parallel
- **[Story] label**: Maps task to specific user story for traceability
- Each user story should be independently completable and testable
- **Existing Docusaurus**: Tasks assume frontend-book/ directory already exists with Docusaurus 3.9.2 and Module 1 complete
- **Research Integration**: Content tasks implement decisions from research.md (Bullet physics, Unity Robotics Hub 0.7.0+, Gazebo sensors)
- **Companion repository**: Code examples reference companion repo structure defined in companion-repo-spec.md (to be created in T007)
- **No tests**: Educational content validation is manual (reading, link checking, diagram rendering, build success)
- Commit after each completed user story (not after every task)
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

---

## Validation Checklist (Before Completion)

- [ ] All 3 chapters created in frontend-book/docs/module-2/
- [ ] All Mermaid diagrams render correctly (physics comparison, Unity-ROS pipeline, sensor fusion)
- [ ] All code examples have proper syntax highlighting (XML for Gazebo, C# for Unity, Python for ROS 2)
- [ ] All external links use stable URLs and return 200 status (Gazebo tutorials, Unity Robotics Hub, ROS 2 docs)
- [ ] Sidebar navigation includes Module 2 with all chapters
- [ ] Docusaurus build succeeds (npm run build in frontend-book/)
- [ ] Content aligns with learning outcomes from spec.md (physics engines, Unity rendering, sensor simulation)
- [ ] Acceptance criteria from spec.md verified: SC-001 to SC-007 addressed in content
- [ ] Research.md decisions correctly implemented: Bullet primary (not ODE), Unity Robotics Hub (not ROS-TCP-Connector), Gazebo sensors (with Unity comparison)
- [ ] Performance guidance for GTX 1060 included in all relevant sections

---

**Total Tasks**: 104
**User Story 1 (Gazebo)**: 24 tasks (T009-T032)
**User Story 2 (Unity)**: 28 tasks (T033-T060)
**User Story 3 (Sensors)**: 33 tasks (T061-T093)
**Setup + Foundational**: 8 tasks (T001-T008)
**Polish**: 11 tasks (T094-T104)
