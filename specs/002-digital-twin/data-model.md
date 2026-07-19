# Data Model: Module 2 - The Digital Twin (Gazebo & Unity)

**Date**: 2025-12-25
**Phase**: 1 - Design & Contracts
**Purpose**: Define content structure model for Module 2

## Content Entities

Module 2 follows the same content entity model as Module 1 (Module, Chapter, Section, CodeExample, Diagram, ExternalLink). See Module 1 data-model.md for detailed entity definitions.

---

## Module 2 Specific Instance

### Module Entity

```yaml
module_id: "module-2"
module_number: 2
title: "The Digital Twin (Gazebo & Unity)"
description: "Learn physics-based simulation with Gazebo, create high-fidelity environments in Unity, and simulate realistic sensors (LiDAR, depth cameras, IMUs) for robotic systems."
learning_objectives:
  - "Configure Gazebo physics engines (ODE, Bullet, DART) for humanoid robot simulation"
  - "Design photorealistic Unity environments with interactive objects for HRI scenarios"
  - "Simulate multi-sensor systems (LiDAR, depth cameras, IMUs) with realistic noise models"
  - "Integrate Unity scenes with ROS 2 using Unity Robotics Hub"
  - "Implement basic sensor fusion algorithms combining LiDAR, depth, and IMU data"
prerequisites:
  - "Module 1 completion (ROS 2 fundamentals, URDF modeling)"
  - "Basic 3D math understanding (vectors, rotations, transforms)"
  - "Ubuntu 22.04 with ROS 2 Humble, Gazebo Garden/11, Unity 2022 LTS installed"
estimated_duration: "4-5 hours"
```

---

## Chapter Instances

### Chapter 1: Gazebo Physics Simulation

```yaml
chapter_id: "chapter-1-gazebo-physics"
chapter_number: 1
title: "Gazebo Physics Simulation"
description: "Understand Gazebo's physics engine architecture and simulate realistic humanoid robot dynamics with gravity, collisions, and rigid-body mechanics."
learning_outcomes:
  - "Explain differences between ODE, Bullet, and DART physics engines"
  - "Configure gravity, friction, and damping parameters in Gazebo world files"
  - "Implement collision detection for humanoid robot links"
  - "Calculate inertia tensors for robot links based on geometry"
  - "Debug common physics simulation pitfalls (timestep, inertia, friction)"
sections:
  - "physics-engine-overview"
  - "gravity-and-world-configuration"
  - "collision-detection-setup"
  - "inertia-and-center-of-mass"
  - "hands-on-falling-robot"
code_examples:
  - "bullet_world_config"
  - "humanoid_urdf_physics"
  - "collision_geometry_example"
  - "inertia_calculator"
external_links:
  - "gazebo-tutorials"
  - "bullet-physics-docs"
  - "urdf-inertia-spec"
estimated_reading_time: 60
```

### Chapter 2: Unity Environments

```yaml
chapter_id: "chapter-2-unity-environments"
chapter_number: 2
title: "High-Fidelity Environments with Unity"
description: "Create photorealistic indoor environments in Unity with interactive objects, lighting, and ROS 2 integration for human-robot interaction scenarios."
learning_outcomes:
  - "Import robot models (URDF/FBX) into Unity projects"
  - "Design interactive environments with movable objects and physics"
  - "Configure Unity's rendering pipeline for realistic lighting and materials"
  - "Integrate Unity scenes with ROS 2 using Unity Robotics Hub"
  - "Export camera and depth sensor data to ROS 2 topics"
sections:
  - "unity-robotics-hub-setup"
  - "importing-robot-models"
  - "environment-design-basics"
  - "lighting-and-materials"
  - "ros2-integration-workflow"
  - "camera-sensor-export"
  - "hands-on-indoor-environment"
code_examples:
  - "unity_ros_connection_script"
  - "camera_publisher_csharp"
  - "depth_camera_ros_export"
  - "interactive_object_script"
external_links:
  - "unity-robotics-hub-github"
  - "unity-learn-tutorials"
  - "ros2-for-unity-docs"
estimated_reading_time: 75
```

### Chapter 3: Sensor Simulation

```yaml
chapter_id: "chapter-3-sensor-simulation"
chapter_number: 3
title: "Sensor Simulation in Virtual Environments"
description: "Simulate realistic LiDAR, depth cameras, and IMU sensors in Gazebo and Unity with noise models, synchronization, and ROS 2 integration for perception algorithms."
learning_outcomes:
  - "Configure Gazebo LiDAR plugins with range, resolution, and noise parameters"
  - "Simulate depth cameras in both Gazebo and Unity with realistic noise"
  - "Set up IMU sensors publishing linear acceleration and angular velocity"
  - "Synchronize multiple sensor streams using ROS 2 message_filters"
  - "Implement basic sensor fusion combining LiDAR, depth, and IMU data"
sections:
  - "lidar-simulation-gazebo"
  - "depth-camera-configuration"
  - "imu-sensor-setup"
  - "sensor-noise-models"
  - "gazebo-vs-unity-sensors"
  - "sensor-synchronization"
  - "hands-on-multisensor-fusion"
code_examples:
  - "lidar_plugin_sdf"
  - "depth_camera_sdf"
  - "imu_sensor_sdf"
  - "ros2_sensor_subscriber"
  - "approximate_time_sync"
  - "basic_sensor_fusion"
external_links:
  - "gazebo-sensor-plugins"
  - "ros2-message-filters"
  - "sensor-msgs-docs"
estimated_reading_time: 70
```

---

## CodeExample Instances (Sampling)

### Example: Bullet World Configuration

```yaml
example_id: "bullet_world_config"
title: "Bullet Physics World Configuration"
language: "xml"
file_path: "module-2-digital-twin/chapter-1-gazebo-physics/simple_world.world"
explanation: "Demonstrates configuring Gazebo world file with Bullet physics engine, realistic gravity, and ground plane friction for humanoid simulation."
expected_output: |
  Gazebo GUI loads with ground plane visible.
  Physics engine shows "Bullet" in simulation settings.
  Real-time factor maintains ~1.0x with default timestep.
dependencies:
  - "Gazebo Garden or Gazebo 11"
run_instructions: |
  1. gazebo simple_world.world
  2. Verify physics engine in World → Physics tab
  3. Observe real-time factor in bottom-left corner
```

### Example: Unity ROS Connection Script

```yaml
example_id: "unity_ros_connection_script"
title: "Unity ROS 2 Connection Script (C#)"
language: "csharp"
file_path: "module-2-digital-twin/chapter-2-unity-environments/UnityProject/Assets/Scripts/ROSConnection.cs"
explanation: "C# script to establish connection between Unity scene and ROS 2 Humble using Unity Robotics Hub package."
expected_output: |
  Unity Console shows: "ROS 2 connection established at ws://localhost:10000"
  ROS 2 terminal: ros2 topic list shows /unity/clock topic
dependencies:
  - "Unity 2022.3 LTS"
  - "Unity Robotics Hub 0.7.0+"
  - "ROS 2 Humble"
run_instructions: |
  1. Attach ROSConnection.cs to GameObject in Unity scene
  2. Configure ROS_DOMAIN_ID=0 in Unity and ROS 2 terminal
  3. Press Play in Unity Editor
  4. ros2 topic list to verify topics
```

### Example: LiDAR Plugin SDF

```yaml
example_id: "lidar_plugin_sdf"
title: "Gazebo LiDAR Sensor Plugin Configuration"
language: "xml"
file_path: "module-2-digital-twin/chapter-3-sensor-simulation/gazebo_sensors/lidar_world.world"
explanation: "Configures Gazebo ray sensor plugin to simulate 360-degree 2D LiDAR with Gaussian noise and 20 Hz update rate."
expected_output: |
  Gazebo loads robot with visible LiDAR rays (enable visualization).
  ROS 2 terminal: ros2 topic echo /lidar/scan shows sensor_msgs/LaserScan messages at ~20 Hz.
  Range values: 0.1m to 30m with ~1% noise.
dependencies:
  - "Gazebo Garden or Gazebo 11"
  - "ROS 2 Humble"
  - "ros_gz_bridge"
run_instructions: |
  1. gazebo lidar_world.world
  2. ros2 topic hz /lidar/scan (verify 20 Hz)
  3. ros2 topic echo /lidar/scan | head -20
```

---

## Diagram Instances (Sampling)

### Diagram: Physics Engine Comparison

```yaml
diagram_id: "physics-engine-comparison-table"
title: "Gazebo Physics Engine Comparison (ODE vs Bullet vs DART)"
type: "table"
source: |
  | Criterion | ODE | Bullet | DART |
  |-----------|-----|--------|------|
  | Ease of Setup | Moderate | Easy ✓ | Moderate |
  | Performance | 10-50 Hz | 50-200 Hz ✓ | 50-300 Hz |
  | Gazebo 11+ Default | No | Yes ✓ | No |
alt_text: "Comparison table showing Bullet as the recommended physics engine for educational humanoid simulation"
caption: "Bullet offers the best balance of performance and ease-of-use for beginners (research.md Decision 1)"
```

### Diagram: Unity-ROS 2 Pipeline

```yaml
diagram_id: "unity-ros-pipeline-mermaid"
title: "Unity to ROS 2 Data Flow"
type: "mermaid"
source: |
  graph LR
      A[Unity Camera] --> B[CameraPublisher.cs]
      B --> C[ROS2-For-Unity]
      C --> D[sensor_msgs/Image]
      D --> E[ROS 2 Topic: /camera/image_raw]
      E --> F[Perception Algorithm]
alt_text: "Flowchart showing Unity camera frame being serialized by C# script, sent through ROS2-For-Unity middleware, published as sensor_msgs/Image to ROS 2 topic for perception processing"
caption: "Unity camera data flows through Unity Robotics Hub to ROS 2 topics for perception algorithms (FR-010, FR-011)"
```

### Diagram: Sensor Fusion Architecture

```yaml
diagram_id: "sensor-fusion-architecture"
title: "Multi-Sensor Fusion with message_filters"
type: "mermaid"
source: |
  graph TD
      A[LiDAR /scan] --> D[ApproximateTimeSynchronizer]
      B[Depth Camera /depth/image] --> D
      C[IMU /imu/data] --> D
      D --> E[sensor_fusion_callback]
      E --> F[Fused Odometry Estimate]
alt_text: "Architecture diagram showing three sensor topics (LiDAR, depth camera, IMU) synchronized by ROS 2 message_filters ApproximateTimeSynchronizer before processing in sensor fusion callback"
caption: "ROS 2 message_filters synchronizes multi-sensor data within ±100ms window for sensor fusion (FR-019, research.md Decision 3)"
```

---

## External Link Instances (Sampling)

```yaml
link_id: "gazebo-tutorials"
title: "Gazebo Official Tutorials"
url: "http://gazebosim.org/tutorials"
description: "Comprehensive Gazebo tutorials covering sensors, plugins, and physics configuration for robot simulation."
link_type: "official_docs"
```

```yaml
link_id: "unity-robotics-hub-github"
title: "Unity Robotics Hub Repository"
url: "https://github.com/Unity-Technologies/Unity-Robotics-Hub"
description: "Official Unity Technologies repository for ROS 2 integration packages, examples, and setup instructions."
link_type: "official_docs"
```

```yaml
link_id: "ros2-message-filters"
title: "ROS 2 message_filters Package"
url: "https://github.com/ros-perception/message_filters"
description: "Time synchronization utilities for combining multiple ROS 2 sensor topics with ApproximateTimeSynchronizer."
link_type: "reference"
```

---

## Content Quality Constraints

Same as Module 1 (see Module 1 data-model.md), with additions:

### Simulation-Specific Constraints

**Gazebo World Files**:
- Use SDF 1.9 format for Gazebo Garden (or SDF 1.6 for Gazebo 11)
- Include comments explaining each physics parameter
- Provide default values optimized for GTX 1060-level hardware (research.md Decision 3)

**Unity C# Scripts**:
- Follow Unity C# naming conventions (PascalCase for public members)
- Include XML documentation comments for public methods
- Specify exact Unity Robotics Hub version in package.json

**Configuration Examples**:
- All sensor configurations must include realistic noise models
- Performance guidance for student hardware (GTX 1060 3GB VRAM)
- Expected outputs include FPS, topic Hz, and sample data ranges

---

## Validation Checklist (Before "Published" State)

- [ ] All Gazebo world files load without errors in Gazebo Garden/11
- [ ] All Unity C# scripts compile in Unity 2022.3 LTS
- [ ] All sensor plugins publish to correct ROS 2 topics
- [ ] Physics simulations achieve >= 0.9x real-time factor on GTX 1060
- [ ] All external links return 200 status (official docs accessible)
- [ ] All Mermaid diagrams render correctly in Docusaurus
- [ ] Learning outcomes align with spec.md success criteria (SC-001 to SC-007)
- [ ] Companion repository CI/CD validates all examples

---

## Relationship to Spec Entities

| Spec Entity | Content Entity | Relationship |
|-------------|----------------|-------------|
| User Story 1 (Gazebo Physics P1) | Chapter 1 | 1:1 mapping (MVP) |
| User Story 2 (Unity Environments P2) | Chapter 2 | 1:1 mapping |
| User Story 3 (Sensor Simulation P3) | Chapter 3 | 1:1 mapping |
| FR-001 to FR-006 (Gazebo) | Chapter 1 Sections + CodeExamples | Requirements → Content |
| FR-007 to FR-012 (Unity) | Chapter 2 Sections + CodeExamples | Requirements → Content |
| FR-013 to FR-019 (Sensors) | Chapter 3 Sections + CodeExamples | Requirements → Content |
| SC-001 to SC-007 | Entire Module 2 + Companion Repo | Success validation |
| Research Decisions | All Chapters | Tech choices inform content |

---

## Notes

- **Research Integration**: Content creation directly implements research.md decisions (Bullet primary, Unity Robotics Hub, Gazebo-first sensors)
- **Performance Focus**: All examples optimized for student hardware (GTX 1060 benchmarks in research.md)
- **Troubleshooting**: Quickstart.md will document Ubuntu 22.04 setup issues (Unity Hub, Python conflicts, ROS_DOMAIN_ID)
- **Companion Repo**: Separate repository structure mirrors chapter organization (module-2-digital-twin/chapter-N/)
