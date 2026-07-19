# Companion Repository Specification: Module 2

**Repository**: `physical-ai-book-examples` (existing)
**Module Directory**: `module-2-digital-twin/` (NEW)
**Purpose**: Provide complete, tested simulation environments for Module 2 chapters

## Overview

This specification defines the structure and contents of the Module 2 companion repository. All code examples referenced in the Docusaurus content must exist in this repository and be validated by CI/CD before publication.

## Repository Structure

```text
physical-ai-book-examples/        # Companion code repository (existing from Module 1)
├── module-1-ros2/                # Module 1 examples (existing)
└── module-2-digital-twin/        # Module 2 examples (NEW)
    ├── README.md                 # Module 2 examples overview and quickstart
    ├── chapter-1-gazebo-physics/
    │   ├── README.md             # Chapter 1: Gazebo Physics Simulation
    │   ├── simple_world.world    # Basic Gazebo world with gravity
    │   ├── humanoid_physics.world # Humanoid robot with Bullet physics config
    │   ├── collision_demo.world  # Collision detection example
    │   ├── urdf/
    │   │   └── simple_humanoid.urdf  # Humanoid URDF for physics demo
    │   └── expected_output.txt   # What to see in Gazebo GUI (benchmarks, FPS)
    ├── chapter-2-unity-environments/
    │   ├── README.md             # Chapter 2: Unity Environments
    │   ├── UnityProject/         # Unity 2022.3 LTS project
    │   │   ├── Assets/
    │   │   │   ├── Scenes/
    │   │   │   │   └── IndoorEnvironment.unity  # Example scene with interactive objects
    │   │   │   ├── Scripts/
    │   │   │   │   ├── ROSConnection.cs         # ROS 2 integration script
    │   │   │   │   └── CameraPublisher.cs       # Camera sensor publisher
    │   │   │   └── RobotModels/
    │   │   │       └── HumanoidRobot.fbx        # Imported robot model (placeholder)
    │   │   └── Packages/
    │   │       └── manifest.json # Unity Robotics Hub dependency
    │   └── expected_output.txt   # What to see in Unity + ROS 2 topics
    └── chapter-3-sensor-simulation/
        ├── README.md             # Chapter 3: Sensor Simulation
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

## Chapter 1: Gazebo Physics Simulation

### simple_world.world
- **Purpose**: Minimal Gazebo world demonstrating Bullet physics engine configuration
- **Physics Config**: Bullet, 1ms timestep, 1.0x real-time factor, gravity -9.81 m/s²
- **Contents**: Ground plane with friction 0.8, basic lighting
- **Expected Output**: Gazebo loads in <5s, maintains 1.0x real-time factor

### humanoid_physics.world
- **Purpose**: Complete humanoid robot simulation with Bullet-tuned parameters
- **Physics Config**: Bullet, damping 5.0, friction 1.0, proper inertia tensors
- **Contents**: Simple humanoid URDF with 6 DOF (2 legs, torso, arms), collision geometries
- **Expected Output**: Robot falls realistically, no limb penetration, 50-200 Hz simulation on GTX 1060

### collision_demo.world
- **Purpose**: Demonstrate collision detection between robot limbs and objects
- **Physics Config**: Bullet, surface friction 0.5-1.5, contact properties
- **Contents**: Humanoid robot, obstacles (boxes, spheres), visual vs collision geometry examples
- **Expected Output**: Robot detects collisions, no pass-through, contact forces visualized

### urdf/simple_humanoid.urdf
- **Purpose**: Minimal humanoid URDF for physics demonstrations
- **Links**: base_link (torso), left_leg, right_leg, left_arm, right_arm
- **Joints**: 6 revolute joints with damping 5.0, friction 1.0
- **Inertia**: Correct inertia tensors (I = 1/12 * m * (h² + d²) for boxes)
- **Collision**: All links have collision geometry matching visual

## Chapter 2: Unity Environments

### UnityProject/
- **Unity Version**: 2022.3 LTS (exact version documented in README)
- **Packages**: Unity Robotics Hub 0.7.0+ (ROS2-For-Unity)
- **Platform**: Tested on Ubuntu 22.04 with ROS 2 Humble

### Assets/Scenes/IndoorEnvironment.unity
- **Purpose**: Indoor HRI scenario with interactive objects
- **Contents**: Room (walls, floor, ceiling), furniture (table, chairs), 5 interactive objects (grabbable)
- **Lighting**: Mixed lighting (directional + point lights), baked lightmaps
- **Performance**: 60+ FPS on GTX 1060 at 1080p

### Assets/Scripts/ROSConnection.cs
- **Purpose**: Establish WebSocket connection to ROS 2
- **Configuration**: ws://localhost:10000, ROS_DOMAIN_ID=0
- **Output**: Console log "ROS 2 connection established"
- **Error Handling**: Retry logic, connection status UI

### Assets/Scripts/CameraPublisher.cs
- **Purpose**: Publish Unity camera frames to ROS 2 topic
- **Topic**: /unity/camera/image_raw (sensor_msgs/Image)
- **Format**: RGB8, 640x480, 30 Hz
- **Performance**: <10ms latency from frame capture to ROS 2 publish

## Chapter 3: Sensor Simulation

### gazebo_sensors/lidar_world.world
- **Purpose**: LiDAR sensor configuration example
- **Sensor**: ray sensor plugin, 360 samples, 0.1-30m range, 20 Hz
- **Noise**: Gaussian, mean=0, stddev=0.01 (1%)
- **Output**: ROS 2 topic /lidar/scan (sensor_msgs/LaserScan)

### gazebo_sensors/depth_camera.world
- **Purpose**: Depth camera configuration example
- **Sensor**: depth camera plugin, 640x480, 20 Hz
- **Noise**: Gaussian, stddev=0.02 (2%)
- **Output**: ROS 2 topic /depth/image (sensor_msgs/Image)

### gazebo_sensors/imu_robot.world
- **Purpose**: IMU sensor configuration example
- **Sensor**: IMU plugin, 100 Hz, linear acceleration + angular velocity
- **Noise**: Gaussian on both measurements
- **Output**: ROS 2 topic /imu/data (sensor_msgs/Imu)

### unity_sensors/SensorScene.unity
- **Purpose**: Multi-sensor Unity scene
- **Sensors**: LiDAR (raycasting), depth camera, IMU (Unity physics)
- **Robot**: Humanoid with sensors attached to base_link
- **Performance**: 60% GPU utilization on GTX 1060 (all sensors active)

### ros2_subscribers/sensor_fusion_basic.py
- **Purpose**: Demonstrate ROS 2 message_filters synchronization
- **Inputs**: /lidar/scan, /depth/image, /imu/data
- **Synchronization**: ApproximateTimeSynchronizer, ±100ms slop
- **Output**: Fused odometry estimate printed to console

## Validation Checklist

All examples must pass these validation criteria before publication:

### Gazebo Examples
- [ ] All .world files load without errors in Gazebo Garden/11
- [ ] Physics simulations achieve >= 0.9x real-time factor on GTX 1060
- [ ] URDF files have correct inertia tensors (verified with inertia calculator)
- [ ] Collision geometries prevent limb penetration

### Unity Examples
- [ ] UnityProject compiles in Unity 2022.3 LTS on Ubuntu 22.04
- [ ] All C# scripts have no compilation errors
- [ ] ROS 2 topics publish successfully (verified with ros2 topic echo)
- [ ] Scenes render at 60+ FPS on GTX 1060 at 1080p

### ROS 2 Integration
- [ ] All sensor plugins publish to correct ROS 2 topics
- [ ] Message types match sensor_msgs standard (LaserScan, Image, Imu)
- [ ] Synchronization works with ApproximateTimeSynchronizer
- [ ] Topic Hz matches configuration (verified with ros2 topic hz)

## CI/CD Pipeline

The companion repository will include GitHub Actions workflow:

1. **Gazebo Validation**: Launch all .world files, verify they load without errors
2. **Unity Compilation**: Compile C# scripts, check for errors (Unity headless mode)
3. **ROS 2 Topics**: Launch simulations, verify topics publish at correct Hz
4. **Documentation**: Validate all README files have correct structure and links

## Expected Output Files

Each chapter includes `expected_output.txt` with:

- **Console Logs**: What to see in terminal (ROS 2 topics, Gazebo messages)
- **Performance Benchmarks**: FPS, real-time factor, GPU utilization
- **Verification Commands**: Commands to run to verify setup (ros2 topic list, gazebo --version)
- **Sample Data**: Example sensor data values (LiDAR ranges, image dimensions, IMU readings)

## Usage Instructions

Each chapter README.md includes:

1. **Prerequisites**: Software versions, hardware requirements
2. **Setup**: Installation steps for Gazebo/Unity/ROS 2
3. **Running Examples**: Step-by-step commands to launch simulations
4. **Verification**: How to verify examples work correctly
5. **Troubleshooting**: Common issues and solutions (from research.md)

## Version Control

- **Git LFS**: Use for large binary files (.fbx models, .unity scenes, .world files >1MB)
- **Branching**: main branch = stable, dev branch = work in progress
- **Tagging**: Tag releases matching Docusaurus module versions (v2.0.0 for Module 2)

## External Links

The companion repository README will link to:

- **Gazebo Tutorials**: http://gazebosim.org/tutorials
- **Unity Robotics Hub**: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- **ROS 2 Humble Docs**: https://docs.ros.org/en/humble/
- **Bullet Physics**: https://pybullet.org/

---

**Status**: Specification complete, ready for companion repository creation
**Next Step**: Create companion repository and populate with validated examples
