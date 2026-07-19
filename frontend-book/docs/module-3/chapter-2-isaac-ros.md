# Chapter 2: Isaac ROS for Perception & Localization

Learn how NVIDIA Isaac ROS provides GPU-accelerated perception pipelines for real-time SLAM, object detection, and semantic segmentation on humanoid robots.

## Introduction: GPU-Accelerated Perception

Traditional ROS 2 perception packages run on CPU, limiting performance to 5-15 FPS for complex tasks like SLAM or semantic segmentation. **Isaac ROS** leverages NVIDIA GPUs to accelerate perception pipelines by 10-50x, enabling real-time performance (30+ FPS) for humanoid robot navigation and manipulation.

### Why Isaac ROS for Digital Twins?

A **digital twin** for perception validates that your AI models work correctly before deploying to physical hardware. Isaac ROS provides:

- **GPU Acceleration**: CUDA-optimized ROS 2 nodes for SLAM, DNN inference, image processing
- **Hardware Consistency**: Same codebase runs on Jetson (embedded) and desktop RTX GPUs
- **Isaac Sim Integration**: Seamless connection between synthetic data generation and perception testing
- **Pre-trained Models**: Production-ready DNNs for segmentation, detection, depth estimation

### Isaac ROS vs Traditional ROS 2 Perception

| Feature | Traditional ROS 2 | Isaac ROS ✓ Recommended |
|---------|------------------|------------------------|
| **SLAM Performance** | 5-10 Hz (CPU) | **30+ Hz (GPU)** |
| **DNN Inference** | 1-5 FPS (CPU) | **30-60 FPS (TensorRT GPU)** |
| **Point Cloud Processing** | 10-20 Hz (PCL) | **60+ Hz (CUDA)** |
| **Jetson Support** | Limited | **Native optimized** |
| **Isaac Sim Bridge** | Manual setup | **Official bridge** |
| **Learning Curve** | Low | **Medium** |

:::tip When to Use Isaac ROS
Use Isaac ROS when:
1. You have an NVIDIA GPU (RTX 2060+ or Jetson Xavier/Orin)
2. You need real-time perception (>20 FPS)
3. You want seamless Isaac Sim integration for testing
4. You're deploying to Jetson embedded platforms
:::

---

## What is Isaac ROS?

**Isaac ROS** is a collection of GPU-accelerated ROS 2 packages for perception, localization, and manipulation. It uses NVIDIA CUDA for parallel processing and TensorRT for optimized deep learning inference.

### Isaac ROS Architecture

```mermaid
graph TD
    A[Sensor Data] --> B[Isaac ROS Nodes]
    B --> C[Graph Execution Manager]
    C --> D[CUDA Kernels]
    C --> E[TensorRT Engine]
    D --> F[Processed Output]
    E --> F
    F --> G[/map Topic]
    F --> H[/odom Topic]
    F --> I[/detections Topic]
```

**Key Components**:

1. **GEM (Graph Execution Manager)**: Optimizes ROS 2 node graphs for GPU execution
2. **CUDA Kernels**: Custom GPU code for image processing, point clouds, SLAM
3. **TensorRT**: NVIDIA's inference optimizer (converts PyTorch/ONNX to optimized engine)
4. **Hardware Abstraction**: Same API for Jetson and desktop RTX GPUs

### Isaac ROS Package Categories

| Package | Function | Performance Gain | Use Case |
|---------|----------|-----------------|----------|
| **nvblox_ros** | 3D SLAM + occupancy mapping | 10-20x vs CPU | Humanoid navigation, obstacle avoidance |
| **isaac_ros_dnn_inference** | TensorRT DNN inference | 20-50x vs CPU | Object detection, segmentation |
| **isaac_ros_depth_segmentation** | Depth-based segmentation | 15-30x vs CPU | Person tracking, workspace monitoring |
| **isaac_ros_apriltag** | Fiducial marker detection | 5-10x vs CPU | Robot localization, calibration |
| **isaac_ros_image_proc** | Image rectification, resize | 8-15x vs CPU | Camera preprocessing |

---

## Installation with Docker

Isaac ROS packages require specific NVIDIA drivers and CUDA versions. The recommended installation method is **Docker** to ensure reproducibility across different systems.

### Prerequisites

**Hardware**:
- NVIDIA GPU (RTX 2060+ or Jetson Xavier/Orin)
- 16GB+ RAM (32GB recommended)
- 50GB+ free disk space

**Software**:
- Ubuntu 22.04 LTS (native or WSL2)
- NVIDIA Driver 525+ (`nvidia-smi` should show driver version)
- Docker 24.0+ (`docker --version`)

### Step-by-Step Docker Installation

**1. Install NVIDIA Container Toolkit**:

```bash
# Add NVIDIA container toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA container toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker to apply changes
sudo systemctl restart docker
```

**2. Verify NVIDIA Runtime**:

```bash
# Test NVIDIA runtime with Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output: nvidia-smi table showing GPU info
# If error: check NVIDIA driver installation
```

**3. Pull Isaac ROS Docker Image**:

```bash
# Pull Isaac ROS 2 Humble image (3-5 GB download)
docker pull nvcr.io/nvidia/isaac/ros:humble-isaac_ros_dev-aarch64

# For x86_64 (desktop), use:
docker pull dustynv/ros:humble-pytorch-l4t-r35.2.1

# Verify image exists
docker images | grep isaac
```

**4. Run Isaac ROS Container**:

```bash
# Create workspace directory on host
mkdir -p ~/isaac_ros_ws/src

# Run container with GPU access and workspace mount
docker run --rm -it \
  --gpus all \
  --network host \
  --privileged \
  -v ~/isaac_ros_ws:/workspaces/isaac_ros_ws \
  -v /dev:/dev \
  nvcr.io/nvidia/isaac/ros:humble-isaac_ros_dev-aarch64 \
  /bin/bash

# Inside container, verify ROS 2 installation
source /opt/ros/humble/setup.bash
ros2 --version
# Expected: ros2 cli version humble

# Verify Isaac ROS packages
ros2 pkg list | grep isaac_ros
# Expected: list of isaac_ros_* packages
```

### Installation Verification Checklist

- [ ] NVIDIA driver 525+ installed (`nvidia-smi` shows version)
- [ ] Docker can access GPU (`docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`)
- [ ] Isaac ROS container starts without errors
- [ ] ROS 2 Humble available in container (`ros2 --version`)
- [ ] Isaac ROS packages listed (`ros2 pkg list | grep isaac_ros`)

### Troubleshooting Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **NVIDIA runtime not found** | Error: "could not select device driver" | Install nvidia-container-toolkit: `sudo apt install nvidia-container-toolkit` |
| **CUDA not available** | Error: "CUDA driver version is insufficient" | Update NVIDIA driver: `sudo ubuntu-drivers autoinstall && sudo reboot` |
| **Network issues** | Container can't reach internet | Use `--network host` flag in docker run command |
| **Permission denied /dev** | Can't access camera/sensors | Add `--privileged -v /dev:/dev` flags |
| **Slow startup** | Container takes >2 minutes to start | Use SSD for Docker storage, increase RAM allocation |

:::warning WSL2 GPU Support
Isaac ROS on WSL2 requires WSL 2.0+ with GPU passthrough enabled. Performance may be 20-30% slower than native Ubuntu. For production, use native installation.
:::

---

## Native Installation (Advanced)

For students who prefer native installation or need maximum performance, follow this apt-based installation.

### Native Installation Steps

**1. Install ROS 2 Humble**:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt install -y curl
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble Desktop
sudo apt update
sudo apt install -y ros-humble-desktop
```

**2. Install Isaac ROS Dependencies**:

```bash
# Install CUDA 12.1 (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install TensorRT 8.6
sudo apt-get install -y tensorrt python3-libnvinfer-dev

# Install additional dependencies
sudo apt-get install -y \
  ros-humble-vision-msgs \
  ros-humble-image-transport \
  ros-humble-tf-transformations
```

**3. Clone and Build Isaac ROS Packages**:

```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws/src

# Clone Isaac ROS common (meta-package)
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git

# Install dependencies
cd ~/isaac_ros_ws
rosdep install --from-paths src --ignore-src -r -y

# Build packages
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace
source ~/isaac_ros_ws/install/setup.bash
```

:::info Native vs Docker Trade-offs
**Native Pros**: 10-20% better performance, easier debugging
**Native Cons**: Complex dependency management, version conflicts

**Docker Pros**: Reproducible, isolated environment, easier setup
**Docker Cons**: 10-20% slower, requires nvidia-container-toolkit

For learning, use Docker. For production deployment on Jetson, use native.
:::

---

## Visual SLAM with nvblox_ros

**nvblox** is NVIDIA's GPU-accelerated 3D reconstruction and SLAM library. It creates real-time occupancy maps for robot navigation using depth cameras or LiDAR.

### nvblox Architecture

```mermaid
graph LR
    A[Depth Camera] --> B[nvblox_ros Node]
    C[IMU] --> B
    B --> D[TSDF Volume]
    B --> E[ESDF Layer]
    D --> F[/map Topic]
    E --> G[/distance_map Topic]
    F --> H[Nav2 Planner]
    G --> H
    B --> I[/odom Topic]
    I --> H
```

**Key Concepts**:

- **TSDF (Truncated Signed Distance Field)**: 3D voxel grid storing distance to nearest surface
- **ESDF (Euclidean Signed Distance Field)**: Distance map for obstacle avoidance
- **Visual Odometry**: Estimates robot motion from camera images
- **Loop Closure**: Detects revisited locations to reduce drift

### nvblox_ros Launch File

Create a launch file to start nvblox SLAM with a depth camera:

```python
#!/usr/bin/env python3
"""
Launch nvblox_ros VSLAM with RealSense D435 depth camera.
Publishes /map (OccupancyGrid) and /odom (Odometry) for Nav2.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # nvblox_ros node (GPU-accelerated SLAM)
    nvblox_node = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'global_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'depth_topic': '/camera/depth/image_rect_raw',
            'depth_camera_info_topic': '/camera/depth/camera_info',
            'color_topic': '/camera/color/image_raw',
            'color_camera_info_topic': '/camera/color/camera_info',
            'imu_topic': '/imu/data',
            'voxel_size': 0.02,  # 2cm voxel resolution (accuracy)
            'esdf_slice_height': 1.0,  # ESDF slice at 1m height (humanoid torso)
            'esdf_slice_min_height': 0.0,
            'esdf_slice_max_height': 2.0,
            'update_esdf_every_n_sec': 0.1,  # Update distance map at 10 Hz
            'max_integration_time_s': 0.5,  # Maximum SLAM delay
            'mesh_update_rate_hz': 5.0,  # Mesh visualization rate
            'publish_esdf_distance_slice': True,
            'publish_occupancy_grid': True,  # For Nav2 compatibility
        }],
        remappings=[
            ('depth/image', '/camera/depth/image_rect_raw'),
            ('depth/camera_info', '/camera/depth/camera_info'),
            ('color/image', '/camera/color/image_raw'),
            ('color/camera_info', '/camera/color/camera_info'),
        ]
    )

    # RealSense camera driver (if using real hardware)
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        output='screen',
        parameters=[{
            'depth_module.profile': '640x480x30',  # 640x480 @ 30 Hz
            'rgb_camera.profile': '640x480x30',
            'enable_gyro': True,
            'enable_accel': True,
            'unite_imu_method': 2,  # Combine gyro + accel to /imu/data
            'align_depth.enable': True,  # Align depth to RGB
        }]
    )

    # TF broadcaster (publish camera to base_link transform)
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.1', '0', '0.15', '0', '0', '0', 'base_link', 'camera_link']
        # Camera positioned 10cm forward, 15cm up from base_link
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation time'),
        nvblox_node,
        realsense_node,  # Comment out if using Isaac Sim
        tf_node,
    ])
```

**Save as**: `~/isaac_ros_ws/src/nvblox_launch.py`

### nvblox Configuration Parameters

Create a YAML file for fine-tuning nvblox performance:

```yaml
# nvblox_params.yaml
# GPU-accelerated SLAM parameters for humanoid robot navigation

nvblox_node:
  ros__parameters:
    # Frame configuration
    global_frame: "map"
    odom_frame: "odom"
    base_frame: "base_link"

    # TSDF Volume (3D reconstruction)
    voxel_size: 0.02  # 2cm resolution (balance: accuracy vs memory)
    tsdf_integrator_max_integration_distance_m: 10.0  # Integrate depth up to 10m
    tsdf_integrator_truncation_distance_vox: 4.0  # TSDF truncation (4 voxels = 8cm)

    # ESDF Distance Map (for obstacle avoidance)
    esdf_integrator_min_weight: 0.0  # Minimum TSDF weight to convert to ESDF
    esdf_integrator_max_distance_m: 4.0  # Compute distance up to 4m from obstacles
    esdf_slice_height: 1.0  # Slice height for 2D occupancy grid (humanoid torso)
    esdf_slice_min_height: 0.0  # Minimum height to consider obstacles
    esdf_slice_max_height: 2.0  # Maximum height (humanoid head)

    # Update rates
    update_mesh_every_n_sec: 0.2  # Update 3D mesh at 5 Hz
    update_esdf_every_n_sec: 0.033  # Update distance map at 30 Hz
    publish_esdf_distance_slice: true
    publish_occupancy_grid: true  # Publish /map for Nav2

    # Integration settings
    max_back_projection_distance: 10.0  # Maximum depth to integrate
    max_integration_time_s: 0.5  # Maximum time to spend integrating per frame

    # Memory management
    layer_cake_size_m: 20.0  # Total map size (20m x 20m x 20m cube)

    # GPU settings
    cuda_streams: 4  # Number of CUDA streams (parallelism)
```

**Save as**: `~/isaac_ros_ws/src/nvblox_params.yaml`

**Load parameters in launch file**:

```python
# Add to nvblox_node parameters
parameters=[
    'nvblox_params.yaml',
    {'use_sim_time': use_sim_time}
]
```

### Understanding /map and /odom Topics

nvblox_ros publishes two critical topics for navigation:

**1. /map (nav_msgs/OccupancyGrid)**:
- 2D occupancy grid for Nav2 global planner
- Resolution: 2cm (matches `voxel_size`)
- Values: 0 (free), 100 (occupied), -1 (unknown)
- Frame: `map` (global, non-drifting)

**2. /odom (nav_msgs/Odometry)**:
- Robot pose estimate from visual odometry
- Frame: `odom` (local, drifts over time)
- Covariance: Uncertainty in x, y, θ

**Message Formats**:

```bash
# Inspect /map topic
ros2 topic echo /map --once

# Expected output:
# header:
#   frame_id: map
# info:
#   resolution: 0.02  # 2cm per cell
#   width: 1000
#   height: 1000
# data: [0, 0, 100, 0, ...]  # Occupancy values

# Inspect /odom topic
ros2 topic echo /odom --once

# Expected output:
# header:
#   frame_id: odom
# pose:
#   pose:
#     position: {x: 1.23, y: 0.45, z: 0.0}
#     orientation: {x: 0, y: 0, z: 0.1, w: 0.99}
#   covariance: [0.01, 0, 0, ..., 0.001]  # Position uncertainty
```

### RViz Visualization

Visualize the SLAM output in RViz:

```bash
# Source workspace
source ~/isaac_ros_ws/install/setup.bash

# Launch nvblox_ros
ros2 launch nvblox_launch.py

# In new terminal, launch RViz
rviz2
```

**RViz Configuration**:
1. Set **Fixed Frame** to `map`
2. Add **Map** display:
   - Topic: `/map`
   - Color Scheme: costmap
3. Add **Odometry** display:
   - Topic: `/odom`
   - Covariance: Position (show uncertainty ellipse)
4. Add **TF** display (show coordinate frames)
5. Add **PointCloud2** display (if using depth camera):
   - Topic: `/camera/depth/points`

**Expected Visualization**:
- Gray grid shows occupancy map
- Red arrows show robot pose trajectory
- Blue ellipse shows odometry uncertainty
- Point cloud shows depth camera data

:::tip SLAM Performance Benchmarking
Monitor SLAM frequency:
```bash
# Check nvblox update rate
ros2 topic hz /map
# Target: 5-10 Hz (mesh updates)

ros2 topic hz /odom
# Target: 30+ Hz (odometry updates)
```

On RTX 3060 (12GB VRAM), expect:
- TSDF integration: 30-60 Hz
- ESDF updates: 30 Hz
- Mesh publishing: 5-10 Hz
- Latency: less than 50ms end-to-end
:::

---

## GPU-Accelerated Semantic Segmentation

Isaac ROS provides TensorRT-accelerated DNN inference for semantic segmentation, object detection, and pose estimation. We'll use **PeopleSemSegNet** for real-time person segmentation.

### Isaac ROS DNN Image Encoder

The **isaac_ros_dnn_inference** package wraps TensorRT for ROS 2 integration. It automatically converts ONNX models to optimized TensorRT engines.

**Architecture**:

```mermaid
graph LR
    A[Camera Image] --> B[isaac_ros_dnn_image_encoder]
    B --> C[TensorRT Engine]
    C --> D[Segmentation Mask]
    D --> E[/segmentation Topic]
```

### PeopleSemSegNet Launch File

```python
#!/usr/bin/env python3
"""
Launch PeopleSemSegNet for real-time person segmentation.
Uses TensorRT on GPU for 30+ FPS inference.
"""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # DNN Image Encoder (TensorRT inference)
    dnn_encoder_container = ComposableNodeContainer(
        name='dnn_encoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded container
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_dnn_image_encoder',
                plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
                name='dnn_encoder',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'network_image_width': 960,  # Input width for PeopleSemSegNet
                    'network_image_height': 544,  # Input height
                    'image_mean': [0.485, 0.456, 0.406],  # ImageNet normalization
                    'image_stddev': [0.229, 0.224, 0.225],
                }],
                remappings=[
                    ('encoded_tensor', 'tensor_pub'),
                    ('image', '/camera/color/image_raw'),
                ]
            ),
            ComposableNode(
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensor_rt',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'model_file_path': '/workspaces/isaac_ros_ws/models/peoplesemsegnet.onnx',
                    'engine_file_path': '/tmp/peoplesemsegnet.engine',  # Cached TensorRT engine
                    'input_tensor_names': ['input_tensor'],
                    'input_binding_names': ['input_1'],
                    'output_tensor_names': ['output_tensor'],
                    'output_binding_names': ['argmax_1'],
                    'verbose': True,
                    'force_engine_update': False,  # Reuse cached engine
                }]
            ),
            ComposableNode(
                package='isaac_ros_unet',
                plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
                name='unet_decoder',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'network_output_type': 'argmax',  # Segmentation mask (class per pixel)
                    'color_segmentation_mask_encoding': 'rgb8',
                }],
                remappings=[
                    ('tensor_sub', 'tensor_pub'),
                    ('unet/colored_segmentation_mask', '/segmentation_mask'),
                ]
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        dnn_encoder_container,
    ])
```

**Save as**: `~/isaac_ros_ws/src/peoplesemsegnet_launch.py`

### DNN Encoder Configuration

Create YAML for TensorRT optimization:

```yaml
# dnn_encoder_params.yaml
# TensorRT optimization for PeopleSemSegNet (person segmentation)

dnn_encoder:
  ros__parameters:
    # Input preprocessing
    network_image_width: 960
    network_image_height: 544
    image_mean: [0.485, 0.456, 0.406]  # ImageNet mean (RGB)
    image_stddev: [0.229, 0.224, 0.225]  # ImageNet std
    keep_aspect_ratio: true  # Preserve aspect ratio during resize

tensor_rt:
  ros__parameters:
    # TensorRT engine configuration
    model_file_path: "/workspaces/isaac_ros_ws/models/peoplesemsegnet.onnx"
    engine_file_path: "/tmp/peoplesemsegnet.engine"
    force_engine_update: false  # Reuse cached engine (faster startup)

    # Precision mode (affects speed vs accuracy)
    # Options: fp32 (slowest, most accurate), fp16 (balanced), int8 (fastest, requires calibration)
    precision: "fp16"

    # Workspace size (MB) - increase for larger models
    workspace_size: 1024  # 1 GB

    # Dynamic batching (if processing multiple images)
    max_batch_size: 1

unet_decoder:
  ros__parameters:
    # Segmentation output configuration
    network_output_type: "argmax"  # Class ID per pixel
    color_segmentation_mask_encoding: "rgb8"  # RGB color mask
    mask_width: 960
    mask_height: 544
```

**Save as**: `~/isaac_ros_ws/src/dnn_encoder_params.yaml`

### Performance Benchmarking

Measure inference performance on your GPU:

```bash
# Launch PeopleSemSegNet
source ~/isaac_ros_ws/install/setup.bash
ros2 launch peoplesemsegnet_launch.py

# In new terminal, benchmark topic rate
ros2 topic hz /segmentation_mask

# Expected output (RTX 3060):
# average rate: 32.5
#   min: 0.028s max: 0.035s std dev: 0.002s window: 100
# Interpretation: 32.5 FPS segmentation (30+ Hz target met!)

# Measure latency
ros2 topic delay /segmentation_mask

# Expected output:
# average delay: 0.045
#   min: 0.030s max: 0.060s std dev: 0.008s window: 50
# Interpretation: 45ms average latency (acceptable for real-time)
```

**Performance by GPU**:

| GPU Model | Precision | FPS | Latency | VRAM Usage |
|-----------|-----------|-----|---------|------------|
| RTX 4090 | FP16 | 120+ | 8ms | 2GB |
| RTX 3060 | FP16 | 30-40 | 30ms | 3GB |
| RTX 2060 | FP16 | 20-25 | 50ms | 4GB |
| Jetson Orin | FP16 | 15-20 | 60ms | 2GB |
| Jetson Xavier | FP16 | 8-12 | 100ms | 2GB |

:::tip Optimizing Inference Speed
To increase FPS:
1. **Lower resolution**: 960x544 → 640x360 (2x speedup)
2. **Use INT8 precision**: Requires calibration dataset (3-5x speedup)
3. **Reduce batch size**: If using batch > 1
4. **Close background GPU apps**: Discord, Chrome with hardware acceleration
:::

---

## Isaac Sim-ROS 2 Bridge Integration

The **Isaac Sim ROS 2 bridge** allows Isaac Sim sensors to publish ROS 2 topics, enabling you to test Isaac ROS nodes with synthetic data before deploying to real hardware.

### Bridge Architecture

```mermaid
graph LR
    A[Isaac Sim Sensors] --> B[ROS 2 Bridge]
    B --> C[/camera/depth]
    B --> D[/camera/color]
    B --> E[/imu/data]
    C --> F[nvblox_ros]
    D --> F
    E --> F
    F --> G[/map]
    G --> H[RViz Visualization]
```

### Bridge Launch File

Create a launch file to start Isaac Sim with ROS 2 bridge:

```python
#!/usr/bin/env python3
"""
Launch Isaac Sim with ROS 2 bridge for nvblox_ros testing.
Publishes depth camera and IMU to ROS 2 topics.
"""

import omni
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np

# Import ROS 2 bridge
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")

import rclpy
from rclpy.node import Node

def create_isaac_sim_scene():
    """
    Create Isaac Sim scene with humanoid robot and camera.
    Configures ROS 2 publishers for depth, RGB, and IMU.
    """
    # Create world
    world = World()

    # Add ground plane
    world.scene.add_default_ground_plane()

    # Load humanoid robot
    robot_prim_path = "/World/humanoid"
    prim_utils.create_prim(
        robot_prim_path,
        usd_path="/home/user/isaac_sim_assets/robots/humanoid.usd",
        position=np.array([0, 0, 1.0])
    )

    # Create depth camera (for nvblox_ros)
    camera_path = f"{robot_prim_path}/head/depth_camera"
    camera = Camera(
        prim_path=camera_path,
        position=np.array([0.1, 0.0, 0.15]),
        orientation=euler_angles_to_quat(np.array([0, 0, 0])),
        frequency=30,
        resolution=(640, 480),
    )
    camera.add_distance_to_camera_to_frame()  # Enable depth

    # Enable ROS 2 publisher for camera
    enable_ros2_camera_publishers(camera_path)

    # Create IMU sensor
    imu_path = f"{robot_prim_path}/torso/imu"
    enable_ros2_imu_publisher(imu_path)

    print("Isaac Sim scene created with ROS 2 bridge")
    print("Publishing topics:")
    print("  - /camera/depth/image_rect_raw")
    print("  - /camera/depth/camera_info")
    print("  - /camera/color/image_raw")
    print("  - /camera/color/camera_info")
    print("  - /imu/data")

    return world


def enable_ros2_camera_publishers(camera_prim_path):
    """
    Enable ROS 2 publishers for camera RGB and depth.
    """
    import omni.graph.core as og

    # Create ROS 2 camera graph
    graph_path = "/ActionGraph/CameraPublisher"
    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2CameraHelper.inputs:execIn"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ROS2CameraHelper.inputs:cameraPrim", camera_prim_path),
                ("ROS2CameraHelper.inputs:topicName", "/camera/color/image_raw"),
                ("ROS2CameraHelper.inputs:depthTopicName", "/camera/depth/image_rect_raw"),
                ("ROS2CameraHelper.inputs:cameraInfoTopicName", "/camera/color/camera_info"),
                ("ROS2CameraHelper.inputs:depthCameraInfoTopicName", "/camera/depth/camera_info"),
                ("ROS2CameraHelper.inputs:frameId", "camera_link"),
            ],
        },
    )

    print(f"ROS 2 camera publishers enabled for {camera_prim_path}")


def enable_ros2_imu_publisher(imu_prim_path):
    """
    Enable ROS 2 publisher for IMU data.
    """
    import omni.graph.core as og

    # Create ROS 2 IMU graph
    graph_path = "/ActionGraph/IMUPublisher"
    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2PublishImu", "omni.isaac.ros2_bridge.ROS2PublishImu"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2PublishImu.inputs:execIn"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ROS2PublishImu.inputs:topicName", "/imu/data"),
                ("ROS2PublishImu.inputs:frameId", "imu_link"),
            ],
        },
    )

    print(f"ROS 2 IMU publisher enabled for {imu_prim_path}")


if __name__ == "__main__":
    # Create scene with ROS 2 bridge
    world = create_isaac_sim_scene()

    # Reset and run simulation
    world.reset()

    print("\n=== Isaac Sim ROS 2 Bridge Running ===")
    print("Verify topics with: ros2 topic list")
    print("Launch nvblox_ros with: ros2 launch nvblox_launch.py use_sim_time:=true")

    # Run simulation loop
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()
```

**Save as**: `~/isaac_ros_ws/src/isaac_sim_bridge.py`

**Run the bridge**:

```bash
# Terminal 1: Launch Isaac Sim with ROS 2 bridge
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh isaac_sim_bridge.py

# Terminal 2: Verify ROS 2 topics
source /opt/ros/humble/setup.bash
ros2 topic list | grep camera
# Expected:
#   /camera/color/image_raw
#   /camera/depth/image_rect_raw
#   /camera/color/camera_info
#   /camera/depth/camera_info

# Terminal 3: Launch nvblox_ros with simulated data
source ~/isaac_ros_ws/install/setup.bash
ros2 launch nvblox_launch.py use_sim_time:=true

# Terminal 4: Visualize in RViz
rviz2
```

### Verification: Depth Camera Publishing

Check that depth camera is publishing at correct rate:

```bash
# Check depth topic rate
ros2 topic hz /camera/depth/image_rect_raw
# Expected: average rate: 30.0 (Hz)

# Inspect depth image
ros2 topic echo /camera/depth/image_rect_raw --once
# Expected:
# header:
#   frame_id: camera_link
# height: 480
# width: 640
# encoding: 32FC1  # 32-bit float depth in meters
# data: [...]
```

:::info Bridge Latency
Isaac Sim → ROS 2 bridge adds 5-15ms latency. This is negligible for 30 Hz sensors but may affect 100+ Hz IMU data. For high-frequency sensors, use native Isaac Sim APIs instead of ROS 2 bridge.
:::

---

## Hands-On Exercise: VSLAM Pipeline

Apply everything you've learned by building a complete visual SLAM pipeline with Isaac Sim and Isaac ROS.

### Exercise Steps

1. **Launch Isaac Sim with ROS 2 Bridge**:
   - Run `isaac_sim_bridge.py` to start simulation
   - Verify depth camera publishes to `/camera/depth/image_rect_raw` at 30 Hz
   - Verify IMU publishes to `/imu/data`

2. **Launch nvblox_ros**:
   - Start nvblox_ros with `use_sim_time:=true`
   - Verify SLAM node subscribes to depth and IMU topics
   - Check SLAM rate with `ros2 topic hz /map` (target: 5+ Hz)

3. **Build 3D Map**:
   - Manually move humanoid robot in Isaac Sim (use transform tools)
   - Observe 3D occupancy map building in RViz
   - Add obstacles (cubes, cylinders) to scene and verify they appear in map

4. **Measure SLAM Accuracy**:
   - Move robot in 5m square loop
   - Check final odometry error: `ros2 topic echo /odom --once`
   - Target: less than 2cm error after 5m loop (low drift)

### Verification Checklist

- [ ] Isaac Sim publishes depth at 30 Hz
- [ ] nvblox_ros publishes /map at 5+ Hz
- [ ] nvblox_ros publishes /odom at 30+ Hz
- [ ] RViz shows 3D occupancy map with obstacles
- [ ] SLAM odometry drift < 2cm after 5m loop
- [ ] No CUDA errors in nvblox terminal output

### Expected Performance

**SLAM Metrics** (RTX 3060):
- TSDF integration: 30-60 Hz
- Occupancy grid updates: 5-10 Hz
- Odometry: 30+ Hz
- Latency: 30-50ms end-to-end

**Accuracy**:
- Voxel resolution: 2cm (voxel_size parameter)
- Odometry drift: 0.5-2% of distance traveled
- Loop closure error: less than 5cm for 10m loop

---

## Performance Optimization

Maximize Isaac ROS performance with these GPU optimizations:

### GPU Utilization Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 0.5 nvidia-smi

# Expected output during SLAM:
# GPU Utilization: 60-80%
# Memory Used: 4-8 GB (out of 12 GB on RTX 3060)
# Power Draw: 80-120 W (out of 170 W max)

# If GPU utilization less than 40%, bottleneck is CPU (increase cuda_streams)
# If GPU memory greater than 90%, reduce voxel map size (layer_cake_size_m)
```

### Latency Reduction Techniques

| Optimization | Latency Improvement | Trade-off |
|--------------|-------------------|-----------|
| **Increase cuda_streams** (4 → 8) | 10-20% faster | Higher GPU memory usage |
| **Reduce voxel_size** (2cm → 5cm) | 2-3x faster | Lower map resolution |
| **Disable mesh publishing** | 15-25% faster | No RViz mesh visualization |
| **Use FP16 precision** (TensorRT) | 2x faster | Minimal accuracy loss |
| **Reduce camera resolution** (640x480 → 320x240) | 2-4x faster | Lower SLAM accuracy |

### Memory Optimization

```yaml
# Low-memory configuration for 6GB VRAM GPUs (RTX 2060)
nvblox_node:
  ros__parameters:
    voxel_size: 0.05  # 5cm (larger voxels = less memory)
    layer_cake_size_m: 10.0  # 10m x 10m map (smaller map)
    esdf_integrator_max_distance_m: 2.0  # Shorter distance field
    cuda_streams: 2  # Fewer streams = less memory
```

---

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **CUDA OOM** | Error: "CUDA out of memory" | Reduce voxel map size: `layer_cake_size_m: 10.0`, increase `voxel_size: 0.05` |
| **Topic not publishing** | `ros2 topic list` shows topic but no data | Check camera is enabled in Isaac Sim, verify TF frames exist |
| **Docker networking** | Container can't see host ROS topics | Use `--network host` in docker run command |
| **Low FPS (less than 10 Hz)** | SLAM slower than expected | Close background GPU apps, enable FP16 precision, reduce resolution |
| **TensorRT build fails** | Error during .engine generation | Check CUDA version matches TensorRT (12.1+), verify disk space (5GB+ free) |
| **RViz shows no map** | Empty grid in RViz | Set Fixed Frame to "map", verify /map topic exists, check nvblox is publishing |

---

## External Resources

Continue learning with these official NVIDIA resources:

- **[Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)**: Complete API reference and tutorials
- **[nvblox_ros GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox)**: Source code, examples, parameter tuning
- **[Isaac ROS GEM Architecture](https://nvidia-isaac-ros.github.io/concepts/graph_execution_manager.html)**: Understanding GPU execution optimization
- **[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)**: Advanced inference optimization

:::tip Performance Comparison: Isaac ROS vs CPU SLAM

| SLAM Method | Hardware | Update Rate | Accuracy | Latency |
|-------------|----------|-------------|----------|---------|
| **nvblox_ros (Isaac ROS)** | RTX 3060 | 30 Hz | 2cm | 30ms |
| ORB-SLAM2 (CPU) | i7-10700K | 10 Hz | 5cm | 100ms |
| RTAB-Map (CPU) | i7-10700K | 5 Hz | 3cm | 200ms |
| Cartographer (CPU) | i7-10700K | 8 Hz | 4cm | 150ms |

Isaac ROS provides 3-6x faster updates with 2-3x better accuracy.
:::

---

## Summary

In this chapter, you learned:

- ✅ **Isaac ROS advantages**: GPU acceleration, 10-50x faster than CPU perception
- ✅ **Installation**: Docker with NVIDIA Container Toolkit, verify GPU access
- ✅ **nvblox_ros VSLAM**: 30 Hz SLAM, 2cm accuracy, /map and /odom topics
- ✅ **TensorRT DNN inference**: 30+ FPS semantic segmentation on RTX 3060
- ✅ **Isaac Sim bridge**: Test Isaac ROS with synthetic sensors before hardware deployment
- ✅ **Performance optimization**: GPU utilization, latency reduction, memory management

**Key Takeaways**:
1. Isaac ROS enables real-time perception (30+ Hz) on NVIDIA GPUs
2. nvblox_ros provides production-ready VSLAM with 2cm accuracy
3. TensorRT optimizes DNNs for 20-50x faster inference vs CPU
4. Isaac Sim bridge allows safe testing before physical robot deployment

Ready for autonomous navigation? Continue to Chapter 3!

:::success Next Chapter: Navigation with Nav2
You now have real-time perception and localization working. In Chapter 3, we'll integrate Isaac ROS with Nav2 to enable autonomous navigation for humanoid robots.

**Next**: [Chapter 3: Navigation with Nav2 →](./chapter-3-nav2.md)
:::
