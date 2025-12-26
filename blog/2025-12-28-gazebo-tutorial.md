---
title: "Getting Started with Gazebo: Your First Simulation"
date: 2025-12-28
authors: [default]
tags: [Gazebo, Tutorial]
description: "Learn how to create your first Gazebo simulation world and spawn a mobile robot in this beginner-friendly tutorial."
---

Ready to dive into robot simulation? Let's build your first Gazebo world and get a robot moving in a virtual environment!

## Why Gazebo?

**Gazebo** is the industry-standard simulator for robot development. It provides:

- **Physics Simulation**: Realistic robot behavior with friction, inertia, and collision detection
- **Sensor Models**: Cameras, LiDAR, IMU, and more for perception testing
- **ROS Integration**: Seamless connection with ROS 2 nodes
- **Plugin System**: Extend functionality with custom plugins

## What You'll Build

In this tutorial, you'll create:

1. A custom Gazebo world with terrain and obstacles
2. A mobile robot model (URDF)
3. Basic navigation controls

## Prerequisites

Before starting, ensure you have:

```bash
# Install Gazebo (Fortress for ROS 2 Humble)
sudo apt install ros-humble-gazebo-ros-pkgs

# Verify installation
gz sim --version
```

## Step 1: Create a World File

Create a new SDF world file `my_world.sdf`:

```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="my_first_world">
    <!-- Sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="box_link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
```

## Step 2: Launch the World

```bash
gz sim my_world.sdf
```

You should see a 3D environment with:
- A sun (lighting source)
- A ground plane
- A box obstacle

## Step 3: Add a Mobile Robot

For this tutorial, we'll use a pre-built differential drive robot. In a production setup, you'd create a custom URDF, but let's start simple:

```bash
# Spawn a TurtleBot3 model
ros2 run gazebo_ros spawn_entity.py \
  -entity my_robot \
  -database turtlebot3_waffle \
  -x 0 -y 0 -z 0.1
```

## Step 4: Control the Robot

Open a new terminal and send velocity commands:

```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Rotate
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
```

## Common Issues & Solutions

### Issue: "Model not found"

**Solution**: Ensure Gazebo model path includes ROS 2 models:

```bash
export GZ_SIM_RESOURCE_PATH=/usr/share/gazebo/models:$GZ_SIM_RESOURCE_PATH
```

### Issue: Robot falls through ground

**Solution**: Add a small Z offset when spawning (e.g., `z: 0.1`)

### Issue: Physics unstable

**Solution**: Check your URDF inertia tensors and collision geometries

## Next Steps

Now that you've created your first Gazebo simulation, you're ready to:

- Build custom robot models with URDF/SDF
- Add sensors (cameras, LiDAR) to your robot
- Implement autonomous navigation with Nav2
- Test perception algorithms in simulated environments

## Continue Learning

Ready to dive deeper into Gazebo? Explore Module 2 for comprehensive simulation training!

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="Gazebo"
  moduleNumber={2}
  moduleTitle="Gazebo Simulation"
  moduleUrl="/docs/intro"
/>

Happy simulating! 🏗️
