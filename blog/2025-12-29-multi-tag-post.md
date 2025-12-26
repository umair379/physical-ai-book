---
title: "Integrating ROS 2 with Gazebo: A Complete Workflow"
date: 2025-12-29
authors: [default]
tags: [ROS2, Gazebo, Tutorial]
description: "Learn how to seamlessly integrate ROS 2 nodes with Gazebo simulation for testing robot behaviors before deploying to hardware."
---

Combining **ROS 2** and **Gazebo** creates a powerful development workflow that lets you test robot algorithms in simulation before deploying to real hardware.

## Why Integrate ROS 2 with Gazebo?

The ROS 2 + Gazebo combination offers several advantages:

- **Safe Testing**: Test dangerous maneuvers without risking hardware damage
- **Rapid Iteration**: Modify code and retest in seconds
- **Reproducible Experiments**: Same environment every time
- **Cost Effective**: No need for multiple physical robots

## Prerequisites

Ensure you have both ROS 2 Humble and Gazebo Fortress installed:

```bash
# Check ROS 2 installation
ros2 --version

# Check Gazebo installation
gz sim --version

# Install ROS 2 Gazebo bridge
sudo apt install ros-humble-ros-gz
```

## Workflow Overview

Here's the typical development cycle:

1. **Design** your robot in URDF/SDF
2. **Spawn** the robot in Gazebo
3. **Connect** ROS 2 nodes to simulated sensors/actuators
4. **Test** your algorithms in simulation
5. **Deploy** to real hardware

## Step 1: Create a ROS 2 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create a package
ros2 pkg create --build-type ament_python my_robot_control
```

## Step 2: Launch Gazebo with ROS 2 Bridge

Create a launch file `gazebo_ros2.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_gz_sim'),
                            'launch', 'gz_sim.launch.py')
            ]),
            launch_arguments={'gz_args': 'empty.sdf'}.items()
        ),

        # Spawn robot
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'my_robot',
                '-topic', '/robot_description',
                '-x', '0', '-y', '0', '-z', '0.5'
            ],
            output='screen'
        ),

        # ROS-Gazebo bridge for cmd_vel
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            output='screen'
        ),
    ])
```

## Step 3: Create a Simple Controller

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        msg = Twist()
        # Simple forward motion
        msg.linear.x = 0.5
        msg.angular.z = 0.0
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Bridge Sensor Data

For camera/LiDAR data, add more bridge topics:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /camera/image@sensor_msgs/msg/Image@gz.msgs.Image \
  /lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan
```

## Step 5: Visualize in RViz2

```bash
ros2 run rviz2 rviz2
```

Add displays for:
- `/camera/image` (Image)
- `/lidar/scan` (LaserScan)
- `/tf` (TF)

## Common Patterns

### Pattern 1: Teleoperation

```python
# Subscribe to keyboard input, publish to /cmd_vel
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Pattern 2: Navigation Testing

```bash
# Launch Nav2 stack with Gazebo
ros2 launch nav2_bringup tb3_simulation_launch.py
```

### Pattern 3: Perception Pipeline

```python
# Camera → Object Detection → Control
class PerceptionController(Node):
    def __init__(self):
        super().__init__('perception_controller')
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        # Process image, detect objects
        # Generate control commands
        pass
```

## Debugging Tips

### Check Topic Connections

```bash
# List all topics
ros2 topic list

# Check message rate
ros2 topic hz /cmd_vel

# Echo messages
ros2 topic echo /camera/image
```

### Monitor Transform Tree

```bash
# View TF tree
ros2 run tf2_tools view_frames
```

### Gazebo-ROS Bridge Status

```bash
# Check bridge status
ros2 topic info /cmd_vel
gz topic -l
```

## Performance Optimization

1. **Real-time factor**: Adjust Gazebo physics update rate
2. **Sensor frequency**: Reduce camera/LiDAR rates for faster simulation
3. **QoS settings**: Match ROS 2 QoS to your use case

## Next Steps

Now that you understand ROS 2 + Gazebo integration:

- Implement autonomous navigation with Nav2
- Test perception algorithms with synthetic data
- Develop multi-robot coordination strategies
- Simulate sensor failures and edge cases

## Dive Deeper

Ready to master both ROS 2 and Gazebo? Check out our comprehensive modules!

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="ROS2"
  moduleNumber={1}
  moduleTitle="ROS 2 Fundamentals"
  moduleUrl="/docs/intro"
/>

<ModuleCTA
  moduleName="Gazebo"
  moduleNumber={2}
  moduleTitle="Gazebo Simulation"
  moduleUrl="/docs/intro"
/>

Happy coding! 🤖🏗️
