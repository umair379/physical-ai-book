---
title: "Module 1: ROS 2 Fundamentals Released"
date: 2025-12-27
authors: [default]
tags: [ROS2, Announcement]
description: "Announcing the release of Module 1 covering ROS 2 core concepts, nodes, topics, services, and action servers."
---

We're thrilled to announce that **Module 1: ROS 2 Fundamentals** is now live!

This comprehensive module introduces you to the Robot Operating System 2 (ROS 2), the industry-standard middleware for building modern robot applications.

## What You'll Learn

Module 1 covers the essential building blocks of ROS 2:

### Core Concepts
- **ROS 2 Architecture**: Understanding the distributed communication model
- **Nodes**: Creating and managing independent processes
- **Topics**: Implementing publish-subscribe communication patterns
- **Services**: Request-response interactions between nodes
- **Actions**: Long-running tasks with feedback and cancellation

### Practical Skills
- Setting up a ROS 2 development workspace
- Creating custom messages and service definitions
- Building a multi-node robot application
- Debugging ROS 2 applications with CLI tools
- Using RViz2 for visualization

## Prerequisites

Before starting Module 1, you should have:
- Basic Python programming knowledge
- Familiarity with Linux command line
- Ubuntu 22.04 or ROS 2 Humble installed (we provide installation guides)

## Module Structure

The module is organized into 5 chapters:

1. **Introduction to ROS 2** - History, architecture, and key concepts
2. **Nodes and Topics** - Creating publishers and subscribers
3. **Services and Parameters** - Synchronous communication patterns
4. **Actions** - Asynchronous long-running tasks
5. **Practical Project** - Build a simple mobile robot controller

Each chapter includes:
- Conceptual explanations with diagrams
- Hands-on coding exercises
- Quiz questions to test understanding
- A practical mini-project

## Code Examples

Here's a quick preview of creating a ROS 2 publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello, ROS 2! Count: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = SimplePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Estimated Time

Plan to spend **2-3 weeks** working through Module 1 at a comfortable pace:
- Reading and understanding concepts: 5-7 hours
- Hands-on exercises: 10-15 hours
- Mini-projects: 5-8 hours
- Total: ~20-30 hours

## What's Next?

After completing Module 1, you'll be ready to:
- Move on to **Module 2: Gazebo Simulation** to test your ROS 2 skills in simulated environments
- Build more complex multi-node applications
- Explore advanced ROS 2 features like lifecycle nodes and quality of service (QoS)

## Get Started Now

Ready to dive in? Visit the module page to begin your ROS 2 journey!

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="ROS2"
  moduleNumber={1}
  moduleTitle="ROS 2 Fundamentals"
  moduleUrl="/docs/intro"
/>

Happy coding! 🤖
