---
sidebar_position: 1
title: Module 1 Overview
description: Learn ROS 2 fundamentals, Python AI agent integration, and humanoid robot modeling with URDF
keywords: [ROS 2, robotics, middleware, Python, rclpy, URDF, humanoid robots]
---

# Module 1: The Robotic Nervous System (ROS 2)

Welcome to Module 1 of the Physical AI Book! This module teaches you how ROS 2 (Robot Operating System 2) serves as the "nervous system" for robotic systems, enabling communication between sensors, decision-making AI agents, and physical actuators.

## What You'll Learn

By the end of this module, you will:

1. **Understand ROS 2 nodes, topics, and services** - Learn how independent processes communicate in distributed robot systems
2. **Integrate Python AI agents with ROS 2** - Bridge intelligent decision-making code with robot hardware using rclpy
3. **Create URDF models for humanoid robots** - Describe robot physical structure for simulation and real-world deployment

## Learning Objectives

### Module-Level Goals

- Explain ROS 2's role as middleware for robot control systems
- Create publisher and subscriber nodes for asynchronous communication
- Differentiate between topics, services, and actions
- Implement Python AI agents that subscribe to sensor data and publish control commands
- Model humanoid robot kinematic chains using URDF (Unified Robot Description Format)
- Visualize robot models in RViz simulation environment

## Prerequisites

Before starting this module, you should have:

- **Basic Python programming knowledge**: Variables, functions, classes, and imports
- **Command-line familiarity**: Running terminal commands, installing packages
- **Development environment**: Ubuntu 22.04 (or Docker) with ROS 2 Humble installed

:::tip Environment Setup
If you haven't set up your ROS 2 environment yet, follow the [official ROS 2 Humble installation guide](https://docs.ros.org/en/humble/Installation.html) for detailed installation instructions.
:::

## Module Structure

This module consists of three chapters, each building on the previous one:

### [Chapter 1: ROS 2 Fundamentals](./chapter-1-fundamentals.md)

**Estimated Time**: 60 minutes

Learn the core concepts of ROS 2:
- What is ROS 2 and why it matters
- Nodes and communication patterns
- Topics for pub/sub messaging
- Services for request/response
- Lifecycle management for robust initialization

**Hands-On**: Create your first publisher and subscriber nodes

### [Chapter 2: Python Agents & ROS 2 Integration](./chapter-2-python-integration.md)

**Estimated Time**: 50 minutes

Bridge AI and robotics:
- Introduction to rclpy (ROS 2 Python library)
- AI agent architecture with ROS 2
- Subscribing to sensor topics
- Publishing control commands
- Complete sensor-to-actuator workflow

**Hands-On**: Build a Python AI agent that integrates with ROS 2

### [Chapter 3: Humanoid Robot Description with URDF](./chapter-3-urdf-modeling.md)

**Estimated Time**: 55 minutes

Model robots for simulation:
- What is URDF?
- Links and joints
- Humanoid robot kinematic chains
- Adding sensors to robot models
- Visualizing in RViz

**Hands-On**: Create and visualize a humanoid URDF model

## Estimated Duration

**Total module time**: 3-4 hours (including hands-on exercises)

## How to Use This Module

1. **Read sequentially**: Chapters build on each other, so start with Chapter 1
2. **Code along**: Don't just read - type and run the examples yourself
3. **Use the companion repository**: All code examples are tested and runnable (link coming soon)
4. **Check expected outputs**: Each example shows what you should see in your terminal
5. **Experiment**: After completing each chapter, modify the code and observe the results

## Success Criteria

You'll know you've mastered this module when you can:

- ✅ Create and run a ROS 2 publisher-subscriber pair
- ✅ Explain when to use topics vs. services vs. actions
- ✅ Integrate a Python AI agent with ROS 2 in under 30 minutes
- ✅ Load and visualize a humanoid URDF model in RViz
- ✅ Adapt code examples to your own robotic projects

## Next Steps

Ready to begin? Start with [Chapter 1: ROS 2 Fundamentals →](./chapter-1-fundamentals.md)

:::info Questions or Issues?
- Official ROS 2 documentation: [docs.ros.org/en/humble](https://docs.ros.org/en/humble/)
- Community support: [ROS Discourse](https://discourse.ros.org/)
:::
