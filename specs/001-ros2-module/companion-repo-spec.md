# Companion Repository Specification: Module 1 ROS 2 Examples

**Date**: 2025-12-23
**Purpose**: Define structure and requirements for the companion code repository

## Repository Overview

**Repository Name**: `physical-ai-book-examples`
**Location**: Separate GitHub repository (not in main book repo)
**License**: MIT License
**Purpose**: Runnable ROS 2 code examples for Module 1 chapters

## Directory Structure

```
physical-ai-book-examples/
├── README.md                          # Setup and usage instructions
├── .gitignore                         # Ignore build artifacts
├── LICENSE                            # MIT License
├── module-1-ros2/                     # Module 1 examples
│   ├── README.md                      # Module-specific setup
│   ├── chapter-1-fundamentals/
│   │   ├── README.md                  # Chapter 1 examples overview
│   │   ├── simple_publisher.py        # Basic publisher example
│   │   ├── simple_subscriber.py       # Basic subscriber example
│   │   ├── lifecycle_node.py          # Lifecycle management example
│   │   ├── publisher_node_complete.py # Full publisher with package
│   │   ├── subscriber_node_complete.py# Full subscriber with package
│   │   ├── package.xml                # ROS 2 package manifest
│   │   ├── setup.py                   # Python package setup
│   │   └── expected_output.txt        # Expected console output
│   ├── chapter-2-python-integration/
│   │   ├── README.md                  # Chapter 2 examples overview
│   │   ├── rclpy_basic_node.py        # Basic rclpy node
│   │   ├── sensor_subscriber.py       # Sensor data subscription
│   │   ├── controller_publisher.py    # Control command publishing
│   │   ├── ai_agent_node_complete.py  # Complete AI agent workflow
│   │   ├── package.xml                # ROS 2 package manifest
│   │   ├── setup.py                   # Python package setup
│   │   └── expected_output.txt        # Expected console output
│   └── chapter-3-urdf-modeling/
│       ├── README.md                  # Chapter 3 examples overview
│       ├── simple_link_definition.urdf # Basic link example (snippet)
│       ├── revolute_joint_definition.urdf # Basic joint example (snippet)
│       ├── simple_humanoid.urdf       # Complete humanoid URDF
│       ├── humanoid_with_sensors.urdf # URDF with sensor plugins
│       ├── launch/
│       │   └── visualize_urdf.launch.py # RViz visualization launch
│       └── expected_output.txt        # What to see in RViz
├── .github/
│   └── workflows/
│       └── test-examples.yml          # CI/CD to validate examples
└── Dockerfile                         # Optional: Docker environment
```

## Code Example Requirements

### All Examples Must Include

1. **Complete code**: No pseudocode, no `# ... rest of code` placeholders
2. **Dependencies documented**: Exact ROS 2 packages and Python modules
3. **Expected output**: What readers should see when running
4. **Run instructions**: Step-by-step execution commands
5. **Comments**: Explain WHY, not WHAT (assume reader can read Python)

### Python Code Style

- **PEP-8 compliant**: Use `black` formatter
- **ROS 2 naming conventions**: Follow official style guide
- **Type hints**: Include where helpful for clarity
- **Docstrings**: For functions and classes

### Example Template

Each code file should follow this structure:

```python
"""
Simple Publisher Node - Chapter 1 Example

This example demonstrates creating a basic ROS 2 publisher node that
sends string messages to a topic at 1 Hz.

Expected output:
[INFO] [timestamp] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [timestamp] [minimal_publisher]: Publishing: "Hello World: 1"
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """Publishes string messages to 'topic' at 1 Hz."""

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        """Called every 1 second to publish a message."""
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## ROS 2 Package Structure

Each chapter directory should be a valid ROS 2 Python package with:

### package.xml

```xml
<?xml version="1.0"?>
<package format="3">
  <name>chapter_1_fundamentals</name>
  <version>1.0.0</version>
  <description>ROS 2 Fundamentals Examples from Physical AI Book</description>
  <maintainer email="book@example.com">Physical AI Book</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py

```python
from setuptools import setup

package_name = 'chapter_1_fundamentals'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Physical AI Book',
    maintainer_email='book@example.com',
    description='ROS 2 Fundamentals Examples',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = chapter_1_fundamentals.simple_publisher:main',
            'simple_subscriber = chapter_1_fundamentals.simple_subscriber:main',
            'lifecycle_node = chapter_1_fundamentals.lifecycle_node:main',
        ],
    },
)
```

## Build and Run Instructions

### Prerequisites

- ROS 2 Humble LTS installed
- Python 3.10+
- colcon build tools

### Build Process

```bash
# Clone repository
git clone https://github.com/[user]/physical-ai-book-examples.git
cd physical-ai-book-examples/module-1-ros2

# Build all chapter examples
cd chapter-1-fundamentals
colcon build
source install/setup.bash

# Run an example
ros2 run chapter_1_fundamentals simple_publisher
```

## CI/CD Validation

### GitHub Actions Workflow

The repository MUST include CI/CD to validate all examples run correctly:

```yaml
name: Test ROS 2 Examples

on: [push, pull_request]

jobs:
  test-module-1:
    runs-on: ubuntu-22.04
    container:
      image: osrf/ros:humble-desktop
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y python3-colcon-common-extensions

      - name: Build Chapter 1
        run: |
          cd module-1-ros2/chapter-1-fundamentals
          colcon build
          source install/setup.bash
          colcon test
          colcon test-result --verbose

      - name: Build Chapter 2
        run: |
          cd module-1-ros2/chapter-2-python-integration
          colcon build
          source install/setup.bash
          colcon test
          colcon test-result --verbose

      - name: Build Chapter 3
        run: |
          cd module-1-ros2/chapter-3-urdf-modeling
          colcon build
          source install/setup.bash
```

## Expected Output Documentation

Each chapter directory includes `expected_output.txt` with:

```text
Chapter 1: ROS 2 Fundamentals - Expected Outputs
=================================================

Simple Publisher (simple_publisher.py)
--------------------------------------
Terminal Output:
[INFO] [1703341200.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1703341201.123456789] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [1703341202.123456789] [minimal_publisher]: Publishing: "Hello World: 2"

Behavior:
- New message published every 1 second
- Counter increments with each message
- Press Ctrl+C to stop


Simple Subscriber (simple_subscriber.py)
-----------------------------------------
Terminal Output:
[INFO] [1703341200.234567890] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1703341201.234567890] [minimal_subscriber]: I heard: "Hello World: 1"
[INFO] [1703341202.234567890] [minimal_subscriber]: I heard: "Hello World: 2"

Behavior:
- Receives and logs messages from publisher
- Must run publisher in separate terminal
- Press Ctrl+C to stop
```

## Integration with Book

### Linking from Docusaurus

Book chapters will link to specific files:

```markdown
View the complete code: [simple_publisher.py](https://github.com/[user]/physical-ai-book-examples/blob/main/module-1-ros2/chapter-1-fundamentals/simple_publisher.py)
```

### Embedding Code Snippets

For inline examples, use syntax-highlighted fenced blocks:

````markdown
```python title="simple_publisher.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # ...
```
````

## Version Control

- **Main branch**: Stable, tested examples
- **Development branch**: Work-in-progress
- **Tag releases**: Match book module versions (e.g., `module-1-v1.0`)

## Maintenance

- **Weekly CI runs**: Validate examples still work with latest ROS 2 Humble updates
- **Issue tracking**: Enable GitHub Issues for reader questions
- **Pull requests**: Accept community contributions with review

## Success Criteria

Companion repository is successful when:

- ✅ All examples build without errors on clean Ubuntu 22.04 + ROS 2 Humble
- ✅ CI/CD passes for every commit
- ✅ Each example produces expected output documented in expected_output.txt
- ✅ Readers can clone, build, and run examples in under 5 minutes
- ✅ No external dependencies beyond standard ROS 2 Humble installation

---

**Next Steps**: Create the repository, populate with Chapter 1 examples, and link from book content.
