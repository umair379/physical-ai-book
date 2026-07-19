# Quickstart Guide: Module 1 - ROS 2 Development Environment

**Date**: 2025-12-23
**Phase**: 1 - Design & Contracts
**Purpose**: Document exact environment setup for Module 1 code examples

---

## Prerequisites

Before starting Module 1, ensure you have:

- **Operating System**: Ubuntu 22.04 LTS (recommended) OR Docker Desktop 4.x+
- **Hardware**: 4GB RAM minimum (8GB recommended), 20GB disk space
- **Network**: Internet connection for package downloads
- **Skills**: Basic command-line familiarity, text editor usage

---

## Option 1: Native Ubuntu Installation (Recommended)

### Step 1: Install ROS 2 Humble

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble Desktop (includes RViz, demos)
sudo apt update
sudo apt install ros-humble-desktop python3-argcomplete -y

# Install development tools
sudo apt install ros-dev-tools -y
```

**Verify Installation**:
```bash
source /opt/ros/humble/setup.bash
ros2 --version
# Expected output: ros2 doctor 0.10.x
```

### Step 2: Install Python 3.10 and Dependencies

Ubuntu 22.04 ships with Python 3.10 by default. Verify:

```bash
python3 --version
# Expected output: Python 3.10.x

# Install pip and venv
sudo apt install python3-pip python3-venv -y

# Install colcon (ROS 2 build tool)
sudo apt install python3-colcon-common-extensions -y
```

### Step 3: Set Up ROS 2 Environment

Add to `~/.bashrc` for automatic sourcing:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**Verify Environment**:
```bash
printenv | grep ROS
# Expected: ROS_VERSION=2, ROS_DISTRO=humble, ROS_PYTHON_VERSION=3
```

### Step 4: Install RViz and Visualization Tools

```bash
sudo apt install ros-humble-rviz2 ros-humble-joint-state-publisher-gui -y
```

**Test RViz**:
```bash
ros2 run rviz2 rviz2
# RViz window should open
```

---

## Option 2: Docker Installation (Cross-Platform)

### Step 1: Install Docker Desktop

- **Windows/macOS**: Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Install Docker Engine via [docs.docker.com/engine/install](https://docs.docker.com/engine/install/)

Verify:
```bash
docker --version
# Expected: Docker version 24.x or later
```

### Step 2: Pull ROS 2 Humble Docker Image

```bash
docker pull osrf/ros:humble-desktop
```

### Step 3: Run ROS 2 Container

```bash
# Linux/macOS
docker run -it --rm \
  --name ros2-humble \
  -v ~/physical-ai-book-examples:/workspace \
  osrf/ros:humble-desktop

# Windows (PowerShell)
docker run -it --rm `
  --name ros2-humble `
  -v ${PWD}/physical-ai-book-examples:/workspace `
  osrf/ros:humble-desktop
```

**Inside Container**:
```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

**Note**: RViz requires X11 forwarding. For GUI:

**Linux**:
```bash
docker run -it --rm \
  --name ros2-humble \
  -v ~/physical-ai-book-examples:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  osrf/ros:humble-desktop
```

**Windows**: Use VcXsrv or Xming for X11 server
**macOS**: Use XQuartz

---

## Clone Companion Repository

All code examples for Module 1 are in the companion repository:

```bash
# Clone the repository
git clone https://github.com/[user]/physical-ai-book-examples.git
cd physical-ai-book-examples/module-1-ros2
```

**Repository Structure**:
```
module-1-ros2/
├── chapter-1-fundamentals/
│   ├── publisher_node.py
│   ├── subscriber_node.py
│   └── package.xml
├── chapter-2-python-integration/
│   ├── ai_agent_node.py
│   └── package.xml
└── chapter-3-urdf-modeling/
    ├── simple_humanoid.urdf
    └── launch/
```

---

## Build and Run First Example

### Test ROS 2 Publisher-Subscriber (Chapter 1)

```bash
cd module-1-ros2/chapter-1-fundamentals

# Build the ROS 2 package
colcon build

# Source the workspace
source install/setup.bash

# Terminal 1: Run publisher
ros2 run my_package publisher_node
```

**Expected Output (Terminal 1)**:
```
[INFO] [1703341200.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1703341201.123456789] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [1703341202.123456789] [minimal_publisher]: Publishing: "Hello World: 2"
```

Open a new terminal:

```bash
# Terminal 2: Source environment
cd module-1-ros2/chapter-1-fundamentals
source install/setup.bash

# Run subscriber
ros2 run my_package subscriber_node
```

**Expected Output (Terminal 2)**:
```
[INFO] [1703341200.234567890] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1703341201.234567890] [minimal_subscriber]: I heard: "Hello World: 1"
[INFO] [1703341202.234567890] [minimal_subscriber]: I heard: "Hello World: 2"
```

**Success**: If you see both nodes communicating, your environment is ready!

---

## Troubleshooting

### Issue: `ros2: command not found`

**Solution**: Source the ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

Add to `~/.bashrc` to make permanent:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

---

### Issue: `ModuleNotFoundError: No module named 'rclpy'`

**Solution**: Ensure ROS 2 Python packages are installed:
```bash
sudo apt install python3-rclpy -y
source /opt/ros/humble/setup.bash
```

---

### Issue: Nodes can't discover each other (no communication)

**Possible Causes**:
1. **DDS configuration**: ROS 2 uses DDS for discovery. Try:
   ```bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   ```

2. **Firewall blocking multicast**: Allow UDP multicast on port 7400:
   ```bash
   sudo ufw allow 7400/udp
   ```

3. **Different ROS_DOMAIN_ID**: Ensure both nodes use same domain:
   ```bash
   export ROS_DOMAIN_ID=0
   ```

---

### Issue: RViz crashes or doesn't display 3D view

**Solution**: Check graphics drivers:
```bash
glxinfo | grep "OpenGL version"
# Should show OpenGL 3.3 or higher
```

For Docker, ensure X11 forwarding is configured (see Docker setup above).

---

### Issue: `colcon build` fails with permission errors

**Solution**: Ensure you own the workspace directory:
```bash
sudo chown -R $USER:$USER ~/physical-ai-book-examples
```

---

## Verify Full Environment

Run this checklist to confirm everything works:

```bash
# Check ROS 2 version
ros2 --version
# Expected: ros2 doctor 0.10.x

# Check Python version
python3 --version
# Expected: Python 3.10.x

# Check colcon
colcon version-check
# Expected: colcon-core 0.x.x

# List ROS 2 packages
ros2 pkg list | grep rclpy
# Expected: rclpy listed

# Test RViz
ros2 run rviz2 rviz2 &
# RViz window opens (close after verification)
```

**All checks pass?** You're ready to start Module 1!

---

## Additional Tools (Optional)

### Visual Studio Code with ROS 2 Extension

1. Install VS Code: [code.visualstudio.com](https://code.visualstudio.com/)
2. Install ROS extension: Search "ROS" in Extensions, install by Microsoft
3. Open workspace: `code ~/physical-ai-book-examples`

### Python IDE (PyCharm)

1. Download PyCharm Community: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
2. Configure Python interpreter to use system Python 3.10
3. Add ROS 2 environment variables to run configurations

---

## Next Steps

Environment ready? Start learning:

1. **Read Chapter 1**: [ROS 2 Fundamentals](../docs/module-1/chapter-1-fundamentals.md)
2. **Run examples**: Follow along in `chapter-1-fundamentals/`
3. **Experiment**: Modify code examples and see what happens!

---

## Reference

- **ROS 2 Humble Documentation**: [docs.ros.org/en/humble](https://docs.ros.org/en/humble/)
- **Ubuntu 22.04 Installation**: [ubuntu.com/download/desktop](https://ubuntu.com/download/desktop)
- **Docker Desktop**: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

---

## Version Summary

| Component | Version | Notes |
|-----------|---------|-------|
| ROS 2 Distribution | Humble Hawksbill | LTS until May 2027 |
| Ubuntu | 22.04 LTS (Jammy Jellyfish) | Recommended OS |
| Python | 3.10.x | Default on Ubuntu 22.04 |
| Docusaurus | 3.x | For viewing book locally |
| Docker (if using) | 24.x+ | Cross-platform alternative |

---

**Questions or Issues?**
- Check [Troubleshooting](#troubleshooting) section above
- Visit ROS 2 community: [discourse.ros.org](https://discourse.ros.org/)
- Report book issues: [GitHub Issues](https://github.com/[user]/physical-ai-book/issues)
