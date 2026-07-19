# Companion Repository Specification: Module 4 - VLA

**Feature**: Module 4 VLA example code repository structure
**Created**: 2025-12-25
**Purpose**: Define structure and content for `physical-ai-book-examples/module-4-vla/`

## Repository Structure

```text
physical-ai-book-examples/module-4-vla/
├── README.md
├── requirements.txt
├── .env.example
├── chapter-1-voice-to-action/
│   ├── README.md
│   ├── scripts/
│   │   ├── install_whisper.sh
│   │   ├── verify_microphone.py
│   │   ├── realtime_transcription.py
│   │   ├── intent_parser.py
│   │   ├── voice_action_client.py
│   │   ├── confidence_filtering.py
│   │   ├── spacy_intent_parser.py
│   │   ├── ambiguity_handler.py
│   │   └── noise_detection.py
│   ├── audio_samples/
│   │   ├── test_commands.wav
│   │   ├── move_forward.wav
│   │   ├── navigate_kitchen.wav
│   │   └── pick_red_cube.wav
│   ├── configs/
│   │   └── action_primitives.json
│   └── expected_output.txt
├── chapter-2-llm-planning/
│   ├── README.md
│   ├── scripts/
│   │   ├── openai_api_example.py
│   │   ├── llama3_ollama_example.py
│   │   ├── prompt_templates.py
│   │   ├── plan_validator.py
│   │   ├── plan_executor.py
│   │   ├── safety_validator.py
│   │   ├── replanner.py
│   │   └── nav2_integration.py
│   ├── prompts/
│   │   ├── system_prompt_robot_planner.txt
│   │   ├── example_tasks.json
│   │   └── guardrails.txt
│   ├── configs/
│   │   ├── action_schema.json
│   │   ├── safety_constraints.yaml
│   │   └── llm_config.yaml
│   └── expected_output.txt
└── chapter-3-capstone/
    ├── README.md
    ├── scripts/
    │   ├── capstone_main.py
    │   ├── voice_handler.py
    │   ├── llm_planner.py
    │   ├── nav_controller.py
    │   ├── perception_handler.py
    │   ├── manipulation_controller.py
    │   ├── metrics_tracker.py
    │   └── test_scenarios.py
    ├── launch/
    │   └── capstone_full_stack.launch.py
    ├── configs/
    │   ├── capstone_params.yaml
    │   ├── test_scenarios.json
    │   └── performance_targets.yaml
    └── expected_output.txt
```

## Top-Level Files

### README.md

```markdown
# Module 4: Vision-Language-Action (VLA) - Example Code

Complete code examples for Physical AI Book Module 4.

## Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10+
- ROS 2 Humble
- NVIDIA Isaac Sim (for Chapter 3 capstone)
- USB microphone or built-in laptop mic

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Whisper:
   ```bash
   cd chapter-1-voice-to-action/scripts
   bash install_whisper.sh
   ```

3. Configure API keys (optional for LLM planning):
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. For free alternative (LLaMA 3):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3
   ```

## Quick Start

### Chapter 1: Voice-to-Action
```bash
cd chapter-1-voice-to-action/scripts
python verify_microphone.py
python realtime_transcription.py
```

### Chapter 2: LLM Planning
```bash
cd chapter-2-llm-planning/scripts
python openai_api_example.py  # OR
python llama3_ollama_example.py
```

### Chapter 3: Capstone
```bash
cd chapter-3-capstone
ros2 launch launch/capstone_full_stack.launch.py
```

## Cost Transparency

- **Free option**: Local LLaMA 3 via Ollama (no API costs)
- **Paid option**: OpenAI GPT-4 API (~$5-10 for complete module)

All examples include both free and paid options.
```

### requirements.txt

```text
# Chapter 1: Voice-to-Action
openai-whisper>=20231117
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
sounddevice>=0.4.6
numpy>=1.24.0
scipy>=1.10.0
spacy>=3.7.0

# Chapter 2: LLM Planning
openai>=1.3.0
anthropic>=0.7.0
requests>=2.31.0
jsonschema>=4.20.0
pyyaml>=6.0

# Chapter 3: Capstone Integration
opencv-python>=4.8.0
matplotlib>=3.7.0

# ROS 2 (installed separately via apt)
# rclpy, nav2_msgs, geometry_msgs, std_msgs
```

### .env.example

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_MAX_TOKENS=1000

# Anthropic API Configuration (optional)
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Whisper Configuration
WHISPER_MODEL=base  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # cpu or cuda

# Microphone Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE=1024

# Safety Constraints
MAX_VELOCITY=1.0  # m/s
MAX_FORCE=50.0    # N
WORKSPACE_X_MIN=-5.0
WORKSPACE_X_MAX=5.0
WORKSPACE_Y_MIN=-5.0
WORKSPACE_Y_MAX=5.0
```

## Chapter 1: Voice-to-Action

### Key Files

**scripts/verify_microphone.py**
- Lists available audio devices
- Records test audio
- Verifies microphone functionality

**scripts/realtime_transcription.py**
- Streams audio from microphone
- Transcribes in real-time with Whisper base model
- Displays confidence scores

**scripts/intent_parser.py**
- Extracts action type, object, parameters from transcribed text
- Uses regex patterns for common robot commands
- Returns structured intent dictionary

**scripts/voice_action_client.py**
- Complete example: microphone → Whisper → intent → ROS 2 action
- Sends cmd_vel or NavigateToPose goals
- Monitors action feedback and result

**configs/action_primitives.json**
```json
{
  "move_forward": {
    "action_type": "cmd_vel",
    "parameters": ["distance"]
  },
  "navigate_to": {
    "action_type": "NavigateToPose",
    "parameters": ["location"]
  },
  "pick_object": {
    "action_type": "GraspObject",
    "parameters": ["object_id", "color"]
  }
}
```

## Chapter 2: LLM Planning

### Key Files

**scripts/openai_api_example.py**
- Calls OpenAI GPT-4 Chat Completions API
- Sends robot planning prompts
- Parses JSON-formatted plans

**scripts/llama3_ollama_example.py**
- Calls local LLaMA 3 via Ollama REST API
- Same prompts as GPT-4 version
- Free alternative for students

**scripts/plan_validator.py**
- Validates JSON schema
- Checks action types exist
- Validates parameter ranges
- Enforces safety constraints

**scripts/plan_executor.py**
- Converts LLM JSON plan to ROS 2 action sequence
- Sends goals sequentially
- Monitors execution state

**prompts/system_prompt_robot_planner.txt**
```text
You are a robot action planner. Given a high-level task, decompose it into a sequence of atomic robot actions.

Available actions:
- navigate_to(location: str)
- grasp_object(object_id: str, color: str)
- place_object(location: str)
- open_door(door_id: str)
- close_door(door_id: str)

Output format (JSON):
{
  "task": "high-level task description",
  "steps": [
    {"step_id": 1, "action": "navigate_to", "parameters": {"location": "kitchen"}, "expected_duration": 10.0},
    ...
  ]
}

Safety constraints:
- Max velocity: 1.0 m/s
- Max gripper force: 50 N
- Workspace boundaries: x[-5,5], y[-5,5]

CRITICAL: Output ONLY valid JSON. Do not include explanations or markdown formatting.
```

**configs/action_schema.json**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "task": {"type": "string"},
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step_id": {"type": "integer"},
          "action": {"type": "string", "enum": ["navigate_to", "grasp_object", "place_object", "open_door", "close_door"]},
          "parameters": {"type": "object"},
          "expected_duration": {"type": "number"}
        },
        "required": ["step_id", "action", "parameters"]
      }
    }
  },
  "required": ["task", "steps"]
}
```

## Chapter 3: Capstone Project

### Key Files

**scripts/capstone_main.py**
- Main orchestration script
- Integrates all modules (voice, LLM, Nav2, YOLO, manipulation)
- State machine for task execution

**scripts/voice_handler.py**
```python
class VoiceHandler:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.intent_parser = IntentParser()

    def listen(self):
        """Stream audio from microphone"""
        pass

    def transcribe(self, audio):
        """Transcribe audio with Whisper"""
        pass

    def extract_intent(self, text):
        """Parse intent from transcribed text"""
        pass
```

**scripts/llm_planner.py**
```python
class LLMPlanner:
    def __init__(self, api_key, model="gpt-4-turbo-preview"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_plan(self, task):
        """Generate multi-step plan from high-level task"""
        pass

    def validate_plan(self, plan):
        """Validate plan for feasibility and safety"""
        pass
```

**scripts/nav_controller.py**
```python
class NavController:
    def __init__(self):
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def navigate_to(self, location):
        """Send NavigateToPose goal to Nav2"""
        pass
```

**scripts/perception_handler.py**
```python
class PerceptionHandler:
    def __init__(self):
        self.yolo_model = YOLO("yolov8n.pt")

    def detect_objects(self, image):
        """Detect objects with YOLOv8"""
        pass

    def filter_by_color(self, detections, color):
        """Filter detections by color attribute"""
        pass
```

**scripts/manipulation_controller.py**
```python
class ManipulationController:
    def __init__(self):
        self.grasp_client = ActionClient(self, GraspObject, 'grasp_object')

    def grasp_object(self, object_id, max_retries=3):
        """Grasp object with retry logic"""
        pass
```

**launch/capstone_full_stack.launch.py**
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='module_4_vla',
            executable='capstone_main',
            name='capstone_main',
            parameters=[{'config_file': 'capstone_params.yaml'}]
        ),
        # Add Nav2, Isaac Sim, YOLO nodes
    ])
```

## Testing and Validation

### Expected Outputs

Each chapter includes `expected_output.txt` documenting:
- Command sequences to run
- Expected console output
- Success criteria (e.g., "Whisper transcription accuracy greater than 95%")
- Failure modes and troubleshooting

### Performance Targets

```yaml
chapter_1:
  transcription_accuracy: 0.95  # quiet environment
  action_success_rate: 0.90
  latency_target: 1.0  # seconds

chapter_2:
  plan_generation_latency: 3.0  # seconds
  plan_feasibility_rate: 0.85
  json_formatting_accuracy: 1.0

chapter_3:
  task_completion_rate: 0.90
  end_to_end_latency: 30.0  # seconds for simple task
  grasp_success_rate: 0.85
```

## CI/CD Validation

Repository includes GitHub Actions workflows:
- Run all code examples on Ubuntu 22.04 + ROS 2 Humble
- Validate expected outputs match actual outputs
- Check API cost estimates (mock OpenAI API)
- Ensure all imports resolve

---

**Maintenance**: Update code examples when dependencies release breaking changes. Test quarterly with latest ROS 2 Humble patches.
