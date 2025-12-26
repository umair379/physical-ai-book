# Module 4: Vision-Language-Action (VLA)

Master voice-controlled humanoid robots using OpenAI Whisper for speech recognition, large language models for cognitive planning, and end-to-end autonomous task execution.

## What You'll Learn

- Convert natural language voice commands into structured robot actions using OpenAI Whisper
- Parse voice transcriptions to extract intent (action type, object parameters, spatial constraints)
- Use LLMs (GPT-4, Claude, LLaMA 3) to generate multi-step robot action plans from high-level tasks
- Validate and execute LLM-generated plans with safety constraints (velocity limits, workspace boundaries)
- Integrate voice recognition, LLM planning, Nav2 navigation, and YOLOv8 perception into autonomous systems

## Prerequisites

Before starting this module, you should have:

- **Module 1 completion**: ROS 2 actions, action clients, action servers
- **Module 2 completion**: Gazebo/Unity simulation for robot testing
- **Module 3 completion**: Isaac ROS perception, Nav2 navigation, YOLOv8 object detection
- **Python 3.10+** with PyTorch 2.0+ installed
- **USB microphone** or laptop mic with Ubuntu 22.04 ALSA/PulseAudio support
- **OpenAI API key** OR local LLaMA 3 installation via Ollama

:::warning API Costs
This module uses LLM APIs for cognitive planning. Estimated cost: **$5-10 per student** for OpenAI GPT-4 usage. **Free alternatives** available (local LLaMA 3 via Ollama) - all exercises include both paid and free options.
:::

## Module Structure

This module is organized into 3 chapters that build progressively:

### [Chapter 1: Voice-to-Action with OpenAI Whisper](./chapter-1-voice-to-action)

Learn to convert natural language voice commands into executable robot actions using OpenAI Whisper for speech-to-text, intent parsing, and ROS 2 action mapping.

**Learning Outcomes**:
- Install OpenAI Whisper (base model) and configure microphone for real-time audio streaming
- Transcribe voice commands with 95%+ word accuracy in quiet environments
- Parse transcribed text to extract action intent and parameters
- Map parsed intents to ROS 2 action goals (cmd_vel, NavigateToPose, custom manipulation)
- Handle low-confidence transcriptions and ambiguous commands with clarification prompts

**Estimated Time**: 100 minutes

### [Chapter 2: Cognitive Planning with LLMs](./chapter-2-llm-planning)

Master LLM-based robot planning where GPT-4 or LLaMA 3 decomposes high-level tasks into multi-step action sequences with validation and error handling.

**Learning Outcomes**:
- Design LLM prompts for robot planning with JSON-formatted output schemas
- Call OpenAI GPT-4 API or local LLaMA 3 to generate 5+ step plans from tasks like "Clean the room"
- Validate LLM-generated plans for feasibility (action types exist, parameters in valid ranges, no safety violations)
- Execute multi-step plans using ROS 2 action sequences from Module 3 (Nav2, manipulation)
- Implement replanning when actions fail (send context to LLM, request alternative plan within 5 seconds)

**Estimated Time**: 110 minutes

### [Chapter 3: Capstone Project - Autonomous Humanoid](./chapter-3-capstone)

Integrate all modules (ROS 2, simulation, Isaac perception, Nav2, voice, LLM) into a complete autonomous humanoid system executing voice-commanded multi-step tasks.

**Learning Outcomes**:
- Build end-to-end pipeline: voice command → Whisper transcription → intent parsing → LLM planning → Nav2 navigation → YOLO detection → manipulation
- Handle dynamic obstacles during navigation (Nav2 local planner re-routing)
- Resolve ambiguities with LLM reasoning (multiple object candidates, unclear spatial references)
- Implement grasp failure recovery (3 retries with adjusted positions, then replan)
- Achieve 90%+ task completion rate on 10 sequential voice-commanded tasks

**Estimated Time**: 120 minutes

## Total Estimated Duration

**7-9 hours** of hands-on learning including reading, coding exercises, and testing

## Performance Targets

By the end of this module, your autonomous humanoid system should achieve:

- **95%+ transcription accuracy** in quiet environments (SNR greater than 20 dB)
- **90%+ action success rate** for voice-triggered robot actions
- **85%+ plan feasibility** for LLM-generated multi-step plans
- **90%+ task completion rate** for end-to-end autonomous tasks

## Cost Transparency

### Paid Option (Recommended for Best Quality)
- **OpenAI GPT-4 API**: $0.01 per 1K input tokens, $0.03 per 1K output tokens
- **Estimated total**: $5-10 per student for complete module
- **Advantages**: Best plan quality, most reliable JSON formatting, fastest latency

### Free Option (Fully Functional Alternative)
- **Local LLaMA 3 8B via Ollama**: No API costs, runs on your machine
- **Requirements**: 16GB RAM recommended, 8GB minimum
- **Trade-offs**: Slightly lower plan quality, slower inference, acceptable for learning

All exercises in this module provide both GPT-4 and LLaMA 3 code examples.

## Chapter Navigation

Ready to begin? Start with [Chapter 1: Voice-to-Action](./chapter-1-voice-to-action) to learn how to convert spoken commands into robot actions.

---

**Next Chapter**: [Voice-to-Action with OpenAI Whisper →](./chapter-1-voice-to-action)
