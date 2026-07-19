# Data Model: Module 1 - The Robotic Nervous System (ROS 2)

**Date**: 2025-12-23
**Phase**: 1 - Design & Contracts
**Purpose**: Define content structure model for Module 1

## Content Entities

While this is a documentation project (not a database-driven application), we model content structure as entities to ensure consistency and completeness.

---

### Entity: Module

Represents a major thematic section of the book (e.g., Module 1: ROS 2)

**Attributes**:
- `module_id`: Unique identifier (e.g., "module-1")
- `module_number`: Sequential number (e.g., 1)
- `title`: Full module name (e.g., "The Robotic Nervous System (ROS 2)")
- `description`: 2-3 sentence overview of module scope
- `learning_objectives`: List of 3-5 high-level outcomes
- `prerequisites`: List of required prior knowledge
- `estimated_duration`: Reading time estimate (e.g., "3-4 hours")

**Relationships**:
- Contains 1+ Chapters
- Maps to User Stories in spec.md

**Validation Rules**:
- Title must be concise (max 60 characters)
- Learning objectives must be measurable and testable
- Prerequisites must reference earlier modules or external resources

**Example**:
```yaml
module_id: "module-1"
module_number: 1
title: "The Robotic Nervous System (ROS 2)"
description: "Learn how ROS 2 serves as middleware for robot control, integrate Python AI agents with robot controllers, and model humanoid robots using URDF."
learning_objectives:
  - "Understand ROS 2 nodes, topics, and services communication patterns"
  - "Integrate Python AI agents with ROS 2 using rclpy"
  - "Create URDF models for humanoid robots"
prerequisites:
  - "Basic Python programming (functions, classes, imports)"
  - "Command-line familiarity (terminal, package installation)"
estimated_duration: "3-4 hours"
```

---

### Entity: Chapter

Represents a focused learning unit within a module (e.g., Chapter 1: ROS 2 Fundamentals)

**Attributes**:
- `chapter_id`: Unique identifier (e.g., "chapter-1-fundamentals")
- `chapter_number`: Sequential number within module (e.g., 1)
- `title`: Chapter name (e.g., "ROS 2 Fundamentals")
- `description`: 1-2 sentence chapter summary
- `learning_outcomes`: List of 3-5 specific skills gained
- `sections`: List of Section IDs (hierarchical structure)
- `code_examples`: List of CodeExample IDs
- `external_links`: List of ExternalLink IDs
- `estimated_reading_time`: Minutes to complete (e.g., 45)

**Relationships**:
- Belongs to one Module
- Contains 3-7 Sections
- References 0+ CodeExamples
- References 0+ ExternalLinks
- Maps to one User Story from spec.md

**Validation Rules**:
- Chapter number must be sequential within module
- Learning outcomes must be specific and actionable (use verbs: "create", "explain", "implement")
- Sections must form logical progression

**Example**:
```yaml
chapter_id: "chapter-1-fundamentals"
chapter_number: 1
title: "ROS 2 Fundamentals"
description: "Learn the core concepts of ROS 2: nodes, topics, services, and lifecycle management."
learning_outcomes:
  - "Explain what a ROS 2 node is and its role in robot systems"
  - "Create publisher and subscriber nodes for topic-based communication"
  - "Differentiate between topics, services, and actions"
  - "Apply lifecycle management to robot components"
sections:
  - "introduction-to-ros2"
  - "nodes-and-communication"
  - "topics-vs-services"
  - "lifecycle-management"
  - "hands-on-publisher-subscriber"
code_examples:
  - "publisher_node"
  - "subscriber_node"
  - "lifecycle_node"
external_links:
  - "ros2-humble-docs"
  - "lifecycle-tutorial"
estimated_reading_time: 45
```

---

### Entity: Section

Represents a subdivision within a chapter (e.g., "Nodes and Communication")

**Attributes**:
- `section_id`: Unique identifier (e.g., "nodes-and-communication")
- `title`: Section heading (e.g., "Nodes and Communication")
- `content_blocks`: Ordered list of content types (paragraph, code, diagram, callout)
- `subsections`: List of nested Section IDs (optional)
- `key_concepts`: List of terms introduced in this section

**Relationships**:
- Belongs to one Chapter
- May contain nested Sections (2-3 levels max for readability)
- References CodeExamples and Diagrams

**Content Block Types**:
- `paragraph`: Explanatory text
- `code`: Fenced code block with syntax highlighting
- `diagram`: Mermaid diagram or image reference
- `callout`: Info/warning/tip admonition
- `list`: Bulleted or numbered list
- `table`: Comparison or reference table

**Example**:
```yaml
section_id: "nodes-and-communication"
title: "Nodes and Communication"
key_concepts:
  - "ROS 2 Node"
  - "Topic"
  - "Publisher"
  - "Subscriber"
content_blocks:
  - type: "paragraph"
    text: "A ROS 2 node is an independent process..."
  - type: "diagram"
    diagram_id: "node-communication-graph"
  - type: "code"
    code_example_id: "publisher_node"
  - type: "callout"
    callout_type: "tip"
    text: "Use descriptive node names like /camera_driver rather than /node1"
```

---

### Entity: CodeExample

Represents a runnable code snippet or complete file

**Attributes**:
- `example_id`: Unique identifier (e.g., "publisher_node")
- `title`: Descriptive name (e.g., "Simple Publisher Node")
- `language`: Programming language (e.g., "python")
- `file_path`: Path in companion repo (e.g., "module-1-ros2/chapter-1-fundamentals/publisher_node.py")
- `code_snippet`: Inline code (if short) or reference to companion repo (if long)
- `explanation`: What the code demonstrates
- `expected_output`: What reader should see when running
- `dependencies`: List of required packages (e.g., "rclpy", "std_msgs")
- `run_instructions`: How to execute the code

**Relationships**:
- Referenced by Sections
- Stored in companion repository
- Maps to Functional Requirements (FR-002, FR-008)

**Validation Rules**:
- Code must be complete and runnable (no pseudocode)
- Expected output must be documented
- Dependencies must be explicit (package.xml, requirements.txt)

**Example**:
```yaml
example_id: "publisher_node"
title: "Simple Publisher Node"
language: "python"
file_path: "module-1-ros2/chapter-1-fundamentals/publisher_node.py"
explanation: "Demonstrates creating a ROS 2 publisher node that sends string messages to a topic at 1 Hz."
expected_output: |
  [INFO] [1703341200.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
  [INFO] [1703341201.123456789] [minimal_publisher]: Publishing: "Hello World: 1"
dependencies:
  - "rclpy"
  - "std_msgs"
run_instructions: |
  1. cd module-1-ros2/chapter-1-fundamentals
  2. colcon build
  3. source install/setup.bash
  4. ros2 run my_package publisher_node
```

---

### Entity: Diagram

Represents a visual illustration (Mermaid or image)

**Attributes**:
- `diagram_id`: Unique identifier (e.g., "node-communication-graph")
- `title`: Descriptive caption
- `type`: "mermaid" or "image"
- `source`: Mermaid syntax or image file path
- `alt_text`: Accessibility description
- `caption`: Explanatory text below diagram

**Example (Mermaid)**:
```yaml
diagram_id: "node-communication-graph"
title: "ROS 2 Node Communication via Topics"
type: "mermaid"
source: |
  graph LR
      A[Publisher Node] -->|chatter topic| B[Subscriber Node]
alt_text: "Diagram showing a publisher node sending messages to a subscriber node via the chatter topic"
caption: "Publishers and subscribers communicate asynchronously through named topics"
```

**Example (Image)**:
```yaml
diagram_id: "urdf-rviz-visualization"
title: "Humanoid Robot in RViz"
type: "image"
source: "docs/assets/module-1/humanoid-urdf-rviz.png"
alt_text: "Screenshot of a humanoid robot model displayed in RViz with joint frames visible"
caption: "URDF model loaded in RViz showing joint coordinate frames and link geometry"
```

---

### Entity: ExternalLink

Represents a hyperlink to external documentation or resources

**Attributes**:
- `link_id`: Unique identifier (e.g., "ros2-humble-docs")
- `title`: Link text (e.g., "ROS 2 Humble Documentation")
- `url`: Full URL (e.g., "https://docs.ros.org/en/humble/")
- `description`: 1-2 sentences explaining what reader will find
- `link_type`: "official_docs", "tutorial", "reference", "video"

**Validation Rules**:
- URLs must use version-specific paths (e.g., `/en/humble/` not `/en/latest/`)
- Description must explain WHY reader should visit link
- Link type helps reader understand resource category

**Example**:
```yaml
link_id: "ros2-humble-docs"
title: "ROS 2 Humble Documentation"
url: "https://docs.ros.org/en/humble/"
description: "Official ROS 2 Humble documentation with comprehensive API reference and tutorials."
link_type: "official_docs"
```

---

## Content Structure Hierarchy

```
Module (e.g., Module 1: ROS 2)
├── Chapter 1: ROS 2 Fundamentals
│   ├── Section: Introduction to ROS 2
│   │   ├── Paragraph (text)
│   │   └── Diagram (mermaid)
│   ├── Section: Nodes and Communication
│   │   ├── Paragraph (text)
│   │   ├── CodeExample: publisher_node
│   │   └── ExternalLink: ros2-humble-docs
│   └── Section: Hands-On Exercise
│       ├── CodeExample: subscriber_node
│       └── Callout (tip)
├── Chapter 2: Python Agents & ROS 2 Integration
│   └── [Similar structure]
└── Chapter 3: Humanoid Robot Description
    └── [Similar structure]
```

---

## Content Quality Constraints

### Writing Style
- **Tone**: Conversational but professional, second person ("you will create...")
- **Sentence length**: Vary between 10-25 words for readability
- **Paragraph length**: 3-5 sentences max before subheading or code example
- **Technical terms**: Define on first use, link to glossary for complex terms

### Code Style
- **Language**: Python 3.10, PEP-8 compliant
- **ROS 2 style**: Follow ROS 2 Python style guide
- **Comments**: Explain WHY, not WHAT (assume reader can read code)
- **Naming**: Descriptive variable/function names (no single letters except loop counters)

### Accessibility
- **Alt text**: All images must have descriptive alt text
- **Color**: Don't rely solely on color for meaning (use labels, patterns)
- **Headings**: Hierarchical structure (H1 → H2 → H3, no skipping levels)
- **Links**: Descriptive link text (avoid "click here", use "ROS 2 Humble documentation")

---

## State Transitions

Content progresses through these states during creation:

1. **Drafted**: Initial content written, may have TODOs or placeholders
2. **Code-Verified**: All code examples tested and run successfully
3. **Link-Validated**: External links checked and accessible
4. **Reviewed**: Content reviewed for accuracy and clarity
5. **Published**: Deployed to GitHub Pages

**Validation Checklist** (before "Published" state):
- [ ] All CodeExamples run without errors
- [ ] All ExternalLinks return 200 status
- [ ] All Diagrams render correctly
- [ ] Learning outcomes align with content
- [ ] Acceptance criteria from spec.md verified

---

## Relationship to Spec Entities

Mapping content model to spec.md entities:

| Spec Entity | Content Entity | Relationship |
|-------------|----------------|-------------|
| User Story 1 (ROS 2 Fundamentals) | Chapter 1 | 1:1 mapping |
| User Story 2 (Python Integration) | Chapter 2 | 1:1 mapping |
| User Story 3 (URDF Modeling) | Chapter 3 | 1:1 mapping |
| FR-001 to FR-003 | Chapter 1 Sections | Requirements → Content |
| FR-004 to FR-005 | Chapter 2 Sections + CodeExamples | Requirements → Content |
| FR-006 to FR-007 | Chapter 3 Sections + CodeExamples | Requirements → Content |
| FR-008 to FR-012 | All CodeExamples | Quality constraints |
| SC-001 to SC-007 | Entire Module 1 | Success validation |

---

## Notes

This data model ensures:
- **Consistency**: All chapters follow same structure
- **Traceability**: Content maps back to spec requirements
- **Quality**: Built-in validation rules prevent incomplete content
- **Scalability**: Model extends to future modules without modification
