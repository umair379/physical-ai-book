# Research: Module 1 - The Robotic Nervous System (ROS 2)

**Date**: 2025-12-23
**Phase**: 0 - Outline & Research
**Purpose**: Document technology decisions, best practices, and implementation approach for Module 1

## Technology Decisions

### Decision 1: Docusaurus Version and Setup

**Decision**: Use Docusaurus 3.x (latest stable) with classic template

**Rationale**:
- Docusaurus 3.x is the current stable release with best React 18 support
- Classic template provides clean, documentation-focused layout suitable for educational content
- Built-in features: versioning, search, dark mode, mobile-responsive
- Native MDX support allows embedding React components if needed in future
- Strong community support and extensive plugin ecosystem

**Alternatives Considered**:
- **VuePress**: Good documentation generator, but smaller ecosystem than Docusaurus
- **MkDocs**: Python-based, simpler but less flexible for future interactive features
- **GitBook**: Commercial product with licensing costs, less control over deployment
- **Custom Next.js site**: More flexibility but significantly more development overhead

**Implementation Notes**:
- Install via: `npx create-docusaurus@latest physical-ai-book classic --typescript`
- Use TypeScript for better type safety in future custom components
- GitHub Pages deployment via official Docusaurus GitHub Actions workflow

---

### Decision 2: Content Organization Strategy

**Decision**: Hierarchical module/chapter structure with index pages

**Rationale**:
- Mirrors educational structure (Module → Chapters → Sections)
- Docusaurus sidebar automatically reflects directory hierarchy
- Easy to navigate and expand as more modules are added
- Clear separation between modules enables independent updates

**Structure**:
```
docs/
├── intro.md                     # Book introduction
├── module-1/
│   ├── index.md                 # Module 1 overview
│   ├── chapter-1-fundamentals.md
│   ├── chapter-2-python-integration.md
│   └── chapter-3-urdf-modeling.md
└── [future modules...]
```

**Alternatives Considered**:
- **Flat structure**: All chapters at root level - rejected because hard to scale with multiple modules
- **Topic-based grouping**: Organize by concept (nodes, topics, etc.) instead of chapters - rejected because less intuitive for linear learning path
- **Single-page per module**: One long scrolling page - rejected because poor UX for navigation and updates

**Implementation Notes**:
- Use sidebar category grouping in `sidebars.js`
- Add "Previous/Next" navigation at bottom of each chapter
- Module index pages provide learning objectives and navigation to chapters

---

### Decision 3: Code Example Strategy

**Decision**: Separate companion repository for runnable code, embed examples in book as fenced code blocks with links

**Rationale**:
- **Separation of concerns**: Book content vs. executable code
- **Testing**: Companion repo can have CI/CD to validate all examples run correctly (addresses FR-008, SC-003)
- **Reproducibility**: Readers clone companion repo to run examples locally (addresses Principle III)
- **Version control**: Code examples can be updated independently without triggering book rebuilds
- **Size management**: Keeps book repository lean

**Implementation**:
- Create `physical-ai-book-examples` repository on GitHub
- Organize by module/chapter matching book structure
- Each chapter's code directory includes:
  - Python files with complete, runnable code
  - `package.xml` for ROS 2 package dependencies
  - `README.md` with setup and run instructions
  - Expected output files for verification
- Book chapters link to specific files: `[View complete code →](https://github.com/user/physical-ai-book-examples/blob/main/module-1-ros2/chapter-1-fundamentals/publisher_node.py)`

**Alternatives Considered**:
- **Embed code directly in book repo**: Rejected because mixing content and code makes testing harder
- **Use code snippets services (CodeSandbox, Replit)**: Rejected because ROS 2 requires specific environment, not browser-compatible
- **GitHub Gists**: Rejected because harder to organize and version control multiple related files

**Implementation Notes**:
- Use GitHub Actions in companion repo to run all examples weekly (validates they don't break with ROS 2 updates)
- Include Dockerfile in companion repo for reproducible environment
- Add badges to README showing test status

---

### Decision 4: External Documentation Linking Strategy

**Decision**: Link to official ROS 2 documentation (docs.ros.org) and URDF tutorials with context and summary

**Rationale**:
- Avoids duplicating official documentation (reduces maintenance burden)
- Official docs are canonical source of truth (aligns with Principle II: Accuracy)
- Provides readers with path to deeper learning
- Book focuses on educational narrative and integration, official docs cover API reference details

**Linking Pattern**:
```markdown
## ROS 2 Lifecycle Management

Lifecycle management allows nodes to transition through defined states...
[explanation with diagrams]

**Learn more**:
- [ROS 2 Lifecycle Management Guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/Lifecycle.html) - Official tutorial with detailed state machine
- [Managed Nodes Design](https://design.ros2.org/articles/node_lifecycle.html) - Architecture rationale
```

**Alternatives Considered**:
- **Copy documentation content**: Rejected because creates maintenance burden and potential for outdated info
- **No external links**: Rejected because readers need authoritative reference material
- **Link without context**: Rejected because readers need guidance on what to read and why

**Implementation Notes**:
- Always use version-specific URLs (e.g., `/en/humble/` for ROS 2 Humble)
- Provide 1-2 sentence context for each external link explaining what reader will find
- Validate links during build (consider using Docusaurus broken-link-checker plugin)

---

### Decision 5: Diagram and Visualization Approach

**Decision**: Use Mermaid diagrams (built into Docusaurus 3) for architecture diagrams, PNG screenshots for RViz/Gazebo visualizations

**Rationale**:
- **Mermaid**: Text-based diagrams version-controlled with content, easy to update, renders in browser
- **PNG screenshots**: RViz/Gazebo visualizations must be actual screenshots for authenticity
- **No external diagram tools**: Avoids dependency on third-party services (Lucidchart, Draw.io)

**Usage**:
- Mermaid for: ROS 2 node communication graphs, data flow diagrams, state machines
- PNG for: URDF robot visualizations in RViz, Gazebo simulation screenshots, terminal outputs

**Example**:
```markdown
### ROS 2 Node Communication

\`\`\`mermaid
graph LR
    A[Sensor Node] -->|sensor_msgs/LaserScan| B[AI Agent Node]
    B -->|geometry_msgs/Twist| C[Controller Node]
\`\`\`
```

**Alternatives Considered**:
- **All PNG images**: Rejected because harder to update and version control
- **External diagram services**: Rejected because creates external dependency
- **D3.js custom visualizations**: Rejected because too much development overhead for static diagrams

**Implementation Notes**:
- Store PNG images in `docs/assets/module-1/`
- Use descriptive filenames: `ros2-pub-sub-architecture.png`, `humanoid-urdf-rviz.png`
- Optimize images (use tools like ImageOptim or Squoosh) for fast page loads

---

### Decision 6: ROS 2 Distribution and Python Version

**Decision**: Target ROS 2 Humble LTS with Python 3.10

**Rationale**:
- **ROS 2 Humble**: Long-term support until May 2027, stable and widely adopted
- **Python 3.10**: Officially supported by ROS 2 Humble, stable with good ecosystem
- **LTS focus**: Ensures examples remain valid for years, reduces reader confusion from version mismatches

**Compatibility Notes**:
- Ubuntu 22.04 LTS is recommended platform for ROS 2 Humble
- Docker images available for other platforms (macOS, Windows)
- All code examples tested on Ubuntu 22.04 + ROS 2 Humble + Python 3.10

**Alternatives Considered**:
- **ROS 2 Iron/Rolling**: Rejected because not LTS (shorter support lifecycle)
- **Python 3.11+**: Rejected because ROS 2 Humble officially targets 3.10
- **Multi-version support**: Rejected because adds complexity and confusion for learners

**Implementation Notes**:
- Document exact versions in quickstart.md
- Use `python3.10` explicitly in all example shebangs and instructions
- Companion repo CI/CD uses Ubuntu 22.04 Docker image with Humble installed

---

### Decision 7: GitHub Pages Deployment Strategy

**Decision**: Use Docusaurus official GitHub Actions workflow with custom domain support

**Rationale**:
- Official workflow maintained by Docusaurus team
- Automatic deployment on push to main branch
- Supports custom domains (e.g., physicalai.book)
- Free hosting via GitHub Pages

**Workflow**:
1. Developer merges PR to main branch
2. GitHub Actions builds Docusaurus site (`npm run build`)
3. Actions deploys build/ directory to gh-pages branch
4. GitHub Pages serves from gh-pages branch

**Alternatives Considered**:
- **Netlify/Vercel**: Rejected because adds external dependency, GitHub Pages is free and integrated
- **Manual deployment**: Rejected because error-prone and doesn't scale
- **Self-hosted**: Rejected because unnecessary complexity for static site

**Implementation Notes**:
- Configure in `.github/workflows/deploy.yml`
- Set GitHub repo settings: Pages source = gh-pages branch
- Add `CNAME` file for custom domain (if used)
- Use `trailingSlash: false` in docusaurus.config.js for better GitHub Pages compatibility

---

## Best Practices for Educational Content

### Content Writing Guidelines

Based on research of effective technical documentation (Docusaurus docs, Stripe docs, Django tutorial):

1. **Progressive disclosure**: Start simple, add complexity gradually
   - Chapter 1: Basic pub/sub (single topic, simple messages)
   - Chapter 2: Multi-topic integration with logic
   - Chapter 3: Complex URDF with sensors

2. **Learning by doing**: Every concept has runnable example within 2 paragraphs
   - Explain concept → Show code → Run it → See output
   - Readers should be typing and running code, not just reading

3. **Expected outputs**: Always show what reader should see
   - Terminal outputs in fenced code blocks with `bash` syntax highlighting
   - Annotate outputs with comments explaining key lines

4. **Troubleshooting sections**: Address common errors proactively
   - "Common Issues" subsection at end of each chapter
   - Include error messages readers might see and solutions

5. **Navigation aids**:
   - Chapter overview at start: "What you'll learn"
   - Summary at end: "Key takeaways"
   - Links to next chapter: "Next up: Chapter 2"

### Code Example Best Practices

1. **Complete, runnable code**: No pseudocode, no `# ... rest of code`
2. **Comments explain WHY, not WHAT**: Assume reader can read Python
3. **One concept per example**: Don't mix multiple new ideas in same file
4. **Standard naming conventions**: Follow ROS 2 and Python PEP-8 style guides
5. **Error handling**: Show graceful shutdown, keyboard interrupt handling

---

## Integration Points (For Future Modules)

While Module 1 is self-contained, document integration points for future work:

### RAG Chatbot Integration (Future Module)
- Docusaurus supports custom React components
- Chatbot can be embedded as `<ChatWidget />` component in MDX files
- Backend API calls to FastAPI service for RAG queries
- Book content indexed in Qdrant vector database

### Search Integration
- Docusaurus has built-in search (Algolia DocSearch or local search plugin)
- Module 1 uses local search (no external dependencies)
- Future: Upgrade to Algolia for better search quality

### Analytics
- Future: Add Google Analytics or privacy-focused Plausible
- Track which chapters readers spend most time on
- Identify drop-off points in learning journey

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ROS 2 Humble becomes outdated | Medium | Low | LTS until 2027, plan update cycle |
| Code examples break with ROS 2 updates | Medium | High | CI/CD in companion repo tests weekly |
| Docusaurus breaking changes in updates | Low | Medium | Pin Docusaurus version, test upgrades in branch |
| External links to ROS docs break | Medium | Low | Use version-specific URLs, validate links in CI |
| GitHub Pages downtime | Low | Low | Static site cached by browsers, rare outages |
| Readers unable to set up ROS 2 environment | High | High | Provide Docker-based setup alternative |

---

## Implementation Phases Summary

**Phase 0 (Complete)**: Research and decisions documented in this file

**Phase 1 (Next)**:
- Generate data-model.md: Content structure model
- Generate contracts/content-structure.yaml: Chapter/section organization
- Generate quickstart.md: Environment setup guide

**Phase 2 (Via /sp.tasks)**:
- Task breakdown for Docusaurus initialization
- Task breakdown for chapter content creation
- Task breakdown for companion repository setup
- Task breakdown for GitHub Pages deployment

---

## References

- [Docusaurus 3.x Documentation](https://docusaurus.io/docs)
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Mermaid Diagram Syntax](https://mermaid.js.org/intro/)
