# Feature Specification: Blog Page

**Feature Branch**: `006-blog`
**Created**: 2025-12-26
**Status**: Draft
**Input**: User description: "Blog Page - Target audience: Students and developers following course updates and insights. Focus: Sharing progress updates, technical articles, and learning reflections. Purpose: Publish course announcements and module updates, Share technical deep-dives and tutorials, Document project milestones and experiments. Content structure: Chronological posts with clear titles and summaries, Tags for modules (ROS2, Gazebo, Isaac, VLA), Author and publish date metadata, Call-to-action links to related modules. Success criteria: Blog posts render correctly in Docusaurus, Posts are easy to navigate and searchable by tags, Clear linkage between blog posts and course modules"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Browse Recent Blog Posts (Priority: P1) 🎯 MVP

As a student or developer, I want to see a chronological list of recent blog posts so I can stay updated on course announcements, module releases, and technical insights without leaving the documentation site.

**Why this priority**: This is the core value proposition of a blog - providing a centralized feed of updates. Without this, there's no way for users to discover new content. This story alone delivers value as a minimal blog.

**Independent Test**: Navigate to the blog page and verify that recent posts are displayed in reverse chronological order (newest first) with clear titles, summaries, publish dates, and author names. Users should be able to click on any post to read the full content.

**Acceptance Scenarios**:

1. **Given** a user visits the blog homepage, **When** the page loads, **Then** they see a list of blog posts sorted by publish date (newest first) with visible titles, excerpts, dates, and authors
2. **Given** a user sees the blog list, **When** they click on a post title, **Then** they are taken to the full blog post page with complete content
3. **Given** a user reads a blog post, **When** they scroll to the bottom, **Then** they see navigation to previous/next posts or a link back to the blog list

---

### User Story 2 - Filter Posts by Module Tag (Priority: P2)

As a student focusing on a specific module (e.g., ROS 2, Isaac Sim, VLA), I want to filter blog posts by module tags so I can quickly find relevant tutorials and updates without reading through unrelated posts.

**Why this priority**: Enhances discoverability for targeted learners. Not critical for MVP since users can still browse all posts, but significantly improves user experience for focused learning paths.

**Independent Test**: On the blog page, click on a tag (e.g., "ROS2" or "Isaac") and verify that only posts tagged with that module are displayed. Tag filters should work independently of the chronological blog list.

**Acceptance Scenarios**:

1. **Given** a user is on the blog homepage, **When** they click on a tag (e.g., "ROS2"), **Then** the page filters to show only posts tagged with "ROS2"
2. **Given** a user has applied a tag filter, **When** they click on another tag (e.g., "VLA"), **Then** the page updates to show only posts tagged with "VLA" (replacing the previous filter)
3. **Given** a user has applied a tag filter, **When** they click "Clear filters" or the blog homepage link, **Then** all posts are displayed again without filtering
4. **Given** a blog post is tagged with multiple modules (e.g., "ROS2" and "Gazebo"), **When** a user filters by either tag, **Then** that post appears in the filtered results

---

### User Story 3 - Navigate to Related Course Modules (Priority: P3)

As a reader of a blog post about a specific topic (e.g., "Setting up Nav2 for autonomous navigation"), I want to see clear call-to-action links to the related course module so I can easily continue learning in the full course after reading the blog post.

**Why this priority**: Bridges blog content to course modules, increasing engagement. Nice-to-have for initial release since users can manually navigate to modules via the main nav.

**Independent Test**: Read a blog post tagged with a module (e.g., "Isaac") and verify that there's a visible CTA link (e.g., "Learn more in Module 3: Isaac Brain") that takes the user to the corresponding module page.

**Acceptance Scenarios**:

1. **Given** a user is reading a blog post tagged with "ROS2", **When** they scroll to the bottom or sidebar, **Then** they see a CTA box with text like "Continue Learning: Module 1 - ROS 2 Fundamentals" and a clickable link
2. **Given** a blog post is tagged with multiple modules (e.g., "ROS2" and "Nav2"), **When** the user views the post, **Then** they see multiple CTA links for each related module
3. **Given** a user clicks on a module CTA link, **When** the link loads, **Then** they are taken to the module's landing page or first chapter

---

### Edge Cases

- **No blog posts exist yet**: Display a friendly message like "No posts yet. Check back soon for course updates!" instead of an empty page
- **Post with no tags**: Blog post without tags should still be browsable on the main blog list, but won't appear in any tag-filtered views
- **Very long blog post**: Ensure code blocks, images, and long-form text render correctly with proper scrolling and don't break the layout
- **Post with broken module link**: If a blog post references a module that doesn't exist or has been renamed, display a generic "Back to Modules" link instead of broken navigation
- **Multiple authors**: If a blog post has multiple authors (co-authored), display all author names clearly (e.g., "By Alice & Bob")
- **Future-dated posts**: Posts with publish dates in the future should not appear in the blog list until that date (draft/scheduled posts)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display blog posts in reverse chronological order (newest first) on the blog homepage
- **FR-002**: System MUST render individual blog post pages with full content, author name, publish date, and tags
- **FR-003**: System MUST support markdown formatting for blog post content (headings, lists, code blocks, links, images)
- **FR-004**: Each blog post MUST have a clear title, summary/excerpt (first 150-200 characters or manually defined), and publish date
- **FR-005**: System MUST display author metadata for each blog post (author name and optional avatar/bio)
- **FR-006**: Blog posts MUST support tags for categorization (e.g., "ROS2", "Gazebo", "Isaac", "VLA", "announcement", "tutorial")
- **FR-007**: System MUST provide a tag filter interface on the blog homepage to filter posts by one tag at a time
- **FR-008**: System MUST display all available tags on the blog homepage for users to click and filter
- **FR-009**: Each blog post page MUST include navigation links to previous/next posts or back to the blog list
- **FR-010**: Blog posts tagged with course modules MUST display call-to-action links to the corresponding module pages
- **FR-011**: System MUST support code syntax highlighting in blog posts for languages used in the course (Python, YAML, Bash, JavaScript, C++)
- **FR-012**: Blog homepage MUST display a brief excerpt or summary for each post (not the full content)
- **FR-013**: System MUST support adding images to blog posts with proper alt text for accessibility
- **FR-014**: Blog posts MUST have unique URLs (e.g., `/blog/2025-12-26-my-post-title` or `/blog/my-post-title`)
- **FR-015**: System MUST display a "Read More" or "Continue Reading" link on truncated post excerpts on the blog homepage
- **FR-016**: System MUST sort tags alphabetically or by frequency on the tag filter interface
- **FR-017**: Each blog post MUST support a table of contents for long-form posts with multiple headings
- **FR-018**: System MUST respect Docusaurus blog configuration (default pagination: 10 posts per page)
- **FR-019**: Blog posts MUST integrate with the site's purple+black theme and maintain visual consistency
- **FR-020**: System MUST support RSS feed generation for blog posts (Docusaurus default feature)

### Key Entities

- **Blog Post**: Represents a single published article with title, content (markdown), author, publish date, tags, excerpt, and optional featured image. Each post is a standalone file in the Docusaurus blog directory.
- **Tag**: Categorization label (e.g., "ROS2", "tutorial", "announcement") used to organize and filter blog posts. Tags are embedded in blog post frontmatter and indexed by Docusaurus.
- **Author**: Person who wrote the blog post, with name and optional bio/avatar. Authors are defined in Docusaurus blog configuration or per-post frontmatter.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can navigate to the blog page from the main navigation menu and see all published posts within 2 clicks
- **SC-002**: Blog posts render correctly in Docusaurus with proper markdown formatting, code syntax highlighting, and embedded images
- **SC-003**: Users can filter blog posts by tag and see only relevant posts within 1 click (e.g., clicking "ROS2" shows only ROS2-tagged posts)
- **SC-004**: 100% of blog posts tagged with a course module display a visible call-to-action link to the corresponding module page
- **SC-005**: Users can read a full blog post and navigate to the next/previous post or back to the blog list without using the browser back button
- **SC-006**: Blog page maintains the purple+black visual theme with consistent styling across all posts and the blog homepage
- **SC-007**: New blog posts can be added by creating a markdown file in the blog directory and appear immediately after build/deployment
- **SC-008**: Users can access an RSS feed for blog posts to follow updates via feed readers (Docusaurus auto-generates this)

## Scope & Constraints

### In Scope

- Chronological blog post list with excerpts, dates, authors, and tags
- Individual blog post pages with full content and markdown support
- Tag-based filtering for module-specific posts (ROS2, Gazebo, Isaac, VLA)
- Call-to-action links from blog posts to related course modules
- RSS feed generation (Docusaurus default)
- Integration with existing purple+black UI theme

### Out of Scope

- **Comments system**: No user comments or discussion threads on blog posts (future enhancement)
- **Search functionality**: No dedicated blog search (rely on site-wide Docusaurus search)
- **Social sharing buttons**: No built-in share-to-Twitter/LinkedIn buttons (future enhancement)
- **Multi-author collaboration**: No collaborative editing or approval workflow for blog posts (single-author or pre-approved content only)
- **Analytics dashboard**: No tracking of blog post views, engagement metrics, or popular posts (can be added via external analytics)
- **Email subscriptions**: No email newsletter signup or automated notifications for new posts (future enhancement)
- **Draft preview mode**: No public preview of unpublished drafts (use local development server for previews)
- **Versioning**: No version history or revision tracking for blog posts (rely on git history)

### Dependencies

- **Docusaurus 3.9.2**: Blog feature relies on Docusaurus classic preset and blog plugin
- **Existing UI theme**: Blog pages must integrate with the purple+black theme implemented in feature 005-purple-ui-homepage
- **Module pages**: Call-to-action links depend on existing module pages (Module 1-4) being available

### Assumptions

- Blog posts are written in markdown format and stored in the `frontend-book/blog/` directory
- Authors are defined in Docusaurus configuration or per-post frontmatter (default: single author for MVP)
- Tags are added manually to each blog post's frontmatter (no auto-tagging or AI suggestions)
- Blog posts are published immediately upon deployment (no scheduled publishing workflow for MVP)
- RSS feed is auto-generated by Docusaurus (no custom feed logic needed)
- Users are familiar with standard blog navigation patterns (chronological lists, tag filters, previous/next links)
- Code examples in blog posts use the same syntax highlighting theme (Dracula) as the rest of the site

## Technical Constraints

- Must use Docusaurus blog plugin (no custom blog implementation)
- Blog styling must extend the existing purple+black CSS custom properties from `custom.css`
- Tag filtering uses Docusaurus built-in tag pages (no custom filtering logic required)
- Blog posts must be markdown files with valid frontmatter (title, date, author, tags)
- URLs follow Docusaurus blog convention: `/blog/YYYY/MM/DD/post-title` or `/blog/post-title`

## Acceptance Criteria Summary

This feature is considered complete when:

1. Blog homepage displays all published posts in reverse chronological order with titles, excerpts, dates, and authors
2. Individual blog post pages render full content with markdown formatting and code syntax highlighting
3. Users can filter blog posts by clicking on module tags (ROS2, Gazebo, Isaac, VLA)
4. Blog posts tagged with course modules display visible CTA links to the corresponding module pages
5. Blog pages integrate seamlessly with the purple+black UI theme
6. RSS feed is accessible at `/blog/rss.xml` (Docusaurus default)
7. All blog functionality works in production build (`npm run build` succeeds without errors)
