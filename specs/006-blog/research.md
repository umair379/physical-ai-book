# Phase 0 Research: Blog Page

**Feature**: Blog Page | **Branch**: `006-blog` | **Date**: 2025-12-26

## Research Summary

This document captures technical research on Docusaurus 3.9.2 blog plugin capabilities, configuration options, and implementation decisions for the Physical AI & Humanoid Robotics course blog.

## Docusaurus Blog Plugin Overview

### Plugin Inclusion
- **Confirmed**: `@docusaurus/preset-classic` version 3.9.2 includes the blog plugin by default
- **Location**: Blog plugin is auto-configured in Docusaurus presets
- **No additional installation required**: Blog functionality is available out-of-the-box

### Blog Plugin Configuration Options

The blog plugin can be configured in `docusaurus.config.ts` under the `blog` key within the classic preset:

```typescript
presets: [
  [
    '@docusaurus/preset-classic',
    {
      blog: {
        routeBasePath: 'blog',           // URL path: /blog
        path: 'blog',                    // Filesystem path: ./blog
        showReadingTime: true,           // Show estimated reading time
        postsPerPage: 10,                // Pagination: 10 posts per page
        blogSidebarCount: 5,             // Number of posts in sidebar
        blogSidebarTitle: 'Recent posts', // Sidebar title
        feedOptions: {
          type: 'all',                   // RSS, Atom, JSON feeds
          copyright: `Copyright © ${new Date().getFullYear()}`,
        },
      },
    },
  ],
]
```

### Default Behaviors (No Configuration Needed)
- **RSS Feed**: Auto-generated at `/blog/rss.xml`
- **Atom Feed**: Auto-generated at `/blog/atom.xml`
- **JSON Feed**: Auto-generated at `/blog/feed.json`
- **Tag Pages**: Auto-generated at `/blog/tags/<tag-slug>`
- **Archive Page**: Auto-generated at `/blog/archive`
- **Pagination**: Automatic with next/previous navigation

## Frontmatter Schema

### Required Fields
```yaml
---
title: "Post Title"        # Required: Post headline
date: YYYY-MM-DD          # Required: Publish date (ISO 8601)
---
```

### Optional Fields
```yaml
---
authors: [key1, key2]     # Optional: Author key(s) from authors.yml
tags: [Tag1, Tag2]        # Optional: Categorization tags
description: "..."        # Optional: Custom excerpt (150-200 chars)
image: /img/blog/post.jpg # Optional: Featured image
slug: custom-url          # Optional: Override default URL slug
---
```

### Frontmatter Examples

**Minimal Post**:
```yaml
---
title: "Welcome to Our Blog"
date: 2025-12-26
---
```

**Full Post**:
```yaml
---
title: "Module 1: ROS 2 Fundamentals Released"
date: 2025-12-27
authors: [default]
tags: [ROS2, Announcement]
description: "Announcing the release of Module 1 covering ROS 2 core concepts, nodes, and communication."
image: /img/blog/module-1-release.jpg
slug: module-1-ros2-release
---
```

## Tag System Behavior

### Auto-Generated Tag Pages
- **Tag Detection**: Docusaurus scans all blog posts and extracts unique tags from frontmatter
- **Tag Page URLs**: `/blog/tags/<tag-slug>` (e.g., `/blog/tags/ros-2`)
- **Tag Slug Conversion**: Display names converted to URL-safe slugs
  - "ROS2" → `/blog/tags/ros-2`
  - "Gazebo" → `/blog/tags/gazebo`
  - "Isaac Sim" → `/blog/tags/isaac-sim`
  - "VLA" → `/blog/tags/vla`

### Tag Filtering Behavior
- **Single-Tag Filtering**: Tag pages show all posts with that specific tag
- **Multi-Tag Support**: Posts can have multiple tags and appear on all corresponding tag pages
- **Tag Count**: Tag pages display post count (e.g., "3 posts tagged with ROS2")
- **No Manual Configuration**: Tags are extracted automatically from post frontmatter

### Recommended Tag Naming Convention
- Use **display names** in frontmatter: `tags: [ROS2, Gazebo, Isaac, VLA, Tutorial, Announcement]`
- Docusaurus auto-converts to URL-safe slugs
- Maintain consistent capitalization across all posts (case-sensitive)

## RSS Feed Generation

### Auto-Generated Feeds
- **RSS 2.0**: `/blog/rss.xml` (default, full content)
- **Atom**: `/blog/atom.xml` (full content)
- **JSON Feed**: `/blog/feed.json` (full content)

### Feed Configuration (Optional)
```typescript
feedOptions: {
  type: 'all',                    // Generate all feed types
  title: 'Physical AI Blog',      // Feed title
  description: 'Course updates and technical tutorials',
  copyright: `Copyright © ${new Date().getFullYear()}`,
  language: 'en',
}
```

### Feed Content
- **Default Behavior**: Feeds include full blog post content (not just excerpts)
- **No Custom Logic Required**: Docusaurus handles feed generation automatically on build

## URL Structure

### Default URL Format
- **Date-Based**: `/blog/YYYY/MM/DD/post-title` (e.g., `/blog/2025/12/26/welcome`)
- **Derived From**: Filename convention `YYYY-MM-DD-post-title.md`

### Custom URL Slugs
- **Override Default**: Use `slug` frontmatter field
- **Example**: `slug: getting-started` → `/blog/getting-started`
- **Use Case**: Evergreen content that shouldn't appear dated

### Blog Homepage
- **URL**: `/blog` (configurable via `routeBasePath`)
- **Content**: Reverse chronological list of posts with excerpts
- **Pagination**: Automatic with configurable `postsPerPage`

## Author Metadata Storage

### Centralized Authors File
- **Location**: `blog/authors.yml` (standard Docusaurus convention)
- **Format**: YAML with author keys and metadata

```yaml
default:
  name: Physical AI Course Team
  title: Instructors
  url: https://github.com/physical-ai-course
  image_url: /img/authors/team.jpg

alice:
  name: Alice Johnson
  title: ROS 2 Expert
  url: https://github.com/alice
  image_url: /img/authors/alice.jpg
```

### Referencing Authors in Posts
```yaml
---
authors: [default]           # Single author
# OR
authors: [alice, bob]        # Multiple authors
---
```

### Author Schema
- **name** (required): Full name
- **title** (optional): Role or title
- **url** (optional): Personal website or GitHub
- **image_url** (optional): Avatar image path

## ModuleCTA Component Design

### Component Approach
- **Type**: React component with TypeScript
- **Location**: `src/components/ModuleCTA/index.tsx`
- **Styling**: CSS Modules (`styles.module.css`)
- **Integration**: Import and use in blog post markdown via MDX

### Component Props
```typescript
interface ModuleCTAProps {
  moduleName: 'ROS2' | 'Gazebo' | 'Isaac' | 'VLA';
  moduleNumber: 1 | 2 | 3 | 4;
  moduleTitle: string;
  moduleUrl: string;
}
```

### Usage in Blog Posts
```markdown
---
title: "ROS 2 Tutorial"
tags: [ROS2, Tutorial]
---

Blog post content here...

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="ROS2"
  moduleNumber={1}
  moduleTitle="ROS 2 Fundamentals"
  moduleUrl="/docs/module-1-ros2/intro"
/>
```

### Module Mapping
```typescript
const MODULE_MAP = {
  ROS2: { number: 1, title: 'ROS 2 Fundamentals', url: '/docs/module-1-ros2/intro' },
  Gazebo: { number: 2, title: 'Gazebo Simulation', url: '/docs/module-2-gazebo/intro' },
  Isaac: { number: 3, title: 'Isaac Sim & Brain', url: '/docs/module-3-isaac/intro' },
  VLA: { number: 4, title: 'Vision-Language-Action Models', url: '/docs/module-4-vla/intro' },
};
```

## Purple+Black Theme Extension Strategy

### Existing Theme
- **Location**: `src/css/custom.css`
- **Colors**:
  - Primary (purple): `#9333ea`
  - Background (black): `#000000`
  - Text (white): `#ffffff`

### Blog-Specific CSS Selectors
```css
/* Blog List Items */
.blog-list__item {
  background-color: var(--ifm-background-color);
  border: 1px solid var(--ifm-color-primary);
}

/* Blog Post Metadata */
.blog-post-meta {
  color: var(--ifm-color-primary);
}

/* Blog Tags */
.blog-tags a {
  background-color: var(--ifm-color-primary);
  color: #ffffff;
}

/* ModuleCTA Component */
.blog-module-cta {
  border: 2px solid var(--ifm-color-primary);
  background: linear-gradient(135deg, #000000, #1a0033);
}
```

### CSS Extension Approach
- **Method**: Extend `src/css/custom.css` (no swizzling)
- **Scope**: ~30 lines of blog-specific CSS
- **Inheritance**: Blog pages inherit global CSS variables
- **WCAG Compliance**: Maintain 4.5:1 contrast ratio for text

## Technical Decisions Summary

### ✅ Decision 1: Use Docusaurus Blog Plugin
- **Rationale**: 80% complexity reduction, auto-generates RSS/tags/pagination
- **Trade-off**: Less customization flexibility
- **Impact**: Fast implementation (1-2 days)

### ✅ Decision 2: Display Names for Tags
- **Rationale**: User-friendly, matches module naming
- **Format**: `tags: [ROS2, Gazebo, Isaac, VLA]`
- **Auto-Conversion**: Docusaurus handles URL-safe slugs

### ✅ Decision 3: Automatic ModuleCTA at Bottom
- **Rationale**: Consistent placement, no manual work per post
- **Integration**: MDX import in blog posts
- **Mapping**: Component reads module URLs from internal config

### ✅ Decision 4: Date-Based URLs
- **Format**: `/blog/YYYY/MM/DD/post-title`
- **Override**: Custom slugs supported via frontmatter
- **SEO**: Provides chronological context

### ✅ Decision 5: Centralized Authors File
- **Location**: `blog/authors.yml`
- **Benefit**: Single source of truth, easy updates
- **Reference**: Author keys in post frontmatter

### ✅ Decision 6: Extend Existing CSS
- **Method**: Add blog selectors to `custom.css`
- **Benefit**: No component swizzling, easier upgrades
- **Scope**: ~30 lines of CSS

## Implementation Constraints

1. **No Swizzling**: Avoid swizzling Docusaurus blog components to maintain upgrade path
2. **WCAG AA Compliance**: All blog pages must meet 4.5:1 contrast ratio
3. **Mobile-First**: Blog layout must be responsive (375px, 768px, 1920px)
4. **Build Performance**: Blog pages must not degrade Lighthouse scores (90+ desktop, 70+ mobile)
5. **Markdown Compatibility**: All standard markdown and MDX features must work

## Testing Approach

### Manual QA Tests
1. Navigate to `/blog` and verify post list
2. Click tags and verify filtering
3. Check RSS feed at `/blog/rss.xml`
4. Test responsive design on mobile/tablet/desktop
5. Verify ModuleCTA component displays and links work
6. Run production build and check for errors

### Success Validation
- All 8 success criteria (SC-001 to SC-008) validated manually
- Production build succeeds with zero errors
- Lighthouse scores: 90+ desktop, 70+ mobile

## References

- [Docusaurus Blog Plugin Docs](https://docusaurus.io/docs/blog)
- [Frontmatter Reference](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-content-blog#markdown-front-matter)
- [MDX Components](https://docusaurus.io/docs/markdown-features/react)
- [Theme Configuration](https://docusaurus.io/docs/styling-layout)
