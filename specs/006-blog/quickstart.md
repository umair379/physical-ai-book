# Phase 1 Quickstart: Blog Page Implementation Guide

**Feature**: Blog Page | **Branch**: `006-blog` | **Date**: 2025-12-26

## 5-Step Blog Setup Guide

This quickstart guide provides step-by-step instructions for implementing the blog functionality in the Physical AI & Humanoid Robotics course documentation site.

---

## Step 1: Enable Docusaurus Blog Plugin

### Verify Blog Plugin Configuration

The blog plugin is already included in `@docusaurus/preset-classic` (version 3.9.2). Verify the configuration in `frontend-book/docusaurus.config.ts`:

```typescript
// docusaurus.config.ts
export default {
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          // ... docs config
        },
        blog: {
          routeBasePath: 'blog',           // Blog URL: /blog
          path: 'blog',                    // Filesystem path: ./blog
          showReadingTime: true,           // Show reading time estimate
          postsPerPage: 10,                // Pagination: 10 posts per page
          blogSidebarCount: 5,             // Recent posts in sidebar
          blogSidebarTitle: 'Recent posts',
          feedOptions: {
            type: 'all',                   // Generate RSS, Atom, JSON feeds
            copyright: `Copyright © ${new Date().getFullYear()} Physical AI Course`,
          },
        },
      },
    ],
  ],
};
```

### Add Blog to Navigation

Add a "Blog" link to the main navbar in `frontend-book/docusaurus.config.ts`:

```typescript
// docusaurus.config.ts
export default {
  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Modules',
        },
        {
          to: '/blog',              // Add this blog link
          label: 'Blog',
          position: 'left',
        },
        {
          href: 'https://github.com/physical-ai-course',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
  },
};
```

### Verify Blog Directory

Ensure the `frontend-book/blog/` directory exists:

```bash
# Create if missing
mkdir -p frontend-book/blog
```

---

## Step 2: Create Author Metadata

### Create `blog/authors.yml`

Create `frontend-book/blog/authors.yml` with author information:

```yaml
default:
  name: Physical AI Course Team
  title: Instructors
  url: https://github.com/physical-ai-course
  image_url: /img/authors/team.jpg
```

### Add Author Avatar (Optional)

1. Create directory: `frontend-book/static/img/authors/`
2. Add author avatar: `frontend-book/static/img/authors/team.jpg`

**Avatar Specifications**:
- **Format**: JPEG or PNG
- **Size**: 200x200 pixels (square)
- **Max File Size**: 50 KB

---

## Step 3: Create First Blog Post

### Create Welcome Post

Create `frontend-book/blog/2025-12-26-welcome.md`:

```markdown
---
title: "Welcome to the Physical AI Blog"
date: 2025-12-26
authors: [default]
tags: [Announcement]
description: "Introducing our new blog for course updates, technical tutorials, and learning insights."
---

Welcome to the **Physical AI & Humanoid Robotics** blog!

## What You'll Find Here

- 📢 **Course Announcements**: New module releases and updates
- 📚 **Technical Tutorials**: Deep-dives into ROS 2, Gazebo, Isaac Sim, and VLA
- 🧪 **Project Showcases**: Student projects and experiments
- 💡 **Learning Insights**: Tips and best practices

## Stay Updated

Subscribe to our [RSS feed](/blog/rss.xml) to get notified of new posts.

## Explore Course Modules

Browse our [course modules](/docs/intro) to start learning about physical AI and humanoid robotics.
```

### File Naming Convention

Blog post filenames must follow this format:

```
YYYY-MM-DD-post-title.md
```

**Examples**:
- ✅ `2025-12-26-welcome.md`
- ✅ `2025-12-27-module-1-announcement.md`
- ❌ `welcome.md` (missing date)
- ❌ `12-26-2025-welcome.md` (wrong date format)

### Test Blog Locally

Run the development server:

```bash
cd frontend-book
npm run start
```

Navigate to `http://localhost:3000/blog` and verify:
- Blog homepage displays
- Welcome post appears
- Post metadata (author, date) displays correctly
- Clicking post title loads full content

---

## Step 4: Create ModuleCTA Component

### Create Component Directory

```bash
mkdir -p frontend-book/src/components/ModuleCTA
```

### Create `index.tsx`

Create `frontend-book/src/components/ModuleCTA/index.tsx`:

```typescript
import React from 'react';
import styles from './styles.module.css';

interface ModuleCTAProps {
  moduleName: 'ROS2' | 'Gazebo' | 'Isaac' | 'VLA';
  moduleNumber: 1 | 2 | 3 | 4;
  moduleTitle: string;
  moduleUrl: string;
}

const MODULE_ICONS = {
  ROS2: '🤖',
  Gazebo: '🏗️',
  Isaac: '🧠',
  VLA: '👁️',
};

export default function ModuleCTA({
  moduleName,
  moduleNumber,
  moduleTitle,
  moduleUrl,
}: ModuleCTAProps): JSX.Element {
  return (
    <div className={styles.moduleCTA}>
      <div className={styles.ctaHeader}>
        <span className={styles.moduleIcon}>{MODULE_ICONS[moduleName]}</span>
        <span className={styles.moduleNumber}>Module {moduleNumber}</span>
      </div>
      <h3 className={styles.ctaTitle}>Continue Learning: {moduleTitle}</h3>
      <p className={styles.ctaDescription}>
        Explore the full module to dive deeper into {moduleName} concepts and hands-on exercises.
      </p>
      <a href={moduleUrl} className={styles.ctaButton}>
        Go to Module →
      </a>
    </div>
  );
}
```

### Create `styles.module.css`

Create `frontend-book/src/components/ModuleCTA/styles.module.css`:

```css
.moduleCTA {
  margin: 2rem 0;
  padding: 1.5rem;
  border: 2px solid var(--ifm-color-primary);
  border-radius: 8px;
  background: linear-gradient(135deg, #000000, #1a0033);
  max-width: 600px;
}

.ctaHeader {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.moduleIcon {
  font-size: 2rem;
}

.moduleNumber {
  font-size: 0.875rem;
  color: var(--ifm-color-primary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.ctaTitle {
  font-size: 1.5rem;
  color: #ffffff;
  margin-bottom: 0.75rem;
}

.ctaDescription {
  color: #cccccc;
  margin-bottom: 1.25rem;
  line-height: 1.6;
}

.ctaButton {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  background-color: var(--ifm-color-primary);
  color: #ffffff;
  text-decoration: none;
  border-radius: 6px;
  font-weight: 600;
  transition: background-color 0.3s ease;
}

.ctaButton:hover {
  background-color: #7c3aed;
  color: #ffffff;
  text-decoration: none;
}

/* Responsive */
@media (max-width: 768px) {
  .moduleCTA {
    max-width: 100%;
  }

  .ctaTitle {
    font-size: 1.25rem;
  }
}
```

### Use ModuleCTA in Blog Post

Update a blog post to include the ModuleCTA component:

```markdown
---
title: "Module 1: ROS 2 Fundamentals Released"
date: 2025-12-27
authors: [default]
tags: [ROS2, Announcement]
---

We're excited to announce Module 1!

## What You'll Learn

- ROS 2 core concepts
- Creating nodes and topics
- Pub/sub communication

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="ROS2"
  moduleNumber={1}
  moduleTitle="ROS 2 Fundamentals"
  moduleUrl="/docs/module-1-ros2/intro"
/>
```

---

## Step 5: Extend Purple+Black Theme to Blog Pages

### Update `custom.css`

Add blog-specific styles to `frontend-book/src/css/custom.css`:

```css
/* ===== Blog Styles ===== */

/* Blog List Items */
.blog-list__item {
  background-color: var(--ifm-background-color);
  border: 1px solid rgba(147, 51, 234, 0.3);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  transition: border-color 0.3s ease;
}

.blog-list__item:hover {
  border-color: var(--ifm-color-primary);
}

/* Blog Post Metadata */
.blog-post-meta {
  color: var(--ifm-color-primary);
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

/* Blog Tags */
.blog-tags {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.blog-tags a {
  background-color: var(--ifm-color-primary);
  color: #ffffff;
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-size: 0.875rem;
  text-decoration: none;
  transition: background-color 0.3s ease;
}

.blog-tags a:hover {
  background-color: #7c3aed;
}

/* Blog Post Title */
article h1 {
  color: #ffffff;
  margin-bottom: 1rem;
}

/* Blog Sidebar */
.blog-sidebar {
  border-left: 2px solid var(--ifm-color-primary);
  padding-left: 1rem;
}

/* Blog Archive */
.blog-archive {
  color: var(--ifm-color-primary);
}

/* Code Blocks in Blog */
article pre {
  background-color: #1a1a1a;
  border: 1px solid rgba(147, 51, 234, 0.3);
}
```

### Verify Theme Consistency

1. Run dev server: `npm run start`
2. Navigate to `/blog`
3. Verify:
   - Purple accent color on links and tags
   - Black background on blog pages
   - White text with sufficient contrast (4.5:1 ratio)
   - Code blocks use purple borders

---

## Verification Checklist

After completing Steps 1-5, verify the following:

- [ ] Blog homepage accessible at `/blog`
- [ ] "Blog" link in main navbar
- [ ] Welcome post displays with author and date
- [ ] Tags are clickable and lead to tag pages
- [ ] ModuleCTA component displays correctly
- [ ] Purple+black theme applied to blog pages
- [ ] RSS feed accessible at `/blog/rss.xml`
- [ ] Production build succeeds: `npm run build`

---

## Next Steps

### Create Additional Blog Posts

Follow the template structure to create more posts:

```markdown
---
title: "Your Post Title"
date: YYYY-MM-DD
authors: [default]
tags: [Tag1, Tag2]
description: "Brief description for SEO and excerpts"
---

Your markdown content here...

import ModuleCTA from '@site/src/components/ModuleCTA';

<ModuleCTA
  moduleName="ROS2"
  moduleNumber={1}
  moduleTitle="ROS 2 Fundamentals"
  moduleUrl="/docs/module-1-ros2/intro"
/>
```

### Recommended Tags

Use these tags for consistency:
- **Modules**: `ROS2`, `Gazebo`, `Isaac`, `VLA`
- **Content Type**: `Tutorial`, `Announcement`, `Project`, `News`

### Production Deployment

Build and deploy:

```bash
cd frontend-book
npm run build
npm run serve  # Test production build locally
```

---

## Troubleshooting

### Blog Page Not Loading

**Issue**: `/blog` returns 404

**Solution**:
1. Verify blog directory exists: `frontend-book/blog/`
2. Check `docusaurus.config.ts` has `blog` config
3. Restart dev server

### Author Not Rendering

**Issue**: Author shows as "Unknown"

**Solution**:
1. Verify `blog/authors.yml` exists
2. Check author key matches frontmatter: `authors: [default]`
3. Ensure YAML syntax is correct (proper indentation)

### Tags Not Generating Pages

**Issue**: Tag links return 404

**Solution**:
1. Verify tags are in array format: `tags: [ROS2, Tutorial]`
2. Check tag capitalization is consistent
3. Rebuild site: `npm run build`

### ModuleCTA Component Not Rendering

**Issue**: Component doesn't display in blog post

**Solution**:
1. Verify import statement: `import ModuleCTA from '@site/src/components/ModuleCTA';`
2. Check component props match interface
3. Ensure MDX is enabled (default in Docusaurus 3.9.2)

---

## Summary

You've successfully set up the blog functionality! The blog now supports:

✅ Chronological post listing at `/blog`
✅ Author metadata from `authors.yml`
✅ Tag-based filtering with auto-generated tag pages
✅ ModuleCTA component for linking to course modules
✅ Purple+black theme integration
✅ RSS feed generation at `/blog/rss.xml`

**Total Implementation Time**: ~1-2 hours for basic setup + sample posts
