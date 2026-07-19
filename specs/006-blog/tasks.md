# Tasks: Blog Page

**Branch**: `006-blog` | **Date**: 2025-12-26 | **Spec**: `/specs/006-blog/spec.md` | **Plan**: `/specs/006-blog/plan.md`

## Task Organization

Tasks are grouped by user story to enable independent delivery:
- **Phase 1-2**: Setup & Foundational (required for all user stories)
- **Phase 3**: User Story 1 (P1-MVP) - Browse Recent Blog Posts
- **Phase 4**: User Story 2 (P2) - Filter Posts by Module Tag
- **Phase 5**: User Story 3 (P3) - Navigate to Related Course Modules
- **Phase 6**: Polish & Validation

## Task Format

```
- [ ] T### [P?] [Story?] Description with file path
```

- **T###**: Unique task number (T001, T002, etc.)
- **[P?]**: Priority (P1=MVP, P2=High, P3=Medium, P4=Low)
- **[Story?]**: User story reference (US1, US2, US3) or SETUP/POLISH

---

## Phase 1: Setup & Research

### Research & Verification

- [X] T001 [P1] [SETUP] Verify Docusaurus blog plugin is included in `@docusaurus/preset-classic` dependency in `frontend-book/package.json`
- [X] T002 [P1] [SETUP] Create Phase 0 research document at `specs/006-blog/research.md` documenting blog plugin configuration options, frontmatter schema, tag system, RSS generation, and theme extension strategy
- [X] T003 [P1] [SETUP] Create Phase 1 data model document at `specs/006-blog/data-model.md` defining BlogPost, Author, Tag, and ModuleCTA entity schemas
- [X] T004 [P1] [SETUP] Create Phase 1 quickstart guide at `specs/006-blog/quickstart.md` with 5-step implementation instructions

### Blog Directory Setup

- [X] T005 [P1] [SETUP] Create blog directory at `frontend-book/blog/` if it doesn't exist
- [X] T006 [P1] [SETUP] Verify blog plugin configuration in `frontend-book/docusaurus.config.ts` under `presets` → `@docusaurus/preset-classic` → `blog`
- [X] T007 [P1] [SETUP] Configure blog plugin settings: `routeBasePath: 'blog'`, `showReadingTime: true`, `postsPerPage: 10`, `blogSidebarCount: 5` in `frontend-book/docusaurus.config.ts`

---

## Phase 2: Foundational Components

### Author Metadata

- [X] T008 [P1] [US1] Create author metadata file at `frontend-book/blog/authors.yml` with default author entry (name, title, url, image_url)
- [X] T009 [P1] [US1] Add author schema validation: name (required), title (optional), url (optional), image_url (optional) in `frontend-book/blog/authors.yml`
- [X] T010 [P1] [US1] Test author rendering by creating temporary test post referencing default author in `frontend-book/blog/test-author.md`

### Blog Post Template & First Post

- [X] T011 [P1] [US1] Create first blog post at `frontend-book/blog/2025-12-26-welcome.md` with complete frontmatter schema (title, date, authors, tags, description)
- [X] T012 [P1] [US1] Add markdown content to welcome post with headings, lists, code blocks, links, and images to test formatting in `frontend-book/blog/2025-12-26-welcome.md`
- [X] T013 [P1] [US1] Tag welcome post with `["Announcement"]` in frontmatter of `frontend-book/blog/2025-12-26-welcome.md`
- [ ] T014 [P1] [US1] Run local dev server (`npm run start`) and verify blog homepage displays at `/blog` with welcome post

---

## Phase 3: User Story 1 (P1-MVP) - Browse Recent Blog Posts

### Blog List Functionality

- [X] T015 [P1] [US1] Create second blog post at `frontend-book/blog/2025-12-27-module-1-announcement.md` with frontmatter (title: "Module 1: ROS 2 Fundamentals Released", date: 2025-12-27, tags: ["ROS2", "Announcement"])
- [X] T016 [P1] [US1] Add markdown content to Module 1 announcement post with code examples and links to Module 1 page in `frontend-book/blog/2025-12-27-module-1-announcement.md`
- [ ] T017 [P1] [US1] Verify blog homepage at `/blog` displays both posts in reverse chronological order (Module 1 announcement first, welcome post second)
- [ ] T018 [P1] [US1] Verify blog list displays post excerpts (first 150-200 characters or custom description from frontmatter) on blog homepage
- [ ] T019 [P1] [US1] Verify blog list displays author name and publish date for each post on blog homepage

### Individual Blog Post Pages

- [ ] T020 [P1] [US1] Click on welcome post title from blog homepage and verify full content loads at `/blog/2025/12/26/welcome`
- [ ] T021 [P1] [US1] Verify markdown formatting renders correctly: headings, lists, code syntax highlighting (Python, YAML), links, images in blog post page
- [ ] T022 [P1] [US1] Verify blog post metadata displays: author name, publish date, tags at top of post page
- [ ] T023 [P1] [US1] Verify previous/next post navigation links appear at bottom of blog post page

### Navigation Integration

- [X] T024 [P1] [US1] Add "Blog" navigation link to main navbar in `frontend-book/docusaurus.config.ts` under `themeConfig` → `navbar` → `items`
- [ ] T025 [P1] [US1] Verify navbar "Blog" link navigates to `/blog` from homepage (1 click total)
- [ ] T026 [P1] [US1] Verify "Back to Blog" or breadcrumb navigation available from individual post pages

---

## Phase 4: User Story 2 (P2) - Filter Posts by Module Tag

### Tag System Configuration

- [X] T027 [P2] [US2] Create third blog post at `frontend-book/blog/2025-12-28-gazebo-tutorial.md` with tags `["Gazebo", "Tutorial"]`
- [X] T028 [P2] [US2] Add markdown content to Gazebo tutorial post with code examples and screenshots in `frontend-book/blog/2025-12-28-gazebo-tutorial.md`
- [ ] T029 [P2] [US2] Verify Docusaurus auto-generates tag pages at `/blog/tags/ros-2`, `/blog/tags/gazebo`, `/blog/tags/announcement`, `/blog/tags/tutorial`

### Tag Filtering Functionality

- [ ] T030 [P2] [US2] Verify blog homepage displays clickable tag list or tag cloud with all available tags
- [ ] T031 [P2] [US2] Click on "ROS2" tag from blog homepage and verify navigation to `/blog/tags/ros-2` showing only ROS2-tagged posts
- [ ] T032 [P2] [US2] Verify tag page displays tag name and post count (e.g., "2 posts tagged with ROS2") at `/blog/tags/ros-2`
- [ ] T033 [P2] [US2] Click on "Gazebo" tag and verify navigation to `/blog/tags/gazebo` showing only Gazebo-tagged post
- [ ] T034 [P2] [US2] From tag page, click "All Posts" or blog homepage link and verify all posts displayed without filtering
- [X] T035 [P2] [US2] Create fourth blog post at `frontend-book/blog/2025-12-29-multi-tag-post.md` with multiple tags `["ROS2", "Gazebo", "Tutorial"]`
- [ ] T036 [P2] [US2] Verify multi-tag post appears in both `/blog/tags/ros-2` and `/blog/tags/gazebo` filtered views

### Tag Display & Styling

- [ ] T037 [P2] [US2] Verify tags display as clickable badges/chips on individual blog post pages
- [ ] T038 [P2] [US2] Verify tag list on blog homepage is sorted alphabetically or by frequency

---

## Phase 5: User Story 3 (P3) - Navigate to Related Course Modules

### ModuleCTA Component Creation

- [X] T039 [P3] [US3] Create ModuleCTA component directory at `frontend-book/src/components/ModuleCTA/`
- [X] T040 [P3] [US3] Create ModuleCTA component TypeScript file at `frontend-book/src/components/ModuleCTA/index.tsx` with props: moduleName, moduleNumber, moduleTitle, moduleUrl
- [X] T041 [P3] [US3] Implement module mapping logic in ModuleCTA component: map tags (ROS2, Gazebo, Isaac, VLA) to module numbers (1, 2, 3, 4) and URLs in `frontend-book/src/components/ModuleCTA/index.tsx`
- [X] T042 [P3] [US3] Create ModuleCTA styles file at `frontend-book/src/components/ModuleCTA/styles.module.css` with purple+black theme styling
- [X] T043 [P3] [US3] Design ModuleCTA layout: card with module icon, title "Continue Learning: Module X - [Name]", description, and "Go to Module" button in `frontend-book/src/components/ModuleCTA/index.tsx`

### ModuleCTA Integration

- [X] T044 [P3] [US3] Import ModuleCTA component in Module 1 announcement blog post at bottom of content using MDX syntax: `import ModuleCTA from '@site/src/components/ModuleCTA';` in `frontend-book/blog/2025-12-27-module-1-announcement.md`
- [X] T045 [P3] [US3] Add ModuleCTA component invocation in Module 1 announcement post: `<ModuleCTA moduleName="ROS2" moduleNumber={1} moduleTitle="ROS 2 Fundamentals" moduleUrl="/docs/module-1-ros2/intro" />` in `frontend-book/blog/2025-12-27-module-1-announcement.md`
- [ ] T046 [P3] [US3] Verify ModuleCTA component displays at bottom of Module 1 announcement post with correct styling and link
- [ ] T047 [P3] [US3] Click ModuleCTA "Go to Module" button and verify navigation to Module 1 landing page at `/docs/module-1-ros2/intro`

### Multi-Module CTA Support

- [X] T048 [P3] [US3] Add multiple ModuleCTA components to multi-tag post (ROS2 + Gazebo) in `frontend-book/blog/2025-12-29-multi-tag-post.md`
- [ ] T049 [P3] [US3] Verify both ModuleCTA components (ROS2 and Gazebo) display correctly at bottom of multi-tag post
- [ ] T050 [P3] [US3] Test ModuleCTA fallback behavior: if module URL is broken or missing, display generic "Back to Modules" link

---

## Phase 6: Theme Integration & Polish

### Purple+Black Theme Extension

- [X] T051 [P2] [POLISH] Add blog-specific CSS selectors to `frontend-book/src/css/custom.css`: `.blog-list__item`, `.blog-post-meta`, `.blog-tags`, `.blog-module-cta`
- [X] T052 [P2] [POLISH] Apply purple accent color (`--ifm-color-primary: #9333ea`) to blog links, tags, and CTA buttons in `frontend-book/src/css/custom.css`
- [X] T053 [P2] [POLISH] Apply black background (`--ifm-background-color: #000000`) to blog pages and maintain white text contrast in `frontend-book/src/css/custom.css`
- [ ] T054 [P2] [POLISH] Verify WCAG 2.1 AA contrast compliance for blog text (4.5:1 ratio) using browser DevTools color contrast checker
- [ ] T055 [P2] [POLISH] Test blog page styling on multiple viewport sizes: 375px (mobile), 768px (tablet), 1920px (desktop)

### Responsive Design

- [ ] T056 [P2] [POLISH] Verify blog list layout is responsive: single column on mobile, multi-column on desktop
- [ ] T057 [P2] [POLISH] Verify ModuleCTA component is responsive: full-width on mobile, max-width 600px on desktop
- [ ] T058 [P2] [POLISH] Verify tag list wraps correctly on small screens without horizontal overflow
- [ ] T059 [P2] [POLISH] Test blog post images are responsive with max-width 100% and proper aspect ratio

### Content Creation

- [ ] T060 [P3] [POLISH] Create fifth blog post at `frontend-book/blog/2026-01-05-isaac-sim-guide.md` with tags `["Isaac", "Tutorial"]`
- [ ] T061 [P3] [POLISH] Add markdown content to Isaac Sim guide with code examples and screenshots in `frontend-book/blog/2026-01-05-isaac-sim-guide.md`
- [ ] T062 [P3] [POLISH] Create sixth blog post at `frontend-book/blog/2026-01-10-vla-research.md` with tags `["VLA", "Tutorial"]`
- [ ] T063 [P3] [POLISH] Add markdown content to VLA research post with academic references and code in `frontend-book/blog/2026-01-10-vla-research.md`
- [ ] T064 [P4] [POLISH] Create blog images directory at `frontend-book/static/img/blog/` and add featured images for blog posts

---

## Phase 7: Validation & Testing

### RSS Feed Validation

- [ ] T065 [P2] [POLISH] Verify RSS feed is accessible at `/blog/rss.xml` in dev server
- [ ] T066 [P2] [POLISH] Verify RSS feed contains all blog posts with full content (not just excerpts)
- [ ] T067 [P2] [POLISH] Validate RSS feed XML structure using online RSS validator (https://validator.w3.org/feed/)
- [ ] T068 [P2] [POLISH] Test RSS feed in feed reader (e.g., Feedly, RSS reader extension) to confirm proper rendering

### Production Build & Deployment

- [X] T069 [P1] [POLISH] Run production build: `npm run build` in `frontend-book/` and verify zero build errors
- [ ] T070 [P1] [POLISH] Run local production server: `npm run serve` and verify blog pages load correctly
- [X] T071 [P1] [POLISH] Verify blog sitemap is generated at `/sitemap.xml` with blog post URLs
- [ ] T072 [P2] [POLISH] Check Lighthouse performance score: desktop 90+, mobile 70+ for blog homepage and individual post pages

### Manual QA - User Story 1 (P1-MVP)

- [ ] T073 [P1] [US1] Manual QA: Navigate from homepage → "Blog" navbar link → Blog homepage (verify 1 click)
- [ ] T074 [P1] [US1] Manual QA: Verify blog homepage displays all posts in reverse chronological order (newest first)
- [ ] T075 [P1] [US1] Manual QA: Click on a blog post title and verify full content loads with proper formatting
- [ ] T076 [P1] [US1] Manual QA: Verify author name and publish date display correctly on blog post page
- [ ] T077 [P1] [US1] Manual QA: Verify previous/next post navigation links work at bottom of post

### Manual QA - User Story 2 (P2)

- [ ] T078 [P2] [US2] Manual QA: Click on "ROS2" tag from blog homepage and verify filtering to `/blog/tags/ros-2`
- [ ] T079 [P2] [US2] Manual QA: Verify only ROS2-tagged posts appear on tag page
- [ ] T080 [P2] [US2] Manual QA: Click on "Gazebo" tag and verify filter updates to show only Gazebo posts
- [ ] T081 [P2] [US2] Manual QA: Click "All Posts" or blog homepage link and verify all posts displayed without filtering
- [ ] T082 [P2] [US2] Manual QA: Verify multi-tagged post appears in multiple tag filtered views

### Manual QA - User Story 3 (P3)

- [ ] T083 [P3] [US3] Manual QA: Read Module 1 announcement post and verify ModuleCTA component displays at bottom
- [ ] T084 [P3] [US3] Manual QA: Click ModuleCTA "Go to Module" button and verify navigation to Module 1 page
- [ ] T085 [P3] [US3] Manual QA: Read multi-tag post and verify multiple ModuleCTA components display (ROS2 + Gazebo)
- [ ] T086 [P3] [US3] Manual QA: Verify ModuleCTA component styling matches purple+black theme

### Cross-Cutting QA

- [ ] T087 [P2] [POLISH] Manual QA: Verify purple+black theme consistency across blog homepage, post pages, and tag pages
- [ ] T088 [P2] [POLISH] Manual QA: Test responsive design on mobile (375px), tablet (768px), and desktop (1920px)
- [ ] T089 [P2] [POLISH] Manual QA: Test keyboard navigation (Tab, Enter, Escape) on blog pages for accessibility
- [ ] T090 [P2] [POLISH] Manual QA: Test screen reader compatibility (NVDA or VoiceOver) for blog post content and navigation

---

## Success Criteria Validation

### SC-001: Blog accessible within 2 clicks
- **Test**: T073, T025 - Navigate from homepage → "Blog" link (1 click total) ✅

### SC-002: Blog posts render correctly
- **Test**: T021, T075 - Verify markdown, code syntax highlighting, images display ✅

### SC-003: Tag filtering within 1 click
- **Test**: T031, T078 - Click tag → Filtered page loads (1 click) ✅

### SC-004: 100% of tagged posts show module CTAs
- **Test**: T046, T083, T084 - Verify ModuleCTA component on module-tagged posts ✅

### SC-005: Previous/next navigation available
- **Test**: T023, T077 - Check bottom of blog post for navigation links ✅

### SC-006: Purple+black theme maintained
- **Test**: T087 - Visual inspection of blog pages match site theme ✅

### SC-007: New posts appear after deployment
- **Test**: T069, T070 - Add new markdown file, rebuild, verify post appears ✅

### SC-008: RSS feed accessible
- **Test**: T065, T066 - Visit `/blog/rss.xml` and verify XML output ✅

---

## Dependency Graph

### Sequential Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3/4/5 (User Stories) → Phase 6/7 (Polish/Validation)

T001-T007 → T008-T014 → T015-T026 (US1)
                      → T027-T038 (US2)
                      → T039-T050 (US3)
                      → T051-T064 (Polish)
                      → T065-T090 (Validation)
```

### Parallel Execution Examples

**After T014 (foundational setup complete), these can run in parallel:**
- User Story 1 tasks (T015-T026)
- User Story 2 tasks (T027-T038)
- User Story 3 tasks (T039-T050)
- Theme tasks (T051-T059)

**After Phase 3-5 complete, these can run in parallel:**
- Content creation (T060-T064)
- RSS validation (T065-T068)
- Production build (T069-T072)
- Manual QA (T073-T090)

---

## Task Summary

- **Total Tasks**: 90
- **P1 (MVP)**: 23 tasks (Setup + US1 + Build)
- **P2 (High)**: 30 tasks (US2 + Theme + Validation)
- **P3 (Medium)**: 33 tasks (US3 + Content + QA)
- **P4 (Low)**: 4 tasks (Optional polish)

**Estimated Completion Time**: 1-2 days for P1-P2 tasks (MVP + high priority features)

---

## Notes

- All tasks follow the format: `- [ ] T### [P?] [Story?] Description with file path`
- Tasks are organized by user story for independent delivery
- Foundational tasks (T001-T014) must complete before user story tasks
- User story tasks (US1, US2, US3) can be executed independently after foundational setup
- Validation tasks (T065-T090) should run after all implementation tasks complete
- Manual QA tasks reference specific user story acceptance criteria from spec.md
