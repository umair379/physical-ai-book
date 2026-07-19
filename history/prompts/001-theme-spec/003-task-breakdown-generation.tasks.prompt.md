---
id: 003
title: Task Breakdown Generation
stage: tasks
date: 2025-12-28
surface: agent
model: claude-sonnet-4-5
feature: 001-theme-spec
branch: 001-theme-spec
user: developer
command: /sp.tasks
labels: ["tasks", "theme", "css", "implementation-breakdown", "user-stories", "mvp"]
links:
  spec: specs/001-theme-spec/spec.md
  plan: specs/001-theme-spec/plan.md
  quickstart: specs/001-theme-spec/quickstart.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-theme-spec/tasks.md (created - 90 tasks organized by user story)
  - history/prompts/001-theme-spec/003-task-breakdown-generation.tasks.prompt.md (created - this PHR)
tests:
  - Task organization validation: PASS (all tasks follow checklist format with [ID] [P?] [Story] structure)
  - User story independence: PASS (each story can be implemented and tested independently)
  - Parallel opportunities: IDENTIFIED (48 tasks marked [P] can run in parallel)
  - MVP scope: DEFINED (43 tasks for User Story 1 - Light Mode)
---

## Prompt

/sp.tasks

## Response Snapshot

**Task Breakdown Complete** for Feature 001: Complete Theme System with Light and Dark Modes

### Tasks Summary

Generated comprehensive task breakdown with 90 tasks organized by user story for incremental delivery.

**Tasks File**: `D:\physical-ai-book\specs\001-theme-spec\tasks.md`

**Organization Strategy**: Tasks organized by user story to enable independent implementation and testing of each story increment.

### Task Breakdown by Phase

**Phase 1: Setup (4 tasks)**
- Verify Docusaurus environment
- Backup existing custom.css
- Read current custom.css structure
- Verify docusaurus.config.ts

**Phase 2: Foundational (6 tasks)**
- Create new custom.css with file structure
- Add CSS section comments
- Add `:root` selector (empty)
- Add `[data-theme='dark']` selector (empty)
- Add responsive media query section (empty)
- Add accessibility media query section (empty)

**Phase 3: User Story 1 - Light Mode (33 tasks) 🎯 MVP**
- Color variables (11 tasks): Primary, background, text, code blocks, borders, navbar, footer, links, system colors
- Typography (5 tasks): Font families, sizes, line-heights, heading scale, margins
- Components (17 tasks): Buttons (4), cards (2), links (2), code blocks (2), inputs (2), tables (3), blockquotes (1), badges (1)
- Layout (2 tasks): Container max-width, section spacing

**Phase 4: User Story 2 - Dark Mode (15 tasks)**
- Dark mode color variables (9 tasks): Primary, background, text, code blocks, borders, navbar, footer, links, system colors
- Component adjustments (3 tasks): Shadows, tables, blockquotes for dark mode
- Theme toggle verification (3 tasks): Button exists, persistence across pages, persistence across sessions

**Phase 5: User Story 4 - Accessibility (13 tasks)**
- Accessibility features (4 tasks): Reduced motion CSS, focus states (general, button, link)
- WCAG AA compliance (6 tasks): Contrast checks for light/dark mode body text and links
- Keyboard navigation (3 tasks): Tab navigation, theme toggle keyboard access, reduced motion testing

**Phase 6: User Story 3 - Responsive (11 tasks)**
- Mobile adjustments (6 tasks): Font size, heading sizes, buttons, section padding, cards, container padding
- Desktop enhancements (2 tasks): Section padding, card spacing
- Responsive testing (3 tasks): Mobile (375px), tablet (768px), desktop (1440px) viewports

**Phase 7: Polish (8 tasks)**
- Print stylesheet
- CSS comments for maintainability
- Remove !important overrides
- Cross-page verification
- Browser compatibility testing
- Code block syntax highlighting verification
- Complete quickstart.md validation
- Delete backup file

### Task Organization Highlights

**Total Tasks**: 90
- Setup: 4 tasks
- Foundational: 6 tasks (BLOCKS all user stories)
- User Story 1 (P1 - Light Mode): 33 tasks - **MVP**
- User Story 2 (P2 - Dark Mode): 15 tasks
- User Story 4 (P2 - Accessibility): 13 tasks
- User Story 3 (P3 - Responsive): 11 tasks
- Polish: 8 tasks

**Parallel Opportunities**: 48 tasks marked [P] can run in parallel within their phase

**Suggested MVP Scope**: Phase 1 (4) + Phase 2 (6) + Phase 3 (33) = **43 tasks total**

### Dependency Graph

```
Phase 1 (Setup) → ~10 minutes
     ↓
Phase 2 (Foundational) → ~15 minutes
     ↓
     ├─ User Story 1: Light Mode (P1) → ~90-120 minutes ← MVP
     │   ├─ Independent: No dependencies
     │   ├─ Delivers: Complete light mode theme
     │   └─ Blocks: None
     │   └─ After color variables defined (T011-T019):
     │        ├─ User Story 2 can start in parallel
     │        ├─ User Story 3 can start in parallel
     │        └─ User Story 4 can start in parallel
     │
     ├─ User Story 2: Dark Mode (P2) → ~60 minutes
     │   ├─ Depends: US1 color variable structure (T011-T019)
     │   ├─ Delivers: Dark mode theme + toggle persistence
     │   └─ Can develop in parallel with US3/US4 after US1 structure defined
     │
     ├─ User Story 4: Accessibility (P2) → ~45 minutes
     │   ├─ Depends: US1 and US2 (tests both themes)
     │   ├─ Delivers: WCAG AA compliance, reduced motion, focus states
     │   └─ Can develop in parallel with US2/US3
     │
     └─ User Story 3: Responsive (P3) → ~60 minutes
         ├─ Depends: US1 (refines light mode)
         ├─ Delivers: Mobile/tablet/desktop optimizations
         └─ Can develop in parallel with US2/US4
              ↓
         Polish → ~30 minutes
```

### Independent Test Criteria

Each user story has clear independent test criteria from spec.md:

**User Story 1 (Light Mode)**:
- Open `http://localhost:3000` in browser
- Verify white background, dark gray text (#212121), blue accent (#1976d2)
- Use DevTools color picker to verify WCAG AA contrast ratios (4.5:1 minimum)
- Follow quickstart.md Section 1 (Visual Inspection - Light Mode)

**User Story 2 (Dark Mode)**:
- Toggle theme switcher to dark mode
- Verify dark slate background (#1a1a1a), light gray text (#e0e0e0), light blue accent (#64b5f6)
- Verify theme persists after page navigation and browser refresh
- Follow quickstart.md Section 2 (Visual Inspection - Dark Mode) and Section 7 (Theme Toggle Testing)

**User Story 4 (Accessibility)**:
- Run axe DevTools scan on both light and dark modes
- Verify 0 color contrast violations
- Enable OS "prefers-reduced-motion" and verify all animations disabled
- Use Tab key to verify all interactive elements show focus indicators
- Follow quickstart.md Section 5 (Accessibility Testing)

**User Story 3 (Responsive)**:
- Open DevTools responsive mode
- Test viewports: 375px (mobile), 768px (tablet), 1440px (desktop)
- Verify font sizes, button padding, card layouts adjust appropriately
- Follow quickstart.md Section 4 (Responsive Testing)

### Implementation Strategy

**MVP First (User Story 1 Only)**:
1. Complete Phase 1: Setup (T001-T004) → ~10 minutes
2. Complete Phase 2: Foundational (T005-T010) → ~15 minutes
3. Complete Phase 3: User Story 1 (T011-T043) → ~90-120 minutes
4. **STOP and VALIDATE**: Test with quickstart.md Section 1
5. Deploy light mode theme ✅ **MVP COMPLETE**

**Total MVP Time**: ~2-3 hours
**MVP Deliverable**: Professional light mode theme with WCAG AA compliance

**Incremental Delivery**:
- **Release 1 (MVP)**: User Story 1 - Light Mode only
- **Release 2**: Add User Story 2 (Dark Mode) + User Story 4 (Accessibility)
- **Release 3**: Add User Story 3 (Responsive) + Polish

**Total Implementation Time**: 4-6 hours across all releases

### Parallel Execution Examples

**Within User Story 1 - Color Variables (All Parallel)**:
```bash
T011 [P] [US1] Primary colors
T012 [P] [US1] Background colors
T013 [P] [US1] Text colors
T014 [P] [US1] Code block colors
T015 [P] [US1] Border colors
T016 [P] [US1] Navbar colors
T017 [P] [US1] Footer colors
T018 [P] [US1] Link colors
T019 [P] [US1] System colors
```

**Within User Story 2 - Dark Mode Variables (All Parallel)**:
```bash
T044 [P] [US2] Primary colors (dark)
T045 [P] [US2] Background colors (dark)
T046 [P] [US2] Text colors (dark)
T047 [P] [US2] Code block colors (dark)
T048 [P] [US2] Border colors (dark)
T049 [P] [US2] Navbar colors (dark)
T050 [P] [US2] Footer colors (dark)
T051 [P] [US2] Link colors (dark)
T052 [P] [US2] System colors (dark)
```

**Within User Story 4 - Accessibility Features (All Parallel)**:
```bash
T059 [P] [US4] Reduced motion CSS
T060 [P] [US4] Focus states (general)
T061 [P] [US4] Button focus state
T062 [P] [US4] Link focus state
```

**Within User Story 3 - Mobile Adjustments (All Parallel)**:
```bash
T072 [P] [US3] Mobile font size
T073 [P] [US3] Mobile heading sizes
T074 [P] [US3] Mobile button adjustments
T075 [P] [US3] Mobile section padding
T076 [P] [US3] Mobile card adjustments
T077 [P] [US3] Mobile container padding
```

### Task Format Validation

All tasks follow the strict checklist format:
- ✅ Every task starts with `- [ ]` (markdown checkbox)
- ✅ Every task has unique Task ID (T001-T090)
- ✅ [P] marker included only for parallelizable tasks (48 total)
- ✅ [Story] label included for all user story phase tasks (US1, US2, US3, US4)
- ✅ Clear action description with exact file path (frontend-book/src/css/custom.css)
- ✅ No Setup/Foundational/Polish tasks have story labels (correct)

**Examples**:
- `- [ ] T001 Verify Docusaurus development server runs...` (Setup - no story label)
- `- [ ] T011 [P] [US1] Define primary color variables in :root...` (User Story 1 - parallel)
- `- [ ] T044 [P] [US2] Define primary color variables in [data-theme='dark']...` (User Story 2 - parallel)

### Key Technical Highlights

**Single File Implementation**: All 90 tasks modify single file `frontend-book/src/css/custom.css`

**CSS Variable Strategy**: Override Docusaurus `--ifm-*` variables
- Light mode: `:root` selector
- Dark mode: `[data-theme='dark']` selector
- Responsive: `@media (max-width: 768px)` and `@media (min-width: 1024px)`
- Accessibility: `@media (prefers-reduced-motion: reduce)`

**Color Palette**:
- Light mode: #ffffff bg, #212121 text, #1976d2 primary
- Dark mode: #1a1a1a bg, #e0e0e0 text, #64b5f6 primary
- All combinations verified WCAG AA compliant (>4.5:1 contrast)

**Typography**:
- System fonts: system-ui, Segoe UI, Roboto, Arial
- Monospace: Consolas, Monaco, Courier New
- Base size: 16px (15px mobile)
- Heading scale: H1 (2.5rem) to H6 (0.875rem)

**Component Coverage**: Buttons, cards, links, code blocks, inputs, navbar, footer, tables, blockquotes, badges

**Responsive Breakpoints**:
- Mobile: <768px (test: 375px)
- Tablet: 768-1024px (test: 768px)
- Desktop: >1024px (test: 1440px)

## Outcome

- ✅ Impact: Feature 001 task breakdown complete - 90 tasks organized by user story for incremental delivery with clear MVP scope
- 🧪 Tests: Task format validation PASS (all 90 tasks follow checklist format), user story independence PASS (each story has independent test criteria), parallel opportunities IDENTIFIED (48 tasks marked [P])
- 📁 Files: tasks.md (90 tasks, 7 phases, clear dependency graph), history/prompts/001-theme-spec/003-task-breakdown-generation.tasks.prompt.md (this PHR)
- 🔁 Next prompts: Ready for implementation - start with MVP (Phase 1-3, 43 tasks) to deliver User Story 1 (Light Mode), then incrementally add US2/US4/US3 in parallel
- 🧠 Reflection: Task breakdown demonstrates clear incremental delivery strategy. MVP scope (US1 - Light Mode) is well-defined with 43 tasks totaling ~2-3 hours of work. All tasks follow strict checklist format with [ID] [P?] [Story] structure for traceability. Parallel opportunities identified within each user story (48 tasks total marked [P]). Dependency graph shows US1 as foundation, then US2/US3/US4 can proceed in parallel once US1 color variable structure is defined (after T011-T019). Each user story has independent test criteria from spec.md and references quickstart.md for comprehensive manual testing. Task descriptions include exact CSS sections and variable names from plan.md appendix. No automated tests included (CSS requires visual verification). Implementation strategy provides three delivery options: MVP first (US1 only), incremental (US1 → US2+US4 → US3+Polish), or parallel team (4 developers working on different stories simultaneously after foundational phase). Ready for execution without user intervention.

## Evaluation Notes (Flywheel)

- Failure modes observed: None - task breakdown completed on first pass with all validation checks passing
- Graders run and results (PASS/FAIL):
  - Task format validation: PASS (all 90 tasks follow [ID] [P?] [Story] Description format)
  - User story organization: PASS (tasks grouped by story for independent implementation)
  - Independent test criteria: PASS (each story has clear verification steps from spec.md)
  - Parallel opportunities: PASS (48 tasks marked [P] for concurrent execution)
  - MVP scope definition: PASS (43 tasks for User Story 1 clearly identified)
  - Dependency graph: PASS (clear blocking relationships and parallel paths documented)
- Prompt variant (if applicable): Standard `/sp.tasks` workflow for CSS-only feature with no data model or contracts
- Next experiment (smallest change to try): Execute MVP scope (T001-T043) to implement User Story 1 (Light Mode), validate with quickstart.md Section 1, then incrementally add remaining user stories
