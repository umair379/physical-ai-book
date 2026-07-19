---
description: "Task breakdown for Complete Theme System with Light and Dark Modes implementation"
---

# Tasks: Complete Theme System with Light and Dark Modes

**Input**: Design documents from `/specs/001-theme-spec/`
**Prerequisites**: plan.md ✅, spec.md ✅, quickstart.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Target File**: `frontend-book/src/css/custom.css` (complete replacement of existing theme)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different sections, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths or CSS sections in descriptions

---

## Phase 1: Setup (Project Verification)

**Purpose**: Verify Docusaurus environment and existing theme structure

- [ ] T001 Verify Docusaurus development server runs with `cd frontend-book && npm start`
- [ ] T002 Backup existing custom.css to `frontend-book/src/css/custom.css.backup`
- [ ] T003 Read current `frontend-book/src/css/custom.css` to understand structure to replace
- [ ] T004 Verify `frontend-book/docusaurus.config.ts` has color mode configuration (no changes needed)

**Checkpoint**: Environment verified, ready for theme implementation

---

## Phase 2: Foundational (CSS Structure Setup)

**Purpose**: Create base CSS file structure with comments and organization

**⚠️ CRITICAL**: This phase establishes the CSS file structure that all user stories will build upon

- [ ] T005 Create new `frontend-book/src/css/custom.css` with file header comment explaining this is the complete theme replacement
- [ ] T006 Add CSS section comments for organization: Colors, Typography, Components, Responsive, Accessibility
- [ ] T007 Add `:root` selector for light mode variables (empty, ready for US1)
- [ ] T008 Add `[data-theme='dark']` selector for dark mode variables (empty, ready for US2)
- [ ] T009 Add `@media (max-width: 768px)` responsive section (empty, ready for US3)
- [ ] T010 Add `@media (prefers-reduced-motion: reduce)` accessibility section (empty, ready for US4)

**Checkpoint**: Foundation ready - CSS file structure established, user story implementation can begin

---

## Phase 3: User Story 1 - View Documentation in Light Mode (Priority: P1) 🎯 MVP

**Goal**: Implement complete light mode theme with professional colors, typography, and component styling

**Independent Test**: Open `http://localhost:3000` in browser and verify white background, dark gray text (#212121), blue accent (#1976d2), all components styled per spec. Use browser DevTools color picker to verify contrast ratios meet WCAG AA (4.5:1 minimum).

### Color Variables for Light Mode

- [ ] T011 [P] [US1] Define primary color variables in `:root`: --ifm-color-primary (#1976d2), --ifm-color-primary-dark (#1565c0), --ifm-color-primary-darker (#0d47a1), --ifm-color-primary-darkest (#0a3d91), --ifm-color-primary-light (#42a5f5), --ifm-color-primary-lighter (#64b5f6), --ifm-color-primary-lightest (#90caf9) in frontend-book/src/css/custom.css
- [ ] T012 [P] [US1] Define background color variables in `:root`: --ifm-background-color (#ffffff), --ifm-background-surface-color (#ffffff) in frontend-book/src/css/custom.css
- [ ] T013 [P] [US1] Define text color variables in `:root`: --ifm-font-color-base (#212121), --ifm-font-color-base-inverse (#ffffff), --ifm-font-color-secondary (#616161) in frontend-book/src/css/custom.css
- [ ] T014 [P] [US1] Define code block color variables in `:root`: --ifm-code-background (#f5f5f5), --ifm-pre-background (#f5f5f5), --ifm-pre-color (#212121) in frontend-book/src/css/custom.css
- [ ] T015 [P] [US1] Define border color variables in `:root`: --ifm-color-emphasis-300 (#e0e0e0), --ifm-hr-background-color (#e0e0e0) in frontend-book/src/css/custom.css
- [ ] T016 [P] [US1] Define navbar color variables in `:root`: --ifm-navbar-background-color (#ffffff), --ifm-navbar-shadow (0 1px 2px 0 rgba(0,0,0,0.05)) in frontend-book/src/css/custom.css
- [ ] T017 [P] [US1] Define footer color variables in `:root`: --ifm-footer-background-color (#fafafa), --ifm-footer-color (#616161), --ifm-footer-link-color (#757575) in frontend-book/src/css/custom.css
- [ ] T018 [P] [US1] Define link color variables in `:root`: --ifm-link-color (#1976d2), --ifm-link-hover-color (#1565c0) in frontend-book/src/css/custom.css
- [ ] T019 [P] [US1] Define system color variables in `:root`: --ifm-color-success (#4caf50), --ifm-color-info (#2196f3), --ifm-color-warning (#ff9800), --ifm-color-danger (#f44336) in frontend-book/src/css/custom.css

### Typography System

- [ ] T020 [P] [US1] Define font family variables in `:root`: --ifm-font-family-base (system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif), --ifm-font-family-monospace ("Consolas", "Monaco", "Courier New", monospace) in frontend-book/src/css/custom.css
- [ ] T021 [P] [US1] Define font size and line-height variables in `:root`: --ifm-font-size-base (16px), --ifm-line-height-base (1.6) in frontend-book/src/css/custom.css
- [ ] T022 [P] [US1] Define heading font weight in `:root`: --ifm-heading-font-weight (700) in frontend-book/src/css/custom.css
- [ ] T023 [US1] Add heading size overrides: h1 (2.5rem), h2 (2rem), h3 (1.5rem), h4 (1.25rem), h5 (1rem), h6 (0.875rem) with line-height 1.3 in frontend-book/src/css/custom.css
- [ ] T024 [US1] Add heading margins: h1 (0 0 1.5rem), h2 (2.5rem 0 1rem), h3 (2rem 0 0.75rem) in frontend-book/src/css/custom.css

### Component Styling

- [ ] T025 [US1] Style buttons: .button selector with 8px border-radius, 0.75rem vertical padding, 1.5rem horizontal padding, gradient background (linear-gradient primary to primary-dark), white text, font-weight 600 in frontend-book/src/css/custom.css
- [ ] T026 [US1] Add button hover state: .button:hover with elevated shadow (0 4px 12px rgba(0,0,0,0.15)), transform translateY(-2px), transition 200ms in frontend-book/src/css/custom.css
- [ ] T027 [US1] Add button active state: .button:active with reduced shadow, transform translateY(0) in frontend-book/src/css/custom.css
- [ ] T028 [US1] Add button disabled state: .button:disabled with opacity 0.5, cursor not-allowed, no hover effects in frontend-book/src/css/custom.css
- [ ] T029 [US1] Style cards: .card selector with 12px border-radius, 1px border (--ifm-color-emphasis-300), shadow (0 2px 8px rgba(0,0,0,0.08)), padding 1.5rem in frontend-book/src/css/custom.css
- [ ] T030 [US1] Add card hover effect: .card:hover with elevated shadow (0 8px 24px rgba(0,0,0,0.12)), transform translateY(-4px), transition 300ms in frontend-book/src/css/custom.css
- [ ] T031 [US1] Style links: a selector with --ifm-link-color, text-decoration none, transition color 200ms in frontend-book/src/css/custom.css
- [ ] T032 [US1] Add link hover state: a:hover with --ifm-link-hover-color, text-decoration underline in frontend-book/src/css/custom.css
- [ ] T033 [US1] Style code blocks: pre, code selectors with --ifm-code-background, 8px border-radius, 1.5rem padding, font-size 0.9em, shadow (0 1px 3px rgba(0,0,0,0.06)) in frontend-book/src/css/custom.css
- [ ] T034 [US1] Style inline code: code (not in pre) with --ifm-code-background, 4px border-radius, 0.25rem horizontal padding, 0.125rem vertical padding in frontend-book/src/css/custom.css
- [ ] T035 [US1] Style inputs: input, textarea, select selectors with 1px border (--ifm-color-emphasis-300), 4px border-radius, 0.5rem padding, transition border-color 200ms in frontend-book/src/css/custom.css
- [ ] T036 [US1] Add input focus state: input:focus, textarea:focus, select:focus with outline 2px solid --ifm-color-primary, outline-offset 2px in frontend-book/src/css/custom.css
- [ ] T037 [US1] Style tables: table selector with border-collapse collapse, 8px border-radius, overflow hidden, border 1px solid (--ifm-color-emphasis-300) in frontend-book/src/css/custom.css
- [ ] T038 [US1] Style table headers: th selector with background (--ifm-color-primary-lightest), padding 0.75rem, font-weight 600, text-align left in frontend-book/src/css/custom.css
- [ ] T039 [US1] Style table rows: tr:nth-child(even) with background rgba(0,0,0,0.02), td with padding 0.75rem, border-top 1px solid (--ifm-color-emphasis-300) in frontend-book/src/css/custom.css
- [ ] T040 [US1] Style blockquotes: blockquote selector with border-left 4px solid --ifm-color-primary, background rgba(25,118,210,0.05), padding 1rem 1.5rem, margin 1.5rem 0, font-style italic in frontend-book/src/css/custom.css
- [ ] T041 [US1] Style badges: .badge selector with 12px border-radius, padding 0.25rem 0.75rem, font-size 0.875rem, font-weight 600, colored backgrounds (success/info/warning/danger variants) in frontend-book/src/css/custom.css

### Layout and Spacing

- [ ] T042 [US1] Add container max-width: .container, .main-wrapper selectors with max-width 1200px, margin 0 auto, padding 0 1rem in frontend-book/src/css/custom.css
- [ ] T043 [US1] Add section spacing: section selector with padding 2rem 1rem (mobile), margin-bottom 2rem in frontend-book/src/css/custom.css

**Checkpoint**: Light mode theme complete - verify with quickstart.md Section 1 (Visual Inspection - Light Mode)

---

## Phase 4: User Story 2 - View Documentation in Dark Mode (Priority: P2)

**Goal**: Implement dark mode theme with adjusted colors for dark backgrounds, maintaining readability and WCAG AA compliance

**Independent Test**: Toggle theme switcher to dark mode and verify dark slate background (#1a1a1a), light gray text (#e0e0e0), light blue accent (#64b5f6). Verify theme persists after page navigation and browser refresh. Use DevTools to confirm contrast ratios meet WCAG AA.

### Dark Mode Color Variables

- [ ] T044 [P] [US2] Define primary color variables in `[data-theme='dark']`: --ifm-color-primary (#64b5f6), --ifm-color-primary-dark (#42a5f5), --ifm-color-primary-darker (#2196f3), --ifm-color-primary-darkest (#1976d2), --ifm-color-primary-light (#90caf9), --ifm-color-primary-lighter (#bbdefb), --ifm-color-primary-lightest (#e3f2fd) in frontend-book/src/css/custom.css
- [ ] T045 [P] [US2] Define background color variables in `[data-theme='dark']`: --ifm-background-color (#1a1a1a), --ifm-background-surface-color (#242424) in frontend-book/src/css/custom.css
- [ ] T046 [P] [US2] Define text color variables in `[data-theme='dark']`: --ifm-font-color-base (#e0e0e0), --ifm-font-color-base-inverse (#1a1a1a), --ifm-font-color-secondary (#bdbdbd) in frontend-book/src/css/custom.css
- [ ] T047 [P] [US2] Define code block color variables in `[data-theme='dark']`: --ifm-code-background (#2a2a2a), --ifm-pre-background (#2a2a2a), --ifm-pre-color (#e0e0e0) in frontend-book/src/css/custom.css
- [ ] T048 [P] [US2] Define border color variables in `[data-theme='dark']`: --ifm-color-emphasis-300 (#424242), --ifm-hr-background-color (#424242) in frontend-book/src/css/custom.css
- [ ] T049 [P] [US2] Define navbar color variables in `[data-theme='dark']`: --ifm-navbar-background-color (#242424), --ifm-navbar-shadow (0 1px 2px 0 rgba(0,0,0,0.3)) in frontend-book/src/css/custom.css
- [ ] T050 [P] [US2] Define footer color variables in `[data-theme='dark']`: --ifm-footer-background-color (#1a1a1a), --ifm-footer-color (#bdbdbd), --ifm-footer-link-color (#9e9e9e) in frontend-book/src/css/custom.css
- [ ] T051 [P] [US2] Define link color variables in `[data-theme='dark']`: --ifm-link-color (#64b5f6), --ifm-link-hover-color (#90caf9) in frontend-book/src/css/custom.css
- [ ] T052 [P] [US2] Define system color variables in `[data-theme='dark']`: --ifm-color-success (#66bb6a), --ifm-color-info (#42a5f5), --ifm-color-warning (#ffa726), --ifm-color-danger (#ef5350) in frontend-book/src/css/custom.css

### Dark Mode Component Adjustments

- [ ] T053 [US2] Adjust dark mode shadows: Update card and button shadows in `[data-theme='dark']` context with darker rgba values (0 2px 8px rgba(0,0,0,0.4) for cards, 0 4px 12px rgba(0,0,0,0.5) for button hover) in frontend-book/src/css/custom.css
- [ ] T054 [US2] Adjust dark mode table styling: `[data-theme='dark']` th with background (#2a2a2a), tr:nth-child(even) with background rgba(255,255,255,0.03) in frontend-book/src/css/custom.css
- [ ] T055 [US2] Adjust dark mode blockquote: `[data-theme='dark']` blockquote with background rgba(100,181,246,0.08) in frontend-book/src/css/custom.css

### Theme Toggle Verification

- [ ] T056 [US2] Verify Docusaurus theme toggle button exists in navbar (no code changes, just verification)
- [ ] T057 [US2] Test theme persistence: Toggle to dark mode, navigate to different page, verify dark mode persists
- [ ] T058 [US2] Test theme persistence across sessions: Toggle to dark mode, close browser, reopen, verify dark mode persists

**Checkpoint**: Dark mode theme complete - verify with quickstart.md Section 2 (Visual Inspection - Dark Mode) and Section 7 (Theme Toggle Testing)

---

## Phase 5: User Story 4 - Accessible Theme for All Users (Priority: P2)

**Goal**: Implement accessibility features including reduced motion support, focus states, and WCAG AA compliance verification

**Independent Test**: Run axe DevTools scan on homepage in both light and dark modes - verify 0 color contrast violations. Enable OS "prefers-reduced-motion" setting and verify all animations are disabled. Use Tab key to navigate page and verify all interactive elements show focus indicators.

### Accessibility Features

- [ ] T059 [P] [US4] Add reduced motion CSS: `@media (prefers-reduced-motion: reduce)` with *, *::before, *::after selectors setting animation-duration (0.01ms !important), animation-iteration-count (1 !important), transition-duration (0.01ms !important) in frontend-book/src/css/custom.css
- [ ] T060 [P] [US4] Add focus states for all interactive elements: :focus-visible selector with outline (2px solid --ifm-color-primary), outline-offset (2px), border-radius (4px) in frontend-book/src/css/custom.css
- [ ] T061 [P] [US4] Override button focus state: .button:focus-visible with outline (2px solid --ifm-color-primary-light) for better visibility in frontend-book/src/css/custom.css
- [ ] T062 [P] [US4] Override link focus state: a:focus-visible with outline (2px solid --ifm-color-primary), outline-offset (2px), text-decoration underline in frontend-book/src/css/custom.css

### WCAG AA Compliance Verification

- [ ] T063 [US4] Manual contrast check: Use browser DevTools color picker to verify light mode body text (#212121 on #ffffff) meets 4.5:1 ratio
- [ ] T064 [US4] Manual contrast check: Use browser DevTools color picker to verify light mode links (#1976d2 on #ffffff) meets 4.5:1 ratio
- [ ] T065 [US4] Manual contrast check: Use browser DevTools color picker to verify dark mode body text (#e0e0e0 on #1a1a1a) meets 4.5:1 ratio
- [ ] T066 [US4] Manual contrast check: Use browser DevTools color picker to verify dark mode links (#64b5f6 on #1a1a1a) meets 4.5:1 ratio
- [ ] T067 [US4] Run axe DevTools scan in light mode: Verify 0 color contrast violations (follow quickstart.md Section 5C)
- [ ] T068 [US4] Run axe DevTools scan in dark mode: Verify 0 color contrast violations (follow quickstart.md Section 5C)

### Keyboard Navigation Testing

- [ ] T069 [US4] Test keyboard navigation: Press Tab repeatedly and verify all interactive elements (links, buttons, inputs, theme toggle) show focus indicators
- [ ] T070 [US4] Test theme toggle keyboard access: Tab to theme toggle button, press Enter or Space, verify theme switches
- [ ] T071 [US4] Test reduced motion: Enable OS "prefers-reduced-motion" setting, toggle theme, verify instant change with no animation (follow quickstart.md Section 5A)

**Checkpoint**: Accessibility features complete - verify with quickstart.md Section 5 (Accessibility Testing) and confirm WCAG AA compliance

---

## Phase 6: User Story 3 - Responsive Theme Across Devices (Priority: P3)

**Goal**: Implement responsive design adjustments for mobile, tablet, and desktop viewports

**Independent Test**: Open DevTools responsive mode and test viewports: 375px (mobile), 768px (tablet), 1440px (desktop). Verify font sizes, button padding, card layouts, and spacing adjust appropriately while maintaining theme colors and visual identity.

### Mobile Responsive Adjustments (<768px)

- [ ] T072 [P] [US3] Add mobile font size adjustment: `@media (max-width: 768px)` with :root --ifm-font-size-base (15px) in frontend-book/src/css/custom.css
- [ ] T073 [P] [US3] Add mobile heading size adjustments: `@media (max-width: 768px)` with h1 (2rem), h2 (1.5rem), h3 (1.25rem), h4 (1.125rem), h5 (1rem), h6 (0.875rem) in frontend-book/src/css/custom.css
- [ ] T074 [P] [US3] Add mobile button adjustments: `@media (max-width: 768px)` with .button padding (0.625rem 1.25rem), font-size (0.95rem), minimum touch target (min-height: 44px, min-width: 44px) in frontend-book/src/css/custom.css
- [ ] T075 [P] [US3] Add mobile section padding: `@media (max-width: 768px)` with section padding (1.5rem 1rem) in frontend-book/src/css/custom.css
- [ ] T076 [P] [US3] Add mobile card adjustments: `@media (max-width: 768px)` with .card padding (1rem), margin-bottom (1rem) in frontend-book/src/css/custom.css
- [ ] T077 [P] [US3] Add mobile container padding: `@media (max-width: 768px)` with .container, .main-wrapper padding (0 0.75rem) in frontend-book/src/css/custom.css

### Desktop Responsive Enhancements (>1024px)

- [ ] T078 [P] [US3] Add desktop section padding: `@media (min-width: 1024px)` with section padding (3rem 1.5rem) in frontend-book/src/css/custom.css
- [ ] T079 [P] [US3] Add desktop card spacing: `@media (min-width: 1024px)` with .card margin-bottom (2rem) in frontend-book/src/css/custom.css

### Responsive Testing

- [ ] T080 [US3] Test mobile viewport (375px): Verify 15px base font, reduced heading sizes, touch-friendly buttons (44px minimum), single-column layouts (follow quickstart.md Section 4, Step 2)
- [ ] T081 [US3] Test tablet viewport (768px): Verify 16px base font, responsive grid, standard spacing (follow quickstart.md Section 4, Step 3)
- [ ] T082 [US3] Test desktop viewport (1440px): Verify 16px base font, max-width 1200px container, multi-column layouts (follow quickstart.md Section 4, Step 4)

**Checkpoint**: Responsive design complete - verify with quickstart.md Section 4 (Responsive Testing)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, cleanup, and cross-browser testing

- [ ] T083 [P] Add print stylesheet: `@media print` with [data-theme] attribute reset to light mode colors, remove shadows and decorative effects in frontend-book/src/css/custom.css
- [ ] T084 [P] Add CSS comments for maintainability: Document each section (Colors, Typography, Components, Responsive, Accessibility) with clear comments explaining purpose in frontend-book/src/css/custom.css
- [ ] T085 [P] Verify no !important overrides: Review custom.css and remove any !important declarations (except in prefers-reduced-motion for accessibility)
- [ ] T086 Cross-page verification: Test all page types per quickstart.md Section 10 (Homepage /, Docs /docs/intro, Blog /blog, 404 page)
- [ ] T087 Browser compatibility testing: Test theme in Chrome, Firefox, Safari, Edge per quickstart.md Section 6
- [ ] T088 Code block syntax highlighting verification: Navigate to page with code examples, verify GitHub theme (light) and Dracula theme (dark) render correctly per quickstart.md Section 8
- [ ] T089 Run complete quickstart.md validation: Execute all 10 test sections and check Success Criteria Checklist
- [ ] T090 Delete backup file: Remove `frontend-book/src/css/custom.css.backup` after confirming new theme works

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User Story 1 (Light Mode) → MVP - should complete first
  - User Story 2 (Dark Mode) → Can start after US1 structure is defined (shares CSS variable approach)
  - User Story 4 (Accessibility) → Can develop in parallel with US2 (applies to both themes)
  - User Story 3 (Responsive) → Can develop in parallel with US2/US4 (independent media queries)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - Light Mode**: Can start after Foundational (Phase 2) - No dependencies on other stories - **MVP PREREQUISITE**
- **User Story 2 (P2) - Dark Mode**: Depends on US1 for CSS variable structure, can start once US1 color variables are defined (T011-T019 complete)
- **User Story 4 (P2) - Accessibility**: Can start after Foundational (Phase 2) - Independent of other stories but tests both US1 and US2
- **User Story 3 (P3) - Responsive**: Can start after Foundational (Phase 2) - Independent of other stories (uses separate media queries)

### Within Each User Story

**User Story 1 (Light Mode)**:
1. All color variable tasks (T011-T019) can run in parallel
2. All typography tasks (T020-T024) can run in parallel (depends on T021 for base size)
3. Component styling tasks (T025-T041) are mostly parallel, but hover/active states depend on default state
4. Layout tasks (T042-T043) are independent

**User Story 2 (Dark Mode)**:
1. All dark mode color variable tasks (T044-T052) can run in parallel
2. Component adjustments (T053-T055) can run in parallel
3. Theme toggle verification (T056-T058) must run sequentially (test workflow)

**User Story 4 (Accessibility)**:
1. All accessibility feature tasks (T059-T062) can run in parallel
2. WCAG compliance verification (T063-T068) can run in parallel
3. Keyboard navigation tests (T069-T071) can run in parallel

**User Story 3 (Responsive)**:
1. All mobile adjustment tasks (T072-T077) can run in parallel
2. Desktop enhancement tasks (T078-T079) can run in parallel
3. Responsive testing tasks (T080-T082) should run sequentially (different viewports)

**Polish Phase**:
- Most tasks (T083-T085) can run in parallel
- Testing tasks (T086-T089) should run sequentially (comprehensive validation)

### Parallel Opportunities

**After Foundational Phase Completes**:
- User Story 1 (Light Mode) → **Start immediately as MVP**
- Once US1 color variables defined (T011-T019 complete):
  - User Story 2 (Dark Mode) → Can start in parallel
  - User Story 3 (Responsive) → Can start in parallel
  - User Story 4 (Accessibility) → Can start in parallel

**Within User Story 1**:
```bash
# Launch all color variable tasks together:
Task T011: Primary colors
Task T012: Background colors
Task T013: Text colors
Task T014: Code block colors
Task T015: Border colors
Task T016: Navbar colors
Task T017: Footer colors
Task T018: Link colors
Task T019: System colors

# Launch all typography tasks together:
Task T020: Font families
Task T021: Font size and line-height
Task T022: Heading font weight
```

**Within User Story 2**:
```bash
# Launch all dark mode color variable tasks together:
Task T044: Primary colors (dark)
Task T045: Background colors (dark)
Task T046: Text colors (dark)
Task T047: Code block colors (dark)
Task T048: Border colors (dark)
Task T049: Navbar colors (dark)
Task T050: Footer colors (dark)
Task T051: Link colors (dark)
Task T052: System colors (dark)
```

**Within User Story 4**:
```bash
# Launch all accessibility feature tasks together:
Task T059: Reduced motion CSS
Task T060: Focus states (general)
Task T061: Button focus state
Task T062: Link focus state

# Launch all contrast check tasks together:
Task T063: Light mode body text contrast
Task T064: Light mode link contrast
Task T065: Dark mode body text contrast
Task T066: Dark mode link contrast
```

**Within User Story 3**:
```bash
# Launch all mobile adjustment tasks together:
Task T072: Mobile font size
Task T073: Mobile heading sizes
Task T074: Mobile button adjustments
Task T075: Mobile section padding
Task T076: Mobile card adjustments
Task T077: Mobile container padding

# Launch desktop tasks together:
Task T078: Desktop section padding
Task T079: Desktop card spacing
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004) → ~10 minutes
2. Complete Phase 2: Foundational (T005-T010) → ~15 minutes
3. Complete Phase 3: User Story 1 (T011-T043) → ~90-120 minutes
4. **STOP and VALIDATE**: Test User Story 1 independently with quickstart.md Section 1 (Light Mode Visual Inspection)
5. Deploy/demo light mode theme ✅ **MVP COMPLETE**

**Total MVP Time**: ~2-3 hours
**MVP Deliverable**: Professional light mode theme with WCAG AA compliance

### Incremental Delivery

**Release 1 (MVP)**: User Story 1 - Light Mode
- Complete Setup + Foundational + US1
- Test light mode independently (quickstart.md Section 1)
- Verify WCAG AA contrast ratios
- **Outcome**: Documentation has professional light theme, replaces old theme completely

**Release 2**: User Story 2 + User Story 4
- Add dark mode color variables (T044-T052)
- Add dark mode component adjustments (T053-T055)
- Verify theme toggle persistence (T056-T058)
- Add accessibility features (T059-T062)
- Verify WCAG AA in both modes (T063-T068)
- Test keyboard navigation (T069-T071)
- **Outcome**: Users can switch to dark mode with full accessibility support

**Release 3**: User Story 3 + Polish
- Add responsive adjustments (T072-T082)
- Test all viewports (mobile, tablet, desktop)
- Complete polish tasks (T083-T090)
- Run comprehensive validation (quickstart.md all sections)
- **Outcome**: Complete theme system ready for production

**Total Implementation Time**: 4-6 hours across all releases

### Parallel Team Strategy

With multiple developers:

1. **Team completes Setup + Foundational together** (T001-T010) → ~25 minutes
2. **Once Foundational is done**:
   - **Developer A**: User Story 1 (T011-T043) → Light Mode (MVP)
3. **Once US1 color structure is defined** (after T011-T019):
   - **Developer A**: Continue US1 components (T023-T043)
   - **Developer B**: User Story 2 (T044-T058) → Dark Mode
   - **Developer C**: User Story 4 (T059-T071) → Accessibility
   - **Developer D**: User Story 3 (T072-T082) → Responsive
4. **Stories complete independently**, then polish together (T083-T090)

---

## Task Summary

**Total Tasks**: 90
- **Setup**: 4 tasks
- **Foundational**: 6 tasks
- **User Story 1 (Light Mode)**: 33 tasks (11 color variables + 5 typography + 17 components + 2 layout) - **MVP**
- **User Story 2 (Dark Mode)**: 15 tasks (9 color variables + 3 component adjustments + 3 verification)
- **User Story 4 (Accessibility)**: 13 tasks (4 features + 6 WCAG checks + 3 keyboard tests)
- **User Story 3 (Responsive)**: 11 tasks (6 mobile + 2 desktop + 3 testing)
- **Polish**: 8 tasks

**Parallel Opportunities**: 48 tasks marked [P] can run in parallel within their phase

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1 only) = 43 tasks total

**Independent Test Criteria Met**:
- ✅ US1: Can verify light mode works without any other user stories
- ✅ US2: Can verify dark mode works after US1 is complete (shares structure)
- ✅ US4: Can verify accessibility in both modes independently
- ✅ US3: Can verify responsive behavior independently with DevTools

---

## Notes

- [P] tasks = different CSS sections, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- All file paths point to single file: `frontend-book/src/css/custom.css`
- Commit after each phase or logical group of tasks
- Stop at any checkpoint to validate story independently
- Use quickstart.md for comprehensive manual testing procedures
- No automated tests for CSS (visual verification required)
