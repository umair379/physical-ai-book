# Quickstart Guide: Theme Verification

**Feature**: 001-theme-spec
**Purpose**: Manual testing steps for verifying theme implementation

## Prerequisites

- Docusaurus dev server running: `cd frontend-book && npm start`
- Browser with DevTools (Chrome, Firefox, Safari, or Edge)
- axe DevTools browser extension (optional, for accessibility testing)

## 1. Visual Inspection - Light Mode

**Goal**: Verify light mode theme colors match specification

### Steps:

1. **Open Documentation**:
   - Navigate to `http://localhost:3000`
   - Ensure theme toggle is set to **light mode** (sun icon visible)

2. **Verify Colors**:
   - **Background**: Should be clean white (#ffffff)
   - **Text**: Should be dark gray (#212121), NOT pure black
   - **Primary accent**: Links and buttons should be blue (#1976d2)
   - **Secondary text**: Subtle gray (#616161)

3. **Check Components**:
   - **Buttons**: Should have blue gradient, white text, rounded corners (8px)
   - **Cards**: Should have subtle shadows, light borders, rounded corners (12px)
   - **Code blocks**: Should have light gray background (#f5f5f5)
   - **Navbar**: White background with subtle shadow
   - **Footer**: Light gray background (#fafafa)

4. **Test Interactions**:
   - **Hover buttons**: Should show shadow elevation and slight upward movement
   - **Hover links**: Should show underline with smooth transition
   - **Hover cards**: Should show elevated shadow

## 2. Visual Inspection - Dark Mode

**Goal**: Verify dark mode theme colors match specification

### Steps:

1. **Toggle to Dark Mode**:
   - Click theme toggle button (moon icon should appear)
   - Page should smoothly transition to dark theme

2. **Verify Colors**:
   - **Background**: Should be dark slate (#1a1a1a), NOT pure black
   - **Text**: Should be light gray (#e0e0e0), NOT pure white
   - **Primary accent**: Links and buttons should be light blue (#64b5f6)
   - **Secondary text**: Medium gray (#bdbdbd)

3. **Check Components**:
   - **Buttons**: Should have light blue gradient
   - **Cards**: Should have darker shadows, visible borders
   - **Code blocks**: Should have darker background (#2a2a2a)
   - **Navbar**: Dark background (#242424) with shadow
   - **Footer**: Very dark background (#1a1a1a)

4. **Verify Theme Persistence**:
   - Navigate to another page (e.g., /docs/intro)
   - Dark mode should persist (no flash of light mode)
   - Refresh page - dark mode should persist
   - Close browser, reopen, navigate to site - dark mode should persist

## 3. Contrast Testing (WCAG AA Compliance)

**Goal**: Verify all text meets 4.5:1 contrast ratio, UI components meet 3:1

### Steps:

1. **Install Contrast Checker** (if not already):
   - Chrome: "Colour Contrast Checker" extension
   - Or use browser DevTools built-in contrast checker

2. **Test Light Mode Contrast**:
   - **Body text on white**: Should be 16:1 or higher ✅
   - **Blue links on white**: Should be 4.5:1 or higher ✅
   - **Secondary text on white**: Should be 5.7:1 or higher ✅

3. **Test Dark Mode Contrast**:
   - **Body text on dark slate**: Should be 12.6:1 or higher ✅
   - **Light blue links on dark slate**: Should be 8.6:1 or higher ✅
   - **Secondary text on dark slate**: Should be 9.7:1 or higher ✅

### Using DevTools:

1. Open DevTools (F12)
2. Select "Elements" tab
3. Click on text element
4. In "Styles" pane, hover over color value
5. Color picker shows contrast ratio automatically

## 4. Responsive Testing

**Goal**: Verify theme adapts correctly to mobile, tablet, and desktop viewports

### Steps:

1. **Open Responsive Mode**:
   - Chrome: DevTools → Toggle Device Toolbar (Ctrl+Shift+M)
   - Firefox: DevTools → Responsive Design Mode (Ctrl+Shift+M)

2. **Test Mobile (375px)**:
   - Set viewport to 375x667 (iPhone SE)
   - **Font size**: Should be slightly smaller (15px base)
   - **Buttons**: Should have touch-friendly size (min 44x44px)
   - **Cards**: Should stack vertically
   - **Navigation**: Should use hamburger menu
   - **Layout**: Single-column

3. **Test Tablet (768px)**:
   - Set viewport to 768x1024 (iPad)
   - **Font size**: Standard (16px base)
   - **Layout**: Responsive grid, some multi-column
   - **Navigation**: May collapse to hamburger
   - **Content**: Well-spaced, readable

4. **Test Desktop (1440px)**:
   - Set viewport to 1440x900 (laptop)
   - **Font size**: Standard (16px base)
   - **Container**: Max-width 1200px, centered
   - **Layout**: Multi-column where appropriate
   - **Navigation**: Full horizontal navbar

## 5. Accessibility Testing

**Goal**: Verify theme meets accessibility requirements

### A. Reduced Motion Test

1. **Enable Reduced Motion** (OS Setting):
   - **Windows**: Settings → Accessibility → Visual effects → Animation effects (OFF)
   - **macOS**: System Preferences → Accessibility → Display → Reduce motion (ON)

2. **Verify Behavior**:
   - Toggle theme - should change instantly (no smooth fade)
   - Hover buttons/cards - no animations
   - All transitions disabled

### B. Keyboard Navigation Test

1. **Tab through page**:
   - Press Tab repeatedly
   - All interactive elements should show focus outline
   - Focus outline should use primary color (blue)
   - Theme toggle should be keyboard accessible (Tab + Enter/Space)

2. **Verify Focus States**:
   - Links should show visible focus ring
   - Buttons should show visible focus ring
   - Inputs should show visible focus ring

### C. Automated Accessibility Scan

1. **Install axe DevTools**:
   - Chrome/Firefox: Install "axe DevTools" extension

2. **Run Scan**:
   - Open DevTools → axe DevTools tab
   - Click "Scan ALL of my page"
   - Verify **0 violations** for color contrast
   - Verify **0 violations** for keyboard accessibility

3. **Repeat for Dark Mode**:
   - Toggle to dark mode
   - Run scan again
   - Verify **0 violations**

## 6. Browser Compatibility Testing

**Goal**: Verify theme works across major browsers

### Browsers to Test:

1. **Chrome** (latest version)
2. **Firefox** (latest version)
3. **Safari** (latest version, macOS only)
4. **Edge** (latest version)

### For Each Browser:

1. Open `http://localhost:3000`
2. Verify colors appear correct (not broken)
3. Toggle theme - verify toggle works
4. Check responsive behavior (mobile, tablet, desktop)
5. Verify no console errors related to CSS

## 7. Theme Toggle Testing

**Goal**: Verify theme toggle functionality works correctly

### Steps:

1. **Toggle Multiple Times**:
   - Click theme toggle 5 times rapidly
   - Should switch smoothly each time
   - No flashing or glitches

2. **Test Persistence Across Pages**:
   - Set theme to dark mode
   - Navigate to /docs/intro
   - Verify dark mode persists
   - Navigate to /blog
   - Verify dark mode persists

3. **Test Persistence Across Sessions**:
   - Set theme to dark mode
   - Close browser completely
   - Reopen browser
   - Navigate to documentation site
   - Verify dark mode is still active

4. **Test OS Preference Detection**:
   - Close browser
   - Set OS to dark mode preference
   - Open browser for first time (clear localStorage)
   - Navigate to site
   - Should default to dark mode automatically

## 8. Code Block Verification

**Goal**: Verify syntax highlighting works in both themes

### Steps:

1. **Light Mode**:
   - Navigate to page with code examples (e.g., /docs/module-1/chapter-1)
   - Code blocks should have light gray background
   - Syntax highlighting should be visible (GitHub theme)
   - Colors should be readable

2. **Dark Mode**:
   - Toggle to dark mode
   - Code blocks should have dark background
   - Syntax highlighting should adapt (Dracula theme)
   - Colors should be readable with high contrast

## 9. Print Preview Test

**Goal**: Verify theme handles print styling correctly

### Steps:

1. **Open Print Preview**:
   - Chrome: File → Print (Ctrl+P)
   - Or use DevTools → More Tools → Rendering → Emulate CSS media: print

2. **Verify Print Styling**:
   - Should use light mode colors (even if dark mode active)
   - Text should be dark, background should be white
   - No shadows or decorative effects
   - Content should be readable when printed

## 10. Cross-Page Verification

**Goal**: Verify theme applies consistently to all page types

### Pages to Test:

1. **Homepage** (`/`):
   - Verify hero section, module cards, quick links
   - All colors match theme

2. **Documentation Page** (`/docs/intro`):
   - Verify article text, headings, links
   - Sidebar navigation styled correctly

3. **Blog List** (`/blog`):
   - Verify blog post cards
   - Preview text readable

4. **Blog Post** (`/blog/2025-12-26-welcome`):
   - Verify article content
   - Code blocks styled correctly

5. **404 Page** (navigate to `/nonexistent`):
   - Error page styled with theme
   - Links work correctly

## Success Criteria Checklist

After completing all tests, verify:

- [ ] Light mode colors match specification (#ffffff bg, #212121 text, #1976d2 primary)
- [ ] Dark mode colors match specification (#1a1a1a bg, #e0e0e0 text, #64b5f6 primary)
- [ ] All contrast ratios meet WCAG AA (4.5:1 text, 3:1 UI)
- [ ] Theme toggle works and persists across sessions
- [ ] Responsive design works on mobile (375px), tablet (768px), desktop (1440px)
- [ ] Reduced motion preference is respected (no animations)
- [ ] Keyboard navigation works (all elements have focus states)
- [ ] axe DevTools scan shows 0 color contrast violations
- [ ] Theme works in Chrome, Firefox, Safari, Edge
- [ ] Code blocks have correct syntax highlighting in both themes
- [ ] All page types (home, docs, blog, 404) styled consistently

## Troubleshooting

**Issue**: Colors don't match specification
- **Solution**: Clear browser cache (Ctrl+Shift+Delete), hard refresh (Ctrl+Shift+R)

**Issue**: Theme toggle doesn't persist
- **Solution**: Check browser localStorage is enabled, check for console errors

**Issue**: Responsive breakpoints don't work
- **Solution**: Verify DevTools is not overriding CSS, check viewport meta tag

**Issue**: Accessibility scan shows violations
- **Solution**: Check specific elements flagged, adjust colors or add ARIA labels

**Issue**: Theme looks different in different browsers
- **Solution**: Verify browser is latest version, check for browser-specific CSS bugs

## Estimated Testing Time

- **Quick verification** (light + dark mode visual check): 5 minutes
- **Full verification** (all 10 test sections): 30-45 minutes
- **Comprehensive testing** (all browsers, all devices): 1-2 hours
