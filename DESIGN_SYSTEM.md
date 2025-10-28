# 🎨 Qallow Design System - Unified Across Web & Native Apps

## Color Palette

### Primary Colors
```
Cyan Primary:     #00d4ff  RGB(0, 212, 255)   - Main UI elements, borders, text
Green Success:    #00ff64  RGB(0, 255, 100)   - Start buttons, success states
Red Danger:       #ff6464  RGB(255, 100, 100) - Stop buttons, error states
Yellow Warning:   #ffd166  RGB(255, 209, 102) - Pause buttons, warnings
```

### Background Colors
```
Dark Background:  #0a0e27  RGB(10, 14, 39)    - Main background
Accent Background:#1a1f3a  RGB(26, 31, 58)    - Secondary background, panels
```

### Text Colors
```
Light Text:       #e8eefc  RGB(232, 238, 252) - Primary text
Muted Text:       #8aa1c1  RGB(138, 161, 193) - Secondary text, labels
```

---

## Typography

### Font Stack
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 
             'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 
             'Helvetica Neue', sans-serif;
```

### Monospace Font
```css
font-family: ui-monospace, monospace;
```

### Font Sizes
```
H1 (Header):      28px (web), 18px (native)
H2 (Subheader):   20px
H3 (Section):     16px
Body:             14px
Small:            12px
Label:            11px
```

### Font Weights
```
Regular:          400
Medium:           500
Semibold:         600
Bold:             700
```

---

## Spacing System

### Padding
```
xs:  4px
sm:  8px
md:  12px
lg:  16px
xl:  20px
2xl: 30px
```

### Gaps
```
Compact:   8px
Normal:    12px
Relaxed:   16px
Loose:     20px
```

### Margins
```
Component:  20px (bottom)
Section:    30px (bottom)
```

---

## Border & Radius

### Border Width
```
Thin:      1px
Medium:    2px
```

### Border Radius
```
Small:     4px
Medium:    6px
Large:     8px
XLarge:    12px
```

### Border Color
```
Primary:   rgba(0, 212, 255, 0.2)   - Subtle cyan border
Accent:    rgba(0, 212, 255, 0.3)   - Stronger cyan border
```

---

## Shadows & Effects

### Box Shadows
```
Small:     0 2px 4px rgba(0, 0, 0, 0.1)
Medium:    0 4px 8px rgba(0, 0, 0, 0.15)
Large:     0 8px 16px rgba(0, 0, 0, 0.2)
Glow:      0 0 10px rgba(0, 212, 255, 0.3)
```

### Text Effects
```
Glow:      text-shadow: 0 0 10px rgba(0, 212, 255, 0.5)
Subtle:    text-shadow: 0 0 5px rgba(0, 212, 255, 0.3)
```

---

## Component Styles

### Buttons

**Primary Button (Start)**
```
Background:  linear-gradient(135deg, #00ff64 0%, #00d4ff 100%)
Color:       #0a0e27
Padding:     12px 24px
Border:      none
Radius:      8px
Shadow:      0 0 20px rgba(0, 255, 100, 0.3)
Hover:       transform: translateY(-2px), shadow: 0 0 30px rgba(0, 255, 100, 0.5)
```

**Danger Button (Stop)**
```
Background:  linear-gradient(135deg, #ff6464 0%, #ff4444 100%)
Color:       white
Padding:     12px 24px
Border:      none
Radius:      8px
Shadow:      0 0 20px rgba(255, 100, 100, 0.3)
Hover:       transform: translateY(-2px), shadow: 0 0 30px rgba(255, 100, 100, 0.5)
```

**Secondary Button**
```
Background:  rgba(0, 212, 255, 0.1)
Color:       #00d4ff
Padding:     10px 20px
Border:      1px solid #00d4ff
Radius:      8px
Hover:       background: rgba(0, 212, 255, 0.2), shadow: 0 0 15px rgba(0, 212, 255, 0.3)
```

### Input Fields

**Text Input**
```
Background:  rgba(0, 212, 255, 0.05)
Color:       #00d4ff
Border:      1px solid rgba(0, 212, 255, 0.3)
Radius:      6px
Padding:     10px
Focus:       border-color: #00ff64, shadow: 0 0 10px rgba(0, 255, 100, 0.2)
```

### Cards/Panels

**Component Card**
```
Background:  rgba(26, 31, 58, 0.8)
Border:      1px solid rgba(0, 212, 255, 0.2)
Radius:      12px
Padding:     20px
Shadow:      0 4px 20px rgba(0, 212, 255, 0.1)
```

### Status Indicators

**Running (Green)**
```
Background:  #00ff64
Color:       white
Border:      1px solid #00ff64
```

**Stopped (Red)**
```
Background:  #ff6464
Color:       white
Border:      1px solid #ff6464
```

**Paused (Yellow)**
```
Background:  #ffd166
Color:       black
Border:      1px solid #ffd166
```

---

## Animations

### Transitions
```
Default:     all 0.3s ease
Fast:        all 0.15s ease
Slow:        all 0.5s ease
```

### Keyframes

**Slide In**
```css
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

**Glow Pulse**
```css
@keyframes glowPulse {
  0%, 100% {
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
  }
  50% {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
  }
}
```

---

## Responsive Breakpoints

```
Mobile:    < 480px
Tablet:    480px - 768px
Desktop:   768px - 1200px
Wide:      > 1200px
```

---

## Accessibility

### Color Contrast
```
Text on Background:  4.5:1 (WCAG AA)
UI Components:       3:1 (WCAG AA)
```

### Focus States
```
Outline:   2px solid #00d4ff
Offset:    2px
```

### Keyboard Navigation
```
Tab Order:  Logical flow
Focus:      Visible indicator
Shortcuts:  Alt + key combinations
```

---

## Implementation Guide

### Web App (CSS)
```css
:root {
  --primary: #00d4ff;
  --success: #00ff64;
  --danger: #ff6464;
  --bg-dark: #0a0e27;
  --bg-accent: #1a1f3a;
  --text: #e8eefc;
  --text-muted: #8aa1c1;
}
```

### Native App (Rust)
```rust
pub const COLOR_PRIMARY: u32 = 0x00d4ff;
pub const COLOR_SUCCESS: u32 = 0x00ff64;
pub const COLOR_DANGER: u32 = 0xff6464;
pub const COLOR_BG_DARK: u32 = 0x0a0e27;
pub const COLOR_BG_ACCENT: u32 = 0x1a1f3a;
pub const COLOR_TEXT: u32 = 0xe8eefc;
pub const COLOR_MUTED: u32 = 0x8aa1c1;
```

---

## Usage Examples

### Web App Button
```jsx
<button className="btn btn-start">▶️ Start VM</button>
```

### Native App Button
```rust
let mut btn = button::Button::default()
    .with_label("▶️ Start VM");
btn.set_color(Color::from_hex(COLOR_SUCCESS));
```

### Web App Card
```jsx
<div className="component">
  <h3 className="component-title">Dashboard</h3>
  {/* content */}
</div>
```

### Native App Card
```rust
let mut card = group::Group::default();
card.set_color(Color::from_hex(COLOR_BG_ACCENT));
```

---

**Design System Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: ✅ Complete and Unified

