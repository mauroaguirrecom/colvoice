# ColVoice design system

## Direction

ColVoice is a calm production workspace inspired by modern audio editors. It uses a warm light canvas, white working surfaces, near-black actions, restrained violet accents, and clear Spanish copy. The interface should feel focused and trustworthy during repeated generation work.

## Layout

- A persistent left sidebar carries product identity, mode navigation, saved-voice count, and local GPU status.
- A compact workspace bar provides context and the local-privacy cue.
- Each mode uses a wide composition area and a narrower result rail.
- Common inputs remain visible. Reference metadata and model settings live in accordions.
- On narrow screens, the sidebar becomes a full-screen drawer and the result rail follows the editor in document order.

## Tokens

- Canvas: `#f5f4f1`
- Surface: `#ffffff`
- Surface subtle: `#faf9f7`
- Text: `#181817`
- Secondary text: `#686762`
- Border: `#e3e1dc`
- Accent: `#6656d9`
- Success: `#167653`
- Error: `#b42318`
- Radius: 12px for panels and fields; pills only for compact status controls.

## Typography

Use Inter when available with Segoe UI and Arial fallbacks. Headings use weight and spacing rather than decorative display effects. Interface text is 14–16px with tabular numerals for measured controls.

## Components

- Primary actions are near-black, full-width, and at least 44px high.
- Secondary actions are white with a single neutral border.
- Inputs use white or subtle warm surfaces with visible focus rings.
- Audio upload areas use a dashed border and short instructional copy.
- Status fields use plain language: the result, the problem, and the next action.

## Motion

The active workspace enters once with a short upward settle. Hover and focus transitions remain under 180ms and respect reduced-motion preferences.

## Responsive behavior

The editor and result rail stack below 900px. Header metadata collapses below 700px. Controls stay at least 44px tall and no horizontal scrolling is required.
