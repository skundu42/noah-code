# Noah Code — Official Landing Page (`website/`)

Zero-dependency, static developer landing page and interactive TUI cockpit simulator for **Noah Code v2.2** (`noah-code`).

## Files

- `index.html` — Semantic HTML5 structure, inline SVG architecture diagrams and Lucide-style icons, ARIA tablists, and interactive Noah Code TUI simulator.
- `styles.css` — Custom-property design system faithfully mapped from `src/noah_code/themes.py` (**Atom One Dark**, **Noah Ocean**, and WCAG AAA **Daylight**), fluid 12-column Bento layout, intraline diff styling, and `@media (prefers-reduced-motion: reduce)` support.
- `app.js` — Vanilla JavaScript controller (zero build step, zero runtime dependencies) powering:
  - Live bidirectional palette switching (`Atom One Dark`, `Noah Ocean`, `Daylight`) persisted in `localStorage`
  - 1-click install command switcher (`install.sh`, `uv tool install noah-code`, `pipx install noah-code`) with clipboard verification feedback
  - 4-scenario interactive TUI Cockpit (`1`–`4` hotkeys or click):
    1. **Multi-Agent Swarm (`task.orchestrate`)** with live inspectable subagent pills and context rail
    2. **Intraline Diff Review (`Ctrl+D`)** with Split/Unified toggle (`S`), Stage Hunk (`G`), and Revert Hunk (`H`)
    3. **Zero-Typing Model Picker (`Ctrl+L`)** with live search across curated 2025–2026 coding models & OpenRouter routes
    4. **`/resume`, Claude Skills & 1-Click MCPs** with session previews and toggleable MCP server presets
  - Filterable Keyboard & Slash Command Explorer

## Local Preview

No bundler or Node.js build step is required. Serve the folder with Python's built-in HTTP server:

```bash
python3 -m http.server 8080 --directory website
```

Then open `http://localhost:8080` in any modern browser.

## Deployment

Deploy `website/` directly as a static root directory to GitHub Pages, Cloudflare Pages, Netlify, or S3/CloudFront with zero build command required.
