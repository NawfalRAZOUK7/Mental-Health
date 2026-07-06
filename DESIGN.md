# Design System — Mental Health Viz

One "refined warm editorial" language across every surface: the Next.js website,
the guide chatbot, the four Streamlit dashboards, and the report figures. The goal
is a calm, trustworthy look appropriate to the subject — not flashy.

## Sources of truth

| Surface | File |
| --- | --- |
| Python (Streamlit app + report figures) | `src/theme.py` |
| Streamlit native chrome (widgets, sidebar) | `.streamlit/config.toml` `[theme]` |
| Website + chatbot | `web/lib/theme.js` and `web/app/globals.css` (`:root`) |

Keep all four in sync when a token changes.

## Tokens

**Color**

| Token | Hex | Use |
| --- | --- | --- |
| bg | `#f4efe7` | page background |
| bg-soft | `#faf6ef` | alternating section |
| surface | `#ffffff` | cards |
| surface-alt | `#f7f2ea` | inset fills |
| ink | `#1c1b1a` | primary text |
| muted | `#6b6460` | secondary text |
| subtle | `#8f877f` | hints |
| **accent** | `#1f6f8b` | primary (teal) — links, buttons, key numbers |
| accent-dark | `#185a72` | hover |
| coral | `#b0453c` | secondary / alert |
| gold | `#d99a2b` | sparing highlight |
| green | `#2f8f6b` | success |
| border | `#e7ded1` | hairline |
| border-strong | `#d9d2c8` | emphasis divider |

**Chart palette (categorical):** teal → coral → gold → green → gray → purple.
**Sequential (intensity):** cream → deep teal. Both defined once in `src/theme.py`
and applied to Plotly (Streamlit) and matplotlib (figures).

**Typography:** headings `Source Serif 4` (600); UI/body `Space Grotesk`
(400 / 500 / 600). Scale: h1 30–52, h2 26–34, h3 18–20, body 16, label 12 uppercase.

**Shape:** radius 10px (controls) / 16px (cards); shadow `0 12px 30px rgba(28,27,26,.08)`.

## Components

Stat/KPI cards, primary/ghost buttons, pills/badges, section headers (serif title +
muted subtitle), inputs/selects, dataframes, the disclaimer banner, and chat bubbles —
all derived from the tokens above with consistent hover/focus states.

## Accessibility

- Text/background pairs meet WCAG AA (ink on cream, teal `#1f6f8b` on white).
- Visible focus ring: `0 0 0 3px rgba(31,111,139,.35)`.
- `prefers-reduced-motion` disables transitions/animations.
- Minimum body font 13–16px.

## Regenerating themed assets

```bash
# report figures pick up src/theme.py automatically
python scripts/run_advanced.py
# refresh the figure copies + predictions used by the web app
python scripts/build_web_data.py
```
