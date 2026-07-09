# Phase D prompt — real dashboard screenshots for the README

Paste the block below to Codex / Claude Code, running from the repo root.

---

You are working in the `Mental-Health` repo (root contains `README.md`, `src/`, `web/`,
`v1/ v2/ v3/`, `report_latex/`). Do **Phase D** of the v1.1 plan: replace placeholder/
diagram images in the README hero with **real screenshots** of the live product, wire
them in cleanly, and optimize them.

## Live URLs
- Website (Next.js, Vercel): https://mental-health-iota-vert.vercel.app
- Dashboards (Streamlit): https://mental-health-razouk.streamlit.app
  - Versions are selected from the website version cards; the dashboard opens with a
    `?version=v0` … `?version=v3` query param. Language via `?lang=en` / `?lang=fr`.

## Capture (5 screenshots, PNG, ~1600px wide, retina if possible)
Save all to `docs/img/`. Use these exact filenames:
1. `web_hero.png`       — website landing / hero (EN).
2. `web_predictor.png`  — the "predict any country" widget with a country selected and a result shown.
3. `dash_overview.png`  — a v2 dashboard overview page (open `...streamlit.app/?version=v2`), a clean chart in view.
4. `dash_chart_guide.png` — a dashboard page showing one of the chart-guide explanation panels.
5. `web_fr.png`         — the website with the FR toggle active (shows i18n works).

Capture tips: use a real browser at ~1440px viewport, hide personal bookmarks/toolbars,
wait for charts to fully render, crop out OS chrome. Keep each file **< 400 KB** (run
them through `pngquant`/`oxipng`/`squoosh`, or `sips`/`ffmpeg` if those aren't installed).

## Optional (nice to have)
- `docs/img/toggle.gif` — a 6–8s GIF of switching the website EN→FR. Keep < 3 MB.
  (`ffmpeg` + `gifski`, or any screen-recorder → GIF.)

## Wire into README.md
- Replace the current hero/figure images at the top of `README.md` with `web_hero.png`
  (and `dash_overview.png` beside it if a two-up layout already exists).
- Add a short **"Gallery"** section (place it right after "Live demo" or near the top),
  a 2-column HTML `<table>` like the existing figure tables in the README, showing
  `web_predictor.png`, `dash_overview.png`, `dash_chart_guide.png`, and `web_fr.png`,
  each with a one-line `<sub>` caption.
- Every `<img>` MUST have descriptive **alt text**, and captions should read naturally in
  English. Use relative paths (`docs/img/...`), not absolute.
- Do not remove the existing analytical figures (SHAP/UMAP/etc.) — this only adds product
  screenshots.

## Constraints (important)
- Do not change any code, data, or the `web/` app — this task is images + README only.
- Keep the $0 stack; no paid tools.
- After wiring, run `python -m ruff check src` (should stay clean — you touched no Python)
  and confirm the README still renders (no broken image links; `grep -o 'docs/img/[^)"]*'
  README.md` files all exist in `docs/img/`).

## Commit (strict)
- Stage `docs/img/*` and `README.md` only.
- One commit, conventional style, e.g.: `docs: add real dashboard and website screenshots`.
- **No attribution of any kind** — no "Co-authored-by", no "Generated with", no tool
  names, no AI mentions anywhere in the commit message or body.
- Push, then confirm the images render on the GitHub repo page.

When done, report: the 5 (or 6) files added with their byte sizes, and the README sections
changed.
