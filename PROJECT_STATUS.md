# Project status — v1.0 · Complete · Live · $0

**Mental Health Viz is feature-complete for its defined scope, deployed, and built entirely on free, open tools — no paid services and no payment/billing integrations of any kind.**

Live: website <https://mental-health-iota-vert.vercel.app> · dashboards <https://mental-health-razouk.streamlit.app> · code <https://github.com/NawfalRAZOUK7/Mental-Health>

---

## Definition of Done — all criteria met ✅

**Data & analysis**
- [x] Public sources integrated (WHO 2021, IHME GBD 2023, World Bank) for 183 countries
- [x] Reproducible pipelines (one command per version)
- [x] Leakage-free modeling, with the leakage documented honestly (R² 0.75 → 0.19)
- [x] Machine learning: LightGBM + Optuna (nested CV), SHAP, conformal intervals, hierarchical/ICC
- [x] Data mining: UMAP, country similarity network, subgroup discovery, FP-Growth rules
- [x] Deep learning: global N-BEATS forecaster with honest baselines
- [x] LaTeX report (PDF)

**Product & engineering**
- [x] Streamlit dashboards (v0–v3), unified design system
- [x] FastAPI prediction service (+ Swagger) with 90% conformal intervals
- [x] Docker + docker-compose
- [x] Next.js (React) website with live predictor + version links
- [x] Rule-based guide chatbot (no LLM, no API cost)
- [x] CI (GitHub Actions), smoke tests, ruff-clean
- [x] Professional repo: README, DESIGN.md, DEPLOY.md, CITATION.cff, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates

**Internationalization**
- [x] Full bilingual EN/FR — website, chatbot, and all Streamlit UI text (labels, headers, guides, messages, dynamic sentences)

**Ethics**
- [x] Non-clinical / educational disclaimers throughout; sources credited; MIT + CC BY 4.0

**Go-live**
- [x] Deployed on Streamlit Community Cloud + Vercel
- [x] Live-demo links in README

### Scope notes (by design, not gaps)
- Dropdown **option values** (e.g. "Both sexes", region/cause names) render in the source data's language — they are **data values, not UI text**, so they are intentionally not translated.
- The written **report is in French**; developer docs (README etc.) are in English by convention.
- "v1.0 complete" reflects the defined scope. Future ideas (more models, deeper data-value i18n) are **optional enhancements**, not open work.

---

## v1.1 — shipped ✅ (optional enhancements, delivered)

The v1.0-era backlog is now built, each opt-in and on the same $0 stack:

- [x] **Deep forecasting** — library-backed **darts N-BEATS + darts TFT** alongside the pure-PyTorch N-BEATS, with a `MHV_DEEP_BACKEND=torch|darts|all` switch. Deep deps isolated in `requirements-deep.txt`; base install and deployment unchanged.
- [x] **Spatial autocorrelation** — `src/17_spatial_model.py`: global Moran's I + LISA clusters over the similarity graph (opt-in `MHV_SPATIAL=1`). Honest result: the raw rate clusters (I ≈ 0.21, p = 0.001) but model residuals do **not** (I ≈ 0.00, p = 0.42).
- [x] **Dropdown data-value i18n** — region/cause/sex/age/metric option labels now render in French (display-only via `format_func`; raw values unchanged, filters intact).
- [x] **Real screenshots** — website + dashboard captures wired into the README hero and a new Gallery section.

## Optional backlog (v1.2+) — genuinely future

- Longer time series / more covariates if new public data is added.
- Bayesian / additional hierarchical variants.

These are "nice to have." None is required for the project to be complete, correct, deployed, or free.

---

## Zero-cost stack — no payment solutions

Every component is free and open; the project uses **no paid tiers, no API keys that bill, and no payment/checkout code**.

| Layer | Tool | Cost |
| --- | --- | --- |
| Data | WHO, IHME GBD, World Bank Open Data | Free / public |
| Analysis | Python, pandas, scikit-learn, LightGBM, Optuna, SHAP, statsmodels, UMAP, networkx, PyTorch, darts | Free / open-source |
| Dashboards | Streamlit | Free / open-source |
| API | FastAPI + Uvicorn | Free / open-source |
| Website | Next.js / React | Free / open-source |
| Chatbot | Rule-based + keyword retrieval (no LLM) | Free — no API cost |
| Fonts | Google Fonts | Free |
| Containers | Docker | Free |
| Dashboard hosting | Streamlit Community Cloud | Free tier |
| Website hosting | Vercel (Hobby) | Free tier |
| Code + CI | GitHub + GitHub Actions | Free tier |

**Result: fully functional, deployed, and reproducible at $0 — anyone can run or host it for free.**

---

## Final go-live checklist

1. Commit and push all changes (see `DEPLOY.md`).
2. Confirm Vercel + Streamlit auto-redeployed.
3. Smoke test: website EN/FR toggle · predictor · version cards · chatbot; dashboards `?lang=fr` and v1–v3.
4. Add the website URL to the GitHub repo **About** box.

Once these are checked, the project is **done**.
