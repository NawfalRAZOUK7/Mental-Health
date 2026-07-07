# v1.1 plan — optional enhancements

Scope: the backlog items listed in `PROJECT_STATUS.md`. None of these is required for v1.0; this plan turns them into concrete, verifiable work. Same constraints as v1.0: **$0 stack, no paid services, no attribution in commits.**

Guiding rules
- Every model stays **opt-in** behind an env flag so the default pipeline stays fast and dependency-light (pattern already used: `MHV_DEEP=1`).
- Every item ends with a verification step (script runs clean, `ruff check src`, `py_compile`, tests still pass).
- Deep-learning deps (darts, pytorch-lightning) go in a separate `requirements-deep.txt`, never the base install.

---

## Phase A — darts deep N-BEATS / TFT path

**Goal.** Add a darts-based implementation alongside the existing pure-PyTorch N-BEATS, and add a Temporal Fusion Transformer (TFT) option, so forecasts have a second, library-backed cross-check.

**Approach.**
1. Extend `src/v2_dl_forecast.py` with a `backend` switch: `"torch"` (current) or `"darts"`.
2. darts path builds a `TimeSeries` panel from the synthetic/real panel, trains `NBEATSModel` and `TFTModel` (both darts) with fixed epochs and `pl_trainer_kwargs={"enable_progress_bar": True}` so progress is visible.
3. Reuse the existing naive + LightGBM baselines for the comparison table; report MAE/RMSE side by side (torch N-BEATS vs darts N-BEATS vs darts TFT vs baselines).
4. Guard the OpenMP conflict already known from v1.0: keep `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`.
5. Gate the whole thing behind `MHV_DEEP=1` **and** `MHV_DEEP_BACKEND=darts`; if darts/lightning aren't installed, print a clear "install requirements-deep.txt" message and skip (no crash).

**Files.** `src/v2_dl_forecast.py` (main), `requirements-deep.txt` (new: `darts`, `pytorch-lightning`, `torch`), `scripts/run_advanced.py` (wire the flag), `README.md`/`DEPLOY.md` (document the opt-in).

**Dependencies.** darts, pytorch-lightning (both free). Adds ~1–2 GB install → **must** stay out of base and off the Streamlit Cloud deploy (deploy uses precomputed artifacts, not live training).

**Effort.** ~1 day. **Risk:** medium — lightning version pinning can be finicky; mitigate by pinning exact versions and documenting the tested combo.

**Acceptance.** `MHV_DEEP=1 MHV_DEEP_BACKEND=darts python scripts/run_advanced.py` runs end to end, prints a 4-row comparison table, writes a forecast artifact, and the base pipeline still runs with deep deps absent.

---

## Phase B — spatial statistics model

**Goal.** Add a spatial-autocorrelation layer: does a country's suicide rate correlate with its neighbours', beyond shared covariates?

**Approach.**
1. New module `src/17_spatial_model.py`.
2. Build a spatial weights matrix from the existing k-NN country network (`src/15_country_network.py` already computes country similarity) — reuse it as the "neighbour" graph (feature-space adjacency, since we have no shared borders dataset for free).
3. Compute **global Moran's I** (spatial autocorrelation of the residuals from the leakage-free model) and **local Moran's I (LISA)** to flag hot/cold clusters.
4. Optionally fit a spatial lag / spatial error regression (`spreg` from PySAL) and compare fit vs the non-spatial baseline.
5. Report: Moran's I + p-value, list of significant clusters, and whether the spatial term improves out-of-sample error.

**Files.** `src/17_spatial_model.py` (new), `scripts/run_advanced.py` (wire), a small results JSON for the dashboard, optional Streamlit section in `src/app.py` (guarded, English + FR label via `i18n`).

**Dependencies.** `libpysal`, `esda`, `spreg` (PySAL family, all free/open). Base-installable (lightweight), but keep behind `MHV_SPATIAL=1` to keep default fast.

**Effort.** ~1 day. **Risk:** low–medium — the "neighbour" definition is feature-space, not geographic; document that honestly so it isn't oversold as true geographic spatial analysis.

**Acceptance.** Script prints Moran's I with p-value and a cluster list, writes results JSON, `ruff` clean. If a Streamlit panel is added, it renders bilingually.

---

## Phase C — translate dropdown data-values (region / cause / sex)

**Goal.** The remaining untranslated UI: dropdown **option values** (e.g. "Both sexes", region names, cause names) show in source-data language. Add FR labels while keeping the underlying data value unchanged.

**Approach.**
1. Add value-mapping dicts to `src/i18n_fr.py`: `SEX_FR`, `REGION_FR`, `CAUSE_FR`, `AGE_FR` (source value → FR label). Keys = exact source strings.
2. Add a helper in `src/i18n.py`: `label_value(value, category)` → returns FR label when `lang == "fr"`, else the original.
3. Use Streamlit's `format_func` on the relevant `selectbox`/`multiselect` widgets so the **displayed** label is translated but the **returned value** stays the raw data string (no downstream filtering breaks). This is the key trick — filtering logic is untouched.
4. Audit every `st.selectbox`/`st.multiselect` in `src/app.py` that lists data values; apply `format_func`.

**Files.** `src/i18n_fr.py` (new dicts), `src/i18n.py` (helper), `src/app.py` (add `format_func=` to the relevant widgets).

**Dependencies.** none.

**Effort.** ~half a day, mostly careful enumeration + a coverage check. **Risk:** low — but must verify no filter compares against the *translated* string. Mitigate with a grep that every widget with `format_func` still passes raw values to filters.

**Acceptance.** In FR mode, dropdown options read in French; selecting them still filters correctly (data unchanged); a coverage script confirms 0 data-value widgets left without `format_func`; existing 44 tests still pass.

---

## Phase D — real dashboard screenshot (README hero)

**Goal.** Replace any placeholder hero with a real, high-quality screenshot (or short GIF) of the live dashboard + website.

**Approach.**
1. Run the app locally, capture 2–3 clean shots: v2 overview, a chart-guide panel, the website predictor. (This step is on your Mac — the sandbox can't screenshot your screen; I can do it via the desktop tools if you want, or you capture and drop them in `docs/img/`.)
2. Save to `docs/img/` at a consistent width (e.g. 1600px), lightly compressed.
3. Reference them in `README.md` hero + a small "Gallery" section; add FR alt text.
4. Optional: a 6–8s GIF of the EN→FR toggle for the README top.

**Files.** `docs/img/*.png` (new), `README.md`.

**Dependencies.** none (optional `gifski`/`ffmpeg` for the GIF, both free).

**Effort.** ~1–2 hours. **Risk:** none.

**Acceptance.** README renders real screenshots on GitHub; images optimized (<400 KB each); alt text present.

---

## Suggested sequence & effort

1. **C (dropdown i18n)** — fastest, highest visible polish, no heavy deps. ~0.5 day.
2. **D (screenshots)** — quick, improves first impression. ~2 h.
3. **B (spatial model)** — self-contained analytical addition. ~1 day.
4. **A (darts N-BEATS/TFT)** — heaviest deps, do last, keep fully opt-in. ~1 day.

Total ~3 focused days. Each phase is independent and shippable on its own; nothing here blocks the live v1.0.

## Definition of done for v1.1
- All four phases pass their acceptance checks.
- `ruff check src` clean, `py_compile` clean, 44 tests still green.
- Base install and Streamlit Cloud deploy unchanged (deep deps isolated in `requirements-deep.txt`, all new models opt-in).
- New env flags documented in `README.md` + `DEPLOY.md`.
- Commits grouped and clean, **no attribution lines**.
