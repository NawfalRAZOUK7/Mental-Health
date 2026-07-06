# Deployment checklist

Get everything live in this order: **GitHub → Streamlit dashboards → website → wire the URLs**. The FastAPI service is optional (the website works without it — predictions are baked in).

Key fact: the four "versions" are **one** Streamlit app (`src/app.py`) served via `?version=v0…v3`. You deploy the app **once** and link the four query-param URLs.

---

## 0. Push to GitHub

```bash
# from the repo root
git add -A
git commit -m "Deploy-ready: analytics, API, website, chatbot, unified design"
git push origin main
```

Make sure these are committed (they must be, for the website build): `web/data/predictions.json` and `web/public/assets/*.png`. Not committed (correct): `node_modules/`, `web/.next/`, `web/out/`, `.venv/`.

---

## 1. Streamlit dashboards (one app, four versions)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick this repo → branch `main` → **Main file path**: `src/app.py` → **Deploy**.
3. Wait for the build (installs `requirements.txt` — the core deps are enough; v2/v3 read pre-computed CSVs, so no torch/LightGBM needed at runtime).
4. Copy the app URL, e.g. `https://<your-app>.streamlit.app`.

Your four version URLs are then:

| Version | URL |
| --- | --- |
| v0 | `https://<your-app>.streamlit.app/?version=v0` |
| v1 | `https://<your-app>.streamlit.app/?version=v1` |
| v2 | `https://<your-app>.streamlit.app/?version=v2` |
| v3 | `https://<your-app>.streamlit.app/?version=v3` |

> If v2/v3 pages ever error on a missing package, add the extra lines from `requirements-advanced.txt` into `requirements.txt` and redeploy. For a normal run they aren't needed.

---

## 2. Wire the version links into the website

Edit `web/lib/versions.js` and replace each `url: "#"` with the matching URL above:

```js
{ id: "v0", ..., url: "https://<your-app>.streamlit.app/?version=v0" },
{ id: "v1", ..., url: "https://<your-app>.streamlit.app/?version=v1" },
{ id: "v2", ..., url: "https://<your-app>.streamlit.app/?version=v2" },
{ id: "v3", ..., url: "https://<your-app>.streamlit.app/?version=v3" },
```

Commit and push:

```bash
git add web/lib/versions.js && git commit -m "Wire live dashboard URLs" && git push
```

---

## 3. Website (Next.js) — deploy on Vercel (easiest)

1. Go to <https://vercel.com>, sign in with GitHub, **Add New → Project**, import this repo.
2. **Root Directory**: set to `web`.
3. Framework preset auto-detects **Next.js**. Leave build/output defaults.
4. **Deploy**. Copy the resulting URL, e.g. `https://mental-health-viz.vercel.app`.

Every push to `main` auto-redeploys, so step 2's URL wiring is picked up automatically.

**Alternatives**
- **Netlify**: base directory `web`, build `npm run build`, publish `web/out`.
- **GitHub Pages**: build locally with a base path, then push `web/out` to a `gh-pages` branch:
  ```bash
  cd web && npm ci && NEXT_PUBLIC_BASE_PATH=/Mental-Health npm run build
  npx gh-pages -d out            # or push web/out to the gh-pages branch
  ```

---

## 4. Finish the README

Paste the website URL into the **live-demo** badge at the top of `README.md` (replace the placeholder), commit, and push:

```bash
git add README.md && git commit -m "Add live demo URL" && git push
```

Optionally add the site URL to the repo's **About** box on GitHub (top-right "⚙" → Website).

---

## 5. (Optional) Prediction API

The website doesn't need it, but to host the FastAPI `/predict` service publicly:

- **Render**: New → Web Service → this repo → Environment **Docker** → Dockerfile path `api/Dockerfile` → Deploy. Free tier sleeps when idle.
- Or **Railway** / **Fly.io** with the same `api/Dockerfile`.

Then `https://<api-host>/docs` gives the interactive Swagger UI.

---

## Done — quick verification

- [ ] Streamlit app loads; switching `?version=` shows v0/v1/v2/v3.
- [ ] Website loads; **Predict a country** works; version cards open the dashboards in new tabs.
- [ ] Chatbot answers (try "what drives suicide risk?" and a country name).
- [ ] README live-demo badge points to the website.
- [ ] GitHub repo **About** shows the website + topics.
