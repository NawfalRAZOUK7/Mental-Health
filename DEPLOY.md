# Deployment checklist

Two free services, each for one job:

- **Streamlit Community Cloud** → the dashboards (`src/app.py`, versions v0–v3).
- **Vercel** → the website (`web/`, Next.js).

The website's cards link out to the Streamlit app. Both auto-deploy from GitHub on every push.

> The four "versions" are **one** Streamlit app served via `?version=v0…v3` — you deploy it **once**.
> The prediction API (`api/`) is optional; the website works without it (predictions are baked in).

---

## 1. Push to GitHub

```bash
git add -A
git commit -m "Deploy-ready"
git push origin main
```

These must be committed for the website build (they are, unless you removed them):
`web/data/predictions.json`, `web/public/assets/*.png`.

---

## 2. Dashboards → Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with GitHub.
2. **New app** → this repo → branch `main` → **Main file**: `src/app.py` → **Deploy**.
3. Copy the URL, e.g. `https://<your-app>.streamlit.app`.

Your version URLs are that URL plus `?version=v0` / `v1` / `v2` / `v3`.

---

## 3. Wire the dashboard links into the site

Edit `web/lib/versions.js` — replace each `url: "#"`:

```js
{ id: "v0", ..., url: "https://<your-app>.streamlit.app/?version=v0" },
{ id: "v1", ..., url: "https://<your-app>.streamlit.app/?version=v1" },
{ id: "v2", ..., url: "https://<your-app>.streamlit.app/?version=v2" },
{ id: "v3", ..., url: "https://<your-app>.streamlit.app/?version=v3" },
```

```bash
git add web/lib/versions.js && git commit -m "Wire live dashboard URLs" && git push
```

---

## 4. Website → Vercel

1. Sign in at <https://vercel.com> with GitHub → **Add New → Project** → import this repo.
2. **Root Directory: `web`** (this is the one setting people miss).
3. Framework auto-detects **Next.js** — leave the defaults → **Deploy**.
4. Copy the URL, e.g. `https://mental-health-viz.vercel.app`. Pushes to `main` redeploy automatically.

---

## 5. Finish

- Paste the website URL into the **live-demo badge** at the top of `README.md`, commit, push.
- Add the same URL to the GitHub repo **About** box (⚙, top-right).

## Verify

- [ ] Streamlit app loads; `?version=` switches v0–v3.
- [ ] Website loads; **Predict a country** works; version cards open the dashboards.
- [ ] Chatbot answers ("what drives suicide risk?", a country name).
- [ ] README badge points to the site.
