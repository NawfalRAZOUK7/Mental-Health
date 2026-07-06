// Streaming dashboards for each project version.
// Paste your deployed Streamlit URLs here after deploying (Streamlit Community Cloud).
// Until then, "#" keeps the buttons inert. The same app serves all versions via
// the ?version= query param, so you can point them all at one deployment.
export const VERSIONS = [
  {
    id: "v0",
    title: "v0 · Visual gallery",
    blurb: "Static, high-variety visuals straight from the raw WHO & IHME data.",
    tag: "Real data",
    url: "#", // e.g. "https://your-app.streamlit.app/?version=v0"
    accent: "#8a6d3b",
  },
  {
    id: "v1",
    title: "v1 · Main dashboard",
    blurb: "Real-data BI dashboard, ML baseline, and the leakage-free enriched model.",
    tag: "Real data",
    url: "#",
    accent: "#1f6f8b",
  },
  {
    id: "v2",
    title: "v2 · Advanced analytics",
    blurb: "Methods showcase: clustering, forecasting, graphs, explainability.",
    tag: "Synthetic",
    url: "#",
    accent: "#5c6bc0",
  },
  {
    id: "v3",
    title: "v3 · Risk estimator",
    blurb: "Interactive probability tool with calibration and what-if scenarios.",
    tag: "Synthetic",
    url: "#",
    accent: "#b0453c",
  },
];
