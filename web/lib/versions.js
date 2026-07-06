// Streaming dashboards for each project version.
// One Streamlit deployment serves every version via the ?version= query param;
// each button below opens the app locked to that version.
const STREAMLIT = "https://mental-health-razouk.streamlit.app";

export const VERSIONS = [
  {
    id: "v0",
    title: "v0 · Visual gallery",
    blurb: "Static, high-variety visuals straight from the raw WHO & IHME data.",
    tag: "Real data",
    url: `${STREAMLIT}/?version=v0`,
    accent: "#8a6d3b",
  },
  {
    id: "v1",
    title: "v1 · Main dashboard",
    blurb: "Real-data BI dashboard, ML baseline, and the leakage-free enriched model.",
    tag: "Real data",
    url: `${STREAMLIT}/?version=v1`,
    accent: "#1f6f8b",
  },
  {
    id: "v2",
    title: "v2 · Advanced analytics",
    blurb: "Methods showcase: clustering, forecasting, graphs, explainability.",
    tag: "Synthetic",
    url: `${STREAMLIT}/?version=v2`,
    accent: "#5c6bc0",
  },
  {
    id: "v3",
    title: "v3 · Risk estimator",
    blurb: "Interactive probability tool with calibration and what-if scenarios.",
    tag: "Synthetic",
    url: `${STREAMLIT}/?version=v3`,
    accent: "#b0453c",
  },
];
