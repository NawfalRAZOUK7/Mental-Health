// Streaming dashboards for each project version. One Streamlit deployment serves
// every version via ?version=; each button opens the app locked to that version.
// Titles/blurbs/tags are localized in lib/i18n.js by id.
const STREAMLIT = "https://mental-health-razouk.streamlit.app";

export const VERSIONS = [
  { id: "v0", url: `${STREAMLIT}/?version=v0`, accent: "#8a6d3b", tagType: "real" },
  { id: "v1", url: `${STREAMLIT}/?version=v1`, accent: "#1f6f8b", tagType: "real" },
  { id: "v2", url: `${STREAMLIT}/?version=v2`, accent: "#5c6bc0", tagType: "synthetic" },
  { id: "v3", url: `${STREAMLIT}/?version=v3`, accent: "#b0453c", tagType: "synthetic" },
];
