// Shared design tokens for the website + chatbot.
// Mirror of src/theme.py (Python surfaces) and .streamlit/config.toml.
// The values are also declared as CSS variables in app/globals.css (:root).
export const theme = {
  color: {
    bg: "#f4efe7",
    bgSoft: "#faf6ef",
    surface: "#ffffff",
    surfaceAlt: "#f7f2ea",
    ink: "#1c1b1a",
    muted: "#6b6460",
    subtle: "#8f877f",
    accent: "#1f6f8b",
    accentDark: "#185a72",
    coral: "#b0453c",
    gold: "#d99a2b",
    green: "#2f8f6b",
    border: "#e7ded1",
    borderStrong: "#d9d2c8",
  },
  // Categorical sequence shared with charts (color = meaning, not rainbow).
  categorical: ["#1f6f8b", "#b0453c", "#d99a2b", "#2f8f6b", "#6b6460", "#7c6f9e"],
  font: {
    ui: "'Space Grotesk', system-ui, sans-serif",
    serif: "'Source Serif 4', Georgia, serif",
  },
  radius: { control: "10px", card: "16px" },
  shadow: "0 12px 30px rgba(28,27,26,.08)",
};
