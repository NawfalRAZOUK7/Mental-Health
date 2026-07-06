"""Shared design tokens for every Python surface (Streamlit app + report figures).

One source of truth for the "refined warm editorial" look, mirrored in
web/lib/theme.js (website/chatbot) and .streamlit/config.toml (Streamlit chrome).
See DESIGN.md.
"""
from __future__ import annotations

# ---- Color tokens ----
BG = "#f4efe7"
BG_SOFT = "#faf6ef"
SURFACE = "#ffffff"
SURFACE_ALT = "#f7f2ea"
INK = "#1c1b1a"
MUTED = "#6b6460"
SUBTLE = "#8f877f"
ACCENT = "#1f6f8b"          # primary — teal
ACCENT_DARK = "#185a72"
CORAL = "#b0453c"           # secondary / alert — warm coral
GOLD = "#d99a2b"            # sparing highlight
GREEN = "#2f8f6b"           # success
PURPLE = "#7c6f9e"          # extra categorical
BORDER = "#e7ded1"
BORDER_STRONG = "#d9d2c8"

# Categorical sequence shared by Plotly + matplotlib (color = meaning, not rainbow).
CATEGORICAL = [ACCENT, CORAL, GOLD, GREEN, MUTED, PURPLE]
# Sequential ramp for heat/intensity (light cream -> deep teal).
SEQUENTIAL = ["#f4efe7", "#cfe0e2", "#93c0c4", "#4f97a0", "#1f6f8b", "#124a5e"]

FONT_UI = "Space Grotesk, system-ui, sans-serif"
FONT_SERIF = "Source Serif 4, Georgia, serif"


def apply_matplotlib() -> None:
    """Apply the shared style to matplotlib (call before plotting)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BORDER_STRONG,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": BORDER,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "figure.titlesize": 15,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def plotly_template():
    """Return a Plotly template matching the design system."""
    import plotly.graph_objects as go

    return go.layout.Template(
        layout=dict(
            colorway=CATEGORICAL,
            font=dict(family=FONT_UI, color=INK, size=13),
            title=dict(font=dict(family=FONT_SERIF, size=18, color=INK)),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER_STRONG, tickcolor=MUTED),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER_STRONG, tickcolor=MUTED),
            colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=56, r=20, b=44, l=52),
        )
    )


def register_plotly(name: str = "mhv") -> str:
    """Register + set the template as Plotly's default. Returns its name."""
    import plotly.io as pio

    pio.templates[name] = plotly_template()
    pio.templates.default = f"plotly_white+{name}"
    return name
