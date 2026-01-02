#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit(
        "Missing core deps. Install with: pip install -r requirements.txt"
    ) from exc

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    ColumnTransformer = None
    LogisticRegression = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "report_latex" / "figures"


def save_fig(fig: go.Figure, name: str) -> Path | None:
    out_path = FIG_DIR / f"{name}.png"
    try:
        fig.write_image(out_path, scale=2)
    except Exception as exc:
        print(f"[warn] PNG export failed for {name}: {exc}")
        return None
    return out_path


def copy_v0_map() -> None:
    src = REPO_ROOT / "v0" / "assets" / "v0_who_global_map_age_std.png"
    dest = FIG_DIR / "fig_v0_who_map.png"
    if src.exists():
        shutil.copy2(src, dest)
        print(f"[ok] Copied {src} -> {dest}")
    else:
        print(f"[warn] Missing v0 map: {src}")


def v1_depression_top10() -> None:
    path = REPO_ROOT / "v1" / "data_clean" / "gbd_depression_dalys_clean.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    df = df[
        (df["cause_name"] == "Depressive disorders")
        & (df["measure_name"] == "DALYs (Disability-Adjusted Life Years)")
        & (df["metric_name"] == "Rate")
        & (df["sex_name"] == "Both")
        & (df["year"] == 2023)
    ].copy()
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    if df.empty:
        print("[warn] v1 depression filter returned empty")
        return
    top = df.sort_values("val", ascending=False).groupby("age_name", as_index=False).head(10)
    fig = px.bar(
        top,
        x="val",
        y="location_name",
        facet_col="age_name",
        orientation="h",
        title="Depressive disorders DALYs rate (Top 10 by age group, 2023)",
    )
    fig.update_layout(margin=dict(l=220, r=40, t=70, b=40), height=620)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("age_name=", "")))
    fig.update_yaxes(matches=None)
    fig.update_xaxes(showticklabels=True)
    out = save_fig(fig, "fig_v1_depression_top10")
    if out:
        print(f"[ok] Wrote {out}")


def v1_relationships_scatter() -> None:
    path = REPO_ROOT / "v1" / "data_clean" / "merged_ml_country.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    numeric_cols = [
        "age_standardized_suicide_rate_2021",
        "gbd_depression_dalys_rate_both",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    group_cols = ["iso3", "location_name", "region_name"]
    df = df[group_cols + numeric_cols].groupby(group_cols, as_index=False).mean()
    fig = px.scatter(
        df,
        x="gbd_depression_dalys_rate_both",
        y="age_standardized_suicide_rate_2021",
        color="region_name",
        hover_name="location_name",
        title="Suicide vs depression (v1, country means)",
    )
    out = save_fig(fig, "fig_v1_relationships_scatter")
    if out:
        print(f"[ok] Wrote {out}")


def v1_who_crude_vs_age_std() -> None:
    path = REPO_ROOT / "v1" / "data_clean" / "who_2021_clean.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    for col in [
        "crude_suicide_rate_2021",
        "age_standardized_suicide_rate_2021",
        "number_suicides_2021",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["crude_suicide_rate_2021", "age_standardized_suicide_rate_2021"])
    if df.empty:
        print("[warn] v1 WHO scatter filter returned empty")
        return
    fig = px.scatter(
        df,
        x="crude_suicide_rate_2021",
        y="age_standardized_suicide_rate_2021",
        color="income_group",
        size="number_suicides_2021",
        hover_name="location_name",
        title="WHO 2021: taux brut vs taux age-standardise",
    )
    if len(df) > 2:
        coeff = np.polyfit(df["crude_suicide_rate_2021"], df["age_standardized_suicide_rate_2021"], 1)
        line_x = np.linspace(
            df["crude_suicide_rate_2021"].min(),
            df["crude_suicide_rate_2021"].max(),
            100,
        )
        line_y = coeff[0] * line_x + coeff[1]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Trend"))
    out = save_fig(fig, "fig_v1_who_crude_vs_age_std")
    if out:
        print(f"[ok] Wrote {out}")


def v1_allcause_trends() -> None:
    path = REPO_ROOT / "v1" / "data_clean" / "context_tables" / "context_allcauses_trend.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["val", "year"])
    df = df[df["metric_name"].isin(["Number", "Rate"])]
    if "measure_name" in df.columns:
        df = df[df["measure_name"].str.contains("DALYs", na=False)]
    if "location_name" in df.columns:
        if "Global" in df["location_name"].unique():
            df = df[df["location_name"] == "Global"]
    if df.empty:
        print("[warn] v1 all-cause trends filter returned empty")
        return
    fig = px.line(
        df,
        x="year",
        y="val",
        color="sex_name",
        facet_col="metric_name",
        markers=True,
        title="Tendances toutes causes (Global, 2021-2023)",
    )
    fig.update_yaxes(matches=None)
    out = save_fig(fig, "fig_v1_allcause_trends")
    if out:
        print(f"[ok] Wrote {out}")


def v1_big_categories_treemap() -> None:
    path = REPO_ROOT / "v1" / "data_clean" / "context_tables" / "context_big_categories_2023.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    if df.empty:
        print("[warn] v1 big categories filter returned empty")
        return
    location = "Global" if "Global" in df["location_name"].unique() else df["location_name"].iloc[0]
    sex = "Both" if "Both" in df["sex_name"].unique() else df["sex_name"].iloc[0]
    age = "All ages" if "All ages" in df["age_name"].unique() else df["age_name"].iloc[0]
    metric = "Rate" if "Rate" in df["metric_name"].unique() else df["metric_name"].iloc[0]
    filtered = df[
        (df["location_name"] == location)
        & (df["sex_name"] == sex)
        & (df["age_name"] == age)
        & (df["metric_name"] == metric)
    ]
    filtered = filtered.groupby("cause_name", as_index=False)["val"].mean()
    val_map = dict(zip(filtered["cause_name"], filtered["val"]))
    all_val = val_map.get("All causes")
    comm_val = val_map.get("Communicable, maternal, neonatal, and nutritional diseases")
    ncd_val = val_map.get("Non-communicable diseases")
    inj_val = val_map.get("Injuries")
    sub_val = val_map.get("Substance use disorders")
    alc_val = val_map.get("Alcohol use disorders")
    drug_val = val_map.get("Drug use disorders")

    rows = []

    def add_leaf(level1: str, level2: str | None, level3: str | None, value: float | None) -> None:
        if value is None or pd.isna(value) or value <= 0:
            return
        rows.append(
            {
                "level0": "All causes",
                "level1": level1,
                "level2": level2 or "",
                "level3": level3 or "",
                "val": value,
            }
        )

    if comm_val is not None:
        add_leaf("Communicable, maternal, neonatal, and nutritional diseases", "", "", float(comm_val))
    if inj_val is not None:
        add_leaf("Injuries", "", "", float(inj_val))

    if ncd_val is not None:
        if sub_val is not None:
            if alc_val is not None:
                add_leaf(
                    "Non-communicable diseases",
                    "Substance use disorders",
                    "Alcohol use disorders",
                    float(alc_val),
                )
            if drug_val is not None:
                add_leaf(
                    "Non-communicable diseases",
                    "Substance use disorders",
                    "Drug use disorders",
                    float(drug_val),
                )
            other_sub = sub_val
            if alc_val is not None:
                other_sub -= alc_val
            if drug_val is not None:
                other_sub -= drug_val
            add_leaf(
                "Non-communicable diseases",
                "Substance use disorders",
                "Other substance use disorders",
                float(other_sub),
            )
            other_ncd = ncd_val - sub_val
            add_leaf(
                "Non-communicable diseases",
                "Other non-communicable diseases",
                "",
                float(other_ncd),
            )
        else:
            add_leaf("Non-communicable diseases", "", "", float(ncd_val))

    if all_val is not None and None not in (comm_val, ncd_val, inj_val):
        other_all = all_val - comm_val - ncd_val - inj_val
        add_leaf("Other causes", "", "", float(other_all))

    tree = pd.DataFrame(rows)
    if tree.empty:
        tree = filtered.rename(columns={"cause_name": "level1"})
        tree["level0"] = "All causes"
        tree["level2"] = ""
        tree["level3"] = ""

    fig = px.treemap(
        tree,
        path=["level0", "level1", "level2", "level3"],
        values="val",
        title=f"{location} | {metric}",
    )
    out = save_fig(fig, "fig_v1_big_categories_treemap")
    if out:
        print(f"[ok] Wrote {out}")


def v2_clusters_scatter() -> None:
    path = REPO_ROOT / "v2" / "data_clean" / "v2_clusters.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    sex_values = df["sex_name"].dropna().unique().tolist()
    sex_pick = "Both" if "Both" in sex_values else (sex_values[0] if sex_values else None)
    age_values = df["age_name"].dropna().unique().tolist()
    age_pick = "All ages" if "All ages" in age_values else (age_values[0] if age_values else None)
    if sex_pick:
        df = df[df["sex_name"] == sex_pick]
    if age_pick:
        df = df[df["age_name"] == age_pick]
    for col in ["suicide_rate", "depression_dalys_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["suicide_rate", "depression_dalys_rate", "cluster_label"])
    if df.empty:
        print("[warn] v2 cluster filter returned empty")
        return
    fig = px.scatter(
        df,
        x="depression_dalys_rate",
        y="suicide_rate",
        color="cluster_label",
        hover_name="location_name",
        title="v2 clustering: suicide vs depression",
    )
    out = save_fig(fig, "fig_v2_clusters_scatter")
    if out:
        print(f"[ok] Wrote {out}")


def v3_calibration_or_hist() -> None:
    path = REPO_ROOT / "v3" / "data_clean" / "v3_features_v1.csv"
    if not path.exists():
        print(f"[warn] Missing {path}")
        return
    df = pd.read_csv(path)
    for col in ["suicide_rate", "depression_dalys_rate", "addiction_death_rate", "selfharm_death_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["suicide_rate"])

    cutoff = 0.7
    threshold = float(df["suicide_rate"].quantile(cutoff))

    if ColumnTransformer is None or LogisticRegression is None:
        fig = px.histogram(
            df,
            x="suicide_rate",
            nbins=30,
            title="v3 risk: distribution du taux de suicide (cutoff 70%)",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        out = save_fig(fig, "fig_v3_calibration")
        if out:
            print(f"[ok] Wrote {out}")
        return

    cat_cols = [c for c in ["region_name", "income_group", "sex_name"] if c in df.columns]
    num_cols = [c for c in ["depression_dalys_rate", "addiction_death_rate", "selfharm_death_rate"] if c in df.columns]
    if not num_cols:
        print("[warn] v3 missing numeric cols for calibration")
        return

    X = df[cat_cols + num_cols]
    y = (df["suicide_rate"] >= threshold).astype(int)
    if y.nunique() < 2 or len(df) < 40:
        fig = px.histogram(
            df,
            x="suicide_rate",
            nbins=30,
            title="v3 risk: distribution du taux de suicide (cutoff 70%)",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        out = save_fig(fig, "fig_v3_calibration")
        if out:
            print(f"[ok] Wrote {out}")
        return

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    rel = pd.DataFrame({"proba": proba, "y": y})
    rel["bin"] = pd.qcut(rel["proba"], q=8, duplicates="drop")
    grouped = rel.groupby("bin", observed=True).agg(
        mean_pred=("proba", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size"),
    )
    grouped = grouped.reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["mean_pred"],
            y=grouped["observed_rate"],
            mode="markers+lines",
            name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal",
        )
    )
    fig.update_layout(
        title="v3 calibration: predicted vs observed (quantile bins)",
        xaxis_title="Predicted probability",
        yaxis_title="Observed rate",
    )
    out = save_fig(fig, "fig_v3_calibration")
    if out:
        print(f"[ok] Wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copy_v0_map()
    v1_who_crude_vs_age_std()
    v1_depression_top10()
    v1_relationships_scatter()
    v1_allcause_trends()
    v1_big_categories_treemap()
    v2_clusters_scatter()
    v3_calibration_or_hist()


if __name__ == "__main__":
    main()
