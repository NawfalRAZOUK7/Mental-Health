"use client";

import { useMemo, useState } from "react";
import { useLang } from "./LanguageContext";

export default function Predictor({ predictions, scaleMax }) {
  const { t } = useLang();
  const sorted = useMemo(
    () => [...predictions].sort((a, b) => a.name.localeCompare(b.name)),
    [predictions]
  );
  const [iso, setIso] = useState("");
  const p = sorted.find((x) => x.iso3 === iso);
  const pct = (v) => Math.max(0, Math.min(100, (v / scaleMax) * 100));

  return (
    <div className="predictor">
      <label htmlFor="country">{t("predictor.choose")}</label>
      <select id="country" value={iso} onChange={(e) => setIso(e.target.value)}>
        <option value="">{t("predictor.select")}</option>
        {sorted.map((c) => (
          <option key={c.iso3} value={c.iso3}>
            {c.name}
          </option>
        ))}
      </select>

      {p && (
        <div>
          <div className="bignum">
            <span>{t("predictor.predicted")} </span>
            {p.pred.toFixed(1)}
            <span> {t("predictor.per100k")}</span>
          </div>
          <div className="rowline">
            <span>
              {t("predictor.interval")}{" "}
              <b>
                {p.lower.toFixed(1)} – {p.upper.toFixed(1)}
              </b>
            </span>
            <span>
              {t("predictor.observed")} <b>{p.actual.toFixed(1)}</b>
            </span>
          </div>
          <div className="bar">
            <div
              className="band"
              style={{ left: `${pct(p.lower)}%`, width: `${pct(p.upper) - pct(p.lower)}%` }}
            />
            <div className="mark" style={{ left: `${pct(p.pred)}%`, background: "var(--accent)" }} />
            <div
              className="mark"
              style={{ left: `${pct(p.actual)}%`, background: "var(--accent2)" }}
            />
          </div>
          <div className="legend">
            <span>
              <span className="swatch" style={{ background: "var(--accent)" }} />
              {t("predictor.legPred")}
            </span>
            <span>
              <span className="swatch" style={{ background: "var(--accent2)" }} />
              {t("predictor.legObs")}
            </span>
            <span>
              <span className="swatch" style={{ background: "rgba(31,111,139,.3)" }} />
              {t("predictor.legInt")}
            </span>
            <span>
              {t("predictor.scale")} 0–{scaleMax}
            </span>
          </div>
          <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 14 }}>{t("predictor.note")}</p>
        </div>
      )}
    </div>
  );
}
