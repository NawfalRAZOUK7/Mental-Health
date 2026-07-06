"use client";

import { useMemo, useState } from "react";

export default function Predictor({ predictions, scaleMax }) {
  const sorted = useMemo(
    () => [...predictions].sort((a, b) => a.name.localeCompare(b.name)),
    [predictions]
  );
  const [iso, setIso] = useState("");
  const p = sorted.find((x) => x.iso3 === iso);
  const pct = (v) => Math.max(0, Math.min(100, (v / scaleMax) * 100));

  return (
    <div className="predictor">
      <label htmlFor="country">Choose a country</label>
      <select id="country" value={iso} onChange={(e) => setIso(e.target.value)}>
        <option value="">— select —</option>
        {sorted.map((c) => (
          <option key={c.iso3} value={c.iso3}>
            {c.name}
          </option>
        ))}
      </select>

      {p && (
        <div>
          <div className="bignum">
            <span>predicted </span>
            {p.pred.toFixed(1)}
            <span> per 100k</span>
          </div>
          <div className="rowline">
            <span>
              90% interval:{" "}
              <b>
                {p.lower.toFixed(1)} – {p.upper.toFixed(1)}
              </b>
            </span>
            <span>
              observed (WHO): <b>{p.actual.toFixed(1)}</b>
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
              model prediction
            </span>
            <span>
              <span className="swatch" style={{ background: "var(--accent2)" }} />
              observed rate
            </span>
            <span>
              <span className="swatch" style={{ background: "rgba(31,111,139,.3)" }} />
              90% interval
            </span>
            <span>scale 0–{scaleMax} per 100k</span>
          </div>
          <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 14 }}>
            Educational estimate, not a clinical or individual risk assessment.
          </p>
        </div>
      )}
    </div>
  );
}
