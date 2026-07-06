import Predictor from "../components/Predictor";
import Chatbot from "../components/Chatbot";
import { VERSIONS } from "../lib/versions";
import predictions from "../data/predictions.json";

const REPO_URL = "https://github.com/NawfalRAZOUK7/Mental-Health";
const BP = process.env.NEXT_PUBLIC_BASE_PATH || "";
const scaleMax = Math.ceil(Math.max(...predictions.map((p) => p.upper)) / 5) * 5;

const FIGURES = [
  ["fig_v1_shap_summary.png", "SHAP — low life expectancy and high alcohol push predicted risk up."],
  ["fig_v1_umap_countries.png", "UMAP — countries mapped by mental-health profile, with a suicide-rate gradient."],
  ["fig_v1_country_network.png", "Similarity network — 10 communities (modularity 0.72) with bridge countries."],
  ["fig_v1_source_agreement.png", "Cross-source check — WHO vs IHME agreement and measurement difference."],
];

export default function Home() {
  return (
    <main>
      <div className="disclaimer">
        ⚠️ Educational project — not a clinical tool, not medical advice. National, aggregate estimates only.
      </div>

      <header className="hero">
        <div className="wrap">
          <div className="eyebrow">WHO 2021 · IHME GBD 2023 · World Bank</div>
          <h1>Turning global mental-health data into clear, honest insight.</h1>
          <p className="lede">
            A full-stack analytics project: reproducible pipelines, leakage-free machine learning,
            data mining, deep-learning forecasting, and a live predictor of national suicide rates —
            built for transparency, not hype.
          </p>
          <div className="cta">
            <a className="btn" href="#predict">Try the predictor</a>
            <a className="btn ghost" href="#versions">Explore the dashboards</a>
            <a className="btn ghost" href={REPO_URL} target="_blank" rel="noopener noreferrer">
              Code on GitHub
            </a>
          </div>
          <div className="tags">
            {["scikit-learn", "LightGBM · Optuna", "SHAP", "conformal intervals", "UMAP · networkx", "N-BEATS", "FastAPI · Docker", "Next.js"].map((t) => (
              <span className="tag" key={t}>{t}</span>
            ))}
          </div>
        </div>
      </header>

      <section>
        <div className="wrap">
          <h2 className="section-title">What the data actually says</h2>
          <p className="section-sub">Findings from the real WHO + IHME + World Bank data (183 countries).</p>
          <div className="grid cols-3">
            <div className="card"><div className="stat">r = 0.84<small>WHO and IHME agree on measuring suicide / self-harm (Pearson)</small></div></div>
            <div className="card"><div className="stat">0.75 → 0.19<small>Cross-validated R² drops once the circular self-harm feature is removed — most apparent skill was leakage</small></div></div>
            <div className="card"><div className="stat">Life expectancy &amp; alcohol<small>Strongest independent drivers of the national suicide rate (SHAP)</small></div></div>
          </div>
        </div>
      </section>

      <section id="predict" style={{ background: "var(--bg2)" }}>
        <div className="wrap">
          <h2 className="section-title">Predict a country</h2>
          <p className="section-sub">
            A leakage-free model estimates the age-standardized suicide rate (per 100k) with a 90%
            conformal interval. Runs entirely in your browser — predictions are baked in from the trained model.
          </p>
          <Predictor predictions={predictions} scaleMax={scaleMax} />
        </div>
      </section>

      <section id="versions">
        <div className="wrap">
          <h2 className="section-title">Explore every version</h2>
          <p className="section-sub">
            Four progressive dashboards, from static visuals to an interactive risk estimator. Each opens in a new tab.
          </p>
          <div className="grid cols-4">
            {VERSIONS.map((v) => (
              <a
                key={v.id}
                className="card vcard"
                href={v.url}
                target="_blank"
                rel="noopener noreferrer"
                style={{ borderTopColor: v.accent }}
              >
                <span className="vtag">{v.tag}</span>
                <h3 style={{ margin: "2px 0" }}>{v.title}</h3>
                <p style={{ color: "var(--muted)", fontSize: 14, margin: 0 }}>{v.blurb}</p>
                <span className="vlink" style={{ color: v.accent }}>Open dashboard →</span>
              </a>
            ))}
          </div>
          <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 14 }}>
            Dashboard links are configured in <code>web/lib/versions.js</code> — paste your deployed Streamlit URLs there.
          </p>
        </div>
      </section>

      <section style={{ background: "var(--bg2)" }}>
        <div className="wrap">
          <h2 className="section-title">Under the hood</h2>
          <p className="section-sub">Modern, honest methods — with baselines kept visible so nothing is oversold.</p>
          <div className="grid cols-2">
            {FIGURES.map(([file, caption]) => (
              <figure className="card" key={file}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${BP}/assets/${file}`} alt={caption} />
                <figcaption>{caption}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2 className="section-title">How it's built</h2>
          <div className="grid cols-3">
            <div className="card"><h3 style={{ margin: "0 0 8px" }}>Machine learning</h3><ul className="clean"><li>LightGBM + Optuna, nested CV</li><li>SHAP explainability</li><li>Conformal prediction intervals</li><li>Hierarchical mixed-effects (ICC)</li></ul></div>
            <div className="card"><h3 style={{ margin: "0 0 8px" }}>Data mining</h3><ul className="clean"><li>UMAP embedding</li><li>Country similarity network</li><li>Subgroup discovery</li><li>FP-Growth association rules</li></ul></div>
            <div className="card"><h3 style={{ margin: "0 0 8px" }}>Engineering</h3><ul className="clean"><li>Reproducible pipelines</li><li>FastAPI prediction service</li><li>Docker + docker-compose</li><li>CI, tests, LaTeX report</li></ul></div>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          <p>
            <b>Mental Health Viz</b> — educational analytics on public data. Sources: WHO suicide
            statistics (2021), IHME Global Burden of Disease (2023), World Bank Open Data.
          </p>
          <p>
            Not medical advice. If you or someone you know is struggling, please contact a local health
            professional or crisis line. Code: MIT · Content: CC BY 4.0.
          </p>
          <p><a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub repository</a></p>
        </div>
      </footer>

      <Chatbot predictions={predictions} />
    </main>
  );
}
