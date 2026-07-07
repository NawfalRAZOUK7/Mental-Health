"use client";

import Chatbot from "./Chatbot";
import { LanguageProvider, useLang } from "./LanguageContext";
import Predictor from "./Predictor";
import { LANGS } from "../lib/i18n";
import { VERSIONS } from "../lib/versions";

const REPO_URL = "https://github.com/NawfalRAZOUK7/Mental-Health";
const BP = process.env.NEXT_PUBLIC_BASE_PATH || "";

const FIGURES = [
  ["fig_v1_shap_summary.png", "fig.shap"],
  ["fig_v1_umap_countries.png", "fig.umap"],
  ["fig_v1_country_network.png", "fig.network"],
  ["fig_v1_source_agreement.png", "fig.agreement"],
];

function LangToggle() {
  const { lang, setLang } = useLang();
  return (
    <div className="lang-toggle" role="group" aria-label="Language">
      {LANGS.map((l) => (
        <button
          key={l}
          className={l === lang ? "on" : ""}
          onClick={() => setLang(l)}
          aria-pressed={l === lang}
        >
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function Site({ predictions, scaleMax }) {
  const { t } = useLang();
  return (
    <main>
      <div className="disclaimer">⚠️ {t("disclaimer")}</div>

      <header className="hero">
        <div className="wrap">
          <div className="topbar">
            <span className="eyebrow">{t("hero.eyebrow")}</span>
            <LangToggle />
          </div>
          <h1>{t("hero.h1")}</h1>
          <p className="lede">{t("hero.lede")}</p>
          <div className="cta">
            <a className="btn" href="#predict">{t("nav.try")}</a>
            <a className="btn ghost" href="#versions">{t("nav.explore")}</a>
            <a className="btn ghost" href={REPO_URL} target="_blank" rel="noopener noreferrer">
              {t("nav.code")}
            </a>
          </div>
          <div className="tags">
            {["scikit-learn", "LightGBM · Optuna", "SHAP", "conformal intervals", "UMAP · networkx", "N-BEATS", "FastAPI · Docker", "Next.js"].map((x) => (
              <span className="tag" key={x}>{x}</span>
            ))}
          </div>
        </div>
      </header>

      <section>
        <div className="wrap">
          <h2 className="section-title">{t("findings.title")}</h2>
          <p className="section-sub">{t("findings.sub")}</p>
          <div className="grid cols-3">
            <div className="card"><div className="stat">{t("findings.s1.v")}<small>{t("findings.s1.l")}</small></div></div>
            <div className="card"><div className="stat">{t("findings.s2.v")}<small>{t("findings.s2.l")}</small></div></div>
            <div className="card"><div className="stat">{t("findings.s3.v")}<small>{t("findings.s3.l")}</small></div></div>
          </div>
        </div>
      </section>

      <section id="predict" style={{ background: "var(--bg2)" }}>
        <div className="wrap">
          <h2 className="section-title">{t("predict.title")}</h2>
          <p className="section-sub">{t("predict.sub")}</p>
          <Predictor predictions={predictions} scaleMax={scaleMax} />
        </div>
      </section>

      <section id="versions">
        <div className="wrap">
          <h2 className="section-title">{t("versions.title")}</h2>
          <p className="section-sub">{t("versions.sub")}</p>
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
                <span className="vtag">{t(v.tagType === "real" ? "versions.tagReal" : "versions.tagSynthetic")}</span>
                <h3 style={{ margin: "2px 0" }}>{t(`versions.${v.id}.title`)}</h3>
                <p style={{ color: "var(--muted)", fontSize: 14, margin: 0 }}>{t(`versions.${v.id}.blurb`)}</p>
                <span className="vlink" style={{ color: v.accent }}>{t("versions.open")}</span>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section style={{ background: "var(--bg2)" }}>
        <div className="wrap">
          <h2 className="section-title">{t("methods.title")}</h2>
          <p className="section-sub">{t("methods.sub")}</p>
          <div className="grid cols-2">
            {FIGURES.map(([file, key]) => (
              <figure className="card" key={file}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${BP}/assets/${file}`} alt={t(key)} />
                <figcaption>{t(key)}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2 className="section-title">{t("built.title")}</h2>
          <div className="grid cols-3">
            {[["built.ml", "built.mlItems"], ["built.dm", "built.dmItems"], ["built.eng", "built.engItems"]].map(([h, items]) => (
              <div className="card" key={h}>
                <h3 style={{ margin: "0 0 8px" }}>{t(h)}</h3>
                <ul className="clean">
                  {t(items).map((it) => <li key={it}>{it}</li>)}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          <p><b>Mental Health Viz</b> — {t("footer.p1").replace("Mental Health Viz — ", "")}</p>
          <p>{t("footer.p2")}</p>
          <p><a href={REPO_URL} target="_blank" rel="noopener noreferrer">{t("footer.github")}</a></p>
        </div>
      </footer>

      <Chatbot predictions={predictions} />
    </main>
  );
}

export default function SiteClient({ predictions, scaleMax }) {
  return (
    <LanguageProvider>
      <Site predictions={predictions} scaleMax={scaleMax} />
    </LanguageProvider>
  );
}
