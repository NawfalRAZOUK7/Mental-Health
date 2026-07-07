"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CHAT, CHIP_LABELS, INTENTS, SUGGESTIONS } from "../lib/knowledge";
import { useLang } from "./LanguageContext";

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function detectCountry(text, predictions) {
  const lower = text.toLowerCase();
  let best = null;
  for (const p of predictions) {
    const name = p.name.toLowerCase();
    const nameRe = new RegExp(`\\b${escapeRe(name)}\\b`, "i");
    const isoRe = new RegExp(`\\b${escapeRe(p.iso3)}\\b`);
    if (nameRe.test(lower) || isoRe.test(text)) {
      if (!best || name.length > best.name.toLowerCase().length) best = p;
    }
  }
  return best;
}

function bestIntent(text) {
  const lower = text.toLowerCase();
  let top = null;
  let topScore = 0;
  for (const intent of INTENTS) {
    const kws = [...(intent.keywords.en || []), ...(intent.keywords.fr || [])];
    let score = 0;
    for (const kw of kws) if (lower.includes(kw)) score += 1;
    if (score > topScore) {
      topScore = score;
      top = intent;
    }
  }
  return topScore > 0 ? top : null;
}

function suggestionLabel(id, lang) {
  const s = CHIP_LABELS[id];
  return s ? s[lang] : id;
}

export default function Chatbot({ predictions }) {
  const { lang } = useLang();
  const c = CHAT[lang] || CHAT.en;
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const bodyRef = useRef(null);
  const preds = useMemo(() => predictions || [], [predictions]);

  // Reset the greeting + suggestions whenever the language changes.
  useEffect(() => {
    setMessages([{ role: "bot", text: c.greeting, chips: SUGGESTIONS }]);
  }, [lang, c.greeting]);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [messages, open]);

  function respond(text) {
    const intent = bestIntent(text);
    const country = detectCountry(text, preds);
    if (country && (!intent || intent.id === "predict")) {
      return { role: "bot", text: c.country(country), chips: ["drivers", "accuracy"] };
    }
    if (intent) return { role: "bot", text: intent.answer[lang], chips: intent.followups || [] };
    if (country) return { role: "bot", text: c.country(country), chips: ["drivers", "accuracy"] };
    return { role: "bot", text: c.fallback, chips: SUGGESTIONS };
  }

  function send(text) {
    const q = (text ?? input).trim();
    if (!q) return;
    setMessages((m) => [...m, { role: "user", text: q }, respond(q)]);
    setInput("");
  }

  return (
    <>
      <button className="cb-fab" onClick={() => setOpen((o) => !o)} aria-label={c.open}>
        {open ? c.close : c.open}
      </button>

      {open && (
        <div className="cb-panel" role="dialog" aria-label={c.header}>
          <div className="cb-head">
            <b>{c.header}</b>
            <button className="cb-x" onClick={() => setOpen(false)} aria-label={c.close}>
              ×
            </button>
          </div>

          <div className="cb-body" ref={bodyRef}>
            {messages.map((m, i) => (
              <div key={i} style={{ display: "contents" }}>
                <div className={`cb-msg ${m.role === "user" ? "cb-user" : "cb-bot"}`}>{m.text}</div>
                {m.chips && m.chips.length > 0 && (
                  <div className="cb-chips">
                    {m.chips.map((id) => (
                      <button key={id} className="cb-chip" onClick={() => send(suggestionLabel(id, lang))}>
                        {suggestionLabel(id, lang)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="cb-foot">
            <input
              value={input}
              placeholder={c.placeholder}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              aria-label={c.placeholder}
            />
            <button onClick={() => send()} aria-label={c.send}>
              {c.send}
            </button>
          </div>
          <div className="cb-note">{c.note}</div>
        </div>
      )}
    </>
  );
}
