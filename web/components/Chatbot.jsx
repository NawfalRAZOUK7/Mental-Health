"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { FALLBACK, INTENTS, SUGGESTIONS } from "../lib/knowledge";

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
    let score = 0;
    for (const kw of intent.keywords) if (lower.includes(kw)) score += 1;
    if (score > topScore) {
      topScore = score;
      top = intent;
    }
  }
  return topScore > 0 ? top : null;
}

function countryReply(p) {
  return {
    role: "bot",
    text: `${p.name}: predicted ${p.pred.toFixed(1)} per 100k (90% interval ${p.lower.toFixed(
      1
    )}–${p.upper.toFixed(1)}); observed WHO rate ${p.actual.toFixed(
      1
    )}. Educational estimate, not a clinical assessment.`,
    chips: ["What drives suicide risk?", "How accurate is the model?"],
  };
}

function respond(text, predictions) {
  const intent = bestIntent(text);
  const country = detectCountry(text, predictions);
  if (country && (!intent || intent.id === "predict_help")) return countryReply(country);
  if (intent) return { role: "bot", text: intent.answer, chips: intent.followups || [] };
  if (country) return countryReply(country);
  return { role: "bot", text: FALLBACK, chips: SUGGESTIONS };
}

export default function Chatbot({ predictions }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi! I'm the project guide. Ask me about the findings, the models, or type a country to see its prediction. I only share results — not the raw data.",
      chips: SUGGESTIONS,
    },
  ]);
  const bodyRef = useRef(null);
  const preds = useMemo(() => predictions || [], [predictions]);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [messages, open]);

  function send(text) {
    const q = (text ?? input).trim();
    if (!q) return;
    setMessages((m) => [...m, { role: "user", text: q }, respond(q, preds)]);
    setInput("");
  }

  return (
    <>
      <button className="cb-fab" onClick={() => setOpen((o) => !o)} aria-label="Open project guide">
        {open ? "Close guide" : "Ask the guide"}
      </button>

      {open && (
        <div className="cb-panel" role="dialog" aria-label="Project guide chatbot">
          <div className="cb-head">
            <b>Project guide</b>
            <button className="cb-x" onClick={() => setOpen(false)} aria-label="Close">
              ×
            </button>
          </div>

          <div className="cb-body" ref={bodyRef}>
            {messages.map((m, i) => (
              <div key={i} style={{ display: "contents" }}>
                <div className={`cb-msg ${m.role === "user" ? "cb-user" : "cb-bot"}`}>{m.text}</div>
                {m.chips && m.chips.length > 0 && (
                  <div className="cb-chips">
                    {m.chips.map((c) => (
                      <button key={c} className="cb-chip" onClick={() => send(c)}>
                        {c}
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
              placeholder="Ask about results, or type a country…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              aria-label="Message"
            />
            <button onClick={() => send()} aria-label="Send">
              Send
            </button>
          </div>
          <div className="cb-note">Educational guide · answers from project results, not raw data · no medical advice</div>
        </div>
      )}
    </>
  );
}
