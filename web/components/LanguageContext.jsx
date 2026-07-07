"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { DEFAULT_LANG, LANGS, makeT } from "../lib/i18n";

const LangContext = createContext({ lang: DEFAULT_LANG, setLang: () => {}, t: (k) => k });

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState(DEFAULT_LANG);

  // Restore saved choice on mount (default stays EN if none).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("mhv_lang");
      if (saved && LANGS.includes(saved)) setLang(saved);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("mhv_lang", lang);
      document.documentElement.lang = lang;
    } catch {
      /* ignore */
    }
  }, [lang]);

  const value = useMemo(() => ({ lang, setLang, t: makeT(lang) }), [lang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  return useContext(LangContext);
}
