"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { useSession } from "next-auth/react";
import { getDictionary, type Locale, type TranslationKeys } from "@/lib/i18n";

type Theme = "system" | "light" | "dark";

interface SettingsContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: TranslationKeys;
  loading: boolean;
}

const SettingsContext = createContext<SettingsContextType | null>(null);

const THEME_STORAGE_KEY = "speakai-theme";
const LOCALE_STORAGE_KEY = "speakai-locale";

function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyTheme(theme: Theme) {
  if (typeof document === "undefined") return;
  const resolved = theme === "system" ? getSystemTheme() : theme;
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

export function UserSettingsProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("system");
  const [locale, setLocaleState] = useState<Locale>("en");
  const [loading, setLoading] = useState(true);
  const { status } = useSession();
  const hasFetchedRef = useRef(false);

  const t = getDictionary(locale);

  useEffect(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as Theme | null;
    const savedLocale = localStorage.getItem(LOCALE_STORAGE_KEY) as Locale | null;
    if (savedTheme) setThemeState(savedTheme);
    if (savedLocale) setLocaleState(savedLocale);
  }, []);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    if (theme === "system") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      const handler = () => applyTheme("system");
      mq.addEventListener("change", handler);
      return () => mq.removeEventListener("change", handler);
    }
  }, [theme]);

  useEffect(() => {
    if (status === "authenticated" && !hasFetchedRef.current) {
      hasFetchedRef.current = true;
      fetch("/api/user/settings")
        .then((res) => (res.ok ? res.json() : null))
        .then((data: { theme?: string; locale?: string } | null) => {
          if (data?.theme && ["system", "light", "dark"].includes(data.theme)) {
            const localTheme = localStorage.getItem(THEME_STORAGE_KEY) as Theme | null;
            const resolved = localTheme ?? (data.theme as Theme);
            setThemeState(resolved);
            localStorage.setItem(THEME_STORAGE_KEY, resolved);
            applyTheme(resolved);
          }
          if (data?.locale && ["en", "zh-TW"].includes(data.locale)) {
            setLocaleState(data.locale as Locale);
            localStorage.setItem(LOCALE_STORAGE_KEY, data.locale);
          }
        })
        .catch(() => {})
        .finally(() => setLoading(false));
    } else if (status === "unauthenticated") {
      setLoading(false);
    }
  }, [status]);

  const persistSetting = useCallback(
    async (updates: { theme?: Theme; locale?: Locale }) => {
      try {
        await fetch("/api/user/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        });
      } catch {}
    },
    []
  );

  const setTheme = useCallback(
    (newTheme: Theme) => {
      setThemeState(newTheme);
      localStorage.setItem(THEME_STORAGE_KEY, newTheme);
      void persistSetting({ theme: newTheme });
    },
    [persistSetting]
  );

  const setLocale = useCallback(
    (newLocale: Locale) => {
      setLocaleState(newLocale);
      localStorage.setItem(LOCALE_STORAGE_KEY, newLocale);
      document.documentElement.lang = newLocale === "zh-TW" ? "zh-Hant" : "en";
      void persistSetting({ locale: newLocale });
    },
    [persistSetting]
  );

  return (
    <SettingsContext.Provider value={{ theme, setTheme, locale, setLocale, t, loading }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useUserSettings() {
  const context = useContext(SettingsContext);
  if (!context)
    throw new Error("useUserSettings must be used within UserSettingsProvider");
  return context;
}
