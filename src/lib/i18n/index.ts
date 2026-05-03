import en, { type TranslationKeys } from "./locales/en";
import zhTW from "./locales/zh-TW";

export type Locale = "en" | "zh-TW";

const dictionaries = {
  en,
  "zh-TW": zhTW,
} as const satisfies Record<Locale, TranslationKeys>;

export function getDictionary(locale: Locale): TranslationKeys {
  return dictionaries[locale] ?? dictionaries.en;
}

export type { TranslationKeys };
