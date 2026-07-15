export const VOICE_MAP: Record<string, string> = {
  "Female 1": "nova",
  "Male 1": "alloy",
  "Female 2": "phoebe",
  "Male 2": "adam",
  "Female 3": "ava",
  "Male 3": "ollie",
};

export const VOICE_ACCENT: Record<string, string> = {
  "Female 1": "US",
  "Male 1": "US",
  "Female 2": "US",
  "Male 2": "US",
  "Female 3": "US",
  "Male 3": "UK",
};

export const VOICE_OPTIONS = Object.keys(VOICE_MAP);

const OLD_VOICE_MAP: Record<string, string> = {
  "Female 2": "fable",
  "Male 2": "echo",
  "Female 3": "shimmer",
  "Male 3": "onyx",
};

const REVERSE_VOICE_MAP: Record<string, string> = {
  ...Object.fromEntries(Object.entries(VOICE_MAP).map(([k, v]) => [v, k])),
  ...Object.fromEntries(Object.entries(OLD_VOICE_MAP).map(([k, v]) => [v, k])),
};

export function formatVoiceBadge(voice: string): string {
  const displayKey = REVERSE_VOICE_MAP[voice];
  if (displayKey) {
    const name = voice.charAt(0).toUpperCase() + voice.slice(1);
    const gender = displayKey.split(" ")[0].toLowerCase();
    return `${name} (${gender})`;
  }
  return voice;
}

export const SUPPORTED_EXTENSIONS = [
  ".txt",
  ".docx",
  ".pdf",
  ".jpg",
  ".jpeg",
  ".png",
];
