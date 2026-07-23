import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Mr.🆖 SpeakAI",
    short_name: "Mr.🆖 SpeakAI",
    description:
      "AI text-to-speech with karaoke highlighting & pronunciation assessment.",
    start_url: "/",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#000000",
    icons: [
      {
        src: "/icon.png",
        sizes: "any",
        type: "image/png",
        purpose: "any",
      },
    ],
  };
}
