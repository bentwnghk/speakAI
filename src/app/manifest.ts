import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Mr.🆖 SpeakAI - AI Text to Speech",
    short_name: "Mr.🆖 SpeakAI",
    description:
      "Turn text from documents and images into high-quality audio with one click.",
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
