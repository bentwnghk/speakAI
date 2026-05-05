import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Mr.\u{1F196} SpeakAI - Turn text into natural-sounding audio with karaoke-style playback",
};

export default async function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (session) {
    redirect("/");
  }

  return <>{children}</>;
}
