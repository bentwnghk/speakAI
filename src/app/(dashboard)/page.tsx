import { AudioWaveform } from "lucide-react";
import { TtsForm } from "@/components/tts-form";
import { KaraokeSubtitle } from "@/components/karaoke-subtitle";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight flex items-center justify-center gap-2">
          <AudioWaveform className="size-8 text-primary" />
          Mr.🆖 SpeakAI
        </h1>
        <KaraokeSubtitle />
      </div>
      <TtsForm />
    </div>
  );
}
