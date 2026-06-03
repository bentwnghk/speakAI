import { AudioWaveform } from "lucide-react";
import { TtsForm } from "@/components/tts-form";
import { KaraokeSubtitle } from "@/components/karaoke-subtitle";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <AudioWaveform className="size-6 text-primary" />
          Text to Speech
        </h1>
        <KaraokeSubtitle />
      </div>
      <TtsForm />
    </div>
  );
}
