import type { Metadata } from "next";
import { HistoryList } from "@/components/history-list";
import { AudioWaveform } from "lucide-react";

export const metadata: Metadata = {
  title: "History - Mr.🆖 SpeakAI",
};

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <AudioWaveform className="size-6" />
          Audio History
        </h1>
        <p className="text-muted-foreground">
          View and manage your previously generated audio
        </p>
      </div>
      <HistoryList />
    </div>
  );
}
