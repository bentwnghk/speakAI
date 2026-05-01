import type { Metadata } from "next";
import { HistoryList } from "@/components/history-list";

export const metadata: Metadata = {
  title: "History - SpeakAI",
};

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Audio History</h1>
        <p className="text-muted-foreground">
          View and manage your previously generated audio
        </p>
      </div>
      <HistoryList />
    </div>
  );
}
