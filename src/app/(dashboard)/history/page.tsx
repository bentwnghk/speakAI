import type { Metadata } from "next";
import { HistoryPageHeader } from "@/components/history-page-header";
import { HistoryTabs } from "@/components/history-tabs";

export const metadata: Metadata = {
  title: "History - Mr.\u{1F196} SpeakAI",
};

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <HistoryPageHeader />
      <HistoryTabs />
    </div>
  );
}
