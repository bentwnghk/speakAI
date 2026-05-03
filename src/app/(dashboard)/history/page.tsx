import type { Metadata } from "next";
import { HistoryList } from "@/components/history-list";
import { HistoryPageHeader } from "@/components/history-page-header";

export const metadata: Metadata = {
  title: "History - Mr.🆖 SpeakAI",
};

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <HistoryPageHeader />
      <HistoryList />
    </div>
  );
}
