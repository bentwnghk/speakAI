import { TtsForm } from "@/components/tts-form";
import { KaraokeSubtitle } from "@/components/karaoke-subtitle";
import { DashboardPageHeader } from "@/components/dashboard-page-header";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <DashboardPageHeader />
        <KaraokeSubtitle />
      </div>
      <TtsForm />
    </div>
  );
}
