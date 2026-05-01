import { TtsForm } from "@/components/tts-form";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Mr.🆖 SpeakAI</h1>
        <p className="text-muted-foreground">
          Convert text from documents and images into high-quality audio
        </p>
      </div>
      <TtsForm />
    </div>
  );
}
