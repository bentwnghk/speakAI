"use client";

import { AudioWaveform } from "lucide-react";
import { useUserSettings } from "@/hooks/use-settings";

export function DashboardPageHeader() {
  const { t } = useUserSettings();
  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
        <AudioWaveform className="size-6 text-primary" />
        {t.dashboard.title}
      </h1>
    </div>
  );
}
