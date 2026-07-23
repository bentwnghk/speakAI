"use client";

import { History } from "lucide-react";
import { useUserSettings } from "@/hooks/use-settings";

export function HistoryPageHeader() {
  const { t } = useUserSettings();
  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
        <History className="size-6 text-primary" />
        {t.history.title}
      </h1>
      <p className="text-muted-foreground">
        {t.history.description}
      </p>
    </div>
  );
}
