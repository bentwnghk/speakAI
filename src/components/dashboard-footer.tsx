"use client";

import { useUserSettings } from "@/hooks/use-settings";

export function DashboardFooter({ t: serverT }: { t: ReturnType<typeof import("@/lib/i18n")["getDictionary"]> }) {
  const { t } = useUserSettings();
  const dict = t ?? serverT;

  return (
    <footer className="border-t py-4">
      <p className="text-center text-sm text-muted-foreground">
        {dict.dashboard.footerBuiltBy}
      </p>
      <p className="text-center text-xs text-muted-foreground mt-1">
        {dict.dashboard.footerPoweredBy} <a href="https://api.mr5ai.com" target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">Mr.🆖 AI Hub</a>
      </p>
    </footer>
  );
}
