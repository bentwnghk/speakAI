"use client";

import { Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useInstallPrompt } from "@/hooks/use-install-prompt";
import { useUserSettings } from "@/hooks/use-settings";
import { toast } from "sonner";

export function InstallPrompt() {
  const { showPrompt, promptInstall, dismiss } = useInstallPrompt();
  const { t } = useUserSettings();

  if (!showPrompt) return null;

  const handleInstall = async () => {
    const accepted = await promptInstall();
    if (accepted) {
      toast.success(t.install.installed);
    }
  };

  return (
    <div className="fixed bottom-4 left-4 right-4 z-[100] mx-auto max-w-sm animate-in slide-in-from-bottom-4 fade-in duration-300">
      <div className="flex items-center gap-3 rounded-lg border bg-background p-4 shadow-lg">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-primary/10">
          <Download className="size-5 text-primary" />
        </div>
        <div className="flex-1 space-y-1">
          <p className="text-sm font-medium">{t.install.title}</p>
          <p className="text-xs text-muted-foreground">
            {t.install.description}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button size="sm" onClick={() => void handleInstall()}>
            {t.install.button}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={dismiss}
          >
            <X className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
