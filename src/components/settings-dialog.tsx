"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Settings, SunMoon, Languages } from "lucide-react";
import { Label } from "@/components/ui/label";
import { useUserSettings } from "@/hooks/use-settings";
import type { Locale } from "@/lib/i18n";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { theme, setTheme, locale, setLocale, t } = useUserSettings();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Settings className="size-5" />
              {t.settings.title}
            </DialogTitle>
          <DialogDescription>
            {t.settings.theme} & {t.settings.language}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              <SunMoon className="size-4" />
              {t.settings.theme}
            </Label>
            <Select value={theme} onValueChange={(v) => setTheme(v as typeof theme)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="system">{t.settings.themeSystem}</SelectItem>
                <SelectItem value="light">{t.settings.themeLight}</SelectItem>
                <SelectItem value="dark">{t.settings.themeDark}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              <Languages className="size-4" />
              {t.settings.language}
            </Label>
            <Select
              value={locale}
              onValueChange={(v) => setLocale(v as Locale)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">{t.settings.langEn}</SelectItem>
                <SelectItem value="zh-TW">{t.settings.langZhTw}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
