"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Mic } from "lucide-react";
import { VOICE_MAP, VOICE_OPTIONS } from "@/lib/constants";
import { useUserSettings } from "@/hooks/use-settings";

interface VoiceSelectProps {
  value: string;
  onValueChange: (value: string) => void;
}

export function VoiceSelect({ value, onValueChange }: VoiceSelectProps) {
  const { t } = useUserSettings();

  return (
    <div className="space-y-2">
      <Label className="flex items-center gap-2">
        <Mic className="size-4" />
        {t.tts.voice}
      </Label>
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue placeholder={t.tts.selectVoice} />
        </SelectTrigger>
        <SelectContent>
          {VOICE_OPTIONS.map((voice) => (
            <SelectItem key={voice} value={voice}>
              {VOICE_MAP[voice].charAt(0).toUpperCase()}{VOICE_MAP[voice].slice(1)} ({voice.split(" ")[0].toLowerCase()})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
