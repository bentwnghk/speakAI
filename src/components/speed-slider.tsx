"use client";

import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Gauge } from "lucide-react";

interface SpeedSliderProps {
  value: number;
  onValueChange: (value: number) => void;
}

export function SpeedSlider({ value, onValueChange }: SpeedSliderProps) {
  return (
    <div className="space-y-3">
      <Label className="flex items-center justify-between">
        <span className="flex items-center gap-2">
          <Gauge className="size-4" />
          Speed
        </span>
        <span className="text-muted-foreground text-sm tabular-nums">
          {value}%
        </span>
      </Label>
      <Slider
        value={[value]}
        min={25}
        max={200}
        step={25}
        onValueChange={(v) => onValueChange(v[0])}
      />
      <div className="relative h-4 text-xs text-muted-foreground">
        <span className="absolute left-0 -translate-x-1/2">0.25x</span>
        <span className="absolute left-[42.857%] -translate-x-1/2">1x</span>
        <span className="absolute right-0 translate-x-1/2">2x</span>
      </div>
    </div>
  );
}
