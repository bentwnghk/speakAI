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
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>0.25x</span>
        <span>1x</span>
        <span>2x</span>
      </div>
    </div>
  );
}
