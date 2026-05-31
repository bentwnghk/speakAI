"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Mic } from "lucide-react";
import { useUserSettings } from "@/hooks/use-settings";
import type { SelectionState } from "@/hooks/use-text-selection";
import type { RefObject } from "react";

interface SelectionPopoverProps {
  selection: SelectionState;
  popupRef: RefObject<HTMLDivElement | null>;
}

export function SelectionPopover({ selection, popupRef }: SelectionPopoverProps) {
  const router = useRouter();
  const { t } = useUserSettings();

  return (
    <div
      ref={popupRef}
      className="selection-popup fixed z-[9999] shadow-md flex gap-0.5 bg-background border rounded-md p-0.5"
      style={{
        left: selection.x,
        top: selection.y,
        transform: selection.above
          ? "translate(-50%, -100%)"
          : "translateX(-50%)",
      }}
    >
      <Button
        size="sm"
        variant="ghost"
        onClick={() => {
          router.push(
            `/assessment?text=${encodeURIComponent(selection.text)}`,
          );
        }}
        onTouchEnd={(e) => {
          e.preventDefault();
          router.push(
            `/assessment?text=${encodeURIComponent(selection.text)}`,
          );
        }}
      >
        <Mic className="h-4 w-4" />
        {t.tts.practiceReading}
      </Button>
    </div>
  );
}
