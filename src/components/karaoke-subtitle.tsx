"use client";

import { motion } from "motion/react";
import { useUserSettings } from "@/hooks/use-settings";

export function KaraokeSubtitle() {
  const { t } = useUserSettings();
  const words = t.dashboard.subtitle;

  return (
    <p className="text-muted-foreground flex flex-wrap items-center gap-x-1.5">
      {words.map((word, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0.25 }}
          animate={{ opacity: [0.25, 1, 1, 0.6] }}
          transition={{
            duration: words.length * 0.35,
            repeat: Infinity,
            repeatDelay: 1,
            delay: i * 0.35,
            ease: "easeInOut",
          }}
          className={word === "—" ? "font-bold text-primary" : ""}
        >
          {word}
        </motion.span>
      ))}
    </p>
  );
}
