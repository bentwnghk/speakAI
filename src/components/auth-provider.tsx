"use client";

import { SessionProvider } from "next-auth/react";
import { type ReactNode } from "react";
import { CreditsProvider } from "@/hooks/use-credits";

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <SessionProvider>
      <CreditsProvider>{children}</CreditsProvider>
    </SessionProvider>
  );
}
