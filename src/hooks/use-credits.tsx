"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { useSession } from "next-auth/react";

interface CreditsContextType {
  balance: number | null;
  loading: boolean;
  refreshBalance: () => Promise<void>;
}

interface CreditsResponse {
  balance: number;
}

const CreditsContext = createContext<CreditsContextType | null>(null);

export function CreditsProvider({ children }: { children: ReactNode }) {
  const [balance, setBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const { status } = useSession();
  const hasFetchedRef = useRef(false);

  const refreshBalance = useCallback(async () => {
    try {
      const res = await fetch("/api/user/credits");
      if (res.ok) {
        const data = (await res.json()) as CreditsResponse;
        setBalance(data.balance);
      }
    } catch {
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (status === "authenticated" && !hasFetchedRef.current) {
      hasFetchedRef.current = true;
      void refreshBalance();
    }
  }, [status, refreshBalance]);

  return (
    <CreditsContext.Provider value={{ balance, loading, refreshBalance }}>
      {children}
    </CreditsContext.Provider>
  );
}

export function useCredits() {
  const context = useContext(CreditsContext);
  if (!context)
    throw new Error("useCredits must be used within CreditsProvider");
  return context;
}
