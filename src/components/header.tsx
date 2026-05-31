"use client";

import { useState } from "react";
import Link from "next/link";
import { useSession, signOut } from "next-auth/react";
import {
  AudioWaveform,
  Coins,
  History,
  LogOut,
  Mic,
  Settings,
  ShieldCheck,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useCredits } from "@/hooks/use-credits";
import { useUserSettings } from "@/hooks/use-settings";
import { SettingsDialog } from "@/components/settings-dialog";

export function Header() {
  const { data: session } = useSession();
  const { balance } = useCredits();
  const { t } = useUserSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <>
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <AudioWaveform className="size-6 text-primary" />
            <span>{t.common.appName}</span>
          </Link>

          <nav className="flex items-center gap-2">
            {session && (
              <>
                <Link href="/assessment">
                  <Button variant="ghost" size="sm">
                    <Mic className="size-4" />
                    {t.header.assessment}
                  </Button>
                </Link>
                <Link href="/history">
                  <Button variant="ghost" size="sm">
                    <History className="size-4" />
                    {t.header.history}
                  </Button>
                </Link>
                <Link href="/credits">
                  <Button variant="ghost" size="sm">
                    <Coins className="size-4" />
                    {balance !== null ? balance.toFixed(2) : "..."}
                  </Button>
                </Link>
              </>
            )}

            {session?.user ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="rounded-full">
                    <Avatar className="size-8">
                      <AvatarImage
                        src={session.user.image || ""}
                        alt={session.user.name || ""}
                      />
                      <AvatarFallback>
                        <User className="size-4" />
                      </AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium">{session.user.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {session.user.email}
                      </p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {session.user.isAdmin && (
                    <DropdownMenuItem asChild>
                      <Link href="/admin">
                        <ShieldCheck className="size-4" />
                        {t.admin.dashboard}
                      </Link>
                    </DropdownMenuItem>
                  )}
                  {session.user.isAdmin && <DropdownMenuSeparator />}
                  <DropdownMenuItem onClick={() => setSettingsOpen(true)}>
                    <Settings className="size-4" />
                    {t.settings.title}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => void signOut()}>
                    <LogOut className="size-4" />
                    {t.common.signOut}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Link href="/login">
                <Button size="sm">{t.common.signIn}</Button>
              </Link>
            )}
          </nav>
        </div>
      </header>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
}
