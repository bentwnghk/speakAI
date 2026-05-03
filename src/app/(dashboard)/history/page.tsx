import type { Metadata } from "next";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { userSettings } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { getDictionary, type Locale } from "@/lib/i18n";
import { HistoryList } from "@/components/history-list";
import { AudioWaveform } from "lucide-react";

export const metadata: Metadata = {
  title: "History - Mr.🆖 SpeakAI",
};

export default async function HistoryPage() {
  const session = await auth();

  let locale: Locale = "en";
  try {
    if (session?.user?.id) {
      const rows = await db
        .select({ locale: userSettings.locale })
        .from(userSettings)
        .where(eq(userSettings.userId, session.user.id))
        .limit(1);
      if (rows.length > 0 && rows[0].locale) {
        locale = rows[0].locale as Locale;
      }
    }
  } catch {}

  const t = getDictionary(locale);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <AudioWaveform className="size-6" />
          {t.history.title}
        </h1>
        <p className="text-muted-foreground">
          {t.history.description}
        </p>
      </div>
      <HistoryList />
    </div>
  );
}
