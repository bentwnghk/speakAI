import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { Header } from "@/components/header";
import { db } from "@/lib/db";
import { userSettings } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { getDictionary, type Locale } from "@/lib/i18n";
import { DashboardFooter } from "@/components/dashboard-footer";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session) {
    redirect("/login");
  }

  let locale: Locale = "en";
  try {
    const rows = await db
      .select({ locale: userSettings.locale })
      .from(userSettings)
      .where(eq(userSettings.userId, session.user.id))
      .limit(1);
    if (rows.length > 0 && rows[0].locale) {
      locale = rows[0].locale as Locale;
    }
  } catch {}

  const t = getDictionary(locale);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 mx-auto w-full max-w-5xl px-4 py-6">
        {children}
      </main>
      <DashboardFooter t={t} />
    </div>
  );
}
