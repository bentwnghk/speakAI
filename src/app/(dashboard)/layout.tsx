import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { Header } from "@/components/header";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session) {
    redirect("/login");
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 mx-auto w-full max-w-5xl px-4 py-6">
        {children}
      </main>
      <footer className="border-t py-4">
        <p className="text-center text-sm text-muted-foreground">
          Built with ❤️ by Mr.🆖 for students learning English.
        </p>
        <p className="text-center text-xs text-muted-foreground mt-1">
          Powered by <a href="https://api.mr5ai.com" target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">Mr.🆖 AI Hub</a>
        </p>
      </footer>
    </div>
  );
}
