import type { Metadata, Viewport } from "next";
import { Poppins, Inter } from "next/font/google";
import { AuthProvider } from "@/components/auth-provider";
import { UserSettingsProvider } from "@/hooks/use-settings";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--poppins",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--inter",
  display: "swap",
});


export const metadata: Metadata = {
  title: "Mr.🆖 SpeakAI - AI Text to Speech",
  description:
    "Turn text from documents and images into high-quality audio with one click. Convert articles, stories, and other content into clear speech for easy listening.",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: "/icon.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#000000",
};

const themeScript = `
(function(){
  try {
    var t = localStorage.getItem('speakai-theme') || 'system';
    var d = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme:dark)').matches);
    if (d) document.documentElement.classList.add('dark');
  } catch(e){}
})();
`.trim();

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className={`${poppins.variable} ${inter.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen bg-background font-sans antialiased">
        <AuthProvider>
          <UserSettingsProvider>
            {children}
            <Toaster />
          </UserSettingsProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
