"use client";

import { useRef } from "react";
import { motion, useInView } from "motion/react";
import {
  Upload,
  Type,
  Mic,
  Gauge,
  Sparkles,
  Headphones,
  Download,
  History,
  FileText,
  Brain,
  AudioWaveform,
  Rocket,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const easeOut: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const sectionVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.6, ease: easeOut },
  },
};

const heroItemVariants = {
  hidden: { opacity: 0, y: 30, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.7, ease: easeOut },
  },
};

const sectionTitleVariants = {
  hidden: { opacity: 0, x: -30 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
};

const pillVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.4 } },
};

function AnimatedSection({
  children,
  className,
  staggerDelay = 0.12,
}: {
  children: React.ReactNode;
  className?: string;
  staggerDelay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.div
      ref={ref}
      variants={gridContainer(staggerDelay)}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function gridContainer(stagger: number) {
  return {
    hidden: {},
    visible: { transition: { staggerChildren: stagger } },
  };
}

function seededRandom(seed: number) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function generateGalaxyData() {
  const stars: { x: number; y: number; r: number; o: number }[] = [];
  for (let i = 0; i < 120; i++) {
    stars.push({
      x: seededRandom(i * 2) * 1000,
      y: seededRandom(i * 2 + 1) * 1000,
      r: seededRandom(i * 3) * 2.5 + 0.8,
      o: seededRandom(i * 5) * 0.5 + 0.15,
    });
  }
  const lines: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    o: number;
  }[] = [];
  const maxDist = 180;
  for (let i = 0; i < stars.length; i++) {
    for (let j = i + 1; j < stars.length; j++) {
      const dx = stars[i].x - stars[j].x;
      const dy = stars[i].y - stars[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < maxDist) {
        lines.push({
          x1: stars[i].x,
          y1: stars[i].y,
          x2: stars[j].x,
          y2: stars[j].y,
          o: (1 - d / maxDist) * 0.25,
        });
      }
    }
  }
  return { stars, lines };
}

const galaxyData = generateGalaxyData();

function GalaxyBackground() {
  const { stars, lines } = galaxyData;

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <svg
        viewBox="0 0 1000 1000"
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="xMidYMid slice"
      >
        {lines.map((l, i) => (
          <line
            key={`l-${i}`}
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke="currentColor"
            className="text-white/20 dark:text-white/10"
            strokeWidth="0.5"
            opacity={l.o}
          />
        ))}
        {stars.map((s, i) => (
          <circle
            key={`s-${i}`}
            cx={s.x}
            cy={s.y}
            r={s.r}
            className="text-white/60 dark:text-white/40"
            fill="currentColor"
            opacity={s.o}
          />
        ))}
      </svg>
    </div>
  );
}

function WaveDivider({ flip = false }: { flip?: boolean }) {
  return (
    <div className={`w-full overflow-hidden leading-[0] ${flip ? "rotate-180" : ""}`}>
      <svg
        viewBox="0 0 1440 100"
        preserveAspectRatio="none"
        className="relative block w-full h-[60px]"
      >
        <path
          d="M0,40 C360,100 720,0 1080,60 C1260,80 1380,40 1440,50 L1440,100 L0,100 Z"
          className="fill-background"
        />
      </svg>
    </div>
  );
}

const glassBase =
  "bg-white/[0.08] dark:bg-white/[0.04] backdrop-blur-xl border border-white/[0.15] dark:border-white/[0.08] shadow-[inset_0_0_20px_rgba(255,255,255,0.05)]";
const glassCard = `${glassBase} bg-blue-50/[0.06] dark:bg-blue-900/[0.03]`;

function GoogleIcon() {
  return (
    <svg className="mr-2 h-5 w-5" viewBox="0 0 24 24">
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

interface LandingPageProps {
  onSignIn: () => void;
}

export function LandingPage({ onSignIn }: LandingPageProps) {
  const features = [
    {
      icon: Upload,
      key: "upload",
      color: "text-blue-600 dark:text-blue-400",
      title: "Upload Anything",
      desc: "Seamlessly upload PDFs, DOCX, TXT files, or images — the AI extracts text automatically.",
    },
    {
      icon: Type,
      key: "text",
      color: "text-indigo-600 dark:text-indigo-400",
      title: "Type or Paste",
      desc: "Enter text directly with a rich text input. Perfect for articles, stories, or study notes.",
    },
    {
      icon: Mic,
      key: "voices",
      color: "text-violet-600 dark:text-violet-400",
      title: "6 Natural AI Voices",
      desc: "Choose from 6 distinct voices — 3 female and 3 male — for natural, expressive speech.",
    },
    {
      icon: Gauge,
      key: "speed",
      color: "text-cyan-600 dark:text-cyan-400",
      title: "Speed Control",
      desc: "Adjust playback speed to match your listening preference — slow it down or speed it up.",
    },
    {
      icon: Sparkles,
      key: "ai",
      color: "text-purple-600 dark:text-purple-400",
      title: "AI-Powered TTS",
      desc: "Leverage cutting-edge AI models for crystal-clear, natural-sounding speech synthesis.",
    },
    {
      icon: Headphones,
      key: "playback",
      color: "text-teal-600 dark:text-teal-400",
      title: "Instant Playback",
      desc: "Listen to generated audio instantly with a built-in player featuring seek and progress controls.",
    },
    {
      icon: Download,
      key: "download",
      color: "text-rose-600 dark:text-rose-400",
      title: "Download MP3",
      desc: "Download any generated audio as an MP3 file for offline listening on any device.",
    },
    {
      icon: History,
      key: "history",
      color: "text-amber-600 dark:text-amber-400",
      title: "Generation History",
      desc: "Access all your past audio generations anytime with full playback and download support.",
    },
  ];

  const journey = [
    { num: 1, icon: FileText, key: "input", label: "Upload or Type Text" },
    { num: 2, icon: Mic, key: "voice", label: "Choose Voice & Speed" },
    { num: 3, icon: Brain, key: "generate", label: "AI Generates Audio" },
    { num: 4, icon: Headphones, key: "listen", label: "Listen to Audio" },
    { num: 5, icon: Download, key: "download", label: "Download MP3" },
    { num: 6, icon: History, key: "history", label: "Review History" },
  ];

  const capabilities = [
    "PDF Support",
    "DOCX Support",
    "Image OCR",
    "6 AI Voices",
    "Speed Control",
    "Audio Download",
    "Playback Controls",
    "Generation History",
  ];

  return (
    <div className="relative">
      <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <GalaxyBackground />
        <motion.div
          className="relative z-10 container mx-auto px-4 py-20 text-center"
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={heroItemVariants}>
            <div className="mb-6 inline-flex items-center justify-center gap-2 px-5 pt-3 pb-2 text-white/80 text-sm border-t-2 border-blue-400/60">
              <AudioWaveform className="h-4 w-4 text-blue-400" />
              AI-powered text-to-speech, made simple
            </div>
          </motion.div>

          <motion.h1
            variants={heroItemVariants}
            className="text-6xl sm:text-8xl font-extrabold tracking-tighter text-white"
          >
            Mr.
            <span className="inline-block mx-1">&#x1F197;</span>
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-violet-400 bg-clip-text text-transparent">
              SpeakAI
            </span>
          </motion.h1>

          <motion.p
            variants={heroItemVariants}
            className="mt-3 text-2xl sm:text-3xl font-medium italic text-white/60 bg-gradient-to-r from-blue-300/80 via-indigo-300/80 to-violet-300/80 bg-clip-text"
          >
            Turn any text into natural speech — instantly.
          </motion.p>

          <motion.p
            variants={heroItemVariants}
            className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-white/50"
          >
            Upload documents, images, or paste text and let AI transform it into
            high-quality audio. Choose from 6 natural voices, adjust speed, and
            download MP3s for offline listening. The easiest way to listen to
            any content.
          </motion.p>

          <motion.div variants={heroItemVariants} className="mt-10">
            <Button
              size="lg"
              onClick={onSignIn}
              className="h-12 px-8 text-base bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white border-0 shadow-lg shadow-blue-500/25 cursor-pointer"
            >
              <Rocket className="mr-2 h-5 w-5" />
              Get Started Free
            </Button>
          </motion.div>

          <motion.div
            variants={heroItemVariants}
            className="mt-6 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 backdrop-blur-sm text-white/80 text-sm"
          >
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <span>Free to use — just sign in with Google</span>
          </motion.div>
        </motion.div>
      </section>

      <WaveDivider />

      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400 mb-3"
            >
              Features
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Powerful Features
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              Everything you need to convert text into natural, high-quality
              speech — powered by cutting-edge AI.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 max-w-6xl mx-auto">
            {features.map(({ icon: Icon, key, color, title, desc }) => (
              <motion.div
                key={key}
                variants={cardVariants}
                className={`rounded-2xl p-5 ${glassCard} group hover:scale-[1.02] transition-transform duration-300`}
              >
                <div
                  className={`mb-3 p-2.5 rounded-xl bg-white/10 inline-block ${color}`}
                >
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="font-semibold text-base mb-1">{title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {desc}
                </p>
              </motion.div>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="py-20 px-4">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400 mb-3"
            >
              How It Works
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Your Audio Journey
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <AnimatedSection
            className="max-w-3xl mx-auto space-y-0"
            staggerDelay={0.1}
          >
            {journey.map(({ num, icon: Icon, key, label }, idx) => (
              <motion.div
                key={key}
                variants={cardVariants}
                className="relative flex items-center gap-4"
              >
                <div className="relative flex flex-col items-center">
                  <div
                    className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold text-white shrink-0 ${
                      idx === 0
                        ? "bg-gradient-to-br from-blue-500 to-indigo-500"
                        : "bg-gradient-to-br from-slate-500 to-slate-600"
                    }`}
                  >
                    {num.toString().padStart(2, "0")}
                  </div>
                  {idx < journey.length - 1 && (
                    <div className="w-0.5 h-12 bg-gradient-to-b from-border to-transparent" />
                  )}
                </div>
                <div
                  className={`flex items-center gap-3 ${
                    idx < journey.length - 1 ? "pb-12" : ""
                  }`}
                >
                  <div className="p-2 rounded-lg bg-muted">
                    <Icon className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <span className="font-medium text-base">{label}</span>
                </div>
              </motion.div>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400 mb-3"
            >
              Capabilities
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              What You Get
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <AnimatedSection className="flex flex-wrap justify-center gap-3 max-w-3xl mx-auto">
            {capabilities.map((capability) => (
              <motion.span
                key={capability}
                variants={pillVariants}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium bg-white/60 dark:bg-white/10 border border-white/30 dark:border-white/10 backdrop-blur-sm"
              >
                <CheckCircle2 className="h-3.5 w-3.5 text-blue-500" />
                {capability}
              </motion.span>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="relative py-24 px-4 overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <GalaxyBackground />
        <div className="relative z-10 container mx-auto text-center">
          <AnimatedSection>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white mb-4"
            >
              Ready to Listen?
            </motion.h2>
            <motion.p
              variants={sectionTitleVariants}
              className="text-white/50 mb-8 max-w-xl mx-auto"
            >
              Turn any text into natural speech in seconds. Sign in and start
              generating audio for free.
            </motion.p>
            <motion.div variants={heroItemVariants}>
              <Button
                size="lg"
                onClick={onSignIn}
                className="h-12 px-8 text-base bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white border-0 shadow-lg shadow-blue-500/25 cursor-pointer"
              >
                <GoogleIcon />
                Sign in with Google
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>
    </div>
  );
}
