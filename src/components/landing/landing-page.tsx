"use client";

import { useRef, useEffect, useState } from "react";
import { motion, useInView } from "motion/react";
import {
  Upload,
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
  Captions,
  BookOpen,
  Pencil,
  MessageCircle,
  BarChart3,
  Ear,
  Languages,
  Target,
  Trophy,
  Activity,
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

function ScoreRing({ score, size = 120 }: { score: number; size?: number }) {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;
  const center = size / 2;

  return (
    <svg width={size} height={size} className="shrink-0">
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        className="stroke-muted"
        strokeWidth="8"
      />
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        className="stroke-emerald-500"
        strokeWidth="8"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference - progress}
        transform={`rotate(-90 ${center} ${center})`}
      />
      <text
        x={center}
        y={center - 4}
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-foreground text-2xl font-bold"
        style={{ fontSize: size * 0.22 }}
      >
        {score}
      </text>
      <text
        x={center}
        y={center + size * 0.15}
        textAnchor="middle"
        className="fill-muted-foreground"
        style={{ fontSize: size * 0.09 }}
      >
        /100
      </text>
    </svg>
  );
}

function MiniScoreBar({ label, score }: { label: string; score: number }) {
  const color =
    score >= 80
      ? "bg-emerald-500"
      : score >= 60
        ? "bg-yellow-500"
        : "bg-red-500";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold">{score}</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}

interface LandingPageProps {
  onSignIn: () => void;
}

export function LandingPage({ onSignIn }: LandingPageProps) {
  const [welcomeInfo, setWelcomeInfo] = useState<{
    welcomeCredits: number;
    approxGenerations: number;
  } | null>(null);

  useEffect(() => {
    fetch("/api/user/welcome-info")
      .then(
        (res) =>
          res.json() as Promise<{
            welcomeCredits: number;
            approxGenerations: number;
          }>,
      )
      .then((data) => setWelcomeInfo(data))
      .catch(() => {});
  }, []);

  const ttsFeatures = [
    {
      icon: Upload,
      key: "upload",
      color: "text-blue-600 dark:text-blue-400",
      title: "Upload Anything",
      desc: "Upload PDFs, DOCX, TXT files, or images — AI extracts text automatically.",
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
      icon: Captions,
      key: "karaoke",
      color: "text-yellow-600 dark:text-yellow-400",
      title: "Karaoke Effect",
      desc: "Follow along word-by-word as text highlights in sync with the audio.",
    },
    {
      icon: Download,
      key: "download",
      color: "text-rose-600 dark:text-rose-400",
      title: "Download MP3",
      desc: "Download any generated audio as MP3 for offline listening on any device.",
    },
    {
      icon: History,
      key: "history",
      color: "text-amber-600 dark:text-amber-400",
      title: "Generation History",
      desc: "Access all your past audio from any device — just sign in to pick up where you left off.",
    },
  ];

  const assessmentFeatures = [
    {
      icon: MessageCircle,
      key: "pron-assess",
      color: "text-emerald-600 dark:text-emerald-400",
      title: "Pronunciation Assessment",
      desc: "AI evaluates every word you speak with detailed accuracy, fluency, and prosody scoring.",
    },
    {
      icon: BarChart3,
      key: "phoneme",
      color: "text-teal-600 dark:text-teal-400",
      title: "Phoneme-Level Feedback",
      desc: "Drill down to individual sounds with IPA phoneme breakdowns and accuracy scores.",
    },
    {
      icon: Ear,
      key: "realtime",
      color: "text-sky-600 dark:text-sky-400",
      title: "Real-Time Speech Recognition",
      desc: "Get instant, detailed feedback as soon as you stop speaking.",
    },
    {
      icon: Target,
      key: "error-detect",
      color: "text-orange-600 dark:text-orange-400",
      title: "Error Detection",
      desc: "Identifies mispronunciations, omissions, insertions, and timing errors in your speech.",
    },
    {
      icon: Languages,
      key: "ipa",
      color: "text-indigo-600 dark:text-indigo-400",
      title: "IPA Phonetic Alphabet",
      desc: "See exact phonetic transcriptions of expected vs. spoken sounds in standard IPA notation.",
    },
    {
      icon: Trophy,
      key: "score-tracking",
      color: "text-amber-600 dark:text-amber-400",
      title: "Score Tracking",
      desc: "Review past assessments, compare scores over time, and track your pronunciation improvement.",
    },
  ];

  const listenJourney = [
    { num: 1, icon: FileText, key: "input", label: "Upload or Type Text" },
    { num: 2, icon: Pencil, key: "edit", label: "Edit Extracted Text" },
    { num: 3, icon: Mic, key: "voice", label: "Choose Voice & Speed" },
    { num: 4, icon: Brain, key: "generate", label: "AI Generates Audio" },
    { num: 5, icon: Headphones, key: "listen", label: "Listen to Audio" },
    { num: 6, icon: BookOpen, key: "read", label: "Read Along with Karaoke" },
    { num: 7, icon: Download, key: "download", label: "Download MP3" },
  ];

  const practiceJourney = [
    { num: 1, icon: FileText, key: "ref", label: "Enter Reference Text" },
    { num: 2, icon: Mic, key: "rec", label: "Record Your Speech" },
    { num: 3, icon: Brain, key: "assess", label: "AI Assesses Pronunciation" },
    { num: 4, icon: Activity, key: "scores", label: "View Overall Scores" },
    { num: 5, icon: MessageCircle, key: "transcript", label: "Review Word-by-Word" },
    { num: 6, icon: BarChart3, key: "phonemes", label: "Explore Phoneme Detail" },
    { num: 7, icon: Trophy, key: "improve", label: "Track Your Progress" },
  ];

  const capabilities = [
    "PDF Support",
    "DOCX Support",
    "Image OCR",
    "6 AI Voices",
    "Speed Control",
    "Karaoke Effect",
    "Audio Download",
    "Pronunciation Scoring",
    "Phoneme Analysis",
    "IPA Phonetic Alphabet",
    "Error Detection",
    "Word-by-Word Feedback",
    "Fluency & Prosody Scores",
    "Assessment History",
    "Score Tracking Over Time",
    "Access from Any Device",
  ];

  return (
    <div className="relative">
      {/* ── Hero ── */}
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
              AI-powered text-to-speech &amp; pronunciation coaching
            </div>
          </motion.div>

          <motion.h1
            variants={heroItemVariants}
            className="text-6xl sm:text-8xl font-extrabold tracking-tighter text-white"
          >
            Mr.
            <span className="inline-block mx-1">&#x1F196;</span>
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-violet-400 bg-clip-text text-transparent">
              SpeakAI
            </span>
          </motion.h1>

          <motion.p
            variants={heroItemVariants}
            className="mt-3 text-2xl sm:text-3xl font-medium italic flex flex-wrap items-center justify-center gap-x-2"
          >
            {[
              "Listen",
              "to",
              "any",
              "text",
              "—",
              "master",
              "your",
              "pronunciation",
              "with",
              "AI",
              "feedback",
            ].map((word, i, arr) => (
              <motion.span
                key={i}
                initial={{ opacity: 0.2 }}
                animate={{ opacity: [0.2, 1, 1, 0.5] }}
                transition={{
                  duration: arr.length * 0.35,
                  repeat: Infinity,
                  repeatDelay: 1,
                  delay: i * 0.35,
                  ease: "easeInOut",
                }}
                className={
                  word === "—"
                    ? "font-bold bg-gradient-to-r from-blue-300/80 via-indigo-300/80 to-violet-300/80 bg-clip-text text-transparent"
                    : "text-white/60"
                }
              >
                {word}
              </motion.span>
            ))}
          </motion.p>

          <motion.p
            variants={heroItemVariants}
            className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-white/50"
          >
            Transform text into lifelike speech with karaoke-style highlighting,
            then practice reading aloud and get instant AI-powered pronunciation
            scores — word by word, phoneme by phoneme.
          </motion.p>

          <motion.div
            variants={heroItemVariants}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
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
            className="mt-6 flex flex-col items-center gap-1.5"
          >
            <div className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 backdrop-blur-sm text-white/80 text-sm">
              <Sparkles className="h-4 w-4 text-indigo-400" />
              <span>
                {welcomeInfo
                  ? `${welcomeInfo.welcomeCredits} free credits on sign up — enough for ~${welcomeInfo.approxGenerations} audio generations`
                  : "Free credits on sign up — just sign in with Google"}
              </span>
            </div>
            {welcomeInfo && (
              <p className="text-xs text-white/40">
                Based on ~3,000 characters per generation
              </p>
            )}
          </motion.div>
        </motion.div>
      </section>

      <WaveDivider />

      {/* ── Speaking Assessment Showcase ── */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              New Feature
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Speaking Assessment
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-2xl mx-auto"
            >
              Practice reading any text aloud and receive instant, detailed
              feedback on your pronunciation. AI scores every word and phoneme
              so you know exactly what to improve.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection
            className="max-w-4xl mx-auto"
            staggerDelay={0.15}
          >
            <motion.div
              variants={cardVariants}
              className={`rounded-3xl p-6 sm:p-8 ${glassCard}`}
            >
              <div className="grid sm:grid-cols-2 gap-8 items-center">
                <div className="flex flex-col items-center gap-4">
                  <ScoreRing score={87} size={140} />
                  <div className="w-full space-y-3 max-w-[200px]">
                    <MiniScoreBar label="Accuracy" score={92} />
                    <MiniScoreBar label="Fluency" score={85} />
                    <MiniScoreBar label="Completeness" score={90} />
                    <MiniScoreBar label="Prosody" score={78} />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                      Word-by-Word Feedback
                    </p>
                    <div className="rounded-xl bg-background/60 p-4 leading-relaxed text-base">
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">The </span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">quick </span>
                      <span className="text-yellow-600 dark:text-yellow-400 font-medium">brown </span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">fox </span>
                      <span className="text-red-600 dark:text-red-400 font-semibold underline decoration-red-500/60 underline-offset-2">jumps </span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">over </span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">the </span>
                      <span className="text-muted-foreground line-through opacity-70">lazy </span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">dog.</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                      Error Details
                    </p>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                        <span>
                          <span className="font-medium text-red-600 dark:text-red-400">jumps</span>
                          {" "}— Mispronunciation
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/50" />
                        <span>
                          <span className="font-medium text-muted-foreground line-through">lazy</span>
                          {" "}— Omission
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-yellow-500" />
                        <span>
                          <span className="font-medium text-yellow-600 dark:text-yellow-400">brown</span>
                          {" "}— Fair (80–89)
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-white/10 flex items-center justify-center gap-6 flex-wrap text-sm text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  Word-level scores
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  IPA phoneme breakdown
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  Error classification
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  Progress tracking
                </span>
              </div>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>

      {/* ── TTS Features ── */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400 mb-3"
            >
              Text-to-Speech
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Lifelike AI Voices
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              Upload any content and let AI transform it into natural,
              high-quality speech with karaoke-style highlighting.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
            {ttsFeatures.map(({ icon: Icon, key, color, title, desc }) => (
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

      {/* ── Assessment Features ── */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              Pronunciation Coach
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Master Your Pronunciation
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              Record yourself reading any text and get instant, detailed AI
              feedback on every word and sound.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
            {assessmentFeatures.map(
              ({ icon: Icon, key, color, title, desc }) => (
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
              ),
            )}
          </AnimatedSection>
        </div>
      </section>

      {/* ── How It Works (Two Columns) ── */}
      <section className="py-20 px-4 bg-muted/30">
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
              Two Powerful Tools
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-emerald-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
            <div>
              <AnimatedSection staggerDelay={0.08}>
                <motion.h3
                  variants={sectionTitleVariants}
                  className="text-xl font-bold mb-6 flex items-center gap-2"
                >
                  <Headphones className="h-5 w-5 text-blue-500" />
                  Listen to Any Text
                </motion.h3>
                {listenJourney.map(
                  ({ num, icon: Icon, key, label }, idx) => (
                    <motion.div
                      key={key}
                      variants={cardVariants}
                      className="relative flex items-center gap-4"
                    >
                      <div className="relative flex flex-col items-center">
                        <div
                          className={`flex items-center justify-center w-9 h-9 rounded-full text-xs font-bold text-white shrink-0 ${
                            idx === 0
                              ? "bg-gradient-to-br from-blue-500 to-indigo-500"
                              : "bg-gradient-to-br from-slate-500 to-slate-600"
                          }`}
                        >
                          {num}
                        </div>
                        {idx < listenJourney.length - 1 && (
                          <div className="w-0.5 h-10 bg-gradient-to-b from-border to-transparent" />
                        )}
                      </div>
                      <div
                        className={`flex items-center gap-3 ${
                          idx < listenJourney.length - 1 ? "pb-10" : ""
                        }`}
                      >
                        <div className="p-1.5 rounded-lg bg-muted">
                          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                        </div>
                        <span className="font-medium text-sm">{label}</span>
                      </div>
                    </motion.div>
                  ),
                )}
              </AnimatedSection>
            </div>

            <div>
              <AnimatedSection staggerDelay={0.08}>
                <motion.h3
                  variants={sectionTitleVariants}
                  className="text-xl font-bold mb-6 flex items-center gap-2"
                >
                  <MessageCircle className="h-5 w-5 text-emerald-500" />
                  Practice &amp; Get Scored
                </motion.h3>
                {practiceJourney.map(
                  ({ num, icon: Icon, key, label }, idx) => (
                    <motion.div
                      key={key}
                      variants={cardVariants}
                      className="relative flex items-center gap-4"
                    >
                      <div className="relative flex flex-col items-center">
                        <div
                          className={`flex items-center justify-center w-9 h-9 rounded-full text-xs font-bold text-white shrink-0 ${
                            idx === 0
                              ? "bg-gradient-to-br from-emerald-500 to-teal-500"
                              : "bg-gradient-to-br from-slate-500 to-slate-600"
                          }`}
                        >
                          {num}
                        </div>
                        {idx < practiceJourney.length - 1 && (
                          <div className="w-0.5 h-10 bg-gradient-to-b from-border to-transparent" />
                        )}
                      </div>
                      <div
                        className={`flex items-center gap-3 ${
                          idx < practiceJourney.length - 1 ? "pb-10" : ""
                        }`}
                      >
                        <div className="p-1.5 rounded-lg bg-muted">
                          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                        </div>
                        <span className="font-medium text-sm">{label}</span>
                      </div>
                    </motion.div>
                  ),
                )}
              </AnimatedSection>
            </div>
          </div>
        </div>
      </section>

      {/* ── Capabilities ── */}
      <section className="py-20 px-4">
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
              Everything in One Place
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-emerald-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <AnimatedSection className="flex flex-wrap justify-center gap-3 max-w-4xl mx-auto">
            {capabilities.map((capability) => (
              <motion.span
                key={capability}
                variants={pillVariants}
                className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm border ${
                  capability === "Pronunciation Scoring" ||
                  capability === "Phoneme Analysis" ||
                  capability === "IPA Phonetic Alphabet" ||
                  capability === "Error Detection" ||
                  capability === "Word-by-Word Feedback" ||
                  capability === "Fluency & Prosody Scores" ||
                  capability === "Assessment History" ||
                  capability === "Score Tracking Over Time"
                    ? "bg-emerald-50/80 dark:bg-emerald-900/20 border-emerald-200/50 dark:border-emerald-700/30"
                    : "bg-white/60 dark:bg-white/10 border-white/30 dark:border-white/10"
                }`}
              >
                <CheckCircle2
                  className={`h-3.5 w-3.5 ${
                    capability === "Pronunciation Scoring" ||
                    capability === "Phoneme Analysis" ||
                    capability === "IPA Phonetic Alphabet" ||
                    capability === "Error Detection" ||
                    capability === "Word-by-Word Feedback" ||
                    capability === "Fluency & Prosody Scores" ||
                    capability === "Assessment History" ||
                    capability === "Score Tracking Over Time"
                      ? "text-emerald-500"
                      : "text-blue-500"
                  }`}
                />
                {capability}
              </motion.span>
            ))}
          </AnimatedSection>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="relative py-24 px-4 overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <GalaxyBackground />
        <div className="relative z-10 container mx-auto text-center">
          <AnimatedSection>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white mb-4"
            >
              Ready to Speak Better?
            </motion.h2>
            <motion.p
              variants={sectionTitleVariants}
              className="text-white/50 mb-8 max-w-xl mx-auto"
            >
              Listen to any text and practice your pronunciation with instant AI
              feedback. Sign in and start for free.
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
