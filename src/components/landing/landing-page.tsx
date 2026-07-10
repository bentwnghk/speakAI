"use client";

import { useRef, useEffect, useState, useCallback } from "react";
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
  Globe,
  AlertTriangle,
  Lightbulb,
  Shuffle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  getDictionary,
  type Locale,
} from "@/lib/i18n";

const LANDING_LOCALE_KEY = "speakai-landing-locale";

function detectBrowserLocale(): Locale {
  if (typeof navigator === "undefined") return "en";
  const langs = navigator.languages || [navigator.language];
  for (const lang of langs) {
    const lower = lang.toLowerCase();
    if (
      lower.startsWith("zh") ||
      lower.startsWith("cmn") ||
      lower === "yue"
    ) {
      return "zh-TW";
    }
  }
  return "en";
}

function getInitialLocale(): Locale {
  if (typeof window === "undefined") return "en";
  const saved = localStorage.getItem(LANDING_LOCALE_KEY) as Locale | null;
  if (saved === "en" || saved === "zh-TW") return saved;
  return detectBrowserLocale();
}

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
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    setLocaleState(getInitialLocale());
  }, []);

  const t = getDictionary(locale);
  const l = t.landing;

  const toggleLocale = useCallback(() => {
    setLocaleState((prev) => {
      const next = prev === "en" ? "zh-TW" : "en";
      localStorage.setItem(LANDING_LOCALE_KEY, next);
      return next;
    });
  }, []);

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
      title: l.uploadTitle,
      desc: l.uploadDesc,
    },
    {
      icon: Mic,
      key: "voices",
      color: "text-violet-600 dark:text-violet-400",
      title: l.voicesTitle,
      desc: l.voicesDesc,
    },
    {
      icon: Gauge,
      key: "speed",
      color: "text-cyan-600 dark:text-cyan-400",
      title: l.speedTitle,
      desc: l.speedDesc,
    },
    {
      icon: Captions,
      key: "karaoke",
      color: "text-yellow-600 dark:text-yellow-400",
      title: l.karaokeTitle,
      desc: l.karaokeDesc,
    },
    {
      icon: Download,
      key: "download",
      color: "text-rose-600 dark:text-rose-400",
      title: l.downloadTitle,
      desc: l.downloadDesc,
    },
    {
      icon: History,
      key: "history",
      color: "text-amber-600 dark:text-amber-400",
      title: l.historyTitle,
      desc: l.historyDesc,
    },
  ];

  const assessmentFeatures = [
    {
      icon: MessageCircle,
      key: "pron-assess",
      color: "text-emerald-600 dark:text-emerald-400",
      title: l.pronAssessTitle,
      desc: l.pronAssessDesc,
    },
    {
      icon: BarChart3,
      key: "phoneme",
      color: "text-teal-600 dark:text-teal-400",
      title: l.phonemeTitle,
      desc: l.phonemeDesc,
    },
    {
      icon: Ear,
      key: "realtime",
      color: "text-sky-600 dark:text-sky-400",
      title: l.realtimeTitle,
      desc: l.realtimeDesc,
    },
    {
      icon: Target,
      key: "error-detect",
      color: "text-orange-600 dark:text-orange-400",
      title: l.errorDetectTitle,
      desc: l.errorDetectDesc,
    },
    {
      icon: Languages,
      key: "ipa",
      color: "text-indigo-600 dark:text-indigo-400",
      title: l.ipaTitle,
      desc: l.ipaDesc,
    },
    {
      icon: Trophy,
      key: "score-tracking",
      color: "text-amber-600 dark:text-amber-400",
      title: l.scoreTrackTitle,
      desc: l.scoreTrackDesc,
    },
    {
      icon: Sparkles,
      key: "ai-coach",
      color: "text-emerald-600 dark:text-emerald-400",
      title: l.coachFeatureTitle,
      desc: l.coachFeatureDesc,
    },
    {
      icon: Shuffle,
      key: "confusion",
      color: "text-fuchsia-600 dark:text-fuchsia-400",
      title: l.confusionTitle,
      desc: l.confusionDesc,
    },
  ];

  const listenJourney = [
    { num: 1, icon: FileText, key: "input", label: l.stepInput },
    { num: 2, icon: Pencil, key: "edit", label: l.stepEdit },
    { num: 3, icon: Mic, key: "voice", label: l.stepVoice },
    { num: 4, icon: Brain, key: "generate", label: l.stepGenerate },
    { num: 5, icon: Headphones, key: "listen", label: l.stepListen },
    { num: 6, icon: BookOpen, key: "read", label: l.stepRead },
    { num: 7, icon: Download, key: "download", label: l.stepDownload },
  ];

  const practiceJourney = [
    { num: 1, icon: FileText, key: "ref", label: l.stepRef },
    { num: 2, icon: Mic, key: "rec", label: l.stepRec },
    { num: 3, icon: Brain, key: "assess", label: l.stepAssess },
    { num: 4, icon: Activity, key: "scores", label: l.stepScores },
    { num: 5, icon: MessageCircle, key: "transcript", label: l.stepTranscript },
    { num: 6, icon: BarChart3, key: "phonemes", label: l.stepPhonemes },
    { num: 7, icon: Sparkles, key: "coach", label: l.stepCoach },
    { num: 8, icon: Trophy, key: "improve", label: l.stepImprove },
  ];

  const capabilities = [
    l.capPdf,
    l.capDocx,
    l.capOcr,
    l.capVoices,
    l.capSpeed,
    l.capKaraoke,
    l.capAudio,
    l.capPronScoring,
    l.capPhoneme,
    l.capIpa,
    l.capErrorDetect,
    l.capWordFeedback,
    l.capFluency,
    l.capAssessHistory,
    l.capScoreTrack,
    l.capAiCoach,
    l.capConfusionMatrix,
    l.capAnyDevice,
  ];

  const assessmentCapabilities = new Set([
    l.capPronScoring,
    l.capPhoneme,
    l.capIpa,
    l.capErrorDetect,
    l.capWordFeedback,
    l.capFluency,
    l.capAssessHistory,
    l.capScoreTrack,
    l.capAiCoach,
    l.capConfusionMatrix,
  ]);

  return (
    <div className="relative">
      {/* ── Language Switcher ── */}
      <button
        onClick={toggleLocale}
        className="fixed top-4 right-4 z-50 flex items-center gap-1.5 px-3 py-2 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-md border border-white/20 text-white/80 hover:text-white transition-colors cursor-pointer"
        aria-label={locale === "en" ? "切換至繁體中文" : "Switch to English"}
      >
        <Globe className="h-4 w-4" />
        <span className="text-sm font-medium">
          {locale === "en" ? "中文" : "EN"}
        </span>
      </button>

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
              {l.heroTagline}
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
            {l.heroSubtitle.map((word, i, arr) => (
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
            {l.heroDesc}
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
              {l.getStartedFree}
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
                  ? l.welcomeCredits
                      .replace("${credits}", String(welcomeInfo.welcomeCredits))
                      .replace("${gens}", String(welcomeInfo.approxGenerations))
                  : l.welcomeCreditsDefault}
              </span>
            </div>
            {welcomeInfo && (
              <p className="text-xs text-white/40">{l.basedOnChars}</p>
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
              {l.assessmentBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              {l.assessmentTitle}
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-2xl mx-auto"
            >
              {l.assessmentDesc}
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
                    <MiniScoreBar label={l.accuracy} score={92} />
                    <MiniScoreBar label={l.fluency} score={85} />
                    <MiniScoreBar label={l.completeness} score={90} />
                    <MiniScoreBar label={l.prosody} score={78} />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                      {l.wordByWordFeedback}
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
                      {l.errorDetails}
                    </p>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                        <span>
                          <span className="font-medium text-red-600 dark:text-red-400">jumps</span>
                          {" "}— {l.mispronunciation}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/50" />
                        <span>
                          <span className="font-medium text-muted-foreground line-through">lazy</span>
                          {" "}— {l.omission}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-full bg-yellow-500" />
                        <span>
                          <span className="font-medium text-yellow-600 dark:text-yellow-400">brown</span>
                          {" "}— {l.fairScore}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-white/10 flex items-center justify-center gap-6 flex-wrap text-sm text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  {l.wordLevelScores}
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  {l.ipaPhonemeBreakdown}
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  {l.errorClassification}
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  {l.progressTracking}
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
              {l.ttsBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              {l.ttsTitle}
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              {l.ttsDesc}
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
              {l.pronBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              {l.pronTitle}
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              {l.pronDesc}
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

      {/* ── AI Pronunciation Coach Showcase ── */}
      <section className="relative py-20 px-4 overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#0b2515] to-[#0f172a]">
        <GalaxyBackground />
        <div className="relative z-10 container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-400 mb-3"
            >
              {l.coachBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4 text-white"
            >
              {l.coachTitle}
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-white/50 mb-12 max-w-2xl mx-auto"
            >
              {l.coachDesc}
            </motion.p>
          </AnimatedSection>

          <AnimatedSection
            className="grid gap-5 sm:grid-cols-3 max-w-5xl mx-auto"
            staggerDelay={0.15}
          >
            <motion.div
              variants={cardVariants}
              className="rounded-2xl p-6 bg-green-500/10 border border-green-500/20 backdrop-blur-xl"
            >
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="size-5 text-green-400" />
                <h3 className="font-semibold text-white">
                  {l.coachStrengthsTitle}
                </h3>
              </div>
              <ul className="space-y-2.5 text-sm text-white/70">
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-green-400/60" />
                  <span>{l.coachStrengths1}</span>
                </li>
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-green-400/60" />
                  <span>{l.coachStrengths2}</span>
                </li>
              </ul>
            </motion.div>

            <motion.div
              variants={cardVariants}
              className="rounded-2xl p-6 bg-amber-500/10 border border-amber-500/20 backdrop-blur-xl"
            >
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle className="size-5 text-amber-400" />
                <h3 className="font-semibold text-white">
                  {l.coachWeaknessesTitle}
                </h3>
              </div>
              <ul className="space-y-2.5 text-sm text-white/70">
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-amber-400/60" />
                  <span>{l.coachWeaknesses1}</span>
                </li>
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-amber-400/60" />
                  <span>{l.coachWeaknesses2}</span>
                </li>
              </ul>
            </motion.div>

            <motion.div
              variants={cardVariants}
              className="rounded-2xl p-6 bg-yellow-500/10 border border-yellow-500/20 backdrop-blur-xl"
            >
              <div className="flex items-center gap-2 mb-4">
                <Lightbulb className="size-5 text-yellow-400" />
                <h3 className="font-semibold text-white">{l.coachTipsTitle}</h3>
              </div>
              <ul className="space-y-2.5 text-sm text-white/70">
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-yellow-400/60" />
                  <span>{l.coachTips1}</span>
                </li>
                <li className="flex gap-2 leading-relaxed">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-yellow-400/60" />
                  <span>{l.coachTips2}</span>
                </li>
              </ul>
            </motion.div>
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
              {l.howBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              {l.howTitle}
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
                  {l.listenTitle}
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
                  {l.practiceTitle}
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
              {l.capBadge}
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              {l.capTitle}
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
                  assessmentCapabilities.has(capability)
                    ? "bg-emerald-50/80 dark:bg-emerald-900/20 border-emerald-200/50 dark:border-emerald-700/30"
                    : "bg-white/60 dark:bg-white/10 border-white/30 dark:border-white/10"
                }`}
              >
                <CheckCircle2
                  className={`h-3.5 w-3.5 ${
                    assessmentCapabilities.has(capability)
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
              {l.ctaTitle}
            </motion.h2>
            <motion.p
              variants={sectionTitleVariants}
              className="text-white/50 mb-8 max-w-xl mx-auto"
            >
              {l.ctaDesc}
            </motion.p>
            <motion.div variants={heroItemVariants}>
              <Button
                size="lg"
                onClick={onSignIn}
                className="h-12 px-8 text-base bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white border-0 shadow-lg shadow-blue-500/25 cursor-pointer"
              >
                <GoogleIcon />
                {l.signInGoogle}
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>
    </div>
  );
}
