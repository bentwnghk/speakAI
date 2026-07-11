"use client";

import { useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import { RotateCcw, History, Mic, FileText, SlidersHorizontal, Timer, Hand, BookOpen, Download, Clock, Coins } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useUserSettings } from "@/hooks/use-settings";
import { useCredits } from "@/hooks/use-credits";
import { RecordingControls } from "./recording-controls";
import { ScoreOverview } from "./score-overview";
import { TranscriptView } from "./transcript-view";
import { WordDetail } from "./word-detail";
import { PhonemeAnalysis } from "./phoneme-analysis";
import { FeedbackCard } from "./feedback-card";
import { ErrorSummary } from "./error-summary";
import { ReferenceAudioSection } from "./reference-audio-section";
import type {
  AssessmentResult,
  AssessmentDetailResult,
  RecordingMode,
  RecordingState,
  WordResult,
  PronunciationScores,
} from "@/types/assessment";
import { type AssessmentFilter, filterWords } from "@/types/assessment";

const SAMPLE_TEXTS = [
  "sample1",
  "sample2",
  "sample3",
  "sample4",
  "sample5",
] as const;

interface JsonNBest {
  PronunciationAssessment: {
    AccuracyScore: number;
    FluencyScore: number;
    CompletenessScore: number;
    ProsodyScore: number;
    PronScore: number;
  };
  Words: WordResult[];
}

interface JsonResult {
  DisplayText?: string;
  Duration?: number;
  NBest?: JsonNBest[];
}

export function AssessmentForm({ pricePerMinHkd, initialText = "" }: { pricePerMinHkd: number; initialText?: string }) {
  const { t } = useUserSettings();
  const { refreshBalance } = useCredits();
  const at = t.assessment as Record<string, string>;

  const [referenceText, setReferenceText] = useState(initialText);
  const [mode, setMode] = useState<RecordingMode>("auto");
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [errorFilter, setErrorFilter] = useState<AssessmentFilter>("All");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [savedAssessmentId, setSavedAssessmentId] = useState<string | null>(null);
  const [savedCost, setSavedCost] = useState<number | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const handleWordCost = useCallback((cost: number) => {
    setSavedCost((prev) => (prev ?? 0) + cost);
    if (savedAssessmentId) {
      void fetch(`/api/assessment/${savedAssessmentId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ additionalCost: cost }),
      });
    }
  }, [savedAssessmentId]);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognizerRef = useRef<import("microsoft-cognitiveservices-speech-sdk").SpeechRecognizer | null>(null);
  const userStoppedRef = useRef(false);
  const recordStartRef = useRef(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number>(0);
  const micLevelRef = useRef<number>(0);

  const wordCount = referenceText.trim()
    ? referenceText.trim().split(/\s+/).length
    : 0;

  function setupVolumeMeter(stream: MediaStream) {
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const update = () => {
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const v = (dataArray[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      const target = Math.min(1, rms * 4);
      micLevelRef.current = micLevelRef.current * 0.6 + target * 0.4;
      rafRef.current = requestAnimationFrame(update);
    };
    rafRef.current = requestAnimationFrame(update);
  }

  function cleanupVolumeMeter() {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    if (audioCtxRef.current) {
      void audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    micLevelRef.current = 0;
  }

  const handleStart = useCallback(async () => {
    if (!referenceText.trim()) return;

    setResult(null);
    setSavedAssessmentId(null);
    setSavedCost(null);
    setRecordingState("recording");
    audioChunksRef.current = [];
    userStoppedRef.current = false;

    let mediaRecorder: MediaRecorder | null = null;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setupVolumeMeter(stream);
      const mimeType = MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : undefined;
      mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      mediaRecorder.start(250);
      recordStartRef.current = Date.now();
    } catch {
      toast.error(at.micDenied);
      setRecordingState("idle");
      return;
    }

    try {
      const tokenRes = await fetch("/api/speech/token");
      if (!tokenRes.ok) throw new Error("token");
      const tokenData = (await tokenRes.json()) as { token: string; region: string };

      const SpeechSDK = await import(
        "microsoft-cognitiveservices-speech-sdk"
      );

      const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(
        tokenData.token,
        tokenData.region
      );
      speechConfig.speechRecognitionLanguage = "en-US";

      const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();

      const pronunciationConfig = new SpeechSDK.PronunciationAssessmentConfig(
        referenceText.trim(),
        SpeechSDK.PronunciationAssessmentGradingSystem.HundredMark,
        SpeechSDK.PronunciationAssessmentGranularity.Phoneme,
        mode === "auto"
      );
      pronunciationConfig.phonemeAlphabet = "IPA";
      pronunciationConfig.nbestPhonemeCount = 5;
      pronunciationConfig.enableProsodyAssessment = true;

      const recognizer = new SpeechSDK.SpeechRecognizer(
        speechConfig,
        audioConfig
      );
      recognizerRef.current = recognizer;

      pronunciationConfig.applyTo(recognizer);

      if (mode === "auto") {
        await recognizeOnce(recognizer, SpeechSDK);
      } else {
        await recognizeContinuous(recognizer, SpeechSDK);
      }
    } catch (err) {
      console.error("Assessment error:", err);
      cleanupVolumeMeter();
      setRecordingState("idle");
      if (err instanceof Error && err.message === "token") {
        toast.error(at.tokenError);
      } else {
        toast.error(at.recognitionError);
      }
    }
  }, [referenceText, mode, at]);

  type SdkTypes = typeof import("microsoft-cognitiveservices-speech-sdk");

  function recognizeOnce(
    recognizer: import("microsoft-cognitiveservices-speech-sdk").SpeechRecognizer,
    SpeechSDK: SdkTypes
  ) {
    return new Promise<void>((resolve, reject) => {
      setRecordingState("recording");

      recognizer.recognizeOnceAsync(
        (recResult) => {
          recognizerRef.current = null;
          recognizer.close();
          audioConfigCleanup();

          if (recResult.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
            setRecordingState("processing");
            void stopMediaRecorder().then(({ blob, recordingDurationMs }) => {
              processResult(recResult, SpeechSDK, blob, recordingDurationMs);
            });
            setRecordingState("done");
            resolve();
          } else if (recResult.reason === SpeechSDK.ResultReason.NoMatch) {
            void stopMediaRecorder();
            setRecordingState("idle");
            toast.error(at.noSpeech);
            resolve();
          } else {
            void stopMediaRecorder();
            setRecordingState("idle");
            if (userStoppedRef.current) {
              // User pressed Stop intentionally — clean exit, no error
              resolve();
            } else {
              const cancel =
                SpeechSDK.CancellationDetails.fromResult(recResult);
              console.error("Recognition canceled:", cancel.errorDetails);
              reject(new Error("canceled"));
            }
          }
        },
        (error: string) => {
          void stopMediaRecorder();
          recognizerRef.current = null;
          recognizer.close();
          audioConfigCleanup();
          setRecordingState("idle");
          reject(new Error(error));
        }
      );
    });
  }

  function recognizeContinuous(
    recognizer: import("microsoft-cognitiveservices-speech-sdk").SpeechRecognizer,
    SpeechSDK: SdkTypes
  ) {
    return new Promise<void>((resolve, reject) => {
      const allResults: import("microsoft-cognitiveservices-speech-sdk").SpeechRecognitionResult[] =
        [];

      // Guard against double-processing if both the stopContinuousRecognitionAsync
      // success callback and the canceled/sessionStopped events fire.
      let finished = false;

      const finish = () => {
        if (finished) return;
        finished = true;
        recognizerRef.current = null;
        try { recognizer.close(); } catch { /* already closed */ }
        audioConfigCleanup();
        if (allResults.length > 0) {
          const combined = combineResults(allResults, SpeechSDK);
          void stopMediaRecorder().then(({ blob, recordingDurationMs }) => {
            processCombinedResult(combined, blob, recordingDurationMs);
            setRecordingState("done");
          });
        } else {
          void stopMediaRecorder();
          setRecordingState("idle");
          toast.error(at.noSpeech);
        }
        resolve();
      };

      recognizer.recognized = (_s, e) => {
        if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
          allResults.push(e.result);
        }
      };

      // Handle unexpected errors and natural end-of-stream as a fallback.
      // The primary result-processing path is the stopContinuousRecognitionAsync
      // success callback inside window.__stopAssessment.
      recognizer.canceled = (_s, e) => {
        if (e.reason === SpeechSDK.CancellationReason.Error) {
          console.error("Continuous recognition error:", e.errorDetails);
          if (finished) return;
          finished = true;
          recognizerRef.current = null;
          try { recognizer.close(); } catch { /* already closed */ }
          audioConfigCleanup();
          void stopMediaRecorder();
          setRecordingState("idle");
          toast.error(at.recognitionError);
          resolve();
        } else {
          // EndOfStream or other non-error cancellation — treat as a natural finish.
          finish();
        }
      };

      recognizer.startContinuousRecognitionAsync(
        () => {},
        (err: string) => {
          void stopMediaRecorder();
          recognizerRef.current = null;
          try { recognizer.close(); } catch { /* already closed */ }
          audioConfigCleanup();
          setRecordingState("idle");
          reject(new Error(err));
        }
      );

      // Called by handleStop(). stopContinuousRecognitionAsync's success callback
      // fires reliably once the SDK has fully stopped — use it as the primary
      // result-processing trigger instead of waiting for the canceled event.
      window.__stopAssessment = () => {
        recognizer.stopContinuousRecognitionAsync(
          finish,
          (err: string) => {
            if (!finished) {
              finished = true;
              recognizerRef.current = null;
              try { recognizer.close(); } catch { /* already closed */ }
              audioConfigCleanup();
              void stopMediaRecorder();
              setRecordingState("idle");
            }
            reject(new Error(err));
          }
        );
      };
    });
  }

  function stopMediaRecorder(): Promise<{ blob: Blob | null; recordingDurationMs: number }> {
    cleanupVolumeMeter();
    const recorder = mediaRecorderRef.current;
    const duration = recordStartRef.current > 0 ? Date.now() - recordStartRef.current : 0;
    recordStartRef.current = 0;
    if (!recorder || recorder.state === "inactive") return Promise.resolve({ blob: null, recordingDurationMs: duration });

    return new Promise((resolve) => {
      recorder.onstop = () => {
        const tracks = recorder.stream.getTracks();
        for (const track of tracks) track.stop();
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType });
        audioChunksRef.current = [];
        mediaRecorderRef.current = null;
        resolve({ blob: blob.size > 0 ? blob : null, recordingDurationMs: duration });
      };
      recorder.stop();
    });
  }

  function audioConfigCleanup() {
    delete window.__stopAssessment;
  }

  function combineResults(
    results: import("microsoft-cognitiveservices-speech-sdk").SpeechRecognitionResult[],
    SpeechSDK: SdkTypes
  ) {
    const jsonResults: JsonResult[] = results.map((r) => {
      const jsonStr = r.properties.getProperty(
        SpeechSDK.PropertyId.SpeechServiceResponse_JsonResult
      );
      return JSON.parse(jsonStr) as JsonResult;
    });

    const combinedWords: WordResult[] = [];
    let totalDuration = 0;

    for (const jr of jsonResults) {
      const nbest = jr.NBest?.[0];
      if (!nbest) continue;
      for (const w of nbest.Words || []) {
        combinedWords.push(w);
      }
      totalDuration += jr.Duration || 0;
    }

    const scores: PronunciationScores = {
      AccuracyScore: 0,
      FluencyScore: 0,
      CompletenessScore: 0,
      ProsodyScore: 0,
      PronScore: 0,
    };
    let count = 0;
    for (const jr of jsonResults) {
      const pa = jr.NBest?.[0]?.PronunciationAssessment;
      if (pa) {
        scores.AccuracyScore += pa.AccuracyScore || 0;
        scores.FluencyScore += pa.FluencyScore || 0;
        scores.CompletenessScore += pa.CompletenessScore || 0;
        scores.ProsodyScore += pa.ProsodyScore || 0;
        scores.PronScore += pa.PronScore || 0;
        count++;
      }
    }
    if (count > 0) {
      scores.AccuracyScore /= count;
      scores.FluencyScore /= count;
      scores.CompletenessScore /= count;
      scores.ProsodyScore /= count;
      scores.PronScore /= count;
    }

    const displayText = jsonResults
      .map((jr) => jr.DisplayText || "")
      .filter(Boolean)
      .join(" ");

    applyMiscueForContinuous(combinedWords, referenceText.trim());

    return { words: combinedWords, scores, displayText, totalDuration };
  }

  function applyMiscueForContinuous(
    recognizedWords: WordResult[],
    refText: string
  ) {
    // Strip leading/trailing punctuation (periods, commas, exclamation/question marks,
    // quotation marks, etc.) so that "year," matches "year" and "'bun" matches "bun".
    const stripPunct = (w: string) => w.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");
    const refWords = refText
      .toLowerCase()
      .split(/\s+/)
      .map(stripPunct)
      .filter(Boolean);
    const recWords = recognizedWords.map((w) =>
      stripPunct(w.Word.toLowerCase())
    );

    const matched = new Set<number>();

    for (const refWord of refWords) {
      for (let ci = 0; ci < recWords.length; ci++) {
        if (!matched.has(ci) && recWords[ci] === refWord) {
          matched.add(ci);
          break;
        }
      }
    }

    for (let ci = 0; ci < recognizedWords.length; ci++) {
      if (
        !matched.has(ci) &&
        recognizedWords[ci].PronunciationAssessment.ErrorType === "None"
      ) {
        recognizedWords[ci].PronunciationAssessment.ErrorType = "Insertion";
      }
    }
  }

  function processResult(
    recResult: import("microsoft-cognitiveservices-speech-sdk").SpeechRecognitionResult,
    SpeechSDK: SdkTypes,
    audioBlob: Blob | null,
    recordingDurationMs: number
  ) {
    const jsonStr = recResult.properties.getProperty(
      SpeechSDK.PropertyId.SpeechServiceResponse_JsonResult
    );
    const parsed = JSON.parse(jsonStr) as AssessmentDetailResult;
    const nbest = parsed.NBest?.[0];
    if (!nbest) {
      toast.error(at.noSpeech);
      return;
    }

    const paResult =
      SpeechSDK.PronunciationAssessmentResult.fromResult(recResult);

    const assessment: AssessmentResult = {
      detailResult: parsed,
      scores: {
        AccuracyScore: paResult.accuracyScore,
        FluencyScore: paResult.fluencyScore,
        CompletenessScore: paResult.completenessScore,
        ProsodyScore: paResult.prosodyScore,
        PronScore: paResult.pronunciationScore,
      },
      words: nbest.Words || [],
      recognizedText: recResult.text || "",
      durationMs: recordingDurationMs,
    };

    setResult(assessment);
    if (audioBlob) setAudioUrl(URL.createObjectURL(audioBlob));
    void saveAssessment(assessment, audioBlob);
  }

  function processCombinedResult(
    data: {
      words: WordResult[];
      scores: PronunciationScores;
      displayText: string;
      totalDuration: number;
    },
    audioBlob: Blob | null,
    recordingDurationMs: number
  ) {
    const assessment: AssessmentResult = {
      detailResult: {} as AssessmentDetailResult,
      scores: data.scores,
      words: data.words,
      recognizedText: data.displayText,
      durationMs: recordingDurationMs,
    };
    setResult(assessment);
    if (audioBlob) setAudioUrl(URL.createObjectURL(audioBlob));
    void saveAssessment(assessment, audioBlob);
  }

  async function saveAssessment(assessment: AssessmentResult, audioBlob: Blob | null) {
    try {
      const formData = new FormData();
      formData.append(
        "data",
        JSON.stringify({
          referenceText: referenceText.trim(),
          recognizedText: assessment.recognizedText,
          durationMs: Math.round(assessment.durationMs),
          accuracyScore: assessment.scores.AccuracyScore,
          fluencyScore: assessment.scores.FluencyScore,
          completenessScore: assessment.scores.CompletenessScore,
          prosodyScore: assessment.scores.ProsodyScore || null,
          pronScore: assessment.scores.PronScore,
          words: assessment.words,
          phonemes: assessment.words.map((w) => w.Phonemes || []),
        })
      );
      if (audioBlob) {
        const ext = audioBlob.type.includes("mp4") ? "mp4" : "webm";
        formData.append("audio", audioBlob, `recording.${ext}`);
        formData.append("audioMimeType", audioBlob.type || "audio/webm");
      }

      const res = await fetch("/api/assessment", {
        method: "POST",
        body: formData,
      });

      if (res.status === 402) {
        toast.error(at.insufficientCredits);
        return;
      }

      if (res.ok) {
        const data = (await res.json()) as { cost: number; id: string };
        setSavedAssessmentId(data.id);
        setSavedCost(data.cost);
        toast.success(at.saved.replace("${cost}", data.cost.toFixed(2)));
        void refreshBalance();
      } else {
        toast.error(at.saveFailed);
      }
    } catch {
      toast.error(at.saveFailed);
    }
  }

  function handleStop() {
    userStoppedRef.current = true;

    if (mode === "manual" && window.__stopAssessment) {
      // Signal Azure to stop continuous recognition. The recognizer.canceled
      // handler in recognizeContinuous fires asynchronously and handles all
      // cleanup: closes the recognizer, stops the media recorder, processes
      // collected results, and updates recording state.
      // Do NOT close the recognizer here — it must stay alive until canceled fires.
      // Do NOT call stopMediaRecorder here — the canceled handler does it.
      setRecordingState("processing");
      window.__stopAssessment();
    } else {
      // Auto mode: close the recognizer (triggers recognizeOnceAsync callback
      // with cancellation, which resolves silently thanks to userStoppedRef).
      const recognizer = recognizerRef.current;
      if (recognizer) {
        try {
          recognizer.close();
        } catch {}
        recognizerRef.current = null;
      }
      delete window.__stopAssessment;
      void stopMediaRecorder();
      setRecordingState("idle");
    }
  }

  function handleReset() {
    setResult(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setSavedAssessmentId(null);
    setRecordingState("idle");
    setErrorFilter("All");
  }

  function handleSample(key: string) {
    const sample = at[key];
    if (sample) setReferenceText(sample);
  }

  const filteredWords = filterWords(result?.words ?? [], errorFilter);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Mic className="size-6 text-primary" />
          {at.title}
        </h1>
        <p className="text-muted-foreground">{at.subtitle}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-1.5"><FileText className="size-4" />{at.referenceText}</span>
            <Select onValueChange={handleSample}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder={<span className="flex items-center gap-1.5"><BookOpen className="size-3.5" />{at.sampleTexts}</span>} />
              </SelectTrigger>
              <SelectContent>
                {SAMPLE_TEXTS.map((key) => (
                  <SelectItem key={key} value={key}>
                    {String(at[key]).slice(0, 30)}...
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardTitle>

        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={referenceText}
            onChange={(e) => setReferenceText(e.target.value)}
            placeholder={at.referencePlaceholder}
            rows={4}
            disabled={recordingState === "recording"}
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {at.wordCount.replace("{count}", String(wordCount))}{" "}
              &middot;{" "}
              {at.estimatedCost.replace("${cost}", pricePerMinHkd.toFixed(2))}
            </span>
          </div>

          <Separator />

          <div className="space-y-2">
            <p className="text-sm font-medium"><SlidersHorizontal className="inline size-4 mr-1.5 align-text-bottom" />{at.modeLabel}</p>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={mode === "auto" ? "default" : "outline"}
                onClick={() => setMode("auto")}
                disabled={recordingState === "recording"}
              >
                <Timer className="size-3.5" />
                {at.modeAuto}
              </Button>
              <Button
                size="sm"
                variant={mode === "manual" ? "default" : "outline"}
                onClick={() => setMode("manual")}
                disabled={recordingState === "recording"}
              >
                <Hand className="size-3.5" />
                {at.modeManual}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              {mode === "auto" ? at.modeAutoDesc : at.modeManualDesc}
            </p>
          </div>

          <Separator />

          <RecordingControls
            state={recordingState}
            mode={mode}
            onStart={() => void handleStart()}
            onStop={handleStop}
            t={at}
            disabled={!referenceText.trim()}
            micLevelRef={micLevelRef}
          />
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{at.results}</CardTitle>
              <Button variant="outline" size="sm" onClick={handleReset}>
                <RotateCcw className="size-4" />
                {at.tryAgain}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <ScoreOverview scores={result.scores} t={at} />

            <Separator />

            {audioUrl && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold">{at.yourRecording}</h3>
                <Card>
                  <CardContent className="flex items-center gap-3 py-3">
                    <audio controls className="w-full" preload="metadata">
                      <source src={audioUrl} />
                    </audio>
                    <a
                      href={audioUrl}
                      download={`assessment-recording.${audioUrl.includes("mp4") ? "mp4" : "webm"}`}
                    >
                      <Button variant="ghost" size="icon" className="shrink-0" asChild>
                        <span>
                          <Download className="size-4" />
                        </span>
                      </Button>
                    </a>
                  </CardContent>
                </Card>
              </div>
            )}

            <Separator />

            <div className="space-y-3">
              <h3 className="text-sm font-semibold">{at.recognizedText}</h3>
              <TranscriptView words={result.words} t={at} onCostUpdate={handleWordCost} />
            </div>

            <Separator />

            <div className="space-y-3">
              <h3 className="text-sm font-semibold">{at.errorSummary}</h3>
              <ErrorSummary
                words={result.words}
                t={at}
                filter={errorFilter}
                onFilterChange={setErrorFilter}
              />
            </div>

            <Separator />

            <Tabs defaultValue="word">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">{at.granularity}</h3>
                <TabsList>
                  <TabsTrigger value="fulltext">{at.granCoach}</TabsTrigger>
                  <TabsTrigger value="word">{at.granWord}</TabsTrigger>
                  <TabsTrigger value="phoneme">{at.granPhoneme}</TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="fulltext">
                <div className="rounded-lg border p-4">
                  <FeedbackCard
                    assessmentId={savedAssessmentId}
                    t={at}
                    onCostUpdate={handleWordCost}
                  />
                </div>
              </TabsContent>

              <TabsContent value="word">
                <div className="max-h-96 overflow-y-auto">
                  <WordDetail words={filteredWords} t={at} onCostUpdate={handleWordCost} />
                </div>
              </TabsContent>

              <TabsContent value="phoneme">
                <div className="max-h-96 overflow-y-auto">
                  <PhonemeAnalysis words={filteredWords} t={at} onCostUpdate={handleWordCost} />
                </div>
              </TabsContent>
            </Tabs>

            <Separator />

            <ReferenceAudioSection referenceText={referenceText} t={at} assessmentId={savedAssessmentId ?? undefined} hasReferenceAudio={false} onCostUpdate={handleWordCost} />

            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Clock className="size-3" />
                {new Date().toLocaleString("en-HK", {
                  timeZone: "Asia/Hong_Kong",
                  year: "numeric",
                  month: "2-digit",
                  day: "2-digit",
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                  hour12: false,
                })}
              </span>
              <Badge variant="outline">{at.score.replace("{score}", String(Math.round(result.scores.PronScore)))}</Badge>
              <Badge variant="outline">
                {at.duration.replace("{seconds}", String(Math.round(result.durationMs / 1000)))}
              </Badge>
              {savedCost !== null && (
                <span className="flex items-center gap-1">
                  <Coins className="size-3" />
                  HK${savedCost.toFixed(2)}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <Link href="/history?tab=assessment" className="block">
        <Button variant="outline" size="lg" className="w-full">
          <History className="size-4" />
          {at.viewHistory}
        </Button>
      </Link>
    </div>
  );
}
