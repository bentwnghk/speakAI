"use client";

import { useState, useCallback } from "react";
import { toast } from "sonner";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
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
import { ErrorSummary } from "./error-summary";
import { AssessmentHistory } from "./assessment-history";
import type {
  AssessmentResult,
  AssessmentDetailResult,
  RecordingMode,
  RecordingState,
  WordResult,
  ErrorType,
  PronunciationScores,
} from "@/types/assessment";

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

export function AssessmentForm({ cost }: { cost: number }) {
  const { t } = useUserSettings();
  const { refreshBalance } = useCredits();
  const at = t.assessment as Record<string, string>;

  const [referenceText, setReferenceText] = useState("");
  const [mode, setMode] = useState<RecordingMode>("auto");
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [errorFilter, setErrorFilter] = useState<ErrorType | "All">("All");
  const [historyRefresh, setHistoryRefresh] = useState(0);

  const wordCount = referenceText.trim()
    ? referenceText.trim().split(/\s+/).length
    : 0;

  const handleStart = useCallback(async () => {
    if (!referenceText.trim()) return;

    setResult(null);
    setRecordingState("recording");

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

      const pronunciationConfig =
        SpeechSDK.PronunciationAssessmentConfig.fromJSON(
          JSON.stringify({
            referenceText: referenceText.trim(),
            gradingSystem: "HundredMark",
            granularity: "Phoneme",
            phonemeAlphabet: "IPA",
            nBestPhonemeCount: 5,
            enableMiscue: mode === "auto",
            enableProsodyAssessment: true,
          })
        );

      const recognizer = new SpeechSDK.SpeechRecognizer(
        speechConfig,
        audioConfig
      );

      pronunciationConfig.applyTo(recognizer);

      if (mode === "auto") {
        await recognizeOnce(recognizer, SpeechSDK);
      } else {
        await recognizeContinuous(recognizer, SpeechSDK);
      }
    } catch (err) {
      console.error("Assessment error:", err);
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
          recognizer.close();
          audioConfigCleanup();

          if (recResult.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
            setRecordingState("processing");
            processResult(recResult, SpeechSDK);
            setRecordingState("done");
            resolve();
          } else if (recResult.reason === SpeechSDK.ResultReason.NoMatch) {
            setRecordingState("idle");
            toast.error(at.noSpeech);
            resolve();
          } else {
            setRecordingState("idle");
            const cancel =
              SpeechSDK.CancellationDetails.fromResult(recResult);
            console.error("Recognition canceled:", cancel.errorDetails);
            reject(new Error("canceled"));
          }
        },
        (error: string) => {
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

      recognizer.recognized = (_s, e) => {
        if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
          allResults.push(e.result);
        }
      };

      recognizer.canceled = (_s, e) => {
        if (e.reason === SpeechSDK.CancellationReason.Error) {
          console.error("Continuous recognition error:", e.errorDetails);
        }
        recognizer.close();
        audioConfigCleanup();
        if (allResults.length > 0) {
          setRecordingState("processing");
          const combined = combineResults(allResults, SpeechSDK);
          processCombinedResult(combined);
          setRecordingState("done");
        } else {
          setRecordingState("idle");
          toast.error(at.noSpeech);
        }
        resolve();
      };

      recognizer.sessionStopped = () => {
        recognizer.stopContinuousRecognitionAsync(
          () => {},
          (err: string) => {
            console.error("Stop error:", err);
            reject(new Error(err));
          }
        );
      };

      recognizer.startContinuousRecognitionAsync(
        () => {},
        (err: string) => {
          recognizer.close();
          audioConfigCleanup();
          setRecordingState("idle");
          reject(new Error(err));
        }
      );

      window.__stopAssessment = () => {
        recognizer.stopContinuousRecognitionAsync(
          () => {},
          (err: string) => reject(new Error(err))
        );
      };
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
    const refWords = refText.toLowerCase().split(/\s+/);
    const recWords = recognizedWords.map((w) => w.Word.toLowerCase());

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
    SpeechSDK: SdkTypes
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
      durationMs: recResult.duration / 10000,
    };

    setResult(assessment);
    void saveAssessment(assessment);
  }

  function processCombinedResult(data: {
    words: WordResult[];
    scores: PronunciationScores;
    displayText: string;
    totalDuration: number;
  }) {
    const assessment: AssessmentResult = {
      detailResult: {} as AssessmentDetailResult,
      scores: data.scores,
      words: data.words,
      recognizedText: data.displayText,
      durationMs: data.totalDuration / 10000,
    };
    setResult(assessment);
    void saveAssessment(assessment);
  }

  async function saveAssessment(assessment: AssessmentResult) {
    try {
      const res = await fetch("/api/assessment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
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
          syllables: assessment.words.map((w) => w.Syllables || []),
        }),
      });

      if (res.status === 402) {
        toast.error(at.insufficientCredits);
        return;
      }

      if (res.ok) {
        const data = (await res.json()) as { cost: number };
        toast.success(at.saved.replace("${cost}", data.cost.toFixed(2)));
        void refreshBalance();
        setHistoryRefresh((n) => n + 1);
      } else {
        toast.error(at.saveFailed);
      }
    } catch {
      toast.error(at.saveFailed);
    }
  }

  function handleStop() {
    if (mode === "manual" && window.__stopAssessment) {
      window.__stopAssessment();
    }
  }

  function handleReset() {
    setResult(null);
    setRecordingState("idle");
    setErrorFilter("All");
  }

  function handleSample(key: string) {
    const sample = at[key];
    if (sample) setReferenceText(sample);
  }

  const filteredWords: WordResult[] =
    errorFilter === "All"
      ? result?.words ?? []
      : (result?.words ?? []).filter(
          (w) => w.PronunciationAssessment.ErrorType === errorFilter
        );

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold">{at.title}</h1>
        <p className="text-sm text-muted-foreground">{at.subtitle}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{at.referenceText}</span>
            <Select onValueChange={handleSample}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder={at.sampleTexts} />
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
          <CardDescription>
            <span>{at.referencePlaceholder}</span>
          </CardDescription>
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
              {at.estimatedCost.replace("${cost}", cost.toFixed(2))}
            </span>
          </div>

          <Separator />

          <div className="space-y-2">
            <p className="text-sm font-medium">{at.modeLabel}</p>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={mode === "auto" ? "default" : "outline"}
                onClick={() => setMode("auto")}
                disabled={recordingState === "recording"}
              >
                {at.modeAuto}
              </Button>
              <Button
                size="sm"
                variant={mode === "manual" ? "default" : "outline"}
                onClick={() => setMode("manual")}
                disabled={recordingState === "recording"}
              >
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

            <div className="space-y-3">
              <h3 className="text-sm font-semibold">{at.recognizedText}</h3>
              <TranscriptView words={result.words} t={at} />
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
                  <TabsTrigger value="fulltext">{at.granFullText}</TabsTrigger>
                  <TabsTrigger value="word">{at.granWord}</TabsTrigger>
                  <TabsTrigger value="phoneme">{at.granPhoneme}</TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="fulltext">
                <div className="rounded-lg border p-4">
                  <ScoreOverview scores={result.scores} t={at} />
                </div>
              </TabsContent>

              <TabsContent value="word">
                <div className="max-h-96 overflow-y-auto">
                  <WordDetail words={filteredWords} t={at} />
                </div>
              </TabsContent>

              <TabsContent value="phoneme">
                <div className="max-h-96 overflow-y-auto">
                  <WordDetail words={filteredWords} t={at} />
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>{at.history}</CardTitle>
        </CardHeader>
        <CardContent>
          <AssessmentHistory t={at} onRefresh={historyRefresh} />
        </CardContent>
      </Card>
    </div>
  );
}
