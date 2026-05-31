"use client";

import { useState, useCallback, useEffect, useRef, useLayoutEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VoiceSelect } from "@/components/voice-select";
import { SpeedSlider } from "@/components/speed-slider";
import { FileUpload } from "@/components/file-upload";
import { AudioPlayer } from "@/components/audio-player";
import { KaraokeText } from "@/components/karaoke-text";
import { Sparkles, Type, Upload, Loader2, History, FileText, SlidersHorizontal, Mic } from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { Segment } from "@/types/karaoke";
import { processPdf } from "@/lib/pdf-client";
import { useCredits } from "@/hooks/use-credits";
import { useUserSettings } from "@/hooks/use-settings";

interface Generation {
  id: string;
  title: string;
  transcript: string;
  voice: string;
  speed: number;
  audioUrl: string;
  segments?: Segment[];
  ttsCost: string | null;
  createdAt: string;
}

export function TtsForm() {
  const router = useRouter();
  const [inputMethod, setInputMethod] = useState<"text" | "upload">("text");
  const [text, setText] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [voice, setVoice] = useState("Female 1");
  const [speed, setSpeed] = useState(100);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [audioTitle, setAudioTitle] = useState<string>();
  const [audioCreatedAt, setAudioCreatedAt] = useState<string>();
  const [audioSegments, setAudioSegments] = useState<Segment[]>([]);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [karaokeActive, setKaraokeActive] = useState(false);
  const [accumulatedVisionCost, setAccumulatedVisionCost] = useState(0);
  const [selection, setSelection] = useState<{
    text: string;
    x: number;
    y: number;
    above: boolean;
  } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const { refreshBalance } = useCredits();
  const { t } = useUserSettings();

  const handleFilesSelected = useCallback(async (newFiles: File[]) => {
    if (newFiles.length === 0) {
      setExtractedText("");
      setUploadedFileName("");
      setAccumulatedVisionCost(0);
      return;
    }

    setUploadedFileName(newFiles[0].name.replace(/\.[^.]+$/, ""));

    setIsExtracting(true);
    try {
      const allTexts: string[] = [];
      const pdfFiles = newFiles.filter(
        (f) =>
          f.type === "application/pdf" ||
          f.name.toLowerCase().endsWith(".pdf")
      );
      const nonPdfFiles = newFiles.filter((f) => !pdfFiles.includes(f));

      for (const file of nonPdfFiles) {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/api/extract-text", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const err = (await res.json()) as { error?: string };
          throw new Error(err.error ?? t.tts.extractFailed);
        }
        const data = (await res.json()) as { text: string; visionCost?: number };
        allTexts.push(data.text);
        if (data.visionCost) {
          setAccumulatedVisionCost((prev) => prev + data.visionCost!);
        }
      }

      for (const pdfFile of pdfFiles) {
        const result = await processPdf(pdfFile);
        if (result.text) {
          allTexts.push(result.text);
        } else if (result.images.length > 0) {
          for (const img of result.images) {
            const formData = new FormData();
            formData.append("file", img);
            const res = await fetch("/api/extract-text", {
              method: "POST",
              body: formData,
            });
            if (!res.ok) {
              const err = (await res.json()) as { error?: string };
              throw new Error(err.error ?? t.tts.ocrFailed);
            }
            const data = (await res.json()) as { text: string; visionCost?: number };
            allTexts.push(data.text);
            if (data.visionCost) {
              setAccumulatedVisionCost((prev) => prev + data.visionCost!);
            }
          }
        }
      }

      const combined = allTexts.filter(Boolean).join("\n\n");
      setExtractedText(combined);
      toast.success(
        t.tts.extractSuccess.replace("{count}", String(newFiles.length))
      );
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t.tts.extractFailed
      );
      setExtractedText("");
    } finally {
      setIsExtracting(false);
    }
  }, [t]);

  const handleGenerate = async () => {
    const inputText = inputMethod === "text" ? text : extractedText;

    if (!inputText?.trim()) {
      toast.error(t.tts.pleaseEnterText);
      return;
    }

    setIsGenerating(true);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText,
          voice,
          speed,
          visionCost: inputMethod === "upload" ? accumulatedVisionCost : 0,
          title:
            inputMethod === "upload" && uploadedFileName
              ? uploadedFileName
              : undefined,
        }),
      });

      if (!res.ok) {
        const err = (await res.json()) as { error?: string };
        throw new Error(err.error || t.tts.generationFailed);
      }

      const data = (await res.json()) as Generation;
      setAudioSrc(data.audioUrl);
      setAudioTitle(data.title);
      setAudioCreatedAt(data.createdAt);
      setAudioSegments(data.segments ?? []);

      toast.success(
        `${t.tts.audioGenerated}${data.ttsCost ? ` ${t.tts.cost.replace("${cost}", data.ttsCost)}` : ""}`,
        { duration: 5000 }
      );
      void refreshBalance();
      setAccumulatedVisionCost(0);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t.tts.generationFailed
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const displayText = inputMethod === "upload" ? extractedText : text;
  const showKaraoke = karaokeActive && audioSegments.length > 0 && audioSrc;

  const handleSelectionChange = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const selectedText = el.value.substring(el.selectionStart, el.selectionEnd).trim();
    if (!selectedText || selectedText.length === 0 || selectedText.length > 4096) {
      setSelection(null);
      return;
    }

    const mirror = mirrorRef.current;
    if (!mirror) return;

    const textNode = mirror.firstChild;
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) {
      setSelection(null);
      return;
    }

    try {
      const range = document.createRange();
      const start = Math.min(el.selectionStart, textNode.textContent?.length ?? 0);
      const end = Math.min(el.selectionEnd, textNode.textContent?.length ?? 0);
      range.setStart(textNode, start);
      range.setEnd(textNode, end);
      const rect = range.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) {
        setSelection(null);
        return;
      }
      const isIOS =
        /iPad|iPhone|iPod/.test(navigator.userAgent) ||
        (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
      const showAbove = isIOS && rect.top < window.innerHeight / 2;
      setSelection({
        text: selectedText,
        x: rect.left + rect.width / 2,
        y: showAbove ? rect.top - 8 : rect.bottom + 8,
        above: showAbove,
      });
    } catch {
      setSelection(null);
    }
  }, []);

  useLayoutEffect(() => {
    const popup = popupRef.current;
    if (!popup || !selection) return;
    const popupWidth = popup.offsetWidth;
    const MARGIN = 8;
    const desiredLeft = selection.x - popupWidth / 2;
    const left = Math.max(MARGIN, Math.min(window.innerWidth - popupWidth - MARGIN, desiredLeft));
    popup.style.left = `${left}px`;
    popup.style.transform = selection.above ? "translateY(-100%)" : "none";
  }, [selection]);

  const handleDismiss = useCallback((e: MouseEvent | TouchEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest(".selection-popup")) {
      setSelection(null);
    }
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const debounced = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(handleSelectionChange, 150);
    };

    document.addEventListener("selectionchange", debounced);
    document.addEventListener("mousedown", handleDismiss);
    document.addEventListener("touchstart", handleDismiss, { passive: true });

    return () => {
      document.removeEventListener("selectionchange", debounced);
      document.removeEventListener("mousedown", handleDismiss);
      document.removeEventListener("touchstart", handleDismiss);
      if (timer) clearTimeout(timer);
    };
  }, [handleSelectionChange, handleDismiss]);

  useEffect(() => {
    if (isAudioPlaying) {
      setKaraokeActive(true);
    }
  }, [isAudioPlaying]);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="size-4" />
              {t.tts.source}
            </CardTitle>
          </CardHeader>
          <CardContent className="max-h-[50vh] overflow-y-auto">
            {showKaraoke ? (
              <KaraokeText
                text={displayText}
                segments={audioSegments}
                currentTime={audioCurrentTime}
                isPlaying={isAudioPlaying}
              />
            ) : (
              <Tabs
                value={inputMethod}
                onValueChange={(v) => setInputMethod(v as "text" | "upload")}
              >
                <TabsList className="w-full">
                  <TabsTrigger value="text" className="flex-1">
                    <Type className="size-4" />
                    {t.tts.text}
                  </TabsTrigger>
                  <TabsTrigger value="upload" className="flex-1">
                    <Upload className="size-4" />
                    {t.tts.files}
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="mt-3">
                  <div className="relative">
                    <div
                      ref={mirrorRef}
                      aria-hidden="true"
                      className="absolute inset-0 overflow-hidden pointer-events-none whitespace-pre-wrap break-words text-base md:text-base p-3 border border-transparent"
                      style={{ visibility: "hidden" }}
                    >
                      {text}
                    </div>
                    <Textarea
                      ref={textareaRef}
                      placeholder={t.tts.textPlaceholder}
                      value={text}
                      onChange={(e) => {
                        setText(e.target.value);
                        setSelection(null);
                      }}
                      rows={10}
                      className="text-base md:text-base"
                    />
                  </div>
                </TabsContent>

                <TabsContent value="upload" className="mt-3">
                  <FileUpload
                    onFilesSelected={handleFilesSelected}
                    isExtracting={isExtracting}
                  />
                  {extractedText && (
                    <div className="mt-3">
                      <p className="text-xs text-muted-foreground mb-1">
                        {t.tts.editExtracted}
                      </p>
                      <Textarea
                        value={extractedText}
                        onChange={(e) => setExtractedText(e.target.value)}
                        rows={10}
                        className="text-base md:text-base"
                      />
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>

        {selection && (
          <div
            ref={popupRef}
            className="selection-popup fixed z-[9999] shadow-md flex gap-0.5 bg-background border rounded-md p-0.5"
            style={{
              left: selection.x,
              top: selection.y,
              transform: selection.above ? "translate(-50%, -100%)" : "translateX(-50%)",
            }}
          >
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                router.push(`/assessment?text=${encodeURIComponent(selection.text)}`);
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                router.push(`/assessment?text=${encodeURIComponent(selection.text)}`);
              }}
            >
              <Mic className="h-4 w-4" />
              <span className="hidden sm:inline">{t.tts.practiceReading}</span>
            </Button>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <SlidersHorizontal className="size-4" />
              {t.tts.settings}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <VoiceSelect value={voice} onValueChange={setVoice} />
            <SpeedSlider value={speed} onValueChange={setSpeed} />
          </CardContent>
        </Card>

        <Button
          className="w-full"
          size="lg"
          onClick={() => void handleGenerate()}
          disabled={isGenerating || !displayText?.trim()}
        >
          {isGenerating ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              {t.tts.generating}
            </>
          ) : (
            <>
              <Sparkles className="size-4" />
              {t.tts.generate}
            </>
          )}
        </Button>
      </div>

      <div className="space-y-4">
        <AudioPlayer
          src={audioSrc}
          title={audioTitle}
          createdAt={audioCreatedAt}
          onTimeUpdate={setAudioCurrentTime}
          onPlayStateChange={setIsAudioPlaying}
          onStop={() => setKaraokeActive(false)}
          onEnded={() => setKaraokeActive(false)}
        />

        <Link href="/history" className="block">
          <Button variant="outline" size="lg" className="w-full">
            <History className="size-4" />
            {t.tts.viewHistory}
          </Button>
        </Link>
      </div>
    </div>
  );
}
