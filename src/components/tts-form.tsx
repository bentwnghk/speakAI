"use client";

import { useState, useCallback, useEffect, useRef } from "react";
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
  const [selectionPopover, setSelectionPopover] = useState<{
    text: string;
    top: number;
    left: number;
  } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
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

  const handleTextSelect = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const sel = el.value.substring(el.selectionStart, el.selectionEnd).trim();
    if (!sel) {
      setSelectionPopover(null);
      return;
    }
    const rect = el.getBoundingClientRect();
    const cardContent = el.closest("[data-selection-container]");
    const containerRect = cardContent
      ? cardContent.getBoundingClientRect()
      : rect;
    const style = getComputedStyle(el);
    const lineH = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
    const paddingTop = parseFloat(style.paddingTop) || 0;
    const textBefore = el.value.substring(0, el.selectionStart);
    const linesBefore = textBefore.split("\n").length - 1;
    const charsPerLine = Math.max(1, Math.floor((rect.width - 24) / (parseFloat(style.fontSize) * 0.6)));
    const wrappedLines = textBefore.split("\n").reduce(
      (acc, line) => acc + Math.max(1, Math.ceil(line.length / charsPerLine)),
      0,
    ) - 1;
    const topOffset = paddingTop + (linesBefore + wrappedLines) * lineH;
    const top = topOffset - el.scrollTop;
    setSelectionPopover({
      text: sel,
      top: top - 8 + rect.top - containerRect.top,
      left: rect.width / 2,
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    setTimeout(handleTextSelect, 10);
  }, [handleTextSelect]);

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
                  <div className="relative" data-selection-container>
                    <Textarea
                      ref={textareaRef}
                      placeholder={t.tts.textPlaceholder}
                      value={text}
                      onChange={(e) => {
                        setText(e.target.value);
                        setSelectionPopover(null);
                      }}
                      onMouseUp={handleMouseUp}
                      onKeyUp={handleMouseUp}
                      onBlur={() => setSelectionPopover(null)}
                      rows={10}
                      className="text-base md:text-base"
                    />
                    {selectionPopover && (
                      <Link
                        href={`/assessment?text=${encodeURIComponent(selectionPopover.text)}`}
                        className="absolute z-10 -translate-x-1/2 -translate-y-full"
                        style={{ top: selectionPopover.top, left: selectionPopover.left }}
                      >
                        <Button
                          size="sm"
                          className="shadow-lg whitespace-nowrap"
                          onMouseDown={(e) => e.preventDefault()}
                        >
                          <Mic className="size-3.5" />
                          {t.tts.practiceReading}
                        </Button>
                      </Link>
                    )}
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
