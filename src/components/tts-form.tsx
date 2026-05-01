"use client";

import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VoiceSelect } from "@/components/voice-select";
import { SpeedSlider } from "@/components/speed-slider";
import { FileUpload } from "@/components/file-upload";
import { AudioPlayer } from "@/components/audio-player";
import { KaraokeText } from "@/components/karaoke-text";
import { Sparkles, Type, Upload, Loader2, History, FileText, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";
import type { Segment } from "@/types/karaoke";
import { processPdf } from "@/lib/pdf-client";

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

  const handleFilesSelected = useCallback(async (newFiles: File[]) => {
    if (newFiles.length === 0) {
      setExtractedText("");
      setUploadedFileName("");
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
          throw new Error(err.error ?? "Text extraction failed");
        }
        const data = (await res.json()) as { text: string };
        allTexts.push(data.text);
      }

      for (const pdfFile of pdfFiles) {
        try {
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
                throw new Error(err.error ?? "OCR failed for scanned PDF page");
              }
              const data = (await res.json()) as { text: string };
              allTexts.push(data.text);
            }
          }
        } catch {
          const formData = new FormData();
          formData.append("file", pdfFile);
          const res = await fetch("/api/extract-text", {
            method: "POST",
            body: formData,
          });
          if (!res.ok) {
            const err = (await res.json()) as { error?: string };
            throw new Error(err.error ?? "Text extraction failed");
          }
          const data = (await res.json()) as { text: string };
          allTexts.push(data.text);
        }
      }

      const combined = allTexts.filter(Boolean).join("\n\n");
      setExtractedText(combined);
      toast.success(`Extracted text from ${newFiles.length} file(s)`);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to extract text"
      );
      setExtractedText("");
    } finally {
      setIsExtracting(false);
    }
  }, []);

  const handleGenerate = async () => {
    const inputText = inputMethod === "text" ? text : extractedText;

    if (!inputText?.trim()) {
      toast.error("Please enter text or upload files first");
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
          title:
            inputMethod === "upload" && uploadedFileName
              ? `Audio - ${uploadedFileName}`
              : undefined,
        }),
      });

      if (!res.ok) {
        const err = (await res.json()) as { error?: string };
        throw new Error(err.error || "Generation failed");
      }

      const data = (await res.json()) as Generation;
      setAudioSrc(data.audioUrl);
      setAudioTitle(data.title);
      setAudioCreatedAt(data.createdAt);
      setAudioSegments(data.segments ?? []);

      toast.success(
        `Audio generated! ${data.ttsCost ? `Cost: HK$${data.ttsCost}` : ""}`,
        { duration: 5000 }
      );
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Audio generation failed"
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const displayText = inputMethod === "upload" ? extractedText : text;
  const showKaraoke = isAudioPlaying && audioSegments.length > 0 && audioSrc;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="size-4" />
              Source
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
                    Text
                  </TabsTrigger>
                  <TabsTrigger value="upload" className="flex-1">
                    <Upload className="size-4" />
                    Files
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="mt-3">
                  <Textarea
                    placeholder="Paste or type the text to read aloud here..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={10}
                  />
                </TabsContent>

                <TabsContent value="upload" className="mt-3">
                  <FileUpload
                    onFilesSelected={handleFilesSelected}
                    isExtracting={isExtracting}
                  />
                  {extractedText && (
                    <div className="mt-3">
                      <p className="text-xs text-muted-foreground mb-1">
                        Edit the extracted text as needed before generating:
                      </p>
                      <Textarea
                        value={extractedText}
                        onChange={(e) => setExtractedText(e.target.value)}
                        rows={10}
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
              Settings
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
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="size-4" />
              Generate Audio
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
        />

        <Link href="/history" className="block">
          <Button variant="outline" className="w-full">
            <History className="size-4" />
            View Audio History
          </Button>
        </Link>
      </div>
    </div>
  );
}
