"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { FileUp, X, Loader2, FileText, Image as ImageIcon } from "lucide-react";
import { SUPPORTED_EXTENSIONS } from "@/lib/constants";

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void | Promise<void>;
  isExtracting: boolean;
}

export function FileUpload({ onFilesSelected, isExtracting }: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const acceptStr = SUPPORTED_EXTENSIONS.join(",");

  const handleFiles = useCallback(
    (newFiles: FileList | File[]) => {
      const fileArray = Array.from(newFiles);
      setFiles((prev) => {
        const updated = [...prev, ...fileArray];
        void onFilesSelected(updated);
        return updated;
      });
    },
    [onFilesSelected]
  );

  const removeFile = (index: number) => {
    setFiles((prev) => {
      const updated = prev.filter((_, i) => i !== index);
      void onFilesSelected(updated);
      return updated;
    });
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const getFileIcon = (name: string) => {
    const ext = name.split(".").pop()?.toLowerCase() || "";
    const imageExts = ["jpg", "jpeg", "png"];
    if (imageExts.includes(ext)) return <ImageIcon className="size-4" />;
    return <FileText className="size-4" />;
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50"
        }`}
      >
        {isExtracting ? (
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <Loader2 className="size-8 animate-spin" />
            <p className="text-sm">Extracting text from files...</p>
          </div>
        ) : (
          <>
            <FileUp className="size-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground text-center">
              Drag & drop files or{" "}
              <label className="text-primary cursor-pointer hover:underline">
                browse
                <input
                  type="file"
                  multiple
                  accept={acceptStr}
                  className="hidden"
                  onChange={(e) => e.target.files && handleFiles(e.target.files)}
                />
              </label>
            </p>
            <p className="text-xs text-muted-foreground/70 mt-1">
              PDF, DOCX, TXT, JPG, PNG
            </p>
          </>
        )}
      </div>

      {files.length > 0 && (
        <div className="space-y-1">
          {files.map((file, i) => (
            <div
              key={`${file.name}-${i}`}
              className="flex items-center gap-2 rounded-md bg-muted/50 px-3 py-1.5 text-sm"
            >
              {getFileIcon(file.name)}
              <span className="truncate flex-1">{file.name}</span>
              <Button
                variant="ghost"
                size="icon"
                className="size-6 shrink-0"
                onClick={() => removeFile(i)}
              >
                <X className="size-3" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
