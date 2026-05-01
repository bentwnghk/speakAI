/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-return */
const MIN_TEXT_LENGTH = 50;
const BLANK_THRESHOLD = 0.001;
const MAX_SAMPLE_PIXELS = 10000;

export interface PdfProcessResult {
  text?: string;
  images: File[];
  fileName: string;
}

function isBlankCanvas(canvas: HTMLCanvasElement): boolean {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width;
  const h = canvas.height;
  const totalPixels = w * h;
  const sampleCount = Math.min(MAX_SAMPLE_PIXELS, totalPixels);
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  let nonWhite = 0;

  for (let i = 0; i < sampleCount; i++) {
    const idx = Math.floor((i * totalPixels) / sampleCount);
    const offset = idx * 4;
    if (
      data[offset] < 250 ||
      data[offset + 1] < 250 ||
      data[offset + 2] < 250
    ) {
      nonWhite++;
    }
  }

  return nonWhite / sampleCount < BLANK_THRESHOLD;
}

async function extractTextLayer(pdf: any): Promise<string> {
  const texts: string[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item: any) => item.str)
      .join(" ")
      .trim();
    if (pageText) texts.push(pageText);
  }

  return texts.join("\n\n").trim();
}

async function renderPagesToImages(pdf: any): Promise<File[]> {
  const images: File[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 2 });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext("2d")!;

    await page.render({ canvasContext: ctx, viewport }).promise;

    if (!isBlankCanvas(canvas)) {
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), "image/png");
      });
      images.push(
        new File([blob], `page-${i}.png`, { type: "image/png" })
      );
    }
  }

  return images;
}

export async function processPdf(file: File): Promise<PdfProcessResult> {
  const pdfjsLib = (await import("pdfjs-dist")) as any;
  pdfjsLib.GlobalWorkerOptions.workerSrc = "/scripts/pdf.worker.min.mjs";

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  const text = await extractTextLayer(pdf);

  if (text.length >= MIN_TEXT_LENGTH) {
    return { text, images: [], fileName: file.name };
  }

  const images = await renderPagesToImages(pdf);
  return { text: undefined, images, fileName: file.name };
}
