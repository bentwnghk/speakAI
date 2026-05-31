"use client";

import {
  useState,
  useCallback,
  useEffect,
  useRef,
  useLayoutEffect,
} from "react";

export interface SelectionState {
  text: string;
  x: number;
  y: number;
  above: boolean;
}

export function useTextSelectionPopover() {
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  const handleSelectionChange = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const selectedText = el.value
      .substring(el.selectionStart, el.selectionEnd)
      .trim();
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
      const start = Math.min(
        el.selectionStart,
        textNode.textContent?.length ?? 0,
      );
      const end = Math.min(
        el.selectionEnd,
        textNode.textContent?.length ?? 0,
      );
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
    const left = Math.max(
      MARGIN,
      Math.min(window.innerWidth - popupWidth - MARGIN, desiredLeft),
    );
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

  const clearSelection = useCallback(() => setSelection(null), []);

  return {
    selection,
    textareaRef,
    mirrorRef,
    popupRef,
    clearSelection,
  };
}
