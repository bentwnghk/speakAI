"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { WordResult, NBestPhoneme } from "@/types/assessment";
import { PlayWordButton } from "./play-word-button";

interface PhonemeAnalysisProps {
  words: WordResult[];
  t: Record<string, string>;
  onCostUpdate?: (cost: number) => void;
}

type SortMode = "worst" | "best" | "freq";
type FilterMode = "all" | "low" | "confused";

interface PhonemeInstance {
  phoneme: string;
  accuracyRaw: number;
  word: string;
  spoken: string | null;
  spokenScore: number | null;
  nbest: NBestPhoneme[];
}

interface PhonemeAggregate {
  phoneme: string;
  count: number;
  avgAccuracy: number;
  minAccuracy: number;
  confusionCount: number;
  instances: PhonemeInstance[];
}

interface ConfusionPair {
  expected: string;
  spoken: string;
  count: number;
  avgScore: number;
  words: string[];
}

function getAccuracyBarColor(score: number): string {
  if (score >= 80) return "bg-green-500";
  if (score >= 60) return "bg-yellow-500";
  return "bg-red-500";
}

function getAccuracyTextColor(score: number): string {
  if (score >= 80) return "text-green-600 dark:text-green-400";
  if (score >= 60) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

export function PhonemeAnalysis({ words, t, onCostUpdate }: PhonemeAnalysisProps) {
  const [sortMode, setSortMode] = useState<SortMode>("worst");
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [expanded, setExpanded] = useState<string | null>(null);

  const aggregates = useMemo<PhonemeAggregate[]>(() => {
    const map = new Map<string, PhonemeAggregate>();
    for (const w of words) {
      if (!w.Phonemes) continue;
      for (const p of w.Phonemes) {
        const raw = p.PronunciationAssessment.AccuracyScore;
        const nbest = p.PronunciationAssessment.NBestPhonemes ?? [];
        const top = nbest[0];
        const isConfused = !!top && top.Phoneme !== p.Phoneme;

        let agg = map.get(p.Phoneme);
        if (!agg) {
          agg = {
            phoneme: p.Phoneme,
            count: 0,
            avgAccuracy: 0,
            minAccuracy: Infinity,
            confusionCount: 0,
            instances: [],
          };
          map.set(p.Phoneme, agg);
        }
        agg.count++;
        const acc = Number.isFinite(raw) ? raw : 0;
        agg.avgAccuracy += acc;
        if (acc < agg.minAccuracy) agg.minAccuracy = acc;
        if (isConfused) agg.confusionCount++;
        agg.instances.push({
          phoneme: p.Phoneme,
          accuracyRaw: raw,
          word: w.Word,
          spoken: isConfused ? top.Phoneme : null,
          spokenScore: isConfused ? top.Score : null,
          nbest,
        });
      }
    }
    for (const agg of map.values()) {
      agg.avgAccuracy = agg.count > 0 ? agg.avgAccuracy / agg.count : 0;
      if (!Number.isFinite(agg.minAccuracy)) agg.minAccuracy = 0;
    }
    return Array.from(map.values());
  }, [words]);

  const confusions = useMemo<ConfusionPair[]>(() => {
    const map = new Map<string, ConfusionPair>();
    for (const w of words) {
      if (!w.Phonemes) continue;
      for (const p of w.Phonemes) {
        const nbest = p.PronunciationAssessment.NBestPhonemes ?? [];
        const top = nbest[0];
        if (!top || top.Phoneme === p.Phoneme) continue;
        const key = `${p.Phoneme}>${top.Phoneme}`;
        let pair = map.get(key);
        if (!pair) {
          pair = {
            expected: p.Phoneme,
            spoken: top.Phoneme,
            count: 0,
            avgScore: 0,
            words: [],
          };
          map.set(key, pair);
        }
        pair.count++;
        pair.avgScore += top.Score;
        pair.words.push(w.Word);
      }
    }
    for (const pair of map.values()) {
      pair.avgScore = pair.count > 0 ? pair.avgScore / pair.count : 0;
    }
    return Array.from(map.values()).sort((a, b) => b.count - a.count);
  }, [words]);

  const sortedAggregates = useMemo(() => {
    const copy = [...aggregates];
    if (sortMode === "worst") copy.sort((a, b) => a.avgAccuracy - b.avgAccuracy);
    else if (sortMode === "best") copy.sort((a, b) => b.avgAccuracy - a.avgAccuracy);
    else copy.sort((a, b) => b.count - a.count || a.avgAccuracy - b.avgAccuracy);
    return copy;
  }, [aggregates, sortMode]);

  const visibleAggregates = useMemo(() => {
    if (filterMode === "low") return sortedAggregates.filter((a) => a.avgAccuracy < 80);
    if (filterMode === "confused") return sortedAggregates.filter((a) => a.confusionCount > 0);
    return sortedAggregates;
  }, [sortedAggregates, filterMode]);

  if (aggregates.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        {t.phonemeNoData ?? "No phoneme data available."}
      </p>
    );
  }

  const totalInstances = aggregates.reduce((s, a) => s + a.count, 0);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            {t.phonemeSortLabel ?? "Sort"}
          </span>
          <Select value={sortMode} onValueChange={(v) => setSortMode(v as SortMode)}>
            <SelectTrigger className="h-8 w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="worst">{t.phonemeSortWorst ?? "Worst accuracy"}</SelectItem>
              <SelectItem value="best">{t.phonemeSortBest ?? "Best accuracy"}</SelectItem>
              <SelectItem value="freq">{t.phonemeSortFreq ?? "Most frequent"}</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <FilterChip
            active={filterMode === "all"}
            onClick={() => setFilterMode("all")}
            label={`${t.phonemeFilterAll ?? "All"} (${totalInstances})`}
          />
          <FilterChip
            active={filterMode === "low"}
            onClick={() => setFilterMode("low")}
            label={t.phonemeFilterLow ?? "Below 80%"}
            className="bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-500/20"
          />
          <FilterChip
            active={filterMode === "confused"}
            onClick={() => setFilterMode("confused")}
            label={t.phonemeFilterConfused ?? "Confused"}
            className="bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20"
          />
        </div>
      </div>

      {confusions.length > 0 && (
        <div className="space-y-2 rounded-lg border bg-card p-3">
          <p className="text-xs font-semibold text-muted-foreground">
            {t.phonemeConfusions ?? "Common Confusions"}
          </p>
          <div className="flex flex-wrap gap-2">
            {confusions.slice(0, 8).map((pair, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 rounded-full border bg-background px-2.5 py-1 text-xs"
                title={`${t.phonemeSources ?? "In words"}: ${pair.words.join(", ")}`}
              >
                <span className="font-mono font-medium">/{pair.expected}/</span>
                <span className="text-muted-foreground">&rarr;</span>
                <span className="font-mono text-red-600 dark:text-red-400">/{pair.spoken}/</span>
                <span className="rounded-full bg-muted px-1.5 tabular-nums text-muted-foreground">
                  &times;{pair.count}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-1">
        {visibleAggregates.length === 0 && (
          <p className="py-4 text-center text-sm text-muted-foreground">
            {t.phonemeNoData ?? "No phoneme data available."}
          </p>
        )}
        {visibleAggregates.map((agg) => {
          const avg = Math.round(agg.avgAccuracy);
          const worst = agg.instances.reduce((min, inst) => {
            const acc = Number.isFinite(inst.accuracyRaw) ? inst.accuracyRaw : 0;
            return acc < (Number.isFinite(min.accuracyRaw) ? min.accuracyRaw : 0) ? inst : min;
          }, agg.instances[0]);
          const worstScore = Number.isFinite(worst.accuracyRaw) ? Math.round(worst.accuracyRaw) : null;
          const isOpen = expanded === agg.phoneme;

          return (
            <div key={agg.phoneme}>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left text-sm transition-colors hover:bg-accent/50",
                  isOpen && "bg-accent/30"
                )}
                onClick={() => setExpanded(isOpen ? null : agg.phoneme)}
              >
                <span className="shrink-0 text-muted-foreground">
                  {isOpen ? (
                    <ChevronDown className="size-4" />
                  ) : (
                    <ChevronRight className="size-4" />
                  )}
                </span>

                <span className="w-14 shrink-0 font-mono text-base font-semibold text-foreground">
                  /{agg.phoneme}/
                </span>

                <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs tabular-nums text-muted-foreground">
                  {(t.phonemeOccurrences ?? "{count}\u00d7").replace("{count}", String(agg.count))}
                </span>

                <div className="flex-1">
                  <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className={cn("h-full rounded-full transition-all duration-500", getAccuracyBarColor(avg))}
                      style={{ width: `${avg}%` }}
                    />
                  </div>
                </div>

                <span className={cn("w-8 text-right tabular-nums font-semibold", getAccuracyTextColor(avg))}>
                  {avg}
                </span>

                {worstScore !== null && (
                  <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
                    {t.phonemeWorstIn ?? "Worst in"} {worst.word} ({worstScore})
                  </span>
                )}
              </button>

              {isOpen && (
                <div className="ml-6 mt-1 space-y-1 rounded-lg border bg-card p-3">
                  {agg.confusionCount > 0 && (
                    <p className="text-xs text-muted-foreground">
                      {(t.phonemeConfusionCount ?? "{count} confused").replace(
                        "{count}",
                        String(agg.confusionCount),
                      )}
                    </p>
                  )}
                  {agg.instances.map((inst, j) => {
                    const acc = Number.isFinite(inst.accuracyRaw) ? Math.round(inst.accuracyRaw) : null;
                    const displayAcc = acc ?? 0;
                    return (
                      <div
                        key={j}
                        className="flex items-center gap-2 rounded border bg-background px-2 py-1.5 text-sm"
                      >
                        <span className="w-20 shrink-0 truncate font-medium text-foreground">
                          {inst.word}
                        </span>

                        <div className="flex-1">
                          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                            <div
                              className={cn(
                                "h-full rounded-full transition-all duration-500",
                                getAccuracyBarColor(displayAcc),
                              )}
                              style={{ width: `${displayAcc}%` }}
                            />
                          </div>
                        </div>

                        <span
                          className={cn(
                            "w-8 text-right tabular-nums font-semibold",
                            getAccuracyTextColor(displayAcc),
                          )}
                        >
                          {acc ?? "\u2013"}
                        </span>

                        {inst.spoken && (
                          <span
                            className="shrink-0 text-xs text-red-600 dark:text-red-400"
                            title={
                              inst.nbest
                                .slice(0, 3)
                                .map(
                                  (nb) =>
                                    `/${nb.Phoneme}/ ${Math.round(nb.Score)}%`,
                                )
                                .join("  ")
                            }
                          >
                            {t.phonemeConfusedAs ?? "said as"} /{inst.spoken}/
                          </span>
                        )}

                        <PlayWordButton
                          word={inst.word}
                          label={t.playPronunciation}
                          costLabel={t.wordPronunciationCost}
                          onCostUpdate={onCostUpdate}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  label,
  className,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-primary text-primary-foreground"
          : className ?? "bg-muted text-muted-foreground hover:bg-accent",
      )}
    >
      {label}
    </button>
  );
}
