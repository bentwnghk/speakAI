"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Loader2,
  Volume2,
  ShieldCheck,
  ShoppingCart,
  LogIn,
  ChevronLeft,
  ChevronRight,
  Mic,
  Clock,
  Coins,
} from "lucide-react";
import { useUserSettings } from "@/hooks/use-settings";
import { AudioPlayer } from "@/components/audio-player";
import { ScoreOverview } from "@/components/assessment/score-overview";
import { TranscriptView } from "@/components/assessment/transcript-view";
import { ErrorSummary } from "@/components/assessment/error-summary";
import { ReferenceAudioSection } from "@/components/assessment/reference-audio-section";
import {
  WordDetail,
  SyllableView,
} from "@/components/assessment/word-detail";
import { PhonemeAnalysis } from "@/components/assessment/phoneme-analysis";
import { FeedbackCard } from "@/components/assessment/feedback-card";
import { Separator } from "@/components/ui/separator";
import type { SavedAssessment } from "@/types/assessment";
import { type AssessmentFilter, filterWords } from "@/types/assessment";
import { cn } from "@/lib/utils";

const PAGE_SIZES = [10, 20, 30, 50, 100] as const;
const DEFAULT_PER_PAGE = 20;

interface GenerationRow {
  id: string;
  userName: string | null;
  email: string | null;
  title: string;
  voice: string;
  createdAt: string;
  ttsCost: string | null;
  cumulativeCost: string;
}

interface PurchaseRow {
  id: string;
  userName: string | null;
  email: string | null;
  planName: string;
  creditsAmount: number;
  amountHKD: number;
  status: string;
  createdAt: string;
}

interface SignInRow {
  id: string;
  userName: string | null;
  email: string | null;
  provider: string;
  createdAt: string;
}

interface AssessmentRow {
  id: string;
  userName: string | null;
  email: string | null;
  referenceText: string;
  durationMs: number;
  pronScore: number;
  accuracyScore: number;
  fluencyScore: number;
  completenessScore: number;
  prosodyScore: number | null;
  cost: number;
  createdAt: string;
  cumulativeCost: string;
}

type GenerationSortKey = "userName" | "createdAt" | "title" | "voice";
type PurchaseSortKey = "userName" | "createdAt" | "planName" | "amountHKD";
type SignInSortKey = "userName" | "createdAt";
type AssessmentSortKey = "userName" | "createdAt" | "pronScore" | "cost";

function formatDateHK(dateStr: string): string {
  return new Date(dateStr).toLocaleString("en-HK", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SortIcon({ active, desc }: { active: boolean; desc: boolean }) {
  if (!active) return <ArrowUpDown className="ml-1 h-3 w-3 opacity-50" />;
  return desc ? (
    <ArrowDown className="ml-1 h-3 w-3" />
  ) : (
    <ArrowUp className="ml-1 h-3 w-3" />
  );
}

function SortableTh({
  label,
  sortKey,
  activeSortKey,
  isDesc,
  onSort,
}: {
  label: string;
  sortKey: string;
  activeSortKey: string;
  isDesc: boolean;
  onSort: (key: string) => void;
}) {
  return (
    <th
      className="p-3 text-left font-medium cursor-pointer select-none hover:bg-muted/80 transition-colors"
      onClick={() => onSort(sortKey)}
    >
      <span className="inline-flex items-center">
        {label}
        <SortIcon active={activeSortKey === sortKey} desc={isDesc} />
      </span>
    </th>
  );
}

function Pagination({
  page,
  total,
  perPage,
  onPageChange,
  onPerPageChange,
  perPageLabel,
  pageOfLabel,
}: {
  page: number;
  total: number;
  perPage: number;
  onPageChange: (p: number) => void;
  onPerPageChange: (pp: number) => void;
  perPageLabel: string;
  pageOfLabel: (page: number, total: number) => string;
}) {
  const totalPages = Math.max(1, Math.ceil(total / perPage));
  return (
    <div className="flex items-center justify-between gap-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">{perPageLabel}</span>
        <Select
          value={String(perPage)}
          onValueChange={(v) => onPerPageChange(Number(v))}
        >
          <SelectTrigger className="w-[70px] h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PAGE_SIZES.map((s) => (
              <SelectItem key={s} value={String(s)}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">
          {pageOfLabel(page, totalPages)}
        </span>
        <Button
          variant="outline"
          size="icon"
          className="size-8"
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 1}
        >
          <ChevronLeft className="size-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="size-8"
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages}
        >
          <ChevronRight className="size-4" />
        </Button>
      </div>
    </div>
  );
}

export default function AdminDashboardPage() {
  const { t } = useUserSettings();
  const [generations, setGenerations] = useState<GenerationRow[]>([]);
  const [gTotal, setGTotal] = useState(0);
  const [purchases, setPurchases] = useState<PurchaseRow[]>([]);
  const [pTotal, setPTotal] = useState(0);
  const [signIns, setSignIns] = useState<SignInRow[]>([]);
  const [sTotal, setSTotal] = useState(0);
  const [assessmentRows, setAssessmentRows] = useState<AssessmentRow[]>([]);
  const [aTotal, setATotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [gSortBy, setGSortBy] = useState<GenerationSortKey>("createdAt");
  const [gSortDesc, setGSortDesc] = useState(true);
  const [gPage, setGPage] = useState(1);
  const [gPerPage, setGPerPage] = useState(DEFAULT_PER_PAGE);
  const [pSortBy, setPSortBy] = useState<PurchaseSortKey>("createdAt");
  const [pSortDesc, setPSortDesc] = useState(true);
  const [pPage, setPPage] = useState(1);
  const [pPerPage, setPPerPage] = useState(DEFAULT_PER_PAGE);
  const [sSortBy, setSSortBy] = useState<SignInSortKey>("createdAt");
  const [sSortDesc, setSSortDesc] = useState(true);
  const [sPage, setSPage] = useState(1);
  const [sPerPage, setSPerPage] = useState(DEFAULT_PER_PAGE);
  const [aSortBy, setASortBy] = useState<AssessmentSortKey>("createdAt");
  const [aSortDesc, setASortDesc] = useState(true);
  const [aPage, setAPage] = useState(1);
  const [aPerPage, setAPerPage] = useState(DEFAULT_PER_PAGE);
  const [aSearch, setASearch] = useState("");
  const initialLoad = useRef(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<{
    transcript: string;
    audioUrl: string;
    title: string;
    createdAt: string;
  } | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedAssessmentId, setSelectedAssessmentId] = useState<string | null>(null);
  const [assessmentDetail, setAssessmentDetail] = useState<SavedAssessment | null>(null);
  const [assessmentDetailLoading, setAssessmentDetailLoading] = useState(false);
  const [assessmentErrorFilter, setAssessmentErrorFilter] = useState<AssessmentFilter>("All");

  const handleAssessmentWordCost = useCallback((cost: number) => {
    setAssessmentDetail((prev) => {
      if (!prev) return prev;
      void fetch(`/api/assessment/${prev.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ additionalCost: cost }),
      });
      return { ...prev, cost: prev.cost + cost };
    });
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    fetch(`/api/generations/${selectedId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) {
          const d = data as { transcript: string; audioUrl: string; title: string; createdAt: string };
          setDetail({
            transcript: d.transcript,
            audioUrl: d.audioUrl,
            title: d.title,
            createdAt: d.createdAt,
          });
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => { cancelled = true; };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedAssessmentId) {
      setAssessmentDetail(null);
      return;
    }
    let cancelled = false;
    setAssessmentDetailLoading(true);
    fetch(`/api/assessment/${selectedAssessmentId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) {
          setAssessmentDetail(data as SavedAssessment);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setAssessmentDetailLoading(false);
      });
    return () => { cancelled = true; };
  }, [selectedAssessmentId]);

  const toggleGSort = useCallback(
    (key: string) => {
      const k = key as GenerationSortKey;
      if (gSortBy === k) {
        setGSortDesc((prev) => !prev);
      } else {
        setGSortBy(k);
        setGSortDesc(false);
      }
      setGPage(1);
    },
    [gSortBy]
  );

  const togglePSort = useCallback(
    (key: string) => {
      const k = key as PurchaseSortKey;
      if (pSortBy === k) {
        setPSortDesc((prev) => !prev);
      } else {
        setPSortBy(k);
        setPSortDesc(false);
      }
      setPPage(1);
    },
    [pSortBy]
  );

  const toggleSSort = useCallback(
    (key: string) => {
      const k = key as SignInSortKey;
      if (sSortBy === k) {
        setSSortDesc((prev) => !prev);
      } else {
        setSSortBy(k);
        setSSortDesc(false);
      }
      setSPage(1);
    },
    [sSortBy]
  );

  const toggleASort = useCallback(
    (key: string) => {
      const k = key as AssessmentSortKey;
      if (aSortBy === k) {
        setASortDesc((prev) => !prev);
      } else {
        setASortBy(k);
        setASortDesc(false);
      }
      setAPage(1);
    },
    [aSortBy]
  );

  useEffect(() => {
    let cancelled = false;
    async function loadGenerations() {
      try {
        const params = new URLSearchParams({
          sortBy: gSortBy,
          sortOrder: gSortDesc ? "desc" : "asc",
          q: search,
          page: String(gPage),
          perPage: String(gPerPage),
        });
        const res = await fetch(`/api/admin/generations?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { generations: GenerationRow[]; total: number };
          setGenerations(data.generations || []);
          setGTotal(data.total ?? 0);
        }
      } catch {}
    }

    async function loadPurchases() {
      try {
        const params = new URLSearchParams({
          sortBy: pSortBy,
          sortOrder: pSortDesc ? "desc" : "asc",
          page: String(pPage),
          perPage: String(pPerPage),
        });
        const res = await fetch(`/api/admin/purchases?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { purchases: PurchaseRow[]; total: number };
          setPurchases(data.purchases || []);
          setPTotal(data.total ?? 0);
        }
      } catch {}
    }

    async function loadSignIns() {
      try {
        const params = new URLSearchParams({
          sortBy: sSortBy,
          sortOrder: sSortDesc ? "desc" : "asc",
          page: String(sPage),
          perPage: String(sPerPage),
        });
        const res = await fetch(`/api/admin/sign-ins?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { signIns: SignInRow[]; total: number };
          setSignIns(data.signIns || []);
          setSTotal(data.total ?? 0);
        }
      } catch {}
    }

    async function loadAssessments() {
      try {
        const params = new URLSearchParams({
          sortBy: aSortBy,
          sortOrder: aSortDesc ? "desc" : "asc",
          q: aSearch,
          page: String(aPage),
          perPage: String(aPerPage),
        });
        const res = await fetch(`/api/admin/assessments?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { assessments: AssessmentRow[]; total: number };
          setAssessmentRows(data.assessments || []);
          setATotal(data.total ?? 0);
        }
      } catch {}
    }

    if (initialLoad.current) {
      void Promise.all([loadGenerations(), loadPurchases(), loadSignIns(), loadAssessments()]).finally(
        () => {
          if (!cancelled) {
            setLoading(false);
            initialLoad.current = false;
          }
        }
      );
    } else {
      void loadGenerations();
      void loadPurchases();
      void loadSignIns();
      void loadAssessments();
    }

    return () => {
      cancelled = true;
    };
  }, [gSortBy, gSortDesc, gPage, gPerPage, pSortBy, pSortDesc, pPage, pPerPage, sSortBy, sSortDesc, sPage, sPerPage, search, aSortBy, aSortDesc, aPage, aPerPage, aSearch]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">
            {t.admin.loading}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <ShieldCheck className="size-6" />
        <h1 className="text-2xl font-bold">{t.admin.title}</h1>
      </div>

      <Tabs defaultValue="usage">
        <TabsList className="flex w-full h-auto">
          <TabsTrigger
            value="usage"
            className="flex-1 gap-0.5 px-1 py-1.5 text-[10px] sm:gap-1 sm:px-3 sm:text-xs sm:py-2"
          >
            <Volume2 className="hidden sm:inline size-3" />
            <span className="truncate">{t.admin.tabUsage}</span>
            <Badge
              variant="secondary"
              className="ml-0.5 h-4 min-w-4 px-1 text-[10px]"
            >
              {gTotal}
            </Badge>
          </TabsTrigger>
          <TabsTrigger
            value="assessments"
            className="flex-1 gap-0.5 px-1 py-1.5 text-[10px] sm:gap-1 sm:px-3 sm:text-xs sm:py-2"
          >
            <Mic className="hidden sm:inline size-3" />
            <span className="truncate">{t.admin.tabAssessments}</span>
            <Badge
              variant="secondary"
              className="ml-0.5 h-4 min-w-4 px-1 text-[10px]"
            >
              {aTotal}
            </Badge>
          </TabsTrigger>
          <TabsTrigger
            value="purchases"
            className="flex-1 gap-0.5 px-1 py-1.5 text-[10px] sm:gap-1 sm:px-3 sm:text-xs sm:py-2"
          >
            <ShoppingCart className="hidden sm:inline size-3" />
            <span className="truncate">{t.admin.tabPurchases}</span>
            <Badge
              variant="secondary"
              className="ml-0.5 h-4 min-w-4 px-1 text-[10px]"
            >
              {pTotal}
            </Badge>
          </TabsTrigger>
          <TabsTrigger
            value="signins"
            className="flex-1 gap-0.5 px-1 py-1.5 text-[10px] sm:gap-1 sm:px-3 sm:text-xs sm:py-2"
          >
            <LogIn className="hidden sm:inline size-3" />
            <span className="truncate">{t.admin.tabSignIns}</span>
            <Badge
              variant="secondary"
              className="ml-0.5 h-4 min-w-4 px-1 text-[10px]"
            >
              {sTotal}
            </Badge>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="usage" className="mt-4 space-y-4">
          <div className="relative max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder={t.admin.searchPlaceholder}
              value={search}
              onChange={(e) => { setSearch(e.target.value); setGPage(1); }}
              className="pl-9"
            />
          </div>

          {generations.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">
                  {search
                    ? t.admin.noUsageMatch.replace("{search}", search)
                    : t.admin.noUsage}
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="rounded-md border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <SortableTh
                        label={t.admin.colUser}
                        sortKey="userName"
                        activeSortKey={gSortBy}
                        isDesc={gSortDesc}
                        onSort={toggleGSort}
                      />
                      <SortableTh
                        label={t.admin.colDate}
                        sortKey="createdAt"
                        activeSortKey={gSortBy}
                        isDesc={gSortDesc}
                        onSort={toggleGSort}
                      />
                      <SortableTh
                        label={t.admin.colTitle}
                        sortKey="title"
                        activeSortKey={gSortBy}
                        isDesc={gSortDesc}
                        onSort={toggleGSort}
                      />
                      <SortableTh
                        label={t.admin.colVoice}
                        sortKey="voice"
                        activeSortKey={gSortBy}
                        isDesc={gSortDesc}
                        onSort={toggleGSort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colCost}
                      </th>
                      <th className="p-3 text-left font-medium">
                        {t.admin.colCumulative}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {generations.map((g) => (
                      <tr
                        key={g.id}
                        className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                      >
                        <td className="p-3">
                          <div className="font-medium">
                            {g.userName || "Unknown"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {g.email}
                          </div>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          {formatDateHK(g.createdAt)}
                        </td>
                        <td className="p-3 max-w-[300px]">
                          <button
                            type="button"
                            className="text-left truncate hover:underline cursor-pointer w-full"
                            onClick={() => setSelectedId(g.id)}
                          >
                            {g.title}
                          </button>
                        </td>
                        <td className="p-3">
                          <Badge variant="secondary" className="capitalize">
                            {g.voice}
                          </Badge>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          {g.ttsCost ? (
                            <span className="text-sm">
                              HK${g.ttsCost}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                        <td className="p-3 whitespace-nowrap text-sm font-medium">
                          HK${parseFloat(g.cumulativeCost).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={gPage}
                total={gTotal}
                perPage={gPerPage}
                onPageChange={setGPage}
                onPerPageChange={(pp) => { setGPerPage(pp); setGPage(1); }}
                perPageLabel={t.admin.perPage}
                pageOfLabel={(p, tp) => t.admin.pageOf.replace("{page}", String(p)).replace("{total}", String(tp))}
              />
            </>
          )}
        </TabsContent>

        <TabsContent value="assessments" className="mt-4 space-y-4">
          <div className="relative max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder={t.admin.searchAssessments}
              value={aSearch}
              onChange={(e) => { setASearch(e.target.value); setAPage(1); }}
              className="pl-9"
            />
          </div>

          {assessmentRows.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">
                  {aSearch
                    ? t.admin.noAssessmentsMatch.replace("{search}", aSearch)
                    : t.admin.noAssessments}
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="rounded-md border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <SortableTh
                        label={t.admin.colUser}
                        sortKey="userName"
                        activeSortKey={aSortBy}
                        isDesc={aSortDesc}
                        onSort={toggleASort}
                      />
                      <SortableTh
                        label={t.admin.colDate}
                        sortKey="createdAt"
                        activeSortKey={aSortBy}
                        isDesc={aSortDesc}
                        onSort={toggleASort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colReferenceText}
                      </th>
                      <SortableTh
                        label={t.admin.colPronScore}
                        sortKey="pronScore"
                        activeSortKey={aSortBy}
                        isDesc={aSortDesc}
                        onSort={toggleASort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colDuration}
                      </th>
                      <SortableTh
                        label={t.admin.colSttCost}
                        sortKey="cost"
                        activeSortKey={aSortBy}
                        isDesc={aSortDesc}
                        onSort={toggleASort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colCumulative}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {assessmentRows.map((a) => (
                      <tr
                        key={a.id}
                        className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                      >
                        <td className="p-3">
                          <div className="font-medium">
                            {a.userName || "Unknown"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {a.email}
                          </div>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          {formatDateHK(a.createdAt)}
                        </td>
                        <td className="p-3 max-w-[250px]">
                          <button
                            type="button"
                            className="text-left truncate hover:underline cursor-pointer w-full"
                            onClick={() => setSelectedAssessmentId(a.id)}
                          >
                            {a.referenceText}
                          </button>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          <Badge
                            className={cn(
                              "font-semibold tabular-nums",
                              a.pronScore >= 80
                                ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-500/25"
                                : a.pronScore >= 60
                                  ? "bg-amber-500/15 text-amber-700 dark:text-amber-400 hover:bg-amber-500/25"
                                  : "bg-red-500/15 text-red-700 dark:text-red-400 hover:bg-red-500/25",
                            )}
                          >
                            {Math.round(a.pronScore)}
                          </Badge>
                        </td>
                        <td className="p-3 whitespace-nowrap text-xs">
                          {(a.durationMs / 1000).toFixed(1)}s
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          HK${a.cost.toFixed(2)}
                        </td>
                        <td className="p-3 whitespace-nowrap text-sm font-medium">
                          HK${parseFloat(a.cumulativeCost).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={aPage}
                total={aTotal}
                perPage={aPerPage}
                onPageChange={setAPage}
                onPerPageChange={(pp) => { setAPerPage(pp); setAPage(1); }}
                perPageLabel={t.admin.perPage}
                pageOfLabel={(p, tp) => t.admin.pageOf.replace("{page}", String(p)).replace("{total}", String(tp))}
              />
            </>
          )}
        </TabsContent>

        <TabsContent value="purchases" className="mt-4 space-y-4">
          {purchases.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">{t.admin.noPurchases}</p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="rounded-md border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <SortableTh
                        label={t.admin.colUser}
                        sortKey="userName"
                        activeSortKey={pSortBy}
                        isDesc={pSortDesc}
                        onSort={togglePSort}
                      />
                      <SortableTh
                        label={t.admin.colDate}
                        sortKey="createdAt"
                        activeSortKey={pSortBy}
                        isDesc={pSortDesc}
                        onSort={togglePSort}
                      />
                      <SortableTh
                        label={t.admin.colPackage}
                        sortKey="planName"
                        activeSortKey={pSortBy}
                        isDesc={pSortDesc}
                        onSort={togglePSort}
                      />
                      <SortableTh
                        label={t.admin.colAmount}
                        sortKey="amountHKD"
                        activeSortKey={pSortBy}
                        isDesc={pSortDesc}
                        onSort={togglePSort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colCredits}
                      </th>
                      <th className="p-3 text-left font-medium">
                        {t.admin.colStatus}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {purchases.map((p) => (
                      <tr
                        key={p.id}
                        className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                      >
                        <td className="p-3">
                          <div className="font-medium">
                            {p.userName || "Unknown"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {p.email}
                          </div>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          {formatDateHK(p.createdAt)}
                        </td>
                        <td className="p-3">{p.planName}</td>
                        <td className="p-3 whitespace-nowrap font-medium">
                          HK${p.amountHKD.toFixed(2)}
                        </td>
                        <td className="p-3">{p.creditsAmount}</td>
                        <td className="p-3">
                          <Badge
                            variant={
                              p.status === "completed"
                                ? "default"
                                : p.status === "pending"
                                  ? "secondary"
                                  : "destructive"
                            }
                          >
                            {p.status}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={pPage}
                total={pTotal}
                perPage={pPerPage}
                onPageChange={setPPage}
                onPerPageChange={(pp) => { setPPerPage(pp); setPPage(1); }}
                perPageLabel={t.admin.perPage}
                pageOfLabel={(p, tp) => t.admin.pageOf.replace("{page}", String(p)).replace("{total}", String(tp))}
              />
            </>
          )}
        </TabsContent>

        <TabsContent value="signins" className="mt-4 space-y-4">
          {signIns.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">{t.admin.noSignIns}</p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="rounded-md border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <SortableTh
                        label={t.admin.colUser}
                        sortKey="userName"
                        activeSortKey={sSortBy}
                        isDesc={sSortDesc}
                        onSort={toggleSSort}
                      />
                      <SortableTh
                        label={t.admin.colDateTime}
                        sortKey="createdAt"
                        activeSortKey={sSortBy}
                        isDesc={sSortDesc}
                        onSort={toggleSSort}
                      />
                      <th className="p-3 text-left font-medium">
                        {t.admin.colProvider}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {signIns.map((s) => (
                      <tr
                        key={s.id}
                        className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                      >
                        <td className="p-3">
                          <div className="font-medium">
                            {s.userName || "Unknown"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {s.email}
                          </div>
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          {formatDateHK(s.createdAt)}
                        </td>
                        <td className="p-3">
                          <Badge variant="secondary">{s.provider}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={sPage}
                total={sTotal}
                perPage={sPerPage}
                onPageChange={setSPage}
                onPerPageChange={(pp) => { setSPerPage(pp); setSPage(1); }}
                perPageLabel={t.admin.perPage}
                pageOfLabel={(p, tp) => t.admin.pageOf.replace("{page}", String(p)).replace("{total}", String(tp))}
              />
            </>
          )}
        </TabsContent>
      </Tabs>

      <Dialog open={!!selectedId} onOpenChange={(open) => { if (!open) setSelectedId(null); }}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{detail?.title}</DialogTitle>
          </DialogHeader>
          {detailLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : !detail ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              {t.admin.notFound}
            </div>
          ) : (
            <div className="space-y-4">
              <AudioPlayer src={detail.audioUrl} createdAt={detail.createdAt} />
              <div>
                <h3 className="text-sm font-medium mb-2">{t.admin.sourceText}</h3>
                <pre className="text-sm whitespace-pre-wrap bg-muted/50 rounded-md p-4 max-h-[40vh] overflow-y-auto">
                  {detail.transcript}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={!!selectedAssessmentId} onOpenChange={(open) => { if (!open) setSelectedAssessmentId(null); }}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t.admin.assessmentDetail}</DialogTitle>
          </DialogHeader>
          {assessmentDetailLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : !assessmentDetail ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              {t.admin.notFound}
            </div>
          ) : (
            (() => {
              const scores = {
                AccuracyScore: assessmentDetail.accuracyScore,
                FluencyScore: assessmentDetail.fluencyScore,
                CompletenessScore: assessmentDetail.completenessScore,
                ProsodyScore: assessmentDetail.prosodyScore ?? 0,
                PronScore: assessmentDetail.pronScore,
              };
              const words = assessmentDetail.words;
              const filteredWords = filterWords(words, assessmentErrorFilter);
              return (
                <div className="space-y-4">
                  <ScoreOverview scores={scores} t={t.assessment} />

                  <Separator />

                  {assessmentDetail.audioPath && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold">{t.assessment.yourRecording}</h3>
                      <Card>
                        <CardContent className="flex items-center gap-3 py-3">
                          <audio controls className="w-full" preload="metadata">
                            <source
                              src={`/api/assessment/${selectedAssessmentId}/audio`}
                            />
                          </audio>
                        </CardContent>
                      </Card>
                    </div>
                  )}

                  <Separator />

                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold">{t.assessment.recognizedText}</h3>
                    <TranscriptView words={words} t={t.assessment} onCostUpdate={handleAssessmentWordCost} />
                  </div>

                  <Separator />

                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold">{t.assessment.errorSummary}</h3>
                    <ErrorSummary
                      words={words}
                      t={t.assessment}
                      filter={assessmentErrorFilter}
                      onFilterChange={setAssessmentErrorFilter}
                    />
                  </div>

                  <Separator />

                  <Tabs defaultValue="word">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold">{t.assessment.granularity}</h3>
                      <TabsList>
                        <TabsTrigger value="fulltext">{t.assessment.granCoach}</TabsTrigger>
                        <TabsTrigger value="word">{t.assessment.granWord}</TabsTrigger>
                        <TabsTrigger value="syllable">{t.assessment.granSyllable}</TabsTrigger>
                        <TabsTrigger value="phoneme">{t.assessment.granPhoneme}</TabsTrigger>
                      </TabsList>
                    </div>

                    <TabsContent value="fulltext">
                      <div className="rounded-lg border p-4">
                        <FeedbackCard
                          assessmentId={assessmentDetail.id}
                          initialFeedback={assessmentDetail.feedback ?? undefined}
                          t={t.assessment}
                          onCostUpdate={() => { void fetch(`/api/assessment/${assessmentDetail.id}`).then((res) => (res.ok ? res.json() : null)).then((data) => { if (data) setAssessmentDetail(data as SavedAssessment); }); }}
                        />
                      </div>
                    </TabsContent>

                    <TabsContent value="word">
                      <div className="max-h-96 overflow-y-auto">
                        <WordDetail words={filteredWords} t={t.assessment} onCostUpdate={handleAssessmentWordCost} />
                      </div>
                    </TabsContent>

                    <TabsContent value="syllable">
                      <div className="max-h-96 overflow-y-auto">
                        <SyllableView words={filteredWords} t={t.assessment} onCostUpdate={handleAssessmentWordCost} />
                      </div>
                    </TabsContent>

                    <TabsContent value="phoneme">
                      <div className="max-h-96 overflow-y-auto">
                        <PhonemeAnalysis words={filteredWords} t={t.assessment} onCostUpdate={handleAssessmentWordCost} />
                      </div>
                    </TabsContent>
                  </Tabs>

                  <Separator />

                  <ReferenceAudioSection referenceText={assessmentDetail.referenceText} t={t.assessment} assessmentId={assessmentDetail.id} hasReferenceAudio={!!assessmentDetail.referenceAudioPath} onCostUpdate={() => { void fetch(`/api/assessment/${assessmentDetail.id}`).then((res) => (res.ok ? res.json() : null)).then((data) => { if (data) setAssessmentDetail(data as SavedAssessment); }); }} />

                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Clock className="size-3" />
                      {formatDateHK(assessmentDetail.createdAt)}
                    </span>
                    <Badge variant="outline">{t.assessment.score.replace("{score}", String(Math.round(assessmentDetail.pronScore)))}</Badge>
                    <Badge variant="outline">
                      {t.assessment.duration.replace("{seconds}", String(Math.round(assessmentDetail.durationMs / 1000)))}
                    </Badge>
                    <span className="flex items-center gap-1">
                      <Coins className="size-3" />
                      HK${assessmentDetail.cost.toFixed(2)}
                    </span>
                  </div>
                </div>
              );
            })()
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
