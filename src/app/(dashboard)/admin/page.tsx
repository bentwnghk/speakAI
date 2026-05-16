"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
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
  Coins,
  ShieldCheck,
  ShoppingCart,
  LogIn,
} from "lucide-react";
import { useUserSettings } from "@/hooks/use-settings";
import { AudioPlayer } from "@/components/audio-player";

interface GenerationRow {
  id: string;
  userName: string | null;
  email: string | null;
  title: string;
  voice: string;
  createdAt: string;
  ttsCost: string | null;
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

type GenerationSortKey = "userName" | "createdAt" | "title" | "voice";
type PurchaseSortKey = "userName" | "createdAt" | "planName" | "amountHKD";
type SignInSortKey = "userName" | "createdAt";

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

export default function AdminDashboardPage() {
  const { t } = useUserSettings();
  const [generations, setGenerations] = useState<GenerationRow[]>([]);
  const [purchases, setPurchases] = useState<PurchaseRow[]>([]);
  const [signIns, setSignIns] = useState<SignInRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [gSortBy, setGSortBy] = useState<GenerationSortKey>("createdAt");
  const [gSortDesc, setGSortDesc] = useState(true);
  const [pSortBy, setPSortBy] = useState<PurchaseSortKey>("createdAt");
  const [pSortDesc, setPSortDesc] = useState(true);
  const [sSortBy, setSSortBy] = useState<SignInSortKey>("createdAt");
  const [sSortDesc, setSSortDesc] = useState(true);
  const initialLoad = useRef(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<{
    transcript: string;
    audioUrl: string;
    title: string;
    createdAt: string;
  } | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

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

  const toggleGSort = useCallback(
    (key: string) => {
      const k = key as GenerationSortKey;
      if (gSortBy === k) {
        setGSortDesc((prev) => !prev);
      } else {
        setGSortBy(k);
        setGSortDesc(false);
      }
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
    },
    [sSortBy]
  );

  useEffect(() => {
    let cancelled = false;
    async function loadGenerations() {
      try {
        const params = new URLSearchParams({
          sortBy: gSortBy,
          sortOrder: gSortDesc ? "desc" : "asc",
          q: search,
        });
        const res = await fetch(`/api/admin/generations?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { generations: GenerationRow[] };
          setGenerations(data.generations || []);
        }
      } catch {}
    }

    async function loadPurchases() {
      try {
        const params = new URLSearchParams({
          sortBy: pSortBy,
          sortOrder: pSortDesc ? "desc" : "asc",
        });
        const res = await fetch(`/api/admin/purchases?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { purchases: PurchaseRow[] };
          setPurchases(data.purchases || []);
        }
      } catch {}
    }

    async function loadSignIns() {
      try {
        const params = new URLSearchParams({
          sortBy: sSortBy,
          sortOrder: sSortDesc ? "desc" : "asc",
        });
        const res = await fetch(`/api/admin/sign-ins?${params}`);
        if (res.ok && !cancelled) {
          const data = (await res.json()) as { signIns: SignInRow[] };
          setSignIns(data.signIns || []);
        }
      } catch {}
    }

    if (initialLoad.current) {
      void Promise.all([loadGenerations(), loadPurchases(), loadSignIns()]).finally(
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
    }

    return () => {
      cancelled = true;
    };
  }, [gSortBy, gSortDesc, pSortBy, pSortDesc, sSortBy, sSortDesc, search]);

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
        <TabsList className="flex w-full">
          <TabsTrigger value="usage" className="flex-1 gap-1 text-xs sm:text-sm">
            <Coins className="h-4 w-4" />
            {t.admin.tabUsage}
            <Badge variant="secondary" className="ml-1 text-xs">
              {generations.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="purchases" className="flex-1 gap-1 text-xs sm:text-sm">
            <ShoppingCart className="h-4 w-4" />
            {t.admin.tabPurchases}
            <Badge variant="secondary" className="ml-1 text-xs">
              {purchases.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="signins" className="flex-1 gap-1 text-xs sm:text-sm">
            <LogIn className="h-4 w-4" />
            {t.admin.tabSignIns}
            <Badge variant="secondary" className="ml-1 text-xs">
              {signIns.length}
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
              onChange={(e) => setSearch(e.target.value)}
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
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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
    </div>
  );
}
