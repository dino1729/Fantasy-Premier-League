import { useMeta } from "@/hooks/use-api"
import { timeAgo } from "@/lib/fpl"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { PanelLeft, RefreshCw } from "lucide-react"

const TRACKED_JOBS = ["bootstrap", "analysis", "solver", "league"] as const

function freshnessTone(state: "fresh" | "stale" | "degraded" | "warming"): string {
  if (state === "fresh") return "border-emerald-500/50 text-emerald-400"
  if (state === "stale") return "border-amber-500/50 text-amber-400"
  if (state === "degraded") return "border-red-500/50 text-red-400"
  return "border-zinc-500/50 text-zinc-300"
}

interface AppHeaderProps {
  onOpenSidebar: () => void
}

export function AppHeader({ onOpenSidebar }: AppHeaderProps) {
  const { data: meta } = useMeta()

  const refreshMap = meta?.refresh_status ?? {}
  const statusRows = Object.values(refreshMap)
  const lastRefreshMs = statusRows
    .map((row) => Date.parse(row.last_run_at))
    .filter((ms) => Number.isFinite(ms))
  const lastRefresh =
    lastRefreshMs.length > 0 ? new Date(Math.max(...lastRefreshMs)).toISOString() : undefined

  const hasError = statusRows.some((row) => row.status !== "ok")
  const isStale =
    lastRefresh != null ? Date.now() - Date.parse(lastRefresh) > 2 * 60 * 60 * 1000 : false
  const freshnessState: "fresh" | "stale" | "degraded" | "warming" =
    !meta?.ready ? "warming" : hasError ? "degraded" : isStale ? "stale" : "fresh"
  const freshnessText =
    freshnessState === "fresh"
      ? "Fresh"
      : freshnessState === "stale"
        ? "Stale"
        : freshnessState === "degraded"
          ? "Degraded"
          : "Warming"

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-3 md:px-6">
      <div className="flex items-center gap-3">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="size-8 md:hidden"
          onClick={onOpenSidebar}
        >
          <PanelLeft className="size-4" />
          <span className="sr-only">Open navigation</span>
        </Button>
        <h1 className="text-sm font-medium text-foreground">
          {meta?.team_info?.team_name ?? "Loading..."}
        </h1>
        {meta?.current_gameweek && (
          <Badge variant="secondary" className="text-xs">
            GW{meta.current_gameweek}
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className={freshnessTone(freshnessState)}>
          {freshnessText}
        </Badge>
        <div className="hidden items-center gap-1 md:flex">
          {TRACKED_JOBS.map((job) => {
            const status = refreshMap[job]?.status
            const tone =
              status === "ok"
                ? "bg-emerald-500/70"
                : status
                  ? "bg-red-500/70"
                  : "bg-zinc-500/50"
            return <span key={job} title={`${job}: ${status ?? "pending"}`} className={`h-1.5 w-1.5 rounded-full ${tone}`} />
          })}
        </div>
        <RefreshCw className={`size-3 ${freshnessState === "warming" ? "animate-spin" : ""}`} />
        <span>Updated {timeAgo(lastRefresh)}</span>
      </div>
    </header>
  )
}
