import { useSquad, useSquadIssues, useManagerOverview } from "@/hooks/use-api"
import { StatCard } from "@/components/shared/stat-card"
import { PlayerCard } from "@/components/shared/player-card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { formatNumber, formatPrice } from "@/lib/fpl"
import { POSITION_ORDER } from "@/lib/fpl"
import type { Player } from "@/lib/api"
import { AlertTriangle } from "lucide-react"

function FormationGrid({ squad }: { squad: Player[] }) {
  const starters = squad.filter((p) => p.multiplier > 0).sort(
    (a, b) => POSITION_ORDER[a.position] - POSITION_ORDER[b.position]
  )
  const bench = squad.filter((p) => p.multiplier === 0)

  // Group starters by position for formation rows
  const rows: Record<string, Player[]> = {}
  for (const p of starters) {
    ;(rows[p.position] ??= []).push(p)
  }

  return (
    <div className="space-y-4">
      {/* Formation rows */}
      {["GKP", "DEF", "MID", "FWD"].map((pos) => {
        const players = rows[pos] ?? []
        if (players.length === 0) return null
        return (
          <div key={pos} className="flex items-center justify-center gap-2 flex-wrap">
            {players.map((p) => (
              <PlayerCard key={p.id} player={p} />
            ))}
          </div>
        )
      })}

      {/* Bench */}
      {bench.length > 0 && (
        <div className="border-t border-border pt-3">
          <p className="mb-2 text-center text-xs font-medium text-muted-foreground">Bench</p>
          <div className="flex items-center justify-center gap-2 flex-wrap">
            {bench.map((p) => (
              <PlayerCard key={p.id} player={p} compact />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function IssuesPanel() {
  const { data: issues, isLoading } = useSquadIssues()
  if (isLoading) return <Skeleton className="h-24" />
  if (!issues || issues.total_issues === 0) return null

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="size-4 text-destructive" />
        <h3 className="text-sm font-semibold">Squad Issues ({issues.total_issues})</h3>
      </div>
      <div className="space-y-2 text-xs">
        {issues.injuries?.map((inj) => (
          <div key={inj.name} className="flex items-center gap-2">
            <Badge variant="destructive" className="text-[10px]">INJ</Badge>
            <span>{inj.name}</span>
            <span className="text-muted-foreground">{inj.news}</span>
          </div>
        ))}
        {issues.suspension_risk?.map((sr) => (
          <div key={sr.name} className="flex items-center gap-2">
            <Badge className="bg-amber-600 text-[10px]">YC</Badge>
            <span>{sr.name}</span>
            <span className="text-muted-foreground">{sr.yellows}/{sr.threshold} yellows</span>
          </div>
        ))}
        {issues.price_drops?.map((pd) => (
          <div key={pd.name} className="flex items-center gap-2">
            <Badge variant="secondary" className="text-[10px]">PRICE</Badge>
            <span>{pd.name}</span>
            <span className="text-muted-foreground">trend: {pd.trend}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function DashboardPage() {
  const { data: squadData, isLoading: squadLoading, error: squadError } = useSquad()
  const { data: overview } = useManagerOverview()

  if (squadError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-2">
          <p className="text-muted-foreground">Loading squad data...</p>
          <p className="text-xs text-muted-foreground">The server is warming up. This may take a minute.</p>
        </div>
      </div>
    )
  }

  const info = squadData?.team_info
  const lastGW = overview?.gw_history?.[overview.gw_history.length - 1]

  return (
    <div className="space-y-6">
      {/* GW Stats Bar */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <StatCard
          label="GW Points"
          value={lastGW?.points ?? "-"}
          sub={`GW${lastGW?.event ?? "?"}`}
        />
        <StatCard
          label="Total Points"
          value={formatNumber(info?.overall_points)}
        />
        <StatCard
          label="Overall Rank"
          value={formatNumber(info?.overall_rank)}
        />
        <StatCard
          label="Team Value"
          value={formatPrice(overview?.team_value)}
          sub={`Bank: ${formatPrice(overview?.bank)}`}
        />
        <StatCard
          label="Free Transfers"
          value={overview?.free_transfers ?? "-"}
        />
      </div>

      {/* Formation Grid + Issues */}
      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-4 text-sm font-semibold">
            Squad {squadData?.gameweek ? `(GW${squadData.gameweek})` : ""}
          </h2>
          {squadLoading ? (
            <div className="space-y-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex justify-center gap-2">
                  <Skeleton className="h-24 w-20" />
                  <Skeleton className="h-24 w-20" />
                  <Skeleton className="h-24 w-20" />
                </div>
              ))}
            </div>
          ) : squadData?.squad ? (
            <FormationGrid squad={squadData.squad} />
          ) : null}
        </div>

        <div className="space-y-4">
          <IssuesPanel />
          {/* Chips used */}
          {overview?.chips_used && overview.chips_used.length > 0 && (
            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="mb-2 text-sm font-semibold">Chips Used</h3>
              <div className="space-y-1">
                {overview.chips_used.map((chip) => (
                  <div key={chip.name} className="flex items-center justify-between text-xs">
                    <Badge variant="outline" className="text-[10px] capitalize">{chip.name}</Badge>
                    <span className="text-muted-foreground">GW{chip.event}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
