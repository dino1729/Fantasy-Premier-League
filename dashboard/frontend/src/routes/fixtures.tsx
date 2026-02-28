import { useMemo, useState } from "react"
import { useFDRGrid } from "@/hooks/use-api"
import type { FDRGridResponse, FixtureEntry } from "@/lib/api"
import { FDR_BG } from "@/lib/fpl"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { ArrowDownNarrowWide, ArrowUpNarrowWide, Loader2 } from "lucide-react"

type DifficultySort = "asc" | "desc"

interface FixtureRow {
  teamId: number
  name: string
  shortName: string
  fixturesByGw: Map<number, FixtureEntry[]>
  avgDifficulty: number
}

function buildGwRange(currentGameweek: number | undefined, width: number): number[] {
  const current = currentGameweek ?? 1
  return Array.from({ length: width }, (_, index) => current + index + 1)
}

function buildTeamRows(data: FDRGridResponse | undefined, gws: number[]): FixtureRow[] {
  if (!data?.fdr_grid) return []

  const rows: FixtureRow[] = []
  for (const [teamIdRaw, team] of Object.entries(data.fdr_grid)) {
    const teamId = Number(teamIdRaw)
    const fixturesByGw = new Map<number, FixtureEntry[]>()
    for (const fixture of team.fixtures) {
      if (!gws.includes(fixture.gameweek)) continue
      const bucket = fixturesByGw.get(fixture.gameweek) ?? []
      bucket.push(fixture)
      fixturesByGw.set(fixture.gameweek, bucket)
    }

    const cellDifficulties = gws.map((gw) => {
      const cellFixtures = fixturesByGw.get(gw)
      if (!cellFixtures || cellFixtures.length === 0) return 5
      const sum = cellFixtures.reduce((acc, f) => acc + f.difficulty, 0)
      return sum / cellFixtures.length
    })
    const avgDifficulty = cellDifficulties.reduce((acc, d) => acc + d, 0) / gws.length

    rows.push({
      teamId,
      name: team.name,
      shortName: team.short_name,
      fixturesByGw,
      avgDifficulty,
    })
  }

  return rows
}

function mapGwTeams(
  entries: Array<{ gw: number; team_ids: number[] }> | undefined
): Map<number, Set<number>> {
  const result = new Map<number, Set<number>>()
  if (!entries) return result

  for (const row of entries) {
    result.set(row.gw, new Set(row.team_ids))
  }
  return result
}

function difficultyCellTheme(difficulty: number): { bg: string; fg: string } {
  const rounded = Math.max(1, Math.min(5, Math.round(difficulty)))
  const bg = FDR_BG[rounded] ?? "#71717a"
  const fg = rounded <= 2 ? "#111827" : "#ffffff"
  return { bg, fg }
}

function formatProb(value: number): string {
  return `${Math.max(0, Math.min(100, Math.round(value * 100)))}%`
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  const width = Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{label}</span>
        <span>{width}%</span>
      </div>
      <div className="h-1.5 w-full rounded bg-secondary">
        <div className="h-full rounded" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

function FixtureTooltip({ fixtures }: { fixtures: FixtureEntry[] }) {
  return (
    <div className="min-w-56 space-y-3">
      {fixtures.map((fixture, index) => (
        <div key={`${fixture.gameweek}-${fixture.opponent}-${index}`} className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold">
              {fixture.opponent} ({fixture.is_home ? "H" : "A"})
            </p>
            <Badge variant="outline" className="text-[10px]">
              FDR {fixture.difficulty}
            </Badge>
          </div>
          <div className="space-y-1.5">
            <ProbBar label="Win" value={fixture.win_prob} color="#10b981" />
            <ProbBar label="Draw" value={fixture.draw_prob} color="#f59e0b" />
            <ProbBar label="Loss" value={fixture.loss_prob} color="#ef4444" />
          </div>
        </div>
      ))}
    </div>
  )
}

function FixtureCell({
  fixtures,
  isBlank,
  isDouble,
}: {
  fixtures: FixtureEntry[]
  isBlank: boolean
  isDouble: boolean
}) {
  if (fixtures.length === 0) {
    return (
      <div
        className={cn(
          "flex min-h-20 items-center justify-center rounded-md border border-border px-1 py-2 text-[10px] font-semibold uppercase tracking-wide",
          isBlank
            ? "bg-zinc-700/40 text-zinc-300"
            : "bg-muted/30 text-muted-foreground",
          isDouble && "border-2 border-amber-400"
        )}
      >
        {isBlank ? "BGW" : "-"}
      </div>
    )
  }

  const avgDifficulty =
    fixtures.reduce((acc, fixture) => acc + fixture.difficulty, 0) / fixtures.length
  const theme = difficultyCellTheme(avgDifficulty)
  const topFixture = fixtures[0]

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className={cn(
            "min-h-20 w-full rounded-md border px-1 py-1.5 text-left transition-opacity hover:opacity-90",
            isDouble ? "border-2 border-amber-400" : "border-zinc-700/50"
          )}
          style={{ backgroundColor: theme.bg, color: theme.fg }}
        >
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold">{topFixture.opponent}</span>
            <span className="text-[10px] font-bold">{topFixture.is_home ? "H" : "A"}</span>
          </div>
          {fixtures.length > 1 && (
            <div className="mt-1 space-y-0.5">
              {fixtures.slice(1).map((fixture, index) => (
                <div key={`${fixture.opponent}-${index}`} className="flex items-center justify-between text-[10px]">
                  <span>{fixture.opponent}</span>
                  <span>{fixture.is_home ? "H" : "A"}</span>
                </div>
              ))}
            </div>
          )}
          <div className="mt-1.5 text-[10px] font-medium">
            {formatProb(topFixture.win_prob)} W
          </div>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-72 p-3">
        <FixtureTooltip fixtures={fixtures} />
      </TooltipContent>
    </Tooltip>
  )
}

export function FixturesPage() {
  const { data, isLoading, error } = useFDRGrid()
  const [sortDirection, setSortDirection] = useState<DifficultySort>("asc")

  const gwColumns = useMemo(() => buildGwRange(data?.current_gameweek, 8), [data?.current_gameweek])
  const bgwByGw = useMemo(() => mapGwTeams(data?.bgw_dgw?.bgw), [data?.bgw_dgw?.bgw])
  const dgwByGw = useMemo(() => mapGwTeams(data?.bgw_dgw?.dgw), [data?.bgw_dgw?.dgw])

  const rows = useMemo(() => {
    const baseRows = buildTeamRows(data, gwColumns)
    return baseRows.sort((a, b) => {
      if (a.avgDifficulty === b.avgDifficulty) {
        return a.shortName.localeCompare(b.shortName)
      }
      return sortDirection === "asc"
        ? a.avgDifficulty - b.avgDifficulty
        : b.avgDifficulty - a.avgDifficulty
    })
  }, [data, gwColumns, sortDirection])

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-4 w-80" />
        </div>
        <Skeleton className="h-[640px] w-full" />
      </div>
    )
  }

  if (error) {
    const isWarmingUp = error.message === "warming_up"
    if (isWarmingUp) {
      return (
        <div className="flex flex-col items-center justify-center gap-3 py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Fixture planner is warming up...</p>
        </div>
      )
    }
    return <div className="py-10 text-center text-red-400 text-sm">Failed to load fixtures: {error.message}</div>
  }

  return (
    <TooltipProvider>
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">Fixture Planner</h2>
            <p className="text-sm text-muted-foreground">
              Elo-based FDR grid for the next 8 gameweeks with BGW/DGW markers.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"))}
            className="gap-1.5"
          >
            {sortDirection === "asc" ? (
              <ArrowUpNarrowWide className="h-3.5 w-3.5" />
            ) : (
              <ArrowDownNarrowWide className="h-3.5 w-3.5" />
            )}
            Avg FDR {sortDirection === "asc" ? "Easiest" : "Hardest"}
          </Button>
        </div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="border-zinc-500 text-zinc-300">BGW</Badge>
          <Badge variant="outline" className="border-amber-400 text-amber-300">DGW</Badge>
          <span>Hover any fixture for win/draw/loss probability breakdown.</span>
        </div>

        <div className="rounded-lg border border-border bg-card">
          <ScrollArea className="w-full">
            <div className="min-w-[980px]">
              <div
                className="grid gap-1 border-b border-border bg-muted/40 p-2"
                style={{ gridTemplateColumns: `130px 68px repeat(${gwColumns.length}, minmax(96px, 1fr))` }}
              >
                <div className="text-xs font-semibold text-muted-foreground">Team</div>
                <div className="text-center text-xs font-semibold text-muted-foreground">Avg</div>
                {gwColumns.map((gw) => (
                  <div key={gw} className="text-center text-xs font-semibold text-muted-foreground">
                    GW{gw}
                  </div>
                ))}
              </div>

              <div className="space-y-1 p-2">
                {rows.map((row) => (
                  <div
                    key={row.teamId}
                    className="grid gap-1"
                    style={{ gridTemplateColumns: `130px 68px repeat(${gwColumns.length}, minmax(96px, 1fr))` }}
                  >
                    <div className="flex items-center rounded-md border border-border bg-muted/20 px-2 text-sm font-semibold">
                      {row.shortName}
                    </div>
                    <div className="flex items-center justify-center rounded-md border border-border bg-muted/20 text-sm font-medium">
                      {row.avgDifficulty.toFixed(2)}
                    </div>
                    {gwColumns.map((gw) => {
                      const fixtures = row.fixturesByGw.get(gw) ?? []
                      const bgwTeams = bgwByGw.get(gw)
                      const dgwTeams = dgwByGw.get(gw)
                      const isBlank = Boolean(bgwTeams?.has(row.teamId))
                      const isDouble = Boolean(dgwTeams?.has(row.teamId))

                      return (
                        <FixtureCell
                          key={`${row.teamId}-${gw}`}
                          fixtures={fixtures}
                          isBlank={isBlank}
                          isDouble={isDouble}
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </div>
      </div>
    </TooltipProvider>
  )
}
