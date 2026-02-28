import { useMemo, useState } from "react"
import { AxisBottom, AxisLeft } from "@visx/axis"
import { Group } from "@visx/group"
import { ParentSize } from "@visx/responsive"
import { scaleLinear } from "@visx/scale"
import { Bar, LinePath } from "@visx/shape"
import { useCompetitors, useGlobalManagers, useMeta } from "@/hooks/use-api"
import type { LeagueEntry, LeagueTransferTimelineEntry } from "@/lib/api"
import { cn, } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Loader2 } from "lucide-react"

type LeagueMode = "competitors" | "global"
type ChartMode = "rank" | "points"

const LINE_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f59e0b",
  "#a78bfa",
  "#f87171",
  "#14b8a6",
  "#facc15",
  "#60a5fa",
]

function formatNumber(value: number | undefined): string {
  if (value == null) return "-"
  return value.toLocaleString()
}

function getManagerLabel(entry: LeagueEntry): string {
  return entry.team_info?.manager_name || entry.team_info?.team_name || `Entry ${entry.entry_id}`
}

function getTeamLabel(entry: LeagueEntry): string {
  return entry.team_info?.team_name || `Entry ${entry.entry_id}`
}

function getRankPoint(row: Record<string, unknown> | undefined): number | null {
  if (!row) return null
  const raw = row.overall_rank ?? row.rank_sort ?? row.rank
  if (typeof raw === "number" && Number.isFinite(raw)) return raw
  return null
}

function getPointsPoint(row: Record<string, unknown> | undefined): number | null {
  if (!row) return null
  const raw = row.total_points
  if (typeof raw === "number" && Number.isFinite(raw)) return raw
  return null
}

function ProgressionChart({
  entries,
  mode,
  highlightEntryId,
}: {
  entries: LeagueEntry[]
  mode: ChartMode
  highlightEntryId?: number
}) {
  const series = useMemo(() => {
    return entries
      .map((entry) => {
        const points = (entry.gw_history ?? [])
          .map((row) => {
            const gw = Number(row.event)
            const value =
              mode === "rank"
                ? getRankPoint(row as unknown as Record<string, unknown>)
                : getPointsPoint(row as unknown as Record<string, unknown>)
            if (!Number.isFinite(gw) || value == null) return null
            return { gw, value }
          })
          .filter((point): point is { gw: number; value: number } => point != null)
          .sort((a, b) => a.gw - b.gw)

        return {
          entry,
          points,
        }
      })
      .filter((row) => row.points.length >= 2)
  }, [entries, mode])

  const allGws = series.flatMap((s) => s.points.map((p) => p.gw))
  const allValues = series.flatMap((s) => s.points.map((p) => p.value))

  if (series.length === 0 || allGws.length === 0 || allValues.length === 0) {
    return <p className="text-sm text-muted-foreground">Not enough data for this chart yet.</p>
  }

  const gwMin = Math.min(...allGws)
  const gwMax = Math.max(...allGws)
  const valMin = Math.min(...allValues)
  const valMax = Math.max(...allValues)

  return (
    <div className="h-72 w-full">
      <ParentSize>
        {({ width, height }) => {
          const chartWidth = Math.max(width, 420)
          const chartHeight = Math.max(height, 240)
          const margin = { top: 16, right: 28, bottom: 36, left: 58 }
          const innerWidth = chartWidth - margin.left - margin.right
          const innerHeight = chartHeight - margin.top - margin.bottom

          const xScale = scaleLinear<number>({
            domain: [gwMin, gwMax],
            range: [0, innerWidth],
          })

          const yScale =
            mode === "rank"
              ? scaleLinear<number>({
                  domain: [valMax, valMin], // lower rank is better
                  range: [innerHeight, 0],
                  nice: true,
                })
              : scaleLinear<number>({
                  domain: [valMin, valMax],
                  range: [innerHeight, 0],
                  nice: true,
                })

          return (
            <svg width={chartWidth} height={chartHeight}>
              <Group left={margin.left} top={margin.top}>
                {series.map(({ entry, points }, index) => {
                  const isHighlighted = highlightEntryId != null && entry.entry_id === highlightEntryId
                  const stroke = isHighlighted ? "#facc15" : LINE_COLORS[index % LINE_COLORS.length]
                  const width = isHighlighted ? 3 : 1.6
                  return (
                    <g key={`${mode}-${entry.entry_id}`}>
                      <LinePath
                        data={points}
                        x={(d) => xScale(d.gw)}
                        y={(d) => yScale(d.value)}
                        stroke={stroke}
                        strokeOpacity={isHighlighted ? 1 : 0.72}
                        strokeWidth={width}
                      />
                      {points.slice(-1).map((last, pointIndex) => (
                        <text
                          key={`${entry.entry_id}-label-${pointIndex}`}
                          x={xScale(last.gw) + 6}
                          y={yScale(last.value) + 3}
                          fontSize={10}
                          fill={stroke}
                        >
                          {entry.team_info?.team_name ?? entry.entry_id}
                        </text>
                      ))}
                    </g>
                  )
                })}

                <AxisBottom
                  top={innerHeight}
                  scale={xScale}
                  numTicks={Math.min(8, Math.max(2, gwMax - gwMin + 1))}
                  tickFormat={(value) => `GW${Math.round(Number(value))}`}
                  tickLabelProps={() => ({
                    fill: "#a1a1aa",
                    fontSize: 10,
                    textAnchor: "middle",
                  })}
                  stroke="#52525b"
                  tickStroke="#52525b"
                />
                <AxisLeft
                  scale={yScale}
                  numTicks={6}
                  tickLabelProps={() => ({
                    fill: "#a1a1aa",
                    fontSize: 10,
                    textAnchor: "end",
                    dx: "-0.25em",
                    dy: "0.25em",
                  })}
                  stroke="#52525b"
                  tickStroke="#52525b"
                />
              </Group>
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}

function latestTransfers(entry: LeagueEntry): LeagueTransferTimelineEntry | null {
  const timeline = entry.transfer_history?.transfer_timeline ?? []
  if (timeline.length === 0) return null
  return [...timeline].sort((a, b) => b.gw - a.gw)[0]
}

function TransferDiffTable({ entries }: { entries: LeagueEntry[] }) {
  const managers = entries.slice(0, 5)
  const gws = Array.from(
    new Set(
      managers.flatMap((entry) =>
        (entry.transfer_history?.transfer_timeline ?? []).map((item) => item.gw)
      )
    )
  )
    .sort((a, b) => b - a)
    .slice(0, 5)

  if (managers.length === 0 || gws.length === 0) {
    return <p className="text-sm text-muted-foreground">No transfer timeline data available.</p>
  }

  return (
    <ScrollArea className="w-full">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-16">GW</TableHead>
            {managers.map((entry) => (
              <TableHead key={`head-${entry.entry_id}`} className="min-w-56">
                {getTeamLabel(entry)}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {gws.map((gw) => (
            <TableRow key={`gw-${gw}`}>
              <TableCell className="font-medium">GW{gw}</TableCell>
              {managers.map((entry) => {
                const row = (entry.transfer_history?.transfer_timeline ?? []).find((item) => item.gw === gw)
                const ins = row?.transfers_in ?? []
                const outs = row?.transfers_out ?? []
                return (
                  <TableCell key={`${entry.entry_id}-${gw}`}>
                    <div className="space-y-1 text-xs">
                      <div>
                        <span className="text-muted-foreground">In:</span>{" "}
                        {ins.length ? ins.map((player) => player.name).join(", ") : "-"}
                      </div>
                      <div>
                        <span className="text-muted-foreground">Out:</span>{" "}
                        {outs.length ? outs.map((player) => player.name).join(", ") : "-"}
                      </div>
                      {row?.chip ? <Badge variant="outline" className="mt-1">{row.chip}</Badge> : null}
                    </div>
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  )
}

function ChipTimeline({ entries }: { entries: LeagueEntry[] }) {
  if (entries.length === 0) {
    return <p className="text-sm text-muted-foreground">No chip history found.</p>
  }

  return (
    <div className="space-y-2">
      {entries.slice(0, 8).map((entry) => {
        const chips = [...(entry.chips_used ?? [])].sort((a, b) => a.event - b.event)
        return (
          <div key={`chip-${entry.entry_id}`} className="rounded border border-border p-2">
            <p className="text-sm font-medium">{getTeamLabel(entry)}</p>
            <div className="mt-1 flex flex-wrap gap-1">
              {chips.length ? (
                chips.map((chip, index) => (
                  <Badge key={`${entry.entry_id}-${chip.event}-${index}`} variant="outline">
                    GW{chip.event} {chip.name}
                  </Badge>
                ))
              ) : (
                <span className="text-xs text-muted-foreground">No chips used yet.</span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function buildContributionData(entry: LeagueEntry): Array<{ name: string; value: number }> {
  const totals = new Map<string, number>()

  for (const gw of entry.season_history ?? []) {
    const squad = gw.squad ?? []
    for (const player of squad) {
      const name = player.name
      if (!name) continue
      const stats = (player.stats ?? {}) as Record<string, unknown>
      const eventPointsRaw = stats.event_points
      const eventPoints = typeof eventPointsRaw === "number" ? eventPointsRaw : 0
      totals.set(name, (totals.get(name) ?? 0) + eventPoints)
    }
  }

  return [...totals.entries()]
    .map(([name, value]) => ({ name, value }))
    .filter((row) => row.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 14)
}

function ContributionTreemap({ entry }: { entry: LeagueEntry | null }) {
  if (!entry) {
    return <p className="text-sm text-muted-foreground">No entry selected for contribution map.</p>
  }

  const rows = buildContributionData(entry)
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">Contribution data unavailable for this entry.</p>
  }

  const firstRow: Array<{ name: string; value: number }> = []
  const secondRow: Array<{ name: string; value: number }> = []
  let firstTotal = 0
  let secondTotal = 0
  for (const row of rows) {
    if (firstTotal <= secondTotal) {
      firstRow.push(row)
      firstTotal += row.value
    } else {
      secondRow.push(row)
      secondTotal += row.value
    }
  }

  const total = rows.reduce((sum, row) => sum + row.value, 0)
  const palette = ["#0891b2", "#0284c7", "#0ea5e9", "#06b6d4", "#14b8a6", "#22c55e", "#84cc16", "#f59e0b"]

  const renderRow = (
    data: Array<{ name: string; value: number }>,
    y: number,
    height: number,
    rowTotal: number
  ) => {
    let xOffset = 0
    return data.map((item, idx) => {
      const widthRatio = rowTotal > 0 ? item.value / rowTotal : 0
      const width = widthRatio
      const color = palette[idx % palette.length]
      const node = {
        key: `${item.name}-${idx}`,
        x: xOffset,
        y,
        width,
        height,
        color,
        label: item.name,
        points: item.value,
      }
      xOffset += width
      return node
    })
  }

  const boxes = [
    ...renderRow(firstRow, 0, 0.5, firstTotal),
    ...renderRow(secondRow, 0.5, 0.5, secondTotal),
  ]

  return (
    <div className="h-60 w-full">
      <ParentSize>
        {({ width, height }) => {
          const w = Math.max(width, 320)
          const h = Math.max(height, 220)
          return (
            <svg width={w} height={h}>
              <Group>
                {boxes.map((box) => {
                  const x = box.x * w
                  const y = box.y * h
                  const bw = Math.max(box.width * w - 2, 0)
                  const bh = Math.max(box.height * h - 2, 0)
                  const pct = ((box.points / total) * 100).toFixed(1)
                  return (
                    <g key={box.key}>
                      <Bar
                        x={x + 1}
                        y={y + 1}
                        width={bw}
                        height={bh}
                        rx={4}
                        fill={box.color}
                        fillOpacity={0.82}
                      />
                      {bw > 72 && bh > 28 ? (
                        <text x={x + 8} y={y + 18} fontSize={11} fill="#f5f5f5">
                          {box.label}
                        </text>
                      ) : null}
                      {bw > 54 && bh > 40 ? (
                        <text x={x + 8} y={y + 34} fontSize={10} fill="#e4e4e7">
                          {box.points} pts ({pct}%)
                        </text>
                      ) : null}
                    </g>
                  )
                })}
              </Group>
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}

export function LeaguePage() {
  const [mode, setMode] = useState<LeagueMode>("competitors")
  const competitorQuery = useCompetitors()
  const globalQuery = useGlobalManagers()
  const { data: meta } = useMeta()

  const activeQuery = mode === "competitors" ? competitorQuery : globalQuery
  const entries = mode === "competitors"
    ? (competitorQuery.data?.competitors ?? [])
    : (globalQuery.data?.global_managers ?? [])

  const myEntryId = meta?.team_info?.team_id
  const focusEntry =
    entries.find((entry) => entry.entry_id === myEntryId) ??
    entries[0] ??
    null

  if (activeQuery.isLoading && entries.length === 0) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-80 w-full" />
      </div>
    )
  }

  if (activeQuery.error) {
    const isWarmingUp = activeQuery.error.message === "warming_up"
    if (isWarmingUp) {
      return (
        <div className="flex flex-col items-center justify-center gap-3 py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">League data is computing...</p>
        </div>
      )
    }
    return (
      <div className="py-10 text-center text-sm text-red-400">
        Failed to load league data: {activeQuery.error.message}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">League Comparison</h2>
          <p className="text-sm text-muted-foreground">
            Rank and points progression, transfer deltas, chips, and contribution breakdowns.
          </p>
        </div>
        <Tabs value={mode} onValueChange={(value) => setMode(value as LeagueMode)}>
          <TabsList>
            <TabsTrigger value="competitors">Competitors</TabsTrigger>
            <TabsTrigger value="global">Top Global</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Entries</CardDescription>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">{entries.length}</CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Focus Team</CardDescription>
          </CardHeader>
          <CardContent className="text-sm font-medium">{focusEntry ? getTeamLabel(focusEntry) : "-"}</CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Focus Rank</CardDescription>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">
            {focusEntry ? formatNumber(getRankPoint((focusEntry.gw_history ?? []).slice(-1)[0] as unknown as Record<string, unknown> | undefined) ?? undefined) : "-"}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Rank Progression</CardTitle>
            <CardDescription>Overall rank by gameweek (lower is better).</CardDescription>
          </CardHeader>
          <CardContent>
            <ProgressionChart entries={entries} mode="rank" highlightEntryId={myEntryId} />
          </CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Points Progression</CardTitle>
            <CardDescription>Total points by gameweek.</CardDescription>
          </CardHeader>
          <CardContent>
            <ProgressionChart entries={entries} mode="points" highlightEntryId={myEntryId} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Transfer Diff Table</CardTitle>
            <CardDescription>Side-by-side recent GW transfer activity per manager.</CardDescription>
          </CardHeader>
          <CardContent>
            <TransferDiffTable entries={entries} />
          </CardContent>
        </Card>

        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Chip Timeline</CardTitle>
            <CardDescription>When each manager activated chips.</CardDescription>
          </CardHeader>
          <CardContent>
            <ChipTimeline entries={entries} />
          </CardContent>
        </Card>
      </div>

      <Card className="bg-card/70">
        <CardHeader>
          <CardTitle className="text-base">Player Contribution Treemap</CardTitle>
          <CardDescription>
            Season contribution share by player for {focusEntry ? getManagerLabel(focusEntry) : "selected manager"}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ContributionTreemap entry={focusEntry} />
        </CardContent>
      </Card>

      <Card className="bg-card/70">
        <CardHeader>
          <CardTitle className="text-base">Manager Snapshot</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {entries.slice(0, 9).map((entry) => {
              const latest = (entry.gw_history ?? []).slice(-1)[0] as unknown as Record<string, unknown> | undefined
              const latestRank = getRankPoint(latest)
              const latestPoints = getPointsPoint(latest)
              const latestTransfer = latestTransfers(entry)
              return (
                <div key={`snapshot-${entry.entry_id}`} className={cn("rounded border border-border p-3", entry.entry_id === myEntryId && "border-amber-400/60 bg-amber-400/10")}>
                  <p className="text-sm font-semibold">{getTeamLabel(entry)}</p>
                  <p className="text-xs text-muted-foreground">{entry.team_info?.manager_name}</p>
                  <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                    <span className="text-muted-foreground">Rank</span>
                    <span>{formatNumber(latestRank ?? undefined)}</span>
                    <span className="text-muted-foreground">Points</span>
                    <span>{formatNumber(latestPoints ?? undefined)}</span>
                    <span className="text-muted-foreground">Team Value</span>
                    <span>{entry.team_value?.toFixed(1)}m</span>
                    <span className="text-muted-foreground">Hits</span>
                    <span>{entry.total_hits}</span>
                  </div>
                  {latestTransfer ? (
                    <div className="mt-2 text-xs text-muted-foreground">
                      GW{latestTransfer.gw}: {latestTransfer.transfers_in.length} in / {latestTransfer.transfers_out.length} out
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
