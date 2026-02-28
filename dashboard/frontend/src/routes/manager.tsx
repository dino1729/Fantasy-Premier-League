import { useMemo } from "react"
import { AxisBottom, AxisLeft } from "@visx/axis"
import { Group } from "@visx/group"
import { ParentSize } from "@visx/responsive"
import { scaleBand, scaleLinear } from "@visx/scale"
import { AreaClosed, Bar, LinePath } from "@visx/shape"
import { useCaptains, useManagerOverview } from "@/hooks/use-api"
import type { CaptainPick, GWHistoryEntry } from "@/lib/api"
import { formatNumber } from "@/lib/fpl"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Loader2 } from "lucide-react"

interface ManagerGwPoint {
  gw: number
  points: number
  rank: number
  value: number
}

function toManagerGwPoints(history: GWHistoryEntry[] | undefined): ManagerGwPoint[] {
  return (history ?? [])
    .map((row) => ({
      gw: Number(row.event),
      points: Number(row.points ?? 0),
      rank: Number(row.overall_rank ?? 0),
      value: Number(row.value ?? 0) / 10,
    }))
    .filter((row) => Number.isFinite(row.gw) && row.gw > 0)
    .sort((a, b) => a.gw - b.gw)
}

function RankJourneyChart({ rows }: { rows: ManagerGwPoint[] }) {
  if (rows.length < 2) {
    return <p className="text-sm text-muted-foreground">Not enough GW data for rank journey yet.</p>
  }

  const rankRows = rows.filter((row) => row.rank > 0)
  if (rankRows.length < 2) {
    return <p className="text-sm text-muted-foreground">Rank data unavailable for this season snapshot.</p>
  }

  const gwMin = Math.min(...rankRows.map((row) => row.gw))
  const gwMax = Math.max(...rankRows.map((row) => row.gw))
  const rankMin = Math.min(...rankRows.map((row) => row.rank))
  const rankMax = Math.max(...rankRows.map((row) => row.rank))

  return (
    <div className="h-64 w-full">
      <ParentSize>
        {({ width, height }) => {
          const chartWidth = Math.max(width, 360)
          const chartHeight = Math.max(height, 240)
          const margin = { top: 16, right: 20, bottom: 36, left: 64 }
          const innerWidth = chartWidth - margin.left - margin.right
          const innerHeight = chartHeight - margin.top - margin.bottom

          const xScale = scaleLinear<number>({
            domain: [gwMin, gwMax],
            range: [0, innerWidth],
          })
          const yScale = scaleLinear<number>({
            domain: [rankMax, rankMin], // lower rank is better
            range: [innerHeight, 0],
            nice: true,
          })

          return (
            <svg width={chartWidth} height={chartHeight}>
              <Group left={margin.left} top={margin.top}>
                <AreaClosed
                  data={rankRows}
                  x={(d) => xScale(d.gw)}
                  y={(d) => yScale(d.rank)}
                  yScale={yScale}
                  stroke="none"
                  fill="#0891b2"
                  fillOpacity={0.2}
                />
                <LinePath
                  data={rankRows}
                  x={(d) => xScale(d.gw)}
                  y={(d) => yScale(d.rank)}
                  stroke="#22d3ee"
                  strokeWidth={2.2}
                />

                <AxisBottom
                  top={innerHeight}
                  scale={xScale}
                  tickFormat={(value) => `GW${Math.round(Number(value))}`}
                  numTicks={Math.min(8, Math.max(3, gwMax - gwMin + 1))}
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

function PointsBarChart({ rows }: { rows: ManagerGwPoint[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No GW points available yet.</p>
  }

  return (
    <div className="h-64 w-full">
      <ParentSize>
        {({ width, height }) => {
          const chartWidth = Math.max(width, 360)
          const chartHeight = Math.max(height, 240)
          const margin = { top: 16, right: 18, bottom: 40, left: 48 }
          const innerWidth = chartWidth - margin.left - margin.right
          const innerHeight = chartHeight - margin.top - margin.bottom

          const xScale = scaleBand<string>({
            domain: rows.map((row) => String(row.gw)),
            range: [0, innerWidth],
            padding: 0.22,
          })
          const yScale = scaleLinear<number>({
            domain: [0, Math.max(...rows.map((row) => row.points), 1) * 1.15],
            range: [innerHeight, 0],
            nice: true,
          })

          return (
            <svg width={chartWidth} height={chartHeight}>
              <Group left={margin.left} top={margin.top}>
                {rows.map((row) => {
                  const x = xScale(String(row.gw)) ?? 0
                  const y = yScale(row.points)
                  const h = innerHeight - y
                  return (
                    <Bar
                      key={`gw-${row.gw}`}
                      x={x}
                      y={y}
                      width={xScale.bandwidth()}
                      height={h}
                      fill="#22c55e"
                      fillOpacity={0.82}
                      rx={3}
                    />
                  )
                })}

                <AxisBottom
                  top={innerHeight}
                  scale={xScale}
                  numTicks={Math.min(10, rows.length)}
                  tickFormat={(value) => `GW${value}`}
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

function TeamValueChart({ rows }: { rows: ManagerGwPoint[] }) {
  if (rows.length < 2) {
    return <p className="text-sm text-muted-foreground">Need at least 2 gameweeks for team value trend.</p>
  }

  const gwMin = Math.min(...rows.map((row) => row.gw))
  const gwMax = Math.max(...rows.map((row) => row.gw))
  const valueMin = Math.min(...rows.map((row) => row.value))
  const valueMax = Math.max(...rows.map((row) => row.value))

  return (
    <div className="h-64 w-full">
      <ParentSize>
        {({ width, height }) => {
          const chartWidth = Math.max(width, 360)
          const chartHeight = Math.max(height, 240)
          const margin = { top: 16, right: 18, bottom: 36, left: 52 }
          const innerWidth = chartWidth - margin.left - margin.right
          const innerHeight = chartHeight - margin.top - margin.bottom

          const xScale = scaleLinear<number>({
            domain: [gwMin, gwMax],
            range: [0, innerWidth],
          })
          const yScale = scaleLinear<number>({
            domain: [valueMin * 0.995, valueMax * 1.005],
            range: [innerHeight, 0],
            nice: true,
          })

          return (
            <svg width={chartWidth} height={chartHeight}>
              <Group left={margin.left} top={margin.top}>
                <AreaClosed
                  data={rows}
                  x={(d) => xScale(d.gw)}
                  y={(d) => yScale(d.value)}
                  yScale={yScale}
                  stroke="none"
                  fill="#f59e0b"
                  fillOpacity={0.18}
                />
                <LinePath
                  data={rows}
                  x={(d) => xScale(d.gw)}
                  y={(d) => yScale(d.value)}
                  stroke="#fbbf24"
                  strokeWidth={2.1}
                />
                <AxisBottom
                  top={innerHeight}
                  scale={xScale}
                  tickFormat={(value) => `GW${Math.round(Number(value))}`}
                  numTicks={Math.min(8, Math.max(3, gwMax - gwMin + 1))}
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
                  tickFormat={(value) => `${Number(value).toFixed(1)}m`}
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

function CaptainTable({ rows }: { rows: CaptainPick[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No captain picks available yet.</p>
  }

  return (
    <ScrollArea className="w-full">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-20">GW</TableHead>
            <TableHead>Captain</TableHead>
            <TableHead className="text-right">Return</TableHead>
            <TableHead>Optimal</TableHead>
            <TableHead className="text-right">Optimal Return</TableHead>
            <TableHead className="text-right">Lost</TableHead>
            <TableHead className="text-right">Result</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={`cap-${row.gameweek}`}>
              <TableCell>GW{row.gameweek}</TableCell>
              <TableCell>
                <div className="text-sm">{row.captain_name}</div>
                <div className="text-xs text-muted-foreground">
                  {row.captain_points} pts x{row.captain_multiplier}
                </div>
              </TableCell>
              <TableCell className="text-right">{row.captain_return}</TableCell>
              <TableCell>{row.optimal_name}</TableCell>
              <TableCell className="text-right">{row.optimal_return}</TableCell>
              <TableCell className="text-right">{row.points_lost}</TableCell>
              <TableCell className="text-right">
                {row.was_optimal ? (
                  <Badge variant="outline" className="border-emerald-500/50 text-emerald-400">
                    Optimal
                  </Badge>
                ) : (
                  <Badge variant="outline" className="border-amber-500/50 text-amber-400">
                    Missed
                  </Badge>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  )
}

export function ManagerPage() {
  const overviewQuery = useManagerOverview()
  const captainsQuery = useCaptains()

  const points = useMemo(
    () => toManagerGwPoints(overviewQuery.data?.gw_history),
    [overviewQuery.data?.gw_history]
  )
  const latest = points.length > 0 ? points[points.length - 1] : null
  const captainRows = captainsQuery.data?.captain_picks ?? []
  const captainSummary = captainsQuery.data?.summary

  if (overviewQuery.isLoading || captainsQuery.isLoading) {
    return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
        <Skeleton className="h-72 w-full" />
      </div>
    )
  }

  const errors = [overviewQuery.error, captainsQuery.error].filter(Boolean)
  if (errors.length > 0) {
    const warmup = errors.some((err) => err instanceof Error && err.message === "warming_up")
    return (
      <Card className="border-zinc-700/70 bg-zinc-900/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            {warmup ? <Loader2 className="h-5 w-5 animate-spin text-amber-400" /> : null}
            Manager Report
          </CardTitle>
          <CardDescription>
            {warmup
              ? "Manager data is still warming up. This page auto-refreshes as jobs complete."
              : "Failed to load manager data."}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">Manager Report</h2>
        <p className="text-sm text-muted-foreground">
          Rank journey, weekly points, team value trend, and captain decision accuracy.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Current Overall Rank</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">{formatNumber(latest?.rank)}</p>
          </CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Total Points</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">
              {formatNumber(overviewQuery.data?.team_info?.overall_points)}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Team Value</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">
              {overviewQuery.data?.team_value != null ? `${overviewQuery.data.team_value.toFixed(1)}m` : "-"}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/70">
          <CardHeader className="pb-2">
            <CardDescription>Free Transfers</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">{overviewQuery.data?.free_transfers ?? "-"}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Rank Journey</CardTitle>
            <CardDescription>Overall rank progression by gameweek.</CardDescription>
          </CardHeader>
          <CardContent>
            <RankJourneyChart rows={points} />
          </CardContent>
        </Card>

        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Points Per Gameweek</CardTitle>
            <CardDescription>Weekly scoring distribution.</CardDescription>
          </CardHeader>
          <CardContent>
            <PointsBarChart rows={points} />
          </CardContent>
        </Card>
      </div>

      <Card className="bg-card/70">
        <CardHeader>
          <CardTitle className="text-base">Team Value Trend</CardTitle>
          <CardDescription>Squad value progression in millions.</CardDescription>
        </CardHeader>
        <CardContent>
          <TeamValueChart rows={points} />
        </CardContent>
      </Card>

      <Card className="bg-card/70">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base">Captain Analysis</CardTitle>
              <CardDescription>
                Counterfactual view of actual captain returns versus best in-squad option.
              </CardDescription>
            </div>
            {captainSummary ? (
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">
                  Accuracy {(captainSummary.accuracy * 100).toFixed(1)}%
                </Badge>
                <Badge variant="outline">
                  Lost {captainSummary.captain_points_lost} pts
                </Badge>
                <Badge variant="outline">
                  {captainSummary.correct_picks}/{captainSummary.total_gws} optimal
                </Badge>
              </div>
            ) : null}
          </div>
        </CardHeader>
        <CardContent>
          <CaptainTable rows={[...captainRows].sort((a, b) => b.gameweek - a.gameweek)} />
        </CardContent>
      </Card>
    </div>
  )
}
