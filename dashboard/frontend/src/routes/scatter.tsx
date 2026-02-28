import { useMemo, useState } from "react"
import { AxisBottom, AxisLeft } from "@visx/axis"
import { GridColumns, GridRows } from "@visx/grid"
import { Group } from "@visx/group"
import { ParentSize } from "@visx/responsive"
import { scaleLinear, scaleSqrt } from "@visx/scale"
import { useScatter } from "@/hooks/use-api"
import type { ScatterPoint } from "@/lib/api"
import { POSITION_COLORS } from "@/lib/fpl"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2 } from "lucide-react"

type ScatterType = "xg_goals" | "xa_assists" | "usage_output" | "defensive"

const SCATTER_CONFIG: Array<{
  type: ScatterType
  label: string
  xLabel: string
  yLabel: string
  description: string
}> = [
  {
    type: "xg_goals",
    label: "xG vs Goals",
    xLabel: "Expected Goals (xG)",
    yLabel: "Goals",
    description: "Finishing efficiency: players above the diagonal are overperforming xG.",
  },
  {
    type: "xa_assists",
    label: "xA vs Assists",
    xLabel: "Expected Assists (xA)",
    yLabel: "Assists",
    description: "Creative conversion: players converting chances into real assists.",
  },
  {
    type: "usage_output",
    label: "Minutes vs Points",
    xLabel: "Minutes Played",
    yLabel: "Total Points",
    description: "Reliability and output: who turns minutes into FPL returns.",
  },
  {
    type: "defensive",
    label: "CS vs BPS",
    xLabel: "Clean Sheets",
    yLabel: "Bonus Points",
    description: "Defensive contribution versus BPS accumulation.",
  },
]

interface HoveredPoint {
  point: ScatterPoint
  x: number
  y: number
}

function clampDomain(min: number, max: number): [number, number] {
  if (min === max) {
    return [min - 1, max + 1]
  }
  const pad = (max - min) * 0.08
  return [min - pad, max + pad]
}

function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2
  }
  return sorted[mid]
}

function formatNumber(value: number, digits = 1): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function PositionLegend() {
  return (
    <div className="flex flex-wrap gap-2">
      {Object.entries(POSITION_COLORS).map(([position, color]) => (
        <div key={position} className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
          <span>{position}</span>
        </div>
      ))}
    </div>
  )
}

function ScatterChart({
  points,
  xLabel,
  yLabel,
  xMid,
  yMid,
}: {
  points: ScatterPoint[]
  xLabel: string
  yLabel: string
  xMid: number
  yMid: number
}) {
  const [hovered, setHovered] = useState<HoveredPoint | null>(null)

  const xValues = points.map((p) => p.x)
  const yValues = points.map((p) => p.y)
  const minuteValues = points.map((p) => p.minutes)
  const [xMinRaw, xMaxRaw] = [Math.min(...xValues), Math.max(...xValues)]
  const [yMinRaw, yMaxRaw] = [Math.min(...yValues), Math.max(...yValues)]
  const [xMin, xMax] = clampDomain(xMinRaw, xMaxRaw)
  const [yMin, yMax] = clampDomain(yMinRaw, yMaxRaw)
  const minutesMin = Math.min(...minuteValues)
  const minutesMax = Math.max(...minuteValues)

  return (
    <div className="relative h-[26rem] w-full">
      <ParentSize>
        {({ width, height }) => {
          const chartWidth = Math.max(width, 420)
          const chartHeight = Math.max(height, 320)
          const margin = { top: 18, right: 22, bottom: 54, left: 56 }
          const innerWidth = chartWidth - margin.left - margin.right
          const innerHeight = chartHeight - margin.top - margin.bottom

          const xScale = scaleLinear<number>({
            domain: [xMin, xMax],
            range: [0, innerWidth],
            nice: true,
          })
          const yScale = scaleLinear<number>({
            domain: [yMin, yMax],
            range: [innerHeight, 0],
            nice: true,
          })
          const rScale = scaleSqrt<number>({
            domain: [minutesMin, Math.max(minutesMin + 1, minutesMax)],
            range: [3, 10],
          })

          return (
            <svg width={chartWidth} height={chartHeight}>
              <Group left={margin.left} top={margin.top}>
                <GridRows
                  scale={yScale}
                  width={innerWidth}
                  stroke="#3f3f46"
                  strokeOpacity={0.35}
                  numTicks={6}
                />
                <GridColumns
                  scale={xScale}
                  height={innerHeight}
                  stroke="#3f3f46"
                  strokeOpacity={0.2}
                  numTicks={8}
                />

                <line
                  x1={0}
                  x2={innerWidth}
                  y1={yScale(yMid)}
                  y2={yScale(yMid)}
                  stroke="#f59e0b"
                  strokeDasharray="4,4"
                  strokeOpacity={0.9}
                />
                <line
                  x1={xScale(xMid)}
                  x2={xScale(xMid)}
                  y1={0}
                  y2={innerHeight}
                  stroke="#f59e0b"
                  strokeDasharray="4,4"
                  strokeOpacity={0.9}
                />

                {points.map((point) => {
                  const cx = xScale(point.x)
                  const cy = yScale(point.y)
                  const radius = rScale(point.minutes)
                  const color = POSITION_COLORS[point.position] ?? "#a1a1aa"
                  return (
                    <circle
                      key={`${point.player_id}-${point.name}`}
                      cx={cx}
                      cy={cy}
                      r={radius}
                      fill={color}
                      fillOpacity={0.8}
                      stroke="#18181b"
                      strokeWidth={1}
                      onMouseEnter={() => setHovered({ point, x: margin.left + cx, y: margin.top + cy })}
                      onMouseMove={() => setHovered({ point, x: margin.left + cx, y: margin.top + cy })}
                      onMouseLeave={() => setHovered(null)}
                    />
                  )
                })}

                <AxisBottom
                  top={innerHeight}
                  scale={xScale}
                  numTicks={7}
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

                <text x={innerWidth / 2} y={innerHeight + 42} textAnchor="middle" fontSize={11} fill="#d4d4d8">
                  {xLabel}
                </text>
                <text
                  x={-innerHeight / 2}
                  y={-38}
                  transform="rotate(-90)"
                  textAnchor="middle"
                  fontSize={11}
                  fill="#d4d4d8"
                >
                  {yLabel}
                </text>
              </Group>
            </svg>
          )
        }}
      </ParentSize>

      {hovered ? (
        <div
          className="pointer-events-none absolute z-10 min-w-44 rounded border border-zinc-700 bg-zinc-950/95 px-3 py-2 text-xs shadow-xl"
          style={{ left: hovered.x + 10, top: hovered.y - 10 }}
        >
          <p className="font-semibold text-zinc-100">{hovered.point.name}</p>
          <p className="text-zinc-400">
            {hovered.point.team} - {hovered.point.position}
          </p>
          <p className="mt-1 text-zinc-300">{xLabel}: {formatNumber(hovered.point.x, 2)}</p>
          <p className="text-zinc-300">{yLabel}: {formatNumber(hovered.point.y, 2)}</p>
          <p className="text-zinc-400">Minutes: {formatNumber(hovered.point.minutes, 0)}</p>
        </div>
      ) : null}
    </div>
  )
}

export function ScatterPage() {
  const [chartType, setChartType] = useState<ScatterType>("xg_goals")
  const chartConfig = SCATTER_CONFIG.find((c) => c.type === chartType) ?? SCATTER_CONFIG[0]
  const { data, isLoading, error } = useScatter(chartType)

  const points = data?.data ?? []
  const xMedian = useMemo(() => median(points.map((p) => p.x)), [points])
  const yMedian = useMemo(() => median(points.map((p) => p.y)), [points])
  const quadrants = useMemo(() => {
    const counts = { q1: 0, q2: 0, q3: 0, q4: 0 }
    for (const point of points) {
      if (point.x >= xMedian && point.y >= yMedian) counts.q1 += 1
      else if (point.x < xMedian && point.y >= yMedian) counts.q2 += 1
      else if (point.x < xMedian && point.y < yMedian) counts.q3 += 1
      else counts.q4 += 1
    }
    return counts
  }, [points, xMedian, yMedian])

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-8 w-52" />
          <Skeleton className="h-4 w-80" />
        </div>
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-[26rem] w-full" />
      </div>
    )
  }

  if (error) {
    const warmup = error instanceof Error && error.message === "warming_up"
    return (
      <Card className="border-zinc-700/70 bg-zinc-900/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            {warmup ? <Loader2 className="h-5 w-5 animate-spin text-amber-400" /> : null}
            Scatter Analysis
          </CardTitle>
          <CardDescription>
            {warmup
              ? "Analysis job is still running. This page auto-refreshes when data is ready."
              : `Unable to load scatter data: ${error instanceof Error ? error.message : "unknown error"}`}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">Scatter Analysis</h2>
        <p className="text-sm text-muted-foreground">
          Four player relationship charts with median quadrants and hover drill-down.
        </p>
      </div>

      <Tabs value={chartType} onValueChange={(value) => setChartType(value as ScatterType)}>
        <TabsList className="grid h-auto w-full grid-cols-2 gap-1 lg:grid-cols-4">
          {SCATTER_CONFIG.map((config) => (
            <TabsTrigger key={config.type} value={config.type} className="py-2 text-xs md:text-sm">
              {config.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <Card className="bg-card/70">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base">{chartConfig.label}</CardTitle>
              <CardDescription>{chartConfig.description}</CardDescription>
            </div>
            <Badge variant="outline">{points.length} players</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {points.length === 0 ? (
            <p className="text-sm text-muted-foreground">No points available for this scatter type yet.</p>
          ) : (
            <>
              <ScatterChart
                points={points}
                xLabel={chartConfig.xLabel}
                yLabel={chartConfig.yLabel}
                xMid={xMedian}
                yMid={yMedian}
              />
              <PositionLegend />
              <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2 xl:grid-cols-4">
                <div className="rounded border border-border p-2">
                  <p className="font-medium text-foreground">Q1: High x / High y</p>
                  <p>{quadrants.q1} players</p>
                </div>
                <div className="rounded border border-border p-2">
                  <p className="font-medium text-foreground">Q2: Low x / High y</p>
                  <p>{quadrants.q2} players</p>
                </div>
                <div className="rounded border border-border p-2">
                  <p className="font-medium text-foreground">Q3: Low x / Low y</p>
                  <p>{quadrants.q3} players</p>
                </div>
                <div className="rounded border border-border p-2">
                  <p className="font-medium text-foreground">Q4: High x / Low y</p>
                  <p>{quadrants.q4} players</p>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
