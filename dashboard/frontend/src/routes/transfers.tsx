import { useEffect, useMemo, useState } from "react"
import { Group } from "@visx/group"
import { ParentSize } from "@visx/responsive"
import { scaleBand, scaleLinear } from "@visx/scale"
import { Bar } from "@visx/shape"
import { useSolver, useTransferHistory } from "@/hooks/use-api"
import type {
  SolverResponse,
  SolverScenario,
  SolverWeeklyPlan,
  TransferHistoryEntry,
  Underperformer,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { AlertTriangle, Loader2 } from "lucide-react"

type ScenarioKey = "conservative" | "balanced" | "aggressive"

const SCENARIO_META: Array<{ key: ScenarioKey; label: string; color: string }> = [
  { key: "conservative", label: "Conservative", color: "#60a5fa" },
  { key: "balanced", label: "Balanced", color: "#22c55e" },
  { key: "aggressive", label: "Aggressive", color: "#ef4444" },
]

function toScenarioKey(value: string | undefined): ScenarioKey {
  if (value === "conservative" || value === "balanced" || value === "aggressive") {
    return value
  }
  return "balanced"
}

function formatPoints(value: number | undefined): string {
  if (value == null) return "-"
  return value.toFixed(1)
}

function SummaryStat({
  label,
  value,
  sub,
}: {
  label: string
  value: string
  sub?: string
}) {
  return (
    <Card className="bg-card/70">
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold">{value}</div>
        {sub ? <p className="text-xs text-muted-foreground">{sub}</p> : null}
      </CardContent>
    </Card>
  )
}

function TransferChip({ name, position, team }: { name?: string; position?: string; team?: string }) {
  return (
    <div className="inline-flex items-center gap-1 rounded border border-border bg-muted/30 px-2 py-0.5 text-xs">
      <span className="font-medium">{name ?? "Unknown"}</span>
      {position ? <span className="text-muted-foreground">{position}</span> : null}
      {team ? <span className="text-muted-foreground">{team}</span> : null}
    </div>
  )
}

function WeeklyPlanCard({ plan }: { plan: SolverWeeklyPlan }) {
  const confidenceTone =
    plan.confidence === "high"
      ? "border-emerald-500/60 text-emerald-400"
      : plan.confidence === "moderate"
        ? "border-amber-500/60 text-amber-400"
        : "border-zinc-500/60 text-zinc-300"

  return (
    <Card className="bg-card/70">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base">GW{plan.gameweek}</CardTitle>
          <div className="flex items-center gap-2">
            {plan.is_hold ? <Badge variant="outline">Hold</Badge> : null}
            {plan.hit_cost > 0 ? <Badge variant="destructive">-{plan.hit_cost}</Badge> : <Badge variant="outline">No Hit</Badge>}
            <Badge variant="outline" className={confidenceTone}>
              {(plan.confidence ?? "unknown").toUpperCase()}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Transfers Out</p>
            <div className="flex flex-wrap gap-1">
              {plan.transfers_out.length > 0 ? (
                plan.transfers_out.map((player, index) => (
                  <TransferChip
                    key={`out-${plan.gameweek}-${index}`}
                    name={player.name}
                    position={player.position}
                    team={player.team}
                  />
                ))
              ) : (
                <span className="text-xs text-muted-foreground">None</span>
              )}
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Transfers In</p>
            <div className="flex flex-wrap gap-1">
              {plan.transfers_in.length > 0 ? (
                plan.transfers_in.map((player, index) => (
                  <TransferChip
                    key={`in-${plan.gameweek}-${index}`}
                    name={player.name}
                    position={player.position}
                    team={player.team}
                  />
                ))
              ) : (
                <span className="text-xs text-muted-foreground">None</span>
              )}
            </div>
          </div>
        </div>
        <Separator />
        <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
          <p>Expected xP: <span className="text-foreground">{formatPoints(plan.expected_xp)}</span></p>
          <p>FT used: <span className="text-foreground">{plan.ft_used}/{plan.ft_available}</span></p>
          <p>Formation: <span className="text-foreground">{plan.formation ?? "-"}</span></p>
          <p>Captain: <span className="text-foreground">{plan.captain?.name ?? "-"}</span></p>
        </div>
        {plan.reasoning ? <p className="text-xs text-muted-foreground">{plan.reasoning}</p> : null}
      </CardContent>
    </Card>
  )
}

function UnderperformersPanel({ players }: { players: Underperformer[] }) {
  return (
    <Card className="bg-card/70">
      <CardHeader>
        <CardTitle className="text-base">Underperformers</CardTitle>
        <CardDescription>Flagged by form trend, expected vs actual, and peer ranking.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {players.length === 0 ? (
          <p className="text-sm text-muted-foreground">No underperformers flagged.</p>
        ) : (
          players.slice(0, 10).map((player) => {
            const tone =
              player.severity >= 6
                ? "border-red-500/40 bg-red-500/10"
                : player.severity >= 4
                  ? "border-amber-500/40 bg-amber-500/10"
                  : "border-zinc-500/40 bg-zinc-500/10"
            return (
              <div key={`${player.player_id}-${player.name}`} className={cn("rounded border p-2", tone)}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-medium">{player.name}</p>
                  <Badge variant="outline">Severity {player.severity}</Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{player.reasons.join(" - ")}</p>
              </div>
            )
          })
        )}
      </CardContent>
    </Card>
  )
}

function BaselineXpChart({ solver }: { solver: SolverResponse }) {
  const chartData = SCENARIO_META.map(({ key, label, color }) => {
    const scenario = solver[key]
    const baseline = scenario?.baseline_xp ?? solver.baseline_xp ?? 0
    const expected = scenario?.expected_points ?? baseline
    return {
      key,
      label,
      color,
      status: scenario?.status ?? "pending",
      baseline,
      expected,
    }
  })

  return (
    <Card className="bg-card/70">
      <CardHeader>
        <CardTitle className="text-base">Baseline xP Comparison</CardTitle>
        <CardDescription>Scenario expected points vs no-transfer baseline.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-56 w-full">
          <ParentSize>
            {({ width, height }) => {
              const chartWidth = Math.max(width, 320)
              const chartHeight = Math.max(height, 220)
              const margin = { top: 16, right: 24, bottom: 42, left: 36 }
              const innerWidth = chartWidth - margin.left - margin.right
              const innerHeight = chartHeight - margin.top - margin.bottom

              const maxValue = Math.max(
                ...chartData.map((row) => Math.max(row.expected, row.baseline)),
                1
              )

              const xScale = scaleBand<string>({
                domain: chartData.map((row) => row.label),
                range: [0, innerWidth],
                padding: 0.35,
              })
              const yScale = scaleLinear<number>({
                domain: [0, maxValue * 1.15],
                range: [innerHeight, 0],
                nice: true,
              })

              return (
                <svg width={chartWidth} height={chartHeight}>
                  <Group left={margin.left} top={margin.top}>
                    {chartData.map((row) => {
                      const x = xScale(row.label) ?? 0
                      const y = yScale(row.expected)
                      const barHeight = innerHeight - y
                      const barWidth = xScale.bandwidth()
                      const muted = row.status !== "optimal"
                      return (
                        <g key={row.key}>
                          <Bar
                            x={x}
                            y={y}
                            width={barWidth}
                            height={barHeight}
                            rx={4}
                            fill={muted ? "#52525b" : row.color}
                            fillOpacity={muted ? 0.45 : 0.85}
                          />
                          <text
                            x={x + barWidth / 2}
                            y={y - 6}
                            textAnchor="middle"
                            fontSize={11}
                            fill="#d4d4d8"
                          >
                            {row.expected.toFixed(1)}
                          </text>
                          <text
                            x={x + barWidth / 2}
                            y={innerHeight + 18}
                            textAnchor="middle"
                            fontSize={11}
                            fill="#a1a1aa"
                          >
                            {row.label}
                          </text>
                        </g>
                      )
                    })}

                    {(() => {
                      const baseline = solver.baseline_xp ?? chartData[0]?.baseline ?? 0
                      const baselineY = yScale(baseline)
                      return (
                        <g>
                          <line
                            x1={0}
                            x2={innerWidth}
                            y1={baselineY}
                            y2={baselineY}
                            stroke="#f59e0b"
                            strokeDasharray="4,4"
                            strokeWidth={1.5}
                          />
                          <text x={innerWidth} y={baselineY - 6} textAnchor="end" fontSize={11} fill="#fbbf24">
                            Baseline {baseline.toFixed(1)}
                          </text>
                        </g>
                      )
                    })()}
                  </Group>
                </svg>
              )
            }}
          </ParentSize>
        </div>
      </CardContent>
    </Card>
  )
}

function TransferHistoryTable({ transfers }: { transfers: TransferHistoryEntry[] }) {
  return (
    <Card className="bg-card/70">
      <CardHeader>
        <CardTitle className="text-base">Recent Transfer History</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="w-full">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-24">GW</TableHead>
                <TableHead>Out</TableHead>
                <TableHead>In</TableHead>
                <TableHead className="w-28 text-right">Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {transfers.slice(0, 12).map((transfer, index) => (
                <TableRow key={`${transfer.event}-${transfer.element_in}-${transfer.element_out}-${index}`}>
                  <TableCell>GW{transfer.event}</TableCell>
                  <TableCell>
                    <div className="text-sm">{transfer.element_out_name ?? transfer.element_out}</div>
                    <div className="text-xs text-muted-foreground">{transfer.element_out_team ?? "-"}</div>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm">{transfer.element_in_name ?? transfer.element_in}</div>
                    <div className="text-xs text-muted-foreground">{transfer.element_in_team ?? "-"}</div>
                  </TableCell>
                  <TableCell className="text-right text-xs text-muted-foreground">
                    {new Date(transfer.time).toLocaleDateString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function getScenario(solver: SolverResponse | undefined, key: ScenarioKey): SolverScenario {
  const baseline = solver?.baseline_xp ?? 0
  return (
    solver?.[key] ?? {
      status: "pending",
      hit_cost: 0,
      num_transfers: 0,
      expected_points: baseline,
      baseline_xp: baseline,
      weekly_plans: [],
      message: "Scenario not available yet.",
    }
  )
}

function SolverLoadingSkeleton({ message }: { message?: string }) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-72" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Skeleton className="h-[28rem] w-full" />
        <div className="space-y-4">
          <Skeleton className="h-52 w-full" />
          <Skeleton className="h-56 w-full" />
        </div>
      </div>
      {message ? (
        <div className="flex items-center justify-center gap-2 rounded border border-border bg-card/60 py-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span>{message}</span>
        </div>
      ) : null}
    </div>
  )
}

export function TransfersPage() {
  const { data: solver, isLoading, error } = useSolver()
  const { data: history } = useTransferHistory()

  const recommended = toScenarioKey(solver?.recommended)
  const [tab, setTab] = useState<ScenarioKey>(recommended)

  useEffect(() => {
    setTab(recommended)
  }, [recommended])

  const scenario = useMemo(() => getScenario(solver, tab), [solver, tab])
  const weeklyPlans = scenario.weekly_plans ?? []
  const totalFtUsed = weeklyPlans.reduce((acc, plan) => acc + (plan.ft_used ?? 0), 0)
  const baseline = scenario.baseline_xp ?? solver?.baseline_xp ?? 0
  const netGain = scenario.net_gain ?? (scenario.expected_points ?? baseline) - baseline

  if (isLoading) {
    return <SolverLoadingSkeleton />
  }

  if (error) {
    const isWarmingUp = error.message === "warming_up"
    if (isWarmingUp) {
      return <SolverLoadingSkeleton message="Solver is computing transfer scenarios..." />
    }
    return <div className="py-10 text-center text-red-400 text-sm">Failed to load solver data: {error.message}</div>
  }

  if (!solver) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Waiting for solver output...</p>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold">Transfer Hub</h2>
        <p className="text-sm text-muted-foreground">
          Multi-scenario MIP planning with weekly transfer timeline and underperformer flags.
        </p>
      </div>

      <Tabs value={tab} onValueChange={(value) => setTab(toScenarioKey(value))}>
        <TabsList>
          {SCENARIO_META.map(({ key, label }) => (
            <TabsTrigger key={key} value={key} className="min-w-32">
              {label}
            </TabsTrigger>
          ))}
        </TabsList>

        {SCENARIO_META.map(({ key }) => (
          <TabsContent key={key} value={key} className="space-y-4">
            {getScenario(solver, key).status !== "optimal" ? (
              <Card className="border-amber-500/30 bg-amber-500/10">
                <CardContent className="flex items-start gap-2 pt-6 text-sm text-amber-200">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <p>{getScenario(solver, key).message ?? "Scenario is still being prepared."}</p>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>
        ))}
      </Tabs>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryStat label="Net xP Gain" value={`${netGain >= 0 ? "+" : ""}${formatPoints(netGain)}`} sub={`vs ${formatPoints(baseline)} baseline`} />
        <SummaryStat label="Expected xP" value={formatPoints(scenario.expected_points)} />
        <SummaryStat label="Hits Taken" value={scenario.hit_cost.toString()} sub={`Transfers: ${scenario.num_transfers}`} />
        <SummaryStat label="FT Used" value={totalFtUsed.toString()} sub={`Recommended: ${solver.recommended}`} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card className="bg-card/70">
          <CardHeader>
            <CardTitle className="text-base">Weekly Plan Timeline</CardTitle>
            <CardDescription>Transfers in/out, FT usage, captain, and expected xP by GW.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {weeklyPlans.length > 0 ? (
                weeklyPlans.map((plan) => <WeeklyPlanCard key={`plan-${plan.gameweek}`} plan={plan} />)
              ) : (
                <p className="text-sm text-muted-foreground">No weekly plans available for this scenario yet.</p>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <UnderperformersPanel players={scenario.underperformers ?? []} />
          <BaselineXpChart solver={solver} />
        </div>
      </div>

      <TransferHistoryTable transfers={history?.transfers ?? []} />
    </div>
  )
}
