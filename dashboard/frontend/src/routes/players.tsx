import { useState, useMemo, useRef, useCallback } from "react"
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
} from "@tanstack/react-table"
import { useVirtualizer } from "@tanstack/react-virtual"
import { usePlayers } from "@/hooks/use-api"
import type { PlayerAnalysis } from "@/lib/api"
import { POSITION_COLORS, formatPrice } from "@/lib/fpl"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ArrowUpDown, ArrowUp, ArrowDown, Columns3, Loader2 } from "lucide-react"

// ---- LocalStorage persistence for column visibility ----
const STORAGE_KEY = "fpl-players-columns"

function loadColumnVisibility(): VisibilityState {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) : {}
  } catch {
    return {}
  }
}

function saveColumnVisibility(state: VisibilityState) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}

// ---- Column definitions ----

function numCell(val: number | null | undefined, decimals = 0) {
  if (val == null) return <span className="text-muted-foreground">-</span>
  return decimals > 0 ? val.toFixed(decimals) : val.toLocaleString()
}

function diffCell(val: number | null | undefined) {
  if (val == null) return <span className="text-muted-foreground">-</span>
  const color = val > 0.5 ? "text-emerald-400" : val < -0.5 ? "text-red-400" : "text-muted-foreground"
  return <span className={color}>{val > 0 ? "+" : ""}{val.toFixed(2)}</span>
}

function pctCell(val: number | null | undefined) {
  if (val == null) return <span className="text-muted-foreground">-</span>
  const color = val >= 75 ? "text-emerald-400" : val >= 50 ? "text-zinc-300" : val >= 25 ? "text-amber-400" : "text-red-400"
  return <span className={color}>{val.toFixed(0)}%</span>
}

function confidenceCell(val: number | null | undefined) {
  if (val == null) return <span className="text-muted-foreground">-</span>
  if (val >= 0.8) return <Badge variant="outline" className="border-emerald-500 text-emerald-400 text-xs">High</Badge>
  if (val >= 0.5) return <Badge variant="outline" className="border-amber-500 text-amber-400 text-xs">Med</Badge>
  return <Badge variant="outline" className="border-red-500 text-red-400 text-xs">Low</Badge>
}

const columns: ColumnDef<PlayerAnalysis>[] = [
  // Identity
  {
    accessorKey: "web_name",
    header: "Player",
    enableHiding: false,
    size: 140,
    cell: ({ row }) => (
      <div className="flex items-center gap-1.5">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
          style={{ backgroundColor: POSITION_COLORS[row.original.position] ?? "#888" }}
        />
        <span className="font-medium truncate">{row.original.web_name}</span>
      </div>
    ),
  },
  {
    accessorKey: "position",
    header: "Pos",
    size: 50,
    cell: ({ getValue }) => {
      const pos = getValue<string>()
      return <span style={{ color: POSITION_COLORS[pos] }}>{pos}</span>
    },
  },
  { accessorKey: "team", header: "Team", size: 50 },
  {
    accessorKey: "price",
    header: "Price",
    size: 60,
    cell: ({ getValue }) => formatPrice(getValue<number>()),
  },
  // FPL Stats
  { accessorKey: "total_points", header: "Pts", size: 50, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "form", header: "Form", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "minutes", header: "Min", size: 55, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "goals", header: "G", size: 40, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "assists", header: "A", size: 40, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "clean_sheets", header: "CS", size: 40, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "bps", header: "BPS", size: 55, cell: ({ getValue }) => numCell(getValue<number>()) },
  // xG/xA
  { accessorKey: "xg", header: "xG", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 2) },
  { accessorKey: "xa", header: "xA", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 2) },
  { accessorKey: "xg_diff", header: "xG+/-", size: 60, cell: ({ getValue }) => diffCell(getValue<number>()) },
  { accessorKey: "xa_diff", header: "xA+/-", size: 60, cell: ({ getValue }) => diffCell(getValue<number>()) },
  // ICT
  { accessorKey: "influence", header: "Inf", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "creativity", header: "Cre", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "threat", header: "Thr", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "ict_index", header: "ICT", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  // Percentiles
  { accessorKey: "pct_form", header: "%Form", size: 60, cell: ({ getValue }) => pctCell(getValue<number>()) },
  { accessorKey: "pct_ict", header: "%ICT", size: 60, cell: ({ getValue }) => pctCell(getValue<number>()) },
  { accessorKey: "pct_xg", header: "%xG", size: 60, cell: ({ getValue }) => pctCell(getValue<number>()) },
  { accessorKey: "pct_xp", header: "%xP", size: 60, cell: ({ getValue }) => pctCell(getValue<number>()) },
  // Predictions
  { accessorKey: "xp_gw1", header: "xP1", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "xp_gw2", header: "xP2", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "xp_gw3", header: "xP3", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "xp_gw4", header: "xP4", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "xp_gw5", header: "xP5", size: 55, cell: ({ getValue }) => numCell(getValue<number>(), 1) },
  { accessorKey: "xp_confidence", header: "Conf", size: 60, cell: ({ getValue }) => confidenceCell(getValue<number>()) },
  // Ownership
  {
    accessorKey: "selected_by_percent",
    header: "Sel%",
    size: 60,
    cell: ({ getValue }) => {
      const v = getValue<number>()
      return v != null ? `${v.toFixed(1)}%` : "-"
    },
  },
  { accessorKey: "transfers_in_event", header: "In", size: 60, cell: ({ getValue }) => numCell(getValue<number>()) },
  { accessorKey: "transfers_out_event", header: "Out", size: 60, cell: ({ getValue }) => numCell(getValue<number>()) },
]

// ---- Column group labels for header ----
const COLUMN_GROUPS: Record<string, string[]> = {
  Identity: ["web_name", "position", "team", "price"],
  "FPL Stats": ["total_points", "form", "minutes", "goals", "assists", "clean_sheets", "bps"],
  "xG/xA": ["xg", "xa", "xg_diff", "xa_diff"],
  ICT: ["influence", "creativity", "threat", "ict_index"],
  Percentiles: ["pct_form", "pct_ict", "pct_xg", "pct_xp"],
  Predictions: ["xp_gw1", "xp_gw2", "xp_gw3", "xp_gw4", "xp_gw5", "xp_confidence"],
  Ownership: ["selected_by_percent", "transfers_in_event", "transfers_out_event"],
}

// ---- Main component ----

const ROW_HEIGHT = 36

export function PlayersPage() {
  // Filters (server-side via query params)
  const [position, setPosition] = useState<string>("all")
  const [priceRange, setPriceRange] = useState<[number, number]>([3.5, 15.0])
  const [minMinutes, setMinMinutes] = useState(0)

  // Build query params for API
  const apiParams = useMemo(() => {
    const p: Record<string, string> = {}
    if (position !== "all") p.position = position
    if (priceRange[0] > 3.5) p.min_price = priceRange[0].toString()
    if (priceRange[1] < 15.0) p.max_price = priceRange[1].toString()
    if (minMinutes > 0) p.min_minutes = minMinutes.toString()
    return Object.keys(p).length > 0 ? p : undefined
  }, [position, priceRange, minMinutes])

  const { data, isLoading, error } = usePlayers(apiParams)
  const players = data?.players ?? []

  // Table state
  const [sorting, setSorting] = useState<SortingState>([{ id: "total_points", desc: true }])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(loadColumnVisibility)

  const handleVisibilityChange = useCallback((updater: VisibilityState | ((old: VisibilityState) => VisibilityState)) => {
    setColumnVisibility((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater
      saveColumnVisibility(next)
      return next
    })
  }, [])

  const table = useReactTable({
    data: players,
    columns,
    state: { sorting, columnVisibility },
    onSortingChange: setSorting,
    onColumnVisibilityChange: handleVisibilityChange,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  })

  // Virtual scrolling
  const parentRef = useRef<HTMLDivElement>(null)
  const { rows } = table.getRowModel()

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 15,
  })

  // Computing / error states
  if (error) {
    const isWarmingUp = error.message === "warming_up"
    if (isWarmingUp) {
      return (
        <div className="flex flex-col items-center justify-center gap-3 py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Server is warming up...</p>
        </div>
      )
    }
    return <div className="py-10 text-center text-red-400 text-sm">Failed to load players: {error.message}</div>
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Position toggle */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Position</span>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={position}
            onValueChange={(v) => v && setPosition(v)}
          >
            <ToggleGroupItem value="all">All</ToggleGroupItem>
            {["GKP", "DEF", "MID", "FWD"].map((pos) => (
              <ToggleGroupItem key={pos} value={pos}>
                <span style={{ color: POSITION_COLORS[pos] }}>{pos}</span>
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>

        {/* Price range */}
        <div className="flex items-center gap-2 min-w-[180px]">
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            Price {priceRange[0].toFixed(1)}-{priceRange[1].toFixed(1)}
          </span>
          <Slider
            min={3.5}
            max={15.0}
            step={0.5}
            value={priceRange}
            onValueChange={(v) => setPriceRange(v as [number, number])}
            className="w-32"
          />
        </div>

        {/* Min minutes */}
        <div className="flex items-center gap-2 min-w-[160px]">
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            Min {minMinutes}
          </span>
          <Slider
            min={0}
            max={2500}
            step={100}
            value={[minMinutes]}
            onValueChange={(v) => setMinMinutes(v[0])}
            className="w-28"
          />
        </div>

        {/* Column visibility dropdown */}
        <div className="ml-auto">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-1.5">
                <Columns3 className="h-3.5 w-3.5" />
                Columns
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48 max-h-80 overflow-y-auto">
              {Object.entries(COLUMN_GROUPS).map(([group, colIds]) => (
                <div key={group}>
                  <DropdownMenuLabel>{group}</DropdownMenuLabel>
                  {colIds.map((colId) => {
                    const col = table.getColumn(colId)
                    if (!col || !col.getCanHide()) return null
                    return (
                      <DropdownMenuCheckboxItem
                        key={colId}
                        checked={col.getIsVisible()}
                        onCheckedChange={(v) => col.toggleVisibility(!!v)}
                      >
                        {col.columnDef.header as string}
                      </DropdownMenuCheckboxItem>
                    )
                  })}
                  <DropdownMenuSeparator />
                </div>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Player count */}
      <div className="text-xs text-muted-foreground">
        {isLoading ? "Loading..." : `${players.length} players`}
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-1">
          <Skeleton className="h-9 w-full" />
          {Array.from({ length: 15 }, (_, i) => (
            <Skeleton key={i} className="h-9 w-full" />
          ))}
        </div>
      )}

      {/* Empty state: analysis not run yet */}
      {!isLoading && players.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 py-16">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Analyzing players... This takes a few minutes on first run.</p>
        </div>
      )}

      {/* Table */}
      {!isLoading && players.length > 0 && (
        <div
          ref={parentRef}
          className="relative overflow-auto rounded-md border flex-1"
          style={{ maxHeight: "calc(100vh - 200px)" }}
        >
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-zinc-900">
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    const isSorted = header.column.getIsSorted()
                    return (
                      <TableHead
                        key={header.id}
                        style={{ width: header.getSize(), minWidth: header.getSize() }}
                        className={`cursor-pointer select-none ${
                          header.column.id === "web_name"
                            ? "sticky left-0 z-20 bg-zinc-900"
                            : ""
                        }`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center gap-1">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {isSorted === "asc" ? (
                            <ArrowUp className="h-3 w-3 text-primary" />
                          ) : isSorted === "desc" ? (
                            <ArrowDown className="h-3 w-3 text-primary" />
                          ) : (
                            <ArrowUpDown className="h-3 w-3 text-muted-foreground/40" />
                          )}
                        </div>
                      </TableHead>
                    )
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {virtualizer.getVirtualItems().length > 0 && (
                /* Top spacer */
                <tr>
                  <td
                    colSpan={table.getVisibleFlatColumns().length}
                    style={{ height: virtualizer.getVirtualItems()[0]?.start ?? 0 }}
                  />
                </tr>
              )}
              {virtualizer.getVirtualItems().map((virtualRow) => {
                const row = rows[virtualRow.index]
                return (
                  <TableRow
                    key={row.id}
                    data-index={virtualRow.index}
                    style={{ height: ROW_HEIGHT }}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        style={{ width: cell.column.getSize(), minWidth: cell.column.getSize() }}
                        className={`text-xs ${
                          cell.column.id === "web_name"
                            ? "sticky left-0 z-10 bg-zinc-950"
                            : ""
                        }`}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                )
              })}
              {virtualizer.getVirtualItems().length > 0 && (
                /* Bottom spacer */
                <tr>
                  <td
                    colSpan={table.getVisibleFlatColumns().length}
                    style={{
                      height:
                        virtualizer.getTotalSize() -
                        (virtualizer.getVirtualItems().at(-1)?.end ?? 0),
                    }}
                  />
                </tr>
              )}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
