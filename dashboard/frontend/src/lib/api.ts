// API response types matching the FastAPI backend

export interface TeamInfo {
  team_id: number
  team_name: string
  manager_name: string
  overall_points: number
  overall_rank: number
  gameweek_points: number
  gameweek_rank: number
  season: string
}

export interface Player {
  id: number
  name: string
  full_name: string
  position: string
  team: string
  team_id: number
  is_captain: boolean
  is_vice_captain: boolean
  multiplier: number
  position_in_squad: number
  purchase_price: number
  selling_price: number
  purchase_price_m: number
  selling_price_m: number
  stats: Record<string, unknown>
}

export interface SquadIssues {
  injuries: Array<{ name: string; chance: number; return_estimate: string; news: string }>
  suspension_risk: Array<{ name: string; yellows: number; threshold: number; risk_level: string }>
  price_drops: Array<{ name: string; change: number; change_season: number; trend: string }>
  ownership_decline: Array<{
    name: string
    net_transfers: number
    transfers_in: number
    transfers_out: number
    ownership: number
  }>
  total_issues: number
  summary: string
}

export interface SquadResponse {
  gameweek: number
  squad: Player[]
  team_info: TeamInfo
  refreshed_at: string
}

export interface MetaResponse {
  ready: boolean
  current_gameweek: number
  season: string
  team_info: TeamInfo
  teams: Array<{ id: number; name: string; short_name: string }>
  refresh_status: Record<string, { job_name: string; last_run_at: string; status: string; message: string }>
}

export interface GWHistoryEntry {
  event: number
  points: number
  total_points: number
  overall_rank: number
  rank_sort: number
  event_transfers: number
  event_transfers_cost: number
  value: number
  bank: number
}

export interface ManagerOverview {
  team_info: TeamInfo
  gw_history: GWHistoryEntry[]
  season_history: unknown[]
  chips_used: Array<{ name: string; time: string; event: number }>
  team_value: number
  bank: number
  free_transfers: number
  refreshed_at: string
}

export interface CaptainPick {
  gameweek: number
  captain_id: number
  captain_name: string
  captain_points: number
  captain_multiplier: number
  captain_return: number
  optimal_id: number
  optimal_name: string
  optimal_points: number
  optimal_return: number
  points_lost: number
  was_optimal: boolean
}

export interface CaptainSummary {
  total_gws: number
  correct_picks: number
  accuracy: number
  captain_points_earned: number
  captain_points_optimal: number
  captain_points_lost: number
}

export interface CaptainsResponse {
  captain_picks: CaptainPick[]
  summary: CaptainSummary
  refreshed_at: string
}

export interface FixtureEntry {
  gameweek: number
  opponent: string
  is_home: boolean
  difficulty: number
  difficulty_ordinal: number
  win_prob: number
  draw_prob: number
  loss_prob: number
}

export interface FDRGridTeam {
  name: string
  short_name: string
  fixtures: FixtureEntry[]
}

export interface FDRGridResponse {
  fdr_grid: Record<string, FDRGridTeam>
  bgw_dgw: {
    bgw: Array<{ gw: number; teams_missing: number; team_ids: number[] }>
    dgw: Array<{ gw: number; teams_doubled: number; team_ids: number[] }>
    normal: number[]
  }
  current_gameweek: number
  refreshed_at: string
}

export interface PlayerAnalysis {
  player_id: number
  web_name: string
  position: string
  team: string
  price: number
  form: number
  total_points: number
  minutes: number
  goals: number
  assists: number
  clean_sheets: number
  bps: number
  xg: number
  xa: number
  xg_diff: number
  xa_diff: number
  influence: number
  creativity: number
  threat: number
  ict_index: number
  xp_gw1: number | null
  xp_gw2: number | null
  xp_gw3: number | null
  xp_gw4: number | null
  xp_gw5: number | null
  xp_confidence: number | null
  pct_form: number | null
  pct_ict: number | null
  pct_xg: number | null
  pct_xp: number | null
  transfers_in_event: number
  transfers_out_event: number
  selected_by_percent: number
  form_trend: unknown
  ict_breakdown: unknown
  raw_stats: unknown
}

export interface SolverTransferPlayer {
  name: string
  position?: string
  team?: string
  xp?: number[] | number
  sell_price?: number
  buy_price?: number
}

export interface SolverCaptain {
  name: string
  xp?: number
  eo?: number
}

export interface SolverWeeklyPlan {
  gameweek: number
  is_hold?: boolean
  transfers_in: SolverTransferPlayer[]
  transfers_out: SolverTransferPlayer[]
  ft_available: number
  ft_used: number
  ft_remaining: number
  hit_cost: number
  expected_xp: number
  confidence?: string
  reasoning?: string
  captain?: SolverCaptain | null
  differential_captain?: SolverCaptain | null
  formation?: string
}

export interface Underperformer {
  player_id: number
  name: string
  position: string
  team: string
  price: number
  severity: number
  reasons: string[]
  current_form: number
  total_points: number
}

export interface SolverScenario {
  status: string
  transfers_out?: SolverTransferPlayer[]
  transfers_in?: SolverTransferPlayer[]
  new_squad?: SolverTransferPlayer[]
  starting_xi?: SolverTransferPlayer[]
  bench?: SolverTransferPlayer[]
  formation?: string
  captain?: SolverCaptain | null
  hit_cost: number
  num_transfers: number
  expected_points: number
  per_gw_xp?: number[]
  baseline_xp: number
  net_gain?: number
  weekly_plans: SolverWeeklyPlan[]
  transfer_sequence?: Array<Record<string, unknown>>
  underperformers?: Underperformer[]
  scenario?: string
  message?: string
}

export interface SolverResponse {
  gameweek: number
  conservative: SolverScenario | null
  balanced: SolverScenario | null
  aggressive: SolverScenario | null
  recommended: string
  baseline_xp: number
  refreshed_at: string
}

export interface TransferHistoryEntry {
  element_in: number
  element_in_cost: number
  element_out: number
  element_out_cost: number
  entry: number
  event: number
  time: string
  element_in_name?: string
  element_out_name?: string
  element_in_team?: string
  element_out_team?: string
  element_in_position?: string
  element_out_position?: string
  element_in_cost_m?: number
  element_out_cost_m?: number
}

export interface LeagueTransferPlayer {
  player_id: number
  name: string
  position?: string
  team?: string
  gw_points?: number
  price?: number
}

export interface LeagueTransferTimelineEntry {
  gw: number
  chip?: string | null
  transfers_in: LeagueTransferPlayer[]
  transfers_out: LeagueTransferPlayer[]
}

export interface LeagueTransferHistory {
  gw_range: number[]
  transfer_timeline: LeagueTransferTimelineEntry[]
  chips_timeline?: Record<string, string>
  gw_squads_data?: Record<string, unknown>
}

export interface LeagueGWTransferSummary {
  transfers_in: LeagueTransferPlayer[]
  transfers_out: LeagueTransferPlayer[]
  net_points: number
  chip_used?: string | null
  prior_chip_used?: string | null
  transfer_cost: number
  is_wildcard: boolean
  is_free_hit: boolean
  num_changes: number
}

export interface LeagueSquadPlayer {
  player_id?: number
  id?: number
  name: string
  position: string
  team?: string
  position_in_squad?: number
  is_captain?: boolean
  is_vice_captain?: boolean
  stats?: Record<string, unknown>
}

export interface LeagueSeasonHistoryEntry {
  gameweek: number
  points?: number
  total_points?: number
  overall_rank?: number
  squad?: LeagueSquadPlayer[]
}

export interface LeagueEntry {
  entry_id: number
  team_info: TeamInfo
  gw_history: GWHistoryEntry[]
  squad: LeagueSquadPlayer[]
  season_history: LeagueSeasonHistoryEntry[]
  chips_used: Array<{ name: string; time: string; event: number }>
  total_hits: number
  team_value: number
  bank: number
  gw_transfers: LeagueGWTransferSummary
  transfer_history: LeagueTransferHistory
}

export interface ScatterPoint {
  player_id: number
  name: string
  team: string
  position: string
  x: number
  y: number
  minutes: number
}

export interface ScatterResponse {
  chart_type: string
  data: ScatterPoint[]
}

// Fetch helpers

const BASE = ""  // same-origin in prod, proxy in dev

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (res.status === 503) {
    throw new Error("warming_up")
  }
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const api = {
  meta: () => fetchJSON<MetaResponse>("/api/meta"),
  squad: () => fetchJSON<SquadResponse>("/api/squad"),
  squadIssues: () => fetchJSON<SquadIssues>("/api/squad/issues"),
  players: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : ""
    return fetchJSON<{ players: PlayerAnalysis[]; total: number }>(`/api/players${qs}`)
  },
  player: (id: number) => fetchJSON<PlayerAnalysis>(`/api/players/${id}`),
  fdrGrid: () => fetchJSON<FDRGridResponse>("/api/fixtures/fdr-grid"),
  solver: () => fetchJSON<SolverResponse>("/api/transfers/solver"),
  transferHistory: () => fetchJSON<{ transfers: TransferHistoryEntry[]; refreshed_at: string }>("/api/transfers/history"),
  competitors: () => fetchJSON<{ competitors: LeagueEntry[]; refreshed_at: string }>("/api/league/competitors"),
  globalManagers: () => fetchJSON<{ global_managers: LeagueEntry[]; refreshed_at: string }>("/api/league/global"),
  scatter: (type: string) => fetchJSON<ScatterResponse>(`/api/scatter/${type}`),
  managerOverview: () => fetchJSON<ManagerOverview>("/api/manager/overview"),
  captains: () => fetchJSON<CaptainsResponse>("/api/manager/captains"),
  health: () => fetchJSON<{ ready: boolean; jobs: Record<string, unknown> }>("/api/health"),
}
