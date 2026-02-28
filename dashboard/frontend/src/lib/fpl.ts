// FDR difficulty color mapping (1-5 scale)
export const FDR_COLORS: Record<number, string> = {
  1: "bg-emerald-600 text-white",    // very easy
  2: "bg-emerald-400 text-black",    // easy
  3: "bg-zinc-400 text-black",       // medium
  4: "bg-red-400 text-white",        // hard
  5: "bg-red-700 text-white",        // very hard
}

export const FDR_BG: Record<number, string> = {
  1: "#059669",
  2: "#34d399",
  3: "#a1a1aa",
  4: "#f87171",
  5: "#b91c1c",
}

// Position colors
export const POSITION_COLORS: Record<string, string> = {
  GKP: "#f59e0b",
  DEF: "#3b82f6",
  MID: "#10b981",
  FWD: "#ef4444",
}

export const POSITION_ORDER: Record<string, number> = {
  GKP: 0,
  DEF: 1,
  MID: 2,
  FWD: 3,
}

// Player photo from PL CDN
export function playerPhotoUrl(code: number | undefined): string {
  if (!code) return ""
  return `https://resources.premierleague.com/premierleague/photos/players/250x250/p${code}.png`
}

// Team badge from PL CDN
export function teamBadgeUrl(teamCode: number | undefined): string {
  if (!teamCode) return ""
  return `https://resources.premierleague.com/premierleague/badges/70/t${teamCode}.png`
}

// Format number with commas
export function formatNumber(n: number | null | undefined): string {
  if (n == null) return "-"
  return n.toLocaleString()
}

// Format price in millions
export function formatPrice(p: number | null | undefined): string {
  if (p == null) return "-"
  return `${p.toFixed(1)}m`
}

// Time ago string
export function timeAgo(isoString: string | undefined): string {
  if (!isoString) return "never"
  const diff = Date.now() - new Date(isoString).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}
