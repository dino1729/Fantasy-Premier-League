import type { Player } from "@/lib/api"
import { cn } from "@/lib/utils"
import { POSITION_COLORS, playerPhotoUrl } from "@/lib/fpl"

interface PlayerCardProps {
  player: Player
  compact?: boolean
}

export function PlayerCard({ player, compact }: PlayerCardProps) {
  const photoCode = (player.stats as Record<string, unknown>)?.code as number | undefined
  const points = (player.stats as Record<string, unknown>)?.event_points as number | undefined
  const posColor = POSITION_COLORS[player.position] ?? "#888"

  return (
    <div
      className={cn(
        "relative flex flex-col items-center rounded-lg border border-border bg-card transition-colors hover:bg-accent/30",
        compact ? "w-16 p-1.5" : "w-20 p-2"
      )}
    >
      {/* Captain badge */}
      {player.is_captain && (
        <div className="absolute -top-1.5 -right-1.5 z-10 flex size-5 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground">
          C
        </div>
      )}
      {player.is_vice_captain && (
        <div className="absolute -top-1.5 -right-1.5 z-10 flex size-5 items-center justify-center rounded-full border border-primary bg-card text-[10px] font-bold text-primary">
          V
        </div>
      )}

      {/* Player photo */}
      {photoCode ? (
        <img
          src={playerPhotoUrl(photoCode)}
          alt={player.name}
          className={cn("rounded-full object-cover", compact ? "size-10" : "size-12")}
          loading="lazy"
        />
      ) : (
        <div
          className={cn(
            "flex items-center justify-center rounded-full bg-muted text-xs font-bold",
            compact ? "size-10" : "size-12"
          )}
        >
          {player.name.slice(0, 2)}
        </div>
      )}

      {/* Name */}
      <p className={cn("mt-1 truncate text-center font-medium", compact ? "w-14 text-[9px]" : "w-18 text-[10px]")}>
        {player.name}
      </p>

      {/* Price + Position */}
      <div className="flex items-center gap-1">
        <span
          className="rounded px-1 text-[8px] font-bold text-white"
          style={{ backgroundColor: posColor }}
        >
          {player.position}
        </span>
        <span className="text-[9px] text-muted-foreground">
          {player.selling_price_m?.toFixed(1)}
        </span>
      </div>

      {/* Points */}
      {points != null && (
        <p className="mt-0.5 text-xs font-bold text-foreground">{points} pts</p>
      )}
    </div>
  )
}
