import { cn } from "@/lib/utils"

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  className?: string
}

export function StatCard({ label, value, sub, className }: StatCardProps) {
  return (
    <div className={cn("rounded-lg border border-border bg-card p-4", className)}>
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-bold text-foreground">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
    </div>
  )
}
