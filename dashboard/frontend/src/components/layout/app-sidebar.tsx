import { Link, useRouterState } from "@tanstack/react-router"
import { cn } from "@/lib/utils"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  LayoutDashboard,
  Users,
  Calendar,
  ArrowLeftRight,
  Trophy,
  ScatterChart,
  UserCircle,
} from "lucide-react"

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/players", label: "Players", icon: Users },
  { to: "/fixtures", label: "Fixtures", icon: Calendar },
  { to: "/transfers", label: "Transfers", icon: ArrowLeftRight },
  { to: "/league", label: "League", icon: Trophy },
  { to: "/scatter", label: "Scatter", icon: ScatterChart },
  { to: "/manager", label: "Manager", icon: UserCircle },
] as const

interface AppSidebarProps {
  mobileOpen: boolean
  onMobileOpenChange: (open: boolean) => void
}

function SidebarNav({
  currentPath,
  onNavigate,
}: {
  currentPath: string
  onNavigate?: () => void
}) {
  return (
    <nav className="flex-1 space-y-1 p-2">
      {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
        const isActive = to === "/" ? currentPath === "/" : currentPath.startsWith(to)
        return (
          <Link
            key={to}
            to={to}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            )}
          >
            <Icon className="size-4" />
            {label}
          </Link>
        )
      })}
    </nav>
  )
}

function SidebarBrand() {
  return (
    <div className="flex h-14 items-center gap-2 border-b border-border px-4">
      <div className="size-8 rounded-lg bg-primary flex items-center justify-center">
        <span className="text-sm font-bold text-primary-foreground">FPL</span>
      </div>
      <span className="font-semibold text-sm">FPL Analytics</span>
    </div>
  )
}

export function AppSidebar({ mobileOpen, onMobileOpenChange }: AppSidebarProps) {
  const router = useRouterState()
  const currentPath = router.location.pathname

  return (
    <>
      <aside className="hidden h-full w-56 flex-col border-r border-border bg-card md:flex">
        <SidebarBrand />
        <SidebarNav currentPath={currentPath} />
      </aside>

      <Sheet open={mobileOpen} onOpenChange={onMobileOpenChange}>
        <SheetContent side="left" className="w-64 border-r border-border bg-card p-0 sm:max-w-none">
          <SheetHeader className="sr-only">
            <SheetTitle>Navigation</SheetTitle>
          </SheetHeader>
          <div className="flex h-full flex-col">
            <SidebarBrand />
            <SidebarNav currentPath={currentPath} onNavigate={() => onMobileOpenChange(false)} />
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
