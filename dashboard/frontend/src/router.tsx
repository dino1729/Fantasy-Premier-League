import {
  createRouter,
  createRoute,
  createRootRoute,
} from "@tanstack/react-router"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RootLayout } from "@/routes/root-layout"
import { DashboardPage } from "@/routes/dashboard"
import { PlayersPage } from "@/routes/players"
import { FixturesPage } from "@/routes/fixtures"
import { TransfersPage } from "@/routes/transfers"
import { LeaguePage } from "@/routes/league"
import { ScatterPage } from "@/routes/scatter"
import { ManagerPage } from "@/routes/manager"

function RouteErrorFallback({
  error,
  reset,
}: {
  error: unknown
  reset: () => void
}) {
  const message =
    error instanceof Error ? error.message : "Unexpected route error"
  return (
    <div className="p-4 md:p-6">
      <Card className="max-w-2xl border-red-500/30 bg-red-500/10">
        <CardHeader>
          <CardTitle>Page Error</CardTitle>
          <CardDescription>
            This route failed to render. You can retry without reloading the app.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-red-300">{message}</p>
          <Button type="button" onClick={reset}>
            Retry
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

const rootRoute = createRootRoute({
  component: RootLayout,
  errorComponent: RouteErrorFallback,
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: DashboardPage,
  errorComponent: RouteErrorFallback,
})

const playersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/players",
  component: PlayersPage,
  errorComponent: RouteErrorFallback,
})

const fixturesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/fixtures",
  component: FixturesPage,
  errorComponent: RouteErrorFallback,
})

const transfersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/transfers",
  component: TransfersPage,
  errorComponent: RouteErrorFallback,
})

const leagueRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/league",
  component: LeaguePage,
  errorComponent: RouteErrorFallback,
})

const scatterRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scatter",
  component: ScatterPage,
  errorComponent: RouteErrorFallback,
})

const managerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/manager",
  component: ManagerPage,
  errorComponent: RouteErrorFallback,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  playersRoute,
  fixturesRoute,
  transfersRoute,
  leagueRoute,
  scatterRoute,
  managerRoute,
])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
