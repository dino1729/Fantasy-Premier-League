import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: api.meta, refetchInterval: 60_000 })
}

export function useSquad() {
  return useQuery({ queryKey: ["squad"], queryFn: api.squad })
}

export function useSquadIssues() {
  return useQuery({ queryKey: ["squad", "issues"], queryFn: api.squadIssues })
}

export function usePlayers(params?: Record<string, string>) {
  const query = useQuery({
    queryKey: ["players", params],
    queryFn: () => api.players(params),
    // Poll every 10s while analysis_job hasn't populated data yet
    refetchInterval: (query) => {
      const total = query.state.data?.total ?? 0
      return total === 0 ? 10_000 : false
    },
  })
  return query
}

export function usePlayer(id: number) {
  return useQuery({
    queryKey: ["players", id],
    queryFn: () => api.player(id),
    enabled: id > 0,
  })
}

export function useFDRGrid() {
  return useQuery({ queryKey: ["fixtures", "fdr-grid"], queryFn: api.fdrGrid })
}

export function useSolver() {
  return useQuery({
    queryKey: ["transfers", "solver"],
    queryFn: api.solver,
    refetchInterval: 30_000,  // poll until solver completes
  })
}

export function useTransferHistory() {
  return useQuery({ queryKey: ["transfers", "history"], queryFn: api.transferHistory })
}

export function useCompetitors() {
  return useQuery({
    queryKey: ["league", "competitors"],
    queryFn: api.competitors,
    refetchInterval: (query) => {
      const total = query.state.data?.competitors?.length ?? 0
      return total === 0 ? 30_000 : false
    },
  })
}

export function useGlobalManagers() {
  return useQuery({
    queryKey: ["league", "global"],
    queryFn: api.globalManagers,
    refetchInterval: (query) => {
      const total = query.state.data?.global_managers?.length ?? 0
      return total === 0 ? 30_000 : false
    },
  })
}

export function useScatter(chartType: string) {
  return useQuery({
    queryKey: ["scatter", chartType],
    queryFn: () => api.scatter(chartType),
  })
}

export function useManagerOverview() {
  return useQuery({
    queryKey: ["manager", "overview"],
    queryFn: api.managerOverview,
    refetchInterval: (query) => {
      if (query.state.data) return false
      const err = query.state.error
      return err instanceof Error && err.message === "warming_up" ? 30_000 : false
    },
  })
}

export function useCaptains() {
  return useQuery({
    queryKey: ["manager", "captains"],
    queryFn: api.captains,
    refetchInterval: (query) => {
      if (query.state.data) return false
      const err = query.state.error
      return err instanceof Error && err.message === "warming_up" ? 30_000 : false
    },
  })
}
